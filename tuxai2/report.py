"""Generate reports for options."""
from collections import defaultdict
import logging
import hashlib
import json

from tqdm.auto import tqdm
import pandas as pd
from cachetools import cached
from pathlib import Path

from tuxai2.misc import version_str
from tuxai2.dataset import Dataset, Columns
from tuxai2.misc import get_config, config_logger, cache, filter_options
from tuxai2.kerneldoc import KernelDoc
from tuxai2.models import XGBoost

LOG = logging.getLogger(__name__)

TARGETS = ("uncompressed", "compressed")


class Report:
    """Report by option or version."""

    def __init__(self, json_path: str | Path | None = None) -> None:
        """"""
        self._config = get_config()
        self._cache = cache()
        if json_path is None:
            raw_db = self._get_all_options_info()
            self._db = self._analysis(raw_db)
        else:
            self._db = self.load_json(json_path)

    def __getitem__(self, option) -> dict:
        """Get all available data from a given option."""
        return self._db[option.upper()]

    def dump_json(self, path: str | Path) -> None:
        """Save db to json."""
        with open(path, "w") as json_file:
            json.dump(self._db, json_file, indent=4)

    def load_json(self, path: str | Path) -> dict[str, dict]:
        """Load json DB."""
        with open(path, "r") as json_file:
            return json.load(json_file)

    @cached(cache={})
    def versions(self) -> list[str]:
        """Get list of available versions."""
        versions = set()
        for values in self._db.values():
            versions.update(values["versions"])
        return sorted(versions)

    @cached(cache={})
    def options(self) -> list[str]:
        """Get list of available options."""
        return sorted(self._db.keys())

    def find(
        self,
        is_top: int | None = None,
        has_version: str | float | int = None,
        has_not_version: str | float | int = None,
    ) -> dict[str, dict]:
        """Find list of options."""
        matches = dict()
        if is_top is not None:
            matches["is_top"] = defaultdict(list)
            for option, data in self._db.items():
                for version, targets in data["feature_importance"].items():
                    for target, rank in targets.items():
                        if rank is not None and rank <= is_top:
                            matches["is_top"][option].append(
                                {"version": version, "target": target, "rank": rank}
                            )
            matches["is_top"] = dict(matches["is_top"])
        if has_version is not None:
            ver = version_str(has_version)
            matches["has_version"] = dict()
            for option, data in self._db.items():
                if ver in data["versions"]:
                    matches["has_version"][option] = data["versions"]
        if has_not_version is not None:
            ver = version_str(has_not_version)
            matches["has_not_version"] = dict()
            for option, data in self._db.items():
                if ver not in data["versions"]:
                    matches["has_not_version"][option] = data["versions"]

        # TODO: doublons

        return matches

    def show(self, option: str, version: int | float | str) -> None:
        """Show available data for this option."""
        data = self._db[option.upper()]
        version = version_str(version)

        # kconfig
        if version in data["kconfig"]:
            data["kconfig"][version]
            print("* kconfig(s) found:\n")
            for path, item in data["kconfig"][version].items():
                print(f"\t[{path}]")
                content = item["content"].split("\n")
                content = "\t" + "\n\t".join(content)
                print(content)
                print("\n")
        # versions
        if data["versions"]:
            print(f"* found in version(s): {', '.join(data['versions'])}")
        if data["filtered"]:
            print(f"* filtered in version(s): {', '.join(data['filtered'])}")

        # collinearity
        if version in data["collinear"]:
            print(f'* collinearity with: {", ".join(data["collinear"][version])}')

        # feature size impact
        for target in TARGETS:
            if target in data["feature_size_impact"][version]:
                item = data["feature_size_impact"][version][target]
                item_diff_sign = "+" if item["diff"] > 0 else ""
                item_diff = f'{item_diff_sign}{item["diff"]:.2f}'
                item_diff_ratio = f'{100 * item["diff_ratio"]:.2f}'
                print(
                    f"* feature size impact ({target}): {item_diff} Mb ({item_diff_ratio}%)"
                )

    def _analysis(self, db: dict[str, dict]) -> dict[str, dict]:
        """Add analysis elements for each option."""

        for option, data in db.items():
            opt_an = dict()
            # unbalanced yes & no ratios
            thresh = self._config["report_analysis"]["yes_no_ratio_threshold"]
            unbalanced_versions = [
                version
                for version, item in data["yes_no_ratio"].items()
                if item and (item["yes_ratio"] < thresh or item["no_ratio"] < thresh)
            ]
            if unbalanced_versions:
                opt_an["unbalanced_yes_no_ratio"] = unbalanced_versions

            # top n features
            for top_n in self._config["report_analysis"]["top_n_features"]:
                top_n_label = f"top {top_n}"
                for target in TARGETS:
                    for version, item in data["feature_importance"].items():
                        if item[target] is not None and item[target] <= top_n:
                            if top_n_label not in opt_an:
                                opt_an[top_n_label] = dict()
                            if target not in opt_an[top_n_label]:
                                opt_an[top_n_label][target] = list()
                            opt_an[top_n_label][target].append(version)

            # # feature always important
            # for top_n in self._config["report_analysis"]["top_n_features"]:
            #     for version, item in data["feature_importance"].items():
            #         for target in TARGETS:

            # save
            db[option]["analysis"] = opt_an
        return db

    def _get_all_options_info(self) -> dict[str, dict]:
        """Return a dict containing all available options and informations for each option.
        WARNING: several hours of computing expected, before using cache."""
        cache_key_config = str(self._config["report"]) + str(self._config["kernel"])
        cache_key = (
            f"all_options_info|{hashlib.md5(cache_key_config.encode()).hexdigest()}"
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        versions = self._config["report"]["versions"]
        targets = {
            TARGETS[0]: self._config["report"]["uncompressed_target"],
            TARGETS[1]: self._config["report"]["compressed_target"],
        }
        res = defaultdict(dict)
        for version in tqdm(versions, desc="versions"):
            ver = version_str(version)
            dataset = Dataset(version)
            raw_list = set(dataset.raw_option_list())

            # init
            for option in raw_list:
                for key, dtype in (
                    ("versions", list),
                    ("filtered", list),
                    ("collinear", dict),
                    ("kconfig", dict),
                    ("feature_importance", dict),
                    ("feature_size_impact", dict),
                    ("yes_no_ratio", dict),
                ):
                    if key not in res[option]:
                        res[option][key] = dtype()

            # this option is available for this list of versions
            for option in raw_list:
                res[option]["versions"].append(ver)

            # feature size impact
            for target_label, target in targets.items():
                fsi = self._feature_mean_size_impact(ver, target)
                # for option, data in fsi.items():
                for option in raw_list:
                    if ver not in res[option]["feature_size_impact"]:
                        res[option]["feature_size_impact"][ver] = dict()
                    if target_label not in res[option]["feature_size_impact"][ver]:
                        res[option]["feature_size_impact"][ver][target_label] = dict()
                    res[option]["feature_size_impact"][ver][target_label] = fsi.get(
                        option, {}
                    )

            # feature importance (collinear options)
            df_coll = dataset.get_dataframe(
                col_filter=Columns.all,
                group_collinear_options=True,
                collinearity_threshold=0.0,
                return_collinear_groups=False,
                add_features=False,
                drop_outliers=False,
            )
            for target_label, target in targets.items():
                fi = self._feature_importance(
                    dataframe=df_coll,
                    target=target,
                    collinear_options=dataset.collinear_options_,
                    version=version,
                )
                for pos, options in fi["group"].items():
                    for option in options:
                        if ver not in res[option]["feature_importance"]:
                            res[option]["feature_importance"][ver] = dict()
                        res[option]["feature_importance"][ver][target_label] = (
                            (pos + 1) if fi["importance"][pos] > 0 else None
                        )

            # this option is filtered or not in this version
            df_no_coll = dataset.get_dataframe(
                col_filter=Columns.options,
                group_collinear_options=False,
                collinearity_threshold=0.0,
                return_collinear_groups=False,
                add_features=False,
                drop_outliers=False,
            )
            df_list = set(df_no_coll.columns)
            missing_options = raw_list - df_list
            for option in tqdm(raw_list, position=1, leave=False, desc="missing"):
                if option in missing_options:
                    res[option]["filtered"].append(ver)

            # yes frequency
            yes_freqs = self._yes_frequency(df_no_coll)
            for option in raw_list:
                res[option]["yes_no_ratio"][ver] = yes_freqs.get(option, {})

            # this option is collinear with these versions
            _, coll = dataset.get_dataframe(
                col_filter=Columns.options,
                group_collinear_options=True,
                collinearity_threshold=0.0,
                return_collinear_groups=True,
                add_features=False,
                drop_outliers=False,
            )
            for options in tqdm(
                coll.values(), position=1, leave=False, desc="collinear"
            ):
                for option in options:
                    coll_options = list(options)
                    coll_options.remove(option)
                    res[option]["collinear"][ver] = coll_options

            # kernel doc info
            kd = KernelDoc(version)
            kconf = kd.get_merged_kconfig_options()
            for option in raw_list:
                res[option]["kconfig"][ver] = dict()
                if option in kconf:
                    res[option]["kconfig"][ver] = kconf[option]
        res = dict(res)
        self._cache[cache_key] = res
        return res

    def _yes_frequency(self, dataframe: pd.DataFrame) -> dict[str:float]:
        """Yes / (Yes + No) for each option."""
        res = dict()
        # thresh = self._config["report"]["yes_no_ratio_threshold"]
        df_len = len(dataframe)
        for col in tqdm(dataframe.columns, desc="yes & no ratio."):
            val_count = dataframe[col].value_counts().to_dict()
            yes_count = val_count.get(True, 0)
            no_count = val_count.get(False, 0)
            yes_ratio = yes_count / df_len
            no_ratio = no_count / df_len
            res[col] = {
                "yes_count": yes_count,
                "no_count": no_count,
                "yes_ratio": yes_ratio,
                "no_ratio": no_ratio,
                # "yes_perc_str": f"{100 * yes_ratio:.2f}%",
                # "no_perc_str": f"{100 * no_ratio:.2f}%",
            }

        return res

    def _feature_importance(
        self,
        dataframe: pd.DataFrame,
        target: str,
        collinear_options: list,
        version: str,
    ):
        """Feature importance for a given version."""

        xgb = XGBoost(
            dataframe=dataframe,
            target=target,
            collinear_options=collinear_options,
            version=version,
            test_size=0,
        )
        xgb.fit()
        return xgb.options_scores()

    def _feature_mean_size_impact(self, version: str, target: str) -> dict:
        """Kernel size impact for each option."""
        key = f"size_diffs_dict_{version}_{target}."
        # if key in self._cache:
        #     return self._cache[key]

        sizes = dict()
        df = Dataset(version).get_dataframe(
            Columns.all,
            group_collinear_options=False,
            return_collinear_groups=False,
            add_features=False,
        )
        options = filter_options(columns=df.columns, config=self._config)

        for option in tqdm(options, desc="feature mean size impact"):
            try:
                yes_size = df[df[option]][target].mean()
                no_size = df[~df[option]][target].mean()
                # duplicate info for grouped options
                # for option in groups[option] if option in groups else [option]:
                sizes[option] = {
                    "yes": yes_size,
                    "no": no_size,
                    "diff": yes_size - no_size,
                    "diff_ratio": (yes_size - no_size) / no_size,
                }
            except KeyError:
                LOG.debug(f"not an original option, skip: {option}")
        self._cache[key] = sizes
        return sizes


if __name__ == "__main__":
    config_logger()

    # report = Report("db.json")
    # report.find(is_top=5)
    report = Report()
    report.find(is_top=3, has_version=5.08, has_not_version=5.08)
    # report.dump_json("db.json")
    # report.load_json("db.json")
    # data = report["KASAN"]
    report.show("KASAN", 4.13)
    # report.display("CC_OPTIMIZE_FOR_SIZE", 5.08)
    # report.versions()
    pass
