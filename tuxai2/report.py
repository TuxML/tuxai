"""Generate reports for options."""

from collections import defaultdict
import logging
import hashlib
import json
from typing import Literal, get_args

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
FEATURE_IMPORTANCE_ARGS = Literal[
    "always_important",
    "always_very_important",
    "never_important",
    "sometime_important",
    "sometime_very_important",
    "",
]


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
        merge: str = "and",  # and/or
        target_filter: str | list[str] = TARGETS,
        is_top: int | None = None,
        has_version: str | float | int | list[str | float | int] = None,
        has_not_version: str | float | int | list[str | float | int] = None,
        has_collinearity: bool = False,
        has_not_kconfig: bool = False,
        has_unbalanced_yes_no_ratio: bool = False,
        feature_importance: FEATURE_IMPORTANCE_ARGS = "",
        verbose: bool = False,
    ) -> list[str] | dict[str, dict]:
        """Find list of options."""
        matches = dict()

        # arguments
        if isinstance(target_filter, str):
            if target_filter not in TARGETS:
                raise ValueError(target_filter)
            target_filter = [target_filter]
        if feature_importance not in get_args(FEATURE_IMPORTANCE_ARGS):
            raise ValueError(feature_importance)

        if is_top is not None:
            matches["is_top"] = defaultdict(list)
            for option, data in self._db.items():
                for version, targets in data["feature_importance"].items():
                    for target, rank in targets.items():
                        if (
                            target in target_filter
                            and rank is not None
                            and rank <= is_top
                        ):
                            matches["is_top"][option].append(
                                {"version": version, "target": target, "rank": rank}
                            )
            matches["is_top"] = dict(matches["is_top"])

        if has_version is not None:
            if not isinstance(has_version, (list, tuple)):
                has_version = [has_version]
            vers = [version_str(ver) for ver in has_version]
            matches["has_version"] = dict()
            for option, data in self._db.items():
                if set(vers).issubset(set(data["versions"])):
                    matches["has_version"][option] = data["versions"]

        if has_not_version is not None:
            if not isinstance(has_not_version, (list, tuple)):
                has_not_version = [has_not_version]
            vers = [version_str(ver) for ver in has_not_version]
            matches["has_not_version"] = dict()
            for option, data in self._db.items():
                if not set(vers).issubset(set(data["versions"])):
                    matches["has_not_version"][option] = list(
                        set(self.versions()) - set(data["versions"])
                    )

        if has_collinearity:
            matches["has_collinearity"] = dict()
            for option, data in self._db.items():
                if data["collinear"]:
                    matches["has_collinearity"][option] = data["collinear"]

        if has_not_kconfig:
            matches["has_not_kconfig"] = defaultdict(list)
            for option, data in self._db.items():
                for version in data["versions"]:
                    if version not in data["kconfig"] or not data["kconfig"][version]:
                        matches["has_not_kconfig"][option].append(version)
            matches["has_not_kconfig"] = dict(matches["has_not_kconfig"])

        if has_unbalanced_yes_no_ratio:
            matches["has_unbalanced_yes_no_ratio"] = defaultdict(dict)
            for option, data in self._db.items():
                if "analysis" in data and "unbalanced_yes_no_ratio" in data["analysis"]:
                    matches["has_unbalanced_yes_no_ratio"][option] = data["analysis"][
                        "unbalanced_yes_no_ratio"
                    ]
            matches["has_unbalanced_yes_no_ratio"] = dict(
                matches["has_unbalanced_yes_no_ratio"]
            )

        if feature_importance:
            matches["feature_importance"] = defaultdict(dict)
            for option, data in self._db.items():
                if (
                    "analysis" in data
                    and "feature_importance" in data["analysis"]
                    and feature_importance in data["analysis"]["feature_importance"]
                ):
                    fi = data["analysis"]["feature_importance"][feature_importance]
                    if fid := {
                        target: versions
                        for target, versions in fi.items()
                        if target in target_filter
                    }:
                        matches["feature_importance"][option][feature_importance] = fid

        # intersection
        if len(matches) > 0:
            if merge.lower() == "and":
                # intersection
                options = list()
                for key, res in matches.items():
                    options.append(set(res.keys()))
                options = options[0].intersection(*options)
                for key in matches.keys():
                    matches[key] = {
                        option: data
                        for option, data in matches[key].items()
                        if option in options
                    }
            # group by option
            by_opt = defaultdict(dict)
            for key, item in matches.items():
                for option, data in item.items():
                    by_opt[option][key] = data
            matches = dict(by_opt)

        # else:
        #     LOG.error(
        #         f'invalid merge parameter: {merge} (valid parameters are "or" or "and")'
        #     )

        return matches if verbose else list(matches.keys())

    def show(self, option: str, version: int | float | str | None = None) -> None:
        print(self.info(option, version, html=False))

    def option_help(self, option: str, version: int | float | str) -> dict:
        """Get Help string for this version of the option."""
        version = version_str(version)
        option = option.upper()

        if option not in self._db:
            return {"error": f"option not found: {option}"}

        data = self._db[option]
        new_help = ""
        # feature size impact
        if version not in data["filtered"]:
            fi = data["feature_size_impact"][version]["uncompressed"]
            add_word = "adds" if fi["diff"] >= 0 else "removes"
            sign = "+" if fi["diff"] >= 0 else "-"
            new_help += (
                f'In average, this option {add_word} {abs(fi["diff"]):.2f} Mb to kernel'
                + f' ({sign}{100* fi["diff_ratio"]:.2f} % from {fi["no"]:.2f} Mb to {fi["yes"]:.2f} Mb).'
            )
            if "analysis" in data:
                analysis = data["analysis"]
                top = False
                # top
                for key, item in analysis.items():
                    if key.startswith("top"):
                        ktop = int(key.split(" ")[1])
                        if (
                            "uncompressed" in item
                            and version in item["uncompressed"]
                            and (not top or ktop < top)
                        ):
                            top = ktop
                if top:
                    new_help += f" This option is in the top {top} for this version."

                # comp with other versions
                fid = analysis["feature_importance"]
                tested_versions = sorted(set(data["versions"]) - set(data["filtered"]))
                tested_versions_str = f"Tested versions ({', '.join(tested_versions)})"
                new_help_len = len(new_help)

                def _has_fi(fi_name: str) -> bool:
                    """Get wether option have feature importance for this versions."""
                    # always, never
                    if fi_name.startswith("always") or fi_name.startswith("never"):
                        return fi_name in fid and "uncompressed" in fid[fi_name]
                    # sometime
                    return fi_name in fid and "uncompressed" in fid[fi_name]

                if _has_fi("always_very_important"):
                    new_help += f" {tested_versions_str} always play a very important role in kernel size."
                elif _has_fi("always_important"):
                    new_help += f" {tested_versions_str} always play an important role in kernel size."
                elif _has_fi("never_important"):
                    new_help += f" {tested_versions_str} never play an important role in kernel size."
                elif _has_fi("sometime_very_important"):
                    imp = " / ".join(
                        [
                            f'{key.replace(" ", "")}: {", ".join(versions)}'
                            for key, versions in fid["sometime_very_important"][
                                "uncompressed"
                            ].items()
                        ]
                    )
                    new_help += f" {tested_versions_str} sometime play a very important role in kernel size ({imp})"
                elif _has_fi("sometime_important"):
                    imp = " / ".join(
                        [
                            f'{key.replace(" ", "")}: {", ".join(versions)}'
                            for key, versions in fid["sometime_important"][
                                "uncompressed"
                            ].items()
                        ]
                    )
                    new_help += f" {tested_versions_str} sometime play an important role in kernel size ({imp})"
                if len(new_help) > new_help_len:
                    if (
                        "unbalanced_yes_no_ratio" in analysis
                        and version in analysis["unbalanced_yes_no_ratio"]
                    ):
                        ynr = data["yes_no_ratio"][version]
                        yes_no_ratio_str = f"yes: {100* ynr['yes_ratio']:.2f} %, no: {100* ynr['no_ratio']:.2f} %"
                        new_help += f" But this version was tested with unbalanced yes/no ratio ({yes_no_ratio_str})"

        return {
            "current": data["kconfig"][version] if version in data["kconfig"] else {},
            "new_help": new_help,
        }

    def info(
        self, option: str, version: int | float | str | None = None, html: bool = False
    ) -> str:
        """Show available data for this option."""
        option = option.upper()
        data = self._db[option]
        output = list()

        if version is None:
            if html:
                output.append("<h1>Showing all versions</h1>")
            else:
                output.append("showing all versions")
            for version in self.versions():
                if html:
                    output.append(f"<h2>version {version}</h2>")
                else:
                    output.append(f"\n< version {version} >\n")

                output.append(self.info(option, version, html=html))
            if html:
                return "<br>".join(output)
            return "\n".join(output)

        version = version_str(version)

        if version not in data["versions"]:
            if html:
                return f'<p style="color:#FF0000">!!! {option} not found for version {version}</p>'
            else:
                return f"!!! {option} not found for version {version}"

        # kconfig
        if version in data["kconfig"]:
            data["kconfig"][version]
            if html:
                output.append("<p><b>* kconfig(s) found:</b></p>")
            else:
                output.append("* kconfig(s) found:\n")
            if len(data["kconfig"][version]) > 0:
                for path, item in data["kconfig"][version].items():
                    if html:
                        output.append(f"<i>[{path}]</i>")
                        content = item["content"].split("\n")
                        content = f"<blockquote>{'<br>'.join(content)}</blockquote>"
                        output.append(content + "<br>")
                    else:
                        output.append(f"\t[{path}]")
                        content = item["content"].split("\n")
                        content = "\t" + "\n\t".join(content)
                        output.append(content + "\n")
            else:
                if html:
                    output.append('<p style="color:#FF0000">No file found</p>')
                else:
                    output.append("(No file found)\n")

        # versions
        if data["versions"]:
            versions_list = ", ".join(data["versions"])
            if html:
                output.append(f"<p><b>* found in version(s):</b> {versions_list}</p>")
            else:
                output.append(f"* found in version(s): {versions_list}")
        if data["filtered"]:
            versions_list = ", ".join(data["filtered"])
            if html:
                output.append(
                    f"<p><b>* filtered in version(s):</b> {versions_list}</p>"
                )
            else:
                output.append(f"* filtered in version(s): {versions_list}")

        # collinearity
        if version in data["collinear"]:
            version_list = ", ".join(data["collinear"][version])
            if html:
                output.append(f"<p><b>* collinearity with:</b> {version_list}</p>")
            else:
                output.append(f"* collinearity with: {version_list}")

        # feature size impact
        for target in TARGETS:
            if target in data["feature_size_impact"][version]:
                item = data["feature_size_impact"][version][target]
                if "diff" in item:
                    item_diff_sign = "+" if item["diff"] > 0 else ""
                    item_diff = f'{item_diff_sign}{item["diff"]:.2f}'
                    item_diff_ratio = f'{100 * item["diff_ratio"]:.2f}'
                    if html:
                        output.append(
                            f"<p><b>* feature size impact (<i>{target}</i>):</b>"
                            f" {item_diff} Mb ({item_diff_ratio}%)</p>"
                        )
                    else:
                        output.append(
                            f"* feature size impact ({target}): {item_diff} Mb ({item_diff_ratio}%)"
                        )
        # feature importance
        for target in TARGETS:
            fi = [
                f"{version}: {position[target]}"
                for version, position in data["feature_importance"].items()
            ]
            if html:
                output.append(
                    f"<p><b>* feature importance ranks (<i>{target}</i>):</b> {', '.join(fi)}</p>"
                )
            else:
                output.append(f"* feature importance ranks ({target}): {', '.join(fi)}")

        # yes/no ratios
        ynr = [
            f"{version}: (y={100*item['yes_ratio']:.2f}%/n={100*item['no_ratio']:.1f}%)"
            for version, item in data["yes_no_ratio"].items()
            if item
        ]
        if html:
            output.append(f"<p><b>* yes/no % in dataset:</b> {', '.join(ynr)}</p>")
        else:
            output.append(f"* yes/no % in dataset: {', '.join(ynr)}")

        # analysis
        def _versions_by_targets(item: dict[str, list[str]]) -> str:
            """Get displayable string."""
            return ", ".join(
                [
                    f"{target}= {', '.join(versions)}"
                    for target, versions in item.items()
                ]
            )

        if "analysis" in data:
            if html:
                output.append("<h3> analysis </h3>")
            else:
                output.append("*** analysis ***")
            for key, item in data["analysis"].items():
                if key.startswith("top"):
                    if html:
                        output.append(
                            f"<p><b>* {key}:</b> {_versions_by_targets(item)}</p>"
                        )
                    else:
                        output.append(f"* {key}: {_versions_by_targets(item)}")
                elif key == "feature_importance":
                    for importance, imp_item in item.items():
                        if html:
                            output.append(
                                f"<p><b>* {importance}:</b> {_versions_by_targets(imp_item)}</p>"
                            )
                        else:
                            output.append(
                                f"* {importance}: {_versions_by_targets(imp_item)}"
                            )
                elif key == "unbalanced_yes_no_ratio":
                    if html:
                        output.append(
                            f"<p><b>* (warning) unbalanced yes/no ratio:</b> {', '.join(item)}</p>"
                        )
                    else:
                        output.append(
                            f"* (warning) unbalanced yes/no ratio: {', '.join(item)}"
                        )
                else:
                    raise ValueError(key)
        if html:
            return "".join(output)
        return "\n".join(output)

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

            # feature importance
            very_important = self._config["report_analysis"]["very_important_threshold"]
            important = self._config["report_analysis"]["important_threshold"]
            not_important = self._config["report_analysis"]["not_important_threshold"]
            fi = {
                target: {
                    version: data["feature_importance"][version][target]
                    for version in data["versions"]
                    if version not in data["filtered"]
                }
                for target in TARGETS
            }
            fid = defaultdict(dict)
            for target in TARGETS:
                very_important_versions = [
                    version
                    for version, value in fi[target].items()
                    if value is not None and value <= very_important
                ]
                important_versions = [
                    version
                    for version, value in fi[target].items()
                    if value is not None and value <= important
                ]

                not_important_versions = [
                    version
                    for version, value in fi[target].items()
                    if value is None or value >= not_important
                ]
                if target_fi_len := len(fi[target]):
                    # always very important
                    if len(very_important_versions) == target_fi_len:
                        fid["always_very_important"][target] = very_important_versions
                    # always important
                    elif len(important_versions) == target_fi_len:
                        fid["always_important"][target] = important_versions
                    # never important
                    if len(not_important_versions) == target_fi_len:
                        fid["never_important"][target] = not_important_versions
                    # sometime important
                    if len(important_versions) > 0 and len(not_important_versions) > 0:
                        fid["sometime_important"][target] = {
                            "important": important_versions,
                            "not_important": not_important_versions,
                        }
                    # sometime very important
                    if (
                        len(very_important_versions) > 0
                        and len(not_important_versions) > 0
                    ):
                        fid["sometime_very_important"][target] = {
                            "very_important": very_important_versions,
                            "not_important": not_important_versions,
                        }
                if fid:
                    opt_an["feature_importance"] = dict(fid)

            # save
            db[option]["analysis"] = opt_an
        return db

    def _get_all_options_info(self) -> dict[str, dict]:
        """Return a dict containing all available options and informations for each option.
        WARNING: several hours of computing expected, before being able to use cache."""
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
        if key in self._cache:
            return self._cache[key]

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
    # report = Report()
    report = Report("db.json")
    report.option_help("ARCNET_COM20020_CS", 413)

    # report.option_help("KASAN", 413)
    report.option_help("CC_OPTIMIZE_FOR_SIZE", 413)
    # report.find(is_top=5)
    report = Report()
    # report.dump_json("db.json")
    report.find(
        feature_importance="always_very_important", target_filter="uncompressed"
    )
    report.find(is_top=10, has_unbalanced_yes_no_ratio=True)
    report.find(is_top=10, has_not_kconfig=True)
    # report.find(is_top=3, has_collinearity=True, has_not_version=4.13)
    # report.find(is_top=3, has_collinearity=True)
    # report.find(is_top=3, has_version=5.08, has_not_version=5.08, has_collinearity=True)
    # report.dump_json("db.json")
    # report.load_json("db.json")
    # data = report["KASAN"]
    report.show("KASAN", 4.13)
    report.show("CC_OPTIMIZE_FOR_SIZE", 5.08)
    # report.versions()
    pass
