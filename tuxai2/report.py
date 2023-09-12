"""Generate reports for options."""
from collections import defaultdict
import logging
import hashlib

from tqdm.auto import tqdm
import pandas as pd

from tuxai2.misc import version_str
from tuxai2.dataset import Dataset, Columns
from tuxai2.misc import get_config, config_logger, cache
from tuxai2.kerneldoc import KernelDoc
from tuxai2.models import XGBoost

LOG = logging.getLogger(__name__)


class Report:
    """Report by option or version."""

    def __init__(self) -> None:
        """"""
        self._config = get_config()
        self._cache = cache()
        self._db = self._get_all_options_info()

    def __getitem__(self, option) -> dict:
        """Get all available data from a given option."""
        return self._db[option]

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
        targets = self._config["report"]["targets"]
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
            for target in targets:
                fsi = self._feature_mean_size_impact(ver, target)
                # for option, data in fsi.items():
                for option in raw_list:
                    if ver not in res[option]["feature_size_impact"]:
                        res[option]["feature_size_impact"][ver] = dict()
                    if target not in res[option]["feature_size_impact"][ver]:
                        res[option]["feature_size_impact"][ver][target] = dict()
                    res[option]["feature_size_impact"][ver][target] = fsi.get(
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
            for target in targets:
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
                        res[option]["feature_importance"][ver][target] = (
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
                "yes_perc_str": f"{100 * yes_ratio:.2f}%",
                "no_perc_str": f"{100 * no_ratio:.2f}%",
                # "very_unbalanced_ratio": (yes_ratio <= thresh or no_ratio <= thresh),
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

        df, groups = Dataset(version).get_dataframe(
            Columns.options, return_collinear_groups=True, add_features=False
        )

        features = df.columns
        df[target] = Dataset(version).get_dataframe(Columns.targets)[target]
        sizes = dict()
        for feature in tqdm(features, desc="feature mean size impact"):
            try:
                yes_size = df[df[feature]][target].mean()
                no_size = df[~df[feature]][target].mean()
                # group = ", ".join(groups[feature]) if feature in groups else feature
                # duplicate info for grouped options
                for option in groups[feature] if feature in groups else [feature]:
                    sizes[option] = {
                        "yes": yes_size,
                        "no": no_size,
                        "diff": yes_size - no_size,
                        "diff_ratio": (yes_size - no_size) / no_size,
                    }
            except KeyError:
                LOG.debug(f"not an original feature, skip: {feature}")
        self._cache[key] = sizes
        return sizes


if __name__ == "__main__":
    config_logger()

    report = Report()
    data = report["KASAN"]
    pass
