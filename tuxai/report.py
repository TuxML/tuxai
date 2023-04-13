"""Generate metrics, docs, plots, etc."""

from pathlib import Path
import logging
from itertools import combinations, product
from collections import defaultdict

import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
from tqdm.auto import tqdm
from typing import Callable
import xgboost

from tuxai.misc import get_config, cache  # , bio2df, df2bio
from tuxai.dataset import Dataset, Columns
from tuxai.models import XGBoost
from tuxai.features import CORR_PREFIX

LOG = logging.getLogger(__name__)

DEFAULT_REPORT_FILENAME = "report.xlsx"
GROUP_COL = "group"
MERGED_GROUPS_COL = "merged_group"
VMLINUX_TARGET = "vmlinux"


def model_metrics(
    y_pred: list, y_true: list, all_metrics: bool = False
) -> dict[str, float]:
    """Get some metrics from model prediction."""
    if all_metrics:
        metrics_list = [
            metrics.explained_variance_score,
            metrics.max_error,
            metrics.mean_absolute_error,
            metrics.mean_squared_error,
            # metrics.mean_squared_log_error,
            metrics.mean_absolute_percentage_error,
            metrics.r2_score,
            metrics.mean_poisson_deviance,
            metrics.mean_gamma_deviance,
            metrics.mean_tweedie_deviance,
            metrics.d2_tweedie_score,
            metrics.mean_pinball_loss,
            metrics.d2_pinball_score,
            metrics.d2_absolute_error_score,
        ]
    else:
        metrics_list = [
            metrics.mean_absolute_error,
            metrics.mean_absolute_percentage_error,
            metrics.mean_squared_error,
            metrics.max_error,
        ]
    res = {
        metric.__name__.replace("_", " "): metric(y_true, y_pred)
        for metric in metrics_list
    }
    res["root mean squared error"] = np.sqrt(res["mean squared error"])
    return res


def plot_pred_true(y_pred: pd.Series, y_true: pd.Series) -> Line2D:
    """Sort y_true and y_pred + scatter plot. TODO: better plot (this one is just a draft)."""
    fig, ax = plt.subplots(figsize=(20, 8))
    df_ap = (
        pd.DataFrame(
            {"actual": y_true, "predicted": y_pred, "error": abs(y_true - y_pred)}
        )
        .sort_values("actual", ascending=True)
        .reset_index()
    )
    sns.scatterplot(df_ap.predicted, ax=ax, s=1, color="k", label="predicted")
    sns.lineplot(df_ap.actual, ax=ax, alpha=0.5, label="actual")
    sns.scatterplot(df_ap.error, ax=ax, s=1, color="r", label="error")
    return fig


class BasicReport:
    """Basic report for each version, target, collinearity, with different options."""

    def __init__(self, path: str | Path | None = None) -> None:
        """Provide a path or get generated one."""
        # self._path = (
        #     Path(__file__).resolve().parent.parent
        #     / "reports"
        #     / date2dir()
        #     / DEFAULT_REPORT_FILENAME
        # )
        self._config = get_config()

    def feature_importance(
        self,
        versions: list[str] | None = None,
        targets: list[str] | None = None,
        collinearities: list[bool] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate feature importance report."""
        LOG.info("generating feature importance report")

        def _callback(xgb: xgboost.XGBRegressor) -> dict[str, float]:
            """Retur metrics."""
            feature_limit = self._config["report"]["feature_limit"]
            if feature_limit < 0:
                feature_limit = None
            return xgb.options_scores(limit=feature_limit)

        return self._all_config(
            callback=_callback,
            versions=versions,
            targets=targets,
            collinearities=collinearities,
            **kwargs,
        )

    def xgboost_model(
        self,
        versions: list[str] | None = None,
        targets: list[str] | None = None,
        collinearities: list[bool] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Test and generate report for each version/target/options.

        If not specified, get parameters from config file.
        """
        LOG.info("generating xgboost report")

        def _callback(xgb: xgboost.XGBRegressor) -> dict[str, float]:
            """Retur metrics."""
            return model_metrics(y_pred=xgb.pred(), y_true=xgb.y_test)

        return self._all_config(
            callback=_callback,
            versions=versions,
            targets=targets,
            collinearities=collinearities,
            **kwargs,
        )

    def feature_importance_stability(
        self,
        versions: list[str] | None = None,
        targets: list[str] | None = None,
        collinearities: list[bool] | None = None,
        top_count: int | None = None,
    ) -> dict:
        """Compare feature importance between each configuration.

        WARNING: somewhat deprecated version.
        """
        # pick all configurations
        df = self.feature_importance(
            versions=versions,
            targets=targets,
            collinearities=collinearities,
        )
        # from tuxai.misc import cache
        # df = cache()["feature_importance_2022_12_13"]
        top_count = (
            self._config["report"]["top_feature_count"]
            if top_count is None
            else top_count
        )

        res = dict()

        # we are working on variations of these parameters
        base_cols = ("version", "collinearity", "target")
        # pick 2 (set_col_a and set_col_b) and observe variation on remaing parameter (var_col)
        for set_col_a, set_col_b in (
            pbar_0 := tqdm(list(combinations(base_cols, 2)), position=0)
        ):
            var_col = [col for col in base_cols if col not in (set_col_a, set_col_b)][0]
            # now, loop on all set_col_1, set_col2 combinations
            for set_col_a_i, set_col_b_i in (
                pbar_1 := tqdm(
                    list(product(df[set_col_a].unique(), df[set_col_b].unique())),
                    position=1,
                    leave=False,
                )
            ):
                conf_str = f"{set_col_a} = {set_col_a_i}, {set_col_b} = {set_col_b_i} -> {var_col}"
                pbar_0.set_description(f"{set_col_a} + {set_col_b} -> {var_col}")
                pbar_1.set_description(conf_str)

                LOG.debug(f"comparing feature importance for {conf_str}")
                # keep what we need to check variation on var_col
                df_i = df[
                    (df[set_col_a] == set_col_a_i) & (df[set_col_b] == set_col_b_i)
                ]
                # extracts every rankings for each config
                options_ranks = defaultdict(list)
                group_corr = dict()
                corr_count = 0
                for row_id, row in df_i.iterrows():
                    # for idx, (rank, options) in enumerate(row.group.items()):
                    for idx, ((rank, option), (_, group)) in enumerate(
                        zip(row.option.items(), row.group.items())
                    ):
                        # handle collinear options
                        if len(group) > 1:
                            corr_count += 1
                            sgroup = tuple(sorted(group))
                            if sgroup in group_corr:
                                option = group_corr[sgroup]
                            else:
                                option = group_corr[
                                    sgroup
                                ] = f"{CORR_PREFIX}{corr_count:04}"

                        options_ranks[option].append(rank)
                        # is_col_str = f" [{idx}] ({option})" if len(group) > 1 else ""
                        # for item in group:
                        #     options_ranks[f"{item}{is_col_str}"].append(rank)

                # extract n best options (based on all time best position)
                top_options = (
                    pd.DataFrame.from_dict(
                        {option: min(ranks) for option, ranks in options_ranks.items()},
                        orient="index",
                    )
                    .sort_values(0)
                    .iloc[:top_count]
                    .index.to_list()
                )
                # keep only n best options (python >3.7, so, dict is insertion ordered)
                top_options_ranks = {
                    option: sorted(options_ranks[option]) for option in top_options
                }
                # flatten dataframe for plotting
                flatten_ranks = pd.DataFrame.from_dict(
                    [
                        {"option": col, "rank": rank}
                        for col, ranks in top_options_ranks.items()
                        for rank in ranks
                    ]
                )
                # plot
                sns.set_theme(style="ticks")

                fig, ax = plt.subplots(figsize=(15, 30))
                plt.xlim(0, 100)
                ax.xaxis.grid(True)
                ax.set(ylabel="")
                plt.title(conf_str)

                sns.boxplot(
                    x="rank",
                    y="option",
                    data=flatten_ranks,
                    whis=[0, 100],
                    width=0.6,
                    palette="vlag",
                    ax=ax,
                )

                # save result in a dataframe
                df_res = pd.DataFrame.from_dict(
                    {
                        col: ", ".join(str(rank) for rank in sorted(ranks))
                        for col, ranks in top_options_ranks.items()
                    },
                    orient="index",
                    columns=["ranks"],
                )

                # add collinearity data
                inv_group_corr = {
                    corr_name: options for options, corr_name in group_corr.items()
                }
                df_res["option"] = df_res.index
                df_res["group"] = df_res.option.apply(
                    lambda option: (", ".join(inv_group_corr[option]))
                    if option in inv_group_corr
                    else ""
                )
                df_res = df_res.drop(columns=["option"])

                # add new entry to result
                res[conf_str] = {
                    "config": (
                        var_col,
                        (set_col_a, set_col_a_i),
                        (set_col_b, set_col_b_i),
                    ),
                    "plot": fig,
                    "dataframe": df_res,
                }
                plt.close()
        return res

    def _all_config(
        self,
        callback: Callable,
        versions: list[str] | None = None,
        targets: list[str] | None = None,
        collinearities: list[bool] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Test and generate report for each version/target/options (common loop for reports).

        If not specified, get parameters from config file.
        """
        versions = self._config["report"]["versions"] if versions is None else versions
        targets = (
            self._config["report"]["xgboost_targets"] if targets is None else targets
        )
        collinearities = [True, False] if collinearities is None else collinearities
        res = list()
        for version in (pbar_0 := tqdm(versions, position=0)):
            dataset = Dataset(version)
            for target in (pbar_1 := tqdm(targets, position=1, leave=False)):
                for collinearity in collinearities:
                    coll_str = (
                        "with collinearity" if collinearity else "no collinearity"
                    )
                    pbar_0.set_description(self._version_str(version))
                    pbar_1.set_description(f"{target} - {coll_str}")

                    xgb = XGBoost(
                        dataset=dataset,
                        target=target,
                        group_collinear_options=collinearity,
                        **kwargs,
                    )
                    xgb.fit()
                    res.append(
                        {
                            "version": self._version_str(version),
                            "collinearity": collinearity,
                            "target": target,
                            **callback(xgb),
                        }
                    )
        return pd.DataFrame.from_dict(res)

    @staticmethod
    def _version_str(version: int | str) -> str:
        """Convert version before displaying."""
        str_ver = str(version)
        return f"{str_ver[0]}.{str_ver[1:]}"


class FeatureImportanceReport:
    """Answer some preliminary feature importance questions."""

    def __init__(self, use_cache: str | None = None, **kwargs) -> None:
        """All raw data come from Report class."""
        # pick all configurations
        report = BasicReport()

        if use_cache is None:
            df = report.feature_importance(**kwargs)
        else:
            cache_ = cache()
            if use_cache in cache_:
                LOG.info(f"loading feature importance data from cache: {use_cache}")
                # pyarrow cannot handle dicts in columns
                # df = bio2df(cache_[use_cache])
                df = cache_[use_cache]
            else:
                df = report.feature_importance(**kwargs)
                # pyarrow cannot handle dicts in columns
                # cache_[use_cache] = df2bio(df)
                cache_[use_cache] = df

        self._df = self._explode_df(df)

    def merged_groups(
        self,
        target: str,
        collinearity: bool = True,
    ) -> dict:
        """Get dict of merged groups."""
        df = self._df.copy()
        df = self._keep(df, targets=[target], collinearities=[collinearity])
        return self._add_merged_group(df, return_dict=True)

    def options_always_importants(
        self,
        target: str,
        rank: int = 30,
        collinearity: bool = True,
        merge_groups: bool = False,
        allow_version_gap=True,
    ) -> pd.DataFrame:
        """Options always < top N, regardless of version."""
        df = self._df.copy()
        df = self._keep(df, targets=[target], collinearities=[collinearity])
        group_col = GROUP_COL
        if merge_groups:
            df = self._add_merged_group(df)
            group_col = MERGED_GROUPS_COL
        if not allow_version_gap:
            df = self._drop_version_gap(df, group_col=group_col)

        # list of top N options
        options = list()
        for option in tqdm(df.option.unique()):
            if all(
                [
                    option_rank <= rank
                    for option_rank in df[df.option == option]["rank"].unique()
                ]
            ):
                options.append(option)

        return self._display_rank_by_version(
            df[df.option.isin(options)], ascending=True
        )

    def options_never_importants(
        self,
        target: str,
        rank: int = 300,
        collinearity: bool = True,
        merge_groups: bool = False,
        allow_version_gap=True,
    ) -> pd.DataFrame:
        """Options always > top N, regardless of version."""
        df = self._df.copy()
        df = self._keep(df, targets=[target], collinearities=[collinearity])

        group_col = GROUP_COL
        if merge_groups:
            df = self._add_merged_group(df)
            group_col = MERGED_GROUPS_COL
        if not allow_version_gap:
            df = self._drop_version_gap(df, group_col=group_col)

        # list of worst N options
        options = list()
        for option in tqdm(df.option.unique()):
            if all(
                [
                    option_rank >= rank
                    for option_rank in df[df.option == option]["rank"].unique()
                ]
            ):
                options.append(option)

        return self._display_rank_by_version(
            df[df.option.isin(options)], ascending=False
        )

    def options_not_always_importants(
        self,
        target: str,
        best_rank: int = 30,
        worst_rank=300,
        collinearity: bool = True,
        merge_groups: bool = False,
        allow_version_gap=True,
    ) -> pd.DataFrame:
        """Options at least once <= best_rank and at least once >= worst_rank, regardless of version."""
        df = self._df.copy()
        df = self._keep(df, targets=[target], collinearities=[collinearity])

        group_col = GROUP_COL
        if merge_groups:
            df = self._add_merged_group(df)
            group_col = MERGED_GROUPS_COL
        if not allow_version_gap:
            df = self._drop_version_gap(df, group_col=group_col)

        groups = list()
        for group in tqdm(df[group_col].unique()):
            ranks = sorted(df[df[group_col] == group]["rank"].unique())
            if ranks[0] <= best_rank and ranks[-1] >= worst_rank:
                groups.append(group)

        return self._display_rank_by_version(
            df[df[group_col].isin(groups)], ascending=True
        )

    def target_comparison(
        self, target_1: str, target_2: str, rank: int = 30, collinearity: bool = True
    ) -> pd.DataFrame:
        """Compare top N for 2 targets."""
        df_1 = self.options_always_importants(
            target=target_1, rank=rank, collinearity=collinearity
        )

        df_2 = self.options_always_importants(
            target=target_2, rank=rank, collinearity=collinearity
        )
        s_1 = set(df_1.options)
        s_2 = set(df_2.options)
        oai = {
            f"{target_1} only": df_1[df_1.options.isin(s_1 - s_2)],
            f"{target_2} only": df_2[df_2.options.isin(s_2 - s_1)],
            f"{target_1} and {target_2}": s_1 & s_2,
        }

        return {f"always top {rank}": oai}

    def add_yes_frequencies(
        self, dataframe: pd.DataFrame, str_pct: bool = False, collinearity: bool = True
    ) -> pd.DataFrame:
        """Add yes/(yes+no) on all features for each version."""
        res = dict()
        for version in tqdm(dataframe.columns):
            if collinearity:
                df, groups = Dataset(version).get_dataframe(
                    add_features=False,
                    col_filter=Columns.options,
                    group_collinear_options=True,
                    return_collinear_groups=True,
                )
                groups = {k: ", ".join(sorted(v)) for k, v in groups.items()}
                groups_inv = {v: k for k, v in groups.items()}
            else:
                df = Dataset(version).get_dataframe(
                    add_features=False,
                    col_filter=Columns.options,
                    group_collinear_options=False,
                    return_collinear_groups=False,
                )

            nai_options = [option for option in dataframe.index if option in df.columns]
            nai_options_col = (
                [option for option in dataframe.index if option in groups_inv]
                if collinearity
                else []
            )

            nai_df = df.mean().loc[
                nai_options + [groups_inv[option] for option in nai_options_col]
            ]
            if collinearity:
                nai_df = nai_df.rename(index=groups)
            res[version] = nai_df

        df_freq = pd.DataFrame.from_dict(res)
        if str_pct:
            df_freq = df_freq.applymap(lambda f: f"{100 * f:.2f}%")
        df_freq_rename = df_freq.rename(
            columns={col: f"freq {col}" for col in df_freq.columns}
        )
        return pd.concat([dataframe, df_freq_rename], axis=1).sort_index()

    def _keep(
        self,
        dataframe: pd.DataFrame,
        versions: list[str] | None = None,
        targets: list[str] | None = None,
        collinearities: list[bool] | None = None,
    ):
        if versions:
            dataframe = dataframe[dataframe.version.isin(versions)]
        if targets:
            dataframe = dataframe[dataframe.target.isin(targets)]
        if collinearities:
            dataframe = dataframe[dataframe.collinearity.isin(collinearities)]
        return dataframe

    def _display_rank_by_version(
        self, dataframe: pd.DataFrame, ascending: bool | None = None
    ) -> pd.DataFrame:
        if dataframe.empty:
            return dataframe
        group_col = (
            MERGED_GROUPS_COL if MERGED_GROUPS_COL in dataframe.columns else GROUP_COL
        )
        df = (
            dataframe.groupby(["target", group_col])[["version", "rank"]]
            .agg(list)
            .reset_index()
        )
        # convert to dict because it's easier to manipulate here
        split_cols = dict()
        for idx, item in df.to_dict(orient="index").items():
            split_cols[idx] = {"options": item[group_col]}
            split_cols[idx].update(
                {version: rank for version, rank in zip(item["version"], item["rank"])}
            )

        # back to dataframe
        df = pd.DataFrame.from_dict(split_cols, orient="index")
        df[df.columns[1:]] = df[df.columns[1:]].astype("Int64", errors="ignore")
        df = df[[df.columns[0]] + sorted(df.columns[1:])]

        # sort by rank mean
        if ascending is not None:
            df["rank_mean"] = df[df.columns[1:]].mean(axis=1)
            df = df.sort_values("rank_mean", ascending=ascending)
            df = df.drop(columns=["rank_mean"])
            return df

    def _explode_df(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """One line for each dict item in option, rank and group."""
        res = list()
        for _, row in dataframe.iterrows():
            for rank, group in row.group.items():
                for option in group:
                    res.append(
                        {
                            "version": row.version,
                            "collinearity": row.collinearity,
                            "target": row.target,
                            "option": option,
                            "group": ", ".join(sorted(group)),
                            "rank": rank,
                        }
                    )

        return pd.DataFrame.from_dict(res)

    def _drop_version_gap(
        self, dataframe: pd.DataFrame, group_col: str
    ) -> pd.DataFrame:
        """Remove each options which are missing versions."""
        LOG.debug("drop options with versions gap.")

        versions = set(dataframe.version.unique())
        # all available versions for each group of options
        df_ver = dataframe.groupby(group_col).agg({"version": lambda x: set(x)})
        # keep groups with all available versions
        df_ver_no_gap = df_ver[df_ver.version.apply(lambda x: x == versions)]
        return dataframe[dataframe[group_col].isin(df_ver_no_gap.index)]

    def _add_merged_group(
        self, dataframe: pd.DataFrame, return_dict: bool = False
    ) -> pd.DataFrame | dict:
        """Try to merge groups that seems related (add merged_group column)."""
        LOG.debug("add merged_group column.")
        by_opt = defaultdict(dict)

        for _, row in dataframe.iterrows():
            for option in row.group.split(", "):
                if row.version in by_opt[option].keys():
                    assert by_opt[option][row.version] == row.group
                by_opt[option][row.version] = row.group

        def _rename_group(item: dict) -> str:
            """Rename/merge groups."""
            # nothing to merge
            groups = set(item.values())
            if len(groups) == 1:
                return groups.pop()

            # unique options
            options = set(option for group in groups for option in group.split(", "))
            # versions = set(item.keys())

            item_by_opt = defaultdict(set)
            for version, group in item.items():
                for option in group.split(", "):
                    item_by_opt[option].add(version)

            res = list()
            for option in sorted(options):
                versions = item_by_opt[option]
                if len(versions) == len(item.keys()):
                    res.append(option)
                else:
                    res.append(f'{option}({", ".join(sorted(versions))})')
            return ", ".join(res)

        if return_dict:
            return {
                option: _rename_group(by_opt[option]) for option in dataframe.option
            }
        else:
            dataframe[MERGED_GROUPS_COL] = dataframe.option.apply(
                lambda option: _rename_group(by_opt[option])
            )
            return dataframe

    # def get_merged_groups(self, dataf)


class YesFrequencyOutliers:
    """Detect yes/no option frequency outliers."""

    def get_dataframe(
        self,
        target: str,
        use_fir_cache: str | None = None,
        threshold: float = 0.1,
        collinearity: bool = True,
    ) -> pd.DataFrame:
        """Dataframe with frequencies and outliers + score."""
        fir = FeatureImportanceReport(use_cache=use_fir_cache)
        df_ai = fir.options_always_importants(
            target=target,
            rank=99999,
            collinearity=True,
            merge_groups=False,
            allow_version_gap=True,
        ).set_index("options")
        df = fir.add_yes_frequencies(df_ai, str_pct=False)

        freq_cols = [col for col in df.columns if col.startswith("freq")]
        df["outliers_score"] = df[freq_cols].apply(
            lambda row: self.outliers_score(row), axis=1
        )
        df["outliers"] = df[freq_cols].apply(
            lambda row: self.detect_outliers(row), axis=1
        )

        return df[df.outliers_score >= threshold].sort_values(
            "outliers_score", ascending=False
        )

    def detect_outliers(self, row) -> list:
        """List of outliers from a dataframe row."""
        row = [i for i in row if i != np.nan]
        q1, q3 = np.percentile(row, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        return [x for x in row if x < lower_bound or x > upper_bound]

    def outliers_score(self, row) -> float:
        """Quantify outlier distance."""
        outliers = self.detect_outliers(row)
        if not outliers:
            return 0
        mean = np.mean(row)
        score = sum([abs(x - mean) for x in outliers]) / len(outliers)
        return score


class FeatureSizeImpact:
    """Positive and negative feature impact on kernel size."""

    def __init__(self) -> None:
        """Cache and config."""
        self._cache = cache()
        self._config = get_config()

    def get_mean_sizes_dataframe(self, version: str, target: str) -> pd.DataFrame:
        """Dataframe presentation for options mean sizes by value."""
        return pd.DataFrame.from_dict(
            self._get_mean_size_by_value(version=version, target=target), orient="index"
        ).sort_values("diff", ascending=False)

    def plot_mean_sizes(self, version: str, target: str, count: int = 20) -> None:
        """Plot mean sizes."""
        plt.figure(figsize=(10, 10))
        df = self.get_mean_sizes_dataframe(version=version, target=target)
        df_more = df.head(count)
        df_less = df.tail(count)

        plt.subplot(2, 1, 1)

        plt.title("version")

        sns.set_color_codes("muted")
        sns.barplot(df_more, y=df_more.index, x="yes", color="r")
        sns.set_color_codes("pastel")
        sns.barplot(df_more, y=df_more.index, x="no", color="b")
        plt.title("Bigger")

        plt.subplot(2, 1, 2)
        sns.set_color_codes("pastel")
        sns.barplot(df_less, y=df_less.index, x="no", color="r")
        sns.set_color_codes("muted")
        sns.barplot(df_less, y=df_less.index, x="yes", color="b")
        plt.title("Smaller")
        plt.show()

    def _get_mean_size_by_value(self, version: str, target: str) -> dict:
        """Get mean sizes for yes, no (and diff)."""
        key = f"size_diffs_dict_{version}_{target}."
        if key in self._cache:
            return self._cache[key]
        else:
            df, groups = Dataset(version).get_dataframe(
                Columns.options, return_collinear_groups=True, add_features=False
            )
            features = df.columns
            df[target] = Dataset(version).get_dataframe(Columns.targets)[target]
            sizes = dict()
            for feature in tqdm(features):
                try:
                    yes_size = df[df[feature]][target].mean()
                    no_size = df[~df[feature]][target].mean()
                    group = ", ".join(groups[feature]) if feature in groups else feature
                    sizes[group] = {
                        "yes": yes_size,
                        "no": no_size,
                        "diff": yes_size - no_size,
                    }
                except KeyError:
                    LOG.debug(f"not an original feature, skip: {feature}")
            self._cache[key] = sizes
            return sizes


class OverviewReport:
    """All other reports and more."""

    def __init__(
        self,
        versions: list[str] | None = None,
        targets: list[str] | None = None,
        fir_cache: str | None = None,
    ) -> None:
        """Get data for all features once."""
        self._config = get_config()

        self._versions = versions if versions else self._config["report"]["versions"]
        self._versions_p = [
            f"{str(version)[:1]}.{str(version)[1:]}" for version in self._versions
        ]
        self._targets = targets if targets else self._config["report"]["targets"]

        ## debug
        c = cache()

        # self._missing_options = self._get_missing_options()
        # self._ranks_and_yes_frequencies = self._get_ranks_and_yes_frequencies(
        #     fir_cache=fir_cache
        # )
        # self._groups = self._get_groups(fir_cache=fir_cache)
        # c["missing_options"] = self._missing_options
        # c["ryf"] = self._ranks_and_yes_frequencies
        # c["groups"] = self._groups

        self._missing_options = c["missing_options"]
        self._ranks_and_yes_frequencies = c["ryf"]
        self._groups = c["groups"]

        ## /debug

    def __getitem__(self, option: str):
        """Get available data for this option."""
        compressed_target = self._config["report"]["unique_compressed_target"]
        versions = self._config["report"]["versions"]
        ranks_yes_vmlinux = self._ranks_and_yes_frequencies[VMLINUX_TARGET].loc[option]
        ranks_yes_comp = self._ranks_and_yes_frequencies[compressed_target].loc[option]

        missing_in_dataset = [
            version
            for version, options in self._missing_options.items()
            if option in options
        ]
        # TODO: outliers
        # TODO: feature impact
        rank_i = len(versions)
        freq_i = 2 * len(versions)
        item_data = {
            "option": option,
            "missing_in_dataset": missing_in_dataset,
            "collinear_groups": self._groups["collinear_groups"].get(option, []),
            "merged_groups": self._groups["merged_groups"].get(option, option),
            "ranks": {
                "uncomp": {
                    version: value
                    for version, value in dict(ranks_yes_vmlinux.iloc[:rank_i]).items()
                    if isinstance(value, float)
                },
                "comp": {
                    version: value
                    for version, value in dict(ranks_yes_comp.iloc[:rank_i]).items()
                    if isinstance(value, float)
                },
            },
            "yes_freq": {
                "uncomp": {
                    version: value
                    for version, value in dict(
                        ranks_yes_vmlinux.iloc[rank_i:freq_i]
                    ).items()
                    if isinstance(value, float)
                },
                "comp": {
                    version: value
                    for version, value in dict(
                        ranks_yes_comp.iloc[rank_i:freq_i]
                    ).items()
                    if isinstance(value, float)
                },
            },
        }
        return self._analysis(item_data)

    def _analysis(self, item: dict) -> dict:
        """Analyse collected data for a single option."""

        def detect_outliers(ver_dict: dict[str, float]) -> dict[str, float]:
            """List of outliers from a dict: returns version(s)."""
            # row = [i for i in row if i != np.nan]
            q1, q3 = np.percentile(list(ver_dict.values()), [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            return {
                ver: val
                for ver, val in ver_dict.items()
                if val < lower_bound or val > upper_bound
            }

        def outliers_and_score(ver_dict: dict[str, float]) -> tuple[list[float], float]:
            """Quantify outlier distance."""
            outliers = detect_outliers(ver_dict)
            if not outliers:
                return outliers, 0.0
            mean = np.mean(list(ver_dict.values()))
            score = sum([abs(val - mean) for val in outliers.values()]) / len(outliers)
            return outliers, score

        def clean_ver_dict(ver_dict: dict[str, float | int]) -> dict[str, float | int]:
            """Clean key/values."""
            return {
                version.replace("freq ", ""): (
                    round(value, 4) if version.startswith("freq") else int(value)
                )
                for version, value in ver_dict.items()
            }

        def ver_stats(ver_dict: dict[str, float | int]) -> dict:
            ver_dict_ = clean_ver_dict(ver_dict)
            min_value = min(ver_dict_.values())
            max_value = max(ver_dict_.values())
            outliers, outliers_score = outliers_and_score(ver_dict_)
            return {
                "min": {
                    "value": min_value,
                    "versions": [
                        version
                        for version, value in ver_dict_.items()
                        if value == min_value
                    ],
                },
                "max": {
                    "value": max_value,
                    "versions": [
                        version
                        for version, value in ver_dict_.items()
                        if value == max_value
                    ],
                },
                "outliers": outliers,
                "outliers_score": outliers_score,
            }

        ana_res = dict()
        raw_res = dict()
        # ranks and yes freq
        for data in ("ranks", "yes_freq"):
            raw_res[data] = dict()
            for target in ("comp", "uncomp"):
                raw_res[data][target] = ver_stats(item[data][target])
                raw_res[data][target]["all"] = clean_ver_dict(item[data][target])

        # remaining
        for key in (
            "missing_in_dataset",
            "missing_version",
            "collinear_groups",
            "merged_groups",
        ):
            if key in item:
                raw_res[key] = item[key]

        # analysis
        ana_score = 0

        # missing data
        if raw_res["missing_in_dataset"]:
            ana_res[
                "missing_dataset"
            ] = f"Version(s) removed from dataset: {', '.join(raw_res['missing_in_dataset'])}"

        ver_diff = set(self._versions_p) - set(item["ranks"]["uncomp"].keys())
        if raw_res["missing_in_dataset"]:
            ver_diff -= set(raw_res["missing_in_dataset"])
        if ver_diff:
            ana_res[
                "missing_version"
            ] = f"Missing in versions: {', '.join(sorted(ver_diff))}"

        # groups
        if raw_res["collinear_groups"]:
            ana_res[
                "collinear_groups"
            ] = f"Collinear options: {raw_res['collinear_groups']}"

        if raw_res["merged_groups"] != item["option"]:
            ana_res[
                "merged_groups"
            ] = f"Collinearity across versions: {raw_res['merged_groups']}"

        # common rank/yes freq analysis
        for data in ("ranks", "yes_freq"):
            for target in ("comp", "uncomp"):
                dt_item = raw_res[data][target]
                min_str = f"min: {dt_item['min']['value']} ({', '.join(dt_item['min']['versions'])})"
                max_str = f"max: {dt_item['max']['value']} ({', '.join(dt_item['max']['versions'])})"
                out_str = (
                    f"{', '.join(str(val) for val in dt_item['outliers'].values())} "
                )
                out_str += f"({', '.join(sorted(dt_item['outliers'].keys()))})"
                dt_str = ""
                if dt_item["outliers"]:
                    dt_str = f"Outlier found: {out_str}. "
                dt_str += f"{min_str}, {max_str}"
                ana_res[f"{data}-{target}"] = dt_str

        # yes freq analysis
        threshold = self._config["report"]["yes_freq_threshold"]
        for target in ("comp", "uncomp"):
            # dt_item = raw_res["yes_freq"][target]
            if thr_freq := {
                version: freq
                for version, freq in raw_res["yes_freq"][target]["all"].items()
                if freq < threshold or freq > (1 - threshold)
            }:
                hf_str = ", ".join(
                    [
                        f"{'YES' if freq > 0.5 else 'NO'}: {100 * (freq if freq > 0.5 else 1 - freq):.2f}% ({version})"
                        for version, freq in thr_freq.items()
                    ]
                )
                ana_res[
                    f"yes_freq-{target}-high frequency"
                ] = f"High frequency: {hf_str}"

        return {"option": item["option"], "raw": raw_res, "analysis": ana_res}

    def _get_groups(self, fir_cache: str) -> dict[str, list[str]]:
        """Get options groups and similarities."""
        colls = dict()
        fir = FeatureImportanceReport(use_cache=fir_cache)
        merged_groups = fir.merged_groups(target=VMLINUX_TARGET, collinearity=True)
        for version in self._versions:

            # df_ai = fir.options_always_importants(
            #     target=VMLINUX_TARGET,
            #     rank=999999,
            #     collinearity=True,
            #     merge_groups=True,
            #     allow_version_gap=True,
            # ).set_index("options")

            ds = Dataset(version)
            _, coll = ds.get_dataframe(
                Columns.options,
                group_collinear_options=True,
                add_features=False,
                return_collinear_groups=True,
            )
            vcolls = dict()
            for options in coll.values():
                for option in options:
                    vcolls[option] = options
            colls[version] = vcolls
        return {"merged_groups": merged_groups, "collinear_groups": colls}

    def _get_missing_options(self) -> dict[str, list[str]]:
        """List of missing options for each target."""
        res = dict()
        for version in self._versions:
            ds = Dataset(version)
            df_options = ds.get_dataframe(
                Columns.options, group_collinear_options=False, add_features=False
            ).columns
            all_options = ds.raw_option_list()

            res[str(version / 100)] = list(set(all_options) - set(df_options))
        return res

    def _get_ranks_and_yes_frequencies_deprecated(
        self, fir_cache: str
    ) -> dict[str, pd.DataFrame]:
        """Get all ranks and frequencies."""
        # fir = FeatureImportanceReport(use_cache=fir_cache)
        yfo = YesFrequencyOutliers()
        res = dict()
        for target in self._targets:
            res[target] = yfo.get_dataframe(
                target=target, use_fir_cache=fir_cache, threshold=0.0
            )
            # df_ai = fir.options_always_importants(
            #     target=target,
            #     rank=999999,
            #     collinearity=True,
            #     merge_groups=True,
            #     allow_version_gap=True,
            # ).set_index("options")
            # res[target] = fir.add_yes_frequencies(df_ai, str_pct=False)
        return res

    def _get_ranks_and_yes_frequencies(self, fir_cache: str) -> dict[str, pd.DataFrame]:
        """Get all ranks and frequencies."""
        res = dict()
        fir = FeatureImportanceReport(use_cache=fir_cache)
        for target in self._targets:
            df_ai = fir.options_always_importants(
                target=target,
                rank=99999,
                collinearity=False,
                merge_groups=False,
                allow_version_gap=True,
            ).set_index("options")
            df = fir.add_yes_frequencies(df_ai, str_pct=False, collinearity=False)

            freq_cols = [col for col in df.columns if col.startswith("freq")]
            df["outliers_score"] = df[freq_cols].apply(
                lambda row: self._outliers_score(row), axis=1
            )
            res[target] = df

        return res

    def _detect_outliers(self, row) -> list:
        """List of outliers from a dataframe row."""
        row = [i for i in row if i != np.nan]
        q1, q3 = np.percentile(row, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        return [x for x in row if x < lower_bound or x > upper_bound]

    def _outliers_score(self, row) -> float:
        """Quantify outlier distance."""
        outliers = self._detect_outliers(row)
        if not outliers:
            return 0
        mean = np.mean(row)
        score = sum([abs(x - mean) for x in outliers]) / len(outliers)
        return score


if __name__ == "__main__":
    from tuxai.misc import config_logger

    config_logger()
    orep = OverviewReport(fir_cache="fi_const_2023")
    orep["UBSAN_NULL"]
    orep["KASAN"]
    orep["CC_OPTIMIZE_FOR_SIZE"]

    # fsi = FeatureSizeImpact()
    # for version in tqdm(get_config()["report"]["versions"]):
    #     for target in ("vmlinux", "XZ"):
    #         fsi.get_mean_sizes_dataframe(version=version, target=target)

    # fir = FeatureImportanceReport(use_cache="fi_const_2023_do", drop_outliers=True)
    # # oai = fir.options_always_importants(
    # #     "vmlinux", merge_groups=False, allow_version_gap=False
    # # )
    # nai = fir.options_not_always_importants(
    #     target="vmlinux",
    #     best_rank=50,
    #     worst_rank=500,
    #     collinearity=True,
    #     merge_groups=True,
    #     allow_version_gap=False,
    # ).set_index("options")
    # print(nai)
