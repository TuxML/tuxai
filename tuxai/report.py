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

from tuxai.misc import date2dir, get_config, cache  # , bio2df, df2bio
from tuxai.dataset import Dataset
from tuxai.models import XGBoost
from tuxai.features import CORR_PREFIX

LOG = logging.getLogger(__name__)

DEFAULT_REPORT_FILENAME = "report.xlsx"


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


class Report:
    """Generate excel report for each version, with different options."""

    def __init__(self, path: str | Path | None = None) -> None:
        """Provide a path or get generated one."""
        self._path = (
            Path(__file__).resolve().parent.parent
            / "reports"
            / date2dir()
            / DEFAULT_REPORT_FILENAME
        )
        self._config = get_config()

    def feature_importance(
        self,
        versions: list[str] | None = None,
        targets: list[str] | None = None,
        collinearities: list[bool] | None = None,
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

    def __init__(self, use_cache: str | None = None) -> None:
        """All raw data come from Report class."""
        # pick all configurations
        report = Report()

        if use_cache is None:
            df = report.feature_importance()
        else:
            cache_ = cache()
            if use_cache in cache_:
                LOG.info(f"loading feature importance data from cache: {use_cache}")
                # pyarrow cannot handle dicts in columns
                # df = bio2df(cache_[use_cache])
                df = cache_[use_cache]
            else:
                df = report.feature_importance()
                # pyarrow cannot handle dicts in columns
                # cache_[use_cache] = df2bio(df)
                cache_[use_cache] = df

        self._df = self._explode_df(df)

    def options_always_importants(
        self, target, rank: int = 30, collinearity: bool = True
    ) -> pd.DataFrame:
        """Options always < top N, regardless of version."""
        df = self._df.copy()
        df = self._keep(df, targets=[target], collinearities=[collinearity])

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
        self, target, rank: int = 300, collinearity: bool = True
    ) -> pd.DataFrame:
        """Options always > top N, regardless of version."""
        df = self._df.copy()
        df = self._keep(df, targets=[target], collinearities=[collinearity])

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
        self, target, best_rank: int = 30, worst_rank=300, collinearity: bool = True
    ) -> pd.DataFrame:
        """Options at least once <= best_rank and at least once >= worst_rank, regardless of version."""
        df = self._df.copy()
        df = self._keep(df, targets=[target], collinearities=[collinearity])

        # list of top N options
        # options = list()
        # for option in tqdm(df.option.unique()):
        #     ranks = sorted(df[df.option == option]["rank"].unique())
        #     if ranks[0] <= best_rank and ranks[-1] >= worst_rank:
        #         options.append(option)
        groups = list()
        for group in tqdm(df.group.unique()):
            ranks = sorted(df[df.group == group]["rank"].unique())
            if ranks[0] <= best_rank and ranks[-1] >= worst_rank:
                groups.append(group)

        return self._display_rank_by_version(df[df.group.isin(groups)], ascending=True)

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
        df = (
            dataframe.groupby(["target", "group"])[["version", "rank"]]
            .agg(list)
            .reset_index()
        )
        # convert to dict because it's easier to manipulate here
        split_cols = dict()
        for idx, item in df.to_dict(orient="index").items():
            split_cols[idx] = {"options": item["group"]}
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


if __name__ == "__main__":
    from tuxai.misc import config_logger

    config_logger()
    fir = FeatureImportanceReport(use_cache="fi_const_2022_12_21")
    fir.target_comparison("vmlinux", "BZIP2-vmlinux", 30, True)
    print(fir._df)
