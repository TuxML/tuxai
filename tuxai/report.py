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

from tuxai.misc import date2dir, config
from tuxai.dataset import Dataset
from tuxai.models import XGBoost
from tuxai.featureselection import CORR_PREFIX

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
        self._config = config()

    def feature_importance_deprecated(self) -> pd.DataFrame:
        """Generate feature importance report."""
        LOG.info("generating feature importance report")
        for version in (pbar := tqdm(self._config["report"]["versions"])):
            pbar.set_description(self._version_str(version))
            dataset = Dataset(version)
            xgb = XGBoost(dataset)
            xgb.fit()
            return xgb.options_scores(limit=self._config["report"]["feature_count"])

    def feature_importance(
        self,
        versions: list[str] | None = None,
        targets: list[str] | None = None,
        group_collinear_options: bool = True,
    ) -> pd.DataFrame:
        """Generate feature importance report."""
        LOG.info("generating feature importance report")

        def _callback(xgb: xgboost.XGBRegressor) -> dict[str, float]:
            """Retur metrics."""
            return xgb.options_scores(limit=self._config["report"]["feature_count"])

        return self._all_config(
            callback=_callback,
            versions=versions,
            targets=targets,
            group_collinear_options=group_collinear_options,
        )

    def xgboost_model(
        self,
        versions: list[str] | None = None,
        targets: list[str] | None = None,
        group_collinear_options: bool = True,
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
            group_collinear_options=group_collinear_options,
        )

    def feature_importance_stability(
        self,
        versions: list[str] | None = None,
        targets: list[str] | None = None,
        group_collinear_options: bool = True,
        top_count: int | None = None,
    ) -> dict:
        """Compare feature importance between each configuration."""
        # pick all configurations
        df = self.feature_importance(
            version=versions,
            targets=targets,
            group_collinear_options=group_collinear_options,
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
        group_collinear_options: bool = True,
    ) -> pd.DataFrame:
        """Test and generate report for each version/target/options (common loop for reports).

        If not specified, get parameters from config file.
        """
        versions = self._config["report"]["versions"] if versions is None else versions
        targets = (
            self._config["report"]["xgboost_targets"] if targets is None else targets
        )
        group_collinear_options_list = (
            (False, True) if group_collinear_options else (False,)
        )
        res = list()
        for version in (pbar_0 := tqdm(versions, position=0)):
            dataset = Dataset(version)
            for target in (pbar_1 := tqdm(targets, position=1, leave=False)):
                for group_collinear_options in group_collinear_options_list:
                    coll_str = (
                        "with collinearity"
                        if group_collinear_options
                        else "no collinearity"
                    )
                    pbar_0.set_description(self._version_str(version))
                    pbar_1.set_description(f"{target} - {coll_str}")

                    xgb = XGBoost(
                        dataset=dataset,
                        target=target,
                        group_collinear_options=group_collinear_options,
                    )
                    xgb.fit()
                    res.append(
                        {
                            "version": self._version_str(version),
                            "collinearity": group_collinear_options,
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


if __name__ == "__main__":
    from tuxai.misc import config_logger

    config_logger()
    df = Report().feature_importance_stability()
    print(df)
