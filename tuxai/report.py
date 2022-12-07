"""Generate metrics, docs, plots, etc."""

from pathlib import Path
import logging

import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
from tqdm.auto import tqdm

from tuxai.misc import date2dir, config
from tuxai.dataset import Dataset
from tuxai.models import XGBoost

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

    # def generate(self):
    #     """Create excel file."""
    #     LOG.info(f"generating excel report: {self._path}")
    #     self._path.parent.mkdir(parents=True, exist_ok=True)
    #     with pd.ExcelWriter(self._path, engine="xlsxwriter") as writer:
    #         # trained model are stored in cache disk,
    #         # so multiple fit call should not be a problem
    #         if self._config["report"]["xgboost"]:
    #             self._xgboost_model(writer)
    #         if self._config["report"]["feature_importance"]:
    #             self._feature_importance(writer)

    def feature_importance(self) -> pd.DataFrame:
        """Generate feature importance report."""
        LOG.info("generating feature importance report")
        for version in (pbar := tqdm(self._config["report"]["versions"])):
            pbar.set_description(self._version_str(version))
            dataset = Dataset(version)
            xgb = XGBoost(dataset)
            xgb.fit()
            return xgb.options_scores(limit=self._config["report"]["feature_count"])

    def xgboost_model(
        self,
        versions: list[str] | None = None,
        targets: list[str] | None = None,
        group_collinear_options: bool = True,
    ) -> dict:
        """Test and generate report for each version/target/options.

        If not specified, get parameters from config file.
        """
        LOG.info("generating xgboost report")
        versions = self._config["report"]["versions"] if versions is None else versions
        targets = (
            self._config["report"]["xgboost_targets"] if targets is None else targets
        )
        group_collinear_options_list = (
            (False, True) if group_collinear_options else (False,)
        )
        res = list()
        for version in (pbar := tqdm(versions)):
            dataset = Dataset(version)
            for target in targets:
                for group_collinear_options in group_collinear_options_list:
                    coll_str = (
                        "with collinearity"
                        if group_collinear_options
                        else "no collinearity"
                    )
                    pbar.set_description(
                        f"{self._version_str(version)} - {target} - {coll_str}"
                    )

                    xgb = XGBoost(
                        dataset=dataset,
                        target=target,
                        group_collinear_options=group_collinear_options,
                    )
                    xgb.fit()
                    metrics = model_metrics(y_pred=xgb.pred(), y_true=xgb.y_test)
                    res.append(
                        {
                            "version": self._version_str(version),
                            "collinearity": group_collinear_options,
                            "target": target,
                            **metrics,
                        }
                    )
        # TODO write res to excel
        return pd.DataFrame.from_dict(res)

    @staticmethod
    def _version_str(version: int | str) -> str:
        """Convert version before displaying."""
        str_ver = str(version)
        return f"{str_ver[0]}.{str_ver[1:]}"


if __name__ == "__main__":
    from tuxai.misc import config_logger

    config_logger()
    df = Report().xgboost_model()
    print(df)
