"""Only XGBoost for now."""

import xgboost as xgb
import numpy as np
import pandas as pd
import logging

from tuxai.dataset import Dataset
from tuxai.featureselection import CORR_PREFIX

LOG = logging.getLogger(__name__)

DEFAULT_XGBOOST_PARAMS = {
    "alpha": 6,
    "gamma": 2,
    "lambda": 2,
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "eval_metric": "mape",
}


class XGBoost:
    """Extreme Gradient Boosting."""

    def __init__(self, dataset: Dataset, **kwargs) -> None:
        """Remaining parameters are sent to model constructor."""
        self._dataset = dataset
        self._xgb_reg = xgb.XGBRegressor(
            **(DEFAULT_XGBOOST_PARAMS if not kwargs else kwargs)
        )

        # X_train, y_train, X_test, y_test
        self._data_split = dataset.train_test_split()

    def fit(self) -> None:
        """Train model."""
        LOG.info("fit model")
        X_train, y_train, _, _ = self._data_split
        self._xgb_reg.fit(X_train, y_train)

    def pred(self) -> np.ndarray:
        """Get y_pred."""
        _, _, X_test, _ = self._data_split
        return self._xgb_reg.predict(X_test)

    def options_scores(
        self,
        corr_only: bool = False,
        corr_groups: bool = True,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Get more important options for prediction."""
        scores = self._xgb_reg.get_booster().get_score(importance_type="weight")
        df_scores = pd.DataFrame.from_dict(
            {"option": scores.keys(), "importance": scores.values()}
        ).sort_values("importance", ascending=False)
        df_scores["position"] = range(1, len(scores) + 1)
        if corr_only:
            df_scores = df_scores[df_scores.option.str.startswith(CORR_PREFIX)]
        if limit is not None:
            df_scores = df_scores.iloc[:limit]
        if corr_groups:
            collinear_options = self._dataset.collinear_options_
            if collinear_options:
                df_scores["groups"] = df_scores.option.apply(
                    lambda option: (", ".join(collinear_options[option]))
                    if option in collinear_options
                    else option
                )
        return df_scores


if __name__ == "__main__":
    from tuxai.misc import config_logger

    config_logger()

    params = {
        "max_depth": 5,
        "alpha": 10,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "eval_metric": "mape",
    }

    dataset = Dataset(508)

    model = XGBoost(dataset, **params)
    model.fit()
    print(model.options_scores(limit=10))
