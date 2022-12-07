"""Only XGBoost for now."""

import logging
import tempfile
from pathlib import Path

import xgboost as xgb
import pandas as pd
import hashlib

from tuxai.dataset import Dataset
from tuxai.featureselection import CORR_PREFIX
from tuxai.misc import cache

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

    def __init__(
        self,
        dataset: Dataset,
        use_cache: bool = True,
        target: str = "vmlinux",
        group_collinear_options: bool = True,
        collinearity_threshold: float = 0.0,
        alpha: float = 6.0,
        gamma: float = 2.0,
        lambda_: float = 2.0,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        eval_metric: str = "mape",
    ) -> None:
        """Remaining parameters are sent to model constructor."""
        self._dataset = dataset
        self._parameters = {
            "alpha": alpha,
            "gamma": gamma,
            "lambda": lambda_,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "eval_metric": eval_metric,
        }

        self.X_train, self.y_train, self.X_test, self.y_test = dataset.train_test_split(
            target=target,
            group_collinear_options=group_collinear_options,
            collinearity_threshold=collinearity_threshold,
        )

        self._is_trained = False
        self._cache = cache() if use_cache else None

        self._xgb_reg = xgb.XGBRegressor(random_state=2022, **self._parameters)
        self._signature = self._get_signature()

    def fit(self) -> None:
        """Train model."""
        if self._is_trained:
            LOG.debug("model already trained.")
            return
        if self._cache:
            if self._load():
                self._is_trained = True
                return

        LOG.info(f"training model {self._signature}")

        # fit with sorted options (/ pred)
        self._xgb_reg.fit(
            self.X_train.reindex(sorted(self.X_train.columns), axis=1), self.y_train
        )
        self._is_trained = True
        LOG.debug("model trained")
        if self._cache:
            self._save()

    def pred(self) -> list:
        """Get y_pred."""
        LOG.debug("make prediction.")
        # predict with sorted options (/ fit) -> mandatory after loading model from cache
        return list(
            self._xgb_reg.predict(
                self.X_test.reindex(sorted(self.X_test.columns), axis=1)
            )
        )

    def options_scores(
        self,
        corr_only: bool = False,
        corr_groups: bool = True,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Get more important options for prediction."""
        LOG.debug("compute features importance.")
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

    def _get_signature(self) -> str:
        """Get model signature (training data + parameters)."""
        LOG.debug("computing model signature...")
        param = "|".join(
            f"({key}:{self._parameters[key]})"
            for key in sorted(self._parameters.keys())
        )
        # X_train.describe().to_string() is not stable enough for a key,
        # so, we just take columns and hope for the best...
        data = (
            ", ".join(sorted(self.X_train.columns))
            + self.y_train.describe().to_string()
        )

        signature = f"xgboost({self._dataset.version})|{hashlib.md5((param + data).encode()).hexdigest()}"
        LOG.debug(f"model signature : {signature}")
        return signature

    def _save(self) -> None:
        """Serialize model in cache."""
        # create temporary file, read content and store in cache
        with tempfile.TemporaryDirectory() as temp_dir:
            LOG.debug(f"saving model: {self._signature}")
            json_path = Path(temp_dir) / "model.json"
            self._xgb_reg.save_model(json_path)
            self._cache[self._signature] = json_path.read_text()

    def _load(self) -> bool:
        """Load model from cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            if found := self._signature in self._cache:
                LOG.debug(f"loading model: {self._signature}")
                json_path = Path(temp_dir) / "model.json"
                json_path.write_text(self._cache[self._signature])
                self._xgb_reg.load_model(json_path)
            else:
                LOG.debug(f"model not found : {self._signature}")
            return found


if __name__ == "__main__":
    from tuxai.misc import config_logger

    config_logger()
    dataset = Dataset(508)
    for _ in range(2):
        model = XGBoost(dataset)
        model.fit()
        model.pred()

    # model._signature()
    # print(model.options_scores(limit=10))
