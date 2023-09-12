"""Only XGBoost for now."""

import logging
import tempfile
from pathlib import Path

from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
import hashlib

from tuxai.dataset import Dataset
from tuxai.features import CORR_PREFIX
from tuxai.misc import cache, filter_options

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
        dataframe: pd.DataFrame,
        test_size: float = 0.2,
        use_cache: bool = True,
        target: str = "vmlinux",
        alpha: float = 6.0,
        gamma: float = 2.0,
        lambda_: float = 2.0,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        eval_metric: str = "mape",
        version="",
        collinear_options: dict | None = None
        # **kwargs,
    ) -> None:
        """Remaining parameters are sent to model constructor."""
        self._dataframe = dataframe
        self._test_size = test_size
        self._target = target
        self._parameters = {
            "alpha": alpha,
            "gamma": gamma,
            "lambda": lambda_,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "eval_metric": eval_metric,
        }

        self.X_train, self.y_train, self.X_test, self.y_test = self._train_test_split()
        self._version = version
        self._test_size = test_size
        self._collinear_options = collinear_options

        self._is_trained = False
        self._cache = cache() if use_cache else None

        self._xgb_reg = xgb.XGBRegressor(random_state=2022, **self._parameters)
        self._signature = self._get_signature()

    def _train_test_split(self):
        """Train/Test datasets."""
        options = filter_options(self._dataframe.columns)
        if self._test_size > 0:
            df_train, df_test = train_test_split(
                self._dataframe, test_size=self._test_size, random_state=2022
            )
            X_test = df_test[options]
            y_test = df_test[self._target]
        else:
            df_train = self._dataframe
            X_test = None
            y_test = None

        X_train = df_train[options]
        y_train = df_train[self._target]

        return X_train, y_train, X_test, y_test

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
        as_dataframe: bool = False,
    ) -> pd.DataFrame | dict[str, dict]:
        """Get more important options for prediction."""
        LOG.debug("compute features importance.")
        scores = self._xgb_reg.get_booster().get_score(importance_type="weight")

        # xgboost/core.py: get_score(...) notes: "Zero-importance features will not be included"
        # -> add them manually
        scores.update(
            {option: 0.0 for option in (set(self.X_train.columns) - set(scores.keys()))}
        )

        df_scores = (
            pd.DataFrame.from_dict(
                {"option": scores.keys(), "importance": scores.values()}
            )
            .sort_values("importance", ascending=False)
            .reset_index()
            .drop(columns=["index"])
        )

        # df_scores["position"] = range(1, len(scores) + 1)
        if corr_only:
            df_scores = df_scores[df_scores.option.str.startswith(CORR_PREFIX)]
        if limit is not None:
            df_scores = df_scores.iloc[:limit]
        if corr_groups:
            # add original collinear column names
            collinear_options = (
                {} if self._collinear_options is None else self._collinear_options
            )
            df_scores["group"] = df_scores.option.apply(
                lambda option: collinear_options[option]
                if option in collinear_options
                else [option]
            )
        return (
            df_scores
            if as_dataframe
            else {k: v for k, v in df_scores.to_dict().items() if k != "index"}
        )

    def _get_signature(self) -> str:
        """Get model signature (training data + parameters)."""
        LOG.debug("computing model signature...")
        param = "|".join(
            str(val) for val in (self._test_size, self._target, self._version)
        )
        hyper_param = "|".join(
            f"({key}:{self._parameters[key]})"
            for key in sorted(self._parameters.keys())
        )
        # X_train.describe().to_string() is not stable enough for a key,
        # so, we just take columns and hope for the best...
        data = (
            ", ".join(sorted(self.X_train.columns))
            + self.y_train.describe().to_string()
        )

        signature = f"xgboost({self._version})|{hashlib.md5((param + data + hyper_param).encode()).hexdigest()}"
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
    dataset = Dataset(415)
    model = XGBoost(dataset, target="LZ4-vmlinux", group_collinear_options=True)
    model.fit()
    model.options_scores(corr_only=True, corr_groups=True)
