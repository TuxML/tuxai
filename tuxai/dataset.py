"""Dataset management."""

from pathlib import Path
import pickle
import logging
from enum import Enum, unique

import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

from tuxai.misc import get_config, config_logger, filter_options
from tuxai.features import Collinearity, FeatureEngineering

LOG = logging.getLogger(__name__)


@unique
class Columns(Enum):
    """DataFrame column selection."""

    all = 0
    options = 1  # ALL - (TARGET + EXTRA)
    targets = 2  # cf config file
    extra = 3  # cf config file


class Dataset:
    """Dataset management."""

    def __init__(self, version: int | float | str) -> None:
        """Convert version to string (413 == 4.13 == "413" == "4.13")."""
        self._version = str(version).replace(".", "")
        self._config = get_config()
        self.collinear_options_: dict[str, list[str]] | None = None

    @property
    def version(self) -> str:
        """Get kernel version as a string (read only)."""
        return self._version

    def get_dataframe(
        self,
        col_filter: Columns = Columns.all,
        group_collinear_options: bool = True,
        collinearity_threshold: float = 0.0,
        return_collinear_groups: bool = False,
        add_features: bool = True,
        nb_yes_range: tuple[float, float] = (0, 1),
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, list[str]]]:
        """Get raw Pandas DataFrame.

        If parquet file is not found, try with pickle.
        """
        LOG.debug(f"[{self._version}] loading dataframe")
        parquet = (
            Path(self._config["dataset"]["path"]) / f"dataset_{self._version}.parquet"
        )
        # parquet not found ?
        # get pkl and convert it for next time
        if not parquet.exists():
            LOG.info(f"[{self._version}] converting pkl to parquet")
            with open(parquet.with_suffix(".pkl"), "rb") as f:
                df = pickle.load(f)
                for target in self._config["dataframe"]["targets"]:
                    df[target] /= 1024 * 1024
                    df[target] = df[target].astype("float")
                for col in tqdm(filter_options(df.columns, config=self._config)):
                    df[col] = df[col].astype("bool")
            df.to_parquet(parquet, engine="pyarrow")
        df = pd.read_parquet(parquet, engine="pyarrow").drop_duplicates()
        if return_collinear_groups and col_filter in (Columns.targets, Columns.extra):
            LOG.warning(f"collinearity not supported for col_filter={col_filter.name}")

        match col_filter:
            case Columns.all:
                df = (
                    self._group_colinar_options(
                        dataframe=df,
                        threshold=collinearity_threshold,
                        return_groups=return_collinear_groups,
                    )
                    if group_collinear_options
                    else df
                )
                if add_features:
                    self._add_features(dataframe=df, nb_yes_range=nb_yes_range)
                return df
            case Columns.targets:
                return df[self._config["dataframe"]["targets"]]
            case Columns.extra:
                return df[self._config["dataframe"]["extras"]]
            case Columns.options:
                options = filter_options(df.columns, config=self._config)
                df = df[options]
                df = (
                    self._group_colinar_options(
                        dataframe=df,
                        threshold=collinearity_threshold,
                        return_groups=return_collinear_groups,
                    )
                    if group_collinear_options
                    else df
                )
                if add_features:
                    self._add_features(dataframe=df, nb_yes_range=nb_yes_range)
                return df

    def _add_features(
        self, dataframe: pd.DataFrame, nb_yes_range: tuple[float, float]
    ) -> pd.DataFrame:
        """Feature engineering: add everything available."""
        LOG.debug("add features")
        feature_engineering = FeatureEngineering(dataframe)
        return feature_engineering.add_nb_yes_feature(val_range=nb_yes_range)

    def _group_colinar_options(
        self,
        dataframe: pd.DataFrame,
        threshold: float,
        return_groups: bool,
    ) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        """Return dataframe with grouped options + group details."""
        # split options from other columns
        options = filter_options(dataframe.columns)
        not_options = [col for col in dataframe.columns if col not in options]
        df_options = dataframe[options]
        df_not_options = dataframe[not_options]
        # collinearity on options
        df_options, self.collinear_options_ = Collinearity(
            df_options
        ).group_correlated_features(threshold)
        # get other columns back
        df = (
            df_options
            if df_not_options.empty
            else pd.concat([df_options, df_not_options], axis=1)
        )

        if return_groups:
            return df, self.collinear_options_
        return df

    def train_test_split(
        self, test_size: float = 0.2, target: str = "vmlinux", **kwargs
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Get X_train, y_train, X_test, y_test."""
        df = self.get_dataframe(**kwargs)
        df_train, df_test = train_test_split(df, test_size=test_size, random_state=2022)
        options = filter_options(df.columns, config=self._config)
        X_train = df_train[options]
        y_train = df_train[target]
        X_test = df_test[options]
        y_test = df_test[target]

        return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    config_logger()
    LOG.info("log test")
    Dataset(508).get_dataframe(add_features=True, nb_yes_range=(0, 1))

    # precompute
    # for ver in (413, 415, 420, 500, 504, 507, 508):
    #     dataset = Dataset(ver)
    #     dataset.get_dataframe()

    # df, groups = Dataset(508).get_dataframe(return_collinear_groups=True)
    Dataset(508).train_test_split()
    # print(groups)
    # dataset.filter_correlated()
