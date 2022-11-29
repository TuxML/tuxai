"""Dataset management."""

from pathlib import Path
import pickle
import logging
from enum import Enum, unique


import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

from tuxai.misc import config, config_logger

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
        self._config = config()

    @property
    def version(self) -> str:
        """Get kernel version as a string (read only)."""
        return self._version

    def get_dataframe(self, col_filter: Columns = Columns.all) -> pd.DataFrame:
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
                for col in tqdm(self._options(df)):
                    # df[col] = df[col].astype("category")
                    df[col] = df[col].astype("bool")
            df.to_parquet(parquet, engine="pyarrow")
        df = pd.read_parquet(parquet, engine="pyarrow").drop_duplicates()
        match col_filter:
            case Columns.all:
                return df
            case Columns.targets:
                return df[self._config["dataframe"]["targets"]]
            case Columns.extra:
                return df[self._config["dataframe"]["extras"]]
            case Columns.options:
                return df[self._options(df.columns)]

    def train_test_split(
        self, test_size: float = 0.2, target: str = "vmlinux"
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Get X_train, y_train, X_test, y_test."""
        df = self.get_dataframe()
        df_train, df_test = train_test_split(df, test_size=test_size)
        options = self._options(df.columns)
        X_train = df_train[options]
        y_train = df_train[target]
        X_test = df_test[options]
        y_test = df_test[target]
        return X_train, y_train, X_test, y_test

    def _options(self, columns: list[str]) -> list[str]:
        """Keep options only."""
        targets = self._config["dataframe"]["targets"]
        extras = self._config["dataframe"]["extras"]
        return [col for col in columns if col not in targets and col not in extras]


if __name__ == "__main__":
    config_logger()
    LOG.info("log test")

    # precompute
    # for ver in (413, 415, 420, 500, 504, 507, 508):
    #     dataset = Dataset(ver)
    #     dataset.get_dataframe()

    dataset = Dataset(413)
    # dataset.filter_correlated()
