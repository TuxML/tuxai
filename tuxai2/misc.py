"""Collection of miscellaneous functions."""

# from datetime import datetime
from io import BytesIO

from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler

import pandas as pd

from diskcache import Cache
from cachetools import cached

import tomli

DEFAULT_CONFIG_FILE = "config.toml"


@cached(cache={})
def get_config() -> dict:
    """Get config file."""
    config_path = Path(__file__).resolve().parent.parent / DEFAULT_CONFIG_FILE
    with open(config_path, "rb") as config_file:
        return tomli.load(config_file)


def cache() -> Cache:
    """Get diskcache."""
    cache_config = get_config()["diskcache"]
    return Cache(cache_config["path"], size_limit=int(cache_config["size_limit"]))


def config_logger():
    """Terminal + files configuration."""
    # read config
    log_config = get_config()["log"]
    log_path = log_config["path"]
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    terminal_log_level = log_config["terminal_level"].upper()
    file_level = log_config["file_level"].upper()
    file_size = log_config["file_size"]
    file_count = log_config["file_count"]

    # common config
    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s :: %(levelname)s :: %(name)s :: %(message)s"
    )

    # terminal config
    terminal_handler = logging.StreamHandler()
    terminal_handler.setFormatter(formatter)
    terminal_handler.setLevel(getattr(logging, terminal_log_level))
    logger.addHandler(terminal_handler)

    # files config
    file_handler = RotatingFileHandler(
        log_path,
        mode="a",
        maxBytes=file_size,
        backupCount=file_count,
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, file_level))
    logger.addHandler(file_handler)

    logging.getLogger("matplotlib").setLevel(logging.CRITICAL)


# def date2dir(dt: datetime | None = None, hm: bool = True) -> str:
#     """Generate directory name based on date."""
#     return (datetime.now() if dt is None else dt).strftime(
#         "%Y_%m_%d_%H_%M" if hm else "%Y_%m_%d"
#     )


def filter_options(columns: list[str], config: dict | None = None) -> list[str]:
    """Keep options only."""
    if config is None:
        config = get_config()
    targets = config["dataframe"]["targets"]
    extras = config["dataframe"]["extras"]
    # CORR features are kept
    return [col for col in columns if col not in targets and col not in extras]


def df2bio(df: pd.DataFrame) -> BytesIO:
    """Transform DataFrame to BytesIO."""
    return BytesIO(df.to_parquet(engine="pyarrow"))


def bio2df(bio: BytesIO) -> pd.DataFrame:
    """Transform BytesIO to DataFrame."""
    return pd.read_parquet(bio)


def version_str(version: int | str | float) -> str:
    """Convert version to string."""
    str_ver = str(version)
    if "." not in str_ver:
        str_ver = f"{str_ver[0]}.{str_ver[1:]}"
    return str_ver


class AutoVivification(dict):
    """Auto vivification dict."""

    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


if __name__ == "__main__":
    print(get_config())
