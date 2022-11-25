"""Collection of miscellaneous functions."""

from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler

import tomli

DEFAULT_CONFIG_FILE = "config.toml"


def config() -> dict:
    """Get config file."""
    config_path = Path(__file__).resolve().parent.parent / DEFAULT_CONFIG_FILE
    with open(config_path, "rb") as config_file:
        return tomli.load(config_file)


def config_logger():
    """Terminal + files configuration."""
    # read config
    log_config = config()["log"]
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


if __name__ == "__main__":
    print(config())
