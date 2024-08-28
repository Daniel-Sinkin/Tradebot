import logging
import os
import sqlite3
from typing import TYPE_CHECKING

import colorlog

from .constants import _Paths

if TYPE_CHECKING:
    from sqlite3.dbapi2 import _Parameters

os.makedirs(_Paths.LOGS, exist_ok=True)

# Define the color scheme for each log level
log_colors = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_red",
}

# Create a ColoredFormatter with the defined color scheme
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors=log_colors,
)

# Set up the root logger
handler = logging.StreamHandler()
handler.setFormatter(formatter)

# File handler (without colors)
file_handler = logging.FileHandler(_Paths.LOGS.joinpath("main.log"))
file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        handler,
        file_handler,
    ],
)

logger = logging.getLogger(__name__)


def cursor_execute(
    cursor: sqlite3.Cursor, command: str, parameters: "_Parameters" = ()
) -> any:
    logger.debug(
        "Executed the following SQL command:\n%s\nWith parameters\n%s\n",
        command,
        str(parameters),
    )
    cursor.execute(command, parameters)


if __name__ == "__main__":
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
