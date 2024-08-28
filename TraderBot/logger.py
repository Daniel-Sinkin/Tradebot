import logging
import os

import colorlog

from .constants import _Paths

os.makedirs(_Paths.LOGS, exist_ok=True)


def setup_logger():
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


setup_logger()
logger = logging.getLogger(__name__)
