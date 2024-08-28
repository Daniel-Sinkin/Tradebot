import logging

from TraderBot.src.data_manager import main as dm_main

logger = logging.getLogger(__name__)


def main() -> None:
    dm_main()


if __name__ == "__main__":
    main()
