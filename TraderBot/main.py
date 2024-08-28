import logging
import sys
from pathlib import Path

from src.data_manager import main as dm_main

sys.path.append(str(Path(__file__).resolve().parent.joinpath("src")))

logger = logging.getLogger(__name__)


def main() -> None:
    dm_main()


if __name__ == "__main__":
    main()
