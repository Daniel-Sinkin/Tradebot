from traderbot.data_manager import main as dm_main
from traderbot.logger import setup_logger


def main():
    setup_logger()
    dm_main()


if __name__ == "__main__":
    main()
