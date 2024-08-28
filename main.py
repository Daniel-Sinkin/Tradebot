import datetime as dt

import matplotlib.pyplot as plt
import numpy as np

from traderbot.constants import _CandleTimeframe, _Symbol
from traderbot.data_manager import DatabaseDataManager
from traderbot.data_manager import main as dm_main
from traderbot.db_manager import DatabaseManager
from traderbot.ema_module import computeEMA
from traderbot.logger import setup_logger


def fill_db():
    setup_logger()
    dm_main()


def main():
    with DatabaseManager().manage_connection() as conn:
        ddm = DatabaseDataManager(conn)
        candles = ddm.get_candles(
            _Symbol.EURUSD,
            _CandleTimeframe.H1,
            dt.datetime(2023, 10, 1, tzinfo=dt.timezone.utc),
        )
    print(candles)


if __name__ == "__main__":
    main()
