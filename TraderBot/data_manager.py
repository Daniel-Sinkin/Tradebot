import datetime as dt
import logging
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, cast

import numpy as np
import pandas as pd

from .constants import _CandleTimeframe, _Paths, _Symbol
from .db_manager import DatabaseManager
from .logger import cursor_execute

logger = logging.getLogger(__name__)


class DataManager(ABC):
    @abstractmethod
    def _get_candles(
        self,
        symbol: _Symbol,
        timeframe: _CandleTimeframe,
        ts_from: dt.datetime,
        ts_until: Optional[dt.datetime] = None,
    ) -> Optional[pd.DataFrame]: ...

    def get_candles(
        self,
        symbols: _Symbol | list[_Symbol],
        timeframes: _CandleTimeframe | list[_CandleTimeframe],
        ts_from: dt.datetime,
        ts_until: Optional[dt.datetime] = None,
    ) -> Optional[dict[_CandleTimeframe, pd.DataFrame]]:
        if not isinstance(symbols, list):
            symbols = [symbols]
        if not isinstance(timeframes, list):
            timeframes = [timeframes]

        df_list = [
            self._get_candles(symbol, ctf, ts_from, ts_until)
            for symbol in symbols
            for ctf in timeframes
        ]
        if any([df is None for df in df_list]):
            logger.warning("At least one dataframe is None, returning None.")
            return None
        return pd.concat([df for df in df_list if not df.empty])

    @staticmethod
    def validate_candles(candles: pd.DataFrame) -> None:
        assert isinstance(candles, pd.DataFrame)
        ohlc = ["open", "high", "low", "close"]
        assert all(c in candles.columns for c in ohlc)

        assert (candles[ohlc] > 0).all().all()
        assert (candles.high >= candles[["open", "low", "close"]].max(axis=1)).all()
        assert (candles.low <= candles[["open", "high", "close"]].min(axis=1)).all()

        assert candles.columns.str.islower().all()
        for c in ohlc:
            assert candles[c].dtype == np.float64


class PickleDataManager(DataManager):
    def get_ticks(
        self,
        symbol: _Symbol,
        ts_from: dt.datetime,
        ts_until: Optional[dt.datetime] = None,
    ) -> Optional[pd.DataFrame]:
        if not isinstance(symbol, _Symbol):
            raise TypeError(f"{symbol=} is of {type=} but has to be of type 'Symbol'!")

        filepath = _Paths.DATA.joinpath(f"ticks_{symbol}.pkl")
        if not filepath.exists():
            logger.info(f"{filepath=} does not exist, returning None.")
            return None

        ticks = cast(pd.DataFrame, pd.read_pickle(filepath))
        assert isinstance(ticks, pd.DataFrame)

        ticks.reset_index(inplace=True)

        ticks["symbol"] = symbol.value

        ticks = ticks[["symbol", "ts", "bid", "ask"]]
        return ticks

    def _generate_candles_from_ticks(
        self,
        symbol: _Symbol,
        timeframe: _CandleTimeframe,
        ts_from: dt.datetime,
        ts_until: Optional[dt.datetime] = None,
    ) -> Optional[pd.DataFrame]:
        pd_tf = timeframe.to_pandas_timeframe()

        if ts_until is None:
            ts_until = dt.datetime.now(tz=dt.timezone.utc)

        ts_from: dt.datetime = pd.Timestamp(ts_from).ceil(pd_tf).to_pydatetime()
        ts_until: dt.datetime = pd.Timestamp(ts_until).floor(pd_tf).to_pydatetime()

        ticks = self.get_ticks(symbol, ts_from, ts_until)
        if ticks is None:
            logger.info("Could not find the ticks, returning None.")
            return None

        if not isinstance(ticks, pd.DataFrame):
            raise TypeError("ticks has to be a DataFrame.")
        if ticks.empty:
            raise ValueError("No ticks passed!")

        candles = (
            ticks.set_index("ts")
            .resample(timeframe.to_pandas_timeframe())
            .agg(
                open=("bid", "first"),
                high=("bid", "max"),
                low=("bid", "min"),
                close=("bid", "last"),
                volume=("bid", "count"),
            )
            .reset_index()
            .dropna()
        )

        candles["symbol"] = ticks.iloc[0]["symbol"]
        candles["timeframe"] = timeframe.value

        candles = candles[
            ["symbol", "timeframe", "ts", "open", "high", "low", "close", "volume"]
        ]

        return candles

    def _get_candles(
        self,
        symbol: _Symbol,
        timeframe: _CandleTimeframe,
        ts_from: dt.datetime,
        ts_until: Optional[dt.datetime] = None,
        validate_candles: bool = True,
    ) -> Optional[pd.DataFrame]:
        candle_fp = _Paths.DATA.joinpath(f"candles_{symbol}_{timeframe}.pkl")
        if not candle_fp.exists():
            logger.info(
                f"{candle_fp=} does not exist, trying to build the candles from ticks."
            )
            candles = self._generate_candles_from_ticks(
                symbol=symbol, timeframe=timeframe, ts_from=ts_from, ts_until=ts_until
            )
            if candles is None:
                logger.info("Could not build candles.")
                return None
            candles.to_pickle(candle_fp)
            if not candle_fp.exists():
                raise RuntimeError("Could not save candles after building.")

        candles = cast(pd.DataFrame, pd.read_pickle(candle_fp))
        if validate_candles:
            DataManager.validate_candles(candles)
        return candles


def load_tick_pkl(symbol: _Symbol) -> Optional[pd.DataFrame]:
    if not isinstance(symbol, _Symbol):
        raise TypeError(f"{symbol=} is of {type=} but has to be of type 'Symbol'!")

    filepath = _Paths.DATA.joinpath(f"ticks_{symbol}.pkl")
    if not filepath.exists():
        logger.info(f"{filepath=} does not exist, returning None.")
        return None

    ticks = cast(pd.DataFrame, pd.read_pickle(filepath))
    assert isinstance(ticks, pd.DataFrame)

    ticks.reset_index(inplace=True)

    ticks["symbol"] = symbol.value

    ticks = ticks[["symbol", "ts", "bid", "ask"]]
    return ticks


def create_database_connection() -> sqlite3.Connection:
    """Create an in-memory SQLite database and return the connection."""
    conn = sqlite3.connect(":memory:")
    logger.info("Initialized in-memory SQLite database.")
    return conn


def create_ticks_table(conn: sqlite3.Connection) -> None:
    """Create the ticks table in the SQLite database."""
    cursor = conn.cursor()
    cursor_execute(cursor, "DROP TABLE IF EXISTS ticks")
    cursor_execute(
        cursor,
        """
CREATE TABLE IF NOT EXISTS ticks (
    symbol TEXT NOT NULL,
    ts TEXT NOT NULL,
    bid REAL NOT NULL,
    ask REAL NOT NULL,
    PRIMARY KEY (symbol, ts)
)
        """,
    )
    logger.info("Created ticks table.")


def create_candles_table(conn: sqlite3.Connection) -> None:
    """Create the ticks table in the SQLite database."""
    cursor = conn.cursor()
    cursor_execute(cursor, "DROP TABLE IF EXISTS candles")
    cursor_execute(
        cursor,
        """
CREATE TABLE candles (
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    ts TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL,
    PRIMARY KEY (symbol, timeframe, ts)
)
        """,
    )
    logger.info("Created candles table.")


def insert_ticks_into_db(ticks: pd.DataFrame, conn: sqlite3.Connection) -> None:
    """Insert tick data into the ticks table in the SQLite database."""
    ticks.to_sql("ticks", conn, if_exists="append", index=False)
    logger.info("Inserted ticks data into the database.")


def create_candles_from_ticks(
    ticks: pd.DataFrame, timeframe: _CandleTimeframe
) -> pd.DataFrame:
    """
    Create OHLC candles for a specific timeframe and insert them into the database.

    For timeframe either enter a pandas compatible Timeframe grouper string or an internal
    CandleTimeframe Enum.
    """
    if not isinstance(ticks, pd.DataFrame):
        raise TypeError("ticks has to be a DataFrame.")
    if ticks.empty:
        raise ValueError("No ticks passed!")

    candles = (
        ticks.set_index("ts")
        .resample(timeframe.to_pandas_timeframe())
        .agg(
            open=("bid", "first"),
            high=("bid", "max"),
            low=("bid", "min"),
            close=("bid", "last"),
            volume=("bid", "count"),
        )
        .reset_index()
        .dropna()
    )

    candles["symbol"] = ticks.iloc[0]["symbol"]
    candles["timeframe"] = timeframe.value

    candles = candles[
        ["symbol", "timeframe", "ts", "open", "high", "low", "close", "volume"]
    ]

    return candles


def create_connection_and_fill_with_tick_pkl(
    backup_location: Optional[Path] = None, load_backup: bool = True
) -> sqlite3.Connection:
    """
    Creates in-memory database connection, fills it with the tick and (generated) candle data from
    the data folder and returns the connection.

    """
    conn = create_database_connection()

    if load_backup:
        if backup_location is None:
            print("WARNING: 'load_backup' is set but backup_location is None!")
        elif backup_location.exists():
            return sqlite3.connect(backup_location)

    create_ticks_table(conn)
    create_candles_table(conn)

    for i, symbol in enumerate(_Symbol):
        print(f"{i + 1}/{len(_Symbol)}.", symbol.pretty_format())
        print("Loading Ticks")
        ticks = load_tick_pkl(symbol=symbol)
        if ticks is None:
            continue

        print("Pushing Ticks")
        ticks.to_sql("ticks", conn, if_exists="append", index=False)

        print("Creating and Pushing Candles")
        for j, ctf in enumerate(_CandleTimeframe):
            print(f"\t{j + 1}/{len(_CandleTimeframe)}.", ctf)
            candles = create_candles_from_ticks(ticks, ctf)
            candles.to_sql("candles", conn, if_exists="append", index=False)
        break

    conn.commit()

    db_path = _Paths.DATA.joinpath("database_backup.db")

    logger.debug("Creating local backup of the database at '%s'", db_path)
    disc_conn = sqlite3.connect(db_path)
    try:
        conn.backup(disc_conn)
    finally:
        disc_conn.close()

    return conn


def main() -> None:
    candles_nested_dict = PickleDataManager().get_candles(
        symbols=list(_Symbol),
        timeframes=list(_CandleTimeframe),
        ts_from=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
    )
    with DatabaseManager().manage_connection() as conn:
        candles_nested_dict.to_sql("candles", conn, if_exists="append", index=False)

        candle_df_from_sql = pd.read_sql("SELECT * FROM candles", conn)
        logger.info("\n%s", candle_df_from_sql.sample(n=10))
