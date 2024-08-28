import logging
import sqlite3
from pathlib import Path
from typing import Optional, cast

import pandas as pd

from .constants import _CandleTimeframe, _Paths, _Symbol
from .logger import cursor_execute

logger = logging.getLogger(__name__)


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
    filepath_db = _Paths.DATA.joinpath("database_backup.db")
    conn = create_connection_and_fill_with_tick_pkl(filepath_db, load_backup=False)

    cursor = conn.cursor()
    cursor_execute(cursor, "SELECT * FROM ticks LIMIT 5")
    ticks_sample = cursor.fetchall()
    print("Sample data from ticks table:")
    for row in ticks_sample:
        print(row)

    cursor_execute(cursor, "SELECT * FROM candles LIMIT 5")
    candles_sample = cursor.fetchall()
    print("Sample data from candles table:")
    for row in candles_sample:
        print(row)
