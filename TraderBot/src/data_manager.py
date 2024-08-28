import sqlite3
from pathlib import Path
from typing import Optional, cast

import pandas as pd

from .constants import CandleTimeframe, DBTables, Paths_, Symbol


def load_tick_pkl(symbol: Symbol | str, append_symbol: bool = False) -> pd.DataFrame:
    if isinstance(symbol, Symbol):
        symbol = symbol.value

    assert isinstance(symbol, str)

    filename = f"ticks_{symbol}.pkl"
    ticks = cast(pd.DataFrame, pd.read_pickle(Paths_.data.joinpath(filename)))

    if append_symbol:
        ticks["symbol"] = symbol

    return ticks


def create_database_connection() -> sqlite3.Connection:
    """Create an in-memory SQLite database and return the connection."""
    conn = sqlite3.connect(":memory:")
    print("Initialized in-memory SQLite database.")
    return conn


def create_ticks_table(conn: sqlite3.Connection) -> None:
    """Create the ticks table in the SQLite database."""
    cursor = conn.cursor()
    cursor.execute(f"""
CREATE TABLE IF NOT EXISTS {DBTables.candles} (
    symbol TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    bid REAL NOT NULL,
    ask REAL NOT NULL,
    PRIMARY KEY (symbol, timestamp)
)
    """)
    print("Created ticks table.")


def create_candles_table(conn: sqlite3.Connection) -> None:
    """Create the ticks table in the SQLite database."""
    cursor = conn.cursor()
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {DBTables.candles} (
        id INTEGER PRIMARY KEY,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume REAL,
        timestamp TEXT NOT NULL
    )
    """)
    print("Created candles table.")


def insert_ticks_into_db(ticks: pd.DataFrame, conn: sqlite3.Connection) -> None:
    """Insert tick data into the ticks table in the SQLite database."""
    ticks.to_sql("ticks", conn, if_exists="append", index=True, index_label="timestamp")
    print("Inserted ticks data into the database.")


def create_candles_from_ticks(
    ticks: pd.DataFrame,
    timeframe: CandleTimeframe | str,
    append_symbol: bool = False,
    append_timeframe: bool = False,
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

    # If necessary converts timeframe to our internal enum.
    if timeframe not in CandleTimeframe:
        timeframe = CandleTimeframe.from_pandas_timeframe(timeframe)
    if not isinstance(timeframe, str):
        raise TypeError(f"{type(timeframe)=} is not supported.")

    candles = ticks.resample(timeframe.to_pandas_timeframe()).agg(
        open=("bid", "first"),
        high=("bid", "max"),
        low=("bid", "min"),
        close=("bid", "last"),
        volume=("bid", "count"),
    )

    if append_symbol:
        try:
            candles["symbol"] = ticks.iloc[0]["symbol"]
        except KeyError:
            print(
                "WARNING: activated append_symbol while no symbol is availiable in the ticks, skipping the appending."
            )
    if append_timeframe:
        candles["timeframe"] = timeframe.value
    return candles


def create_db_and_fill_with_tick_pkl(
    backup_location: Optional[Path] = None,
) -> sqlite3.Connection:
    """
    Creates in-memory database connection, fills it with the tick and (generated) candle data from
    the data folder and returns the connection.
    """
    conn = create_database_connection()

    # If a backup exists then use that one, no validation is done it on it though.
    if backup_location is not None and backup_location.exists():
        return sqlite3.connect(backup_location)

    for i, symbol in enumerate(Symbol):
        print(f"{i + 1}/{len(Symbol)}.", symbol)
        print("Loading Ticks")
        ticks = load_tick_pkl(symbol=symbol, append_symbol=True)

        print("Pushing Ticks")
        ticks.to_sql(
            "ticks", conn, if_exists="replace", index=True, index_label="timestamp"
        )

        print("Creating and Pushing Candles")
        for j, tf in enumerate(CandleTimeframe):
            print(f"\t{j + 1}/{len(CandleTimeframe)}.", tf)
            candles = create_candles_from_ticks(
                ticks, tf, append_symbol=True, append_timeframe=True
            )
            candles.to_sql(
                DBTables.candles,
                conn,
                if_exists="append",
                index=True,
                index_label="timestamp",
            )
        break

    conn.commit()
    print("Transaction committed.")

    with sqlite3.connect(Paths_.data.joinpath("database_backup.db")) as disc_conn:
        conn.backup(disc_conn)

    return conn


def main() -> None:
    conn = create_db_and_fill_with_tick_pkl(Paths_.data.joinpath("database_backup.db"))

    cursor = conn.cursor()
    cursor.execute("SELECT * FROM ticks LIMIT 5")
    ticks_sample = cursor.fetchall()
    print("Sample data from ticks table:")
    for row in ticks_sample:
        print(row)

    cursor.execute(f"SELECT * FROM {DBTables.candles} LIMIT 5")
    candles_sample = cursor.fetchall()
    print("Sample data from candles table:")
    for row in candles_sample:
        print(row)
