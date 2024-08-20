import os
import sqlite3
from sqlite3 import Connection

import pandas as pd


def init_db(in_memory: bool = True) -> Connection:
    """
    Initialize the SQLite database.

    ### Parameters:
    * in_memory: bool
        * If True, the database is created in memory. If False, it is created on disk.

    ### Returns:
    * Connection
        * SQLite3 database connection object.
    """
    if in_memory:
        conn = sqlite3.connect(":memory:")  # In-memory database
        print("Initialized in-memory SQLite database.")
    else:
        conn = sqlite3.connect("data/tradebot.db")  # Database file on disk
        print("Initialized SQLite database at 'data/tradebot.db'.")

    create_tables(conn)
    return conn


def create_tables(conn: Connection):
    """
    Create the necessary tables in the database.

    ### Parameters:
    * conn: Connection
        * SQLite3 database connection object.
    """
    cursor = conn.cursor()

    # Example table for ticks
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ticks (
        id INTEGER PRIMARY KEY,
        symbol TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        bid REAL NOT NULL,
        ask REAL NOT NULL
    )
    """)

    # Example table for candles
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS candles (
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

    conn.commit()
    print("Created tables in the SQLite database.")


def push_ticks_to_db(filepath: str, db_path: str = "data/tradebot.db"):
    """
    Push the contents of a tick data pickle file to the SQLite database.

    ### Parameters:
    * filepath: str
        * The path to the pickle file containing the tick data.
    * db_path: str, optional
        * The path to the SQLite database file. Default is 'data/tradebot.db'.
    """
    # Extract the symbol from the filename
    filename = os.path.basename(filepath)
    symbol = filename.replace("ticks_", "").replace(".pkl", "")

    # Load the tick data from the pickle file
    tick_data = pd.read_pickle(filepath)

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if the table for the symbol exists
    cursor.execute(
        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{symbol}';"
    )
    table_exists = cursor.fetchone()

    if not table_exists:
        # Create a new table for the symbol if it doesn't exist
        cursor.execute(f"""
        CREATE TABLE {symbol} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            bid REAL NOT NULL,
            ask REAL NOT NULL
        )
        """)
        print(f"Created table for {symbol}.")

    # Insert the tick data into the table
    tick_data.reset_index(inplace=True)  # Ensure that the index (timestamp) is a column
    tick_data.columns = [
        "timestamp",
        "bid",
        "ask",
    ]  # Rename columns to match the database schema

    tick_data.to_sql(symbol, conn, if_exists="append", index=False)
    print(f"Inserted {len(tick_data)} rows into {symbol} table.")

    # Commit the transaction and close the connection
    conn.commit()
    conn.close()

    print(f"Data from {filename} has been pushed to the database.")


def main() -> None:
    conn: Connection = init_db(in_memory=True)
    conn.close()


if __name__ == "__main__":
    main()
