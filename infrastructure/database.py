import sqlite3
from sqlite3 import Connection


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


# Example usage:
# if __name__ == '__main__':
#     conn = init_db(in_memory=True)
#     conn.close()
