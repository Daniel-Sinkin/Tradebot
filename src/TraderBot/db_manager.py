import logging
import sqlite3
from contextlib import contextmanager
from threading import Lock

from .util import format_connection

logger = logging.getLogger(__name__)


class DatabaseManager:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, db_path=":memory:"):
        if not hasattr(self, "initialized"):
            self.db_path = db_path
            self.connection = None
            self.initialized = True
        logger.info("Created %s", self)

    def __str__(self):
        return f"DatabaseManager connected to {self.db_path}"

    def __repr__(self):
        return (
            f"DatabaseManager(db_path={self.db_path!r}, "
            f"connection_active={self.connection is not None})"
        )

    def connect(self) -> sqlite3.Connection:
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path)
            logger.info(
                "Successfully created connection to %s.",
                format_connection(self.connection),
            )
        return self.connection

    def close(self):
        if self.connection:
            logger.info(
                "Trying to close connection to %s.",
                format_connection(self.connection),
            )
            self.connection.close()
            logger.info("Connection successfully closed.")
            self.connection = None

    @contextmanager
    def manage_connection(self):
        try:
            conn = self.connect()
            yield conn
        finally:
            self.close()


# Usage
if __name__ == "__main__":
    db_manager = DatabaseManager()

    with db_manager.manage_connection() as conn:
        # Perform database operations here
        pass
