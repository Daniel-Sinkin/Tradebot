import datetime as dt
from abc import ABC, abstractmethod
from typing import cast

import numpy as np
import pandas as pd


class DataManager(ABC):
    @abstractmethod
    def load_data(self, symbol: str) -> pd.DataFrame:
        """
        Load the data for a given symbol.

        ### Parameters:
        * symbol
            * The symbol for which to load data.

        ### Returns:
        * A pandas DataFrame containing the loaded data.
        """
        pass

    @abstractmethod
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the loaded data. This could involve tasks such as filtering,
        resampling, or computing additional columns.

        ### Parameters:
        * df
            * The DataFrame to preprocess.

        ### Returns:
        * The preprocessed DataFrame.
        """
        pass


class PickleDataManager(DataManager):
    def __init__(self, base_path: str):
        """
        Initialize the PickleDataManager with a base path for data files.

        ### Parameters:
        * base_path
            * The base path where pickle files are stored.
        """
        self.base_path = base_path

    def load_data(self, symbol: str) -> pd.DataFrame:
        """
        Load data from a pickle file for a given symbol.

        ### Parameters:
        * symbol
            * The symbol for which to load data.

        ### Returns:
        * A pandas DataFrame containing the loaded data.
        """
        data = cast(
            pd.DataFrame, pd.read_pickle(f"{self.base_path}/ticks_{symbol}.pkl")
        )
        return data

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the loaded data, such as filtering by date and computing additional columns.

        ### Parameters:
        * df
            * The DataFrame to preprocess.

        ### Returns:
        * The preprocessed DataFrame.
        """
        idx0 = np.searchsorted(
            df.index, dt.datetime(2023, 10, 29, 21, 10, tzinfo=dt.timezone.utc)
        )
        df = df.iloc[idx0:]
        return df


class DatabaseDataManager(DataManager):
    def __init__(self, connection_string: str):
        """
        Initialize the DatabaseDataManager with a connection string.

        ### Parameters:
        * connection_string
            * The connection string used to connect to the database.
        """
        self.connection_string = connection_string

    def load_data(self, symbol: str) -> pd.DataFrame:
        """
        Load data from a database for a given symbol.

        ### Parameters:
        * symbol
            * The symbol for which to load data.

        ### Returns:
        * A pandas DataFrame containing the loaded data.
        """
        # Stub implementation: Replace with actual database query logic.
        raise NotImplementedError("Database loading is not implemented yet.")

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the loaded data.

        ### Parameters:
        * df
            * The DataFrame to preprocess.

        ### Returns:
        * The preprocessed DataFrame.
        """
        # Stub implementation: Replace with actual preprocessing logic.
        raise NotImplementedError("Preprocessing is not implemented yet.")
