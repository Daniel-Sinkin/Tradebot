import datetime as dt
import logging
import sqlite3
from abc import ABC, abstractmethod
from typing import Optional, cast

import numpy as np
import pandas as pd

from .constants import _CandleTimeframe, _Paths, _Symbol
from .db_manager import DatabaseManager
from .util import slice_sorted

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
    ) -> Optional[pd.DataFrame]:
        if not isinstance(symbols, list):
            symbols = [symbols]
        if not isinstance(timeframes, list):
            timeframes = [timeframes]

        if ts_until is None:
            ts_until = dt.datetime.now(tz=dt.timezone.utc)

        df_list = [
            self._get_candles(symbol, ctf, ts_from, ts_until)
            for symbol in symbols
            for ctf in timeframes
        ]
        if any([df is None for df in df_list]):
            logger.warning("At least one dataframe is None, returning None.")
            return None
        return pd.concat([df for df in df_list if not df.empty])

    def push_candles_to_sql(
        self,
        conn: sqlite3.Connection,
        symbols: _Symbol | list[_Symbol],
        timeframes: _CandleTimeframe | list[_CandleTimeframe],
        ts_from: dt.datetime,
        ts_until: Optional[dt.datetime] = None,
    ) -> None:
        candles = self.get_candles(symbols, timeframes, ts_from, ts_until)
        if candles is None:
            logger.warning("No candles found, not pushing anything to sql.")
        candles.to_sql("candles", conn, index=False, if_exists="append")

    @staticmethod
    def validate_candles(candles: pd.DataFrame) -> None:
        assert isinstance(candles, pd.DataFrame)
        ohlc = ["open", "high", "low", "close"]
        assert all(c in candles.columns for c in ohlc)

        assert (candles[ohlc] > 0).all().all()
        assert (candles.high >= candles[["open", "low", "close"]].max(axis=1)).all()
        assert (candles.low <= candles[["open", "high", "close"]].min(axis=1)).all()

        # TODO: Add validation that 'ts' are datetimes/timestamps not str/objects

        assert candles.columns.str.islower().all()
        for c in ohlc:
            # TODO: Change the internal handling of ticks and candles to either use f32
            #       or 8/16 bit deltas for better compression
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
            raise ValueError("No ticks!")

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
        if ts_until is None:
            ts_until = dt.datetime.now(tz=dt.timezone.utc)

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
        candles = slice_sorted(candles, "ts", ts_from, ts_until)

        if validate_candles:
            DataManager.validate_candles(candles)
        return candles


class DatabaseDataManager(DataManager):
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def _get_candles(
        self,
        symbol: _Symbol,
        timeframe: _CandleTimeframe,
        ts_from: dt.datetime,
        ts_until: Optional[dt.datetime] = None,
    ) -> Optional[pd.DataFrame]:
        if ts_until is None:
            ts_until = dt.datetime.now(tz=dt.timezone.utc)
        query = f"""
SELECT * FROM candles 
WHERE symbol = '{symbol}' AND timeframe = '{timeframe}' AND ts BETWEEN '{ts_from.isoformat()}' AND '{ts_until.isoformat()}'
            """
        candles = pd.read_sql(query, self.conn)
        if candles.empty:
            logger.info("No candles found in the database for the given parameters.")
            return None
        candles["ts"] = pd.to_datetime(candles["ts"])

        DataManager.validate_candles(candles)
        return candles


def fill_db_with_pkl() -> None:
    with DatabaseManager().manage_connection() as conn:
        PickleDataManager().push_candles_to_sql(
            conn=conn,
            symbols=list(_Symbol),
            timeframes=list(_CandleTimeframe),
            ts_from=dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc),
        )
