import datetime as dt
import os
import time

import numpy as np
import pandas as pd


# Function to build candles
def build_candle(
    prices: pd.Series, timeframe: str, dropna: bool = True
) -> pd.DataFrame:
    resampled = prices.resample(timeframe)
    candles = resampled.ohlc()
    candles["volume"] = 20.0
    candles.columns = ["Open", "High", "Low", "Close", "Volume"]
    return candles


def get_symbols() -> list[str]:
    return [f for f in os.listdir("data") if os.path.isfile(os.path.join("data", f))]


# Function to get tick data
def get_ticks(symbol: str) -> pd.DataFrame:
    return pd.read_pickle(f"data/ticks_{symbol}.pkl")


SYMBOLS = get_symbols()


