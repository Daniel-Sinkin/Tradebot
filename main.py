import datetime as dt
import os
import time

import numpy as np
import pandas as pd
import vectorbt as vbt


def get_symbol_specs(symbol: str) -> float:
    if symbol[-3:] == "JPY":
        return 0.1
    else:
        return 1.0


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


def evaluate_portfolio(
    candles_dict: dict[str, pd.DataFrame],
    weights: np.ndarray,
    lookback_window: int = 6,
    base_volume: float = 1.0,
) -> float:
    SYMBOLS = list(candles_dict.keys())

    weights = np.array(weights)
    weights = weights / weights.sum()

    portfolio_entries = {}
    portfolio_exits = {}

    for symbol, weight in zip(SYMBOLS, weights):
        candles = candles_dict[symbol]
        deltas = weight * candles["Deltas"]
        deltas_ema = deltas.ewm(span=1000, adjust=False).mean()[1000:]
        deltas_specific = candles["Deltas"][1000:]

        points_sell = deltas_ema[
            np.all(
                [
                    deltas_specific.shift(i) > deltas_ema.shift(i)
                    for i in range(lookback_window)
                ],
                axis=0,
            )
        ]

        points_buy = deltas_ema[
            np.all(
                [
                    deltas_specific.shift(i) < deltas_ema.shift(i)
                    for i in range(lookback_window)
                ],
                axis=0,
            )
        ]

        portfolio_entries[symbol] = pd.Series(False, index=deltas_ema.index)
        portfolio_exits[symbol] = pd.Series(False, index=deltas_ema.index)

        portfolio_entries[symbol].loc[points_buy.index] = True
        portfolio_exits[symbol].loc[points_sell.index] = True

    # Align all signals and close prices by reindexing to a common index
    common_index = portfolio_entries[SYMBOLS[0]].index

    final_entries = pd.concat(
        [portfolio_entries[symbol].reindex(common_index) for symbol in SYMBOLS], axis=1
    ).any(axis=1)
    final_exits = pd.concat(
        [portfolio_exits[symbol].reindex(common_index) for symbol in SYMBOLS], axis=1
    ).any(axis=1)
    close_prices = pd.concat(
        [candles_dict[symbol]["Close"].reindex(common_index) for symbol in SYMBOLS],
        axis=1,
    )
    close_prices.columns = SYMBOLS

    # Backtesting with vectorbt
    portfolio = vbt.Portfolio.from_signals(
        close=close_prices,
        entries=final_entries,
        exits=final_exits,
        direction="both",
        freq="5min",
    )

    return portfolio

