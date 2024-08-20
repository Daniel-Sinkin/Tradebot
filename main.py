import concurrent.futures
import datetime as dt
import os
import time

import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy.optimize import dual_annealing


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
) -> vbt.Portfolio:
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


def optimize_with_simulated_annealing(
    candles_dict, lookback_window, num_symbols, maxiter: int
):
    bounds = [(-1, 1) for _ in range(num_symbols)]
    iteration = 0
    t0 = time.perf_counter()

    def objective(weights):
        return -evaluate_portfolio(
            candles_dict, np.array(weights), lookback_window
        ).stats()["Sharpe Ratio"]

    def callback(x, f, context):
        nonlocal iteration
        iteration += 1
        print(
            f"{lookback_window:2} - Iteration {iteration}/{maxiter}, Current Sharpe Ratio: {-f:.4f}. Total runtime {time.perf_counter() - t0:.2f} seconds."
        )

    result = dual_annealing(
        objective, bounds=bounds, maxiter=maxiter, callback=callback
    )
    best_weights = result.x
    best_sharpe = -result.fun

    return lookback_window, best_sharpe, best_weights


def run_optimizations_in_parallel(
    candles_dict, lookback_windows, num_symbols, maxiter=25
):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                optimize_with_simulated_annealing,
                candles_dict,
                lw,
                num_symbols,
                maxiter,
            )
            for lw in lookback_windows
        ]
        for future in concurrent.futures.as_completed(futures):
            lookback_window, best_sharpe, best_weights = future.result()
            print(
                f"Lookback Window {lookback_window}: Best Sharpe Ratio: {best_sharpe:.4f} with weights: {best_weights}"
            )


def run_optimization():
    print("Getting ticks")
    ticks = {symbol: get_ticks(symbol) for symbol in SYMBOLS}

    print("Building Candles")
    candles_dict = {
        symbol: build_candle((ticks_["ask"] + ticks_["bid"]) / 2.0, "5min")
        for symbol, ticks_ in ticks.items()
    }

    ticks.clear()
    del ticks

    for symbol in candles_dict:
        candles_dict[symbol]["Deltas"] = (
            0.2 * candles_dict[symbol]["Close"].diff() * get_symbol_specs(symbol)
        )
        candles_dict[symbol] = candles_dict[symbol].dropna()

    num_symbols = len(SYMBOLS)

    lookback_windows = [6, 7, 8, 9, 10, 11, 12]
    maxiter = 25
    print("Starting optimization runs.")
    run_optimizations_in_parallel(candles_dict, lookback_windows, num_symbols, maxiter)

