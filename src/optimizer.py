import concurrent.futures
import time
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.optimize import dual_annealing

from .util import build_candle


class Optimizer(ABC):
    def __init__(self, candles_dict: dict[str, pd.DataFrame], strategy, maxiter=25):
        """
        Initialize the Optimizer.

        ### Parameters:
        * candles_dict
            * Dictionary containing the candlestick data for each symbol.
        * strategy
            * The trading strategy function to optimize. This function should accept
              weights and a lookback window as parameters and return a performance metric.
        * num_symbols
            * Number of symbols in the portfolio.
        * maxiter
            * Maximum number of iterations for the optimization process.
        """
        self.candles_dict = candles_dict
        self.strategy = strategy
        self.maxiter = maxiter

    @abstractmethod
    def optimize(self, lookback_window) -> None:
        """
        Optimize the trading strategy.

        ### Parameters:
        * lookback_window: int
            * Lookback window to use in the trading strategy.

        ### Returns:
        * tuple
            * Lookback window, best Sharpe ratio, and the best weights.
        """
        pass

    def run_optimizations_in_parallel(
        self, lookback_windows
    ) -> list[tuple[int, float, np.ndarray]]:
        """
        Run multiple optimizations in parallel over different lookback windows.

        ### Parameters:
        * lookback_windows: list[int]
            * List of lookback windows to optimize over.

        ### Returns:
        * list[tuple]
            * List of tuples containing lookback window, best Sharpe ratio, and best weights.
        """
        results = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.optimize, lw) for lw in lookback_windows]
            for future in concurrent.futures.as_completed(futures):
                lookback_window, best_sharpe, best_weights = future.result()
                print(
                    f"Lookback Window {lookback_window}: Best Total Returns: {best_sharpe:.4f} with weights: {best_weights}"
                )
                results.append((lookback_window, best_sharpe, best_weights))
        return results


class SimulatedAnnealingOptimizer(Optimizer):
    def optimize(self, lookback_window) -> tuple[int, float, np.ndarray]:
        """
        Optimize the trading strategy using simulated annealing.

        ### Parameters:
        * lookback_window: int
            * Lookback window to use in the trading strategy.

        ### Returns:
        * tuple
            * Lookback window, best Sharpe ratio, and the best weights.
        """
        bounds = [(-1, 1) for _ in range(self.num_symbols)]
        iteration = 0
        t0 = time.perf_counter()

        def objective(weights):
            # Negative of the strategy's return because we're minimizing
            return -(
                self.strategy(
                    self.candles_dict, np.array(weights), lookback_window
                ).stats()["End Value"]
                - 100
            )

        def callback(x, f, context):
            nonlocal iteration
            iteration += 1
            print(
                f"{lookback_window:2} - Iteration {iteration}/{self.maxiter}, Current Final Profit: {-f:.2f}%. Total runtime {time.perf_counter() - t0:.2f} seconds."
            )

        result = dual_annealing(
            objective, bounds=bounds, maxiter=self.maxiter, callback=callback
        )
        best_weights = result.x
        best_sharpe = -result.fun

        return lookback_window, best_sharpe, best_weights


def run_optimization(symbols: list[str]):
    print("Getting ticks")
    ticks = {symbol: get_ticks(symbol) for symbol in symbols}

    print("Building Candles")
    candles_dict = {
        symbol: build_candle((ticks_["ask"] + ticks_["bid"]) / 2.0, "5min")
        for symbol, ticks_ in ticks.items()
    }

    ticks.clear()
    del ticks

    for symbol in candles_dict:
        candles_dict[symbol]["Deltas"] = (
            0.2
            * candles_dict[symbol]["Close"].diff()
            * (0.1 if "JPK" == symbol[-3:] else 1.0)
        )
        candles_dict[symbol] = candles_dict[symbol].dropna()

    num_symbols = len(symbols)

    lookback_windows = [6, 7, 8, 9, 10, 11, 12]
    maxiter = 25
    print("Starting optimization runs.")

    # Initialize SimulatedAnnealingOptimizer with the necessary components
    optimizer = SimulatedAnnealingOptimizer(
        candles_dict, run_differential_momentum_strategy, num_symbols, maxiter
    )

    # Run optimizations in parallel
    results = optimizer.run_optimizations_in_parallel(lookback_windows)

    # Results can be further processed or analyzed
    return results
