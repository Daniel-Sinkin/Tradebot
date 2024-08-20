import concurrent.futures
import datetime as dt
import os
import time
from typing import Optional, cast

import matplotlib.pyplot as plt
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
    candles["volume"] = 1.0
    candles.columns = ["Open", "High", "Low", "Close", "Volume"]
    return candles


def get_symbols() -> list[str]:
    return [
        f.removesuffix(".pkl").removeprefix("ticks_")
        for f in os.listdir("data")
        if os.path.isfile(os.path.join("data", f))
    ]


# Function to get tick data
def get_ticks(symbol: str) -> pd.DataFrame:
    data = cast(pd.DataFrame, pd.read_pickle(f"data/ticks_{symbol}.pkl"))
    idx0 = np.searchsorted(
        data.index, dt.datetime(2023, 10, 29, 21, 10, tzinfo=dt.timezone.utc)
    )
    data = data.iloc[idx0:]

    return data


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
        return -(
            evaluate_portfolio(
                candles_dict, np.array(weights), lookback_window
            ).stats()["End Value"]
            - 100
        )

    def callback(x, f, context):
        nonlocal iteration
        iteration += 1
        print(
            f"{lookback_window:2} - Iteration {iteration}/{maxiter}, Current Final Profit: {-f:.2f}%. Total runtime {time.perf_counter() - t0:.2f} seconds."
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
                f"Lookback Window {lookback_window}: Best Total Returns: {best_sharpe:.4f} with weights: {best_weights}"
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


def sample_weight_neighborhood(
    weight: np.ndarray, eps: float, seed: Optional[int] = None
) -> np.ndarray:
    """
    Samples weights around an epsilon ball to validate how stable the procedure is.

    This is not a uniform distribution in the ball, prefering points around the center, but that is good enough for us.
    """
    assert eps > 0, "radius has to be postive"
    _rng = np.random.default_rng(seed)
    radius = eps - _rng.uniform(0, eps)

    sample = _rng.normal(0, 1.0, len(weight))
    sample /= np.linalg.norm(sample, 2)
    sample *= radius

    return weight + sample


def main():
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
        candles_dict[symbol]["Deltas"] = candles_dict[symbol]["Close"].diff() / (
            get_symbol_specs(symbol)
        )
        candles_dict[symbol] = candles_dict[symbol].dropna()

    best_results = {
        6: [0.84833103, -0.55018444, -0.66391149, 0.2049016, 0.26380426],
        7: [0.39744567, 0.06226624, 0.01093088, -0.36537635, -0.03622586],
        8: [0.95629888, 0.3915262, -0.923586, -0.39015492, 0.47217318],
        9: [-0.99443441, -0.4044945, 0.96641785, -0.58417758, 0.70029267],
        10: [-0.08786026, -0.73912231, 0.11903087, -0.16121764, 0.76409417],
        11: [-0.42842743, -0.84678352, 0.20057753, -0.02365986, 0.94337179],
        12: [-0.2708226, 0.7040356, 0.34625536, -0.37657696, -0.25549361],
    }
    lbs = list(best_results.keys())

    portfolio = {}
    for i, (lb, w) in enumerate(best_results.items()):
        print(i)
        portfolio[lb] = evaluate_portfolio(candles_dict, weights=w, lookback_window=lb)

    returns = []
    for lb, portfolio_ in portfolio.items():
        returns.append(portfolio_.stats()["End Value"])

    ret = max(returns)
    for i in range(len(returns)):
        argmax = i
        if np.isclose(ret, returns[i]):
            break

    argmax_lb = lbs[argmax]
    argmax_weight = best_results[argmax_lb]

    print(f"Best lb = {argmax_lb}, best_weights = {argmax_weight}")
    annualization_factor = dt.timedelta(days=365).total_seconds() / (
        (portfolio_.stats()["End"] - portfolio_.stats()["Start"]).total_seconds()
    )
    print(f"Maximal Achieved Results {ret - 100:.2f}%.")
    print(
        f"Maximal Achieved Results (annual) {(ret - 100) * annualization_factor:.2f}%."
    )

    eps = 0.01
    print(f"Sampling neighborhood with {eps=}.")
    end_vals = []
    dists = []
    for i in range(250):
        sample = sample_weight_neighborhood(argmax_weight, eps)
        portfolio_ = evaluate_portfolio(
            candles_dict, weights=sample, lookback_window=argmax_lb
        )
        dist = np.linalg.norm(sample - argmax_weight, 2)
        dists.append(dist)
        print(f"{i + 1}. {sample=} ({dist=:.4f})")
        end_val = portfolio_.stats()["End Value"] - 100
        end_vals.append(end_val)
        end_val_annual = end_val * annualization_factor
        print(f"{end_val=:.2f}%, {end_val_annual=:.2f}%")

    print("\nQuantiles:")
    print(np.quantile(np.array(end_vals), [0, 0.25, 0.5, 0.75, 1.0]))

    plt.scatter(dists, end_vals)
    plt.scatter(0, ret - 100)

    plt.title("Performance of params around optimized weights.")

    plt.xlabel("Distance to determined point")
    plt.ylabel("Relative returns")

    plt.show()


if __name__ == "__main__":
    main()
