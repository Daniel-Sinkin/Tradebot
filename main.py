import datetime as dt

import matplotlib.pyplot as plt
import numpy as np

from src_old.util import build_candle, sample_eps_ball


def main() -> None:
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
        portfolio[lb] = run_differential_momentum_strategy(
            candles_dict, weights=w, lookback_window=lb
        )

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
        sample = sample_eps_ball(argmax_weight, eps)
        portfolio_ = run_differential_momentum_strategy(
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
