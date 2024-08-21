from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import vectorbt as vbt


class Strategy(ABC):
    @abstractmethod
    def run_backtest(self) -> vbt.Portfolio:
        """
        Run the backtest for the strategy.

        ### Returns:
        * A vectorbt Portfolio object containing the results of the backtested strategy.
        """
        pass


class DifferentialMomentumStrategy(Strategy):
    def __init__(
        self,
        candles_dict: dict[str, pd.DataFrame],
        weights: np.ndarray,
        lookback_window: int = 6,
        ema_span: int = 1000,
        timeframe: str = "5min",
        base_volume: float = 1.0,
    ):
        """
        Initialize the Differential Momentum Strategy.

        ### Parameters:
        * candles_dict
            * A dictionary where each key is a symbol and the corresponding value is a DataFrame
              containing OHLC data. Each DataFrame must include a "Deltas" column representing the
              price differences.
        * weights
            * An array of weights to apply to each symbol's deltas. The weights can't all be zero but
              don't have to sum to 1.0 as they get normalized internally.

              Need one weight for search symbol (key in candles_dict).
        * lookback_window
            * The number of periods to consider when comparing recent deltas to their EMA.
        * ema_span
            * The span for calculating the Exponential Moving Average (EMA) used to smooth the deltas.
        * timeframe
            * The frequency of the data used for backtesting, specified in a pandas-compatible string
              format (e.g., "5min", "1H").
        * base_volume
            * The base volume used for scaling the strategy (this parameter is included for potential
              future use but is not currently utilized).
        """
        if len(candles_dict) != len(weights):
            raise ValueError(f"{len(candles_dict)=}!={len(weights)=}")
        if np.isclose(weights, np.zeros_like(weights)):
            raise ValueError("Require non-zero weight vector.")

        self.candles_dict = candles_dict
        self.weights = weights / np.linalg.norm(weights, 2)
        self.lookback_window = lookback_window
        self.ema_span = ema_span
        self.timeframe = timeframe
        self.base_volume = base_volume

    def run_backtest(self) -> vbt.Portfolio:
        """
        Run the backtest for the Differential Momentum Strategy.

        ### Returns:
        * A vectorbt Portfolio object containing the results of the backtested strategy.
        """
        symbols = list(self.candles_dict.keys())

        weights = np.array(self.weights)
        weights = weights / weights.sum()

        portfolio_entries = {}
        portfolio_exits = {}

        for symbol, weight in zip(symbols, weights):
            candles = self.candles_dict[symbol]
            deltas = weight * candles["Deltas"]
            deltas_ema = deltas.ewm(span=self.ema_span, adjust=False).mean()[
                self.ema_span :
            ]
            deltas_specific = candles["Deltas"][self.ema_span :]

            points_sell = deltas_ema[
                np.all(
                    [
                        deltas_specific.shift(i) > deltas_ema.shift(i)
                        for i in range(self.lookback_window)
                    ],
                    axis=0,
                )
            ]

            points_buy = deltas_ema[
                np.all(
                    [
                        deltas_specific.shift(i) < deltas_ema.shift(i)
                        for i in range(self.lookback_window)
                    ],
                    axis=0,
                )
            ]

            portfolio_entries[symbol] = pd.Series(False, index=deltas_ema.index)
            portfolio_exits[symbol] = pd.Series(False, index=deltas_ema.index)

            portfolio_entries[symbol].loc[points_buy.index] = True
            portfolio_exits[symbol].loc[points_sell.index] = True

        # Align all signals and close prices by reindexing to a common index
        common_index = portfolio_entries[symbols[0]].index

        final_entries = pd.concat(
            [portfolio_entries[symbol].reindex(common_index) for symbol in symbols],
            axis=1,
        ).any(axis=1)
        final_exits = pd.concat(
            [portfolio_exits[symbol].reindex(common_index) for symbol in symbols],
            axis=1,
        ).any(axis=1)
        close_prices = pd.concat(
            [
                self.candles_dict[symbol]["Close"].reindex(common_index)
                for symbol in symbols
            ],
            axis=1,
        )
        close_prices.columns = symbols

        portfolio = vbt.Portfolio.from_signals(
            close=close_prices,
            entries=final_entries,
            exits=final_exits,
            direction="both",
            freq=self.timeframe,
        )

        return portfolio
