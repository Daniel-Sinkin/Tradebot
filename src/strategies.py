import numpy as np
import pandas as pd
import vectorbt as vbt


def run_differential_momentum_strategy(
    candles_dict: dict[str, pd.DataFrame],
    weights: np.ndarray,
    lookback_window: int = 6,
    ema_span: int = 1000,
    timeframe: str = "5min",
    base_volume: float = 1.0,
) -> vbt.Portfolio:
    """
    Evaluate a portfolio using a Differential Momentum Strategy.

    The Differential Momentum Strategy generates buy and sell signals based on a comparison
    of recent price differences (deltas) with their exponentially weighted moving average (EMA).
    The strategy signals a buy when recent deltas are consistently below the EMA and a sell
    when they are consistently above the EMA over a specified lookback window.

    ### Parameters:
    * candles_dict : dict[str, pd.DataFrame]
        * A dictionary where each key is a symbol and the corresponding value is a DataFrame
          containing OHLC data. Each DataFrame must include a "Deltas" column representing the
          price differences.
    * weights : np.ndarray
        * An array of weights to apply to each symbol's deltas. The weights should sum to 1.
    * lookback_window : int, optional
        * The number of periods to consider when comparing recent deltas to their EMA. Default is 6.
    * ema_span : int, optional
        * The span for calculating the Exponential Moving Average (EMA) used to smooth the deltas.
          Default is 1000.
    * timeframe : str, optional
        * The frequency of the data used for backtesting, specified in a pandas-compatible string
          format (e.g., "5min", "1H"). Default is "5min".
    * base_volume : float, optional
        * The base volume used for scaling the strategy (this parameter is included for potential
          future use but is not currently utilized). Default is 1.0.

    ### Returns:
    * vbt.Portfolio
        * A vectorbt Portfolio object containing the results of the backtested strategy.

    ### Example:
    ```
    portfolio = evaluate_portfolio(
        candles_dict=my_candles,
        weights=np.array([0.5, 0.3, 0.2]),
        lookback_window=10,
        ema_span=800,
        timeframe="1H"
    )
    ```

    ### Notes:
    * This strategy is momentum-based, focusing on the directional movement of price changes (deltas).
    * The generated buy/sell signals are determined by whether the recent deltas are consistently above
      or below their EMA across the specified lookback window.
    """
    symbols = list(candles_dict.keys())

    weights = np.array(weights)
    weights = weights / weights.sum()

    portfolio_entries = {}
    portfolio_exits = {}

    for symbol, weight in zip(symbols, weights):
        candles = candles_dict[symbol]
        deltas = weight * candles["Deltas"]
        deltas_ema = deltas.ewm(span=ema_span, adjust=False).mean()[ema_span:]
        deltas_specific = candles["Deltas"][ema_span:]

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
    common_index = portfolio_entries[symbols[0]].index

    final_entries = pd.concat(
        [portfolio_entries[symbol].reindex(common_index) for symbol in symbols], axis=1
    ).any(axis=1)
    final_exits = pd.concat(
        [portfolio_exits[symbol].reindex(common_index) for symbol in symbols], axis=1
    ).any(axis=1)
    close_prices = pd.concat(
        [candles_dict[symbol]["Close"].reindex(common_index) for symbol in symbols],
        axis=1,
    )
    close_prices.columns = symbols

    # Backtesting with vectorbt
    portfolio = vbt.Portfolio.from_signals(
        close=close_prices,
        entries=final_entries,
        exits=final_exits,
        direction="both",
        freq=timeframe,
    )

    return portfolio
