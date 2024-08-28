import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from traderbot.constants import _CandleTimeframe, _Symbol
from traderbot.data_manager import DatabaseDataManager
from traderbot.db_manager import DatabaseManager
from traderbot.ta_module import (
    computeEMA,
    computeResistance,
    computeStochasticOscillator,
    computeSupport,
)


def get_symbol_ctf_to_candle_map():
    with DatabaseManager().manage_connection() as conn:
        candles = DatabaseDataManager(conn).get_candles(
            list(_Symbol),
            [_CandleTimeframe.M1, _CandleTimeframe.D1],
            ts_from=dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc),
        )

    symbol_ctf_to_candle_map: dict[str, pd.DataFrame] = {
        (symbol, ctf): candles[
            (candles.symbol == symbol) & (candles.timeframe == ctf)
        ].copy()
        for symbol in candles.symbol.unique()
        for ctf in [_CandleTimeframe.M1.value, _CandleTimeframe.D1.value]
    }
    return symbol_ctf_to_candle_map


def main():
    symbol_ctf_to_candle_map = get_symbol_ctf_to_candle_map()

    candles_daily = symbol_ctf_to_candle_map[("EURUSD", "1d")]
    candles_minute = symbol_ctf_to_candle_map[("EURUSD", "1m")]

    lookback_minutes = 15

    short_ema = computeEMA(candles_daily.close, 12)
    long_ema = computeEMA(candles_daily.close, 26)
    macd_line = long_ema - short_ema
    macd_signal = computeEMA(macd_line, 9)
    macd_delta = macd_line - macd_signal

    bullish_crossovers_idxs = np.where(macd_delta[1:] >= macd_delta[:-1])[0]
    bearish_crossovers_idxs = np.where(macd_delta[1:] <= macd_delta[:-1])[0]

    candles_daily["bullish_crossover"] = 0
    candles_daily["bearish_crossover"] = 0

    candles_daily.loc[bullish_crossovers_idxs[0], "bullish_crossover"] = 1
    candles_daily.loc[bearish_crossovers_idxs[0], "bearish_crossover"] = 1

    candles_daily["stoch_osc"] = computeStochasticOscillator(
        candles_daily.close, candles_daily.low, candles_daily.high, 14
    )
    candles_daily["support"] = computeSupport(candles_daily.close, 20)
    candles_daily["resistance"] = computeResistance(candles_daily.close, 20)

    candles_daily = candles_daily.iloc[19:].copy()

    candles_daily["date"] = pd.to_datetime(candles_daily["ts"]).dt.date
    candles_minute["date"] = pd.to_datetime(candles_minute["ts"]).dt.date

    log_returns = np.log(
        candles_minute["close"] / candles_minute["close"].shift(1)
    ).dropna()
    rolling_volatility = log_returns.rolling(window=60).std()
    annualized_rolling_volatility = rolling_volatility * np.sqrt(252 * 390)

    candles_minute["volatility"] = annualized_rolling_volatility

    candles_minute = candles_minute.merge(
        candles_daily[
            [
                "symbol",
                "date",
                "stoch_osc",
                "support",
                "resistance",
                "bullish_crossover",
                "bearish_crossover",
            ]
        ],
        on=["symbol", "date"],
        how="left",
    )
    # Writes the previous close_deltas, this duplicates memory a lot but helps with processing speed significantly,
    # also it makes the logic simpler, so for now this is okay.
    for i in range(1, lookback_minutes):
        candles_minute[f"close_m{i}"] = (
            (
                candles_minute.close.shift(i + 1, fill_value=True)
                - candles_minute.close.shift(i, fill_value=True)
            )
            / 1e-5
        ).astype(int)

    # Filters out those candles corresponding to days for which resistnace / stochosc could not be computed (because candles are missing)
    candles_minute.dropna(inplace=True)

    candles_minute = candles_minute.iloc[15:]

    # Filters those minute candles out which had a large timejump in the last couple of minutes, usually a new day
    candles_minute = candles_minute[
        candles_minute.ts.diff(lookback_minutes + 1)
        < dt.timedelta(minutes=lookback_minutes + 10)
    ]

    candles_minute_pruned = candles_minute[
        [
            "close",
            "volume",
            "volatility",
            "stoch_osc",
            "support",
            "resistance",
            "bullish_crossover",
            "bearish_crossover",
        ]
        + [f"close_m{i}" for i in range(1, lookback_minutes)]
    ].copy()
    candles_minute.reset_index(inplace=True, drop=True)
    candles_minute_pruned.reset_index(inplace=True, drop=True)


def train_model():
    df = candles_minute_pruned.iloc[:100000]

    # Define your features and target
    features = df.drop(columns=["close"])  # Assuming 'close' is your target variable
    target = df["close"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, shuffle=False
    )

    # Initialize the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    predictions = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, predictions)
    print(f"Mean Absolute Error: {mae}")

    # Backtesting logic here...
    # Generate signals based on predictions, simulate trading, and calculate performance metrics


def determine_signal():
    pct_deltas = np.log(
        model.predict(candles_minute_pruned.drop(columns=["close"]))
        / candles_minute_pruned.close
    )

    candles_minute["buy_signal"] = False
    candles_minute.loc[np.where(pct_deltas > 0.02)[0], "buy_signal"] = True
    candles_minute["sell_signal"] = False
    candles_minute.loc[np.where(pct_deltas < -0.005)[0], "sell_signal"] = True

    candles_minute.dropna()


def plot_signals():
    candles_minute.close.plot()

    plt.scatter(
        candles_minute[candles_minute.buy_signal].index,
        candles_minute[candles_minute.buy_signal].close,
        c="green",
        zorder=3,
        alpha=0.01,
    )
    plt.scatter(
        candles_minute[candles_minute.sell_signal].index,
        candles_minute[candles_minute.sell_signal].close,
        c="red",
        zorder=2,
        alpha=0.01,
    )
