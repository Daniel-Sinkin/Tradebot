# Algorithmic Trading Bot

This project implements an algorithmic trading bot designed to execute and optimize trading strategies based on historical market data. The bot is built with modular components that allow for the easy integration of new strategies, data sources, and optimization techniques. The current focus is on a Differential Momentum Strategy, but the infrastructure is in place to support a variety of strategies.

## Strategy: Differential Momentum Strategy

The Differential Momentum Strategy (DMS) implemented in this project is a momentum-based trading strategy. The core idea is to generate buy and sell signals based on the comparison of recent price differences (deltas) with their exponentially weighted moving average (EMA).
### How It Works
1. Deltas Calculation: For each symbol, the strategy calculates the price differences (Deltas) over time.
2. EMA Calculation: An Exponential Moving Average (EMA) is computed for the deltas.
3. Signal Generation:
    * Buy Signal: Generated when recent deltas are consistently below their EMA over a specified lookback window.
    * Sell Signal: Generated when recent deltas are consistently above their EMA over the lookback window.
4. Backtesting: The strategy is backtested using the vectorbt library, which allows for detailed analysis of the strategy's performance over historical data.

```python
from src.strategies import DifferentialMomentumStrategy
import vectorbt as vbt

# Assume candles_dict and weights are predefined
strategy = DifferentialMomentumStrategy(candles_dict, weights, lookback_window=10, ema_span=800, timeframe="1H")
portfolio = strategy.run_backtest()

# Analyze the portfolio
print(portfolio.stats())
portfolio.plot().show()
```

## Infrastructure

The project uses SQLite as the database for storing and managing tick data. The database can be initialized using the infrastructure/database.py script, which sets up the necessary tables and allows for data insertion from pickle files.

## Optimization

The project includes an Optimizer abstract base class, with a concrete implementation using Simulated Annealing. This allows for the fine-tuning of strategy parameters by optimizing for metrics like the Sharpe ratio or total returns.

```python
from src.optimizer import SimulatedAnnealingOptimizer
from src.strategies import DifferentialMomentumStrategy

# Define candles_dict and other necessary inputs
optimizer = SimulatedAnnealingOptimizer(candles_dict, DifferentialMomentumStrategy, num_symbols=5, maxiter=25)
results = optimizer.run_optimizations_in_parallel(lookback_windows=[6, 7, 8, 9, 10, 11, 12])

for result in results:
    print(result)
```

## Reporting

The reporting.py module provides functionality to generate reports from the backtest results. Reports can be generated in multiple formats, including interactive Dash dashboards, PDF files, and Excel spreadsheets.

## Future Work

Future developments in this project may include:
Adding more sophisticated trading strategies.
Enhancing the optimization framework with additional algorithms.
Expanding data management capabilities to include cloud storage and more complex databases.
Integrating real-time data feeds and live trading capabilities.