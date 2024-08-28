import numpy as np

def computeEMA(data_array: np.ndarray, lookback: int) -> np.ndarray: ...
def computeStochasticOscillator(
    close_prices: np.ndarray,
    low_prices: np.ndarray,
    high_prices: np.ndarray,
    lookback: int,
) -> np.ndarray: ...
def computeResistance(high_prices: np.ndarray, lookback: int) -> np.ndarray: ...
def computeSupport(low_prices: np.ndarray, lookback: int) -> np.ndarray: ...

__all__ = [
    "computeEMA",
    "computeStochasticOscillator",
    "computeResistance",
    "computeSupport",
]
