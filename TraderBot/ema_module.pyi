# ema_module.pyi

import numpy as np

def computeEMA(data: np.ndarray, lookback: int) -> np.ndarray:
    """
    Compute Exponential Moving Average (EMA) without loop unrolling.

    Parameters:
        data (np.ndarray): The input data for which EMA is calculated.
        lookback (int): The lookback period for EMA calculation.

    Returns:
        np.ndarray: The EMA of the input data as a NumPy array.
    """
