import numpy as np
from scipy.signal import lfilter


def ema_filter(data: np.ndarray, lookback: int):
    """
    Compute the Exponential Moving Average (EMA) using a filter.

    This is around 40% faster than using
    ```python
    pd.Series(data).ewm(span=lookback, adjust=False).mean().to_numpy()
    ```

    Initializes the filter the same way the pandas variant does.

    Deprecated for the handrolled `computeEMA` c++ function.
    """
    alpha = 2 / (lookback + 1)
    b = [alpha]
    a = [1, -(1 - alpha)]

    ema_values = lfilter(b, a, data, zi=[data[0] * (1 - alpha)])[0]

    return ema_values
