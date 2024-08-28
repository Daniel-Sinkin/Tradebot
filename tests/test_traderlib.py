# Test cases using pytest
import numpy as np
import pandas as pd
import pytest

from TraderBot.traderlib import ema_filter


@pytest.mark.parametrize(
    "data, span",
    [
        (np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32), 3),
        (np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=np.float32), 5),
        (np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32), 2),
        (np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2], dtype=np.float32), 4),
    ],
)
def test_ema_filter_vs_pandas(data, span):
    # Calculate EMA using our function
    ema_custom = ema_filter(data, span)

    # Calculate EMA using pandas
    ema_pandas = pd.Series(data).ewm(span=span, adjust=False).mean().to_numpy()

    # Compare the two results
    np.testing.assert_allclose(ema_custom, ema_pandas, rtol=1e-5, atol=1e-8)
