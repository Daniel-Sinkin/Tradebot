from typing import Optional

import numpy as np
import pandas as pd


def build_candle(
    prices: pd.Series,
    timeframe: str,
    volumes: Optional[float | pd.Series] = None,
    include_tick_volumes: bool = False,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Constructs a DataFrame of OHLC (Open, High, Low, Close) candles from price data.

    ### Parameters:
    * prices : pd.Series
        * A pandas Series of price data, indexed by datetime.
    * timeframe : str
        * A string representing the resampling timeframe (e.g., '1T' for 1 minute, '1H' for 1 hour).
    * volumes : Optional[float | pd.Series], optional
        * Either a single float representing the volume for each period or a pandas Series of volumes, indexed by datetime.
        * If None, volume data will not be included.
    * include_tick_volumes : bool, optional
        * If True, includes a column for tick volumes (i.e., the count of trades per resampled period). Default is False.
    * dropna : bool, optional
        * If True, drops any NaN values from the resulting DataFrame. Default is True.

    ### Returns:
    * pd.DataFrame
        * A DataFrame with columns for Open, High, Low, Close prices, and optionally Volume and TickVolume.

    ### Raises:
    * TypeError
        * If the provided volumes argument is neither a float nor a pd.Series.
    """
    resampled = prices.resample(timeframe)
    candles = resampled.ohlc()

    if include_tick_volumes:
        candles["TickVolume"] = resampled.count()

    if volumes is not None:
        if isinstance(volumes, float):
            candles["Volume"] = volumes
        elif isinstance(volumes, pd.Series):
            assert len(volumes) == len(
                candles
            ), "Volume series must match the length of the candles DataFrame."
            candles["Volume"] = volumes
        else:
            raise TypeError(f"Invalid type for volumes: {type(volumes)=}.")

    candles.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"},
        inplace=True,
    )

    if dropna:
        candles.dropna(inplace=True)

    return candles


def slice_sorted(
    self, df: pd.DataFrame, key: str, include_left: bool = True, include_right=False
) -> pd.DataFrame:
    """
    Slices a DataFrame according to some key. Significantly faster than boolean slicing which doesn't
    use the sorted property.

    ### Parameters:
    * df : pd.DataFrame
        * The DataFrame to slice.
    * key : str
        * The column name used as the key for slicing. This column should be sorted.
    * include_left : bool, optional
        * If True, the slice includes the left bound. Default is True.
    * include_right : bool, optional
        * If True, the slice includes the right bound. Default is False.

    ### Returns:
    * pd.DataFrame
        * A sliced DataFrame according to the key.

    ### Raises:
    * TypeError
        * If `df` is not a pandas DataFrame.
    * KeyError
        * If the `key` is not found in the DataFrame columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")

    if key not in pd.DataFrame.columns:
        raise KeyError(f"Column '{key}' not found in df.")

    left_side = "left" if include_left else "right"
    right_side = "right" if include_right else "left"

    idx0 = int(df[key].searchsorted(df[key].iloc[0], side=left_side))
    idx1 = int(df[key].searchsorted(df[key].iloc[-1], side=right_side))

    if idx0 == 0:
        idx0 = None
    if idx1 == 0 or idx1 >= len(df):
        idx1 = None

    return df.iloc[idx0:idx1]


def sample_eps_ball(
    center: np.ndarray, eps: float, n_samples: int = 1, seed: Optional[int] = None
) -> np.ndarray:
    """
    Samples points around an epsilon ball to validate how stable the procedure is.

    This is not a uniform distribution in the ball, preferring points around the center, but that is good enough for us.

    ### Parameters:
    * center : np.ndarray
        * An array representing the center of the epsilon ball. The shape is `(n,)` where `n` is the dimensionality of the space.
    * eps : float
        * The radius of the epsilon ball. Must be a positive value.
    * n_samples : int, optional
        * The number of samples to generate. The resulting array will have a shape of `(n_samples, n)`. Default is 1.
    * seed : Optional[int], optional
        * An optional random seed for reproducibility. Default is None.

    ### Returns:
    * np.ndarray
        * An array of shape `(n_samples, n)` representing the sampled points within the epsilon ball.

    ### Raises:
    * AssertionError
        * If `eps` is not greater than 0.
    """
    assert eps > 0, "radius has to be positive"
    _rng = np.random.default_rng(seed)

    # Generate random radii for all samples
    radii = eps - _rng.uniform(0, eps, size=n_samples)

    # Generate normal distributed samples in a vectorized manner
    samples = _rng.normal(0, 1.0, size=(n_samples, len(center)))

    # Normalize each sample to lie on the surface of a unit sphere
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    samples /= norms

    # Scale samples by the corresponding radii
    samples *= radii[:, np.newaxis]

    # Shift the samples by the center
    samples += center

    return samples