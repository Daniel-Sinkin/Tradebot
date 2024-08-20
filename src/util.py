from typing import Optional, TypeVar

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
            # Resample the volumes Series to match the resampling of prices
            resampled_volumes = volumes.resample(timeframe).sum()
            assert len(resampled_volumes) == len(
                candles
            ), "Volume series must match the length of the candles DataFrame."
            candles["Volume"] = resampled_volumes
        else:
            raise TypeError(f"Invalid type for volumes: {type(volumes)=}.")

    candles.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"},
        inplace=True,
    )

    if dropna:
        candles.dropna(inplace=True)

    return candles


T = TypeVar("T", int, float, pd.Timestamp, str)


def slice_sorted(  # pylint: disable=too-many-arguments
    df: pd.DataFrame,
    key: str,
    left: T | None = None,
    right: T | None = None,
    left_inclusive: bool = True,
    right_inclusive: bool = False,
) -> pd.DataFrame:
    """
    Slices a DataFrame based on a key column, with flexible boundary conditions.

    ### Parameters:
    * df : pd.DataFrame
        * The DataFrame to slice.
    * key : str
        * The column name to perform the slicing on.
    * left : T | None, optional
        * Comparable to the key column, this is the lower boundary for slicing (default: None).
    * right : T | None, optional
        * Comparable to the key column, this is the upper boundary for slicing (default: None).
    * left_inclusive : bool, optional
        * If True, the lower bound is inclusive (>=). If False, it is exclusive (>). Default is True.
    * right_inclusive : bool, optional
        * If True, the upper bound is inclusive (<=). If False, it is exclusive (<). Default is False.

    ### Returns:
    * pd.DataFrame
        * The sliced DataFrame based on the specified conditions.

    ### Raises:
    * ValueError
        * If the key column is not found in the DataFrame.
        * If the left boundary is greater than the right boundary.

    ### Notes:
    * Requires `left` and `right` to be comparable to the `df[key]` values, such as numerics, datetime objects, or other classes that implement rich comparison.
    * The default configuration covers the case `df[(left <= df["key"]) & (df["key"] < right)]`.
    * This function is associative, meaning that the following expressions are identical:
        - `slice_sorted(slice_sorted(df, "key", left=t0), right=t1)`
        - `slice_sorted(slice_sorted(df, "key", right=t1), left=t0)`
        - `slice_sorted(df, "key", left=t0, right=t1)`
    """
    if key not in df.columns:
        raise ValueError(f"Column {key} not found in DataFrame.")

    if left is not None and right is not None and left > right:
        raise ValueError("Left boundary is greater than right boundary.")

    if right is None:
        # df[t0 <= df["ts"]] or df[t0 < df["ts"]] or df
        right_idx: int = len(df)
    else:
        right_idx: int = np.searchsorted(
            df[key], right, side="right" if right_inclusive else "left"
        )

    if left is None:
        # df[df["ts"] <= t1] or df[df["ts"] < t1] or df
        left_idx: int = 0
    else:
        left_idx: int = np.searchsorted(
            df[key], left, side="left" if left_inclusive else "right"
        )

    return df.iloc[left_idx:right_idx]


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
