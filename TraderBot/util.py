import sqlite3
from typing import Optional, TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T", int, float, pd.Timestamp, str)


def slice_sorted(
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
    * df
        * The DataFrame to slice.
    * key
        * The column name to perform the slicing on.
    * left
        * Comparable to the key column, this is the lower boundary for slicing.
    * right
        * Comparable to the key column, this is the upper boundary for slicing.
    * left_inclusive
        * If True, the lower bound is inclusive (>=). If False, it is exclusive (>).
    * right_inclusive
        * If True, the upper bound is inclusive (<=). If False, it is exclusive (<).

    ### Returns:
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
    if df is None or df.columns is None:
        raise TypeError("df is None or its columns are None.")

    if key not in df.columns:
        raise KeyError(f"Column {key} not found in DataFrame.")

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


@staticmethod
def format_connection(conn: sqlite3.Connection) -> str:
    """Return a pretty formatted string representation of the connection."""
    db_name = conn.execute("PRAGMA database_list").fetchall()[0][2]
    connection_id = id(conn)
    in_memory = db_name == ":memory:"
    conn_type = "In-Memory" if in_memory else "File-Based"
    return f"[Connection ID: {connection_id}, Database: {db_name}, Type: {conn_type}]"
