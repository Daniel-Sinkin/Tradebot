import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes

from TraderBot.util import sample_eps_ball, slice_sorted


def test_slice_sorted_basic_case():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": ["a", "b", "c", "d", "e"]})

    result = slice_sorted(df, key="A", left=2, right=4)
    expected = df.iloc[1:3]  # Rows with A=2 and A=3

    pd.testing.assert_frame_equal(result, expected)


def test_slice_sorted_left_inclusive():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": ["a", "b", "c", "d", "e"]})

    result = slice_sorted(df, key="A", left=2, right=4, left_inclusive=False)
    expected = df.iloc[2:3]  # Row with A=3

    pd.testing.assert_frame_equal(result, expected)


def test_slice_sorted_right_inclusive():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": ["a", "b", "c", "d", "e"]})

    result = slice_sorted(df, key="A", left=2, right=4, right_inclusive=True)
    expected = df.iloc[1:4]  # Rows with A=2, A=3, A=4

    pd.testing.assert_frame_equal(result, expected)


def test_slice_sorted_no_left_boundary():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": ["a", "b", "c", "d", "e"]})

    result = slice_sorted(df, key="A", right=3)
    expected = df.iloc[:2]  # Rows with A=1, A=2

    pd.testing.assert_frame_equal(result, expected)


def test_slice_sorted_no_right_boundary():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": ["a", "b", "c", "d", "e"]})

    result = slice_sorted(df, key="A", left=3)
    expected = df.iloc[2:]  # Rows with A=3, A=4, A=5

    pd.testing.assert_frame_equal(result, expected)


def test_slice_sorted_non_existent_key():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": ["a", "b", "c", "d", "e"]})

    with pytest.raises(KeyError):
        slice_sorted(df, key="C", left=2, right=4)


def test_slice_sorted_invalid_boundaries():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": ["a", "b", "c", "d", "e"]})

    with pytest.raises(ValueError):
        slice_sorted(df, key="A", left=4, right=2)


def test_slice_sorted_none_df():
    with pytest.raises(TypeError):
        slice_sorted(None, key="A", left=1, right=3)


def test_slice_sorted_dates():
    df = pd.DataFrame(
        {
            "A": pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
            ),
            "B": [10, 20, 30, 40],
        }
    )

    result = slice_sorted(
        df, key="A", left=pd.Timestamp("2023-01-02"), right=pd.Timestamp("2023-01-04")
    )
    expected = df.iloc[1:3]  # Rows with dates "2023-01-02" and "2023-01-03"

    pd.testing.assert_frame_equal(result, expected)


def test_slice_sorted_string_keys():
    df = pd.DataFrame(
        {
            "A": ["apple", "banana", "cherry", "date", "elderberry"],
            "B": [10, 20, 30, 40, 50],
        }
    )

    result = slice_sorted(df, key="A", left="banana", right="date")
    expected = df.iloc[1:3]  # Rows with "banana" and "cherry"

    pd.testing.assert_frame_equal(result, expected)


@given(
    data_frames(
        columns=[
            column("A", st.integers(min_value=0, max_value=1000)),
            column("B", st.floats(min_value=0.0, max_value=1000.0)),
        ],
        index=range_indexes(min_size=1000, max_size=1000),
    ).map(lambda df: df.sort_values(by="A")),
    st.integers(min_value=0, max_value=999),
    st.integers(min_value=0, max_value=999),
    st.booleans(),
    st.booleans(),
)
def test_slice_sorted_fuzzy(df, left, right, left_inclusive, right_inclusive):
    if left > right:
        return  # Skip invalid cases where left > right

    left_bound = df["A"].iloc[left] if left_inclusive else df["A"].iloc[left] + 1
    right_bound = df["A"].iloc[right] if right_inclusive else df["A"].iloc[right] - 1

    # Use boolean slicing for the equivalent operation
    expected = df[(df["A"] >= left_bound) & (df["A"] <= right_bound)]
    result = slice_sorted(
        df,
        key="A",
        left=df["A"].iloc[left],
        right=df["A"].iloc[right],
        left_inclusive=left_inclusive,
        right_inclusive=right_inclusive,
    )

    pd.testing.assert_frame_equal(result, expected)


def test_sample_eps_ball_basic():
    center = np.array([0, 0])
    eps = 1.0
    result = sample_eps_ball(center, eps, n_samples=5)
    assert result.shape == (5, 2)
    norms = np.linalg.norm(result - center, axis=1)
    assert np.all(norms <= eps)


def test_sample_eps_ball_with_seed():
    center = np.array([0, 0])
    eps = 1.0
    result1 = sample_eps_ball(center, eps, n_samples=5, seed=42)
    result2 = sample_eps_ball(center, eps, n_samples=5, seed=42)
    np.testing.assert_array_equal(result1, result2)


def test_sample_eps_ball_large_dimension():
    center = np.array([0, 0, 0, 0])
    eps = 1.0
    result = sample_eps_ball(center, eps, n_samples=3)
    assert result.shape == (3, 4)
    norms = np.linalg.norm(result - center, axis=1)
    assert np.all(norms <= eps)


def test_sample_eps_ball_assertion():
    center = np.array([0, 0])
    eps = -1.0
    with pytest.raises(AssertionError):
        sample_eps_ball(center, eps)
