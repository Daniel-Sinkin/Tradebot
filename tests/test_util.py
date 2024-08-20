import numpy as np
import pandas as pd
import pytest

from src.util import build_candle, sample_eps_ball, slice_sorted


def test_build_candle_basic():
    prices = pd.Series(
        [100, 102, 101, 103, 102],
        index=pd.date_range("2023-01-01 09:00", periods=5, freq="min"),
    )
    result = build_candle(prices, "2min")
    expected_columns = ["Open", "High", "Low", "Close"]
    assert list(result.columns) == expected_columns
    assert len(result) == 3


def test_build_candle_with_volumes():
    prices = pd.Series(
        [100, 102, 101, 103, 102],
        index=pd.date_range("2023-01-01 09:00", periods=5, freq="min"),
    )
    volumes = pd.Series(
        [1000, 1500, 1200, 1800, 1700],
        index=pd.date_range("2023-01-01 09:00", periods=5, freq="min"),
    )
    result = build_candle(prices, "2min", volumes=volumes)
    assert "Volume" in result.columns
    assert len(result) == 3


def test_build_candle_with_invalid_volumes():
    prices = pd.Series(
        [100, 102, 101, 103, 102],
        index=pd.date_range("2023-01-01 09:00", periods=5, freq="min"),
    )
    with pytest.raises(TypeError):
        build_candle(prices, "2min", volumes=[1000, 1500, 1200])


def test_build_candle_with_tick_volumes():
    prices = pd.Series(
        [100, 102, 101, 103, 102],
        index=pd.date_range("2023-01-01 09:00", periods=5, freq="min"),
    )
    result = build_candle(prices, "2min", include_tick_volumes=True)
    assert "TickVolume" in result.columns
    assert len(result) == 3


def test_slice_sorted_basic():
    df = pd.DataFrame({"key": [1, 2, 3, 4, 5], "value": ["a", "b", "c", "d", "e"]})
    result = slice_sorted(None, df, "key")
    assert len(result) == len(df)


def test_slice_sorted_left_bound():
    df = pd.DataFrame({"key": [1, 2, 3, 4, 5], "value": ["a", "b", "c", "d", "e"]})
    result = slice_sorted(None, df, "key", include_left=False)
    assert result["key"].iloc[0] == 2


def test_slice_sorted_right_bound():
    df = pd.DataFrame({"key": [1, 2, 3, 4, 5], "value": ["a", "b", "c", "d", "e"]})
    result = slice_sorted(None, df, "key", include_right=True)
    assert result["key"].iloc[-1] == 5


def test_slice_sorted_key_not_in_df():
    df = pd.DataFrame({"key": [1, 2, 3, 4, 5], "value": ["a", "b", "c", "d", "e"]})
    with pytest.raises(KeyError):
        slice_sorted(None, df, "non_existent_key")


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
