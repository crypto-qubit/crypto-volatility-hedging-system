"""
Tests for src/validation/schema.py — structural OHLCV validation used
internally by every volatility estimator.
"""

import pandas as pd
import pytest

from src.validation.schema import ValidationError, check_sufficient_history, validate_ohlcv
from tests.conftest import make_ohlcv


def test_valid_frame_passes_through():
    df = make_ohlcv(n=20)
    out = validate_ohlcv(df, symbol="BTC")
    assert len(out) == 20
    assert out["timestamp"].dt.tz is not None


def test_missing_columns_raise():
    df = make_ohlcv(n=20).drop(columns=["volume"])
    with pytest.raises(ValidationError):
        validate_ohlcv(df)


def test_empty_frame_raises():
    with pytest.raises(ValidationError):
        validate_ohlcv(
            pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"])
        )


def test_unknown_symbol_raises():
    df = make_ohlcv(n=20)
    with pytest.raises(ValidationError):
        validate_ohlcv(df, symbol="DOGE")


def test_duplicate_timestamps_raise():
    df = make_ohlcv(n=20)
    df.loc[1, "timestamp"] = df.loc[0, "timestamp"]
    with pytest.raises(ValidationError):
        validate_ohlcv(df, symbol="BTC")


def test_nonpositive_price_is_nullified_not_dropped():
    df = make_ohlcv(n=20)
    df.loc[5, "close"] = -1.0
    out = validate_ohlcv(df, symbol="BTC")
    assert out.loc[5, ["open", "high", "low", "close"]].isna().all()
    assert len(out) == 20  # row kept, not dropped


def test_high_below_low_bar_is_nullified():
    df = make_ohlcv(n=20)
    df.loc[3, "high"] = df.loc[3, "low"] - 10
    out = validate_ohlcv(df, symbol="BTC")
    assert out.loc[3, ["open", "high", "low", "close"]].isna().all()


def test_check_sufficient_history_raises_when_short():
    df = make_ohlcv(n=5)
    with pytest.raises(ValidationError):
        check_sufficient_history(df, window=24)


def test_check_sufficient_history_passes_when_enough():
    df = make_ohlcv(n=30)
    check_sufficient_history(df, window=24)  # should not raise
