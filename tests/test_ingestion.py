"""
Tests for src/ingestion/loader.py.
"""

import pandas as pd
import pytest

from src.ingestion.loader import CANONICAL_COLUMNS, load_hourly_csv


def test_load_csv_with_symbol_column(tmp_path):
    path = tmp_path / "btc.csv"
    pd.DataFrame(
        {
            "timestamp": ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"],
            "symbol": ["BTC", "BTC"],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [10.0, 12.0],
        }
    ).to_csv(path, index=False)

    out = load_hourly_csv(path)
    assert list(out.columns) == CANONICAL_COLUMNS
    assert (out["symbol"] == "BTC").all()
    assert pd.api.types.is_datetime64_any_dtype(out["timestamp"])


def test_load_csv_without_symbol_column_requires_override(tmp_path):
    path = tmp_path / "eth.csv"
    pd.DataFrame(
        {
            "timestamp": ["2024-01-01T00:00:00Z"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [5.0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError):
        load_hourly_csv(path)

    out = load_hourly_csv(path, symbol="ETH")
    assert (out["symbol"] == "ETH").all()


def test_load_csv_missing_required_column_raises(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp": ["2024-01-01T00:00:00Z"], "open": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_hourly_csv(path, symbol="BTC")


def test_load_csv_sorts_by_timestamp(tmp_path):
    path = tmp_path / "unsorted.csv"
    pd.DataFrame(
        {
            "timestamp": ["2024-01-01T02:00:00Z", "2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"],
            "symbol": ["BTC"] * 3,
            "open": [1.0, 2.0, 3.0],
            "high": [1.0, 2.0, 3.0],
            "low": [1.0, 2.0, 3.0],
            "close": [1.0, 2.0, 3.0],
            "volume": [1.0, 1.0, 1.0],
        }
    ).to_csv(path, index=False)

    out = load_hourly_csv(path)
    assert out["timestamp"].is_monotonic_increasing
