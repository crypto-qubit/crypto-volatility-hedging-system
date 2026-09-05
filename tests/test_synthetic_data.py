"""
Tests for scripts/generate_sample_data.py — determinism and OHLC validity
of the synthetic demo dataset.
"""

import numpy as np

from scripts.generate_sample_data import (
    BTC_SEED,
    _generate_symbol,
    generate_sample_data,
)


def test_same_seed_is_fully_deterministic():
    a = _generate_symbol("BTC", 50_000.0, 500, BTC_SEED, "2024-01-01", tail_shock=True)
    b = _generate_symbol("BTC", 50_000.0, 500, BTC_SEED, "2024-01-01", tail_shock=True)
    assert a["close"].tolist() == b["close"].tolist()


def test_different_seed_gives_different_series():
    a = _generate_symbol("BTC", 50_000.0, 500, seed=1, start_date="2024-01-01")
    b = _generate_symbol("BTC", 50_000.0, 500, seed=2, start_date="2024-01-01")
    assert a["close"].tolist() != b["close"].tolist()


def test_ohlc_geometry_is_always_valid():
    df = _generate_symbol("BTC", 50_000.0, 2000, BTC_SEED, "2024-01-01", tail_shock=True)
    assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (df["high"] >= df["low"]).all()
    assert (df[["open", "high", "low", "close"]] > 0).all().all()
    assert (df["volume"] > 0).all()


def test_tail_shock_elevates_recent_volatility():
    with_shock = _generate_symbol("BTC", 50_000.0, 1000, BTC_SEED, "2024-01-01", tail_shock=True)
    without_shock = _generate_symbol(
        "BTC", 50_000.0, 1000, BTC_SEED, "2024-01-01", tail_shock=False
    )

    tail_returns_shock = np.diff(np.log(with_shock["close"].tail(45)))
    tail_returns_calm = np.diff(np.log(without_shock["close"].tail(45)))
    assert tail_returns_shock.std() > tail_returns_calm.std()


def test_generate_sample_data_writes_both_symbols(tmp_path):
    paths = generate_sample_data(days=20, out_dir=tmp_path)
    assert set(paths) == {"BTC", "ETH"}
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0
