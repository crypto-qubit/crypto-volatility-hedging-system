"""
Threshold-engine lookahead guard rail.

ThresholdEngine.fit() calibrates 99.75th-percentile thresholds on the FULL
sample passed to it — intentional, and documented as "DUMMY BASELINE — NOT
FOR PRODUCTION USE" in src/thresholds/threshold_engine.py. This is only safe
for latest-bar live alerting ("full sample up to now" is not a lookahead
problem there); it becomes a lookahead problem the moment flag() output is
replayed as a historical time-series input, since every past bar's flag
would then be contaminated by future data.

These tests make that property machine-checkable rather than just documented
in prose, and exercise the assert_safe_for_historical_replay() guard rail
that callers are expected to run before doing exactly that.
"""

import numpy as np
import pandas as pd

from src.thresholds.derived_returns import DerivedReturnsCalculator
from src.thresholds.threshold_engine import (
    _LABEL_PREFIX,
    ThresholdEngine,
    assert_safe_for_historical_replay,
)


def _make_ohlcv(
    n: int,
    symbol: str = "BTC",
    base_price: float = 30_000.0,
    seed: int = 7,
    vol_scale: float = 0.01,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0, vol_scale, size=n)
    close = np.exp(np.cumsum(log_ret) + np.log(base_price))
    noise = rng.uniform(0.001, 0.005, n)
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    ts = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(100, 5000, n),
            "symbol": symbol,
        }
    )


def _derived(ohlcv_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    return DerivedReturnsCalculator().compute(ohlcv_df, symbol)


def test_full_sample_lookahead_changes_thresholds():
    """
    Fitting on a truncated sample vs. the full sample (which contains a
    later volatility spike) produces different 1h thresholds — proving
    fit() is sensitive to data points "in the future" of any given bar.

    When rolling/expanding-window calibration replaces this dummy baseline
    (see docs/limitations.md), this test should be replaced with one
    asserting thresholds for bar t are invariant to data appended after t.
    """
    full_ohlcv = _make_ohlcv(n=600)
    truncated_ohlcv = full_ohlcv.iloc[:400].copy()

    # Inject a single large spike late in the sample (bar 590), well after
    # the truncated window ends at bar 399.
    spike_idx = full_ohlcv.index[590]
    prev_close = full_ohlcv.loc[full_ohlcv.index[589], "close"]
    full_ohlcv.loc[spike_idx, "close"] = prev_close * 1.50

    engine = ThresholdEngine()
    full_thresholds = engine.fit(_derived(full_ohlcv, "BTC"))
    truncated_thresholds = engine.fit(_derived(truncated_ohlcv, "BTC"))

    assert full_thresholds.hourly_return_p9975 != truncated_thresholds.hourly_return_p9975, (
        "Full-sample and truncated-sample 1h thresholds should differ — "
        "if they don't, this regression test no longer demonstrates "
        "full-sample lookahead and should be revisited."
    )


def test_all_flag_labels_carry_dummy_baseline_prefix():
    """
    Machine-checkable marker: every row produced by ThresholdEngine.flag()
    carries the DUMMY_BASELINE prefix in flag_label.
    assert_safe_for_historical_replay() relies on this prefix to detect
    full-sample-lookahead-contaminated rows.
    """
    ohlcv = _make_ohlcv(n=1000, seed=7)
    engine = ThresholdEngine()
    derived = _derived(ohlcv, "BTC")
    thresholds = engine.fit(derived)
    flags = engine.flag(derived, thresholds)

    assert not flags.empty, "Expected at least some threshold flags in this sample"
    assert flags["flag_label"].str.startswith(_LABEL_PREFIX).all()


def test_assert_safe_for_historical_replay_raises_on_dummy_baseline_flags():
    ohlcv = _make_ohlcv(n=1000, seed=7)
    engine = ThresholdEngine()
    derived = _derived(ohlcv, "BTC")
    flags = engine.flag(derived, engine.fit(derived))

    try:
        assert_safe_for_historical_replay(flags)
        raised = False
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for DUMMY_BASELINE-flagged historical replay"


def test_assert_safe_for_historical_replay_passes_on_empty_flags():
    empty = pd.DataFrame(columns=["flag_label"])
    assert_safe_for_historical_replay(empty)  # must not raise
