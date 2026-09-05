"""
Tests for the threshold system (src/thresholds/): derived returns,
threshold engine (dummy-baseline boundary behavior), event windowing, and
the stress summary rollup.
"""

import numpy as np
import pandas as pd
import pytest

from src.thresholds.derived_returns import DerivedReturnsCalculator
from src.thresholds.event_windower import EventWindower
from src.thresholds.stress_summary import StressSummaryBuilder
from src.thresholds.threshold_engine import ThresholdEngine
from tests.conftest import make_ohlcv


@pytest.fixture
def derived(btc_df):
    return DerivedReturnsCalculator().compute(btc_df, "BTC")


def test_derived_returns_schema_and_nan_policy(derived, btc_df):
    assert len(derived) == len(btc_df)
    assert derived["log_return_1h"].iloc[0] != derived["log_return_1h"].iloc[0]  # NaN at bar 0
    assert derived["rolling_drawdown_24h"].iloc[0] == 0.0  # non-NaN from bar 0


def test_derived_returns_drawdown_is_non_positive(derived):
    assert (derived["rolling_drawdown_24h"].dropna() <= 1e-12).all()
    assert (derived["rolling_drawdown_168h"].dropna() <= 1e-12).all()


def test_threshold_engine_fit_produces_positive_thresholds(derived):
    fitted = ThresholdEngine().fit(derived)
    assert fitted.hourly_return_p9975 > 0
    assert fitted.daily_return_p9975 > 0
    assert fitted.symbol == "BTC"


def test_threshold_engine_flags_synthetic_spike():
    """A single 8% one-hour move must clear both the percentile and the hard floor."""
    df = make_ohlcv(n=400, seed=7)
    spike_idx = 300
    df.loc[spike_idx:, ["open", "high", "low", "close"]] *= 1.08

    derived = DerivedReturnsCalculator().compute(df, "BTC")
    engine = ThresholdEngine()
    fitted = engine.fit(derived)
    flags = engine.flag(derived, fitted)

    hourly_flags = flags[flags["horizon"] == "1h"]
    assert (hourly_flags["timestamp"] == derived["timestamp"].iloc[spike_idx]).any()


def test_threshold_engine_floor_is_fixed_not_data_dependent():
    df = make_ohlcv(n=200, seed=3, base_price=100.0)
    df["close"] = df["open"]  # flatten so the calibrated percentile sits near zero
    derived = DerivedReturnsCalculator().compute(df, "BTC")
    fitted = ThresholdEngine().fit(derived)
    assert fitted.hourly_return_floor == pytest.approx(np.log(1.015))
    assert fitted.daily_return_floor == pytest.approx(np.log(1.075))


def test_event_windower_merges_within_gap_and_splits_beyond_it():
    flags = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01T00:00Z", "2024-01-01T01:00Z", "2024-01-05T00:00Z"], utc=True
            ),
            "symbol": "BTC",
            "horizon": "1h",
            "raw_return": [0.02, -0.02, 0.03],
            "abs_return": [0.02, 0.02, 0.03],
            "percentile_rank": [99.8, 99.8, 99.9],
            "rolling_volatility": [0.5, 0.5, 0.6],
            "drawdown_size": [0.0, -0.01, -0.02],
            "threshold_breached": ["RETURN_PCT_9975"] * 3,
            "flag_label": ["DUMMY_BASELINE_1H_STRESS"] * 3,
            "event_window_id": ["", "", ""],
        }
    )
    flags_out, windows = EventWindower(merge_gap_hours=48).window(flags)

    assert windows["event_window_id"].nunique() == 2
    assert set(flags_out["event_window_id"]) == set(windows["event_window_id"])


def test_event_windower_empty_input_returns_empty_frames():
    empty = pd.DataFrame(
        columns=[
            "timestamp",
            "symbol",
            "horizon",
            "raw_return",
            "abs_return",
            "percentile_rank",
            "rolling_volatility",
            "drawdown_size",
            "threshold_breached",
            "flag_label",
            "event_window_id",
        ]
    )
    flags_out, windows = EventWindower().window(empty)
    assert flags_out.empty
    assert windows.empty


def test_stress_summary_rolls_up_by_symbol_and_horizon(derived):
    engine = ThresholdEngine()
    fitted = engine.fit(derived)
    flags = engine.flag(derived, fitted)
    flags_with_ids, windows = EventWindower().window(flags)

    summary = StressSummaryBuilder().build(flags_with_ids, windows)
    if not summary.empty:
        assert set(summary["symbol"]) <= {"BTC"}
        assert (summary["n_total_flags"] >= summary["n_event_windows"]).all()
