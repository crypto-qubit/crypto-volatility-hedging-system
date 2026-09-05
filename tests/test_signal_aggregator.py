"""
Tests for decision-layer signal aggregation.
"""

from datetime import datetime, timezone

from src.decision import signal_aggregator
from src.decision.signal_aggregator import SignalAggregator


def test_signal_aggregator_regime_comes_from_detect_market_regime(monkeypatch):
    calls = []

    def fake_detect(log_returns):
        calls.append(log_returns)
        return {
            "regime": "high_vol",
            "confidence": 0.83,
            "recent_vol_window": 12,
            "recent_vol": 0.02,
            "overall_vol": 0.015,
            "vol_ratio": 1.33,
            "skewness": 0.1,
            "excess_kurtosis": 2.0,
            "mean_return": 0.0,
        }

    monkeypatch.setattr(signal_aggregator, "detect_market_regime", fake_detect)

    returns_1h = [0.001] * 60
    state = SignalAggregator().compute(
        asset="BTC",
        returns_1h=returns_1h,
        returns_1d=[0.001] * 10,
        returns_1w=[0.02] * 5,
        timestamp=datetime(2026, 5, 26, tzinfo=timezone.utc),
    )

    assert calls == [returns_1h]
    assert state.regime == "high_vol"
    assert state.confidence == 0.50
    assert state.pct_5d_return is not None
    assert state.direction_state == "NEUTRAL"
    assert state.dual_trigger is False  # "high_vol" is not "stress"


def test_dual_trigger_true_when_stress(monkeypatch):
    def fake_stress(log_returns):
        return {
            "regime": "stress",
            "confidence": 0.87,
            "recent_vol_window": 12,
            "recent_vol": 0.03,
            "overall_vol": 0.015,
            "vol_ratio": 2.0,
            "skewness": -1.6,
            "excess_kurtosis": 35.0,
            "mean_return": 0.0,
        }

    monkeypatch.setattr(signal_aggregator, "detect_market_regime", fake_stress)

    state = SignalAggregator().compute(
        asset="ETH",
        returns_1h=[0.001] * 60,
        returns_1d=[0.01] * 10,
        returns_1w=[0.02] * 5,
        timestamp=datetime(2026, 5, 26, tzinfo=timezone.utc),
    )

    assert state.regime == "stress"
    assert state.dual_trigger is True


def test_signal_aggregator_applies_direction_overlay(monkeypatch):
    def fake_stress(log_returns):
        return {
            "regime": "stress",
            "confidence": 0.90,
            "recent_vol_window": 12,
            "recent_vol": 0.03,
            "overall_vol": 0.015,
            "vol_ratio": 2.0,
            "skewness": -1.6,
            "excess_kurtosis": 35.0,
            "mean_return": 0.0,
        }

    monkeypatch.setattr(signal_aggregator, "detect_market_regime", fake_stress)

    state = SignalAggregator().compute(
        asset="ETH",
        returns_1h=[0.001] * 60,
        returns_1d=[-0.02] * 5 + [0.007] * 5,
        returns_1w=[0.02] * 5,
        timestamp=datetime(2026, 5, 26, tzinfo=timezone.utc),
    )

    assert state.direction_state == "RISING"
    assert state.confidence == 0.68


def test_dual_trigger_false_for_normal_and_low_vol(monkeypatch):
    for regime_label in ("normal", "low_vol"):

        def make_fake(label):
            def fake(log_returns):
                return {
                    "regime": label,
                    "confidence": 0.80,
                    "recent_vol_window": 12,
                    "recent_vol": 0.01,
                    "overall_vol": 0.012,
                    "vol_ratio": 0.9,
                    "skewness": 0.0,
                    "excess_kurtosis": 1.0,
                    "mean_return": 0.0,
                }

            return fake

        monkeypatch.setattr(signal_aggregator, "detect_market_regime", make_fake(regime_label))

        state = SignalAggregator().compute(
            asset="BTC",
            returns_1h=[0.001] * 60,
            returns_1d=[0.01] * 10,
            returns_1w=[0.02] * 5,
            timestamp=datetime(2026, 5, 26, tzinfo=timezone.utc),
        )

        assert state.dual_trigger is False, f"expected False for regime={regime_label!r}"


def test_missing_horizon_data_is_reported():
    state = SignalAggregator().compute(
        asset="BTC",
        returns_1h=[0.001],  # < 2 bars
        returns_1d=[0.01] * 10,
        returns_1w=[0.02] * 5,
    )
    assert "1H" in state.missing_layers
