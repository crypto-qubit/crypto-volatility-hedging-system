"""
Tests for directional hedge signal boundary helpers.
"""

import pytest

from src.decision.directional_signal import (
    build_directional_hedge_signal,
    classify_direction,
    pct_return_from_log_returns,
)


def test_classify_direction_uses_dead_zone():
    assert classify_direction(-2.5) == "FALLING"
    assert classify_direction(0.0) == "NEUTRAL"
    assert classify_direction(2.5) == "RISING"


def test_build_directional_hedge_signal_combines_confidence():
    signal = build_directional_hedge_signal("stress", -3.5, 0.90)

    assert signal.volatility_regime == "stress"
    assert signal.direction_state == "FALLING"
    assert signal.confidence == 0.68


def test_pct_return_from_log_returns_requires_window():
    assert pct_return_from_log_returns([0.01] * 4) is None
    assert pct_return_from_log_returns([0.01] * 5) == pytest.approx(5.1271096)


def test_build_directional_hedge_signal_rejects_unknown_regime():
    with pytest.raises(ValueError):
        build_directional_hedge_signal("chaotic", -3.5, 0.9)
