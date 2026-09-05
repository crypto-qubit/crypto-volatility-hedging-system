"""
Tests for hedge policy decisions.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

from src.decision.policy_engine import evaluate
from src.decision.signal_aggregator import HorizonSignal, SignalState


def _signal(regime: str, direction_state: str | None = "FALLING") -> SignalState:
    return SignalState(
        asset="BTC",
        timestamp=datetime(2026, 5, 26, tzinfo=timezone.utc),
        layer_a=HorizonSignal("1H", 2.5, 1.5, True, True, -0.02, 60),
        layer_b=HorizonSignal("1D", 1.6, 1.2, False, True, -0.04, 10),
        layer_c=HorizonSignal("1W", None, 0.8, False, False, -0.03, 5),
        dual_trigger=True,
        regime=regime,
        confidence=0.80,
        pct_5d_return=-4.0 if direction_state == "FALLING" else 4.0,
        direction_state=direction_state,
        missing_layers=[],
    )


def _pricing():
    return SimpleNamespace(
        instrument_name="BTC-29MAY26-90000-P",
        strike=90000.0,
        mark_price_usd=1000.0,
        premium_pct=0.01,
        cost_label="cheap",
        spread_recommendation=None,
    )


def test_dual_trigger_high_vol_waits_instead_of_hedging():
    decision = evaluate(_signal("high_vol"), _pricing())

    assert decision.action == "WAIT"
    assert decision.regime == "high_vol"


def test_dual_trigger_stress_can_hedge():
    decision = evaluate(_signal("stress"), _pricing())

    assert decision.action == "HEDGE"
    assert decision.regime == "stress"


def test_dual_trigger_stress_rising_waits():
    decision = evaluate(_signal("stress", "RISING"), _pricing())

    assert decision.action == "WAIT"
    assert decision.hedge_ratio == 0.0


def test_dual_trigger_stress_without_direction_keeps_legacy_hedge_path():
    decision = evaluate(_signal("stress", None), _pricing())

    assert decision.action == "HEDGE"
    assert decision.regime == "stress"


def test_low_confidence_forces_review():
    signal = _signal("stress")
    signal = SignalState(**{**signal.__dict__, "confidence": 0.20})

    decision = evaluate(signal, _pricing())

    assert decision.action == "REVIEW"
    assert decision.review_flag is True


def test_missing_pricing_on_dual_trigger_forces_review():
    decision = evaluate(_signal("stress"), pricing=None)

    assert decision.action == "REVIEW"


def test_no_dual_trigger_always_waits():
    signal = _signal("stress")
    signal = SignalState(**{**signal.__dict__, "dual_trigger": False})

    decision = evaluate(signal, _pricing())

    assert decision.action == "WAIT"


def test_expensive_premium_yields_partial_hedge():
    expensive = SimpleNamespace(
        instrument_name="BTC-29MAY26-90000-P",
        strike=90000.0,
        mark_price_usd=4000.0,
        premium_pct=0.04,  # 4% > 1.5% budget
        cost_label="expensive",
        spread_recommendation=None,
    )
    decision = evaluate(_signal("stress"), expensive, budget_pct=0.015)

    assert decision.action == "PARTIAL_HEDGE"
    assert 0.0 < decision.hedge_ratio < 1.0


def test_extremely_expensive_premium_degrades_to_review():
    very_expensive = SimpleNamespace(
        instrument_name="BTC-29MAY26-90000-P",
        strike=90000.0,
        mark_price_usd=50000.0,
        premium_pct=0.50,  # feasible ratio would be far below the 10% floor
        cost_label="expensive",
        spread_recommendation=None,
    )
    decision = evaluate(_signal("stress"), very_expensive, budget_pct=0.015)

    assert decision.action == "REVIEW"
