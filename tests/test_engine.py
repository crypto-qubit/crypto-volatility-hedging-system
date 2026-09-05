"""
Tests for the pure decision function (src/decision/engine.py).

Covers the priority-ordered rule chain: kill switch, sticky REVIEW, policy
pass-through, anti-whipsaw suppression, already-hedged suppression, and the
PARTIAL_HEDGE -> HEDGE vocabulary simplification.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.decision.engine import (
    DecisionRecord,
    HedgeState,
    ReviewState,
    decide,
)
from src.decision.policy_engine import PolicyDecision

NOW = datetime(2026, 5, 26, 12, 0, tzinfo=timezone.utc)


def _hedge_state(is_hedged=False, instrument=None, hedge_ratio=0.0) -> HedgeState:
    return HedgeState(
        is_hedged=is_hedged,
        instrument=instrument,
        strike=None,
        notional=None,
        premium_paid_usd=None,
        opened_at=None,
        hedge_ratio=hedge_ratio,
    )


def _review_state(active=False, reason="", activated_at=None) -> ReviewState:
    return ReviewState(active=active, reason=reason, activated_at=activated_at)


def _policy(action="HEDGE", regime="stress", hedge_ratio=1.0, instrument="BTC-P") -> PolicyDecision:
    return PolicyDecision(
        action=action,
        confidence=0.85,
        review_flag=(action == "REVIEW"),
        hedge_ratio=hedge_ratio,
        regime=regime,
        rationale=f"policy said {action}",
        recommended_instrument=instrument,
        recommended_strike=90000.0,
        premium_usd=1000.0,
        premium_pct=0.01,
        cost_label="cheap",
        spread_available=False,
        timestamp=NOW,
    )


def _decide(**overrides):
    defaults = dict(
        policy=_policy(),
        hedge_state=_hedge_state(),
        review_state=_review_state(),
        history=[],
        kill_switch_active=False,
        asset="BTC",
        now=NOW,
    )
    defaults.update(overrides)
    return decide(**defaults)


def test_kill_switch_forces_review_even_with_hedge_policy():
    out = _decide(kill_switch_active=True)
    assert out.action == "REVIEW"
    assert out.override_reason == "kill_switch"


def test_sticky_review_state_persists_regardless_of_new_policy():
    out = _decide(review_state=_review_state(active=True, reason="prior anomaly", activated_at=NOW))
    assert out.action == "REVIEW"
    assert out.override_reason == "review_state_active"


def test_policy_review_passes_through_without_override_flag():
    out = _decide(policy=_policy(action="REVIEW"))
    assert out.action == "REVIEW"
    assert out.overridden is False


def test_policy_wait_passes_through():
    out = _decide(policy=_policy(action="WAIT"))
    assert out.action == "WAIT"
    assert out.overridden is False


def test_partial_hedge_maps_to_hedge_vocabulary():
    out = _decide(policy=_policy(action="PARTIAL_HEDGE", hedge_ratio=0.35))
    assert out.action == "HEDGE"
    assert out.overridden is True
    assert out.override_reason == "partial_to_hedge"
    assert "35%" in out.rationale


def test_plain_hedge_passes_through_unmodified():
    out = _decide(policy=_policy(action="HEDGE"))
    assert out.action == "HEDGE"
    assert out.overridden is False


def test_anti_whipsaw_suppresses_rehedge_within_window_at_same_regime():
    recent_hedge = DecisionRecord(
        timestamp=NOW - timedelta(hours=2),
        action="HEDGE",
        policy_action="HEDGE",
        regime="stress",
        confidence=0.8,
        asset="BTC",
        rationale="earlier hedge",
    )
    out = _decide(policy=_policy(action="HEDGE", regime="stress"), history=[recent_hedge])
    assert out.action == "WAIT"
    assert out.override_reason == "anti_whipsaw"


def test_anti_whipsaw_does_not_suppress_after_window_elapses():
    old_hedge = DecisionRecord(
        timestamp=NOW - timedelta(hours=7),
        action="HEDGE",
        policy_action="HEDGE",
        regime="stress",
        confidence=0.8,
        asset="BTC",
        rationale="earlier hedge",
    )
    out = _decide(policy=_policy(action="HEDGE", regime="stress"), history=[old_hedge])
    assert out.action == "HEDGE"


def test_anti_whipsaw_allows_rehedge_when_regime_escalates():
    recent_hedge = DecisionRecord(
        timestamp=NOW - timedelta(hours=1),
        action="HEDGE",
        policy_action="HEDGE",
        regime="high_vol",
        confidence=0.8,
        asset="BTC",
        rationale="earlier hedge",
    )
    out = _decide(policy=_policy(action="HEDGE", regime="stress"), history=[recent_hedge])
    assert out.action == "HEDGE"


def test_already_fully_hedged_at_same_instrument_waits():
    out = _decide(
        policy=_policy(action="HEDGE", instrument="BTC-29MAY26-90000-P"),
        hedge_state=_hedge_state(
            is_hedged=True, instrument="BTC-29MAY26-90000-P", hedge_ratio=0.95
        ),
    )
    assert out.action == "WAIT"
    assert out.override_reason == "already_hedged"


def test_partially_hedged_below_90pct_still_allows_incremental_hedge():
    out = _decide(
        policy=_policy(action="HEDGE", instrument="BTC-29MAY26-90000-P"),
        hedge_state=_hedge_state(
            is_hedged=True, instrument="BTC-29MAY26-90000-P", hedge_ratio=0.40
        ),
    )
    assert out.action == "HEDGE"


@pytest.mark.parametrize("action", ["HEDGE", "WAIT", "REVIEW"])
def test_decision_is_deterministic_given_identical_inputs(action):
    policy = _policy(action=action)
    out1 = _decide(policy=policy)
    out2 = _decide(policy=policy)
    assert out1 == out2
