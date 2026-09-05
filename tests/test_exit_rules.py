"""
Tests for src/decision/exit_rules.py — the spike-exit state machine.

Each test exercises one branch of compute_spike_exit's six-rule evaluation
order (see module docstring). Adapted from the source project's research
prototype; ported here as a genuinely non-duplicative addition — this repo
had no exit-side logic for an open hedge before this module.
"""

from __future__ import annotations

import pytest

from src.decision.exit_rules import (
    OptionSnapshot,
    SpikeExitConfig,
    SpikeHedgeState,
    ValuationHistory,
    advance_hedge_state,
    compute_spike_exit,
)


def _state(entry: float = 1_000.0, peak: float = 1_000.0, days: int = 1) -> SpikeHedgeState:
    return SpikeHedgeState(entry_value=entry, peak_value=peak, days_held=days)


def _snap(value: float = 1_000.0, delta: float = -0.50, intrinsic: float = 0.0) -> OptionSnapshot:
    return OptionSnapshot(current_value=value, delta=delta, intrinsic_value=intrinsic)


def _decide(
    entry: float = 1_000.0,
    peak: float = 1_000.0,
    days: int = 1,
    value: float = 1_000.0,
    delta: float = -0.50,
    intrinsic: float = 0.0,
    config: SpikeExitConfig | None = None,
):
    return compute_spike_exit(
        hedge_state=_state(entry=entry, peak=peak, days=days),
        option_snapshot=_snap(value=value, delta=delta, intrinsic=intrinsic),
        valuation_history=ValuationHistory(values=[value]),
        config=config,
    )


# ---------------------------------------------------------------------------
# Rule 2 — Minimum holding period
# ---------------------------------------------------------------------------


def test_min_holding_blocks_all_rules_on_day_zero():
    config = SpikeExitConfig(min_holding_days=2)
    decision = _decide(entry=1_000.0, value=3_500.0, days=1, config=config)
    assert not decision.should_exit
    assert decision.reason == "MIN_HOLDING_PERIOD"


def test_min_holding_one_day_passes_on_day_one():
    decision = _decide(entry=1_000.0, value=800.0, days=1)
    assert decision.reason != "MIN_HOLDING_PERIOD"


# ---------------------------------------------------------------------------
# Rule 3 — 72-hour explosive tiers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected_reason,expected_fraction",
    [
        (1_600.0, "SPIKE_1_5X", 0.30),
        (2_100.0, "SPIKE_2X", 0.50),
        (2_400.0, "SPIKE_2_3X", 0.75),
        (3_100.0, "SPIKE_3X", 1.00),
    ],
)
def test_spike_tiers_within_72h(value, expected_reason, expected_fraction):
    decision = _decide(entry=1_000.0, value=value, days=2)
    assert decision.should_exit
    assert decision.reason == expected_reason
    assert decision.sell_fraction == pytest.approx(expected_fraction)


def test_spike_tiers_do_not_fire_after_day_3():
    """A 3x multiplier on day 4 must not trigger a Rule 3 tier exit."""
    decision = _decide(entry=1_000.0, value=3_100.0, days=4)
    assert decision.reason not in {"SPIKE_3X", "SPIKE_2X", "SPIKE_1_5X", "SPIKE_2_3X"}


# ---------------------------------------------------------------------------
# Rule 4 — Post-spike trailing stop
# ---------------------------------------------------------------------------


def test_trailing_stop_fires_after_deep_pullback_from_spike_peak():
    # Peak reached 2000 (2x entry), now down 30% from peak -> exceeds 25% stop.
    decision = _decide(entry=1_000.0, peak=2_000.0, value=1_400.0, days=5)
    assert decision.should_exit
    assert decision.reason == "TRAILING_STOP_POST_SPIKE"
    assert decision.sell_fraction == pytest.approx(1.0)


def test_trailing_stop_does_not_fire_without_prior_spike():
    """No trailing stop if the position never exceeded 1.5x entry."""
    decision = _decide(entry=1_000.0, peak=1_100.0, value=900.0, days=5)
    assert decision.reason != "TRAILING_STOP_POST_SPIKE"


def test_trailing_stop_does_not_fire_within_tolerance():
    # Peak 2000, only 10% down from peak -> below the 25% trailing-stop trigger.
    decision = _decide(entry=1_000.0, peak=2_000.0, value=1_800.0, days=5)
    assert decision.reason != "TRAILING_STOP_POST_SPIKE"


# ---------------------------------------------------------------------------
# Rule 5 — Convexity spent
# ---------------------------------------------------------------------------


def test_convexity_spent_on_deep_itm_delta():
    decision = _decide(entry=1_000.0, peak=1_000.0, value=1_100.0, days=2, delta=-0.95)
    assert decision.should_exit
    assert decision.reason == "CONVEXITY_SPENT"
    assert decision.sell_fraction == pytest.approx(0.50)


def test_convexity_spent_on_high_intrinsic_ratio():
    decision = _decide(entry=1_000.0, peak=1_000.0, value=1_100.0, days=2, intrinsic=950.0)
    assert decision.should_exit
    assert decision.reason == "CONVEXITY_SPENT"


def test_convexity_spent_does_not_fire_on_day_zero_even_past_min_holding():
    """Rule 5 requires days_held >= 1; a 0-day-held snapshot never reaches it
    anyway because Rule 2's default min_holding_days=1 blocks first."""
    decision = _decide(entry=1_000.0, peak=1_000.0, value=1_100.0, days=0, delta=-0.99)
    assert decision.reason == "MIN_HOLDING_PERIOD"


# ---------------------------------------------------------------------------
# Rule 6 — Hard time cap
# ---------------------------------------------------------------------------


def test_time_cap_exits_when_no_new_highs():
    decision = _decide(entry=1_000.0, peak=1_000.0, value=400.0, days=10)
    assert decision.should_exit
    assert decision.reason == "TIME_CAP_NO_NEW_HIGHS"
    assert decision.sell_fraction == pytest.approx(1.0)


def test_time_cap_does_not_fire_if_near_peak():
    decision = _decide(entry=1_000.0, peak=1_000.0, value=900.0, days=10)
    assert decision.reason != "TIME_CAP_NO_NEW_HIGHS"


def test_time_cap_does_not_fire_before_max_holding_days():
    decision = _decide(entry=1_000.0, peak=1_000.0, value=400.0, days=9)
    assert decision.reason != "TIME_CAP_NO_NEW_HIGHS"


# ---------------------------------------------------------------------------
# Default hold
# ---------------------------------------------------------------------------


def test_default_hold_when_no_rule_fires():
    decision = _decide(entry=1_000.0, peak=1_000.0, value=1_050.0, days=5)
    assert not decision.should_exit
    assert decision.sell_fraction == 0.0
    assert decision.reason is None


# ---------------------------------------------------------------------------
# State advancement
# ---------------------------------------------------------------------------


def test_advance_hedge_state_increments_days_and_tracks_peak():
    state = SpikeHedgeState(entry_value=1_000.0, peak_value=1_000.0, days_held=2)
    snap = OptionSnapshot(current_value=1_500.0, delta=-0.5, intrinsic_value=0.0)

    advanced = advance_hedge_state(state, snap)

    assert advanced.days_held == 3
    assert advanced.peak_value == 1_500.0
    assert advanced.entry_value == 1_000.0


def test_advance_hedge_state_peak_never_decreases():
    state = SpikeHedgeState(entry_value=1_000.0, peak_value=1_800.0, days_held=4)
    snap = OptionSnapshot(current_value=1_200.0, delta=-0.5, intrinsic_value=0.0)

    advanced = advance_hedge_state(state, snap)

    assert advanced.peak_value == 1_800.0  # unchanged — new value is lower than peak
