"""
services/decision/engine.py — Pure deterministic decision function.

Maps upstream signal + policy + operational state → HEDGE | WAIT | REVIEW.

No I/O, no randomness, no side effects. All state is passed in explicitly
so every decision is replayable from stored inputs.

Decision rules (evaluated in priority order):
  1. Kill switch active            → REVIEW
  2. Sticky REVIEW state active    → REVIEW
  3. Policy emitted REVIEW         → REVIEW (runner will activate sticky state)
  4. Policy emitted WAIT           → WAIT
  5. Anti-whipsaw: HEDGE within window, regime not escalated → WAIT
  6. Already fully hedged at same instrument → WAIT
  7. PARTIAL_HEDGE from policy     → mapped to HEDGE (simplified output vocabulary)
  8. HEDGE                         → HEDGE

Anti-whipsaw window: 6 hours.
Re-hedge IS allowed if:
  - Regime has escalated since last HEDGE
  - 6h window has elapsed
  - Position has changed (different instrument)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.decision.policy_engine import PolicyDecision

_ANTI_WHIPSAW_HOURS = 6

_REGIME_ORDER = {"low_vol": 0, "normal": 1, "high_vol": 2, "stress": 3}


# ---------------------------------------------------------------------------
# State data classes (inputs to the decision function)
# ---------------------------------------------------------------------------


@dataclass
class HedgeState:
    """Current open hedge position. Loaded from DecisionStore."""

    is_hedged: bool
    instrument: Optional[str]
    strike: Optional[float]
    notional: Optional[float]
    premium_paid_usd: Optional[float]
    opened_at: Optional[datetime]
    hedge_ratio: float  # 0.0 when not hedged


@dataclass
class ReviewState:
    """
    Sticky REVIEW flag. Once activated it persists until a human clears it
    via the operational REST endpoint. No automatic recovery.
    """

    active: bool
    reason: str
    activated_at: Optional[datetime]


@dataclass
class DecisionRecord:
    """One entry in the recommendation history."""

    timestamp: datetime
    action: str  # HEDGE | WAIT | REVIEW
    policy_action: str  # raw from policy_engine (may include PARTIAL_HEDGE)
    regime: str
    confidence: float
    asset: str
    rationale: str


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass
class DecisionOutput:
    """Result of the pure decision function."""

    action: str  # HEDGE | WAIT | REVIEW
    rationale: str
    overridden: bool  # True if policy action was modified
    override_reason: Optional[str]
    policy_action: str  # original from policy_engine


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def decide(
    policy: PolicyDecision,
    hedge_state: HedgeState,
    review_state: ReviewState,
    history: list[DecisionRecord],
    kill_switch_active: bool,
    asset: str,
    now: Optional[datetime] = None,
) -> DecisionOutput:
    """
    Pure deterministic decision function.

    Never reads from or writes to disk. Call once per trigger cycle.
    The same inputs always produce the same output — suitable for audit replay.
    """
    now = now or datetime.now(timezone.utc)

    # 1. Kill switch — highest priority override
    if kill_switch_active:
        return DecisionOutput(
            action="REVIEW",
            rationale="Kill switch is active. All hedge decisions suppressed until manually deactivated.",
            overridden=True,
            override_reason="kill_switch",
            policy_action=policy.action,
        )

    # 2. Sticky REVIEW state — system must be manually cleared before resuming
    if review_state.active:
        return DecisionOutput(
            action="REVIEW",
            rationale=(
                f"System is in REVIEW state (activated {_fmt_dt(review_state.activated_at)}). "
                f"Reason: {review_state.reason}. "
                "Manual clearance required via POST /decision/review/clear before resuming."
            ),
            overridden=True,
            override_reason="review_state_active",
            policy_action=policy.action,
        )

    # 3. Policy issued REVIEW — pass through (runner will activate sticky state)
    if policy.action == "REVIEW":
        return DecisionOutput(
            action="REVIEW",
            rationale=policy.rationale,
            overridden=False,
            override_reason=None,
            policy_action=policy.action,
        )

    # 4. Policy issued WAIT (no dual trigger, or regime not stress)
    if policy.action == "WAIT":
        return DecisionOutput(
            action="WAIT",
            rationale=policy.rationale,
            overridden=False,
            override_reason=None,
            policy_action=policy.action,
        )

    # From here: policy is HEDGE or PARTIAL_HEDGE (stress regime confirmed).

    # 5. Anti-whipsaw: suppress re-hedge if recent HEDGE at same or higher regime
    recent_hedge = _last_hedge(history)
    if recent_hedge is not None:
        elapsed = now - recent_hedge.timestamp
        if elapsed < timedelta(hours=_ANTI_WHIPSAW_HOURS):
            if not _regime_escalated(recent_hedge.regime, policy.regime):
                return DecisionOutput(
                    action="WAIT",
                    rationale=(
                        f"Anti-whipsaw: HEDGE issued {_fmt_elapsed(elapsed)} ago at regime "
                        f"{recent_hedge.regime}. Current regime {policy.regime} has not "
                        f"escalated. Suppressing re-hedge for {_ANTI_WHIPSAW_HOURS}h window."
                    ),
                    overridden=True,
                    override_reason="anti_whipsaw",
                    policy_action=policy.action,
                )

    # 6. Already fully hedged at the same instrument — no incremental action needed
    if (
        hedge_state.is_hedged
        and hedge_state.instrument is not None
        and hedge_state.instrument == policy.recommended_instrument
        and hedge_state.hedge_ratio >= 0.90
    ):
        return DecisionOutput(
            action="WAIT",
            rationale=(
                f"Already fully hedged via {hedge_state.instrument} "
                f"(ratio={hedge_state.hedge_ratio:.0%}, "
                f"opened {_fmt_dt(hedge_state.opened_at)}). "
                "No incremental action required."
            ),
            overridden=True,
            override_reason="already_hedged",
            policy_action=policy.action,
        )

    # 7. PARTIAL_HEDGE → map to HEDGE (simplified output vocabulary per spec)
    if policy.action == "PARTIAL_HEDGE":
        ratio_note = f" Hedge ratio: {policy.hedge_ratio:.0%} of notional."
        return DecisionOutput(
            action="HEDGE",
            rationale=policy.rationale + ratio_note,
            overridden=True,
            override_reason="partial_to_hedge",
            policy_action=policy.action,
        )

    # 8. HEDGE
    return DecisionOutput(
        action="HEDGE",
        rationale=policy.rationale,
        overridden=False,
        override_reason=None,
        policy_action=policy.action,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _last_hedge(history: list[DecisionRecord]) -> Optional[DecisionRecord]:
    hedges = [r for r in history if r.action == "HEDGE"]
    return max(hedges, key=lambda r: r.timestamp) if hedges else None


def _regime_escalated(previous: str, current: str) -> bool:
    """True if current regime is strictly higher than previous."""
    return _REGIME_ORDER.get(current, 0) > _REGIME_ORDER.get(previous, 0)


def _fmt_dt(dt: Optional[datetime]) -> str:
    if dt is None:
        return "unknown time"
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def _fmt_elapsed(td: timedelta) -> str:
    total_minutes = int(td.total_seconds() / 60)
    if total_minutes < 60:
        return f"{total_minutes}m"
    hours, mins = divmod(total_minutes, 60)
    return f"{hours}h {mins}m" if mins else f"{hours}h"
