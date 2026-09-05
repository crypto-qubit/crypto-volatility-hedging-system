"""
Policy engine — maps signal state + pricing result to a hedge recommendation.

This is a pure function module: no state, no I/O. Takes the outputs of
SignalAggregator and find_target_put() and applies deterministic rules.
SignalAggregator delegates regime classification to
core.risk_checks.detect_market_regime.

Decision logic
--------------
REVIEW        : confidence < 0.40, or missing horizon data, or pricing unavailable
WAIT          : no dual trigger at Layer A (1H); regime is not stress; or
                available 5-day direction is not FALLING
HEDGE         : dual trigger + stress regime + FALLING direction + premium within budget
PARTIAL_HEDGE : dual trigger + stress regime + FALLING direction + premium exceeds budget
                → hedge ratio scaled to fit budget; spread alternative shown if available

Hedge ratio for PARTIAL_HEDGE
------------------------------
    ratio = min(budget_pct / premium_pct, 0.50)

The 0.50 cap prevents recommending a trivially small hedge (e.g. 5% of notional)
when the option is very expensive. At that point the action degrades to REVIEW.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from src.decision.signal_aggregator import EWMA_ZSCORE_THRESHOLD_A, SignalState
from src.pricing.option_pricer import PricingResult

_CONFIDENCE_REVIEW_FLOOR = 0.40  # below this → always REVIEW
_MIN_PARTIAL_RATIO = 0.10  # below this hedge ratio → degrade to REVIEW
_MAX_PARTIAL_RATIO = 0.50  # cap for partial hedge


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass
class PolicyDecision:
    action: str  # HEDGE | PARTIAL_HEDGE | WAIT | REVIEW
    confidence: float  # [0, 1]
    review_flag: bool
    hedge_ratio: float  # fraction of notional to hedge [0, 1]
    regime: str
    rationale: str
    recommended_instrument: str | None
    recommended_strike: float | None
    premium_usd: float | None
    premium_pct: float | None
    cost_label: str | None
    spread_available: bool
    timestamp: datetime


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def evaluate(
    signal: SignalState,
    pricing: PricingResult | None,
    budget_pct: float = 0.015,
) -> PolicyDecision:
    """
    Apply policy rules to produce a hedge recommendation.

    Parameters
    ----------
    signal : SignalState
        Output of SignalAggregator.compute().
    pricing : PricingResult | None
        Output of find_target_put(), or None if the feed is unavailable.
    budget_pct : float
        Maximum acceptable premium as a fraction of notional. Default: 1.5%.

    Returns
    -------
    PolicyDecision
    """
    ts = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # 1. Review gate — overrides all other logic
    # ------------------------------------------------------------------
    review_reasons: list[str] = []

    if signal.confidence < _CONFIDENCE_REVIEW_FLOOR:
        review_reasons.append(f"low confidence ({signal.confidence:.2f})")
    if signal.missing_layers:
        review_reasons.append(f"missing horizon data: {', '.join(signal.missing_layers)}")
    if pricing is None and signal.dual_trigger:
        # Pricing missing but signal is live → cannot size without cost
        review_reasons.append("pricing unavailable")

    if review_reasons:
        return PolicyDecision(
            action="REVIEW",
            confidence=signal.confidence,
            review_flag=True,
            hedge_ratio=0.0,
            regime=signal.regime,
            rationale="Manual review required — " + "; ".join(review_reasons) + ".",
            recommended_instrument=None,
            recommended_strike=None,
            premium_usd=None,
            premium_pct=None,
            cost_label=None,
            spread_available=False,
            timestamp=ts,
        )

    # ------------------------------------------------------------------
    # 2. No dual trigger → wait
    # ------------------------------------------------------------------
    if not signal.dual_trigger:
        return PolicyDecision(
            action="WAIT",
            confidence=signal.confidence,
            review_flag=False,
            hedge_ratio=0.0,
            regime=signal.regime,
            rationale=_wait_rationale(signal),
            recommended_instrument=None,
            recommended_strike=None,
            premium_usd=None,
            premium_pct=None,
            cost_label=None,
            spread_available=False,
            timestamp=ts,
        )

    # ------------------------------------------------------------------
    # 3. Dual trigger active — evaluate detect_market_regime result + cost
    # ------------------------------------------------------------------

    if signal.regime != "stress":
        return PolicyDecision(
            action="WAIT",
            confidence=signal.confidence,
            review_flag=False,
            hedge_ratio=0.0,
            regime=signal.regime,
            rationale=(
                f"detect_market_regime returned {signal.regime!r} — not stress. Monitoring."
            ),
            recommended_instrument=pricing.instrument_name if pricing else None,
            recommended_strike=pricing.strike if pricing else None,
            premium_usd=pricing.mark_price_usd if pricing else None,
            premium_pct=pricing.premium_pct if pricing else None,
            cost_label=pricing.cost_label if pricing else None,
            spread_available=(pricing.spread_recommendation is not None) if pricing else False,
            timestamp=ts,
        )

    if signal.direction_state is not None and signal.direction_state != "FALLING":
        return PolicyDecision(
            action="WAIT",
            confidence=signal.confidence,
            review_flag=False,
            hedge_ratio=0.0,
            regime=signal.regime,
            rationale=(
                f"Stress regime detected, but 5-day direction is "
                f"{signal.direction_state}. Waiting for confirmed downside momentum."
            ),
            recommended_instrument=pricing.instrument_name if pricing else None,
            recommended_strike=pricing.strike if pricing else None,
            premium_usd=pricing.mark_price_usd if pricing else None,
            premium_pct=pricing.premium_pct if pricing else None,
            cost_label=pricing.cost_label if pricing else None,
            spread_available=(pricing.spread_recommendation is not None) if pricing else False,
            timestamp=ts,
        )

    # stress — act
    assert pricing is not None
    over_budget = pricing.premium_pct > budget_pct
    expensive = pricing.cost_label == "expensive"

    if not (over_budget or expensive):
        action = "HEDGE"
        hedge_ratio = 1.0
    else:
        ratio = _partial_ratio(pricing.premium_pct, budget_pct)
        if ratio < _MIN_PARTIAL_RATIO:
            # Option is so expensive that even a partial hedge is impractical
            return PolicyDecision(
                action="REVIEW",
                confidence=signal.confidence,
                review_flag=True,
                hedge_ratio=0.0,
                regime=signal.regime,
                rationale=(
                    f"Premium {pricing.premium_pct * 100:.2f}% far exceeds budget "
                    f"{budget_pct * 100:.2f}%. Feasible hedge ratio {ratio:.0%} too small. "
                    "Manual review required."
                ),
                recommended_instrument=pricing.instrument_name,
                recommended_strike=pricing.strike,
                premium_usd=pricing.mark_price_usd,
                premium_pct=pricing.premium_pct,
                cost_label=pricing.cost_label,
                spread_available=pricing.spread_recommendation is not None,
                timestamp=ts,
            )
        action = "PARTIAL_HEDGE"
        hedge_ratio = ratio

    return PolicyDecision(
        action=action,
        confidence=signal.confidence,
        review_flag=False,
        hedge_ratio=hedge_ratio,
        regime=signal.regime,
        rationale=_action_rationale(signal, pricing, action, hedge_ratio, budget_pct),
        recommended_instrument=pricing.instrument_name,
        recommended_strike=pricing.strike,
        premium_usd=pricing.mark_price_usd,
        premium_pct=pricing.premium_pct,
        cost_label=pricing.cost_label,
        spread_available=pricing.spread_recommendation is not None,
        timestamp=ts,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _partial_ratio(premium_pct: float, budget_pct: float) -> float:
    if premium_pct <= 0:
        return _MAX_PARTIAL_RATIO
    return round(min(budget_pct / premium_pct, _MAX_PARTIAL_RATIO), 2)


def _wait_rationale(signal: SignalState) -> str:
    a = signal.layer_a
    parts = ["No dual trigger."]
    if a.ewma_zscore is not None:
        parts.append(f"EWMA z-score {a.ewma_zscore:.2f} (threshold {EWMA_ZSCORE_THRESHOLD_A}).")
    if a.last_return is not None:
        parts.append(f"|return| {abs(a.last_return):.3f}.")
    parts.append(f"Regime {signal.regime}.")
    return " ".join(parts)


def _action_rationale(
    signal: SignalState,
    pricing: PricingResult,
    action: str,
    hedge_ratio: float,
    budget_pct: float,
) -> str:
    a, b, c = signal.layer_a, signal.layer_b, signal.layer_c
    parts = [f"Regime {signal.regime}."]
    if signal.direction_state is not None:
        parts.append(f"5-day direction {signal.direction_state}.")

    parts.append(
        f"Layer A (1H): EWMA z-score {a.ewma_zscore:.2f}, |return| {abs(a.last_return or 0):.3f}."
    )
    if b.ewma_threshold_hit:
        parts.append(f"Layer B (1D) confirmed: z-score {b.ewma_zscore:.2f}.")
    if c.naive_threshold_hit:
        parts.append("Layer C (1W) structural stress confirmed.")

    parts.append(
        f"Premium {pricing.premium_pct * 100:.2f}% vs budget {budget_pct * 100:.2f}% "
        f"({pricing.cost_label})."
    )

    if action == "PARTIAL_HEDGE":
        parts.append(f"Premium exceeds budget — partial hedge at {hedge_ratio:.0%} of notional.")
    if pricing.spread_recommendation:
        s = pricing.spread_recommendation
        parts.append(
            f"Put spread alternative: buy {s.long_leg.instrument_name}, "
            f"sell {s.short_leg.instrument_name}, "
            f"net cost {s.net_premium_pct * 100:.2f}%, "
            f"max payoff {s.max_payoff_pct * 100:.1f}%."
        )

    return " ".join(parts)
