"""
services/decision/report.py — Pure deterministic report generator.

Accepts all upstream computation results and produces a structured RiskReport
that is both machine-readable (as_dict()) and human-readable (as_text()).

No I/O, no randomness. Two calls with identical inputs produce identical output.
Suitable for replaying historical reports from stored inputs.

Scenario analysis
-----------------
Four fixed scenarios: -5%, -10%, -20%, -30%.
Uses put intrinsic value at each scenario spot (European payoff, no time value).
This is deliberately conservative — actual option value includes remaining time
value, so real protection is better. The conservative estimate is preferred for
risk management.

Formula (per scenario pct_move):
    spot_new          = spot × (1 + pct_move)
    intrinsic_per_coin = max(strike - spot_new, 0)
    option_payout     = intrinsic_per_coin × (notional × hedge_ratio / spot)
    portfolio_loss    = notional × |pct_move|
    net_loss          = portfolio_loss − option_payout + premium_cost_usd
    protection_ratio  = option_payout / portfolio_loss
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

from src.decision.engine import DecisionOutput, HedgeState
from src.decision.policy_engine import PolicyDecision
from src.decision.signal_aggregator import SignalState
from src.pricing.option_pricer import PricingResult

_SCENARIOS = [-0.05, -0.10, -0.20, -0.30]


# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    pct_move: float  # e.g. -0.10 = -10%
    portfolio_loss_usd: float  # notional × |pct_move|
    option_payout_usd: float  # put intrinsic × hedge_ratio × notional
    net_loss_usd: float  # portfolio_loss − option_payout + premium_cost
    protection_pct: float  # option_payout / portfolio_loss (0.0 if unhedged)


@dataclass
class RiskReport:
    # Metadata
    report_id: str
    generated_at: str  # ISO 8601
    asset: str
    notional: float

    # Trigger context
    trigger_fired: bool
    trigger_sources: list[str]
    what_changed: str

    # Volatility state
    regime: str
    confidence: float
    dual_trigger: bool
    vol_1h_annualized: Optional[float]
    vol_1h_zscore: Optional[float]
    vol_1d_zscore: Optional[float]
    vol_1w_annualized: Optional[float]
    last_1h_return: Optional[float]

    # Decision
    action: str  # HEDGE | WAIT | REVIEW
    rationale: str
    policy_action: str  # raw from policy_engine
    overridden: bool
    override_reason: Optional[str]

    # Hedge execution detail
    instrument: Optional[str]
    strike: Optional[float]
    expiry_date: Optional[str]
    days_to_expiry: Optional[float]
    delta: Optional[float]
    iv: Optional[float]
    premium_usd: Optional[float]
    premium_pct: Optional[float]
    cost_label: Optional[str]
    hedge_ratio: float
    total_hedge_cost_usd: Optional[float]  # premium_usd × hedge_ratio
    spread_available: bool

    # Protection estimates
    protection_at_10pct_usd: Optional[float]
    protection_at_10pct_ratio: Optional[float]

    # Max loss estimates (−30% scenario)
    max_portfolio_loss_usd: float  # notional × 0.30, no hedge
    max_net_loss_usd: float  # −30% with hedge applied
    max_net_loss_pct: float  # max_net_loss / notional

    # Deterministic scenarios
    scenarios: list[ScenarioResult]

    # Operational
    operational_impact: str

    # History context
    recent_actions: list[str]  # e.g. ["2026-05-19T12:00Z HEDGE (stress, 75%)"]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["scenarios"] = [asdict(s) for s in self.scenarios]
        return d

    def as_text(self) -> str:
        return _render_text(self)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def generate_report(
    asset: str,
    notional: float,
    signal: SignalState,
    policy: PolicyDecision,
    decision: DecisionOutput,
    pricing: Optional[PricingResult],
    hedge_state: HedgeState,
    recent_actions: list[str],
    report_id: Optional[str] = None,
    now: Optional[datetime] = None,
) -> RiskReport:
    """
    Build a RiskReport from upstream computation results.

    Pure function — no side effects, deterministic given the same inputs.
    """
    now = now or datetime.now(timezone.utc)
    report_id = report_id or f"{asset}_{now.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"

    scenarios = _compute_scenarios(pricing, hedge_state, notional)
    s10 = next((s for s in scenarios if s.pct_move == -0.10), None)
    s30 = next((s for s in scenarios if s.pct_move == -0.30), None)

    total_hedge_cost: Optional[float] = None
    if pricing is not None and decision.action == "HEDGE":
        ratio = policy.hedge_ratio if policy.hedge_ratio > 0 else 1.0
        total_hedge_cost = round(pricing.mark_price_usd * ratio, 2)

    return RiskReport(
        report_id=report_id,
        generated_at=now.isoformat(),
        asset=asset,
        notional=notional,
        trigger_fired=signal.dual_trigger,
        trigger_sources=_trigger_sources(signal),
        what_changed=_what_changed(signal),
        regime=signal.regime,
        confidence=signal.confidence,
        dual_trigger=signal.dual_trigger,
        vol_1h_annualized=_round(signal.layer_a.naive_vol_annualized, 4),
        vol_1h_zscore=_round(signal.layer_a.ewma_zscore, 4),
        vol_1d_zscore=_round(signal.layer_b.ewma_zscore, 4),
        vol_1w_annualized=_round(signal.layer_c.naive_vol_annualized, 4),
        last_1h_return=_round(signal.layer_a.last_return, 6),
        action=decision.action,
        rationale=decision.rationale,
        policy_action=decision.policy_action,
        overridden=decision.overridden,
        override_reason=decision.override_reason,
        instrument=pricing.instrument_name if pricing else None,
        strike=pricing.strike if pricing else None,
        expiry_date=pricing.expiry.date().isoformat() if pricing else None,
        days_to_expiry=_round(pricing.days_to_expiry, 2) if pricing else None,
        delta=_round(pricing.delta, 4) if pricing else None,
        iv=_round(pricing.iv, 4) if pricing else None,
        premium_usd=_round(pricing.mark_price_usd, 2) if pricing else None,
        premium_pct=_round(pricing.premium_pct, 6) if pricing else None,
        cost_label=pricing.cost_label if pricing else None,
        hedge_ratio=policy.hedge_ratio,
        total_hedge_cost_usd=total_hedge_cost,
        spread_available=policy.spread_available,
        protection_at_10pct_usd=s10.option_payout_usd if s10 else None,
        protection_at_10pct_ratio=s10.protection_pct if s10 else None,
        max_portfolio_loss_usd=round(notional * 0.30, 2),
        max_net_loss_usd=s30.net_loss_usd if s30 else round(notional * 0.30, 2),
        max_net_loss_pct=round((s30.net_loss_usd if s30 else notional * 0.30) / notional, 4),
        scenarios=scenarios,
        operational_impact=_operational_impact(decision, policy, pricing, notional),
        recent_actions=recent_actions,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trigger_sources(signal: SignalState) -> list[str]:
    sources = []
    a = signal.layer_a
    if a.ewma_threshold_hit and a.ewma_zscore is not None:
        sources.append(f"Layer A (1H) EWMA z={a.ewma_zscore:.2f}")
    if a.naive_threshold_hit and a.last_return is not None:
        sources.append(f"Layer A (1H) |return|={abs(a.last_return):.3f}")
    b = signal.layer_b
    if b.ewma_threshold_hit and b.ewma_zscore is not None:
        sources.append(f"Layer B (1D) EWMA z={b.ewma_zscore:.2f}")
    c = signal.layer_c
    if c.naive_threshold_hit:
        vol_str = f"{c.naive_vol_annualized:.1%}" if c.naive_vol_annualized else "elevated"
        sources.append(f"Layer C (1W) structural vol={vol_str}")
    return sources or ["no trigger"]


def _what_changed(signal: SignalState) -> list[str]:
    parts = [f"Regime {signal.regime} (confidence {signal.confidence:.0%})."]
    a = signal.layer_a
    if a.ewma_zscore is not None:
        parts.append(f"1H EWMA vol z-score {a.ewma_zscore:.2f} (threshold 2.0).")
    if a.last_return is not None:
        parts.append(
            f"Most recent 1H log return: {a.last_return:+.4f} ({abs(a.last_return) * 100:.2f}%)."
        )
    if a.naive_vol_annualized is not None:
        parts.append(f"1H realized vol: {a.naive_vol_annualized:.1%} annualized.")
    b = signal.layer_b
    if b.ewma_zscore is not None:
        status = "confirmed" if b.ewma_threshold_hit else "not yet confirmed"
        parts.append(f"1D EWMA z-score {b.ewma_zscore:.2f} ({status}).")
    c = signal.layer_c
    if c.naive_vol_annualized is not None:
        status = "structural stress confirmed" if c.naive_threshold_hit else "within normal range"
        parts.append(f"1W vol {c.naive_vol_annualized:.1%} ann. ({status}).")
    return " ".join(parts)


def _compute_scenarios(
    pricing: Optional[PricingResult],
    hedge_state: HedgeState,
    notional: float,
) -> list[ScenarioResult]:
    spot = pricing.spot if pricing else None
    strike = pricing.strike if pricing else None

    # Use current hedge state if active; fall back to recommendation
    if hedge_state.is_hedged and hedge_state.hedge_ratio > 0:
        hedge_ratio = hedge_state.hedge_ratio
        premium_cost = hedge_state.premium_paid_usd or 0.0
    elif pricing is not None:
        hedge_ratio = 0.0  # not yet hedged; scenario shows pre-hedge exposure
        premium_cost = 0.0
    else:
        hedge_ratio = 0.0
        premium_cost = 0.0

    results = []
    for pct in _SCENARIOS:
        portfolio_loss = notional * abs(pct)

        option_payout = 0.0
        if spot and strike and hedge_ratio > 0 and spot > 0:
            spot_new = spot * (1.0 + pct)
            intrinsic = max(strike - spot_new, 0.0)
            # coins covered = notional × hedge_ratio / spot
            option_payout = intrinsic * (notional * hedge_ratio / spot)

        net_loss = portfolio_loss - option_payout + premium_cost
        protection = option_payout / portfolio_loss if portfolio_loss > 0 else 0.0

        results.append(
            ScenarioResult(
                pct_move=pct,
                portfolio_loss_usd=round(portfolio_loss, 2),
                option_payout_usd=round(option_payout, 2),
                net_loss_usd=round(net_loss, 2),
                protection_pct=round(protection, 4),
            )
        )

    return results


def _operational_impact(
    decision: DecisionOutput,
    policy: PolicyDecision,
    pricing: Optional[PricingResult],
    notional: float,
) -> str:
    if decision.action == "HEDGE":
        ratio = policy.hedge_ratio if policy.hedge_ratio > 0 else 1.0
        if pricing:
            cost_usd = pricing.mark_price_usd * ratio
            cost_pct = pricing.premium_pct * ratio
            return (
                f"Execute hedge: purchase {policy.recommended_instrument or 'selected put'} "
                f"covering {ratio:.0%} of ${notional:,.0f} notional. "
                f"Estimated cost: {cost_pct * 100:.2f}% of notional (${cost_usd:,.2f} USD). "
                f"Pending manual approval via POST /decision/approve."
            )
        return (
            "Execute hedge — live pricing unavailable at trigger time. "
            "Obtain current pricing before executing. "
            "Pending manual approval via POST /decision/approve."
        )
    elif decision.action == "WAIT":
        return "No action. Monitor regime. Re-evaluate on next hourly cycle."
    else:
        return (
            "System has entered REVIEW state. No automated action taken. "
            "Manual investigation required. "
            "Clear REVIEW state via POST /decision/review/clear before resuming."
        )


def _round(val: Optional[float], ndigits: int) -> Optional[float]:
    if val is None:
        return None
    return round(val, ndigits)


# ---------------------------------------------------------------------------
# Human-readable text renderer
# ---------------------------------------------------------------------------


def _render_text(r: RiskReport) -> str:
    lines: list[str] = [
        "=" * 62,
        f"HEDGE DECISION REPORT  —  {r.asset}",
        f"Report ID  : {r.report_id}",
        f"Generated  : {r.generated_at}",
        f"Notional   : ${r.notional:,.0f} USD",
        "=" * 62,
        "",
        "TRIGGER",
        "-------",
        f"  Fired          : {r.trigger_fired}",
        f"  Sources        : {'; '.join(r.trigger_sources)}",
        f"  Summary        : {r.what_changed}",
        "",
        "VOLATILITY STATE",
        "----------------",
        f"  Regime         : {r.regime}",
        f"  Confidence     : {r.confidence:.0%}",
        f"  Dual trigger   : {r.dual_trigger}",
    ]
    if r.vol_1h_annualized is not None:
        lines.append(f"  1H vol (ann)   : {r.vol_1h_annualized:.1%}")
    if r.vol_1h_zscore is not None:
        lines.append(f"  1H EWMA z      : {r.vol_1h_zscore:.2f}")
    if r.vol_1d_zscore is not None:
        lines.append(f"  1D EWMA z      : {r.vol_1d_zscore:.2f}")
    if r.vol_1w_annualized is not None:
        lines.append(f"  1W vol (ann)   : {r.vol_1w_annualized:.1%}")
    if r.last_1h_return is not None:
        lines.append(
            f"  Last 1H return : {r.last_1h_return:+.4f} ({abs(r.last_1h_return) * 100:.2f}%)"
        )

    lines += [
        "",
        "DECISION",
        "--------",
        f"  Action         : {r.action}",
        f"  Policy action  : {r.policy_action}",
    ]
    if r.overridden:
        lines.append(f"  Override       : {r.override_reason}")
    lines.append(f"  Rationale      : {r.rationale}")

    if r.instrument:
        lines += [
            "",
            "HEDGE EXECUTION",
            "---------------",
            f"  Instrument     : {r.instrument}",
        ]
        if r.strike:
            lines.append(f"  Strike         : ${r.strike:,.0f}")
        if r.expiry_date and r.days_to_expiry:
            lines.append(f"  Expiry         : {r.expiry_date} ({r.days_to_expiry:.1f}d)")
        if r.delta:
            lines.append(f"  Delta          : {r.delta:.3f}")
        if r.iv:
            lines.append(f"  Implied vol    : {r.iv:.1%}")
        if r.premium_usd and r.premium_pct:
            lines.append(
                f"  Premium        : ${r.premium_usd:,.2f} ({r.premium_pct * 100:.2f}% of notional)"
            )
        if r.cost_label:
            lines.append(f"  Cost label     : {r.cost_label}")
        lines.append(f"  Hedge ratio    : {r.hedge_ratio:.0%}")
        if r.total_hedge_cost_usd:
            lines.append(f"  Total cost     : ${r.total_hedge_cost_usd:,.2f} USD")
        if r.spread_available:
            lines.append("  Spread alt     : available — see full API response")

    lines += [
        "",
        "SCENARIO ANALYSIS  (conservative: European intrinsic value only)",
        "------------------------------------------------------------------",
        f"  {'Move':>6}  {'Port. Loss':>12}  {'Option Payout':>14}  {'Net Loss':>12}  {'Protection':>11}",
    ]
    for s in r.scenarios:
        lines.append(
            f"  {s.pct_move * 100:>5.0f}%  "
            f"${s.portfolio_loss_usd:>11,.0f}  "
            f"${s.option_payout_usd:>13,.0f}  "
            f"${s.net_loss_usd:>11,.0f}  "
            f"{s.protection_pct:>10.1%}"
        )

    lines += [
        "",
        "MAX LOSS ESTIMATE  (−30% scenario)",
        "-----------------------------------",
        f"  Unhedged loss  : ${r.max_portfolio_loss_usd:,.0f}",
        f"  Hedged loss    : ${r.max_net_loss_usd:,.0f}  ({r.max_net_loss_pct:.1%} of notional)",
        "",
        "OPERATIONAL IMPACT",
        "------------------",
        f"  {r.operational_impact}",
    ]

    if r.recent_actions:
        lines += [
            "",
            "RECENT HISTORY",
            "--------------",
        ]
        for entry in r.recent_actions[-6:]:
            lines.append(f"  {entry}")

    lines += ["", "=" * 62]
    return "\n".join(lines)
