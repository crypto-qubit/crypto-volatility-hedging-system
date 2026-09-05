"""
src/decision/runner.py — Operational wrapper for the decision layer.

Wraps the pure functions in engine.py, policy_engine.py, and report.py with
the side effects needed to run as a stateful service: state load/save,
sticky REVIEW activation, and pending-hedge tracking.

This is the offline counterpart of the production runner: the production
version also emits a Telegram event and writes the report to BigQuery. Those
two steps are intentionally omitted here so this module has zero network
or cloud dependency — see docs/architecture.md for the full production flow.

Notional / budget config
-------------------------
HEDGE_NOTIONAL_USD env var (default 100 000 USD — a round, clearly synthetic
    demo notional, not a real position size)
HEDGE_BUDGET_PCT   env var (default 1.5%)
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from src.decision.engine import DecisionOutput, decide
from src.decision.policy_engine import evaluate
from src.decision.report import RiskReport, generate_report
from src.decision.signal_aggregator import SignalState
from src.decision.store import DecisionStore
from src.pricing.option_pricer import PricingResult

_log = logging.getLogger(__name__)

_DEFAULT_NOTIONAL = float(os.getenv("HEDGE_NOTIONAL_USD", "100000"))
_DEFAULT_BUDGET_PCT = float(os.getenv("HEDGE_BUDGET_PCT", "0.015"))


def run_decision(
    asset: str,
    signal: SignalState,
    pricing: Optional[PricingResult] = None,
    notional: float = _DEFAULT_NOTIONAL,
    budget_pct: float = _DEFAULT_BUDGET_PCT,
    now: Optional[datetime] = None,
    store: Optional[DecisionStore] = None,
) -> tuple[DecisionOutput, RiskReport]:
    """
    Run one full decision + reporting cycle.

    Steps:
      1. Load persisted state (history, hedge state, review state, kill switch)
      2. Evaluate policy_engine  (signal + pricing -> recommendation)
      3. Apply decision rules    (kill switch, anti-whipsaw, review state, etc.)
      4. Generate risk report    (deterministic from inputs)
      5. Persist decision record
      6. Activate sticky REVIEW state if decision is REVIEW
      7. Track pending hedge for approval flow
      8. Save state

    Returns (DecisionOutput, RiskReport). Never raises.
    """
    now = now or datetime.now(timezone.utc)
    report_id = f"{asset}_{now.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"

    store = store or DecisionStore()

    # 1. Load state
    kill_switch = store.kill_switch
    hedge_state = store.load_hedge_state()
    review_state = store.load_review_state()
    history = store.load_history()
    recent_actions = store.recent_actions_strings()

    # 2. Policy evaluation
    policy = evaluate(signal, pricing, budget_pct=budget_pct)

    # 3. Decision
    decision = decide(
        policy=policy,
        hedge_state=hedge_state,
        review_state=review_state,
        history=history,
        kill_switch_active=kill_switch,
        asset=asset,
        now=now,
    )

    # 4. Report
    report = generate_report(
        asset=asset,
        notional=notional,
        signal=signal,
        policy=policy,
        decision=decision,
        pricing=pricing,
        hedge_state=hedge_state,
        recent_actions=recent_actions,
        report_id=report_id,
        now=now,
    )

    _log.info(
        "[decision] %s | action=%s | regime=%s | confidence=%.2f | policy=%s%s",
        asset,
        decision.action,
        signal.regime,
        signal.confidence,
        policy.action,
        f" [override: {decision.override_reason}]" if decision.overridden else "",
    )

    # 5. Persist decision record
    store.append_decision(
        decision=decision,
        regime=signal.regime,
        confidence=signal.confidence,
        asset=asset,
        report_id=report_id,
        now=now,
    )

    # 6. Activate sticky REVIEW state if decision is REVIEW
    if decision.action == "REVIEW":
        store.activate_review_state(reason=decision.rationale[:300], now=now)

    # 7. Track pending hedge for approval flow
    if decision.action == "HEDGE":
        store.set_pending_hedge(
            report_id=report_id,
            asset=asset,
            details={
                "instrument": report.instrument,
                "strike": report.strike,
                "expiry_date": report.expiry_date,
                "premium_usd": report.premium_usd,
                "premium_pct": report.premium_pct,
                "hedge_ratio": report.hedge_ratio,
                "total_hedge_cost_usd": report.total_hedge_cost_usd,
                "regime": signal.regime,
                "confidence": signal.confidence,
            },
            now=now,
        )

    # 8. Save
    store.save()

    return decision, report
