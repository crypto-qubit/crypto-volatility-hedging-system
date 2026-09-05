"""
src/api/main.py — FastAPI interface for the decision layer.

Offline by default: /demo/run executes the same pipeline function as
`python -m src.demo` (src.demo.run_symbol_pipeline) against local sample
data (no BigQuery, no live options feed, no network access).

State scope differs deliberately from the CLI, though: /demo/run reads and
writes the same PERSISTENT per-asset DecisionStore as the operational
endpoints below (/decision/state, /decision/kill-switch,
/decision/review/clear) — a kill switch set via the API correctly suppresses
a subsequent /demo/run for that asset too, which is the point of exposing
this as a live operational endpoint rather than a pure demo. `python -m
src.demo` is the deterministic, fully isolated entry point (a fresh
tempfile.TemporaryDirectory() per run — see src/decision/store.py and
src/demo.py); use that when you need two runs to produce identical output
regardless of local operational state.

Run locally:
    uvicorn src.api.main:app --reload
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/demo/run
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.decision.exit_rules import (
    OptionSnapshot,
    SpikeExitConfig,
    SpikeHedgeState,
    ValuationHistory,
    compute_spike_exit,
)
from src.decision.store import DecisionStore
from src.demo import run_symbol_pipeline
from src.validation.schema import ValidationError

app = FastAPI(
    title="Crypto Volatility & Hedging Decision API",
    description="Offline-runnable BTC/ETH volatility, regime, and HEDGE/WAIT/REVIEW decision service.",
    version="0.1.0",
)

_SAMPLE_DIR = Path("data/sample")
_ASSETS = ("BTC", "ETH")


class DemoRunRequest(BaseModel):
    notional: float = 100_000.0
    budget_pct: float = 0.015


class KillSwitchRequest(BaseModel):
    asset: Literal["BTC", "ETH"]
    active: bool
    set_by: str = "api"


class ReviewClearRequest(BaseModel):
    asset: Literal["BTC", "ETH"]
    cleared_by: str
    note: str = ""


class ExitCheckRequest(BaseModel):
    """Inputs for one spike-exit evaluation of an already-open hedge."""

    entry_value: float
    peak_value: float
    days_held: int
    current_value: float
    delta: float
    intrinsic_value: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mode": "offline"}


@app.post("/demo/run")
def demo_run(body: DemoRunRequest) -> dict:
    """
    Run the full offline pipeline for BTC and ETH against data/sample/*.csv.

    Returns the same structure as reports/demo_output.json. Requires sample
    data to already exist (generate it first with
    `python -m scripts.generate_sample_data`).
    """
    missing = [a for a in _ASSETS if not (_SAMPLE_DIR / f"{a.lower()}_hourly.csv").exists()]
    if missing:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Missing sample data for {missing}. "
                "Generate it with: python -m scripts.generate_sample_data"
            ),
        )

    results = {}
    for asset in _ASSETS:
        csv_path = _SAMPLE_DIR / f"{asset.lower()}_hourly.csv"
        try:
            results[asset] = run_symbol_pipeline(asset, csv_path, body.notional, body.budget_pct)
        except ValidationError as exc:
            results[asset] = {"symbol": asset, "error": str(exc)}
    return results


@app.post("/decision/exit-check")
def exit_check(body: ExitCheckRequest) -> dict:
    """
    Evaluate the spike-exit rules for an already-open hedge position.

    Pure/stateless — does not read or write any DecisionStore. Complements
    the entry-side HEDGE/WAIT/REVIEW decision with the exit-side question:
    given a hedge already held, should it be trimmed, exited, or held today.
    See src/decision/exit_rules.py for the six-rule evaluation order.
    """
    decision = compute_spike_exit(
        hedge_state=SpikeHedgeState(
            entry_value=body.entry_value, peak_value=body.peak_value, days_held=body.days_held
        ),
        option_snapshot=OptionSnapshot(
            current_value=body.current_value, delta=body.delta, intrinsic_value=body.intrinsic_value
        ),
        valuation_history=ValuationHistory(values=[body.current_value]),
        config=SpikeExitConfig(),
    )
    return {
        "should_exit": decision.should_exit,
        "sell_fraction": decision.sell_fraction,
        "reason": decision.reason,
        "message": decision.message,
    }


@app.get("/decision/state")
def decision_state(asset: Literal["BTC", "ETH"]) -> dict:
    store = DecisionStore(asset=asset)
    return store.state_snapshot()


@app.post("/decision/kill-switch")
def set_kill_switch(body: KillSwitchRequest) -> dict:
    store = DecisionStore(asset=body.asset)
    store.set_kill_switch(body.active, set_by=body.set_by)
    store.save()
    return store.state_snapshot()


@app.post("/decision/review/clear")
def clear_review(body: ReviewClearRequest) -> dict:
    store = DecisionStore(asset=body.asset)
    if not store.load_review_state().active:
        raise HTTPException(
            status_code=409, detail=f"{body.asset} is not currently in REVIEW state."
        )
    store.clear_review_state(cleared_by=body.cleared_by, note=body.note)
    store.save()
    return store.state_snapshot()
