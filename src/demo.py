"""
src/demo.py — End-to-end offline demonstration.

Synthetic data -> validation -> volatility estimators -> thresholds ->
regime classification -> decision engine -> dashboard output.

Requires NO BigQuery credentials, NO GCP project, NO API key, NO network
access, NO live Cloud Run service, and NO exchange connection. If it cannot
find local sample data it tells you exactly how to generate it and exits
cleanly rather than reaching for a cloud fallback.

Isolated and deterministic by construction: main() runs the whole pipeline
against a fresh tempfile.TemporaryDirectory() for decision-layer state,
created at the start of the run and removed at the end. It never reads from
or writes to the persistent operational state directory (data/state/ or
DECISION_STATE_DIR) — two clean runs in a row always produce the same
canonical result, and running the demo can never disturb real operational
state. See "Isolated demo state vs. persistent operational state" in
src/decision/store.py for the full rationale.

Usage
-----
    python -m src.demo
    python -m src.demo --input data/sample/btc_hourly.csv
    python -m src.demo --notional 250000 --budget-pct 0.02
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.decision.runner import run_decision
from src.decision.signal_aggregator import SignalAggregator
from src.decision.store import DecisionStore
from src.ingestion.loader import load_hourly_csv
from src.pricing.option_pricer import find_target_put
from src.pricing.synthetic_quotes import generate_put_chain
from src.thresholds.derived_returns import DerivedReturnsCalculator
from src.thresholds.event_windower import EventWindower
from src.thresholds.stress_summary import StressSummaryBuilder
from src.thresholds.threshold_engine import ThresholdEngine
from src.validation.integrity import validate_ohlc_integrity
from src.validation.schema import ValidationError, validate_ohlcv
from src.volatility.ewma import EWMAVolatility
from src.volatility.naive import NaiveVolatility
from src.volatility.yang_zhang import YangZhangVolatility

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("demo")

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # for the shipped dashboard template
_DASHBOARD_TEMPLATE = _PACKAGE_ROOT / "assets" / "dashboard_template.html"
# Outputs (dashboard.html, reports/demo_output.json) are written relative to
# the current working directory at call time — see main() — so
# `python -m src.demo` always writes into whichever checkout you run it from.

SPARKLINE_POINTS = 300  # downsample series embedded in the dashboard


def _multi_horizon_log_returns(df: pd.DataFrame) -> tuple[list[float], list[float], list[float]]:
    """Derive (1H, 1D, 1W) log-return sequences from an hourly close series."""
    close_h = df.set_index("timestamp")["close"]

    log_h = np.log(close_h)
    returns_1h = log_h.diff().dropna().tail(200).tolist()

    daily = close_h.resample("1D").last().dropna()
    returns_1d = np.log(daily).diff().dropna().tail(120).tolist()

    weekly = close_h.resample("1W").last().dropna()
    returns_1w = np.log(weekly).diff().dropna().tail(60).tolist()

    return returns_1h, returns_1d, returns_1w


def run_symbol_pipeline(
    symbol: str,
    csv_path: Path,
    notional: float,
    budget_pct: float,
    state_dir: Path | None = None,
    store: DecisionStore | None = None,
) -> dict:
    """
    Run the full pipeline for one symbol.

    Decision-layer state isolation
    -------------------------------
    By default (state_dir=None, store=None) this reads/writes the per-asset
    DecisionStore's normal state_dir resolution (DECISION_STATE_DIR env var,
    falling back to data/state/) — the persistent operational location. Pass
    `state_dir` to isolate this call to that directory instead (e.g. a
    tempfile.TemporaryDirectory() — see main()'s usage below), or pass a
    fully pre-built `store` directly for tests that need to seed specific
    state first. See src/decision/store.py's module docstring for the full
    isolated-vs-persistent distinction.
    """
    store = store or DecisionStore(asset=symbol, state_dir=state_dir)
    log.info("=" * 62)
    log.info("Processing %s from %s", symbol, csv_path)

    # 1. Ingestion
    raw = load_hourly_csv(csv_path, symbol=symbol)

    # 2. Data-quality validation (FATAL / WARNING / MARKET_ANOMALY audit)
    integrity = validate_ohlc_integrity(raw, asset=symbol, expected_freq="1h")
    log.info(
        "[%s] integrity: fatal=%d warning=%d market_anomaly=%d",
        symbol,
        len(integrity.fatals()),
        len(integrity.warnings()),
        len(integrity.anomalies()),
    )
    if integrity.has_fatal:
        log.error("[%s] FATAL data-quality issues — skipping.", symbol)
        return {"symbol": symbol, "error": "fatal_data_quality", "integrity": integrity.to_dict()}

    # 3. Structural validation (used internally by every estimator below too)
    df = validate_ohlcv(raw, symbol=symbol)

    # 4. Volatility estimators
    naive = NaiveVolatility(window=24).compute(df, symbol)
    yz = YangZhangVolatility(window=24).compute(df, symbol)
    ewma = EWMAVolatility(lam=0.94).compute(df, symbol)

    latest_naive = float(naive["volatility_annualized"].dropna().iloc[-1])
    latest_yz = float(yz["volatility_annualized"].dropna().iloc[-1])
    latest_ewma = float(ewma["volatility_annualized"].dropna().iloc[-1])

    # 5. Thresholds (DUMMY BASELINE — full-sample calibration, see docs/limitations.md)
    derived = DerivedReturnsCalculator().compute(df, symbol)
    thresholds = ThresholdEngine()
    fitted = thresholds.fit(derived)
    flags = thresholds.flag(derived, fitted, vol_series=ewma["volatility_annualized"])
    flags_with_ids, windows = EventWindower().window(flags)
    stress_summary = StressSummaryBuilder().build(flags_with_ids, windows)

    # 6. Regime + multi-horizon signal
    returns_1h, returns_1d, returns_1w = _multi_horizon_log_returns(df)
    signal = SignalAggregator().compute(
        asset=symbol,
        returns_1h=returns_1h,
        returns_1d=returns_1d,
        returns_1w=returns_1w,
    )

    # 7. Synthetic option chain + pricing (offline stand-in for a live Deribit feed)
    spot = float(df["close"].iloc[-1])
    quotes = generate_put_chain(symbol, spot=spot, annualized_vol=latest_ewma)
    pricing = find_target_put(quotes, spot=spot, notional=notional) if signal.dual_trigger else None

    # 8. Decision + report
    decision, report = run_decision(
        asset=symbol,
        signal=signal,
        pricing=pricing,
        notional=notional,
        budget_pct=budget_pct,
        store=store,
    )

    log.info(
        "[%s] regime=%s (conf=%.0f%%) -> %s | %s",
        symbol,
        signal.regime,
        signal.confidence * 100,
        decision.action,
        decision.rationale[:100],
    )

    close_tail = df["close"].tail(SPARKLINE_POINTS)
    ewma_tail = ewma["volatility_annualized"].tail(SPARKLINE_POINTS)

    return {
        "symbol": symbol,
        "spot": spot,
        "data_quality": {
            "fatal": len(integrity.fatals()),
            "warning": len(integrity.warnings()),
            "market_anomaly": len(integrity.anomalies()),
            "n_bars": len(df),
        },
        "volatility": {
            "naive_annualized": latest_naive,
            "yang_zhang_annualized": latest_yz,
            "ewma_annualized": latest_ewma,
        },
        "thresholds": {
            "n_events": int(len(windows)),
            "n_flags": int(len(flags_with_ids)),
            "summary": stress_summary.to_dict("records") if not stress_summary.empty else [],
        },
        "regime": {
            "label": signal.regime,
            "confidence": signal.confidence,
            "dual_trigger": signal.dual_trigger,
            "direction_state": signal.direction_state,
        },
        "decision": {
            "action": decision.action,
            "rationale": decision.rationale,
            "overridden": decision.overridden,
            "override_reason": decision.override_reason,
        },
        "report": report.as_dict(),
        "chart": {
            "timestamps": [t.isoformat() for t in df["timestamp"].tail(SPARKLINE_POINTS)],
            "close": [round(v, 2) for v in close_tail.tolist()],
            "ewma_vol_annualized": [
                None if pd.isna(v) else round(float(v), 4) for v in ewma_tail.tolist()
            ],
        },
    }


def _write_dashboard(result: dict, dashboard_out: Path) -> None:
    if not _DASHBOARD_TEMPLATE.exists():
        log.warning(
            "Dashboard template not found at %s — skipping dashboard render.", _DASHBOARD_TEMPLATE
        )
        return
    template = _DASHBOARD_TEMPLATE.read_text(encoding="utf-8")
    payload = json.dumps(result, default=str)
    rendered = template.replace("/*__DEMO_DATA__*/null", payload)
    dashboard_out.write_text(rendered, encoding="utf-8")
    log.info("Dashboard written to %s — open it directly in a browser.", dashboard_out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=str, default="data/sample/btc_hourly.csv", help="BTC hourly OHLCV CSV."
    )
    parser.add_argument(
        "--eth-input", type=str, default="data/sample/eth_hourly.csv", help="ETH hourly OHLCV CSV."
    )
    parser.add_argument(
        "--notional", type=float, default=100_000.0, help="Synthetic portfolio notional (USD)."
    )
    parser.add_argument(
        "--budget-pct", type=float, default=0.015, help="Hedge budget as a fraction of notional."
    )
    args = parser.parse_args(argv)

    log.info("OFFLINE DEMO MODE — no GCP project, no BigQuery, no network access used.")

    inputs = {"BTC": Path(args.input), "ETH": Path(args.eth_input)}
    missing = [str(p) for p in inputs.values() if not p.exists()]
    if missing:
        log.error("Missing sample data: %s", ", ".join(missing))
        log.error("Generate it with:  python -m scripts.generate_sample_data")
        return 1

    # Isolated, disposable decision-layer state for this run only. The demo
    # must be deterministic and must never read from or write to the
    # persistent operational state directory (data/state/ or
    # DECISION_STATE_DIR) — see "Isolated demo state vs. persistent
    # operational state" in src/decision/store.py. A fresh temp directory is
    # created here and removed automatically when this `with` block exits,
    # regardless of whatever real operational state exists on this machine.
    results = {}
    with tempfile.TemporaryDirectory(prefix="crypto-vol-demo-state-") as tmp_state_dir:
        state_dir = Path(tmp_state_dir)
        for symbol, path in inputs.items():
            try:
                results[symbol] = run_symbol_pipeline(
                    symbol, path, args.notional, args.budget_pct, state_dir=state_dir
                )
            except ValidationError as exc:
                log.error("[%s] validation error: %s", symbol, exc)
                results[symbol] = {"symbol": symbol, "error": str(exc)}

    result_out = Path.cwd() / "reports" / "demo_output.json"
    result_out.parent.mkdir(parents=True, exist_ok=True)
    result_out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    log.info("Result written to %s", result_out)

    _write_dashboard(results, Path.cwd() / "dashboard.html")

    log.info("=" * 62)
    log.info("SUMMARY")
    for symbol, r in results.items():
        if "error" in r:
            log.info("  %s: ERROR — %s", symbol, r["error"])
            continue
        log.info(
            "  %s: regime=%-9s decision=%-6s vol(EWMA ann.)=%.1f%%",
            symbol,
            r["regime"]["label"],
            r["decision"]["action"],
            r["volatility"]["ewma_annualized"] * 100,
        )
    log.info("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
