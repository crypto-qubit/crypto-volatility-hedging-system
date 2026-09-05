"""
Demo decision-layer state isolation.

`python -m src.demo` must be deterministic and must never read from or write
to the persistent operational state directory (data/state/ or
DECISION_STATE_DIR) — see "Isolated demo state vs. persistent operational
state" in src/decision/store.py. These tests prove three distinct properties:

1. Two consecutive clean demo runs produce identical canonical output.
2. Unrelated stale state sitting in the *persistent* location does not leak
   into, or get mutated by, an isolated demo run.
3. Anti-whipsaw still works correctly *within* an isolated state directory —
   isolation prevents cross-run/cross-location leakage, not legitimate
   state-dependent behavior for a single, deliberately-shared state_dir.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from scripts.generate_sample_data import generate_sample_data
from src.decision.engine import DecisionOutput
from src.decision.store import DecisionStore
from src.demo import main as demo_main
from src.demo import run_symbol_pipeline

NOW = datetime.now(timezone.utc)

# Fields that vary run-to-run by design (uuid/wall-clock) and must be
# excluded from an "identical output" comparison.
_VOLATILE_KEYS = {"report"}


def _canonical(result: dict) -> dict:
    """Strip run-to-run-volatile fields (report_id, generated_at, ...) from
    a per-symbol result so what's left is the deterministic, comparable
    "canonical" output: spot, data_quality, volatility, thresholds, regime,
    decision, chart.
    """
    return {k: v for k, v in result.items() if k not in _VOLATILE_KEYS}


@pytest.fixture
def canonical_sample_data(tmp_path):
    """The full 400-day fixture (same as scripts/generate_sample_data.py's
    default) — this is the series that deterministically produces BTC in a
    live tail-shock (-> stress -> HEDGE) and ETH calm (-> normal -> WAIT),
    matching the README's documented canonical result."""
    out_dir = tmp_path / "data" / "sample"
    generate_sample_data(out_dir=out_dir)  # default days=400
    return out_dir


def test_two_consecutive_demo_runs_produce_identical_canonical_output(
    tmp_path, monkeypatch, canonical_sample_data
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DECISION_STATE_DIR", raising=False)

    args = [
        "--input", str(canonical_sample_data / "btc_hourly.csv"),
        "--eth-input", str(canonical_sample_data / "eth_hourly.csv"),
    ]  # fmt: skip

    assert demo_main(args) == 0
    run1 = json.loads((tmp_path / "reports" / "demo_output.json").read_text())

    assert demo_main(args) == 0
    run2 = json.loads((tmp_path / "reports" / "demo_output.json").read_text())

    assert set(run1.keys()) == set(run2.keys()) == {"BTC", "ETH"}
    for symbol in ("BTC", "ETH"):
        assert _canonical(run1[symbol]) == _canonical(run2[symbol]), (
            f"{symbol}: two clean demo runs produced different canonical output — "
            "isolated state is leaking between runs."
        )

    # The deterministic fixture is specifically designed so BTC ends inside a
    # live shock and ETH stays calm — assert the canonical result directly.
    assert run1["BTC"]["decision"]["action"] == "HEDGE"
    assert run1["BTC"]["regime"]["label"] == "stress"
    assert run1["ETH"]["decision"]["action"] == "WAIT"
    assert run1["ETH"]["regime"]["label"] == "normal"


def test_demo_isolation_ignores_stale_persistent_state(
    tmp_path, monkeypatch, canonical_sample_data
):
    """
    Pre-populate the PERSISTENT (DECISION_STATE_DIR) location for BTC with an
    unrelated stale kill switch — if the demo read persistent state, BTC
    would come back REVIEW instead of the canonical HEDGE. It must not.
    """
    persistent_dir = tmp_path / "persistent_ops_state"
    monkeypatch.setenv("DECISION_STATE_DIR", str(persistent_dir))
    monkeypatch.chdir(tmp_path)

    persistent_store = DecisionStore(asset="BTC")  # resolves via DECISION_STATE_DIR
    persistent_store.set_kill_switch(True, set_by="unrelated-ops-test-fixture")
    persistent_store.save()
    assert (persistent_dir / "decision_state_BTC.json").exists()

    args = [
        "--input", str(canonical_sample_data / "btc_hourly.csv"),
        "--eth-input", str(canonical_sample_data / "eth_hourly.csv"),
    ]  # fmt: skip
    assert demo_main(args) == 0

    result = json.loads((tmp_path / "reports" / "demo_output.json").read_text())
    assert result["BTC"]["decision"]["action"] == "HEDGE", (
        "Demo result changed to REVIEW — it must be reading the persistent "
        "kill switch instead of staying isolated in its own temp state dir."
    )

    # The demo run must not have touched the persistent state either.
    reloaded_persistent = DecisionStore(asset="BTC")
    assert reloaded_persistent.kill_switch is True, (
        "Persistent operational state was modified by an isolated demo run."
    )
    assert reloaded_persistent.load_history() == [], (
        "Isolated demo run wrote decision history into the persistent store."
    )


def test_running_demo_creates_no_files_outside_reports_and_dashboard(
    tmp_path, monkeypatch, canonical_sample_data
):
    """The isolated temp state dir must be removed when main() returns — no
    data/state/ (or any other stray directory) should exist afterward."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DECISION_STATE_DIR", raising=False)

    args = [
        "--input", str(canonical_sample_data / "btc_hourly.csv"),
        "--eth-input", str(canonical_sample_data / "eth_hourly.csv"),
    ]  # fmt: skip
    assert demo_main(args) == 0

    assert not (tmp_path / "data" / "state").exists()


def test_anti_whipsaw_suppresses_hedge_within_six_hours(tmp_path, canonical_sample_data):
    """
    Anti-whipsaw must still work correctly *within* one isolated state_dir:
    seed it directly with a recent BTC HEDGE, then run the pipeline against
    that same state_dir and confirm the second HEDGE is suppressed to WAIT.
    """
    iso_state_dir = tmp_path / "iso_state"
    seed_store = DecisionStore(asset="BTC", state_dir=iso_state_dir)
    seed_store.append_decision(
        decision=DecisionOutput(
            action="HEDGE",
            rationale="seeded prior hedge for anti-whipsaw test",
            overridden=False,
            override_reason=None,
            policy_action="HEDGE",
        ),
        regime="stress",
        confidence=0.80,
        asset="BTC",
        report_id="seed-report-1",
        now=NOW - timedelta(hours=1),
    )
    seed_store.save()

    result = run_symbol_pipeline(
        "BTC",
        canonical_sample_data / "btc_hourly.csv",
        notional=100_000.0,
        budget_pct=0.015,
        state_dir=iso_state_dir,
    )

    assert result["decision"]["action"] == "WAIT"
    assert result["decision"]["override_reason"] == "anti_whipsaw"
    assert result["regime"]["label"] == "stress"  # regime is still stress; only the action changed
