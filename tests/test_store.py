"""
Tests for DecisionStore JSON persistence (src/decision/store.py).
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.decision.engine import DecisionOutput, HedgeState
from src.decision.store import DecisionStore

NOW = datetime(2026, 5, 26, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _isolated_state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DECISION_STATE_DIR", str(tmp_path))


def _decision(action="HEDGE") -> DecisionOutput:
    return DecisionOutput(
        action=action,
        rationale="test rationale",
        overridden=False,
        override_reason=None,
        policy_action=action,
    )


def test_fresh_store_has_no_history_and_no_kill_switch():
    store = DecisionStore(asset="BTC")
    assert store.load_history() == []
    assert store.kill_switch is False
    assert store.load_review_state().active is False


def test_append_and_reload_round_trips(tmp_path):
    store = DecisionStore(asset="BTC")
    store.append_decision(
        _decision("HEDGE"), regime="stress", confidence=0.8, asset="BTC", report_id="r1", now=NOW
    )
    store.save()

    reloaded = DecisionStore(asset="BTC")
    history = reloaded.load_history()
    assert len(history) == 1
    assert history[0].action == "HEDGE"
    assert history[0].regime == "stress"


def test_history_pruned_to_24h_window_but_keeps_last_hedge():
    store = DecisionStore(asset="BTC")
    old_hedge_time = NOW - timedelta(hours=48)
    store.append_decision(_decision("HEDGE"), "stress", 0.8, "BTC", "r1", now=old_hedge_time)
    store.append_decision(_decision("WAIT"), "normal", 0.7, "BTC", "r2", now=NOW)
    store.save()

    reloaded = DecisionStore(asset="BTC")
    history = reloaded.load_history()
    # The 48h-old HEDGE is outside the 24h pruning window but must survive
    # regardless of age; the WAIT is recent so it survives on its own merit.
    assert any(r.action == "HEDGE" for r in history)
    assert any(r.action == "WAIT" for r in history)


def test_review_state_activation_is_idempotent():
    store = DecisionStore(asset="BTC")
    store.activate_review_state("first reason", now=NOW)
    store.activate_review_state("second reason", now=NOW + timedelta(hours=1))

    review = store.load_review_state()
    assert review.active is True
    assert review.reason == "first reason"


def test_review_state_clear_logs_clearance():
    store = DecisionStore(asset="BTC")
    store.activate_review_state("anomaly", now=NOW)
    store.clear_review_state(cleared_by="operator", note="confirmed false positive", now=NOW)
    store.save()

    assert store.load_review_state().active is False
    assert store._state["review_clearance_log"][-1]["cleared_by"] == "operator"


def test_kill_switch_round_trips():
    store = DecisionStore(asset="BTC")
    store.set_kill_switch(True, set_by="operator", now=NOW)
    store.save()

    reloaded = DecisionStore(asset="BTC")
    assert reloaded.kill_switch is True


def test_hedge_state_round_trips():
    store = DecisionStore(asset="BTC")
    hedge = HedgeState(
        is_hedged=True,
        instrument="BTC-P",
        strike=90000.0,
        notional=100000.0,
        premium_paid_usd=1500.0,
        opened_at=NOW,
        hedge_ratio=1.0,
    )
    store.update_hedge_state(hedge, now=NOW)
    store.save()

    reloaded = DecisionStore(asset="BTC")
    loaded = reloaded.load_hedge_state()
    assert loaded.is_hedged is True
    assert loaded.instrument == "BTC-P"
    assert loaded.hedge_ratio == 1.0


def test_pending_hedge_and_approval_log():
    store = DecisionStore(asset="BTC")
    store.set_pending_hedge("r1", "BTC", {"instrument": "BTC-P"}, now=NOW)
    assert store.load_pending_hedge()["report_id"] == "r1"

    store.log_approval("r1", approved=True, approved_by="operator", note="ok", now=NOW)
    store.clear_pending_hedge()
    store.save()

    assert store.load_pending_hedge() is None
    assert store._state["approval_log"][-1]["approved"] is True


def test_different_assets_do_not_share_state(tmp_path):
    btc_store = DecisionStore(asset="BTC")
    btc_store.append_decision(_decision("HEDGE"), "stress", 0.8, "BTC", "r1", now=NOW)
    btc_store.save()

    eth_store = DecisionStore(asset="ETH")
    assert eth_store.load_history() == []


def test_corrupt_state_file_falls_back_to_empty(tmp_path):
    state_file = tmp_path / "decision_state_BTC.json"
    state_file.write_text("{not valid json", encoding="utf-8")

    store = DecisionStore(asset="BTC")
    assert store.load_history() == []
