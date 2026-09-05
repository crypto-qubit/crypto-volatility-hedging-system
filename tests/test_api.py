"""
Tests for the FastAPI interface (src/api/main.py).
"""

import pytest
from fastapi.testclient import TestClient

from scripts.generate_sample_data import generate_sample_data
from src.api.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DECISION_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    generate_sample_data(days=60, out_dir=tmp_path / "data" / "sample")
    return TestClient(app)


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok", "mode": "offline"}


def test_demo_run_without_sample_data_returns_409(tmp_path, monkeypatch):
    monkeypatch.setenv("DECISION_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    resp = TestClient(app).post("/demo/run", json={})
    assert resp.status_code == 409


def test_demo_run_produces_both_assets(client):
    resp = client.post("/demo/run", json={"notional": 50_000.0, "budget_pct": 0.02})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"BTC", "ETH"}
    for asset, r in body.items():
        assert "error" not in r
        assert r["decision"]["action"] in {"HEDGE", "WAIT", "REVIEW"}


def test_decision_state_starts_empty(client):
    resp = client.get("/decision/state", params={"asset": "BTC"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["kill_switch"] is False
    assert body["history_count"] == 0


def test_kill_switch_round_trips(client):
    resp = client.post(
        "/decision/kill-switch", json={"asset": "BTC", "active": True, "set_by": "tester"}
    )
    assert resp.status_code == 200
    assert resp.json()["kill_switch"] is True

    state = client.get("/decision/state", params={"asset": "BTC"}).json()
    assert state["kill_switch"] is True

    # ETH must be unaffected — per-asset isolation.
    eth_state = client.get("/decision/state", params={"asset": "ETH"}).json()
    assert eth_state["kill_switch"] is False


def test_clear_review_without_active_review_is_conflict(client):
    resp = client.post(
        "/decision/review/clear", json={"asset": "BTC", "cleared_by": "tester", "note": "n/a"}
    )
    assert resp.status_code == 409


def test_kill_switch_forces_review_on_next_demo_run(client):
    client.post("/decision/kill-switch", json={"asset": "BTC", "active": True})
    resp = client.post("/demo/run", json={})
    assert resp.json()["BTC"]["decision"]["action"] == "REVIEW"

    # Clear it and confirm the endpoint now succeeds.
    client.post("/decision/kill-switch", json={"asset": "BTC", "active": False})


def test_exit_check_holds_below_all_thresholds():
    resp = TestClient(app).post(
        "/decision/exit-check",
        json={
            "entry_value": 1000.0,
            "peak_value": 1000.0,
            "days_held": 5,
            "current_value": 1050.0,
            "delta": -0.5,
            "intrinsic_value": 0.0,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["should_exit"] is False
    assert body["reason"] is None


def test_exit_check_fires_spike_tier():
    resp = TestClient(app).post(
        "/decision/exit-check",
        json={
            "entry_value": 1000.0,
            "peak_value": 1000.0,
            "days_held": 2,
            "current_value": 2100.0,
            "delta": -0.6,
            "intrinsic_value": 500.0,
        },
    )
    body = resp.json()
    assert body["should_exit"] is True
    assert body["reason"] == "SPIKE_2X"
    assert body["sell_fraction"] == pytest.approx(0.50)
