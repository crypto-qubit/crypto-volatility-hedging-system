"""
End-to-end smoke test: synthetic data -> validation -> volatility ->
thresholds -> regime -> decision -> saved result, with zero network access
and zero cloud credentials (asserted by monkeypatching the environment
clean of every GCP-related variable).
"""

import json

import pytest

from scripts.generate_sample_data import generate_sample_data
from src.demo import main as demo_main


@pytest.fixture(autouse=True)
def _no_cloud_env(monkeypatch):
    for var in ("PROJECT_ID", "DATASET", "AUDIT_DATASET", "GOOGLE_APPLICATION_CREDENTIALS"):
        monkeypatch.delenv(var, raising=False)


def test_demo_runs_offline_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("DECISION_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)

    generate_sample_data(days=60, out_dir=tmp_path / "data" / "sample")

    exit_code = demo_main(
        [
            "--input",
            str(tmp_path / "data" / "sample" / "btc_hourly.csv"),
            "--eth-input",
            str(tmp_path / "data" / "sample" / "eth_hourly.csv"),
        ]
    )

    assert exit_code == 0

    result_path = tmp_path / "reports" / "demo_output.json"
    assert result_path.exists()

    result = json.loads(result_path.read_text())
    assert set(result.keys()) == {"BTC", "ETH"}
    for symbol, r in result.items():
        assert "error" not in r
        assert r["decision"]["action"] in {"HEDGE", "WAIT", "REVIEW"}
        assert r["regime"]["label"] in {"stress", "high_vol", "normal", "low_vol"}
        assert r["data_quality"]["fatal"] == 0


def test_demo_reports_missing_sample_data_cleanly(tmp_path, monkeypatch):
    monkeypatch.setenv("DECISION_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)

    exit_code = demo_main(
        [
            "--input",
            str(tmp_path / "does_not_exist_btc.csv"),
            "--eth-input",
            str(tmp_path / "does_not_exist_eth.csv"),
        ]
    )

    assert exit_code == 1
