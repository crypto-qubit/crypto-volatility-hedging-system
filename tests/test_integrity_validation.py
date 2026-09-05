"""
Tests for the OHLC integrity validation layer (src/validation/integrity.py).

Covers the FATAL / WARNING / MARKET_ANOMALY severity taxonomy documented in
the module's audit-fix history.
"""

import numpy as np
import pandas as pd

from src.validation.integrity import validate_all_assets, validate_ohlc_integrity
from tests.conftest import make_ohlcv


def test_clean_data_has_no_fatals():
    df = make_ohlcv(n=100)
    report = validate_ohlc_integrity(df, asset="BTC")
    assert report.has_fatal is False


def test_empty_dataframe_is_fatal():
    report = validate_ohlc_integrity(pd.DataFrame(), asset="BTC")
    assert report.has_fatal is True
    assert report.results[0].check_id == "FATAL-EMPTY-DATAFRAME"


def test_high_below_low_is_fatal():
    df = make_ohlcv(n=50)
    df.loc[10, "high"] = df.loc[10, "low"] - 100
    report = validate_ohlc_integrity(df, asset="BTC")
    assert report.has_fatal is True
    assert any(r.check_id == "FATAL-HIGH-BELOW-LOW" for r in report.fatals())


def test_high_below_open_is_fatal():
    df = make_ohlcv(n=50)
    df.loc[10, "high"] = df.loc[10, "open"] - 50
    report = validate_ohlc_integrity(df, asset="BTC")
    assert any(r.check_id == "FATAL-HIGH-BELOW-OPEN" for r in report.fatals())


def test_nonpositive_price_is_fatal_regardless_of_epsilon():
    df = make_ohlcv(n=50)
    df.loc[5, "close"] = -1.0
    report = validate_ohlc_integrity(df, asset="BTC", epsilon=1.0)
    assert any(r.check_id == "FATAL-NONPOSITIVE-CLOSE" for r in report.fatals())


def test_null_column_is_fatal():
    df = make_ohlcv(n=50)
    df.loc[3, "open"] = np.nan
    report = validate_ohlc_integrity(df, asset="BTC")
    assert any(r.check_id == "FATAL-NULL-OPEN" for r in report.fatals())


def test_missing_required_column_is_fatal():
    df = make_ohlcv(n=50).drop(columns=["low"])
    report = validate_ohlc_integrity(df, asset="BTC")
    assert any(r.check_id == "FATAL-MISSING-COL-LOW" for r in report.fatals())


def test_volume_spike_is_warning_not_fatal():
    df = make_ohlcv(n=60)
    df.loc[40, "volume"] = df["volume"].median() * 50
    report = validate_ohlc_integrity(df, asset="BTC")
    assert report.has_fatal is False
    assert any(r.check_id == "WARNING-VOLUME-SPIKE" for r in report.warnings())


def test_zero_volume_is_warning():
    df = make_ohlcv(n=30)
    df.loc[5, "volume"] = 0.0
    report = validate_ohlc_integrity(df, asset="BTC")
    assert any(r.check_id == "WARNING-ZERO-VOLUME" for r in report.warnings())


def test_stale_ohlc_streak_is_market_anomaly():
    df = make_ohlcv(n=30)
    df.loc[10:20, ["open", "high", "low", "close"]] = 100.0
    report = validate_ohlc_integrity(df, asset="BTC", consecutive_flat_threshold=6)
    assert any(r.check_id == "MARKET_ANOMALY-STALE-OHLC" for r in report.anomalies())


def test_short_dataframe_short_circuits_anomaly_checks():
    df = make_ohlcv(n=3)
    report = validate_ohlc_integrity(df, asset="BTC", consecutive_flat_threshold=6)
    assert not any(r.check_id.startswith("MARKET_ANOMALY") for r in report.results)


def test_validate_all_assets_isolates_failures():
    good = make_ohlcv(n=50, symbol="BTC")
    bad = make_ohlcv(n=50, symbol="ETH")
    bad.loc[0, "close"] = -5.0

    reports = validate_all_assets({"BTC": good, "ETH": bad})

    assert reports["BTC"].has_fatal is False
    assert reports["ETH"].has_fatal is True


def test_report_round_trips_through_json():
    df = make_ohlcv(n=20)
    report = validate_ohlc_integrity(df, asset="BTC")
    payload = report.to_dict()
    assert payload["asset"] == "BTC"
    assert isinstance(report.to_json(), str)
