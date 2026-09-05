"""
Tests for the three volatility estimators (src/volatility/).

Covers output schema, warmup NaN behavior, annualization, parameter
validation, and the no-lookahead property (each is only allowed to use
past/current bars — perturbing the future must never change a past value).
"""

import numpy as np
import pandas as pd
import pytest

from src.volatility.ewma import MODEL_NAME as EWMA_NAME
from src.volatility.ewma import EWMAVolatility
from src.volatility.naive import MODEL_NAME as NAIVE_NAME
from src.volatility.naive import NaiveVolatility
from src.volatility.utils import HOURS_PER_YEAR
from src.volatility.yang_zhang import MODEL_NAME as YZ_NAME
from src.volatility.yang_zhang import YangZhangVolatility
from tests.conftest import make_ohlcv

ESTIMATORS = [
    (NaiveVolatility, {"window": 24}, NAIVE_NAME),
    (YangZhangVolatility, {"window": 24}, YZ_NAME),
    (EWMAVolatility, {"lam": 0.94}, EWMA_NAME),
]


@pytest.mark.parametrize("cls,kwargs,name", ESTIMATORS)
def test_output_schema(btc_df, cls, kwargs, name):
    out = cls(**kwargs).compute(btc_df, "BTC")
    expected_cols = {
        "timestamp",
        "symbol",
        "volatility_raw",
        "volatility_annualized",
        "zscore",
        "percentile_90d",
        "window",
        "model_name",
    }
    assert set(out.columns) == expected_cols
    assert len(out) == len(btc_df)
    assert (out["model_name"] == name).all()
    assert (out["symbol"] == "BTC").all()


@pytest.mark.parametrize("cls,kwargs,name", ESTIMATORS)
def test_annualization_is_consistent_with_raw(btc_df, cls, kwargs, name):
    out = cls(**kwargs).compute(btc_df, "BTC")
    valid = out.dropna(subset=["volatility_raw", "volatility_annualized"])
    assert len(valid) > 0
    ratio = valid["volatility_annualized"] / valid["volatility_raw"]
    assert ratio.round(6).nunique() == 1
    assert ratio.iloc[0] == pytest.approx(np.sqrt(HOURS_PER_YEAR), rel=1e-6)


@pytest.mark.parametrize("cls,kwargs,name", ESTIMATORS)
def test_warmup_period_is_nan(btc_df, cls, kwargs, name):
    out = cls(**kwargs).compute(btc_df, "BTC")
    assert out["volatility_raw"].iloc[0:1].isna().all()


@pytest.mark.parametrize("cls,kwargs,name", ESTIMATORS)
def test_no_lookahead_bias(btc_df, cls, kwargs, name):
    """Changing only the LAST bar must not change any earlier estimator output."""
    estimator = cls(**kwargs)
    baseline = estimator.compute(btc_df, "BTC")

    perturbed = btc_df.copy()
    perturbed.loc[perturbed.index[-1], ["open", "high", "low", "close"]] *= 1.5
    perturbed.loc[perturbed.index[-1], "high"] *= 1.1  # keep OHLC valid after scaling

    mutated = estimator.compute(perturbed, "BTC")

    pd.testing.assert_series_equal(
        baseline["volatility_raw"].iloc[:-1],
        mutated["volatility_raw"].iloc[:-1],
        check_names=False,
    )


def test_naive_rejects_window_below_two():
    with pytest.raises(ValueError):
        NaiveVolatility(window=1)


def test_yang_zhang_rejects_window_below_three():
    with pytest.raises(ValueError):
        YangZhangVolatility(window=2)


def test_ewma_rejects_lambda_outside_unit_interval():
    with pytest.raises(ValueError):
        EWMAVolatility(lam=1.0)
    with pytest.raises(ValueError):
        EWMAVolatility(lam=0.0)


def test_ewma_half_life_matches_formula():
    est = EWMAVolatility(lam=0.94)
    expected = int(np.ceil(np.log(0.5) / np.log(0.94)))
    assert est.half_life == expected


def test_yang_zhang_uses_all_ohlc_columns(btc_df):
    """A pure intrabar-range widening (no close-to-close move) should raise YZ vol
    even though naive (close-only) vol is unaffected."""
    df = btc_df.copy()
    widened = df.copy()
    widened["high"] = widened["high"] * 1.05
    widened["low"] = widened["low"] * 0.95

    naive_before = NaiveVolatility(window=24).compute(df, "BTC")["volatility_raw"].iloc[-1]
    naive_after = NaiveVolatility(window=24).compute(widened, "BTC")["volatility_raw"].iloc[-1]
    yz_before = YangZhangVolatility(window=24).compute(df, "BTC")["volatility_raw"].iloc[-1]
    yz_after = YangZhangVolatility(window=24).compute(widened, "BTC")["volatility_raw"].iloc[-1]

    assert naive_after == pytest.approx(naive_before)
    assert yz_after > yz_before


def test_multi_window_naive_returns_one_frame_per_window(btc_df):
    result = NaiveVolatility(window=24).compute_multi_window(btc_df, "BTC", windows=[12, 24, 48])
    assert set(result.keys()) == {12, 24, 48}
    for w, frame in result.items():
        assert (frame["window"] == w).all()


def test_estimator_rejects_missing_required_columns():
    df = make_ohlcv(n=50)
    df = df.drop(columns=["high"])
    with pytest.raises(Exception):
        NaiveVolatility().compute(df, "BTC")
