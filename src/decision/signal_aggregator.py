"""
Signal aggregator — multi-horizon volatility signal computation.

Computes EWMA and Naive volatility signals at three horizons (1H, 1D, 1W)
from close-price log return series. All regime classification and the hedge
trigger decision are delegated to core.risk_checks.detect_market_regime so
the decision layer uses the same regime model as the research/risk-check
pipeline.

Decision logic
--------------
dual_trigger is True when detect_market_regime classifies the current 1H
return series as "stress". No EWMA z-score or naive return threshold is used
to drive the decision.

Observability
-------------
Layer A/B/C signals (EWMA z-scores, naive vol, threshold flags) are still
computed and carried in SignalState for reporting and monitoring purposes.
They do not affect the trigger.

No OHLCV required: all inputs are log return sequences. This makes the module
usable directly from the live trigger service without a full market data frame.

Vol math (matches vol_models/ exactly):
    EWMA : σ²_t = λ·σ²_{t-1} + (1-λ)·r²_t   (adjust=False, ignore_na=True)
    Naive: rolling std of log returns (ddof=1)
    z    : (vol_raw - rolling_mean) / rolling_std

Regime states (from detect_market_regime):
    stress | high_vol | normal | low_vol

Layer horizons (observability only, from multi-horizon research, 2026-05-11):
    Layer A (1H) : EWMA λ=0.94, Naive rolling std
    Layer B (1D) : EWMA λ=0.97, Naive rolling std
    Layer C (1W) : Naive vol expansion vs rolling median
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.decision.directional_signal import (
    build_directional_hedge_signal,
    pct_return_from_log_returns,
)
from src.regime.risk_checks import detect_market_regime

# ---------------------------------------------------------------------------
# Thresholds (calibrated from research_validation and multi_horizon_validation)
# ---------------------------------------------------------------------------

# Layer A (1H): EWMA z-score trigger — vol must be elevated above recent baseline
EWMA_ZSCORE_THRESHOLD_A: float = 2.0
# Layer A (1H): Naive confirmation — raw |return| on this bar
NAIVE_RETURN_THRESHOLD_A: float = 0.015  # 1.5% move in the triggering hour
# Layer B (1D): EWMA confirmation — lower bar; just confirm deterioration is real
EWMA_ZSCORE_THRESHOLD_B: float = 1.5
# Layer C (1W): structural vol expansion ratio vs rolling median
STRUCTURAL_EXPANSION_THRESHOLD: float = 1.5  # current vol > 1.5× rolling median

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------

_LAYER_A_LAMBDA = 0.94  # RiskMetrics; fastest reaction at 1H
_LAYER_B_LAMBDA = 0.97  # Smoother; better FP rate at 1D (research: FP 12%→8.6%)
_EWMA_INIT_WINDOW_1H = 24  # suppress first 24 bars at 1H (1 day)
_EWMA_INIT_WINDOW_1D = 5  # suppress first 5 bars at 1D (1 week)

_NAIVE_WINDOW_1H = 24  # 1-day rolling std
_NAIVE_WINDOW_1D = 7  # 1-week rolling std
_NAIVE_WINDOW_1W = 4  # 4-week rolling std
_STRUCTURAL_MEDIAN_WINDOW = 12  # 12-week rolling median baseline

_HOURS_PER_YEAR = 8766.0  # 365.25 * 24
_DAYS_PER_YEAR = 365.25
_WEEKS_PER_YEAR = 52.18


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class HorizonSignal:
    horizon: str  # "1H", "1D", "1W"
    ewma_zscore: float | None  # positive when vol is elevated
    naive_vol_annualized: float | None
    naive_threshold_hit: bool
    ewma_threshold_hit: bool
    last_return: float | None  # most recent log return in series
    n_bars: int  # number of return bars provided


@dataclass
class SignalState:
    asset: str
    timestamp: datetime
    layer_a: HorizonSignal  # 1H
    layer_b: HorizonSignal  # 1D
    layer_c: HorizonSignal  # 1W
    dual_trigger: bool  # detect_market_regime returned "stress"
    regime: str  # stress|high_vol|normal|low_vol
    confidence: float  # [0.0, 1.0]
    pct_5d_return: float | None = None
    direction_state: str | None = None  # RISING|FALLING|NEUTRAL when available
    missing_layers: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# EWMA math (mirrors ewma_volatility.py)
# ---------------------------------------------------------------------------


def _ewma_var(returns: pd.Series, lam: float, init_window: int) -> pd.Series:
    """σ²_t = λ·σ²_{t-1} + (1-λ)·r²_t, suppressed for first init_window bars."""
    alpha = 1.0 - lam
    r_sq = returns**2
    return r_sq.ewm(
        alpha=alpha,
        adjust=False,
        ignore_na=True,
        min_periods=init_window,
    ).mean()


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """(x_t - mean_{t-window:t}) / std_{t-window:t}; positive when x is high."""
    roll = series.rolling(window=window, min_periods=2)
    mu = roll.mean()
    sigma = roll.std(ddof=1)
    return (series - mu) / sigma.replace(0.0, float("nan"))


def _ewma_zscore_series(
    returns: pd.Series,
    lam: float,
    init_window: int,
) -> pd.Series:
    var_series = _ewma_var(returns, lam, init_window)
    vol_raw = np.sqrt(var_series.clip(lower=0.0))
    half_life = int(math.ceil(math.log(0.5) / math.log(lam)))
    zscore_window = max(half_life * 4, init_window)
    return _rolling_zscore(vol_raw, window=zscore_window)


def _naive_vol_ann(returns: pd.Series, window: int, periods_per_year: float) -> pd.Series:
    """Rolling std of log returns, annualized."""
    vol_raw = returns.rolling(window=window, min_periods=2).std(ddof=1)
    return vol_raw * math.sqrt(periods_per_year)


def _safe_last(series: pd.Series) -> float | None:
    """Return the last value of a Series, or None if NaN."""
    val = series.iloc[-1] if len(series) > 0 else float("nan")
    return None if math.isnan(val) else float(val)


# ---------------------------------------------------------------------------
# Per-layer computation
# ---------------------------------------------------------------------------


def _layer_a(returns_1h: list[float]) -> HorizonSignal:
    n = len(returns_1h)
    if n < 2:
        return HorizonSignal("1H", None, None, False, False, None, n)

    r = pd.Series(returns_1h, dtype=float)
    zscores = _ewma_zscore_series(r, _LAYER_A_LAMBDA, _EWMA_INIT_WINDOW_1H)
    vol_ann = _naive_vol_ann(r, _NAIVE_WINDOW_1H, _HOURS_PER_YEAR)

    last_z = _safe_last(zscores)
    last_vol = _safe_last(vol_ann)
    last_r = float(r.iloc[-1])

    # EWMA trigger: z-score is POSITIVE when vol is elevated above recent baseline.
    ewma_hit = last_z is not None and last_z >= EWMA_ZSCORE_THRESHOLD_A
    # Naive confirmation: the triggering bar must itself show a meaningful move.
    naive_hit = abs(last_r) >= NAIVE_RETURN_THRESHOLD_A

    return HorizonSignal(
        horizon="1H",
        ewma_zscore=last_z,
        naive_vol_annualized=last_vol,
        naive_threshold_hit=naive_hit,
        ewma_threshold_hit=ewma_hit,
        last_return=last_r,
        n_bars=n,
    )


def _layer_b(returns_1d: list[float]) -> HorizonSignal:
    n = len(returns_1d)
    if n < 2:
        return HorizonSignal("1D", None, None, False, False, None, n)

    r = pd.Series(returns_1d, dtype=float)
    zscores = _ewma_zscore_series(r, _LAYER_B_LAMBDA, _EWMA_INIT_WINDOW_1D)
    vol_ann = _naive_vol_ann(r, _NAIVE_WINDOW_1D, _DAYS_PER_YEAR)

    last_z = _safe_last(zscores)
    last_vol = _safe_last(vol_ann)
    last_r = float(r.iloc[-1])

    ewma_hit = last_z is not None and last_z >= EWMA_ZSCORE_THRESHOLD_B
    naive_hit = abs(last_r) >= 0.03  # 3% daily move

    return HorizonSignal(
        horizon="1D",
        ewma_zscore=last_z,
        naive_vol_annualized=last_vol,
        naive_threshold_hit=naive_hit,
        ewma_threshold_hit=ewma_hit,
        last_return=last_r,
        n_bars=n,
    )


def _layer_c(returns_1w: list[float]) -> HorizonSignal:
    n = len(returns_1w)
    if n < 2:
        return HorizonSignal("1W", None, None, False, False, None, n)

    r = pd.Series(returns_1w, dtype=float)
    vol_ann = _naive_vol_ann(r, _NAIVE_WINDOW_1W, _WEEKS_PER_YEAR)

    # Structural trigger: current vol > 1.5× rolling 12-week median
    median_vol = vol_ann.rolling(window=_STRUCTURAL_MEDIAN_WINDOW, min_periods=2).median()
    expansion = vol_ann / median_vol.replace(0.0, float("nan"))

    last_vol = _safe_last(vol_ann)
    last_expansion = _safe_last(expansion)
    last_r = float(r.iloc[-1])

    structural_hit = last_expansion is not None and last_expansion >= STRUCTURAL_EXPANSION_THRESHOLD

    return HorizonSignal(
        horizon="1W",
        ewma_zscore=None,  # weekly EWMA z-score is unreliable (research finding)
        naive_vol_annualized=last_vol,
        naive_threshold_hit=structural_hit,
        ewma_threshold_hit=False,
        last_return=last_r,
        n_bars=n,
    )


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------


def _market_regime(returns_1h: list[float]) -> dict:
    """
    Authoritative regime classifier for decision-layer signals.

    Keep this as a thin wrapper so all classifications in this module flow
    through core.risk_checks.detect_market_regime.
    """
    return detect_market_regime(returns_1h)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


class SignalAggregator:
    """
    Compute multi-horizon volatility signals from log return sequences.

    All thresholds are class-level defaults that can be overridden at
    construction time for sensitivity analysis or per-asset tuning.
    """

    def __init__(
        self,
        z_threshold_a: float = EWMA_ZSCORE_THRESHOLD_A,
        naive_return_threshold_a: float = NAIVE_RETURN_THRESHOLD_A,
        z_threshold_b: float = EWMA_ZSCORE_THRESHOLD_B,
    ) -> None:
        self.z_threshold_a = z_threshold_a
        self.naive_return_threshold_a = naive_return_threshold_a
        self.z_threshold_b = z_threshold_b

    def compute(
        self,
        asset: str,
        returns_1h: list[float],
        returns_1d: list[float],
        returns_1w: list[float],
        timestamp: datetime | None = None,
    ) -> SignalState:
        """
        Compute a multi-horizon signal state.

        Parameters
        ----------
        asset : str
            "BTC" or "ETH".
        returns_1h : list[float]
            Log returns at hourly frequency, chronological order.
            Minimum 48 bars recommended for a reliable z-score (4× half-life).
        returns_1d : list[float]
            Log returns at daily frequency. Minimum 14 bars recommended.
        returns_1w : list[float]
            Log returns at weekly frequency. Minimum 12 bars recommended.
        timestamp : datetime | None
            Signal timestamp. Defaults to UTC now.

        Returns
        -------
        SignalState
        """
        ts = timestamp or datetime.now(timezone.utc)
        missing: list[str] = []

        layer_a = _layer_a(returns_1h)
        layer_b = _layer_b(returns_1d)
        layer_c = _layer_c(returns_1w)

        if layer_a.n_bars < 2:
            missing.append("1H")
        if layer_b.n_bars < 2:
            missing.append("1D")
        if layer_c.n_bars < 2:
            missing.append("1W")

        # Apply instance-level threshold overrides
        layer_a = HorizonSignal(
            horizon=layer_a.horizon,
            ewma_zscore=layer_a.ewma_zscore,
            naive_vol_annualized=layer_a.naive_vol_annualized,
            naive_threshold_hit=(
                layer_a.last_return is not None
                and abs(layer_a.last_return) >= self.naive_return_threshold_a
            ),
            ewma_threshold_hit=(
                layer_a.ewma_zscore is not None and layer_a.ewma_zscore >= self.z_threshold_a
            ),
            last_return=layer_a.last_return,
            n_bars=layer_a.n_bars,
        )
        layer_b = HorizonSignal(
            horizon=layer_b.horizon,
            ewma_zscore=layer_b.ewma_zscore,
            naive_vol_annualized=layer_b.naive_vol_annualized,
            naive_threshold_hit=layer_b.naive_threshold_hit,
            ewma_threshold_hit=(
                layer_b.ewma_zscore is not None and layer_b.ewma_zscore >= self.z_threshold_b
            ),
            last_return=layer_b.last_return,
            n_bars=layer_b.n_bars,
        )

        regime_result = _market_regime(returns_1h)
        dual_trigger = regime_result["regime"] == "stress"
        reg = regime_result["regime"]
        conf = float(regime_result["confidence"])
        pct_5d_return = pct_return_from_log_returns(returns_1d, periods=5)
        direction_state = None
        if pct_5d_return is not None:
            directional_signal = build_directional_hedge_signal(
                volatility_regime=reg,
                pct_5d_return=pct_5d_return,
                regime_confidence=conf,
            )
            direction_state = directional_signal.direction_state
            conf = directional_signal.confidence

        return SignalState(
            asset=asset,
            timestamp=ts,
            layer_a=layer_a,
            layer_b=layer_b,
            layer_c=layer_c,
            dual_trigger=dual_trigger,
            regime=reg,
            confidence=conf,
            pct_5d_return=pct_5d_return,
            direction_state=direction_state,
            missing_layers=missing,
        )
