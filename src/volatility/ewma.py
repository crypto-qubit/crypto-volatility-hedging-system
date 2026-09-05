"""
EWMA (Exponentially Weighted Moving Average) Volatility
=========================================================

Mathematical Definition
-----------------------
The RiskMetrics EWMA estimator uses the following recursive variance update:

    σ²_t = λ · σ²_{t−1} + (1 − λ) · r²_t

where:
    σ²_t  = conditional variance estimate at time t
    r_t   = log return at time t = ln(close_t / close_{t−1})
    λ     = decay factor (persistence), 0 < λ < 1
    1 − λ = α = EWM alpha parameter in pandas

The estimator gives exponentially decaying weight to past squared returns,
with recent observations weighted more heavily than older ones.

Initialized by running ewm() from the first available return; the first
`init_window` bars are suppressed (NaN) to avoid a poorly-seeded estimate.

Relationship to GARCH
---------------------
EWMA is mathematically equivalent to an integrated GARCH(1,1) with:
    ω = 0,  α = 1 − λ,  β = λ
but it does NOT model mean reversion to an unconditional variance, does NOT
forecast future variance, and does NOT estimate parameters from data. It is
purely a conditional volatility smoother.

Assumptions
-----------
- Log returns are the appropriate input.
- λ is set exogenously (not estimated from data) and held constant.
- No distributional assumption on returns.
- No mean reversion of volatility is assumed or modeled.
- All bars are of equal duration (hourly).

Strengths
---------
- Fastest of the three estimators to react to new volatility: the first
  high-return bar immediately inflates σ²_t via (1−λ)·r²_t.
- Smooth output: no hard cliff when old spikes exit a fixed window.
- Computationally trivial: one multiply and one add per bar (O(1)).
- Well-understood in industry (Basel II, J.P. Morgan RiskMetrics 1994).
- Works on any return frequency without parameter changes.

Weaknesses
----------
- λ is not estimated from data — a misspecified λ can cause systematic
  under- or over-estimation of volatility persistence.
- For λ = 0.94 (calibrated to daily equity data): when applied to hourly
  crypto, the effective half-life is only ~12 bars, which may cause excessive
  noise in some regimes.
- Never mean-reverts to unconditional vol: once vol spikes, it only decays
  geometrically, never "snapping back" regardless of how calm the market is.
- Infinite memory: every past return contributes, with exponentially decaying
  weight. The effective window is unbounded.

Crypto-Specific Risks
---------------------
- λ = 0.94 was calibrated for daily equity data (J.P. Morgan, 1994). For hourly
  crypto, the equivalent calendar-time smoothing requires a higher λ (e.g. 0.99)
  since we sample 24× more frequently.
- Crypto fat tails: a single 20%+ return makes (1−λ)·r²_t extremely large,
  causing an immediate spike in σ²_t that then decays geometrically. A single
  outlier bar dominates the estimate for many subsequent bars.
- Prolonged low-vol periods drive σ²_t toward near-zero. When volatility
  returns, the first few bars undershoot reality because the prior state σ²_{t-1}
  is tiny.

Computational Complexity
------------------------
- O(1) per bar (recursive update, pandas ewm vectorized path).
- Lowest cost of the three estimators. Suitable for any ingestion frequency.

Expected Behavior During Liquidation Cascades
---------------------------------------------
- Reacts in the FIRST bar of the cascade: σ²_t spikes immediately.
- Decay after cascade is geometric: half-life = ln(0.5) / ln(λ) bars.
  At λ=0.94: half-life ≈ 12 bars (12 hours). At λ=0.99: half-life ≈ 69 bars.
- Fastest of the three estimators to respond to sudden volatility onset.

Expected Behavior During Calm Markets
--------------------------------------
- Smoothly decays toward zero.
- No artificial floor; σ²_t can approach 0 during extended calm.
- Transitions are smooth — no cliff effects unlike fixed rolling windows.

When It Tends to React Late
----------------------------
- When λ is set close to 1 (high persistence): the past dominates and new
  signals are barely incorporated per bar.
- During gradual volatility build-up: each small return contributes only
  (1−λ)·r²_t which may be negligible against a large σ²_{t-1}.

When It Tends to Overreact
---------------------------
- On a single outlier return (possible data error): σ²_t spikes and decays
  slowly — there is no mechanism to distinguish a genuine crash from a tick error.
- When λ is set too low (e.g. 0.80): the estimator is very noisy.

When It May Fail Operationally
-------------------------------
- If the first `init_window` bars have anomalous returns, the seeded variance
  is unreliable; the estimator can start from a wrong base and take many bars
  to converge.
- Missing bars (NaN returns) are skipped via `ignore_na=True`; the estimator
  continues from the last valid state, which is generally the correct behavior.

References
----------
J.P. Morgan / Reuters (1996). RiskMetrics Technical Document, 4th Edition.
"""

import numpy as np
import pandas as pd

from src.validation.schema import validate_ohlcv
from src.volatility.rolling_windows import rolling_percentile, rolling_zscore, safe_log_returns
from src.volatility.utils import HOURS_PER_YEAR, annualize_volatility, build_output_frame

MODEL_NAME = "ewma"

PERCENTILE_WINDOW_90D = 90 * 24  # 2160 bars at hourly frequency
DEFAULT_LAMBDA = 0.94
DEFAULT_INIT_WINDOW = 24  # bars used to suppress the startup period


class EWMAVolatility:
    """
    RiskMetrics-style EWMA conditional volatility estimator.

    σ²_t = λ · σ²_{t−1} + (1 − λ) · r²_t

    Parameters
    ----------
    lam : float
        Decay factor λ ∈ (0, 1). Default: 0.94 (J.P. Morgan RiskMetrics).
        For hourly crypto, consider 0.99 for equivalent calendar-time smoothing.
    init_window : int
        Number of bars suppressed (NaN) at startup. Default: 24.
    periods_per_year : float
        Annualization factor. Default: 8766 (hourly, 365.25 days/yr).
    percentile_window : int
        Lookback window (bars) for rolling percentile. Default: 2160 (90 days).
    """

    def __init__(
        self,
        lam: float = DEFAULT_LAMBDA,
        init_window: int = DEFAULT_INIT_WINDOW,
        periods_per_year: float = HOURS_PER_YEAR,
        percentile_window: int = PERCENTILE_WINDOW_90D,
    ) -> None:
        if not (0.0 < lam < 1.0):
            raise ValueError("lam (lambda) must be strictly between 0 and 1.")
        if init_window < 2:
            raise ValueError("init_window must be >= 2.")
        self.lam = lam
        self.alpha = 1.0 - lam  # pandas ewm alpha = 1 - lambda
        self.init_window = init_window
        self.periods_per_year = periods_per_year
        self.percentile_window = percentile_window

    @property
    def half_life(self) -> int:
        """Effective half-life of the EWMA in bars (ceil of ln(0.5)/ln(λ))."""
        return int(np.ceil(np.log(0.5) / np.log(self.lam)))

    def compute(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Compute EWMA volatility for a single symbol.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data. Required columns:
            timestamp, open, high, low, close, volume, symbol.
        symbol : str
            Asset to compute. Must be 'BTC' or 'ETH'.

        Returns
        -------
        pd.DataFrame
            Standardized output schema:
            timestamp, symbol, volatility_raw, volatility_annualized,
            zscore, percentile_90d, window, model_name.

            volatility_raw  = sqrt(σ²_EWMA) in log-return units per bar.
            window column   = effective half-life (bars), not a fixed window.
            NaN for the first `init_window` bars (suppressed startup period).
        """
        df = validate_ohlcv(df, symbol=symbol)

        log_returns = safe_log_returns(df["close"])
        r_sq = log_returns**2

        # Recursive EWMA variance:
        #   adjust=False  → σ²_t = λ·σ²_{t-1} + (1-λ)·r²_t  (recursive, not adjusted)
        #   ignore_na=True → NaN returns are skipped; σ² continues from last valid state
        #   min_periods   → first init_window bars return NaN (suppresses bad seed)
        ewma_var = r_sq.ewm(
            alpha=self.alpha,
            adjust=False,
            ignore_na=True,
            min_periods=self.init_window,
        ).mean()

        vol_raw = np.sqrt(ewma_var.clip(lower=0.0))

        vol_annualized = annualize_volatility(vol_raw, self.periods_per_year)

        # Z-score lookback: 4× half-life gives a reasonable context window
        zscore_window = max(self.half_life * 4, self.init_window)
        zscore = rolling_zscore(vol_raw, window=zscore_window, min_periods=2)

        pct = rolling_percentile(vol_raw, window=self.percentile_window, min_periods=1)

        return build_output_frame(
            timestamps=df["timestamp"],
            symbol=symbol,
            volatility_raw=vol_raw,
            volatility_annualized=vol_annualized,
            zscore=zscore,
            percentile_90d=pct,
            window=self.half_life,  # report effective half-life as the window metric
            model_name=MODEL_NAME,
        )

    def compute_multi_lambda(
        self,
        df: pd.DataFrame,
        symbol: str,
        lambdas: list[float],
    ) -> dict[float, pd.DataFrame]:
        """
        Compute EWMA vol for multiple lambda values.

        Returns
        -------
        dict mapping lambda -> output DataFrame.
        Useful for comparing reaction speeds across persistence levels.
        """
        return {
            lam: EWMAVolatility(
                lam=lam,
                init_window=self.init_window,
                periods_per_year=self.periods_per_year,
                percentile_window=self.percentile_window,
            ).compute(df, symbol)
            for lam in lambdas
        }
