"""
Yang–Zhang Realized Volatility Estimator
==========================================

Mathematical Definition
-----------------------
Yang and Zhang (2000) derived the minimum-variance unbiased volatility estimator
that is simultaneously:
  - independent of drift,
  - consistent under opening jumps (overnight gaps), and
  - efficient relative to all linear combinations of OHLC sub-estimators.

The formula is a weighted combination of three sub-components:

    σ²_YZ = σ²_o + k · σ²_c + (1 − k) · σ²_RS

Sub-components over a rolling window of n bars:

1. Overnight variance (open-jump component):
       σ²_o = Var(o_i)  [sample variance, ddof=1]
       o_i  = ln(Open_i / Close_{i−1})

2. Open-to-close variance (close-volatility component):
       σ²_c = Var(c_i)  [sample variance, ddof=1]
       c_i  = ln(Close_i / Open_i)

3. Rogers-Satchell intraday estimator (drift-independent):
       σ²_RS = (1/n) · Σ [ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)]

Optimal k (Yang-Zhang 2000, minimizes estimator variance):
       k = 0.34 / (1.34 + (n+1) / (n−1))

The k parameter is fixed for a given window size — it is not estimated from data.

All rolling calculations use closed-left windows: bar t's output uses only
bars {t-n+1, ..., t}. No future data is ever accessed.

Assumptions
-----------
- Open, High, Low, Close are all from the same bar and reflect traded prices.
- Open_i is the first traded price of bar i (not necessarily = Close_{i−1}).
- All prices are strictly positive.
- The four OHLC prices are all available; missing any one collapses the RS term
  to NaN for that bar.
- No distributional assumption on returns.

Strengths
---------
- Uses all four OHLC price points, not just close-to-close.
- 4–14× more statistically efficient than close-to-close estimators under
  Brownian motion assumptions (higher efficiency = narrower confidence intervals).
- Captures intrabar range explosions invisible to naive vol.
- Handles overnight gaps correctly via the σ²_o component.
- Path-aware: detects liquidation cascades where H/L range diverges from C-to-C.

Weaknesses
----------
- Requires all four OHLC columns; close-only data sources cannot use this.
- OHLC data quality is critical: exchange-reported H/L on thin books can be
  distorted by single trades or data errors.
- The optimal k is derived under Brownian motion; under fat-tailed crypto returns
  it may not be globally optimal.
- Slightly more complex to implement and audit than naive vol.

Crypto-Specific Risks
---------------------
- "Wick spikes": thin order books allow single trades to set extreme H or L,
  inflating the RS term without reflecting genuine price discovery.
- Perpetual futures: funding rate resets can create artificial open-vs-close gaps
  that inflate σ²_o without representing true overnight risk.
- Exchange downtime during high-vol events can produce stale OHLC bars where
  the true volatility is not captured.
- During exchange outages: a zero-range bar (H=L=O=C) makes RS=0 for that bar,
  understating realized vol in the window.

Computational Complexity
------------------------
- O(N) per bar via vectorized pandas rolling operations.
- Slightly more expensive than naive vol (6 log operations on 4 columns per bar)
  but still suitable for real-time hourly ingestion.

Expected Behavior During Liquidation Cascades
---------------------------------------------
- The RS term (intrabar range) spikes immediately on the first cascade bar, even
  if close-to-close return is small (common during "V-shaped" intrabar cascades).
- The σ²_o component captures gap-opens at the start of a cascade session.
- Yang-Zhang typically responds FASTER than naive vol during liquidation cascades.
- After the cascade, elevated readings persist for N bars (window-dependent).

Expected Behavior During Calm Markets
--------------------------------------
- Low, stable readings.
- RS is non-negative by construction, so σ²_YZ cannot go negative.
- More sensitive than naive vol to small intrabar price oscillations in calm
  conditions — may read slightly higher than naive vol in quiet periods.

When It Tends to React Late
----------------------------
- When volatility is driven purely by close-to-close drift with compressed intrabar
  ranges (e.g., trending with no mean reversion within the bar).
- Large window N buffers many low-vol bars before a single spike shifts the mean.

When It Tends to Overreact
---------------------------
- When single bad OHLC ticks create wick spikes that inflate H/L (RS spike).
- When perpetual funding resets create artificial open gaps (σ²_o spike).

When It May Fail Operationally
-------------------------------
- If open prices are unavailable or equal to close (some data providers omit open),
  σ²_o collapses to zero and σ²_c is computed on zero returns — total vol is
  understated.
- If a bar has H = L (zero-range), that bar contributes 0 to the RS mean.
  Multiple such bars in the window (e.g., exchange halt periods) will compress
  the YZ estimate below true realized vol.

References
----------
Yang, D., & Zhang, Q. (2000). Drift-Independent Volatility Estimation Based on
High, Low, Open, and Close Prices. Journal of Business, 73(3), 477–491.
"""

import numpy as np
import pandas as pd

from src.validation.schema import validate_ohlcv
from src.volatility.rolling_windows import rolling_percentile, rolling_zscore
from src.volatility.utils import HOURS_PER_YEAR, annualize_volatility, build_output_frame

MODEL_NAME = "yang_zhang"

PERCENTILE_WINDOW_90D = 90 * 24  # 2160 bars at hourly frequency


class YangZhangVolatility:
    """
    Rolling Yang-Zhang realized volatility estimator using OHLC prices.

    Parameters
    ----------
    window : int
        Rolling window in bars (n in the YZ formula). Default: 24.
    min_periods : int
        Minimum valid bars required before producing output. Default: 10.
    periods_per_year : float
        Annualization factor. Default: 8766 (hourly, 365.25 days/yr).
    percentile_window : int
        Lookback window (bars) for rolling percentile. Default: 2160 (90 days).
    """

    def __init__(
        self,
        window: int = 24,
        min_periods: int = 10,
        periods_per_year: float = HOURS_PER_YEAR,
        percentile_window: int = PERCENTILE_WINDOW_90D,
    ) -> None:
        if window < 3:
            raise ValueError(
                "window must be >= 3: Yang-Zhang sample variance requires n >= 3 "
                "(ddof=1 collapses at n=2, and k is undefined at n=1)."
            )
        self.window = window
        self.min_periods = max(min_periods, 3)
        self.periods_per_year = periods_per_year
        self.percentile_window = percentile_window

    def compute(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Compute Yang-Zhang realized volatility for a single symbol.

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

            volatility_raw = sqrt(σ²_YZ) in log-return units per bar.
            NaN for the first (window - 1) bars (warmup period).
        """
        df = validate_ohlcv(df, symbol=symbol)
        n = self.window

        # --- Log prices (all strictly positive after validation) ---
        log_o = np.log(df["open"])
        log_h = np.log(df["high"])
        log_l = np.log(df["low"])
        log_c = np.log(df["close"])

        # --- Sub-component returns (causal: bar t uses bar t's own OHLC and
        #     bar t-1's close only) ---

        # o_i = ln(Open_i / Close_{i-1}): overnight (close-to-open) return
        overnight_ret = log_o - log_c.shift(1)

        # c_i = ln(Close_i / Open_i): open-to-close return
        oc_ret = log_c - log_o

        # Rogers-Satchell per-bar term (no window yet):
        #   rs_i = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)
        rs_per_bar = (log_h - log_c) * (log_h - log_o) + (log_l - log_c) * (log_l - log_o)

        # --- Rolling variance components ---
        mp = self.min_periods

        # σ²_o: sample variance of overnight returns (ddof=1)
        var_overnight = overnight_ret.rolling(n, min_periods=mp).var(ddof=1)

        # σ²_c: sample variance of open-to-close returns (ddof=1)
        var_oc = oc_ret.rolling(n, min_periods=mp).var(ddof=1)

        # σ²_RS: rolling mean of per-bar RS terms (1/n weighting)
        mean_rs = rs_per_bar.rolling(n, min_periods=mp).mean()

        # --- Optimal k (YZ 2000) ---
        k = 0.34 / (1.34 + (n + 1) / (n - 1))

        # --- Yang-Zhang combined variance ---
        yz_variance = var_overnight + k * var_oc + (1.0 - k) * mean_rs

        # Clamp floating-point negatives (RS can be negative for individual bars
        # under rare data errors; the mean is theoretically non-negative)
        yz_variance = yz_variance.clip(lower=0.0)

        vol_raw = np.sqrt(yz_variance)

        vol_annualized = annualize_volatility(vol_raw, self.periods_per_year)

        zscore = rolling_zscore(vol_raw, window=n, min_periods=mp)

        pct = rolling_percentile(vol_raw, window=self.percentile_window, min_periods=1)

        return build_output_frame(
            timestamps=df["timestamp"],
            symbol=symbol,
            volatility_raw=vol_raw,
            volatility_annualized=vol_annualized,
            zscore=zscore,
            percentile_90d=pct,
            window=n,
            model_name=MODEL_NAME,
        )

    def compute_multi_window(
        self,
        df: pd.DataFrame,
        symbol: str,
        windows: list[int],
    ) -> dict[int, pd.DataFrame]:
        """
        Compute Yang-Zhang vol for multiple rolling windows.

        Returns
        -------
        dict mapping window_size -> output DataFrame.
        """
        return {
            w: YangZhangVolatility(
                window=w,
                min_periods=self.min_periods,
                periods_per_year=self.periods_per_year,
                percentile_window=self.percentile_window,
            ).compute(df, symbol)
            for w in windows
        }
