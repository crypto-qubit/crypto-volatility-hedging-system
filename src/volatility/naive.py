"""
Naive Realized Volatility Estimator
=====================================

Mathematical Definition
-----------------------
Given log returns r_t = ln(close_t / close_{t-1}), naive realized volatility
over a rolling window of N bars is:

    σ_naive(t, N) = std(r_{t-N+1}, ..., r_t)    [sample std, ddof=1]

Annualized (hourly data, 365.25 days/year):

    σ_annualized = σ_naive × sqrt(8766)

Assumptions
-----------
- Close prices are strictly positive.
- No distributional assumption on returns is required.
- All bars are of equal duration (hourly).
- Stationarity is not assumed; the estimator adapts to a rolling window only.

Strengths
---------
- Fully transparent: every practitioner understands rolling std.
- No parameter sensitivity beyond window length.
- O(N) per bar (pandas vectorized); lowest latency of the three estimators.
- Difficult to break: no division by zero risk beyond log returns.
- Easy to audit: one line of math, one line of code.

Weaknesses
----------
- Equal weighting: a spike from N-1 bars ago counts identically to the last bar.
- Slow to react: the window must accumulate N bars before fully reflecting a new
  volatility regime.
- Slow to forget: a past spike remains in the window for exactly N bars, creating
  a hard cliff when it drops off.
- Uses only close prices: intrabar movement (open, high, low) is invisible.
- Underestimates true realized vol in comparison to OHLC-aware estimators.

Crypto-Specific Risks
---------------------
- BTC/ETH can move 10–30% in a single hour during liquidation cascades (e.g.
  March 2020, May 2021, FTX collapse Nov 2022). A single spike inflates the
  window reading for all N subsequent bars.
- Thin low-liquidity bars (e.g. late-night UTC hours) can produce near-zero
  close-to-close returns even when intrabar range is wide, underestimating vol.
- The hard cliff at bar N is operationally hazardous: a regime change can cause
  an abrupt vol drop on the bar when the old spike exits the window.

Computational Complexity
------------------------
- O(N) per bar, fully vectorized via pandas rolling().std().
- Suitable for real-time hourly ingestion.

Expected Behavior During Liquidation Cascades
---------------------------------------------
- If the cascade produces large close-to-close returns, naive vol spikes on the
  first affected bar and remains elevated for N bars.
- If the cascade produces a large intrabar range but close-to-close is small
  (price recovers within the bar), naive vol does NOT react. Use Yang-Zhang
  instead for this case.
- After the event, vol stays elevated for exactly N bars, then drops abruptly.

Expected Behavior During Calm Markets
--------------------------------------
- Correct, stable, low readings.
- No artificial floor: vol can approach zero after extended calm.
- If a stale spike is still in the window, the reading is artificially elevated
  until bar N when the spike drops off.

When It Tends to React Late
----------------------------
- During intra-bar volatility events where close-to-close is flat.
- During slow, gradual volatility regime escalation (requires N bars to reflect).

When It Tends to Overreact
---------------------------
- After a single extreme return: the bar inflates readings for N bars even if
  current conditions are calm.
- Short windows amplify single-bar outliers disproportionately.

When It May Fail Operationally
-------------------------------
- Missing bars produce NaN returns; if > (N - min_periods) bars are missing,
  the output is NaN for those timestamps.
- The hard cliff at bar N (when a spike exits the window) can produce sudden vol
  step-downs that confuse threshold systems relying on smooth signals.
"""

import pandas as pd

from src.validation.schema import validate_ohlcv
from src.volatility.rolling_windows import rolling_percentile, rolling_zscore, safe_log_returns
from src.volatility.utils import HOURS_PER_YEAR, annualize_volatility, build_output_frame

MODEL_NAME = "naive_realized"

# 90-day lookback for rolling percentile at hourly frequency
PERCENTILE_WINDOW_90D = 90 * 24  # 2160 bars


class NaiveVolatility:
    """
    Rolling realized volatility as the rolling standard deviation of log returns.

    Parameters
    ----------
    window : int
        Rolling window in bars (hours for hourly data). Default: 24 (1 day).
    min_periods : int
        Minimum valid observations before producing output. Default: 2.
    periods_per_year : float
        Used for annualization. Default: 8766 (hourly, 365.25 days/yr).
    percentile_window : int
        Lookback window (bars) for rolling percentile. Default: 2160 (90 days).
    """

    def __init__(
        self,
        window: int = 24,
        min_periods: int = 2,
        periods_per_year: float = HOURS_PER_YEAR,
        percentile_window: int = PERCENTILE_WINDOW_90D,
    ) -> None:
        if window < 2:
            raise ValueError("window must be >= 2 to compute a meaningful std.")
        self.window = window
        self.min_periods = max(min_periods, 2)
        self.periods_per_year = periods_per_year
        self.percentile_window = percentile_window

    def compute(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Compute naive realized volatility for a single symbol.

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

            volatility_raw = rolling std of log returns (per-bar units).
            NaN for the first (window - 1) bars (warmup period).
        """
        df = validate_ohlcv(df, symbol=symbol)

        log_returns = safe_log_returns(df["close"])

        vol_raw = log_returns.rolling(
            window=self.window,
            min_periods=self.min_periods,
        ).std(ddof=1)

        vol_annualized = annualize_volatility(vol_raw, self.periods_per_year)

        zscore = rolling_zscore(vol_raw, window=self.window, min_periods=self.min_periods)

        pct = rolling_percentile(vol_raw, window=self.percentile_window, min_periods=1)

        return build_output_frame(
            timestamps=df["timestamp"],
            symbol=symbol,
            volatility_raw=vol_raw,
            volatility_annualized=vol_annualized,
            zscore=zscore,
            percentile_90d=pct,
            window=self.window,
            model_name=MODEL_NAME,
        )

    def compute_multi_window(
        self,
        df: pd.DataFrame,
        symbol: str,
        windows: list[int],
    ) -> dict[int, pd.DataFrame]:
        """
        Compute naive vol for multiple rolling windows.

        Returns
        -------
        dict mapping window_size -> output DataFrame.
        Useful for comparing 24h, 168h (1 week), 720h (30 day) horizons.
        """
        return {
            w: NaiveVolatility(
                window=w,
                min_periods=self.min_periods,
                periods_per_year=self.periods_per_year,
                percentile_window=self.percentile_window,
            ).compute(df, symbol)
            for w in windows
        }
