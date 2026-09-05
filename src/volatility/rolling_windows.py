"""
Rolling window helpers that strictly avoid lookahead bias.

All operations use only past observations within a fixed-width rolling window.
No expanding-window leakage. No future data access.
"""

import numpy as np
import pandas as pd


def safe_log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute log returns: r_t = ln(p_t / p_{t-1}).

    Handles zero and NaN prices safely — returns NaN for any bar where either
    the current or prior price is non-positive or missing.

    Parameters
    ----------
    prices : pd.Series
        Strictly positive price series (close prices).

    Returns
    -------
    pd.Series
        Log return series (same index). First element is always NaN.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        log_prices = np.log(prices.where(prices > 0))
    return log_prices.diff()


def rolling_zscore(
    series: pd.Series,
    window: int,
    min_periods: int = 2,
) -> pd.Series:
    """
    Rolling z-score of a series using only past observations within the window.

    z_t = (x_t - mean(x_{t-window+1:t})) / std(x_{t-window+1:t}, ddof=1)

    No future leakage: z_t is computed from at most `window` observations
    ending at (and including) time t.

    Parameters
    ----------
    series : pd.Series
    window : int
        Number of observations in the rolling window.
    min_periods : int
        Minimum observations required (default 2, since std needs at least 2).

    Returns
    -------
    pd.Series
        Z-scores. NaN where insufficient history or zero std.
    """
    roll = series.rolling(window=window, min_periods=max(min_periods, 2))
    mu = roll.mean()
    sigma = roll.std(ddof=1)
    return (series - mu) / sigma


def rolling_percentile(
    series: pd.Series,
    window: int,
    min_periods: int = 1,
) -> pd.Series:
    """
    Rolling percentile rank of the current value within the past `window` observations.

    Returns a value in [0, 100]: the fraction of past-window observations that
    are <= the current value, expressed as a percentage.

    Strictly causal: the current bar is ranked against the PRIOR `window` bars
    only, never against itself.

    Parameters
    ----------
    series : pd.Series
    window : int
        Number of PAST bars to include in the comparison set.
    min_periods : int
        Minimum past bars required before returning a result (default 1).

    Returns
    -------
    pd.Series
        Percentile values in [0, 100]. NaN where insufficient history.
    """

    def _pct_rank(arr: np.ndarray) -> float:
        # arr contains window+1 values: arr[:-1] = past window, arr[-1] = current
        current = arr[-1]
        past = arr[:-1]
        past = past[~np.isnan(past)]
        if len(past) == 0 or np.isnan(current):
            return np.nan
        return float(np.sum(past <= current) / len(past) * 100)

    # window+1 so the rolling frame contains `window` past values + 1 current value
    return series.rolling(
        window=window + 1,
        min_periods=min_periods + 1,
    ).apply(_pct_rank, raw=True)
