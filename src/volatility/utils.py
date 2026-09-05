"""
Shared utilities for volatility estimators.

Provides the annualization factor, standardized output schema construction,
and nothing else. No model logic lives here.
"""

import numpy as np
import pandas as pd

# Hourly crypto: 365.25 days × 24 hours = 8766 hours per year
HOURS_PER_YEAR: float = 365.25 * 24  # 8766.0
ANNUALIZATION_FACTOR_HOURLY: float = np.sqrt(HOURS_PER_YEAR)


def annualize_volatility(
    vol_raw: pd.Series,
    periods_per_year: float = HOURS_PER_YEAR,
) -> pd.Series:
    """
    Scale a per-period volatility series to annualized units.

    vol_annualized = vol_raw * sqrt(periods_per_year)

    Parameters
    ----------
    vol_raw : pd.Series
        Per-period volatility (dimensionless; units of log returns per bar).
    periods_per_year : float
        Number of bars per year. Default: 8766 for hourly crypto data.

    Returns
    -------
    pd.Series
        Annualized volatility.
    """
    return vol_raw * np.sqrt(periods_per_year)


def build_output_frame(
    timestamps: pd.Series,
    symbol: str,
    volatility_raw: pd.Series,
    volatility_annualized: pd.Series,
    zscore: pd.Series,
    percentile_90d: pd.Series,
    window: int,
    model_name: str,
) -> pd.DataFrame:
    """
    Construct the standardized output DataFrame for a volatility estimator.

    Output schema
    -------------
    timestamp            UTC datetime
    symbol               Asset symbol (BTC, ETH)
    volatility_raw       Per-period volatility in log-return units
    volatility_annualized Annualized volatility
    zscore               Rolling z-score of raw vol within the estimator window
    percentile_90d       Rolling percentile rank of raw vol over 90-day lookback
    window               Rolling window size used (bars), or effective half-life for EWMA
    model_name           Estimator identifier string

    Parameters
    ----------
    timestamps : pd.Series
    symbol : str
    volatility_raw : pd.Series
    volatility_annualized : pd.Series
    zscore : pd.Series
    percentile_90d : pd.Series
    window : int
    model_name : str

    Returns
    -------
    pd.DataFrame with exactly the schema described above.
    """
    out = pd.DataFrame(
        {
            "timestamp": timestamps.values,
            "symbol": symbol,
            "volatility_raw": volatility_raw.values,
            "volatility_annualized": volatility_annualized.values,
            "zscore": zscore.values,
            "percentile_90d": percentile_90d.values,
            "window": window,
            "model_name": model_name,
        }
    )

    if not pd.api.types.is_datetime64_any_dtype(out["timestamp"]):
        out = out.assign(timestamp=pd.to_datetime(out["timestamp"], utc=True))
    elif out["timestamp"].dt.tz is None:
        out = out.assign(timestamp=out["timestamp"].dt.tz_localize("UTC"))

    return out
