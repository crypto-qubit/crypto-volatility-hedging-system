"""
DUMMY BASELINE THRESHOLD SYSTEM — Layer 2: Derived Returns Calculator
=======================================================================

Computes multi-horizon log returns and rolling drawdowns from hourly OHLCV data.
These are the input features for the dummy threshold system and sit in the
`derived_returns` table — a separate layer from raw OHLCV.

IMPORTANT
---------
This module is part of the DUMMY BASELINE THRESHOLD SYSTEM.
Outputs are labeled "DUMMY BASELINE" and are intended ONLY for:
  - infrastructure testing
  - event window alignment
  - model responsiveness comparison
  - threshold calibration experiments

NOT for production trading or risk management use.

Architecture note
-----------------
Maintains the separation:
    raw_market_data → derived_returns → threshold_flags → event_windows

No derived data is written back to the OHLCV tables.
"""

import numpy as np
import pandas as pd

from src.validation.schema import validate_ohlcv

DERIVED_RETURNS_COLUMNS = [
    "timestamp",
    "symbol",
    "close",
    "log_return_1h",
    "log_return_24h",
    "log_return_168h",
    "rolling_drawdown_24h",
    "rolling_drawdown_168h",
]


class DerivedReturnsCalculator:
    """
    Computes multi-horizon log returns and rolling drawdowns from OHLCV data.

    Horizons
    --------
    1h  : bar-to-bar log return = ln(close_t / close_{t-1})
    24h : 24-bar log return     = ln(close_t / close_{t-24})
    168h: 168-bar log return    = ln(close_t / close_{t-168})

    Rolling drawdowns (always in [-1, 0])
    --------------------------------------
    rolling_drawdown_24h  = (close - rolling_24h_max)  / rolling_24h_max
    rolling_drawdown_168h = (close - rolling_168h_max) / rolling_168h_max

    A drawdown of 0 means the current close is the rolling high.
    A drawdown of -0.30 means the close is 30% below the rolling high.

    No future leakage
    -----------------
    All operations use only bars at or before the current timestamp.
    rolling_max uses min_periods=1 so the first bars get single-bar peaks,
    producing drawdown=0 rather than NaN.

    NaN policy
    ----------
    log_return_1h   : NaN at bar 0 only
    log_return_24h  : NaN for first 24 bars
    log_return_168h : NaN for first 168 bars
    rolling_drawdowns: non-NaN from bar 0 (value = 0.0 until window fills)
    """

    def compute(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Compute derived returns and drawdowns for a single symbol.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data. Required columns:
            timestamp, open, high, low, close, volume, symbol.
        symbol : str
            "BTC" or "ETH".

        Returns
        -------
        pd.DataFrame
            Columns per DERIVED_RETURNS_COLUMNS. One row per input bar.
        """
        df = validate_ohlcv(df, symbol=symbol)
        close = df["close"].reset_index(drop=True)

        # Log prices for efficient multi-horizon differencing
        log_c = np.log(close.where(close > 0))

        log_ret_1h = log_c.diff(1)
        log_ret_24h = log_c.diff(24)
        log_ret_168h = log_c.diff(168)

        # Rolling high-water marks (min_periods=1 → no NaN)
        peak_24 = close.rolling(24, min_periods=1).max()
        dd_24 = (close - peak_24) / peak_24

        peak_168 = close.rolling(168, min_periods=1).max()
        dd_168 = (close - peak_168) / peak_168

        return pd.DataFrame(
            {
                "timestamp": df["timestamp"].reset_index(drop=True),
                "symbol": symbol,
                "close": close.values,
                "log_return_1h": log_ret_1h.values,
                "log_return_24h": log_ret_24h.values,
                "log_return_168h": log_ret_168h.values,
                "rolling_drawdown_24h": dd_24.values,
                "rolling_drawdown_168h": dd_168.values,
            }
        )
