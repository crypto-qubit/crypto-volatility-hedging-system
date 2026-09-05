"""
Input validation for OHLCV crypto data.

Validates DataFrame structure, column types, timestamp ordering, and OHLC
sanity before passing data to volatility estimators.
"""

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume", "symbol"}
OHLCV_PRICE_COLS = ["open", "high", "low", "close"]
VALID_SYMBOLS = {"BTC", "ETH"}


class ValidationError(ValueError):
    pass


def validate_ohlcv(df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """
    Validate and sanitize an OHLCV DataFrame for use in volatility estimators.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with columns: timestamp, open, high, low, close, volume, symbol.
    symbol : str, optional
        If provided, filter to this symbol and validate it is known.

    Returns
    -------
    pd.DataFrame
        Validated (and optionally filtered) DataFrame, sorted ascending by timestamp.

    Raises
    ------
    ValidationError
        If the DataFrame fails any structural or sanity check.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValidationError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValidationError("DataFrame is empty.")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValidationError(f"Missing required columns: {missing}")

    df = df.copy()

    if symbol is not None:
        if symbol not in VALID_SYMBOLS:
            raise ValidationError(f"Unknown symbol '{symbol}'. Valid: {VALID_SYMBOLS}")
        df = df[df["symbol"] == symbol].copy()
        if df.empty:
            raise ValidationError(f"No rows found for symbol '{symbol}'.")

    # Ensure timestamp is UTC datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        except Exception as exc:
            raise ValidationError(f"Cannot parse 'timestamp' as datetime: {exc}") from exc

    if df["timestamp"].dt.tz is None:
        df = df.assign(timestamp=df["timestamp"].dt.tz_localize("UTC"))
    else:
        df = df.assign(timestamp=df["timestamp"].dt.tz_convert("UTC"))

    df = df.sort_values("timestamp").reset_index(drop=True)

    if df.duplicated(subset=["timestamp", "symbol"]).any():
        raise ValidationError("Duplicate (timestamp, symbol) pairs found.")

    # Cast price and volume columns to float
    for col in OHLCV_PRICE_COLS + ["volume"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception as exc:
                raise ValidationError(f"Cannot cast '{col}' to numeric: {exc}") from exc

    # Non-positive prices are invalid for log returns — nullify affected OHLC rows
    for col in OHLCV_PRICE_COLS:
        bad = (df[col] <= 0) & df[col].notna()
        if bad.any():
            df.loc[bad, OHLCV_PRICE_COLS] = np.nan

    # High < Low is a malformed bar — nullify OHLC for that bar
    invalid_hl = df["high"] < df["low"]
    if invalid_hl.any():
        df.loc[invalid_hl, OHLCV_PRICE_COLS] = np.nan

    if len(df) < 2:
        raise ValidationError("Need at least 2 rows to compute log returns.")

    return df


def check_sufficient_history(df: pd.DataFrame, window: int) -> None:
    """
    Raise ValidationError if df has fewer rows than the rolling window.

    This is a soft pre-flight check. Estimators will still run and produce
    NaNs during the warmup period; use this to catch obviously misconfigured
    windows before the compute call.
    """
    if len(df) < window:
        raise ValidationError(
            f"DataFrame has {len(df)} rows but window requires {window}. Results will be all NaN."
        )
