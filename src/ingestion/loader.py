"""
src/ingestion/loader.py — OHLCV ingestion boundary.

In production this layer pulls hourly bars from BigQuery (or an exchange
REST/WebSocket API) and normalizes them to the pipeline's canonical schema:
timestamp, symbol, open, high, low, close, volume. This offline build ships
only the local half of that boundary — a CSV loader — so the rest of the
pipeline (src/validation onward) is exercised identically regardless of
where the bars came from.

See docs/architecture.md for how this is wired to BigQuery in the deployable
version of this system.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CANONICAL_COLUMNS = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]


def load_hourly_csv(path: str | Path, symbol: str | None = None) -> pd.DataFrame:
    """
    Load an hourly OHLCV CSV into the canonical ingestion schema.

    Parameters
    ----------
    path : str | Path
        CSV with at least: timestamp, open, high, low, close, volume.
        A `symbol` column is used if present; otherwise `symbol` is required.
    symbol : str, optional
        Overrides/fills the `symbol` column when the CSV does not carry one.

    Returns
    -------
    pd.DataFrame
        Columns exactly CANONICAL_COLUMNS, timestamp parsed as UTC.

    Raises
    ------
    ValueError
        If the file is missing required columns or no symbol is resolvable.
    """
    path = Path(path)
    df = pd.read_csv(path)

    missing = {"timestamp", "open", "high", "low", "close", "volume"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required column(s) {missing}")

    if "symbol" not in df.columns:
        if symbol is None:
            raise ValueError(f"{path}: no 'symbol' column and no symbol override provided")
        df["symbol"] = symbol
    elif symbol is not None:
        df["symbol"] = symbol

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[CANONICAL_COLUMNS]
