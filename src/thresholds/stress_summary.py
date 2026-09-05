"""
DUMMY BASELINE THRESHOLD SYSTEM — Stress Summary Builder
==========================================================

Produces summary statistics over the threshold_flags and event_windows tables.
Intended for comparing model detection quality, threshold sensitivity, and
event characteristics across assets and horizons.

IMPORTANT: DUMMY BASELINE — NOT for production use.

Output: one row per (symbol, horizon) combination.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STRESS_SUMMARY_COLUMNS = [
    "symbol",
    "horizon",
    "n_total_flags",
    "n_event_windows",
    "avg_flags_per_window",
    "avg_window_duration_hours",
    "max_window_duration_hours",
    "pct_flags_pct9975_triggered",
    "pct_flags_floor_triggered",
    "p50_abs_return",
    "p95_abs_return",
    "p99_abs_return",
    "p50_peak_volatility",
    "p95_peak_volatility",
    "p99_peak_volatility",
    "flag_label",
]


class StressSummaryBuilder:
    """
    Builds summary statistics from threshold_flags and event_windows tables.

    Intended use: compare detection counts, window durations, return sizes,
    and peak volatility levels across assets and horizons.
    """

    def build(
        self,
        flags_df: pd.DataFrame,
        event_windows_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute summary statistics per (symbol, horizon).

        Parameters
        ----------
        flags_df : pd.DataFrame
            threshold_flags with event_window_id populated (output of EventWindower).
        event_windows_df : pd.DataFrame
            event_windows output of EventWindower.

        Returns
        -------
        pd.DataFrame
            One row per (symbol, horizon). Columns per STRESS_SUMMARY_COLUMNS.
            Empty DataFrame if flags_df is empty.
        """
        if flags_df.empty:
            return pd.DataFrame(columns=STRESS_SUMMARY_COLUMNS)

        rows: list[dict] = []

        for (symbol, horizon), f_group in flags_df.groupby(["symbol", "horizon"]):
            # Window stats (join on event_windows_df for this group)
            w_group = event_windows_df[
                (event_windows_df["symbol"] == symbol) & (event_windows_df["horizon"] == horizon)
            ]

            n_flags = len(f_group)
            n_windows = len(w_group)
            avg_flags_per_window = n_flags / n_windows if n_windows > 0 else 0.0
            avg_duration = w_group["duration_hours"].mean() if not w_group.empty else 0.0
            max_duration = w_group["duration_hours"].max() if not w_group.empty else 0.0

            thresh_col = f_group["threshold_breached"]
            n_pct = (thresh_col == "RETURN_PCT_9975").sum() + (
                thresh_col == "DRAWDOWN_PCT_9975"
            ).sum()
            n_floor = (thresh_col == "RETURN_ABS_FLOOR").sum()
            pct_pct9975 = n_pct / n_flags * 100.0
            pct_floor = n_floor / n_flags * 100.0

            abs_ret = f_group["abs_return"].dropna()
            peak_vol = (
                w_group["peak_volatility"].dropna() if not w_group.empty else pd.Series(dtype=float)
            )

            flag_label = f_group["flag_label"].iloc[0] if "flag_label" in f_group.columns else ""

            rows.append(
                {
                    "symbol": symbol,
                    "horizon": horizon,
                    "n_total_flags": n_flags,
                    "n_event_windows": n_windows,
                    "avg_flags_per_window": round(avg_flags_per_window, 2),
                    "avg_window_duration_hours": round(float(avg_duration), 2),
                    "max_window_duration_hours": round(float(max_duration), 2),
                    "pct_flags_pct9975_triggered": round(pct_pct9975, 2),
                    "pct_flags_floor_triggered": round(pct_floor, 2),
                    "p50_abs_return": float(np.nanpercentile(abs_ret, 50))
                    if not abs_ret.empty
                    else np.nan,
                    "p95_abs_return": float(np.nanpercentile(abs_ret, 95))
                    if not abs_ret.empty
                    else np.nan,
                    "p99_abs_return": float(np.nanpercentile(abs_ret, 99))
                    if not abs_ret.empty
                    else np.nan,
                    "p50_peak_volatility": float(np.nanpercentile(peak_vol, 50))
                    if not peak_vol.empty
                    else np.nan,
                    "p95_peak_volatility": float(np.nanpercentile(peak_vol, 95))
                    if not peak_vol.empty
                    else np.nan,
                    "p99_peak_volatility": float(np.nanpercentile(peak_vol, 99))
                    if not peak_vol.empty
                    else np.nan,
                    "flag_label": flag_label,
                }
            )

        return pd.DataFrame(rows, columns=STRESS_SUMMARY_COLUMNS)
