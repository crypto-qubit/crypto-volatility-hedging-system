"""
DUMMY BASELINE THRESHOLD SYSTEM — Event Windower
==================================================

Merges consecutive nearby threshold flags into unified event windows.
A "window" is a contiguous cluster of flags where each consecutive pair
is separated by no more than `merge_gap_hours`.

Rationale
---------
During a market stress period, many hourly bars fire flags in succession.
Without merging, a 48-hour liquidation cascade would produce ~48 independent
alerts. The EventWindower collapses these into ONE event window, allowing:
  - clean event counting ("N stress events in 3 years")
  - duration analysis ("average crisis lasted X hours")
  - model comparison ("EWMA detected event 3 hours before naive vol")

Merge rule
----------
For a sorted sequence of flag timestamps T_0, T_1, ..., T_n (same symbol, horizon):
  - T_i and T_{i+1} are in the SAME window if:  T_{i+1} - T_i <= merge_gap_hours
  - Otherwise T_{i+1} starts a NEW window

Example with merge_gap_hours=48:
  Flags at hours: 0, 1, 2, 5, 10, 100, 101
  → Window 0: [0, 1, 2, 5, 10]   (all gaps ≤ 48h)
  → Window 1: [100, 101]          (gap 10→100 = 90h > 48h)

Windows are per (symbol, horizon) combination.
A BTC 1h window and a BTC 24h window covering the same calendar period are
SEPARATE windows — they represent detections at different horizons.

Event window IDs
----------------
Format: "{SYMBOL}_{HORIZON}_{SEQUENCE:04d}"
Examples: "BTC_1h_0000", "ETH_24h_0003", "BTC_168h_0001"

Architecture note
-----------------
Maintains separation: threshold_flags → event_windows
No data is written back to raw_market_data or derived_returns.
"""

from __future__ import annotations

import pandas as pd

EVENT_WINDOWS_COLUMNS = [
    "event_window_id",
    "symbol",
    "horizon",
    "start_timestamp",
    "end_timestamp",
    "duration_hours",
    "n_flags",
    "max_abs_return",
    "min_return",
    "min_drawdown",
    "peak_volatility",
    "primary_threshold_breached",
]


class EventWindower:
    """
    Merges threshold_flags into event windows.

    Parameters
    ----------
    merge_gap_hours : int
        Maximum gap (in hours) between consecutive flags for them to be
        merged into the same event window. Default: 48.
        Set lower to split events more finely; higher to merge more aggressively.
    """

    def __init__(self, merge_gap_hours: int = 48) -> None:
        if merge_gap_hours < 1:
            raise ValueError("merge_gap_hours must be >= 1.")
        self.merge_gap_hours = merge_gap_hours
        self._merge_gap_td = pd.Timedelta(hours=merge_gap_hours)

    def window(
        self,
        flags_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Assign event window IDs to flags and build the event_windows table.

        Parameters
        ----------
        flags_df : pd.DataFrame
            threshold_flags output from ThresholdEngine.flag().
            Required columns: timestamp, symbol, horizon, raw_return,
            abs_return, rolling_volatility, drawdown_size, threshold_breached.

        Returns
        -------
        flags_with_ids : pd.DataFrame
            Input flags_df with event_window_id column populated.
        event_windows : pd.DataFrame
            One row per distinct event window. Columns per EVENT_WINDOWS_COLUMNS.
        """
        if flags_df.empty:
            flags_out = flags_df.copy()
            flags_out["event_window_id"] = ""
            return flags_out, pd.DataFrame(columns=EVENT_WINDOWS_COLUMNS)

        flags_out = flags_df.copy().reset_index(drop=True)
        window_rows: list[dict] = []

        for (symbol, horizon), group in flags_out.groupby(["symbol", "horizon"], sort=False):
            group = group.sort_values("timestamp")
            timestamps = group["timestamp"].tolist()
            win_nums = self._assign_window_nums(timestamps)

            group = group.copy()
            group["_win_num"] = win_nums

            for w_num, w_flags in group.groupby("_win_num"):
                win_id = f"{symbol}_{horizon}_{w_num:04d}"
                flags_out.loc[w_flags.index, "event_window_id"] = win_id
                window_rows.append(self._build_window_row(win_id, symbol, horizon, w_flags))

        event_windows = (
            pd.DataFrame(window_rows, columns=EVENT_WINDOWS_COLUMNS)
            if window_rows
            else pd.DataFrame(columns=EVENT_WINDOWS_COLUMNS)
        )
        return flags_out, event_windows

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _assign_window_nums(self, timestamps: list[pd.Timestamp]) -> list[int]:
        """
        Assign sequential window numbers to a sorted timestamp list.

        Consecutive timestamps separated by more than merge_gap_hours
        start a new window (increments the counter).
        """
        if not timestamps:
            return []

        nums: list[int] = [0]
        for i in range(1, len(timestamps)):
            gap = pd.Timestamp(timestamps[i]) - pd.Timestamp(timestamps[i - 1])
            if gap > self._merge_gap_td:
                nums.append(nums[-1] + 1)
            else:
                nums.append(nums[-1])
        return nums

    def _build_window_row(
        self,
        win_id: str,
        symbol: str,
        horizon: str,
        w_flags: pd.DataFrame,
    ) -> dict:
        start_ts = w_flags["timestamp"].min()
        end_ts = w_flags["timestamp"].max()
        duration_h = (end_ts - start_ts).total_seconds() / 3600.0

        primary_thresh = w_flags["threshold_breached"].mode()
        primary_thresh = primary_thresh.iloc[0] if not primary_thresh.empty else ""

        return {
            "event_window_id": win_id,
            "symbol": symbol,
            "horizon": horizon,
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "duration_hours": duration_h,
            "n_flags": len(w_flags),
            "max_abs_return": w_flags["abs_return"].max(),
            "min_return": w_flags["raw_return"].min(),
            "min_drawdown": w_flags["drawdown_size"].min(),
            "peak_volatility": w_flags["rolling_volatility"].max(),
            "primary_threshold_breached": primary_thresh,
        }
