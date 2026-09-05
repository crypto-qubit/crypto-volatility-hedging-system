"""
DUMMY BASELINE THRESHOLD SYSTEM — Empirical Threshold Engine
==============================================================

WARNING: DUMMY BASELINE — NOT FOR PRODUCTION USE.
--------------------------------------------------
Thresholds are calibrated on the FULL historical sample provided to fit().
This introduces full-sample lookahead bias relative to every bar in the dataset.
This is intentional: the system exists only for infrastructure testing, event
alignment, and model comparison.

Later versions will replace this with:
  - rolling / expanding-window percentile calibration
  - EVT-based tail models (Peaks Over Threshold)
  - regime-conditioned threshold distributions
  - volatility-normalized stress scores
  - portfolio-risk-based intervention logic

Architecture
------------
Maintains the table separation:
    derived_returns → threshold_flags → (event_windows populated by EventWindower)

No flag data is written back to raw_market_data or derived_returns.

Horizons
--------
1h  : DUMMY BASELINE HOURLY STRESS
      Flag if |log_return_1h| > 99.75th pct  OR  |log_return_1h| > log(1.015)
24h : DUMMY BASELINE DAILY STRESS
      Flag if |log_return_24h| > 99.75th pct  OR  |log_return_24h| > log(1.075)
168h: DUMMY BASELINE WEEKLY STRESS
      Flag if |rolling_drawdown_168h| > 99.75th pct

Threshold floor values
----------------------
hourly  floor: log(1.015) ≈ 0.01489  (~1.5% arithmetic move)
daily   floor: log(1.075) ≈ 0.07232  (~7.5% arithmetic move)
weekly: percentile only (no hard floor per spec)

Percentile rank
---------------
Each flagged bar is assigned an empirical percentile rank of its |return| or
|drawdown| within the full provided sample. This rank uses the full-sample CDF
(intentional lookahead for the dummy system).

Historical-replay guard rail
-----------------------------
fit() calibrating on the full sample is *not* a lookahead problem for a
caller that only ever reads the latest bar for live alerting (thresholds
computed from "everything up to now" are legitimate). It *is* a lookahead
problem if flag() output is ever replayed as a historical time series —
every past bar's flag would be contaminated by future data. Call
assert_safe_for_historical_replay() from any backtest/historical-input data
loader before consuming flags_df that way; see that function's docstring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_LABEL_PREFIX = "DUMMY_BASELINE"
_PERCENTILE_LEVEL = 99.75
_HOURLY_ABS_FLOOR = np.log(1.015)  # ≈ 0.01489 log return (~1.5% arithmetic)
_DAILY_ABS_FLOOR = np.log(1.075)  # ≈ 0.07232 log return (~7.5% arithmetic)


def assert_safe_for_historical_replay(flags_df: pd.DataFrame) -> None:
    """
    Raise if `flags_df` (ThresholdEngine.flag() output) is about to be used
    as a time-series input to a backtest or historical decision-layer test.

    Every row produced by flag() carries a DUMMY_BASELINE-prefixed
    flag_label because fit() calibrates on the full sample (intentional
    lookahead — see module docstring). A caller iterating this table across
    historical timestamps, rather than reading only the latest bar for live
    alerting, would see future-contaminated thresholds at every earlier bar.
    """
    if not flags_df.empty and flags_df["flag_label"].str.startswith(_LABEL_PREFIX).any():
        raise ValueError(
            "threshold_flags contains DUMMY_BASELINE (full-sample lookahead) rows — "
            "unsafe to consume as historical backtest/decision input. See "
            "src/thresholds/threshold_engine.py module docstring."
        )


THRESHOLD_FLAGS_COLUMNS = [
    "timestamp",
    "symbol",
    "horizon",
    "raw_return",
    "abs_return",
    "percentile_rank",
    "rolling_volatility",
    "drawdown_size",
    "threshold_breached",
    "flag_label",
    "event_window_id",
]


@dataclass
class EmpiricalThresholds:
    """
    Fitted dummy baseline thresholds for a single asset.

    All percentile thresholds are computed over the FULL provided sample.
    Hard floor values are fixed constants per the dummy spec.

    DUMMY BASELINE — NOT for production use.
    """

    symbol: str

    # 99.75th percentile of full-sample absolute values
    hourly_return_p9975: float
    daily_return_p9975: float
    weekly_drawdown_p9975: float

    # Fixed hard floors
    hourly_return_floor: float  # log(1.015)
    daily_return_floor: float  # log(1.075)

    # Sample metadata
    n_hourly_obs: int
    n_daily_obs: int
    n_weekly_obs: int
    data_start: pd.Timestamp
    data_end: pd.Timestamp

    def describe(self) -> str:
        lines = [
            f"[DUMMY BASELINE] EmpiricalThresholds for {self.symbol}",
            f"  Data range : {self.data_start}  →  {self.data_end}",
            f"  1h  P99.75 |return|    = {self.hourly_return_p9975:.6f}",
            f"  1h  abs floor          = {self.hourly_return_floor:.6f}  [log(1.015)]",
            f"  1h  n_obs              = {self.n_hourly_obs}",
            f"  24h P99.75 |return|    = {self.daily_return_p9975:.6f}",
            f"  24h abs floor          = {self.daily_return_floor:.6f}  [log(1.075)]",
            f"  24h n_obs              = {self.n_daily_obs}",
            f"  168h P99.75 |drawdown| = {self.weekly_drawdown_p9975:.6f}",
            f"  168h n_obs             = {self.n_weekly_obs}",
        ]
        return "\n".join(lines)


class ThresholdEngine:
    """
    DUMMY BASELINE multi-horizon stress threshold engine.

    Two-phase operation:
    1. fit()  — calibrate empirical thresholds from the full historical sample.
    2. flag() — apply thresholds to produce the threshold_flags table.

    IMPORTANT: fit() uses the FULL sample (lookahead intentional for dummy system).
    """

    def fit(self, derived_df: pd.DataFrame) -> EmpiricalThresholds:
        """
        Compute empirical percentile thresholds from derived_returns data.

        Parameters
        ----------
        derived_df : pd.DataFrame
            Output of DerivedReturnsCalculator.compute().

        Returns
        -------
        EmpiricalThresholds
        """
        symbol = derived_df["symbol"].iloc[0]

        abs_1h = derived_df["log_return_1h"].abs().dropna()
        abs_24h = derived_df["log_return_24h"].abs().dropna()
        abs_168h_dd = derived_df["rolling_drawdown_168h"].abs().dropna()

        if abs_1h.empty:
            raise ValueError(f"[{symbol}] No valid 1h returns available for threshold calibration.")

        def _pct(series: pd.Series) -> float:
            return (
                float(np.nanpercentile(series, _PERCENTILE_LEVEL)) if not series.empty else np.nan
            )

        return EmpiricalThresholds(
            symbol=symbol,
            hourly_return_p9975=_pct(abs_1h),
            daily_return_p9975=_pct(abs_24h),
            weekly_drawdown_p9975=_pct(abs_168h_dd),
            hourly_return_floor=_HOURLY_ABS_FLOOR,
            daily_return_floor=_DAILY_ABS_FLOOR,
            n_hourly_obs=len(abs_1h),
            n_daily_obs=len(abs_24h),
            n_weekly_obs=len(abs_168h_dd),
            data_start=pd.Timestamp(derived_df["timestamp"].min()),
            data_end=pd.Timestamp(derived_df["timestamp"].max()),
        )

    def flag(
        self,
        derived_df: pd.DataFrame,
        thresholds: EmpiricalThresholds,
        vol_series: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Apply thresholds to derived_df and return the threshold_flags table.

        Parameters
        ----------
        derived_df : pd.DataFrame
            Output of DerivedReturnsCalculator.compute(), with RangeIndex.
        thresholds : EmpiricalThresholds
            Fitted thresholds from fit().
        vol_series : pd.Series, optional
            Annualized rolling volatility aligned with derived_df rows (same
            RangeIndex). Pass the Layer 1 NaiveVolatility output's
            volatility_annualized column for best integration.
            If None, a basic 24h rolling std (annualized) is computed internally.

        Returns
        -------
        pd.DataFrame
            threshold_flags table sorted by (timestamp, horizon).
            event_window_id is "" for all rows — populated by EventWindower.

        Notes
        -----
        The percentile_rank column reflects each bar's rank in the FULL sample
        distribution (intentional full-sample calibration for dummy system).
        """
        derived_df = derived_df.reset_index(drop=True)

        if vol_series is None:
            raw_vol = derived_df["log_return_1h"].rolling(24, min_periods=2).std(ddof=1)
            vol_series = (raw_vol * np.sqrt(8766.0)).reset_index(drop=True)
        else:
            vol_series = pd.Series(vol_series.values, dtype=float)

        all_flags = [
            self._flag_1h(derived_df, thresholds, vol_series),
            self._flag_24h(derived_df, thresholds, vol_series),
            self._flag_168h(derived_df, thresholds, vol_series),
        ]

        non_empty = [f for f in all_flags if not f.empty]
        if not non_empty:
            return pd.DataFrame(columns=THRESHOLD_FLAGS_COLUMNS)

        result = (
            pd.concat(non_empty, ignore_index=True)
            .sort_values(["timestamp", "horizon"])
            .reset_index(drop=True)
        )
        return result

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pct_rank(abs_vals: pd.Series) -> pd.Series:
        """Full-sample empirical percentile rank in [0, 100]."""
        return abs_vals.rank(pct=True, na_option="keep") * 100

    def _build_flags(
        self,
        derived_df: pd.DataFrame,
        flag_mask: pd.Series,
        horizon: str,
        return_col: str,
        pct_rank_series: pd.Series,
        vol_series: pd.Series,
        thresh_labels: pd.Series,
        flag_label: str,
    ) -> pd.DataFrame:
        """Extract rows matching flag_mask and assemble the flag DataFrame."""
        flagged = derived_df[flag_mask].reset_index(drop=True)
        if flagged.empty:
            return pd.DataFrame(columns=THRESHOLD_FLAGS_COLUMNS)

        raw_ret = derived_df[return_col][flag_mask].reset_index(drop=True)
        pct_r = pct_rank_series[flag_mask].reset_index(drop=True)
        rolling_v = vol_series[flag_mask].reset_index(drop=True)
        drawdown = derived_df["rolling_drawdown_168h"][flag_mask].reset_index(drop=True)
        t_labels = thresh_labels[flag_mask].reset_index(drop=True)

        return pd.DataFrame(
            {
                "timestamp": flagged["timestamp"].reset_index(drop=True),
                "symbol": flagged["symbol"].values,
                "horizon": horizon,
                "raw_return": raw_ret.values,
                "abs_return": raw_ret.abs().values,
                "percentile_rank": pct_r.values,
                "rolling_volatility": rolling_v.values,
                "drawdown_size": drawdown.values,
                "threshold_breached": t_labels.values,
                "flag_label": flag_label,
                "event_window_id": "",
            }
        )

    def _flag_1h(
        self,
        derived_df: pd.DataFrame,
        thresholds: EmpiricalThresholds,
        vol_series: pd.Series,
    ) -> pd.DataFrame:
        abs_ret = derived_df["log_return_1h"].abs()
        pct_rank = self._pct_rank(abs_ret)

        pct_flag = abs_ret > thresholds.hourly_return_p9975
        floor_flag = abs_ret > thresholds.hourly_return_floor
        flag_mask = (pct_flag | floor_flag) & abs_ret.notna()

        if not flag_mask.any():
            return pd.DataFrame(columns=THRESHOLD_FLAGS_COLUMNS)

        # Precedence: percentile threshold over floor
        thresh_labels = pd.Series("RETURN_ABS_FLOOR", index=derived_df.index)
        thresh_labels[pct_flag] = "RETURN_PCT_9975"

        return self._build_flags(
            derived_df,
            flag_mask,
            "1h",
            "log_return_1h",
            pct_rank,
            vol_series,
            thresh_labels,
            f"{_LABEL_PREFIX}_1H_STRESS",
        )

    def _flag_24h(
        self,
        derived_df: pd.DataFrame,
        thresholds: EmpiricalThresholds,
        vol_series: pd.Series,
    ) -> pd.DataFrame:
        abs_ret = derived_df["log_return_24h"].abs()
        pct_rank = self._pct_rank(abs_ret)

        pct_flag = abs_ret > thresholds.daily_return_p9975
        floor_flag = abs_ret > thresholds.daily_return_floor
        flag_mask = (pct_flag | floor_flag) & abs_ret.notna()

        if not flag_mask.any():
            return pd.DataFrame(columns=THRESHOLD_FLAGS_COLUMNS)

        thresh_labels = pd.Series("RETURN_ABS_FLOOR", index=derived_df.index)
        thresh_labels[pct_flag] = "RETURN_PCT_9975"

        return self._build_flags(
            derived_df,
            flag_mask,
            "24h",
            "log_return_24h",
            pct_rank,
            vol_series,
            thresh_labels,
            f"{_LABEL_PREFIX}_24H_STRESS",
        )

    def _flag_168h(
        self,
        derived_df: pd.DataFrame,
        thresholds: EmpiricalThresholds,
        vol_series: pd.Series,
    ) -> pd.DataFrame:
        abs_dd = derived_df["rolling_drawdown_168h"].abs()
        pct_rank = self._pct_rank(abs_dd)

        pct_flag = abs_dd > thresholds.weekly_drawdown_p9975
        flag_mask = pct_flag & abs_dd.notna()

        if not flag_mask.any():
            return pd.DataFrame(columns=THRESHOLD_FLAGS_COLUMNS)

        thresh_labels = pd.Series("DRAWDOWN_PCT_9975", index=derived_df.index)

        return self._build_flags(
            derived_df,
            flag_mask,
            "168h",
            "rolling_drawdown_168h",
            pct_rank,
            vol_series,
            thresh_labels,
            f"{_LABEL_PREFIX}_168H_STRESS",
        )
