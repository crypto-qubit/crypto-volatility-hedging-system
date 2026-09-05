"""
src/validation/integrity.py — OHLC integrity validation layer (v2, hardened).

Independent OHLC data-quality audit, decoupled from the ingestion path so it
can be run against any DataFrame without importing pipeline internals.

Audit fixes applied (v1 → v2)
──────────────────────────────
FAIL  Float Precision Trap       → epsilon-aware comparisons throughout;
                                   configurable per call (default 1e-8).
RISK  Severity Misclassification → full candle-geometry FATAL set:
                                   high >= max(O,C) − ε
                                   low  <= min(O,C) + ε
                                   high >= low − ε
RISK  Silent Failure             → mandatory structured logging; FATALs
                                   are always emitted to module logger even
                                   if caller discards the return value.
RISK  Stale Detection Edge Case  → len(df) < threshold short-circuit added
                                   to both anomaly checks.
RISK  Base-Rate Challenge        → addressed in docstring (opt-in per caller).
PASS  All-or-Nothing Regression  → validate_all_assets() unchanged.

Immutable design contract
──────────────────────────
• Never raises inside the validator — accumulates and returns.
• Never modifies the input DataFrame.
• Fully JSON-serialisable output.
• Caller owns the halt-on-FATAL decision.
• FATALs are always logged even when return value is ignored.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# Severity taxonomy
# ══════════════════════════════════════════════


class Severity(str, Enum):
    """
    FATAL          — structurally corrupt; downstream consumers must NOT run.
    WARNING        — suspicious; log and proceed with care.
    MARKET_ANOMALY — technically valid but statistically abnormal.
    """

    FATAL = "FATAL"
    WARNING = "WARNING"
    MARKET_ANOMALY = "MARKET_ANOMALY"


# ══════════════════════════════════════════════
# Result primitives
# ══════════════════════════════════════════════


@dataclass
class CheckResult:
    check_id: str
    severity: Severity
    message: str
    affected_rows: list[int]  # iloc positions; [] if not row-specific
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "severity": self.severity.value,
            "message": self.message,
            "affected_rows": self.affected_rows,
            "metadata": self.metadata,
        }


@dataclass
class ValidationReport:
    asset: str
    row_count: int
    epsilon: float
    results: list[CheckResult] = field(default_factory=list)

    @property
    def has_fatal(self) -> bool:
        return any(r.severity == Severity.FATAL for r in self.results)

    @property
    def has_warning(self) -> bool:
        return any(r.severity == Severity.WARNING for r in self.results)

    @property
    def has_market_anomaly(self) -> bool:
        return any(r.severity == Severity.MARKET_ANOMALY for r in self.results)

    @property
    def is_clean(self) -> bool:
        return len(self.results) == 0

    def fatals(self) -> list[CheckResult]:
        return [r for r in self.results if r.severity == Severity.FATAL]

    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if r.severity == Severity.WARNING]

    def anomalies(self) -> list[CheckResult]:
        return [r for r in self.results if r.severity == Severity.MARKET_ANOMALY]

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset": self.asset,
            "row_count": self.row_count,
            "epsilon": self.epsilon,
            "has_fatal": self.has_fatal,
            "has_warning": self.has_warning,
            "has_market_anomaly": self.has_market_anomaly,
            "is_clean": self.is_clean,
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ══════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════


def validate_ohlc_integrity(
    df: pd.DataFrame,
    asset: str,
    consecutive_flat_threshold: int = 6,
    volume_spike_multiplier: float = 10.0,
    expected_freq: str | None = None,
    epsilon: float = 1e-8,
) -> ValidationReport:
    """
    Run all OHLC integrity checks and return a ValidationReport.

    Silent-failure contract
    ───────────────────────
    FATALs are emitted to the module logger at ERROR level before this
    function returns, regardless of whether the caller uses the return value.

    Epsilon policy
    ──────────────
    All OHLC geometry comparisons use epsilon to absorb float serialisation
    drift from exchange feeds, parquet/CSV round-trips, and vendor rounding.
    Set epsilon=0.0 only for synthetic/test data with guaranteed exact values.
    Use epsilon=1e-5 for assets with known vendor rounding (some FX feeds).

    Stale detection edge case
    ─────────────────────────
    When len(df) < consecutive_flat_threshold the anomaly checks are
    short-circuited to avoid misleading results on thin datasets.

    Base-rate note
    ──────────────
    This layer is opt-in per caller.  For manual single-asset pipelines
    the overhead is negligible; for automated multi-asset or ML-consumer
    pipelines it is mandatory.
    """
    report = ValidationReport(asset=asset, row_count=len(df), epsilon=epsilon)

    if df.empty:
        _emit(
            report,
            CheckResult(
                check_id="FATAL-EMPTY-DATAFRAME",
                severity=Severity.FATAL,
                message=f"[{asset}] DataFrame is empty — no OHLC data to validate.",
                affected_rows=[],
            ),
        )
        _log_fatal_summary(report)
        return report

    # FATAL
    for r in _check_fatal_nulls(df, asset):
        _emit(report, r)
    for r in _check_fatal_price_positive(df, asset):
        _emit(report, r)
    for r in _check_fatal_candle_geometry(df, asset, epsilon):
        _emit(report, r)

    # WARNING
    for r in _check_warning_volume(df, asset, volume_spike_multiplier):
        _emit(report, r)
    if expected_freq is not None:
        for r in _check_warning_gaps(df, asset, expected_freq):
            _emit(report, r)

    # MARKET_ANOMALY
    for r in _check_market_anomaly_stale(df, asset, consecutive_flat_threshold, epsilon):
        _emit(report, r)
    for r in _check_market_anomaly_flat_streaks(df, asset, consecutive_flat_threshold, epsilon):
        _emit(report, r)

    _log_fatal_summary(report)
    return report


# ══════════════════════════════════════════════
# Mandatory observability
# ══════════════════════════════════════════════


def _emit(report: ValidationReport, result: CheckResult) -> None:
    """Append result; immediately log if FATAL."""
    report.results.append(result)
    if result.severity == Severity.FATAL:
        _log.error(
            "OHLC_FATAL | asset=%s | check=%s | rows=%d | msg=%s",
            report.asset,
            result.check_id,
            len(result.affected_rows),
            result.message,
        )


def _log_fatal_summary(report: ValidationReport) -> None:
    """
    Emit a summary ERROR log whenever any FATALs are present.

    Called unconditionally at function exit — this is the silent-discard
    defence:  validate_ohlc_integrity(df, asset)  # no assignment → still logged.
    """
    if report.has_fatal:
        ids = [r.check_id for r in report.fatals()]
        _log.error(
            "OHLC_VALIDATION_SUMMARY | asset=%s | fatal_count=%d | checks=%s",
            report.asset,
            len(ids),
            ids,
        )


# ══════════════════════════════════════════════
# FATAL checks
# ══════════════════════════════════════════════

_OHLC_COLS = ("open", "high", "low", "close")


def _check_fatal_nulls(df: pd.DataFrame, asset: str) -> list[CheckResult]:
    results: list[CheckResult] = []
    for col in _OHLC_COLS:
        if col not in df.columns:
            results.append(
                CheckResult(
                    check_id=f"FATAL-MISSING-COL-{col.upper()}",
                    severity=Severity.FATAL,
                    message=f"[{asset}] Required column '{col}' is absent.",
                    affected_rows=[],
                )
            )
            continue
        null_mask = df[col].isna()
        if null_mask.any():
            results.append(
                CheckResult(
                    check_id=f"FATAL-NULL-{col.upper()}",
                    severity=Severity.FATAL,
                    message=f"[{asset}] '{col}' contains {int(null_mask.sum())} null value(s).",
                    affected_rows=_to_iloc_list(df, null_mask),
                    metadata={"null_count": int(null_mask.sum())},
                )
            )
    return results


def _check_fatal_price_positive(df: pd.DataFrame, asset: str) -> list[CheckResult]:
    """
    Price <= 0 is always FATAL regardless of epsilon.

    Epsilon is intentionally NOT applied: a price of 1e-12 is not float
    rounding — it is a dead feed or a data error.
    """
    results: list[CheckResult] = []
    missing = [c for c in _OHLC_COLS if c not in df.columns]
    if missing:
        return results
    for col in _OHLC_COLS:
        try:
            numeric = pd.to_numeric(df[col], errors="coerce")
            mask = numeric.notna() & (numeric <= 0)
        except Exception:
            mask = pd.Series(False, index=df.index)
        if mask.any():
            results.append(
                CheckResult(
                    check_id=f"FATAL-NONPOSITIVE-{col.upper()}",
                    severity=Severity.FATAL,
                    message=(
                        f"[{asset}] '{col}' has {int(mask.sum())} value(s) ≤ 0 "
                        f"(min={_safe_min(df[col], mask)})."
                    ),
                    affected_rows=_to_iloc_list(df, mask),
                    metadata={"count": int(mask.sum())},
                )
            )
    return results


def _check_fatal_candle_geometry(df: pd.DataFrame, asset: str, epsilon: float) -> list[CheckResult]:
    """
    Full epsilon-aware candle geometry invariants.

    All five violations are FATAL — they represent structurally impossible
    price bars that would corrupt any downstream model.

    Invariants (with epsilon tolerance for float serialisation drift):
      1. high  ≥ low   − ε
      2. high  ≥ open  − ε
      3. high  ≥ close − ε
      4. low   ≤ open  + ε
      5. low   ≤ close + ε

    Invariant 1 subsumes the original H < L check.
    Invariants 2-5 are the four geometry violations missing in v1.
    """
    results: list[CheckResult] = []
    missing = [c for c in _OHLC_COLS if c not in df.columns]
    if missing:
        return results

    try:
        H = pd.to_numeric(df["high"], errors="coerce")
        L = pd.to_numeric(df["low"], errors="coerce")
        O = pd.to_numeric(df["open"], errors="coerce")
        C = pd.to_numeric(df["close"], errors="coerce")
    except Exception:
        return results

    spec: list[tuple[str, pd.Series, str]] = [
        (
            "FATAL-HIGH-BELOW-LOW",
            H.notna() & L.notna() & (H < L - epsilon),
            f"[{asset}] High < Low − ε({{n}} bar(s)) — structurally impossible candle.",
        ),
        (
            "FATAL-HIGH-BELOW-OPEN",
            H.notna() & O.notna() & (H < O - epsilon),
            f"[{asset}] High < Open − ε ({{n}} bar(s)) — open above high.",
        ),
        (
            "FATAL-HIGH-BELOW-CLOSE",
            H.notna() & C.notna() & (H < C - epsilon),
            f"[{asset}] High < Close − ε ({{n}} bar(s)) — close above high.",
        ),
        (
            "FATAL-LOW-ABOVE-OPEN",
            L.notna() & O.notna() & (L > O + epsilon),
            f"[{asset}] Low > Open + ε ({{n}} bar(s)) — open below low.",
        ),
        (
            "FATAL-LOW-ABOVE-CLOSE",
            L.notna() & C.notna() & (L > C + epsilon),
            f"[{asset}] Low > Close + ε ({{n}} bar(s)) — close below low.",
        ),
    ]

    for check_id, mask, msg_tmpl in spec:
        if mask.any():
            n = int(mask.sum())
            results.append(
                CheckResult(
                    check_id=check_id,
                    severity=Severity.FATAL,
                    message=msg_tmpl.format(n=n),
                    affected_rows=_to_iloc_list(df, mask),
                    metadata={"count": n, "epsilon": epsilon},
                )
            )

    return results


# ══════════════════════════════════════════════
# WARNING checks
# ══════════════════════════════════════════════


def _check_warning_gaps(df: pd.DataFrame, asset: str, expected_freq: str) -> list[CheckResult]:
    results: list[CheckResult] = []
    ts = _extract_timestamps(df)
    if ts is None or len(ts) < 2:
        return results

    try:
        # pd.Timedelta accepts '1h', '60min', '1H', etc. directly — no .nanos needed.
        expected_td = pd.Timedelta(expected_freq)
    except Exception:
        return results

    diffs = ts.diff().dropna()
    gap_mask = diffs > expected_td * 1.5
    if gap_mask.any():
        gap_on_df = gap_mask.reindex(df.index, fill_value=False)
        results.append(
            CheckResult(
                check_id="WARNING-TIMESTAMP-GAP",
                severity=Severity.WARNING,
                message=(
                    f"[{asset}] {int(gap_mask.sum())} gap(s) > 1.5× {expected_freq} "
                    f"(max={diffs[gap_mask].max()})."
                ),
                affected_rows=_to_iloc_list(df, gap_on_df),
                metadata={
                    "gap_count": int(gap_mask.sum()),
                    "max_gap_seconds": diffs[gap_mask].max().total_seconds(),
                    "expected_freq": expected_freq,
                },
            )
        )
    return results


def _check_warning_volume(
    df: pd.DataFrame, asset: str, spike_multiplier: float
) -> list[CheckResult]:
    results: list[CheckResult] = []
    if "volume" not in df.columns:
        return results
    try:
        vol = pd.to_numeric(df["volume"], errors="coerce")
    except Exception:
        return results

    zero_mask = vol.notna() & (vol == 0)
    if zero_mask.any():
        results.append(
            CheckResult(
                check_id="WARNING-ZERO-VOLUME",
                severity=Severity.WARNING,
                message=f"[{asset}] {int(zero_mask.sum())} bar(s) have volume = 0.",
                affected_rows=_to_iloc_list(df, zero_mask),
                metadata={"zero_count": int(zero_mask.sum())},
            )
        )

    rolling_med = vol.rolling(window=20, min_periods=5).median()
    spike_mask = vol.notna() & rolling_med.notna() & (vol > rolling_med * spike_multiplier)
    if spike_mask.any():
        results.append(
            CheckResult(
                check_id="WARNING-VOLUME-SPIKE",
                severity=Severity.WARNING,
                message=(
                    f"[{asset}] {int(spike_mask.sum())} bar(s) exceed "
                    f"{spike_multiplier}× rolling 20-bar median volume."
                ),
                affected_rows=_to_iloc_list(df, spike_mask),
                metadata={"spike_count": int(spike_mask.sum()), "multiplier": spike_multiplier},
            )
        )
    return results


# ══════════════════════════════════════════════
# MARKET_ANOMALY checks
# ══════════════════════════════════════════════


def _check_market_anomaly_stale(
    df: pd.DataFrame, asset: str, threshold: int, epsilon: float
) -> list[CheckResult]:
    """
    O≈H≈L≈C for ≥ threshold consecutive bars → stale/frozen feed.

    Edge-case: short-circuit when len(df) < threshold (no false positives).
    Epsilon: price spread across all four values ≤ epsilon → treat as stale.
    """
    results: list[CheckResult] = []
    missing = [c for c in _OHLC_COLS if c not in df.columns]
    if missing or len(df) < threshold:
        return results

    try:
        prices = pd.DataFrame(
            {
                "H": pd.to_numeric(df["high"], errors="coerce"),
                "L": pd.to_numeric(df["low"], errors="coerce"),
                "O": pd.to_numeric(df["open"], errors="coerce"),
                "C": pd.to_numeric(df["close"], errors="coerce"),
            }
        )
        spread = prices.max(axis=1) - prices.min(axis=1)
        all_equal = spread.notna() & (spread <= epsilon)
    except Exception:
        return results

    streaks = _find_consecutive_streaks(all_equal, threshold)
    if streaks:
        total = sum(e - s + 1 for s, e in streaks)
        results.append(
            CheckResult(
                check_id="MARKET_ANOMALY-STALE-OHLC",
                severity=Severity.MARKET_ANOMALY,
                message=(
                    f"[{asset}] O≈H≈L≈C (spread ≤ {epsilon}) for {total} bar(s) "
                    f"across {len(streaks)} streak(s) of ≥{threshold} bars — likely stale feed."
                ),
                affected_rows=[i for s, e in streaks for i in range(s, e + 1)],
                metadata={
                    "streak_count": len(streaks),
                    "threshold": threshold,
                    "epsilon": epsilon,
                    "streaks": [{"start": s, "end": e} for s, e in streaks],
                },
            )
        )
    return results


def _check_market_anomaly_flat_streaks(
    df: pd.DataFrame, asset: str, threshold: int, epsilon: float
) -> list[CheckResult]:
    """
    Weaker stale signals:
      • Close unchanged (|Δclose| ≤ ε) for ≥ threshold consecutive bars.
      • |High − Low| ≤ ε for ≥ threshold consecutive bars (zero-range).

    Edge-case: flat-close needs at least threshold+1 rows (shift drops one).
               zero-range needs at least threshold rows.
    """
    results: list[CheckResult] = []

    # flat close
    if "close" in df.columns and len(df) >= threshold + 1:
        try:
            close = pd.to_numeric(df["close"], errors="coerce")
            unchanged = close.notna() & ((close - close.shift(1)).abs() <= epsilon)
            streaks = _find_consecutive_streaks(unchanged, threshold)
            if streaks:
                total = sum(e - s + 1 for s, e in streaks)
                results.append(
                    CheckResult(
                        check_id="MARKET_ANOMALY-FLAT-CLOSE",
                        severity=Severity.MARKET_ANOMALY,
                        message=(
                            f"[{asset}] Close unchanged (|Δ| ≤ {epsilon}) for {total} bar(s) "
                            f"across {len(streaks)} streak(s) of ≥{threshold} bars."
                        ),
                        affected_rows=[i for s, e in streaks for i in range(s, e + 1)],
                        metadata={"streak_count": len(streaks), "threshold": threshold},
                    )
                )
        except Exception:
            pass

    # zero-range
    if "high" in df.columns and "low" in df.columns and len(df) >= threshold:
        try:
            H = pd.to_numeric(df["high"], errors="coerce")
            L = pd.to_numeric(df["low"], errors="coerce")
            zero_rng = H.notna() & L.notna() & ((H - L).abs() <= epsilon)
            streaks = _find_consecutive_streaks(zero_rng, threshold)
            if streaks:
                total = sum(e - s + 1 for s, e in streaks)
                results.append(
                    CheckResult(
                        check_id="MARKET_ANOMALY-ZERO-RANGE",
                        severity=Severity.MARKET_ANOMALY,
                        message=(
                            f"[{asset}] |High − Low| ≤ {epsilon} (zero-range) for {total} bar(s) "
                            f"across {len(streaks)} streak(s) of ≥{threshold} bars."
                        ),
                        affected_rows=[i for s, e in streaks for i in range(s, e + 1)],
                        metadata={"streak_count": len(streaks), "threshold": threshold},
                    )
                )
        except Exception:
            pass

    return results


# ══════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════


def _find_consecutive_streaks(series: pd.Series, threshold: int) -> list[tuple[int, int]]:
    if len(series) == 0:
        return []
    arr = series.to_numpy(dtype=bool)
    n, streaks, i = len(arr), [], 0
    while i < n:
        if arr[i]:
            j = i
            while j < n and arr[j]:
                j += 1
            if (j - i) >= threshold:
                streaks.append((i, j - 1))
            i = j
        else:
            i += 1
    return streaks


def _to_iloc_list(df: pd.DataFrame, mask: pd.Series) -> list[int]:
    return [int(p) for p in np.where(mask.to_numpy())[0]][:100]


def _extract_timestamps(df: pd.DataFrame) -> pd.Series | None:
    """
    Return timestamps as a Series regardless of whether they are stored as a
    DatetimeIndex or in a named column ('open_time' or 'timestamp').
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index.to_series()
    for col in ("open_time", "timestamp"):
        if col in df.columns:
            try:
                return pd.to_datetime(df[col])
            except Exception:
                return None
    return None


def _safe_min(series: pd.Series, mask: pd.Series) -> str:
    try:
        return f"{pd.to_numeric(series[mask], errors='coerce').min():.8g}"
    except Exception:
        return "unknown"


# ══════════════════════════════════════════════
# Batch entry point
# ══════════════════════════════════════════════


def validate_all_assets(
    asset_frames: dict[str, pd.DataFrame],
    consecutive_flat_threshold: int = 6,
    volume_spike_multiplier: float = 10.0,
    expected_freq: str | None = None,
    epsilon: float = 1e-8,
) -> dict[str, ValidationReport]:
    """
    Validate multiple assets independently.

    One failing asset does NOT abort others (fixes pipeline.py GAP-3/GAP-4).
    Each asset gets its own fully-populated ValidationReport.
    """
    return {
        asset: validate_ohlc_integrity(
            df=df,
            asset=asset,
            consecutive_flat_threshold=consecutive_flat_threshold,
            volume_spike_multiplier=volume_spike_multiplier,
            expected_freq=expected_freq,
            epsilon=epsilon,
        )
        for asset, df in asset_frames.items()
    }
