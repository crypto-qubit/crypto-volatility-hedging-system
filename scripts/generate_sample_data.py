"""
Deterministic synthetic OHLCV generator for the offline demo.

Produces hourly BTC and ETH bars that are OBVIOUSLY SYNTHETIC (round starting
prices, documented seeds) with:
  - a long calm-volatility baseline period,
  - several injected volatility shocks (crash-like drawdowns with elevated
    intrabar range, followed by a recovery/decay back to baseline),
  - valid OHLC geometry on every bar (high >= max(o,c), low <= min(o,c)).

This is NOT real market data and NOT a calibrated model of BTC/ETH dynamics.
It exists solely so the rest of the pipeline (validation -> volatility
estimators -> thresholds -> regime -> decision) can run end-to-end with zero
network access. See docs/limitations.md for why this must never be read as a
backtest or a performance claim.

Usage
-----
    python -m scripts.generate_sample_data
    python -m scripts.generate_sample_data --days 365 --out-dir data/sample
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Documented seeds — change these (not the RNG calls) to get a different draw.
BTC_SEED = 42
ETH_SEED = 43

BTC_START_PRICE = 50_000.0  # round, obviously synthetic starting level
ETH_START_PRICE = 3_000.0

BASELINE_HOURLY_VOL = 0.006  # ~ calm-regime hourly return std (log units)
SHOCK_HOURLY_VOL = 0.028  # elevated hourly return std during a shock window

# (start_fraction_of_series, duration_hours, cumulative_drift) — injected shocks.
# drift is the approximate total log-return move over the shock window
# (negative = crash-like drawdown), applied on top of the elevated-vol noise.
SHOCK_WINDOWS = [
    (0.18, 36, -0.22),  # ~3 week mark: sharp 1.5-day liquidation-style drop
    (0.45, 60, -0.30),  # mid-series: longer, deeper stress event
    (0.72, 20, -0.15),  # later: shorter, sharper shock
    (0.88, 40, -0.18),  # near the end: second deep event
]
SHOCK_DECAY_HOURS = 30  # bars of elevated (decaying) vol after each shock window


def _vol_and_drift_path(n_hours: int, tail_shock: bool) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar (vol, drift) arrays: baseline everywhere, elevated inside shocks."""
    vol = np.full(n_hours, BASELINE_HOURLY_VOL)
    drift = np.zeros(n_hours)

    for start_frac, duration, cum_drift in SHOCK_WINDOWS:
        start = int(start_frac * n_hours)
        end = min(start + duration, n_hours)
        if start >= n_hours:
            continue
        vol[start:end] = SHOCK_HOURLY_VOL
        drift[start:end] = cum_drift / max(duration, 1)

        decay_end = min(end + SHOCK_DECAY_HOURS, n_hours)
        if decay_end > end:
            decay_len = decay_end - end
            decay = np.linspace(SHOCK_HOURLY_VOL, BASELINE_HOURLY_VOL, decay_len)
            vol[end:decay_end] = np.maximum(vol[end:decay_end], decay)

    if tail_shock:
        # Guarantees the series ends inside an active shock, so the "current"
        # regime read by the demo (trailing window ending at the last bar) is
        # stress and the decision layer has something live to react to —
        # without this, whether the run ends calm is a coin flip of the RNG.
        tail_start = n_hours - 45
        vol[tail_start:] = SHOCK_HOURLY_VOL
        drift[tail_start:] = -0.24 / 45

    return vol, drift


def _generate_symbol(
    symbol: str,
    start_price: float,
    n_hours: int,
    seed: int,
    start_date: str,
    tail_shock: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    vol, drift = _vol_and_drift_path(n_hours, tail_shock)

    log_returns = rng.normal(loc=drift, scale=vol)
    log_prices = np.log(start_price) + np.cumsum(log_returns)
    close = np.exp(log_prices)
    open_ = np.empty(n_hours)
    open_[0] = start_price
    open_[1:] = close[:-1]

    # Intrabar range scaled off the same per-bar vol so high-vol bars also get
    # wider (but still geometrically valid) wicks.
    intrabar_scale = vol * rng.uniform(0.6, 1.4, size=n_hours)
    upper_wick = np.abs(rng.normal(0.0, 1.0, size=n_hours)) * intrabar_scale
    lower_wick = np.abs(rng.normal(0.0, 1.0, size=n_hours)) * intrabar_scale

    bar_top = np.maximum(open_, close)
    bar_bottom = np.minimum(open_, close)
    high = bar_top * (1.0 + upper_wick)
    low = bar_bottom * (1.0 - lower_wick)
    low = np.minimum(low, bar_bottom)  # guard against float rounding pushing low above bottom
    high = np.maximum(high, bar_top)

    # Volume: baseline log-normal, amplified during shock windows (panic volume).
    base_volume = rng.lognormal(mean=np.log(500.0), sigma=0.35, size=n_hours)
    volume = base_volume * (1.0 + 6.0 * (vol - BASELINE_HOURLY_VOL) / BASELINE_HOURLY_VOL).clip(
        min=1.0
    )

    timestamps = pd.date_range(start=start_date, periods=n_hours, freq="h", tz="UTC")

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": symbol,
            "open": open_.round(2),
            "high": high.round(2),
            "low": low.round(2),
            "close": close.round(2),
            "volume": volume.round(4),
        }
    )


def generate_sample_data(days: int = 400, out_dir: Path | str = "data/sample") -> dict[str, Path]:
    """Generate and write BTC + ETH synthetic hourly OHLCV CSVs. Returns {symbol: path}."""
    n_hours = days * 24
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # BTC ends inside a live shock (demonstrates a HEDGE decision); ETH ends
    # calm (demonstrates WAIT) — deliberately different so the dashboard shows
    # contrast between the two assets rather than two identical outcomes.
    btc = _generate_symbol("BTC", BTC_START_PRICE, n_hours, BTC_SEED, "2024-01-01", tail_shock=True)
    eth = _generate_symbol(
        "ETH", ETH_START_PRICE, n_hours, ETH_SEED, "2024-01-01", tail_shock=False
    )

    paths = {
        "BTC": out_dir / "btc_hourly.csv",
        "ETH": out_dir / "eth_hourly.csv",
    }
    btc.to_csv(paths["BTC"], index=False)
    eth.to_csv(paths["ETH"], index=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days", type=int, default=400, help="Number of days of hourly bars to generate."
    )
    parser.add_argument(
        "--out-dir", type=str, default="data/sample", help="Output directory for CSVs."
    )
    args = parser.parse_args()

    paths = generate_sample_data(days=args.days, out_dir=args.out_dir)
    for symbol, path in paths.items():
        n_rows = sum(1 for _ in open(path)) - 1
        print(f"[generate_sample_data] {symbol}: {n_rows} hourly bars -> {path}")
    print(
        f"[generate_sample_data] Seeds: BTC={BTC_SEED}, ETH={ETH_SEED} "
        f"(deterministic — same seed always reproduces the same series). "
        "All prices are SYNTHETIC."
    )


if __name__ == "__main__":
    main()
