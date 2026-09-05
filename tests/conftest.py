"""
Shared fixtures for the test suite.
"""

import numpy as np
import pandas as pd
import pytest


def make_ohlcv(
    n: int = 300,
    symbol: str = "BTC",
    base_price: float = 30_000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic hourly OHLCV data with valid geometry for tests."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0, 0.01, size=n)
    log_prices = np.cumsum(log_returns) + np.log(base_price)
    close = np.exp(log_prices)

    noise = rng.uniform(0.001, 0.005, size=n)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(close, open_) * (1 + noise)
    low = np.minimum(close, open_) * (1 - noise)

    timestamps = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(100, 10_000, size=n),
            "symbol": symbol,
        }
    )


@pytest.fixture
def btc_df():
    return make_ohlcv(n=300, symbol="BTC")


@pytest.fixture
def eth_df():
    return make_ohlcv(n=300, symbol="ETH", base_price=2_000.0)
