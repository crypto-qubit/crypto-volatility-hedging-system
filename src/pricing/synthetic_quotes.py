"""
Synthetic put-option chain generator for the offline demo.

Produces a small deterministic grid of OTM put quotes priced with the same
Black-Scholes primitives used by option_pricer.py, so find_target_put() has
something realistic to select from with zero network access and no exchange
account. Every value produced here is SYNTHETIC — it does not reflect any
real order book, and must not be treated as market data.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.pricing.option_pricer import OptionQuote, bs_put_delta, bs_put_price

# OTM strikes as a fraction of spot (0.95 = 5% OTM put)
_STRIKE_FRACTIONS = [0.97, 0.95, 0.92, 0.90, 0.85]
_EXPIRIES_DAYS = [7.0, 14.0, 30.0]
_SYNTHETIC_BID_ASK_SPREAD_PCT = 0.06  # 6% of mid, typical for a liquid OTM crypto put


def generate_put_chain(
    symbol: str,
    spot: float,
    annualized_vol: float,
    as_of: datetime | None = None,
) -> list[OptionQuote]:
    """
    Build a synthetic OTM put chain for `symbol` around `spot`.

    Parameters
    ----------
    symbol : str
        "BTC" or "ETH".
    spot : float
        Current (synthetic) spot price in USD.
    annualized_vol : float
        Annualized volatility (decimal) used to price every strike/expiry —
        normally the latest EWMA or Yang-Zhang reading for this symbol.
    as_of : datetime, optional
        Quote timestamp. Defaults to UTC now.

    Returns
    -------
    list[OptionQuote]
        SYNTHETIC quotes only — for demo/testing, not real market data.
    """
    as_of = as_of or datetime.now(timezone.utc)
    sigma = max(annualized_vol, 0.05)  # floor to avoid a degenerate/zero-vol chain

    quotes: list[OptionQuote] = []
    for expiry_days in _EXPIRIES_DAYS:
        expiry = as_of + timedelta(days=expiry_days)
        T = expiry_days / 365.0
        for frac in _STRIKE_FRACTIONS:
            strike = round(spot * frac, -1 if spot > 1000 else 0)
            mark_price_usd = bs_put_price(spot, strike, T, sigma)
            mark_price = (
                mark_price_usd / spot
            )  # underlying-denominated, matches exchange convention
            delta = bs_put_delta(spot, strike, T, sigma)
            mid = mark_price
            half_spread = mid * _SYNTHETIC_BID_ASK_SPREAD_PCT / 2.0

            quotes.append(
                OptionQuote(
                    instrument_name=f"{symbol}-{expiry:%d%b%y}-{int(strike)}-P".upper(),
                    symbol=symbol,
                    strike=strike,
                    expiry=expiry,
                    option_type="P",
                    bid=round(max(mid - half_spread, 0.0), 6),
                    ask=round(mid + half_spread, 6),
                    mark_price=round(mid, 6),
                    mark_iv=round(sigma, 4),
                    delta=round(delta, 4),
                    gamma=None,
                    theta=None,
                    vega=None,
                    underlying_price=spot,
                    updated_at=as_of,
                )
            )
    return quotes
