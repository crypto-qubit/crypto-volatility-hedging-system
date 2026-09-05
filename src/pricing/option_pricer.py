"""
Option pricer — Black-Scholes put pricing and best-fit instrument selection.

Stateless: takes OptionQuote snapshots (e.g. from a synthetic chain, see
src/pricing/synthetic_quotes.py for the offline demo generator) and returns
a PricingResult. When exchange-reported greeks are available they are used
directly; Black-Scholes serves as a fallback and for verification.

Conventions
-----------
- mark_price is in underlying units (BTC for BTC options).
  premium_usd = mark_price * underlying_price.
- mark_iv is already converted to decimal (0.835 = 83.5%).
- Risk-free rate is set to 0.0; crypto has no conventional risk-free rate.
- Delta for puts is negative; target_delta is specified as a positive absolute
  value (0.25 means the 25-delta put).

Put spread recommendation
-------------------------
When the protective put is classified "expensive", find_target_put also returns
a debit put spread alternative:
    - Long leg : the target ~0.25Δ put (higher strike, e.g. K=60 000)
    - Short leg: a further OTM put at ~0.10Δ (lower strike, e.g. K=55 000)
    Max payoff = (K_long - K_short) if spot collapses below K_short.
    Net premium = long_premium - short_premium (always positive since K_long > K_short).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

_RISK_FREE_RATE = 0.0
_EXPENSIVE_PREMIUM_PCT = 0.015  # > 1.5% of notional → expensive
_CHEAP_PREMIUM_PCT = 0.005  # < 0.5% of notional → cheap
_WIDE_SPREAD_PCT = 0.15  # bid/ask spread > 15% of mid → illiquid / expensive


# ---------------------------------------------------------------------------
# Option quote snapshot
# ---------------------------------------------------------------------------
#
# A minimal, exchange-agnostic quote type. In production this is populated by
# a live feed (e.g. a Deribit WebSocket client); here it is populated by the
# offline synthetic-chain generator so the pricer can run with no network
# access. Only the fields the pricer actually needs are included.


@dataclass
class OptionQuote:
    instrument_name: str
    symbol: str  # "BTC" or "ETH"
    strike: float
    expiry: datetime  # UTC
    option_type: str  # "C" or "P"
    bid: float | None
    ask: float | None
    mark_price: float | None  # in underlying units (BTC for BTC options)
    mark_iv: float | None  # annualized IV as decimal (0.835 = 83.5%)
    delta: float | None  # signed: negative for puts
    gamma: float | None
    theta: float | None
    vega: float | None
    underlying_price: float | None  # spot USD price of the underlying
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def days_to_expiry(self) -> float:
        remaining = (self.expiry - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, remaining / 86400.0)

    @property
    def is_expired(self) -> bool:
        return self.days_to_expiry <= 0.0

    @property
    def bid_ask_spread_pct(self) -> float | None:
        if self.bid is None or self.ask is None or self.ask <= 0:
            return None
        return (self.ask - self.bid) / self.ask


# ---------------------------------------------------------------------------
# Black-Scholes primitives (no external deps — uses math.erfc from stdlib)
# ---------------------------------------------------------------------------


def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def bs_put_price(
    spot: float,
    strike: float,
    T: float,  # years to expiry
    sigma: float,  # annualized vol as decimal
    r: float = _RISK_FREE_RATE,
) -> float:
    """European put price via Black-Scholes. Returns intrinsic value when T <= 0."""
    if T <= 0.0 or sigma <= 0.0:
        return max(strike - spot, 0.0)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return strike * math.exp(-r * T) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def bs_put_delta(
    spot: float,
    strike: float,
    T: float,
    sigma: float,
    r: float = _RISK_FREE_RATE,
) -> float:
    """Black-Scholes put delta. Returns a negative value in [-1, 0]."""
    if T <= 0.0 or sigma <= 0.0:
        return -1.0 if spot < strike else 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1) - 1.0


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class SpreadLeg:
    instrument_name: str
    strike: float
    delta: float
    premium_usd: float
    premium_pct: float


@dataclass
class SpreadRecommendation:
    """
    Debit put spread: buy the target OTM put, sell a further OTM put to
    reduce net premium at the cost of capping the maximum payoff.
    """

    long_leg: SpreadLeg  # buy: ~0.25Δ put (higher strike, primary hedge)
    short_leg: SpreadLeg  # sell: ~0.10Δ put (lower strike, premium offset)
    net_premium_usd: float
    net_premium_pct: float
    max_payoff_pct: float  # (K_long - K_short) / spot


@dataclass
class PricingResult:
    instrument_name: str
    symbol: str
    spot: float
    strike: float
    expiry: datetime
    days_to_expiry: float
    delta: float  # signed, negative for puts
    iv: float  # annualized decimal
    mark_price_usd: float  # premium in USD
    premium_pct: float  # mark_price_usd / notional
    cost_label: str  # "cheap" | "fair" | "expensive"
    bid_ask_spread_pct: float | None
    notional: float
    spread_recommendation: SpreadRecommendation | None = None


# ---------------------------------------------------------------------------
# Instrument selection
# ---------------------------------------------------------------------------


def _score(
    quote: OptionQuote,
    target_delta_abs: float,
    target_expiry_days: float,
) -> float:
    """Lower score = better fit. Delta match weighted 2x over expiry proximity."""
    delta_abs = abs(quote.delta) if quote.delta is not None else 0.5
    delta_err = abs(delta_abs - target_delta_abs)
    expiry_err = abs(quote.days_to_expiry - target_expiry_days) / max(target_expiry_days, 1.0)
    return delta_err * 2.0 + expiry_err


def _effective_delta(quote: OptionQuote, spot: float) -> float:
    """Return the exchange-reported delta if available, else a Black-Scholes estimate."""
    if quote.delta is not None:
        return quote.delta
    T = quote.days_to_expiry / 365.0
    sigma = quote.mark_iv if quote.mark_iv is not None else 0.80
    return bs_put_delta(spot, quote.strike, T, sigma)


def _cost_label(premium_pct: float, bid_ask_spread_pct: float | None) -> str:
    spread_wide = bid_ask_spread_pct is not None and bid_ask_spread_pct > _WIDE_SPREAD_PCT
    if premium_pct > _EXPENSIVE_PREMIUM_PCT or spread_wide:
        return "expensive"
    if premium_pct < _CHEAP_PREMIUM_PCT and not spread_wide:
        return "cheap"
    return "fair"


def _build_spread(
    long_quote: OptionQuote,
    candidates: list[OptionQuote],
    spot: float,
    notional: float,
    underlying: float,
    short_delta_target: float = 0.10,
) -> SpreadRecommendation | None:
    """
    Find a short leg for a debit put spread. Short leg must be:
    - Same symbol and same expiry (within 1 hour of long leg).
    - Lower strike than the long leg (further OTM for puts).
    - Have a valid mark_price.
    """
    same_expiry = [
        q
        for q in candidates
        if q.symbol == long_quote.symbol
        and q.strike < long_quote.strike
        and q.mark_price is not None
        and abs((q.expiry - long_quote.expiry).total_seconds()) < 3600.0
    ]
    if not same_expiry:
        return None

    short_quote = min(
        same_expiry,
        key=lambda q: abs(abs(q.delta or 0.0) - short_delta_target),
    )

    long_prem_usd = (long_quote.mark_price or 0.0) * underlying
    short_prem_usd = (short_quote.mark_price or 0.0) * underlying
    net_usd = long_prem_usd - short_prem_usd
    if net_usd <= 0:
        return None  # degenerate: short leg more expensive than long (data error)

    long_leg = SpreadLeg(
        instrument_name=long_quote.instrument_name,
        strike=long_quote.strike,
        delta=_effective_delta(long_quote, spot),
        premium_usd=long_prem_usd,
        premium_pct=long_prem_usd / notional if notional > 0 else 0.0,
    )
    short_leg = SpreadLeg(
        instrument_name=short_quote.instrument_name,
        strike=short_quote.strike,
        delta=_effective_delta(short_quote, spot),
        premium_usd=short_prem_usd,
        premium_pct=short_prem_usd / notional if notional > 0 else 0.0,
    )

    return SpreadRecommendation(
        long_leg=long_leg,
        short_leg=short_leg,
        net_premium_usd=net_usd,
        net_premium_pct=net_usd / notional if notional > 0 else 0.0,
        max_payoff_pct=(long_quote.strike - short_quote.strike) / spot,
    )


def find_target_put(
    quotes: list[OptionQuote],
    spot: float,
    notional: float,
    target_delta: float = 0.25,
    target_expiry_days: float = 7.0,
    max_expiry_days: float = 14.0,
) -> PricingResult | None:
    """
    Find the closest available put to (target_delta, target_expiry_days).

    Parameters
    ----------
    quotes : list[OptionQuote]
        Active puts (e.g. from generate_put_chain()).
    spot : float
        Current USD spot price (used to convert mark_price to USD).
    notional : float
        Portfolio exposure in USD. Used to compute premium_pct.
    target_delta : float
        Absolute value of the target delta (0.25 = 25-delta OTM put).
    target_expiry_days : float
        Preferred days to expiry. Instruments are scored by proximity.
    max_expiry_days : float
        Hard cap on days to expiry. Instruments beyond this are excluded.

    Returns
    -------
    PricingResult | None
        Best-fit instrument, or None if no valid quotes are available.
    """
    candidates = [
        q
        for q in quotes
        if not q.is_expired
        and q.days_to_expiry <= max_expiry_days
        and q.mark_price is not None
        and q.underlying_price is not None
    ]
    if not candidates:
        return None

    best = min(candidates, key=lambda q: _score(q, target_delta, target_expiry_days))
    underlying = best.underlying_price or spot
    mark_price_usd = (best.mark_price or 0.0) * underlying
    premium_pct = mark_price_usd / notional if notional > 0 else 0.0
    delta = _effective_delta(best, underlying)
    iv = best.mark_iv if best.mark_iv is not None else 0.0
    label = _cost_label(premium_pct, best.bid_ask_spread_pct)

    spread = None
    if label == "expensive":
        spread = _build_spread(best, candidates, spot, notional, underlying)

    return PricingResult(
        instrument_name=best.instrument_name,
        symbol=best.symbol,
        spot=spot,
        strike=best.strike,
        expiry=best.expiry,
        days_to_expiry=best.days_to_expiry,
        delta=delta,
        iv=iv,
        mark_price_usd=mark_price_usd,
        premium_pct=premium_pct,
        cost_label=label,
        bid_ask_spread_pct=best.bid_ask_spread_pct,
        notional=notional,
        spread_recommendation=spread,
    )
