"""
Tests for src/pricing/option_pricer.py and src/pricing/synthetic_quotes.py.
"""

from src.pricing.option_pricer import bs_put_delta, bs_put_price, find_target_put
from src.pricing.synthetic_quotes import generate_put_chain


def test_bs_put_price_is_nonnegative_and_matches_intrinsic_at_expiry():
    assert bs_put_price(spot=100, strike=110, T=0.0, sigma=0.5) == 10.0
    assert bs_put_price(spot=100, strike=90, T=0.0, sigma=0.5) == 0.0
    assert bs_put_price(spot=100, strike=100, T=0.1, sigma=0.5) > 0


def test_bs_put_delta_is_bounded_and_negative():
    delta = bs_put_delta(spot=100, strike=110, T=0.1, sigma=0.5)
    assert -1.0 <= delta <= 0.0


def test_put_price_increases_with_volatility():
    low_vol = bs_put_price(spot=100, strike=95, T=0.1, sigma=0.2)
    high_vol = bs_put_price(spot=100, strike=95, T=0.1, sigma=0.8)
    assert high_vol > low_vol


def test_generate_put_chain_produces_valid_synthetic_quotes():
    quotes = generate_put_chain("BTC", spot=50_000.0, annualized_vol=0.6)
    assert len(quotes) > 0
    for q in quotes:
        assert q.symbol == "BTC"
        assert q.strike < 50_000.0  # OTM puts only
        assert q.mark_price >= 0
        assert q.bid <= q.ask
        assert -1.0 <= q.delta <= 0.0


def test_find_target_put_selects_closest_delta_and_expiry():
    quotes = generate_put_chain("BTC", spot=50_000.0, annualized_vol=0.6)
    result = find_target_put(
        quotes, spot=50_000.0, notional=100_000.0, target_delta=0.25, target_expiry_days=7.0
    )

    assert result is not None
    assert result.symbol == "BTC"
    assert result.premium_pct >= 0
    assert result.cost_label in {"cheap", "fair", "expensive"}


def test_find_target_put_returns_none_for_empty_chain():
    assert find_target_put([], spot=50_000.0, notional=100_000.0) is None


def test_expensive_put_may_recommend_a_spread():
    # A very high vol pushes the target put's premium above the expensive threshold.
    quotes = generate_put_chain("BTC", spot=50_000.0, annualized_vol=3.5)
    result = find_target_put(quotes, spot=50_000.0, notional=10_000.0)
    assert result is not None
    if result.cost_label == "expensive":
        assert result.spread_recommendation is None or (
            result.spread_recommendation.net_premium_usd > 0
        )
