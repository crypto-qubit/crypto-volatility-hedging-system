"""
Directional hedge signal helpers.

This module is the decision-boundary adapter between raw regime output and
hedge policy. It centralizes the 5-day direction classification and confidence
overlay so callers do not hand-build direction state before policy evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

VALID_REGIMES = {"low_vol", "normal", "high_vol", "stress"}
VALID_DIRECTIONS = {"RISING", "FALLING", "NEUTRAL"}

DIRECTION_DEAD_ZONE_PCT = 2.5
DIRECTION_HIGH_CONF_PCT = 5.0
STRESS_RISING_STEPDOWN_PCT = 10.0

DIRECTION_CONF_LOW = 0.50
DIRECTION_CONF_MEDIUM = 0.68
DIRECTION_CONF_HIGH = 0.80


@dataclass(frozen=True)
class DirectionalHedgeSignal:
    volatility_regime: str
    direction_state: str
    confidence: float


def pct_return_from_log_returns(log_returns: list[float], periods: int = 5) -> float | None:
    """Convert the latest `periods` log returns into cumulative simple percent."""
    if len(log_returns) < periods:
        return None
    return (math.exp(sum(log_returns[-periods:])) - 1.0) * 100.0


def classify_direction(pct_5d: float) -> str:
    """
    Classify a 5-day simple percent return as RISING, FALLING, or NEUTRAL.

    NEUTRAL is the dead zone where the price move is directionally
    uninformative for hedge sizing.
    """
    if abs(pct_5d) < DIRECTION_DEAD_ZONE_PCT:
        return "NEUTRAL"
    return "RISING" if pct_5d > 0.0 else "FALLING"


def direction_confidence(pct_5d: float, volatility_regime: str) -> float:
    """Return direction confidence in the shared 0.50 / 0.68 / 0.80 bands."""
    abs_pct = abs(pct_5d)

    if abs_pct >= DIRECTION_HIGH_CONF_PCT:
        confidence = DIRECTION_CONF_HIGH
    elif abs_pct >= DIRECTION_DEAD_ZONE_PCT:
        confidence = DIRECTION_CONF_MEDIUM
    else:
        confidence = DIRECTION_CONF_LOW

    if (
        volatility_regime == "stress"
        and pct_5d > STRESS_RISING_STEPDOWN_PCT
        and confidence == DIRECTION_CONF_HIGH
    ):
        confidence = DIRECTION_CONF_MEDIUM

    return confidence


def combine_confidence(
    regime_confidence: float | None,
    dir_confidence: float,
) -> float:
    """Use the lower confidence from regime classification and direction."""
    if regime_confidence is None:
        return dir_confidence
    return min(regime_confidence, dir_confidence)


def build_directional_hedge_signal(
    volatility_regime: str,
    pct_5d_return: float,
    regime_confidence: float | None = None,
) -> DirectionalHedgeSignal:
    """Build directional hedge state from raw regime and recent price return."""
    if volatility_regime not in VALID_REGIMES:
        raise ValueError(
            f"Invalid volatility_regime '{volatility_regime}'. "
            f"Must be one of {sorted(VALID_REGIMES)}."
        )

    direction = classify_direction(pct_5d_return)
    dir_conf = direction_confidence(pct_5d_return, volatility_regime)
    final_confidence = combine_confidence(regime_confidence, dir_conf)

    return DirectionalHedgeSignal(
        volatility_regime=volatility_regime,
        direction_state=direction,
        confidence=final_confidence,
    )
