"""
Spike-exit framework — option-value-driven, time-constrained exit state
machine for an *open* hedge position.

Complements the entry-side decision engine (`src/decision/engine.py`, which
answers "should we open a HEDGE now") with the exit-side question: "given a
hedge we already hold, should we take profit, trim, or roll it today." The
two are deliberately separate pure functions with separate state — this
module never reads or writes `src/decision/store.py`'s persisted HedgeState;
a caller wires the two together (see `src/api/main.py::exit_check`).

Six-rule evaluation order, first match wins:
  1. Effective peak update (folds today's value into the running peak).
  2. Minimum holding period — no exit fires before `min_holding_days`.
  3. 72-hour explosive tiers — sell 30/50/75/100% at 1.5x/2.0x/2.3x/3.0x
     the entry value, while `days_held <= 3`.
  4. Post-spike trailing stop — full exit on a `post_spike_trailing_stop_pct`
     drawdown from a peak that ever exceeded 1.5x entry.
  5. Convexity spent — sell 50% and roll when delta is deep ITM or the
     option's value is mostly intrinsic (time value exhausted).
  6. Hard time cap — full exit after `max_holding_days` if the position
     hasn't made a new high recently (current < 50% of peak).
  Default: HOLD.

`SpikeHedgeState` is intentionally distinct from `engine.HedgeState` — this
module tracks only what the exit rules need (entry value, running peak, days
held), not the full position record (instrument, strike, notional, ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------


@dataclass
class SpikeExitConfig:
    # Rule 3: 72-hour spike tiers — sell fraction of current position
    tier_1_5x_sell_pct: float = 0.30
    tier_2_0x_sell_pct: float = 0.50
    tier_2_3x_sell_pct: float = 0.75
    tier_3_0x_sell_pct: float = 1.00

    # Rule 4: Post-spike trailing stop
    post_spike_trailing_stop_pct: float = 0.25

    # Rule 5: Convexity spent thresholds
    delta_deep_itm_threshold: float = -0.90
    intrinsic_value_ratio_threshold: float = 0.80

    # Rule 6: Time bounds
    max_holding_days: int = 10
    min_holding_days: int = 1


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass
class SpikeHedgeState:
    """Minimal persistent state that advances with each daily evaluation.

    entry_value : option mark-to-market value per unit at inception.
    peak_value  : running maximum option value per unit observed to date.
    days_held   : calendar days the position has been open (0 = day of entry).
    """

    entry_value: float
    peak_value: float
    days_held: int


@dataclass
class OptionSnapshot:
    """Current option valuation and greeks at the evaluation instant.

    current_value   : mark-to-market value per unit.
    delta           : option delta — negative for long puts (e.g. -0.65 to -1.0).
    intrinsic_value : intrinsic value per unit, max(0, K - S) for a put.
    """

    current_value: float
    delta: float
    intrinsic_value: float


@dataclass
class ValuationHistory:
    """Ordered daily mark-to-market values, oldest first.

    The last element is today's value. Must contain at least one entry.
    Used to derive the historical peak independently of SpikeHedgeState.
    """

    values: list[float]

    def peak(self) -> float:
        """Maximum value observed over the full history."""
        return max(self.values) if self.values else 0.0


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass
class SpikeExitDecision:
    """Result of one evaluation step.

    should_exit   : True if any exit or partial-sell rule fired.
    sell_fraction : fraction of the current position to liquidate (0.0-1.0).
    reason        : machine-readable tag identifying the rule that fired.
    message       : human-readable explanation.
    """

    should_exit: bool
    sell_fraction: float
    reason: Optional[str]
    message: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _effective_peak(hedge_state: SpikeHedgeState, option_snapshot: OptionSnapshot) -> float:
    """Peak that accounts for today's value (Rule 1 update-in-place)."""
    return max(hedge_state.peak_value, option_snapshot.current_value)


def _multiplier(option_snapshot: OptionSnapshot, hedge_state: SpikeHedgeState) -> float:
    """Current value as a multiple of the entry value."""
    if hedge_state.entry_value <= 0:
        return 0.0
    return option_snapshot.current_value / hedge_state.entry_value


def _intrinsic_ratio(option_snapshot: OptionSnapshot) -> float:
    """Fraction of current value that is intrinsic (no time value = 1.0)."""
    if option_snapshot.current_value <= 0:
        return 0.0
    return option_snapshot.intrinsic_value / option_snapshot.current_value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_spike_exit(
    hedge_state: SpikeHedgeState,
    option_snapshot: OptionSnapshot,
    valuation_history: ValuationHistory,
    config: Optional[SpikeExitConfig] = None,
) -> SpikeExitDecision:
    """Evaluate the next action for an open hedge position.

    Parameters
    ----------
    hedge_state:
        Persistent state (entry value, running peak, days held). Advance with
        advance_hedge_state() after each call.
    option_snapshot:
        Current mark-to-market value, delta, and intrinsic value.
    valuation_history:
        Full ordered series of daily values — used to validate peak and
        support future analytics. The last element should equal
        option_snapshot.current_value.
    config:
        Tunable thresholds; defaults to SpikeExitConfig() if not supplied.

    Returns
    -------
    SpikeExitDecision
        Exit action, sell fraction, reason tag, and human-readable message.
    """
    if config is None:
        config = SpikeExitConfig()

    # Rule 1: Compute effective peak (includes today's value).
    peak = _effective_peak(hedge_state, option_snapshot)

    # Rule 2: Minimum holding period — no exit before min_holding_days.
    if hedge_state.days_held < config.min_holding_days:
        return SpikeExitDecision(
            should_exit=False,
            sell_fraction=0.0,
            reason="MIN_HOLDING_PERIOD",
            message=(f"Hold: {hedge_state.days_held}d held < {config.min_holding_days}d minimum."),
        )

    multiplier = _multiplier(option_snapshot, hedge_state)

    # Rule 3: 72-hour explosive tiers (days_held <= 3).
    if hedge_state.days_held <= 3:
        if multiplier >= 3.0:
            return SpikeExitDecision(
                should_exit=True,
                sell_fraction=config.tier_3_0x_sell_pct,
                reason="SPIKE_3X",
                message=(
                    f"3.0x spike in {hedge_state.days_held}d (x{multiplier:.2f} vs entry): "
                    f"sell {config.tier_3_0x_sell_pct * 100:.0f}% of position."
                ),
            )
        if multiplier >= 2.3:
            return SpikeExitDecision(
                should_exit=True,
                sell_fraction=config.tier_2_3x_sell_pct,
                reason="SPIKE_2_3X",
                message=(
                    f"2.3x spike in {hedge_state.days_held}d (x{multiplier:.2f} vs entry): "
                    f"sell {config.tier_2_3x_sell_pct * 100:.0f}% of position."
                ),
            )
        if multiplier >= 2.0:
            return SpikeExitDecision(
                should_exit=True,
                sell_fraction=config.tier_2_0x_sell_pct,
                reason="SPIKE_2X",
                message=(
                    f"2.0x spike in {hedge_state.days_held}d (x{multiplier:.2f} vs entry): "
                    f"sell {config.tier_2_0x_sell_pct * 100:.0f}% of position."
                ),
            )
        if multiplier >= 1.5:
            return SpikeExitDecision(
                should_exit=True,
                sell_fraction=config.tier_1_5x_sell_pct,
                reason="SPIKE_1_5X",
                message=(
                    f"1.5x spike in {hedge_state.days_held}d (x{multiplier:.2f} vs entry): "
                    f"sell {config.tier_1_5x_sell_pct * 100:.0f}% of position."
                ),
            )

    # Rule 4: Post-spike trailing stop (only if a >=1.5x spike was ever reached).
    if peak > hedge_state.entry_value * 1.5 and peak > 0:
        drawdown_from_peak = (peak - option_snapshot.current_value) / peak
        if drawdown_from_peak >= config.post_spike_trailing_stop_pct:
            return SpikeExitDecision(
                should_exit=True,
                sell_fraction=1.0,
                reason="TRAILING_STOP_POST_SPIKE",
                message=(
                    f"{drawdown_from_peak:.1%} drawdown from spike peak "
                    f"(peak={peak:.2f}, current={option_snapshot.current_value:.2f}): "
                    "exit remaining position."
                ),
            )

    # Rule 5: Convexity spent — deep ITM delta or high intrinsic ratio.
    if hedge_state.days_held >= 1:
        is_deep_itm = option_snapshot.delta <= config.delta_deep_itm_threshold
        intrinsic_ratio = _intrinsic_ratio(option_snapshot)
        convexity_spent = intrinsic_ratio >= config.intrinsic_value_ratio_threshold

        if is_deep_itm or convexity_spent:
            trigger = (
                f"delta={option_snapshot.delta:.2f} <= {config.delta_deep_itm_threshold}"
                if is_deep_itm
                else (
                    f"intrinsic ratio={intrinsic_ratio:.1%} "
                    f">= {config.intrinsic_value_ratio_threshold:.0%}"
                )
            )
            return SpikeExitDecision(
                should_exit=True,
                sell_fraction=0.50,
                reason="CONVEXITY_SPENT",
                message=f"Convexity spent ({trigger}): sell 50% and roll remainder.",
            )

    # Rule 6: Hard time cap — exit if max days elapsed and no new highs.
    if hedge_state.days_held >= config.max_holding_days:
        if option_snapshot.current_value < 0.5 * peak:
            return SpikeExitDecision(
                should_exit=True,
                sell_fraction=1.0,
                reason="TIME_CAP_NO_NEW_HIGHS",
                message=(
                    f"{hedge_state.days_held}d held >= {config.max_holding_days}d cap and "
                    f"current ({option_snapshot.current_value:.2f}) < 50% of peak "
                    f"({peak:.2f}): exit."
                ),
            )

    # Default.
    return SpikeExitDecision(
        should_exit=False,
        sell_fraction=0.0,
        reason=None,
        message="Hold: no exit conditions met.",
    )


def advance_hedge_state(
    state: SpikeHedgeState,
    option_snapshot: OptionSnapshot,
) -> SpikeHedgeState:
    """Advance one calendar day: update the running peak and increment days_held.

    Call once per daily evaluation cycle, after compute_spike_exit(), to
    persist state for the next iteration.
    """
    return SpikeHedgeState(
        entry_value=state.entry_value,
        peak_value=max(state.peak_value, option_snapshot.current_value),
        days_held=state.days_held + 1,
    )
