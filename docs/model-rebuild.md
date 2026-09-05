# Case Study: Rebuilding the Volatility/Regime Layer After a Lookahead-Bias Audit

This project has a predecessor. An earlier regime-classification implementation
was built, used for exploratory analysis, and then deprecated after an internal
audit found several statistical and correctness defects — including a lookahead
bias that would have made any backtest built on it unreliable. This document is
that audit, and the design decisions it forced, written up honestly rather than
quietly deleted.

## What the old implementation did

The original module computed Garman-Klass and Parkinson volatility estimators
from OHLC data, then classified each bar into a Low / Normal / Elevated / Extreme
regime using rolling percentile thresholds, with a hysteresis rule intended to
stop the regime label from flickering bar-to-bar.

## What the audit found

**1. Lookahead bias via retroactive hysteresis backfill.**
The hysteresis rule rewrote the regime label at bar `i` after observing bar
`i+1`, to smooth out single-bar flicker. That means the regime label attached to
bar `i` was not fully determined until bar `i+1` had already been seen. Any
backtest that walked forward through this "smoothed" series and treated the
label at bar `i` as known-at-time-`i` was silently using future information. This
is the single most serious defect: it does not just weaken results, it inflates
them in a way that is invisible until someone traces the label assignment logic
bar by bar.

**2. Negative-variance clipping introduced systematic bias.**
The Garman-Klass estimator can and does produce arithmetically negative
"variance" on individual bars — that's a known property of the estimator, not a
bug — and the fix was to clip those values to 0 before taking the square root.
The audit found this clipping happened selectively during high-momentum candles,
which is exactly when accurate volatility matters most, biasing the estimate
downward at the moments the system was supposed to be most reliable.

**3. The regime classifier was calibrated to hit fixed proportions, not fixed
meaning.** The percentile thresholds guaranteed roughly 25% Low / 50% Normal /
20% Elevated / 5% Extreme *by construction*, regardless of what the market was
actually doing. "Extreme" was defined as "top 5% of the trailing 30 days" — a
relative label with no absolute meaning, easy to misread as an absolute
statement about market stress.

**4. No OHLC sanity checking.** Impossible bars (`high < open`, `low > close`,
non-positive prices) were not caught before being fed into rolling-window
calculations, so a single bad tick could corrupt every window it touched,
silently.

**5. Blanket warning suppression.** `warnings.filterwarnings("ignore")` was set
at module scope, which meant `log(0) → -inf` and other numeric warnings that
should have been loud failure signals were swallowed and propagated into
downstream sigma estimates undetected.

**6. Parkinson's zero-drift assumption was applied to an asset class where it
routinely fails.** The Parkinson estimator assumes zero drift within the bar; in
strongly trending regimes (which crypto produces regularly) this systematically
underestimates volatility.

**7. No stationarity or ARCH-effect testing was ever performed** on the
underlying return series before these estimators were applied to it.

## Why it was deprecated rather than patched

Item 1 alone was disqualifying: a volatility/regime signal that isn't
determined until after the fact cannot be used to evaluate "would this have
caught the event in time," which is the entire point of the system. Patching
the hysteresis logic in place would have meant re-auditing every historical
result ever produced with it, with no way to distinguish genuine skill from
leaked future information in the existing output. A clean rebuild with an
explicit no-lookahead contract was the more defensible path, and it's the one
this repository reflects.

## What changed in the rebuild

- **Every estimator here is provably causal.** `naive`, `yang_zhang`, and `ewma`
  in `src/volatility/` compute bar `t` from bars `{t-n+1, ..., t}` only — no
  retroactive relabeling, no smoothing pass that looks forward. This is
  exercised directly by `test_no_lookahead_bias` in
  `tests/test_volatility_estimators.py`, which perturbs only the *last* bar of a
  series and asserts every earlier output value is byte-identical before and
  after.
- **OHLC integrity is validated before anything else runs.**
  `src/validation/integrity.py` checks candle geometry, null/non-positive
  prices, and several data-quality anomalies (stale feeds, volume spikes) as a
  FATAL/WARNING/MARKET_ANOMALY-severity report *before* any volatility math
  executes, closing the gap in finding 4.
- **No blanket exception or warning suppression anywhere in the codebase.**
  Numeric edge cases (e.g. non-positive prices before a log-return) are handled
  explicitly and are covered by tests, rather than caught-and-hidden.
- **Yang-Zhang replaces Parkinson as the OHLC-aware estimator**, specifically
  because it is drift-independent by construction — the failure mode in
  finding 6 doesn't apply to it.
- **Regime/threshold outputs are labeled by what they actually are.** The
  threshold system in `src/thresholds/` is explicitly documented, in its own
  module docstrings and in `docs/limitations.md`, as a **DUMMY BASELINE** that
  calibrates on the full sample — the same class of lookahead risk as finding 1,
  except this time it is written on the tin instead of discovered by an
  auditor. See `docs/limitations.md` for exactly what a production-safe
  replacement (rolling/expanding-window calibration) would require.

## What this taught me about validating production-oriented models

The most dangerous defects in this class of system are the ones that make
results look *better*, not worse — a lookahead bug doesn't crash, it quietly
inflates every backtest that depends on it. The practical takeaway I carried
into this rebuild: assume every smoothing, hysteresis, or "let's clean up the
signal" step is a potential lookahead bug until it's proven causal, and prove
it with a test that actively tries to leak future information into the past,
not just a test that checks the happy path. That's exactly what
`test_no_lookahead_bias` does here — it doesn't just check that the estimator
runs correctly, it checks that it *cannot* be influenced by data that hasn't
happened yet. Governance-wise, the other lesson was to keep the label honest
at the source: "DUMMY BASELINE — NOT FOR PRODUCTION USE" is not a comment, it's
enforced in the docstrings, the README, and the test suite, so the next person
(including future me) can't mistake exploratory infrastructure for a validated
production threshold.
