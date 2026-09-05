# Validation

This document distinguishes four different kinds of "this works" claim that
are easy to blur together, and states exactly what evidence backs each one in
this repository.

| Kind of claim | What it means | Where the evidence is |
|---|---|---|
| **Statistical validation** | Unit-level correctness of the math (no lookahead, correct annualization, correct boundary behavior) | `tests/` — 109 tests, see the [root README](../README.md#testing-and-security-controls) |
| **Offline synthetic demonstration** | The pipeline runs end-to-end and produces a coherent, explainable decision | `python -m src.demo`, `tests/test_demo_e2e.py` |
| **Historical analysis** | How the three estimators actually behaved on real BTC/ETH history across known stress events | This document, Sections 2–3 (summarized from prior research; not reproducible from the code in this repository — see caveat below) |
| **Deployable architecture** | The system is *structured* the way a production service would be, with an API, audit trail, and kill switch | `src/api/`, `src/decision/store.py`, `docs/architecture.md` |

None of these is a substitute for the others, and this repository does **not**
claim the fourth category implies the third, or that the third implies proven
trading performance. See `docs/limitations.md` for what is explicitly *not*
established here.

## 1. Statistical validation (this repo, reproducible)

Every volatility estimator is tested against three properties that matter more
than "does it produce a plausible-looking number":

1. **No lookahead bias.** `test_no_lookahead_bias` perturbs only the final bar
   of an OHLCV series and asserts that every earlier estimator output is
   unchanged. This is the direct test-suite answer to the failure mode
   described in `docs/model-rebuild.md`.
2. **Annualization consistency.** `volatility_annualized / volatility_raw`
   must equal `sqrt(HOURS_PER_YEAR)` for every non-NaN row, for all three
   estimators.
3. **OHLC-awareness where claimed.** Yang-Zhang is tested to actually respond
   to a pure intrabar range widening (high/low spread) with unchanged
   close-to-close returns — proving it uses more than just the close price,
   unlike the naive estimator, which is tested to *not* respond to the same
   perturbation.

Run them yourself: `pytest tests/test_volatility_estimators.py -v`.

## 2. Historical analysis (prior research, not reproducible here)

Before this repository was curated, the same three estimator implementations
(naive, Yang-Zhang, EWMA λ=0.94) were run against **~72,000 hourly bars of real
BTC and ETH history, 2018-01-01 → 2026-03-27**, across 10 known market-stress
events (COVID Black Thursday, the May 2021 crash, the China mining ban, Luna/UST,
3AC, FTX, the USDC/SVB depeg, the BTC ETF approval, the April 2024 Iran/Israel
shock, and the August 2024 recession-fear selloff) plus 5 calm reference
periods for false-positive measurement.

**This analysis is not bundled or re-runnable from this repository** — it
depended on a long proprietary historical dataset that is out of scope for an
offline, credential-free demo. It is summarized here because the findings
materially shaped the design of `src/decision/signal_aggregator.py` (the
Layer A/1H, Layer B/1D, Layer C/1W split) and are the most honest evidence
available for *how these specific estimator implementations behave*, as
opposed to a synthetic proxy.

**Headline numbers** (zscore > 2.0 = "strong" detection):

- All three estimators detected **100% of the 10 known events** at this
  threshold — detection coverage was not the differentiator.
- Median detection latency: EWMA -21.5h (BTC) / -24.0h (ETH), Naive -20.5h /
  -31.0h, Yang-Zhang -6.0h / -9.0h (negative = early warning ahead of the
  defined event start).
- False-positive rate in calm reference periods (zscore > 1.5): Yang-Zhang was
  **highest** at 16.1% (BTC) / 16.4% (ETH), Naive middle at 15.0% / 14.3%,
  EWMA **lowest** at 12.7% / 12.9%.

**What surprised the research, and why it matters for how to read this
system's output:**

- **EWMA is not universally fastest.** For sudden single-bar liquidation
  cascades (COVID Black Thursday) EWMA detected 33h early — genuinely fastest.
  For FTX, which unfolded as a multi-day drift rather than a single shock,
  EWMA was **22h late**; Naive and Yang-Zhang both detected within ±1h of the
  event start. EWMA's ~12h half-life at λ=0.94 does not accumulate slow-burn
  stress the way a longer memory does.
- **Yang-Zhang, despite using all four OHLC prices, gave the least early
  warning on average** (median -6h to -9h vs. -20h to -31h for the other two)
  and was the noisiest in calm markets — traced to thin-orderbook "phantom
  wicks" during Asia off-peak hours (02:00–06:00 UTC) inflating its
  Rogers-Satchell component even when the close-to-close move was negligible.
- **Naive volatility handled slow structural deterioration best** in this
  sample (31h early on the China mining ban, a 5-week grind) but exhibits a
  hard-cliff artifact: when a large spike ages out of its 24-bar window
  mid-crisis, the reading drops abruptly even though stress is still active.
- **The dummy-baseline 168h drawdown threshold, at its calibrated P99.75, is
  operationally close to useless** for early warning — it produced only 7
  event windows for BTC across 8 years of data, firing only at the most
  extreme tail events, well after the damage was underway.

The full per-event tables (max volatility reached, z-score, exact latency per
asset and model) existed in `reports/research_validation/` and
`reports/multi_horizon_validation/` in the source project; they are not
included here per the curation scope (they reference a private historical
dataset and are heavy with plots), but the summary above is a faithful
condensation of that report's Sections 3–8, not a fabrication.

## 3. Offline synthetic demonstration (this repo, reproducible)

`python -m src.demo` exercises the full pipeline — ingestion, integrity
validation, all three estimators, the dummy-baseline threshold system, regime
classification, and the HEDGE/WAIT/REVIEW decision engine — against
deterministic synthetic data (`scripts/generate_sample_data.py`, fixed seeds
42/43). This proves the pipeline is *wired correctly end-to-end and produces
explainable output*; it says nothing about whether the decisions would have
been good trading decisions, because the input data is synthetic by
construction. See `tests/test_demo_e2e.py` and `tests/test_synthetic_data.py`.

## 4. Deployable architecture (documented, not claimed as "in production")

`src/api/`, the per-asset `DecisionStore` audit trail, the kill switch, and the
`sql/` BigQuery schemas describe a system *shaped* the way a production
service would be. See `docs/architecture.md` for how these pieces would be
wired to a real BigQuery project and a live options feed. **The presence of
this architecture is not a claim that it has been deployed, load-tested, or
operated live** — see `docs/limitations.md`.
