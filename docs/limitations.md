# Limitations

Read this before drawing any conclusion from this repository stronger than
"the code does what the tests say it does."

## The threshold system is a documented dummy baseline, not a validated production threshold

`src/thresholds/threshold_engine.py` calibrates its percentile thresholds by
calling `.fit()` on the **entire** provided sample and then evaluating every
bar in that same sample against thresholds derived from data that, for the
early bars, hadn't happened yet. This is full-sample lookahead bias, and it is
intentional and explicit — the module docstring is titled `DUMMY BASELINE
THRESHOLD SYSTEM` and states plainly: *"Thresholds are calibrated on the FULL
historical sample provided to fit(). This introduces full-sample lookahead
bias relative to every bar in the dataset. This is intentional: the system
exists only for infrastructure testing, event alignment, and model
comparison."*

**What a production-safe replacement would require:**
- Rolling or expanding-window percentile calibration (minimum ~90 days of
  trailing history before a threshold is used), so that the threshold applied
  to bar `t` only ever depends on data available at or before `t`.
- Re-validation of every historical detection-latency and false-positive
  number in `docs/validation.md` under the new calibration — the numbers cited
  there were produced under the full-sample dummy baseline and will not
  transfer unchanged to a walk-forward version.
- A decision on how to handle the cold-start period before 90 days of history
  exists (fall back to a wider window, a fixed floor, or defer flagging
  entirely).

This is the same category of defect that caused the predecessor system to be
deprecated — see `docs/model-rebuild.md` — the difference here is that it is
labeled, tested against, and never presented as validated.

## The volatility estimators themselves are not the thing under caution

To be clear about scope: `test_no_lookahead_bias` in
`tests/test_volatility_estimators.py` proves the three volatility estimators
(`naive`, `yang_zhang`, `ewma`) are causal. The lookahead risk described above
is confined to the **threshold layer** that consumes their output, not the
estimators.

## Nothing here is a backtested trading strategy

`docs/validation.md` Section 2 summarizes real historical detection behavior
of the volatility estimators. It is a measurement of *when the models would
have flagged elevated volatility*, not a backtest of the HEDGE/WAIT/REVIEW
decision engine, and it says nothing about P&L. No P&L backtest of the
decision engine exists in this repository or its source project.

## No proven alpha, no production SLA, no external usage

This repository does not claim:
- Proven P&L improvement or alpha from following its HEDGE/WAIT/REVIEW output.
- A production SLA — the `src/api/` FastAPI service has no load testing,
  uptime history, or incident record.
- Use by any external investor, client, or third party. Every decision, price,
  and portfolio value produced by `python -m src.demo` is synthetic.
- Any specific test-coverage percentage. The exact number of passing tests is
  stated in the README because it was actually run — see
  `docs/validation.md` — but line/branch coverage is not measured or claimed.

## Deployment infrastructure existing is not deployment infrastructure proven

`sql/` contains BigQuery DDL and `docs/architecture.md` describes how this
system would connect to a live options feed and a real GCP project. Their
presence in this repository is not a claim that this system has been deployed,
operated against real capital, or exercised under live market conditions.

## The synthetic demo data is a scenario generator, not a market simulator

`scripts/generate_sample_data.py` produces geometrically valid OHLCV bars with
injected volatility shocks so the pipeline has something interesting to react
to. It is not calibrated to reproduce the statistical properties of real
BTC/ETH returns (fat tails, volatility clustering beyond what's explicitly
injected, realistic autocorrelation structure) beyond what's needed to
exercise every code path. Do not use it to estimate real-world model
performance — use `docs/validation.md` Section 2 for that, with the caveats
stated there.

## Option pricing uses a synthetic chain, not a live feed

`src/pricing/synthetic_quotes.py` generates a Black-Scholes-priced put chain
from the same volatility reading the decision engine already computed. It
demonstrates how `find_target_put()` and the spread-recommendation logic work,
but it is not a live order book — real bid/ask spreads, skew, and liquidity
constraints are absent by construction.
