-- hedge_alerts
-- One row per Telegram dispatch attempt by the hedging system (signal, decision, or C09P layer).
-- Dataset: AUDIT_DATASET (e.g. crypto_ops_audit). Replace ${DATASET} below with your AUDIT_DATASET value.
-- Note: event_id is the source Event.id for traceability; NOT unique per row (repeats on retry).
-- Partitioned by created_at (DAY). Clustered on symbol, alert_type.
--
-- alert_type vocabulary
-- ---------------------
--   Signal layer  : REGIME_ELEVATED | REGIME_STRESS | REGIME_CRISIS
--                   VOL_LIMIT
--                   THRESHOLD_1H | THRESHOLD_24H | THRESHOLD_168H
--   Decision layer: DECISION_HEDGE | DECISION_WAIT | DECISION_REVIEW
--
-- signal_value semantics (by alert_type)
-- ----------------------------------------
--   REGIME_*        : Layer A EWMA z-score
--   VOL_LIMIT       : 1H naive vol annualized (e.g. 2.15 = 215%)
--   THRESHOLD_*     : abs(log_return) or abs(rolling_drawdown) of the flagged bar
--   DECISION_*      : Layer A EWMA z-score (same as REGIME_* for that bar)
--
-- percentile_rank semantics
-- -------------------------
--   THRESHOLD_* rows : full-sample percentile rank (DUMMY BASELINE -- lookahead)
--   All other rows   : NULL (no percentile computed at signal layer)
--
-- Usage
-- -----
-- Provision once per environment:
--   bq mk --project_id=$PROJECT_ID $DATASET.hedge_alerts  (schema auto-created by writer)
-- Or run this DDL directly in the BigQuery console / bq query.

CREATE TABLE IF NOT EXISTS `${PROJECT_ID}.${DATASET}.hedge_alerts`
(
    -- Identity
    event_id        STRING    NOT NULL OPTIONS (description = 'UUID for this alert row'),
    created_at      TIMESTAMP NOT NULL OPTIONS (description = 'UTC wall-clock time of write (partition column)'),

    -- Asset / classification
    symbol          STRING    NOT NULL OPTIONS (description = 'BTC or ETH'),
    alert_type      STRING    NOT NULL OPTIONS (description = 'REGIME_ELEVATED | REGIME_STRESS | REGIME_CRISIS | VOL_LIMIT | THRESHOLD_1H | THRESHOLD_24H | THRESHOLD_168H | DECISION_HEDGE | DECISION_WAIT | DECISION_REVIEW'),
    regime          STRING    NOT NULL OPTIONS (description = 'NORMAL | ELEVATED | STRESS | CRISIS'),

    -- Bar context
    bar_timestamp   TIMESTAMP          OPTIONS (description = 'open_time of the triggering OHLCV bar (UTC)'),
    horizon         STRING             OPTIONS (description = '1H | 1D | 1W — populated for THRESHOLD alerts only'),

    -- Signal values
    signal_value    FLOAT64            OPTIONS (description = 'Primary signal value; semantics depend on alert_type — see table description'),
    percentile_rank FLOAT64            OPTIONS (description = 'Percentile rank [0,100]; populated for THRESHOLD alerts only (full-sample rank — lookahead)'),

    -- Rationale
    trigger_reason  STRING             OPTIONS (description = 'Human-readable rationale string from policy or signal layer'),

    -- Decision output
    hedge_action    STRING             OPTIONS (description = 'HEDGE | WAIT | REVIEW — populated for DECISION_* alert types only'),

    -- Dispatch
    telegram_sent   BOOL      NOT NULL OPTIONS (description = 'True if Telegram HTTP call returned 200'),

    -- Full context snapshot
    raw_payload     STRING             OPTIONS (description = 'JSON object: all available fields at alert time')
)
PARTITION BY DATE(created_at)
CLUSTER BY symbol, alert_type
OPTIONS (
    description = 'One row per alert dispatched by the hedging pipeline (signal layer + decision layer). Partitioned DAY on created_at, clustered on symbol and alert_type.',
    require_partition_filter = false
);
