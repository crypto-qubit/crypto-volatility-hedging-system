-- regime_hourly_history
-- Dataset: AUDIT_DATASET (e.g. crypto_ops_audit). Replace ${DATASET} below with your AUDIT_DATASET value.
-- Provision: bq mk --dataset ${PROJECT_ID}:${AUDIT_DATASET}
--
-- One row per asset per hourly pipeline run, written immediately after
-- SignalAggregator.compute(). Provides a permanent audit trail of all regime
-- snapshots regardless of local JSON queue pruning.
--
-- Partitioned by created_at (DAY). Clustered on asset, system_regime.
--
-- Provision once per environment:
--   bq mk --project_id=$PROJECT_ID $DATASET.regime_hourly_history  (schema auto-created by writer)
-- Or run this DDL directly in the BigQuery console / bq query.

CREATE TABLE IF NOT EXISTS `${PROJECT_ID}.${DATASET}.regime_hourly_history`
(
    -- Identity
    asset                   STRING    NOT NULL OPTIONS (description = 'BTC or ETH'),
    bar_timestamp           TIMESTAMP NOT NULL OPTIONS (description = 'open_time of the latest OHLCV bar (UTC, partition column)'),
    created_at              TIMESTAMP NOT NULL OPTIONS (description = 'UTC wall-clock time this row was written'),

    -- Regime summary
    system_regime         STRING             OPTIONS (description = 'stress | high_vol | normal | low_vol'),
    confidence              FLOAT64            OPTIONS (description = 'Combined confidence score [0.0, 1.0]'),
    dual_trigger            BOOL               OPTIONS (description = 'True when detect_market_regime returned stress'),
    missing_layers          STRING             OPTIONS (description = 'Comma-separated list of layers with insufficient data'),

    -- Layer A (1H) — EWMA λ=0.94 primary trigger
    z_1h                    FLOAT64            OPTIONS (description = 'Layer A: EWMA z-score at 1H horizon'),
    return_1h               FLOAT64            OPTIONS (description = 'Layer A: most recent 1H log return'),
    naive_vol_1h            FLOAT64            OPTIONS (description = 'Layer A: 1H naive rolling vol (annualized)'),
    ewma_flag_1h            BOOL               OPTIONS (description = 'Layer A: EWMA z-score exceeded threshold'),
    naive_flag_1h           BOOL               OPTIONS (description = 'Layer A: naive return exceeded threshold'),

    -- Layer B (1D) — EWMA λ=0.97 confirmation
    z_1d                    FLOAT64            OPTIONS (description = 'Layer B: EWMA z-score at 1D horizon'),
    return_1d               FLOAT64            OPTIONS (description = 'Layer B: most recent 1D log return'),
    naive_vol_1d            FLOAT64            OPTIONS (description = 'Layer B: 1D naive rolling vol (annualized)'),
    ewma_flag_1d            BOOL               OPTIONS (description = 'Layer B: EWMA z-score exceeded threshold'),
    naive_flag_1d           BOOL               OPTIONS (description = 'Layer B: naive return exceeded threshold'),

    -- Layer C (1W) — structural monitor
    return_1w               FLOAT64            OPTIONS (description = 'Layer C: most recent 1W log return'),
    naive_vol_1w            FLOAT64            OPTIONS (description = 'Layer C: 1W naive rolling vol (annualized)'),
    structural_flag_1w      BOOL               OPTIONS (description = 'Layer C: structural vol expansion flag'),

    -- Daily regime context (from detect_latest_regime_from_frame)
    daily_regime            STRING             OPTIONS (description = 'Daily-horizon regime label'),
    daily_regime_confidence FLOAT64            OPTIONS (description = 'Confidence of the daily regime classification'),
    daily_ann_vol_pct       FLOAT64            OPTIONS (description = 'Daily annualized vol percent'),
    daily_vol_ratio         FLOAT64            OPTIONS (description = 'Daily vol ratio vs rolling baseline'),
    daily_skewness          FLOAT64            OPTIONS (description = 'Return distribution skewness'),
    daily_excess_kurtosis   FLOAT64            OPTIONS (description = 'Return distribution excess kurtosis'),

    -- Full context snapshot
    raw_payload_json        STRING             OPTIONS (description = 'JSON of all scalar fields at write time')
)
PARTITION BY DATE(created_at)
CLUSTER BY asset, system_regime
OPTIONS (
    description = 'Hourly regime snapshot per asset. One row per asset per pipeline run. Permanent audit trail for regime history reconstruction.',
    require_partition_filter = false
);
