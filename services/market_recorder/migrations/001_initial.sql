-- Market Recorder Schema
-- TimescaleDB hypertables for high-frequency market data

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================================
-- Recording Sessions
-- ============================================================================

CREATE TABLE IF NOT EXISTS recordings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    stopped_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_recordings_symbol ON recordings(symbol);
CREATE INDEX IF NOT EXISTS idx_recordings_active ON recordings(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_recordings_started ON recordings(started_at DESC);

-- ============================================================================
-- Order Book Snapshots (Hypertable)
-- ============================================================================

CREATE TABLE IF NOT EXISTS order_book_snapshots (
    time TIMESTAMPTZ NOT NULL,
    recording_id UUID NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    last_update_id BIGINT,
    -- Denormalized for fast queries
    best_bid_price NUMERIC(20, 8),
    best_bid_qty NUMERIC(20, 8),
    best_ask_price NUMERIC(20, 8),
    best_ask_qty NUMERIC(20, 8),
    spread NUMERIC(20, 8) GENERATED ALWAYS AS (best_ask_price - best_bid_price) STORED,
    -- Full order book as JSONB (will be compressed)
    bids JSONB NOT NULL,
    asks JSONB NOT NULL
);

-- Convert to hypertable (time-series optimized)
SELECT create_hypertable('order_book_snapshots', 'time', if_not_exists => TRUE);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_obs_recording_time ON order_book_snapshots(recording_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_obs_symbol_time ON order_book_snapshots(symbol, time DESC);

-- Compression policy (compress chunks older than 1 hour)
ALTER TABLE order_book_snapshots SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'recording_id, symbol',
    timescaledb.compress_orderby = 'time DESC'
);

SELECT add_compression_policy('order_book_snapshots', INTERVAL '1 hour', if_not_exists => TRUE);

-- ============================================================================
-- Trades (Hypertable)
-- ============================================================================

CREATE TABLE IF NOT EXISTS trades (
    time TIMESTAMPTZ NOT NULL,
    recording_id UUID NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    trade_id BIGINT,
    price NUMERIC(20, 8) NOT NULL,
    quantity NUMERIC(20, 8) NOT NULL,
    is_buyer_maker BOOLEAN NOT NULL,
    -- Derived field
    quote_qty NUMERIC(20, 8) GENERATED ALWAYS AS (price * quantity) STORED
);

-- Convert to hypertable
SELECT create_hypertable('trades', 'time', if_not_exists => TRUE);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_trades_recording_time ON trades(recording_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, time DESC);

-- Compression
ALTER TABLE trades SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'recording_id, symbol',
    timescaledb.compress_orderby = 'time DESC'
);

SELECT add_compression_policy('trades', INTERVAL '1 hour', if_not_exists => TRUE);

-- ============================================================================
-- Tension Snapshots (Derived data, Hypertable)
-- ============================================================================

CREATE TABLE IF NOT EXISTS tension_snapshots (
    time TIMESTAMPTZ NOT NULL,
    recording_id UUID NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    tick_number INTEGER NOT NULL,
    -- Center of Mass data
    bid_com_price NUMERIC(20, 8),
    bid_com_qty NUMERIC(20, 8),
    ask_com_price NUMERIC(20, 8),
    ask_com_qty NUMERIC(20, 8),
    -- Derived metrics
    spread NUMERIC(20, 8),
    slope_angle NUMERIC(10, 4),
    bid_dominance NUMERIC(5, 4),
    -- Full tension field as JSONB
    tension_field JSONB
);

SELECT create_hypertable('tension_snapshots', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_tension_recording_time ON tension_snapshots(recording_id, time DESC);

-- ============================================================================
-- Continuous Aggregates for Efficient Replay
-- ============================================================================

-- 1-second trade aggregation (OHLCV)
CREATE MATERIALIZED VIEW IF NOT EXISTS trades_1s
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 second', time) AS bucket,
    recording_id,
    symbol,
    FIRST(price, time) AS open,
    MAX(price) AS high,
    MIN(price) AS low,
    LAST(price, time) AS close,
    SUM(quantity) AS volume,
    COUNT(*) AS trade_count,
    SUM(price * quantity) / NULLIF(SUM(quantity), 0) AS vwap
FROM trades
GROUP BY bucket, recording_id, symbol
WITH NO DATA;

-- Refresh policy for trades_1s
SELECT add_continuous_aggregate_policy('trades_1s',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '1 second',
    schedule_interval => INTERVAL '10 seconds',
    if_not_exists => TRUE
);

-- 100ms order book snapshots (for replay at reduced frequency)
CREATE MATERIALIZED VIEW IF NOT EXISTS order_book_100ms
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('100 milliseconds', time) AS bucket,
    recording_id,
    symbol,
    LAST(best_bid_price, time) AS best_bid_price,
    LAST(best_bid_qty, time) AS best_bid_qty,
    LAST(best_ask_price, time) AS best_ask_price,
    LAST(best_ask_qty, time) AS best_ask_qty,
    LAST(spread, time) AS spread,
    LAST(bids, time) AS bids,
    LAST(asks, time) AS asks,
    LAST(last_update_id, time) AS last_update_id
FROM order_book_snapshots
GROUP BY bucket, recording_id, symbol
WITH NO DATA;

-- Refresh policy for order_book_100ms
SELECT add_continuous_aggregate_policy('order_book_100ms',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '100 milliseconds',
    schedule_interval => INTERVAL '1 second',
    if_not_exists => TRUE
);

-- ============================================================================
-- Retention Policies (Optional - uncomment to enable)
-- ============================================================================

-- Keep raw data for 30 days
-- SELECT add_retention_policy('order_book_snapshots', INTERVAL '30 days', if_not_exists => TRUE);
-- SELECT add_retention_policy('trades', INTERVAL '30 days', if_not_exists => TRUE);

-- Keep aggregates longer
-- SELECT add_retention_policy('trades_1s', INTERVAL '1 year', if_not_exists => TRUE);
-- SELECT add_retention_policy('order_book_100ms', INTERVAL '90 days', if_not_exists => TRUE);
