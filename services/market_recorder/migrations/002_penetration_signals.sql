-- Penetration Signals Schema
-- S/R penetration signals for trading decisions

-- ============================================================================
-- Signal Type Enum
-- ============================================================================

DO $$ BEGIN
    CREATE TYPE signal_type AS ENUM (
        'bullish',        -- Resistance entered tension field
        'bearish',        -- Support entered tension field
        'strong_bullish', -- Bullish + positive ratio confluence
        'strong_bearish', -- Bearish + negative ratio confluence
        'breakout_up',    -- Price crossed from bid to ask COM
        'breakout_down'   -- Price crossed from ask to bid COM
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- ============================================================================
-- Penetration Signals (Hypertable)
-- ============================================================================

CREATE TABLE IF NOT EXISTS penetration_signals (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,

    -- Signal classification
    signal_type signal_type NOT NULL,

    -- Position data (0 = bid side, 1 = ask side)
    support_position NUMERIC(10, 6),
    resistance_position NUMERIC(10, 6),
    last_price_position NUMERIC(10, 6),

    -- Supporting price levels
    bid_com_price NUMERIC(20, 8) NOT NULL,
    ask_com_price NUMERIC(20, 8) NOT NULL,
    trend_support NUMERIC(20, 8),
    trend_resistance NUMERIC(20, 8),
    last_trade_price NUMERIC(20, 8),

    -- Volume context
    bid_active_qty NUMERIC(20, 8),
    ask_active_qty NUMERIC(20, 8),
    bulls_bears_ratio NUMERIC(10, 6),

    -- Range flags
    support_in_range BOOLEAN DEFAULT FALSE,
    resistance_in_range BOOLEAN DEFAULT FALSE,

    -- Optional recording link
    recording_id UUID REFERENCES recordings(id) ON DELETE SET NULL,

    -- Signal confidence
    confidence NUMERIC(5, 4),

    -- Additional context
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Composite primary key for hypertable
    PRIMARY KEY (time, symbol)
);

-- Convert to hypertable
SELECT create_hypertable('penetration_signals', 'time', if_not_exists => TRUE);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_signals_symbol_time
    ON penetration_signals(symbol, time DESC);

CREATE INDEX IF NOT EXISTS idx_signals_type_time
    ON penetration_signals(signal_type, time DESC);

CREATE INDEX IF NOT EXISTS idx_signals_recording
    ON penetration_signals(recording_id, time DESC)
    WHERE recording_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_signals_confidence
    ON penetration_signals(confidence DESC, time DESC)
    WHERE confidence >= 0.7;

-- Compression policy (compress after 1 day - signals are smaller dataset)
ALTER TABLE penetration_signals SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol, signal_type',
    timescaledb.compress_orderby = 'time DESC'
);

SELECT add_compression_policy('penetration_signals', INTERVAL '1 day', if_not_exists => TRUE);

-- ============================================================================
-- Signal Summary View (for dashboards)
-- ============================================================================

CREATE OR REPLACE VIEW signal_summary_hourly AS
SELECT
    time_bucket('1 hour', time) AS hour,
    symbol,
    signal_type,
    COUNT(*) AS signal_count,
    AVG(confidence) AS avg_confidence,
    MAX(confidence) AS max_confidence,
    AVG(bulls_bears_ratio) AS avg_ratio
FROM penetration_signals
GROUP BY hour, symbol, signal_type;

-- ============================================================================
-- Recent Strong Signals View
-- ============================================================================

CREATE OR REPLACE VIEW recent_strong_signals AS
SELECT
    time,
    symbol,
    signal_type,
    confidence,
    last_trade_price,
    bulls_bears_ratio,
    CASE
        WHEN signal_type IN ('strong_bullish', 'breakout_up') THEN 'BUY'
        WHEN signal_type IN ('strong_bearish', 'breakout_down') THEN 'SELL'
        ELSE 'WATCH'
    END AS action
FROM penetration_signals
WHERE time > NOW() - INTERVAL '1 hour'
  AND confidence >= 0.6
ORDER BY time DESC;

-- ============================================================================
-- Continuous Aggregate for Signal Counts (1-minute buckets)
-- ============================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS signals_1m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS bucket,
    symbol,
    signal_type,
    COUNT(*) AS signal_count,
    AVG(confidence) AS avg_confidence,
    AVG(bulls_bears_ratio) AS avg_ratio,
    LAST(last_trade_price, time) AS last_price
FROM penetration_signals
GROUP BY bucket, symbol, signal_type
WITH NO DATA;

-- Refresh policy
SELECT add_continuous_aggregate_policy('signals_1m',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '30 seconds',
    if_not_exists => TRUE
);
