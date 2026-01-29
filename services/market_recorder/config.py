"""
Configuration settings for the Market Recorder service.

Uses pydantic-settings for environment variable loading.
"""

from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = "postgresql+asyncpg://synesthesia:synesthesia@localhost:5433/market_data"

    # Binance WebSocket
    binance_ws_url: str = "wss://stream.binance.us:9443/ws"
    binance_depth_levels: int = 20
    binance_depth_update_ms: int = 100

    # Recording settings
    max_concurrent_recordings: int = 10  # Increased for multi-symbol support
    batch_insert_size: int = 100  # Batch inserts for performance
    batch_insert_interval_ms: int = 500

    # Multi-symbol streaming
    default_symbols: str = "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT"
    auto_start_streaming: bool = False  # Start streaming on server startup

    # Signal detection
    candle_interval_seconds: int = 30  # Candle interval for trend calculation
    signal_ema_period: int = 30  # EMA period for bulls/bears ratio smoothing

    # Replay settings
    max_replay_duration_hours: int = 24
    default_downsample_ms: int = 100

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Production settings
    cors_origins: str = "*"  # Comma-separated list of allowed origins
    log_level: str = "INFO"

    @property
    def symbols_list(self) -> List[str]:
        """Parse default_symbols into a list."""
        return [s.strip().upper() for s in self.default_symbols.split(",") if s.strip()]

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse cors_origins into a list."""
        if self.cors_origins == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    class Config:
        env_file = ".env"
        env_prefix = "MARKET_RECORDER_"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
