"""
Configuration settings for the Market Recorder service.

Uses pydantic-settings for environment variable loading.
"""

from functools import lru_cache
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
    max_concurrent_recordings: int = 5
    batch_insert_size: int = 100  # Batch inserts for performance
    batch_insert_interval_ms: int = 500

    # Replay settings
    max_replay_duration_hours: int = 24
    default_downsample_ms: int = 100

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    class Config:
        env_file = ".env"
        env_prefix = "MARKET_RECORDER_"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
