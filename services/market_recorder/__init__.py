"""
Market Recorder Service

A FastAPI + TimescaleDB backend for recording and replaying Binance market data.

Usage:
    # Start the service
    docker-compose up -d

    # Or run directly
    uvicorn services.market_recorder.main:app --reload
"""

from .config import get_settings

__all__ = ["get_settings"]
