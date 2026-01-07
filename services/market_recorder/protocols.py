"""
Protocol interfaces for the Market Recorder service.

Defines interface contracts between modules using Python typing protocols.
This enables dependency injection and maintains loose coupling between components.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import AsyncGenerator, Protocol, runtime_checkable
from uuid import UUID


# ============================================================================
# Data Transfer Objects
# ============================================================================

@dataclass
class DepthUpdate:
    """Order book depth update from Binance."""
    timestamp: datetime
    symbol: str
    last_update_id: int
    bids: list[tuple[float, float]]  # [(price, qty), ...]
    asks: list[tuple[float, float]]  # [(price, qty), ...]


@dataclass
class TradeUpdate:
    """Aggregated trade from Binance."""
    timestamp: datetime
    symbol: str
    trade_id: int
    price: float
    quantity: float
    is_buyer_maker: bool


@dataclass
class ReplayFrame:
    """Single frame of replay data."""
    timestamp: datetime
    order_book: dict | None = None  # {bids, asks, best_bid, best_ask}
    trades: list[dict] | None = None


@dataclass
class RecordingInfo:
    """Recording session metadata."""
    id: UUID
    name: str
    symbol: str
    started_at: datetime
    stopped_at: datetime | None
    is_active: bool
    metadata: dict


# ============================================================================
# Protocol Interfaces
# ============================================================================

@runtime_checkable
class CanStreamMarketData(Protocol):
    """Interface for streaming market data from an exchange."""

    async def stream_depth(
        self,
        symbol: str,
        depth_levels: int = 20,
        update_speed_ms: int = 100,
    ) -> AsyncGenerator[DepthUpdate, None]:
        """Stream order book depth updates."""
        ...

    async def stream_trades(
        self,
        symbol: str,
    ) -> AsyncGenerator[TradeUpdate, None]:
        """Stream aggregated trades."""
        ...

    def stop(self) -> None:
        """Stop all active streams."""
        ...


@runtime_checkable
class CanRecord(Protocol):
    """Interface for recording market data."""

    async def start_recording(
        self,
        name: str,
        symbol: str,
        metadata: dict | None = None,
    ) -> RecordingInfo:
        """Start a new recording session."""
        ...

    async def stop_recording(
        self,
        recording_id: UUID,
    ) -> RecordingInfo:
        """Stop an active recording session."""
        ...

    async def list_recordings(
        self,
        symbol: str | None = None,
        active_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> list[RecordingInfo]:
        """List recording sessions."""
        ...

    async def get_recording(
        self,
        recording_id: UUID,
    ) -> RecordingInfo | None:
        """Get recording session by ID."""
        ...


@runtime_checkable
class CanReplay(Protocol):
    """Interface for replaying recorded market data."""

    async def replay(
        self,
        recording_id: UUID,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        downsample_ms: int = 100,
        include_order_book: bool = True,
        include_trades: bool = True,
    ) -> AsyncGenerator[ReplayFrame, None]:
        """Stream replay data as frames."""
        ...

    async def get_recording_summary(
        self,
        recording_id: UUID,
    ) -> dict:
        """Get summary statistics for a recording."""
        ...

    async def get_batch(
        self,
        recording_id: UUID,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        downsample_ms: int = 1000,
        limit: int = 10000,
    ) -> list[ReplayFrame]:
        """Get batch of replay data (for ML training)."""
        ...
