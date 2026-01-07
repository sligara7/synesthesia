"""
Pydantic schemas for recording API requests and responses.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class RecordingCreate(BaseModel):
    """Request schema for creating a new recording."""

    name: str = Field(..., min_length=1, max_length=255, description="Name for the recording")
    symbol: str = Field(..., min_length=1, max_length=20, description="Trading pair (e.g., BTCUSDT)")
    metadata: Optional[dict] = Field(default=None, description="Optional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "BTC Morning Session",
                "symbol": "BTCUSDT",
                "metadata": {"notes": "Testing order flow patterns"}
            }
        }


class RecordingResponse(BaseModel):
    """Response schema for a recording."""

    id: UUID
    name: str
    symbol: str
    started_at: datetime
    stopped_at: Optional[datetime] = None
    is_active: bool
    metadata: dict = Field(default_factory=dict)
    duration_seconds: Optional[float] = None
    snapshot_count: Optional[int] = None
    trade_count: Optional[int] = None

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "name": "BTC Morning Session",
                "symbol": "BTCUSDT",
                "started_at": "2024-01-15T09:00:00Z",
                "stopped_at": "2024-01-15T09:30:00Z",
                "is_active": False,
                "metadata": {"notes": "Testing order flow patterns"},
                "duration_seconds": 1800.0,
                "snapshot_count": 18000,
                "trade_count": 5432
            }
        }


class RecordingList(BaseModel):
    """Response schema for listing recordings."""

    recordings: list[RecordingResponse]
    total: int
    limit: int
    offset: int


class RecordingStats(BaseModel):
    """Statistics for a recording."""

    recording_id: UUID
    duration_seconds: float
    order_book_snapshots: int
    trades: int
    first_timestamp: Optional[datetime] = None
    last_timestamp: Optional[datetime] = None
    avg_spread: Optional[float] = None
    total_volume: Optional[float] = None
    vwap: Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "recording_id": "123e4567-e89b-12d3-a456-426614174000",
                "duration_seconds": 1800.0,
                "order_book_snapshots": 18000,
                "trades": 5432,
                "first_timestamp": "2024-01-15T09:00:00Z",
                "last_timestamp": "2024-01-15T09:30:00Z",
                "avg_spread": 0.50,
                "total_volume": 125.5,
                "vwap": 42500.25
            }
        }
