"""
Pydantic schemas for replay API requests and responses.
"""

from datetime import datetime
from typing import Optional, List
from uuid import UUID

from pydantic import BaseModel, Field


class OrderBookData(BaseModel):
    """Order book snapshot data."""

    bids: List[List[float]] = Field(..., description="Bid levels [[price, qty], ...]")
    asks: List[List[float]] = Field(..., description="Ask levels [[price, qty], ...]")
    best_bid: Optional[dict] = Field(None, description="Best bid {price, qty}")
    best_ask: Optional[dict] = Field(None, description="Best ask {price, qty}")
    spread: Optional[float] = Field(None, description="Spread (best_ask - best_bid)")
    last_update_id: Optional[int] = Field(None, description="Binance update ID")


class TradeData(BaseModel):
    """Trade data."""

    price: float
    quantity: float
    timestamp: datetime
    is_buyer_maker: bool
    trade_id: Optional[int] = None


class ReplayFrame(BaseModel):
    """Single frame of replay data."""

    timestamp: datetime
    tick: int = Field(..., description="Frame sequence number")
    order_book: Optional[OrderBookData] = None
    trades: Optional[List[TradeData]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-01-15T09:00:00.100Z",
                "tick": 1,
                "order_book": {
                    "bids": [[42500.00, 1.5], [42499.50, 2.0]],
                    "asks": [[42500.50, 1.0], [42501.00, 3.0]],
                    "best_bid": {"price": 42500.00, "qty": 1.5},
                    "best_ask": {"price": 42500.50, "qty": 1.0},
                    "spread": 0.50
                },
                "trades": []
            }
        }


class ReplayRequest(BaseModel):
    """Request parameters for replay."""

    start_time: Optional[datetime] = Field(None, description="Start of replay window")
    end_time: Optional[datetime] = Field(None, description="End of replay window")
    downsample_ms: int = Field(
        default=100,
        ge=10,
        le=60000,
        description="Time bucket size in milliseconds"
    )
    speed: float = Field(
        default=1.0,
        ge=0.1,
        le=100.0,
        description="Playback speed multiplier (1.0 = realtime)"
    )
    include_order_book: bool = Field(default=True, description="Include order book data")
    include_trades: bool = Field(default=True, description="Include trade data")


class ReplayBatchRequest(BaseModel):
    """Request parameters for batch replay (ML training)."""

    start_time: Optional[datetime] = Field(None, description="Start of replay window")
    end_time: Optional[datetime] = Field(None, description="End of replay window")
    downsample_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Time bucket size in milliseconds"
    )
    limit: int = Field(
        default=10000,
        ge=1,
        le=100000,
        description="Maximum number of frames to return"
    )
    include_order_book: bool = Field(default=True)
    include_trades: bool = Field(default=True)


class ReplayBatchResponse(BaseModel):
    """Response for batch replay."""

    recording_id: UUID
    frames: List[ReplayFrame]
    total_frames: int
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    downsample_ms: int


class ReplayStatus(BaseModel):
    """Status of a replay session."""

    recording_id: UUID
    is_playing: bool
    current_time: Optional[datetime] = None
    progress_percent: float = 0.0
    frames_sent: int = 0
    speed: float = 1.0
