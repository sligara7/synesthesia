"""S/R Penetration Signal models for real-time trading signals."""

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum as SqlEnum,
    ForeignKey,
    Index,
    Numeric,
    String,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PgUUID

from .base import Base

import enum


class SignalType(str, enum.Enum):
    """Type of S/R penetration signal."""

    BULLISH = "bullish"           # Resistance entered tension field (being challenged)
    BEARISH = "bearish"           # Support entered tension field (being tested)
    STRONG_BULLISH = "strong_bullish"  # Bullish + positive ratio confluence
    STRONG_BEARISH = "strong_bearish"  # Bearish + negative ratio confluence
    BREAKOUT_UP = "breakout_up"   # Price crossed from bid COM to ask COM
    BREAKOUT_DOWN = "breakout_down"  # Price crossed from ask COM to bid COM


class PenetrationSignal(Base):
    """
    S/R Penetration signal detected in real-time.

    Generated when support or resistance levels enter the bid-ask COM tension field,
    indicating potential price movement.
    """

    __tablename__ = "penetration_signals"

    # Primary key is (time, symbol) for TimescaleDB hypertable
    time = Column(DateTime(timezone=True), primary_key=True)
    symbol = Column(String(20), primary_key=True)

    # Signal classification
    signal_type = Column(SqlEnum(SignalType), nullable=False)

    # Position data (0 = bid side, 1 = ask side)
    support_position = Column(Numeric(10, 6))  # Where support crosses COM line
    resistance_position = Column(Numeric(10, 6))  # Where resistance crosses COM line
    last_price_position = Column(Numeric(10, 6))  # Where last trade crosses COM line

    # Supporting price levels
    bid_com_price = Column(Numeric(20, 8), nullable=False)
    ask_com_price = Column(Numeric(20, 8), nullable=False)
    trend_support = Column(Numeric(20, 8))
    trend_resistance = Column(Numeric(20, 8))
    last_trade_price = Column(Numeric(20, 8))

    # Volume context
    bid_active_qty = Column(Numeric(20, 8))  # Volume within 1 sigma of bid COM
    ask_active_qty = Column(Numeric(20, 8))  # Volume within 1 sigma of ask COM
    bulls_bears_ratio = Column(Numeric(10, 6))  # (bid - ask) / (bid + ask)

    # Whether support/resistance is within the COM tension field
    support_in_range = Column(Boolean, default=False)
    resistance_in_range = Column(Boolean, default=False)

    # Optional: link to recording if from recorded data
    recording_id = Column(
        PgUUID(as_uuid=True),
        ForeignKey("recordings.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Signal confidence/strength (0-1)
    confidence = Column(Numeric(5, 4))

    # Additional context as JSONB
    metadata = Column(JSONB, default={})

    # Indexes for efficient queries
    __table_args__ = (
        Index("ix_signals_symbol_time", "symbol", "time"),
        Index("ix_signals_type_time", "signal_type", "time"),
        Index("ix_signals_recording", "recording_id", "time"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "time": self.time.isoformat() if self.time else None,
            "symbol": self.symbol,
            "signal_type": self.signal_type.value if self.signal_type else None,
            "positions": {
                "support": float(self.support_position) if self.support_position else None,
                "resistance": float(self.resistance_position) if self.resistance_position else None,
                "last_price": float(self.last_price_position) if self.last_price_position else None,
            },
            "prices": {
                "bid_com": float(self.bid_com_price) if self.bid_com_price else None,
                "ask_com": float(self.ask_com_price) if self.ask_com_price else None,
                "support": float(self.trend_support) if self.trend_support else None,
                "resistance": float(self.trend_resistance) if self.trend_resistance else None,
                "last_trade": float(self.last_trade_price) if self.last_trade_price else None,
            },
            "volume": {
                "bid_active_qty": float(self.bid_active_qty) if self.bid_active_qty else None,
                "ask_active_qty": float(self.ask_active_qty) if self.ask_active_qty else None,
                "bulls_bears_ratio": float(self.bulls_bears_ratio) if self.bulls_bears_ratio else None,
            },
            "in_range": {
                "support": self.support_in_range,
                "resistance": self.resistance_in_range,
            },
            "confidence": float(self.confidence) if self.confidence else None,
            "recording_id": str(self.recording_id) if self.recording_id else None,
            "metadata": self.metadata,
        }
