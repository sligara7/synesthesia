"""Market data models (order book snapshots, trades)."""

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import (
    Boolean,
    Column,
    Computed,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    BigInteger,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PgUUID
from sqlalchemy.orm import relationship

from .base import Base


class OrderBookSnapshot(Base):
    """Order book snapshot at a point in time."""

    __tablename__ = "order_book_snapshots"

    # Primary key is (time, recording_id) for hypertable
    time = Column(DateTime(timezone=True), primary_key=True)
    recording_id = Column(
        PgUUID(as_uuid=True),
        ForeignKey("recordings.id", ondelete="CASCADE"),
        primary_key=True,
    )
    symbol = Column(String(20), nullable=False)
    last_update_id = Column(BigInteger)

    # Denormalized best bid/ask for fast queries
    best_bid_price = Column(Numeric(20, 8))
    best_bid_qty = Column(Numeric(20, 8))
    best_ask_price = Column(Numeric(20, 8))
    best_ask_qty = Column(Numeric(20, 8))
    # spread is a generated column in PostgreSQL

    # Full order book as JSONB
    bids = Column(JSONB, nullable=False)  # [[price, qty], ...]
    asks = Column(JSONB, nullable=False)  # [[price, qty], ...]

    # Relationship
    recording = relationship("Recording", back_populates="order_book_snapshots")

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "time": self.time.isoformat() if self.time else None,
            "recording_id": str(self.recording_id),
            "symbol": self.symbol,
            "last_update_id": self.last_update_id,
            "best_bid": {
                "price": float(self.best_bid_price) if self.best_bid_price else None,
                "qty": float(self.best_bid_qty) if self.best_bid_qty else None,
            },
            "best_ask": {
                "price": float(self.best_ask_price) if self.best_ask_price else None,
                "qty": float(self.best_ask_qty) if self.best_ask_qty else None,
            },
            "bids": self.bids,
            "asks": self.asks,
        }


class Trade(Base):
    """Individual trade record."""

    __tablename__ = "trades"

    # Primary key is (time, recording_id, trade_id) for hypertable
    time = Column(DateTime(timezone=True), primary_key=True)
    recording_id = Column(
        PgUUID(as_uuid=True),
        ForeignKey("recordings.id", ondelete="CASCADE"),
        primary_key=True,
    )
    symbol = Column(String(20), nullable=False)
    trade_id = Column(BigInteger)
    price = Column(Numeric(20, 8), nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    is_buyer_maker = Column(Boolean, nullable=False)
    # quote_qty is a generated column in PostgreSQL

    # Relationship
    recording = relationship("Recording", back_populates="trades")

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "time": self.time.isoformat() if self.time else None,
            "recording_id": str(self.recording_id),
            "symbol": self.symbol,
            "trade_id": self.trade_id,
            "price": float(self.price) if self.price else None,
            "quantity": float(self.quantity) if self.quantity else None,
            "is_buyer_maker": self.is_buyer_maker,
        }


class TensionSnapshot(Base):
    """Tension field snapshot (derived from order book)."""

    __tablename__ = "tension_snapshots"

    time = Column(DateTime(timezone=True), primary_key=True)
    recording_id = Column(
        PgUUID(as_uuid=True),
        ForeignKey("recordings.id", ondelete="CASCADE"),
        primary_key=True,
    )
    symbol = Column(String(20), nullable=False)
    tick_number = Column(Integer, nullable=False)

    # Center of Mass data
    bid_com_price = Column(Numeric(20, 8))
    bid_com_qty = Column(Numeric(20, 8))
    ask_com_price = Column(Numeric(20, 8))
    ask_com_qty = Column(Numeric(20, 8))

    # Derived metrics
    spread = Column(Numeric(20, 8))
    slope_angle = Column(Numeric(10, 4))
    bid_dominance = Column(Numeric(5, 4))

    # Full tension field as JSONB
    tension_field = Column(JSONB)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "time": self.time.isoformat() if self.time else None,
            "recording_id": str(self.recording_id),
            "symbol": self.symbol,
            "tick_number": self.tick_number,
            "bid_com": {
                "price": float(self.bid_com_price) if self.bid_com_price else None,
                "qty": float(self.bid_com_qty) if self.bid_com_qty else None,
            },
            "ask_com": {
                "price": float(self.ask_com_price) if self.ask_com_price else None,
                "qty": float(self.ask_com_qty) if self.ask_com_qty else None,
            },
            "spread": float(self.spread) if self.spread else None,
            "slope_angle": float(self.slope_angle) if self.slope_angle else None,
            "bid_dominance": float(self.bid_dominance) if self.bid_dominance else None,
            "tension_field": self.tension_field,
        }
