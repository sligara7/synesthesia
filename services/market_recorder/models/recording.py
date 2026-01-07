"""Recording session model."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import Boolean, Column, DateTime, String, text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PgUUID
from sqlalchemy.orm import relationship

from .base import Base


class Recording(Base):
    """Recording session metadata."""

    __tablename__ = "recordings"

    id = Column(
        PgUUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    name = Column(String(255), nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    started_at = Column(DateTime(timezone=True), nullable=False)
    stopped_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True, index=True)
    recording_metadata = Column("metadata", JSONB, default=dict)  # 'metadata' in DB, 'recording_metadata' in Python
    created_at = Column(
        DateTime(timezone=True),
        server_default=text("NOW()"),
    )

    # Relationships
    order_book_snapshots = relationship(
        "OrderBookSnapshot",
        back_populates="recording",
        cascade="all, delete-orphan",
    )
    trades = relationship(
        "Trade",
        back_populates="recording",
        cascade="all, delete-orphan",
    )

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "name": self.name,
            "symbol": self.symbol,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "is_active": self.is_active,
            "metadata": self.recording_metadata or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
