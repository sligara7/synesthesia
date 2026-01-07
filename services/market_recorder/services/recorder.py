"""
Recording service for capturing market data to TimescaleDB.

Implements the CanRecord protocol with batch inserts for performance.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List
from uuid import UUID

from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..protocols import DepthUpdate, TradeUpdate, RecordingInfo
from ..models import Recording, OrderBookSnapshot, Trade
from ..config import get_settings
from .binance_client import BinanceWebSocketClient

logger = logging.getLogger(__name__)


class RecordingService:
    """
    Service for recording market data streams to TimescaleDB.

    Implements CanRecord protocol with batched inserts for performance.

    Usage:
        async with get_session_context() as db:
            service = RecordingService(db)
            recording = await service.start_recording("My Recording", "BTCUSDT")
            # ... recording runs in background ...
            await service.stop_recording(recording.id)
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize the recording service.

        Args:
            db: Async database session
        """
        self.db = db
        self.settings = get_settings()

        # Active recordings: recording_id -> recording task
        self._active_recordings: Dict[UUID, asyncio.Task] = {}

        # Buffers for batch inserts: recording_id -> buffer dict
        self._buffers: Dict[UUID, Dict[str, List]] = {}

        # Binance clients: recording_id -> client
        self._clients: Dict[UUID, BinanceWebSocketClient] = {}

    async def start_recording(
        self,
        name: str,
        symbol: str,
        metadata: dict | None = None,
    ) -> RecordingInfo:
        """
        Start a new recording session.

        Args:
            name: Human-readable name for the recording
            symbol: Trading pair (e.g., "BTCUSDT")
            metadata: Optional metadata dict

        Returns:
            RecordingInfo with session details
        """
        # Check concurrent recording limit
        active_count = await self.count_active_recordings()
        if active_count >= self.settings.max_concurrent_recordings:
            raise RuntimeError(
                f"Maximum concurrent recordings ({self.settings.max_concurrent_recordings}) reached"
            )

        # Create recording record
        recording = Recording(
            name=name,
            symbol=symbol.upper(),
            started_at=datetime.now(timezone.utc),
            is_active=True,
            recording_metadata=metadata or {},
        )
        self.db.add(recording)
        await self.db.commit()
        await self.db.refresh(recording)

        logger.info(f"Created recording {recording.id}: {name} for {symbol}")

        # Initialize buffer
        self._buffers[recording.id] = {
            "order_books": [],
            "trades": [],
        }

        # Create Binance client
        client = BinanceWebSocketClient()
        self._clients[recording.id] = client

        # Start recording tasks
        task = asyncio.create_task(
            self._record_streams(recording.id, symbol, client)
        )
        self._active_recordings[recording.id] = task

        return self._to_recording_info(recording)

    async def _record_streams(
        self,
        recording_id: UUID,
        symbol: str,
        client: BinanceWebSocketClient,
    ):
        """Record both depth and trade streams concurrently."""

        async def record_depth():
            """Record order book depth updates."""
            try:
                async for update in client.stream_depth(
                    symbol,
                    depth_levels=self.settings.binance_depth_levels,
                    update_speed_ms=self.settings.binance_depth_update_ms,
                ):
                    await self._buffer_depth(recording_id, update)
            except asyncio.CancelledError:
                logger.info(f"Depth recording cancelled for {recording_id}")
            except Exception as e:
                logger.error(f"Depth recording error for {recording_id}: {e}")

        async def record_trades():
            """Record aggregated trades."""
            try:
                async for trade in client.stream_trades(symbol):
                    await self._buffer_trade(recording_id, trade)
            except asyncio.CancelledError:
                logger.info(f"Trade recording cancelled for {recording_id}")
            except Exception as e:
                logger.error(f"Trade recording error for {recording_id}: {e}")

        async def flush_loop():
            """Periodically flush buffers to database."""
            try:
                interval = self.settings.batch_insert_interval_ms / 1000
                while True:
                    await asyncio.sleep(interval)
                    await self._flush_buffers(recording_id)
            except asyncio.CancelledError:
                # Final flush
                await self._flush_buffers(recording_id)
                logger.info(f"Flush loop cancelled for {recording_id}")

        # Run all tasks concurrently
        await asyncio.gather(
            record_depth(),
            record_trades(),
            flush_loop(),
            return_exceptions=True,
        )

    async def _buffer_depth(self, recording_id: UUID, update: DepthUpdate):
        """Buffer order book update for batch insert."""
        buffer = self._buffers.get(recording_id)
        if not buffer:
            return

        # Extract best bid/ask
        best_bid_price = update.bids[0][0] if update.bids else None
        best_bid_qty = update.bids[0][1] if update.bids else None
        best_ask_price = update.asks[0][0] if update.asks else None
        best_ask_qty = update.asks[0][1] if update.asks else None

        snapshot = {
            "time": update.timestamp,
            "recording_id": recording_id,
            "symbol": update.symbol,
            "last_update_id": update.last_update_id,
            "best_bid_price": best_bid_price,
            "best_bid_qty": best_bid_qty,
            "best_ask_price": best_ask_price,
            "best_ask_qty": best_ask_qty,
            "bids": update.bids,  # Will be stored as JSONB
            "asks": update.asks,
        }
        buffer["order_books"].append(snapshot)

        # Check if we need to flush
        if len(buffer["order_books"]) >= self.settings.batch_insert_size:
            await self._flush_buffers(recording_id)

    async def _buffer_trade(self, recording_id: UUID, trade: TradeUpdate):
        """Buffer trade for batch insert."""
        buffer = self._buffers.get(recording_id)
        if not buffer:
            return

        trade_data = {
            "time": trade.timestamp,
            "recording_id": recording_id,
            "symbol": trade.symbol,
            "trade_id": trade.trade_id,
            "price": trade.price,
            "quantity": trade.quantity,
            "is_buyer_maker": trade.is_buyer_maker,
        }
        buffer["trades"].append(trade_data)

        # Check if we need to flush
        if len(buffer["trades"]) >= self.settings.batch_insert_size:
            await self._flush_buffers(recording_id)

    async def _flush_buffers(self, recording_id: UUID):
        """Flush buffered data to database."""
        buffer = self._buffers.get(recording_id)
        if not buffer:
            return

        try:
            # Bulk insert order books
            if buffer["order_books"]:
                stmt = insert(OrderBookSnapshot.__table__)
                await self.db.execute(stmt, buffer["order_books"])
                count = len(buffer["order_books"])
                buffer["order_books"] = []
                logger.debug(f"Flushed {count} order book snapshots for {recording_id}")

            # Bulk insert trades
            if buffer["trades"]:
                stmt = insert(Trade.__table__)
                await self.db.execute(stmt, buffer["trades"])
                count = len(buffer["trades"])
                buffer["trades"] = []
                logger.debug(f"Flushed {count} trades for {recording_id}")

            await self.db.commit()

        except Exception as e:
            logger.error(f"Error flushing buffers for {recording_id}: {e}")
            await self.db.rollback()

    async def stop_recording(self, recording_id: UUID) -> RecordingInfo:
        """
        Stop an active recording session.

        Args:
            recording_id: ID of recording to stop

        Returns:
            Updated RecordingInfo
        """
        # Stop the client
        client = self._clients.pop(recording_id, None)
        if client:
            client.stop()

        # Cancel the recording task
        task = self._active_recordings.pop(recording_id, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self._flush_buffers(recording_id)

        # Cleanup buffer
        self._buffers.pop(recording_id, None)

        # Update recording record
        stmt = (
            update(Recording)
            .where(Recording.id == recording_id)
            .values(
                stopped_at=datetime.now(timezone.utc),
                is_active=False,
            )
            .returning(Recording)
        )
        result = await self.db.execute(stmt)
        await self.db.commit()

        recording = result.scalar_one()
        logger.info(f"Stopped recording {recording_id}")

        return self._to_recording_info(recording)

    async def get_recording(self, recording_id: UUID) -> RecordingInfo | None:
        """Get recording by ID."""
        stmt = select(Recording).where(Recording.id == recording_id)
        result = await self.db.execute(stmt)
        recording = result.scalar_one_or_none()

        if recording:
            return self._to_recording_info(recording)
        return None

    async def list_recordings(
        self,
        symbol: str | None = None,
        active_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> List[RecordingInfo]:
        """List recordings with optional filters."""
        stmt = select(Recording).order_by(Recording.started_at.desc())

        if symbol:
            stmt = stmt.where(Recording.symbol == symbol.upper())
        if active_only:
            stmt = stmt.where(Recording.is_active == True)

        stmt = stmt.limit(limit).offset(offset)

        result = await self.db.execute(stmt)
        recordings = result.scalars().all()

        return [self._to_recording_info(r) for r in recordings]

    async def count_active_recordings(self) -> int:
        """Count currently active recordings."""
        stmt = select(Recording).where(Recording.is_active == True)
        result = await self.db.execute(stmt)
        return len(result.scalars().all())

    def _to_recording_info(self, recording: Recording) -> RecordingInfo:
        """Convert Recording model to RecordingInfo DTO."""
        return RecordingInfo(
            id=recording.id,
            name=recording.name,
            symbol=recording.symbol,
            started_at=recording.started_at,
            stopped_at=recording.stopped_at,
            is_active=recording.is_active,
            metadata=recording.recording_metadata or {},
        )
