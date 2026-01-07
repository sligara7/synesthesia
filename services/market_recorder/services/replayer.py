"""
Replay service for streaming recorded market data.

Implements the CanReplay protocol with support for downsampling
and real-time playback speed control.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import AsyncGenerator, List, Optional
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..protocols import ReplayFrame as ReplayFrameDTO
from ..models import Recording, OrderBookSnapshot, Trade
from ..schemas.replay import ReplayFrame, OrderBookData, TradeData

logger = logging.getLogger(__name__)


class ReplayService:
    """
    Service for replaying recorded market data.

    Implements CanReplay protocol with downsampling support.

    Usage:
        async with get_session_context() as db:
            service = ReplayService(db)
            async for frame in service.replay(recording_id):
                yield frame
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize the replay service.

        Args:
            db: Async database session
        """
        self.db = db

    async def get_recording(self, recording_id: UUID) -> Optional[Recording]:
        """Get recording by ID."""
        stmt = select(Recording).where(Recording.id == recording_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def replay(
        self,
        recording_id: UUID,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        downsample_ms: int = 100,
        include_order_book: bool = True,
        include_trades: bool = True,
    ) -> AsyncGenerator[ReplayFrame, None]:
        """
        Stream replay data as frames.

        Args:
            recording_id: Recording to replay
            start_time: Start of replay window (default: recording start)
            end_time: End of replay window (default: recording end)
            downsample_ms: Time bucket size in milliseconds
            include_order_book: Include order book data
            include_trades: Include trade data

        Yields:
            ReplayFrame objects
        """
        # Validate recording exists
        recording = await self.get_recording(recording_id)
        if not recording:
            logger.error(f"Recording {recording_id} not found")
            return

        # Set default time bounds
        if start_time is None:
            start_time = recording.started_at
        if end_time is None:
            end_time = recording.stopped_at or datetime.now(timezone.utc)

        logger.info(
            f"Replaying {recording_id} from {start_time} to {end_time} "
            f"at {downsample_ms}ms resolution"
        )

        tick = 0

        if downsample_ms >= 100:
            # Use continuous aggregate for efficiency
            async for frame in self._replay_downsampled(
                recording_id, start_time, end_time, downsample_ms,
                include_order_book, include_trades
            ):
                tick += 1
                frame.tick = tick
                yield frame
        else:
            # Full resolution replay from raw tables
            async for frame in self._replay_full_resolution(
                recording_id, start_time, end_time,
                include_order_book, include_trades
            ):
                tick += 1
                frame.tick = tick
                yield frame

    async def _replay_downsampled(
        self,
        recording_id: UUID,
        start_time: datetime,
        end_time: datetime,
        downsample_ms: int,
        include_order_book: bool,
        include_trades: bool,
    ) -> AsyncGenerator[ReplayFrame, None]:
        """Replay using time_bucket for downsampling."""

        # Build query with interval as literal (TimescaleDB requires this)
        # Using string interpolation for the interval since it's an integer we control
        query = text(f"""
            SELECT
                time_bucket(INTERVAL '{downsample_ms} milliseconds', time) AS bucket,
                LAST(best_bid_price, time) AS best_bid_price,
                LAST(best_bid_qty, time) AS best_bid_qty,
                LAST(best_ask_price, time) AS best_ask_price,
                LAST(best_ask_qty, time) AS best_ask_qty,
                LAST(bids, time) AS bids,
                LAST(asks, time) AS asks,
                LAST(last_update_id, time) AS last_update_id
            FROM order_book_snapshots
            WHERE recording_id = :recording_id
              AND time >= :start_time
              AND time <= :end_time
            GROUP BY bucket
            ORDER BY bucket
        """)

        if include_order_book:
            result = await self.db.execute(
                query,
                {
                    "recording_id": recording_id,
                    "start_time": start_time,
                    "end_time": end_time,
                }
            )

            for row in result:
                order_book = None
                if row.bids is not None and row.asks is not None:
                    spread = None
                    if row.best_ask_price and row.best_bid_price:
                        spread = float(row.best_ask_price) - float(row.best_bid_price)

                    order_book = OrderBookData(
                        bids=row.bids,
                        asks=row.asks,
                        best_bid={
                            "price": float(row.best_bid_price) if row.best_bid_price else None,
                            "qty": float(row.best_bid_qty) if row.best_bid_qty else None,
                        },
                        best_ask={
                            "price": float(row.best_ask_price) if row.best_ask_price else None,
                            "qty": float(row.best_ask_qty) if row.best_ask_qty else None,
                        },
                        spread=spread,
                        last_update_id=row.last_update_id,
                    )

                yield ReplayFrame(
                    timestamp=row.bucket,
                    tick=0,  # Will be set by caller
                    order_book=order_book,
                    trades=None,  # TODO: Add trade aggregation
                )

    async def _replay_full_resolution(
        self,
        recording_id: UUID,
        start_time: datetime,
        end_time: datetime,
        include_order_book: bool,
        include_trades: bool,
    ) -> AsyncGenerator[ReplayFrame, None]:
        """Replay at full resolution from raw tables."""

        if include_order_book:
            stmt = (
                select(OrderBookSnapshot)
                .where(OrderBookSnapshot.recording_id == recording_id)
                .where(OrderBookSnapshot.time >= start_time)
                .where(OrderBookSnapshot.time <= end_time)
                .order_by(OrderBookSnapshot.time)
            )

            result = await self.db.execute(stmt)

            for snapshot in result.scalars():
                spread = None
                if snapshot.best_ask_price and snapshot.best_bid_price:
                    spread = float(snapshot.best_ask_price) - float(snapshot.best_bid_price)

                order_book = OrderBookData(
                    bids=snapshot.bids,
                    asks=snapshot.asks,
                    best_bid={
                        "price": float(snapshot.best_bid_price) if snapshot.best_bid_price else None,
                        "qty": float(snapshot.best_bid_qty) if snapshot.best_bid_qty else None,
                    },
                    best_ask={
                        "price": float(snapshot.best_ask_price) if snapshot.best_ask_price else None,
                        "qty": float(snapshot.best_ask_qty) if snapshot.best_ask_qty else None,
                    },
                    spread=spread,
                    last_update_id=snapshot.last_update_id,
                )

                yield ReplayFrame(
                    timestamp=snapshot.time,
                    tick=0,
                    order_book=order_book,
                    trades=None,
                )

    async def get_batch(
        self,
        recording_id: UUID,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        downsample_ms: int = 1000,
        limit: int = 10000,
        include_order_book: bool = True,
        include_trades: bool = True,
    ) -> List[ReplayFrame]:
        """
        Get batch of replay data for ML training.

        Args:
            recording_id: Recording to replay
            start_time: Start of replay window
            end_time: End of replay window
            downsample_ms: Time bucket size in milliseconds
            limit: Maximum number of frames
            include_order_book: Include order book data
            include_trades: Include trade data

        Returns:
            List of ReplayFrame objects
        """
        frames = []

        async for frame in self.replay(
            recording_id=recording_id,
            start_time=start_time,
            end_time=end_time,
            downsample_ms=downsample_ms,
            include_order_book=include_order_book,
            include_trades=include_trades,
        ):
            frames.append(frame)
            if len(frames) >= limit:
                break

        return frames

    async def get_recording_time_range(
        self,
        recording_id: UUID,
    ) -> tuple[Optional[datetime], Optional[datetime]]:
        """Get the time range of data in a recording."""

        query = text("""
            SELECT
                MIN(time) as first_time,
                MAX(time) as last_time
            FROM order_book_snapshots
            WHERE recording_id = :recording_id
        """)

        result = await self.db.execute(query, {"recording_id": recording_id})
        row = result.first()

        if row:
            return row.first_time, row.last_time
        return None, None

    async def get_frame_count(
        self,
        recording_id: UUID,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        downsample_ms: int = 100,
    ) -> int:
        """Estimate the number of frames in a replay."""

        first_time, last_time = await self.get_recording_time_range(recording_id)

        if not first_time or not last_time:
            return 0

        if start_time:
            first_time = max(first_time, start_time)
        if end_time:
            last_time = min(last_time, end_time)

        duration_ms = (last_time - first_time).total_seconds() * 1000
        return max(1, int(duration_ms / downsample_ms))
