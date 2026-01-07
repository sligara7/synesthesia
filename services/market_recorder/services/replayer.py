"""
Replay service for streaming recorded market data.

Implements the CanReplay protocol with support for downsampling
and real-time playback speed control. Also provides enriched replay
with computed analytics (COM, tension field, S/R, etc).
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import AsyncGenerator, List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..protocols import ReplayFrame as ReplayFrameDTO
from ..models import Recording, OrderBookSnapshot, Trade as TradeModel
from ..schemas.replay import (
    ReplayFrame, OrderBookData, TradeData,
    EnrichedReplayFrame, CenterOfMassData, TensionFieldData,
    QuantileLineData, TradeDensityData, CandleData, TrendStateData,
)
from ..calculations import (
    OrderLevel, Trade, Candle,
    parse_order_book, calculate_com, calculate_tension_field,
    calculate_trade_density, calculate_vwap, aggregate_candles,
    calculate_trend_state,
)

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

    # ============================================================
    # Enriched Replay (with computed analytics)
    # ============================================================

    async def get_trades_in_range(
        self,
        recording_id: UUID,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Trade]:
        """Get all trades in a time range."""

        query = text("""
            SELECT time, price, quantity, is_buyer_maker, trade_id
            FROM trades
            WHERE recording_id = :recording_id
              AND time >= :start_time
              AND time <= :end_time
            ORDER BY time
        """)

        result = await self.db.execute(query, {
            "recording_id": recording_id,
            "start_time": start_time,
            "end_time": end_time,
        })

        trades = []
        for row in result:
            trades.append(Trade(
                price=float(row.price),
                quantity=float(row.quantity),
                timestamp=row.time,
                is_buyer_maker=row.is_buyer_maker,
                trade_id=row.trade_id,
            ))

        return trades

    async def replay_enriched(
        self,
        recording_id: UUID,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        downsample_ms: int = 100,
        candle_interval_seconds: int = 30,
        trade_window_seconds: int = 60,
    ) -> AsyncGenerator[EnrichedReplayFrame, None]:
        """
        Stream enriched replay data with computed analytics.

        Args:
            recording_id: Recording to replay
            start_time: Start of replay window
            end_time: End of replay window
            downsample_ms: Time bucket size in milliseconds
            candle_interval_seconds: Candle aggregation interval
            trade_window_seconds: Rolling window for trade density/VWAP

        Yields:
            EnrichedReplayFrame objects with all computed values
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
            f"Enriched replay {recording_id} from {start_time} to {end_time} "
            f"at {downsample_ms}ms resolution"
        )

        # Prefetch all trades for the recording
        all_trades = await self.get_trades_in_range(recording_id, start_time, end_time)
        logger.info(f"Loaded {len(all_trades)} trades for enrichment")

        # Build candles from all trades
        all_candles = aggregate_candles(all_trades, candle_interval_seconds)
        logger.info(f"Aggregated into {len(all_candles)} candles")

        # Compute trend state for all candles
        candle_trends: Dict[float, Any] = {}  # bucket_time -> (candle, trend_state)
        prev_center = None
        prev_upper_mid = None
        prev_lower_mid = None
        prev_dynamic_support = None
        prev_dynamic_resistance = None
        prev_trend_support = None
        prev_trend_resistance = None

        for candle in all_candles:
            state = calculate_trend_state(
                candle,
                prev_center=prev_center,
                prev_upper_mid=prev_upper_mid,
                prev_lower_mid=prev_lower_mid,
                prev_dynamic_support=prev_dynamic_support,
                prev_dynamic_resistance=prev_dynamic_resistance,
                prev_trend_support=prev_trend_support,
                prev_trend_resistance=prev_trend_resistance,
            )

            bucket_time = candle.timestamp.timestamp()
            candle_trends[bucket_time] = (candle, state)

            # Update previous values
            prev_center = state.center
            prev_upper_mid = state.upper_mid
            prev_lower_mid = state.lower_mid
            prev_dynamic_support = state.dynamic_support
            prev_dynamic_resistance = state.dynamic_resistance
            prev_trend_support = state.trend_support
            prev_trend_resistance = state.trend_resistance

        tick = 0

        # Replay order book snapshots with enrichment
        async for frame in self._replay_downsampled(
            recording_id, start_time, end_time, downsample_ms,
            include_order_book=True, include_trades=False
        ):
            tick += 1
            frame_time = frame.timestamp

            # Parse order book
            bids_raw = frame.order_book.bids if frame.order_book else []
            asks_raw = frame.order_book.asks if frame.order_book else []
            bid_levels, ask_levels = parse_order_book(bids_raw, asks_raw)

            # Compute COM
            bid_com = calculate_com(bid_levels, "bid")
            ask_com = calculate_com(ask_levels, "ask")

            # Compute tension field
            tension = calculate_tension_field(bid_levels, ask_levels)

            # Compute bid dominance
            total_qty = bid_com.total_quantity + ask_com.total_quantity
            bid_dominance = bid_com.total_quantity / total_qty if total_qty > 0 else 0.5

            # Get trades in rolling window
            window_start = frame_time - timedelta(seconds=trade_window_seconds)
            window_trades = [
                t for t in all_trades
                if window_start <= t.timestamp <= frame_time
            ]

            # Compute trade analytics
            trade_density = calculate_trade_density(window_trades)
            vwap = calculate_vwap(window_trades)
            last_trade_price = window_trades[-1].price if window_trades else None

            # Find current candle and trend
            frame_bucket = (frame_time.timestamp() // candle_interval_seconds) * candle_interval_seconds
            current_candle = None
            current_trend = None

            # Find the most recent completed candle
            for bucket_time in sorted(candle_trends.keys(), reverse=True):
                if bucket_time <= frame_time.timestamp():
                    candle, trend = candle_trends[bucket_time]
                    current_candle = candle
                    current_trend = trend
                    break

            # Build enriched frame
            yield EnrichedReplayFrame(
                timestamp=frame_time,
                tick=tick,
                order_book=frame.order_book,
                trades=[
                    TradeData(
                        price=t.price,
                        quantity=t.quantity,
                        timestamp=t.timestamp,
                        is_buyer_maker=t.is_buyer_maker,
                        trade_id=t.trade_id,
                    )
                    for t in window_trades[-20:]  # Last 20 trades
                ] if window_trades else None,
                bid_com=CenterOfMassData(
                    price=bid_com.price,
                    total_quantity=bid_com.total_quantity,
                    active_quantity=bid_com.active_quantity,
                    sigma=bid_com.sigma,
                    side=bid_com.side,
                ),
                ask_com=CenterOfMassData(
                    price=ask_com.price,
                    total_quantity=ask_com.total_quantity,
                    active_quantity=ask_com.active_quantity,
                    sigma=ask_com.sigma,
                    side=ask_com.side,
                ),
                tension_field=TensionFieldData(
                    lines=[
                        QuantileLineData(
                            percentile=line.percentile,
                            bid_price=line.bid_price,
                            ask_price=line.ask_price,
                        )
                        for line in tension.lines
                    ],
                    bid_total=tension.bid_total,
                    ask_total=tension.ask_total,
                    convergence_ratio=tension.convergence_ratio,
                    shape=tension.shape,
                ),
                bid_dominance=bid_dominance,
                trade_density=[
                    TradeDensityData(
                        price=td.price,
                        density=td.density,
                        count=td.count,
                    )
                    for td in trade_density
                ] if trade_density else None,
                vwap=vwap,
                last_trade_price=last_trade_price,
                current_candle=CandleData(
                    timestamp=current_candle.timestamp,
                    open=current_candle.open,
                    high=current_candle.high,
                    low=current_candle.low,
                    close=current_candle.close,
                    volume=current_candle.volume,
                    trade_count=current_candle.trade_count,
                ) if current_candle else None,
                trend=TrendStateData(
                    trend_support=current_trend.trend_support,
                    trend_resistance=current_trend.trend_resistance,
                    dynamic_support=current_trend.dynamic_support,
                    dynamic_resistance=current_trend.dynamic_resistance,
                    bull_strength=current_trend.bull_strength,
                    bear_strength=current_trend.bear_strength,
                    momentum=current_trend.momentum,
                ) if current_trend else None,
            )

    async def get_enriched_batch(
        self,
        recording_id: UUID,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        downsample_ms: int = 1000,
        limit: int = 10000,
        candle_interval_seconds: int = 30,
        trade_window_seconds: int = 60,
    ) -> List[EnrichedReplayFrame]:
        """
        Get batch of enriched replay data.

        Args:
            recording_id: Recording to replay
            start_time: Start of replay window
            end_time: End of replay window
            downsample_ms: Time bucket size in milliseconds
            limit: Maximum number of frames
            candle_interval_seconds: Candle aggregation interval
            trade_window_seconds: Rolling window for trade calculations

        Returns:
            List of EnrichedReplayFrame objects
        """
        frames = []

        async for frame in self.replay_enriched(
            recording_id=recording_id,
            start_time=start_time,
            end_time=end_time,
            downsample_ms=downsample_ms,
            candle_interval_seconds=candle_interval_seconds,
            trade_window_seconds=trade_window_seconds,
        ):
            frames.append(frame)
            if len(frames) >= limit:
                break

        return frames
