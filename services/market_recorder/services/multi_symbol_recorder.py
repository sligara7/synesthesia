"""
Multi-Symbol Market Data Recorder.

Manages concurrent streaming of market data for multiple symbols (top 5 cryptos),
with integrated real-time S/R penetration signal detection.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator, Callable, Dict, List, Optional, Set
from uuid import UUID

from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..models import OrderBookSnapshot, Recording, Trade
from ..models.signals import PenetrationSignal
from ..protocols import DepthUpdate, RecordingInfo, TradeUpdate
from .binance_client import BinanceWebSocketClient
from .signal_detector import DetectedSignal, SignalDetector

logger = logging.getLogger(__name__)


# Default top 5 cryptos by trading volume/market cap
DEFAULT_TOP_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
]


class MultiSymbolRecorder:
    """
    Manages concurrent recording of multiple symbols with signal detection.

    Features:
    - Streams order book and trade data for multiple symbols simultaneously
    - Computes S/R penetration signals in real-time
    - Batched inserts to TimescaleDB for efficiency
    - Single database session shared across all streams
    - Graceful shutdown with data flush
    """

    def __init__(
        self,
        db: AsyncSession,
        symbols: Optional[List[str]] = None,
        on_signal: Optional[Callable[[DetectedSignal], None]] = None,
    ):
        """
        Initialize the multi-symbol recorder.

        Args:
            db: Async database session
            symbols: List of symbols to record (default: top 5 cryptos)
            on_signal: Optional callback for real-time signal notifications
        """
        self.db = db
        self.settings = get_settings()
        self.symbols = [s.upper() for s in (symbols or DEFAULT_TOP_SYMBOLS)]

        # Recording state per symbol
        self._recordings: Dict[str, Recording] = {}
        self._clients: Dict[str, BinanceWebSocketClient] = {}
        self._tasks: Dict[str, asyncio.Task] = {}

        # Shared buffer for batch inserts
        self._order_book_buffer: List[dict] = []
        self._trade_buffer: List[dict] = []
        self._signal_buffer: List[PenetrationSignal] = []
        self._buffer_lock = asyncio.Lock()

        # Signal detector
        self._signal_detector = SignalDetector(
            candle_interval_seconds=30,
            ema_period=30,
            on_signal=on_signal,
        )

        # Control
        self._running = False
        self._flush_task: Optional[asyncio.Task] = None
        self._session_name: Optional[str] = None

        # Signal subscribers (for WebSocket streaming)
        self._signal_subscribers: Set[asyncio.Queue] = set()

    async def start(self, session_name: str = "Top5 Crypto Stream") -> Dict[str, RecordingInfo]:
        """
        Start recording all configured symbols.

        Args:
            session_name: Base name for the recording session

        Returns:
            Dict of symbol -> RecordingInfo
        """
        if self._running:
            raise RuntimeError("Recorder is already running")

        self._running = True
        self._session_name = session_name
        results: Dict[str, RecordingInfo] = {}

        logger.info(f"Starting multi-symbol recording: {self.symbols}")

        # Create recording entries for each symbol
        for symbol in self.symbols:
            recording = Recording(
                name=f"{session_name} - {symbol}",
                symbol=symbol,
                started_at=datetime.now(timezone.utc),
                is_active=True,
                recording_metadata={
                    "session": session_name,
                    "multi_symbol": True,
                    "symbols": self.symbols,
                },
            )
            self.db.add(recording)
            await self.db.commit()
            await self.db.refresh(recording)

            self._recordings[symbol] = recording
            results[symbol] = self._to_recording_info(recording)
            logger.info(f"Created recording {recording.id} for {symbol}")

        # Start WebSocket clients and recording tasks
        for symbol in self.symbols:
            client = BinanceWebSocketClient()
            self._clients[symbol] = client

            task = asyncio.create_task(
                self._record_symbol(symbol, client),
                name=f"record_{symbol}",
            )
            self._tasks[symbol] = task

        # Start flush loop
        self._flush_task = asyncio.create_task(
            self._flush_loop(),
            name="flush_loop",
        )

        return results

    async def stop(self) -> Dict[str, RecordingInfo]:
        """
        Stop all recordings gracefully.

        Returns:
            Dict of symbol -> final RecordingInfo
        """
        if not self._running:
            return {}

        logger.info("Stopping multi-symbol recording...")
        self._running = False

        # Stop all clients
        for symbol, client in self._clients.items():
            client.stop()

        # Cancel all tasks
        for symbol, task in self._tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Cancel flush task
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self._flush_buffers()

        # Update recording records
        results: Dict[str, RecordingInfo] = {}
        for symbol, recording in self._recordings.items():
            recording.stopped_at = datetime.now(timezone.utc)
            recording.is_active = False
            await self.db.commit()
            await self.db.refresh(recording)
            results[symbol] = self._to_recording_info(recording)
            logger.info(f"Stopped recording {recording.id} for {symbol}")

        # Cleanup
        self._recordings.clear()
        self._clients.clear()
        self._tasks.clear()
        self._signal_detector.reset_state()

        return results

    async def _record_symbol(self, symbol: str, client: BinanceWebSocketClient):
        """Record streams for a single symbol."""
        recording_id = self._recordings[symbol].id

        async def record_depth():
            """Record order book depth updates."""
            try:
                async for update in client.stream_depth(
                    symbol,
                    depth_levels=self.settings.binance_depth_levels,
                    update_speed_ms=self.settings.binance_depth_update_ms,
                ):
                    await self._buffer_depth(symbol, recording_id, update)

                    # Detect signals
                    signals = self._signal_detector.process_depth(symbol, update, recording_id)
                    for sig in signals:
                        await self._buffer_signal(sig)
                        await self._notify_subscribers(sig)

            except asyncio.CancelledError:
                logger.info(f"Depth recording cancelled for {symbol}")
            except Exception as e:
                logger.error(f"Depth recording error for {symbol}: {e}")

        async def record_trades():
            """Record aggregated trades."""
            try:
                async for trade in client.stream_trades(symbol):
                    await self._buffer_trade(symbol, recording_id, trade)

                    # Update signal detector
                    signals = self._signal_detector.process_trade(symbol, trade, recording_id)
                    for sig in signals:
                        await self._buffer_signal(sig)
                        await self._notify_subscribers(sig)

            except asyncio.CancelledError:
                logger.info(f"Trade recording cancelled for {symbol}")
            except Exception as e:
                logger.error(f"Trade recording error for {symbol}: {e}")

        # Run both streams concurrently
        await asyncio.gather(
            record_depth(),
            record_trades(),
            return_exceptions=True,
        )

    async def _buffer_depth(
        self, symbol: str, recording_id: UUID, update: DepthUpdate
    ):
        """Buffer order book update for batch insert."""
        snapshot = {
            "time": update.timestamp,
            "recording_id": recording_id,
            "symbol": symbol,
            "last_update_id": update.last_update_id,
            "best_bid_price": update.bids[0][0] if update.bids else None,
            "best_bid_qty": update.bids[0][1] if update.bids else None,
            "best_ask_price": update.asks[0][0] if update.asks else None,
            "best_ask_qty": update.asks[0][1] if update.asks else None,
            "bids": update.bids,
            "asks": update.asks,
        }

        async with self._buffer_lock:
            self._order_book_buffer.append(snapshot)

            if len(self._order_book_buffer) >= self.settings.batch_insert_size:
                await self._flush_buffers()

    async def _buffer_trade(
        self, symbol: str, recording_id: UUID, trade: TradeUpdate
    ):
        """Buffer trade for batch insert."""
        trade_data = {
            "time": trade.timestamp,
            "recording_id": recording_id,
            "symbol": symbol,
            "trade_id": trade.trade_id,
            "price": trade.price,
            "quantity": trade.quantity,
            "is_buyer_maker": trade.is_buyer_maker,
        }

        async with self._buffer_lock:
            self._trade_buffer.append(trade_data)

            if len(self._trade_buffer) >= self.settings.batch_insert_size:
                await self._flush_buffers()

    async def _buffer_signal(self, signal: DetectedSignal):
        """Buffer signal for batch insert."""
        async with self._buffer_lock:
            self._signal_buffer.append(signal.to_penetration_signal())

    async def _flush_loop(self):
        """Periodically flush buffers to database."""
        interval = self.settings.batch_insert_interval_ms / 1000

        try:
            while self._running:
                await asyncio.sleep(interval)
                await self._flush_buffers()
        except asyncio.CancelledError:
            # Final flush
            await self._flush_buffers()

    async def _flush_buffers(self):
        """Flush all buffered data to database."""
        async with self._buffer_lock:
            try:
                # Flush order books
                if self._order_book_buffer:
                    stmt = insert(OrderBookSnapshot.__table__)
                    await self.db.execute(stmt, self._order_book_buffer)
                    count = len(self._order_book_buffer)
                    self._order_book_buffer = []
                    logger.debug(f"Flushed {count} order book snapshots")

                # Flush trades
                if self._trade_buffer:
                    stmt = insert(Trade.__table__)
                    await self.db.execute(stmt, self._trade_buffer)
                    count = len(self._trade_buffer)
                    self._trade_buffer = []
                    logger.debug(f"Flushed {count} trades")

                # Flush signals
                if self._signal_buffer:
                    stmt = insert(PenetrationSignal.__table__)
                    await self.db.execute(
                        stmt,
                        [
                            {
                                "time": s.time,
                                "symbol": s.symbol,
                                "signal_type": s.signal_type,
                                "support_position": s.support_position,
                                "resistance_position": s.resistance_position,
                                "last_price_position": s.last_price_position,
                                "bid_com_price": s.bid_com_price,
                                "ask_com_price": s.ask_com_price,
                                "trend_support": s.trend_support,
                                "trend_resistance": s.trend_resistance,
                                "last_trade_price": s.last_trade_price,
                                "bid_active_qty": s.bid_active_qty,
                                "ask_active_qty": s.ask_active_qty,
                                "bulls_bears_ratio": s.bulls_bears_ratio,
                                "support_in_range": s.support_in_range,
                                "resistance_in_range": s.resistance_in_range,
                                "confidence": s.confidence,
                                "recording_id": s.recording_id,
                            }
                            for s in self._signal_buffer
                        ],
                    )
                    count = len(self._signal_buffer)
                    self._signal_buffer = []
                    logger.debug(f"Flushed {count} signals")

                await self.db.commit()

            except Exception as e:
                logger.error(f"Error flushing buffers: {e}")
                await self.db.rollback()

    def subscribe_signals(self) -> asyncio.Queue:
        """
        Subscribe to real-time signals.

        Returns:
            Queue that will receive DetectedSignal objects
        """
        queue: asyncio.Queue = asyncio.Queue()
        self._signal_subscribers.add(queue)
        return queue

    def unsubscribe_signals(self, queue: asyncio.Queue):
        """Unsubscribe from real-time signals."""
        self._signal_subscribers.discard(queue)

    async def _notify_subscribers(self, signal: DetectedSignal):
        """Notify all subscribers of a new signal."""
        for queue in self._signal_subscribers:
            try:
                queue.put_nowait(signal)
            except asyncio.QueueFull:
                logger.warning("Signal subscriber queue full, dropping signal")

    def get_active_symbols(self) -> List[str]:
        """Get list of currently active symbols."""
        return list(self._recordings.keys())

    def get_recording_ids(self) -> Dict[str, UUID]:
        """Get recording IDs for all active symbols."""
        return {symbol: rec.id for symbol, rec in self._recordings.items()}

    def is_running(self) -> bool:
        """Check if recorder is running."""
        return self._running

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


class StreamingSession:
    """
    Convenience wrapper for managing a multi-symbol streaming session.

    Usage:
        async with StreamingSession(db) as session:
            # Session starts automatically
            signals = session.subscribe_signals()

            async for signal in signals:
                print(f"Signal: {signal}")
    """

    def __init__(
        self,
        db: AsyncSession,
        symbols: Optional[List[str]] = None,
        session_name: str = "Top5 Crypto Stream",
    ):
        self.recorder = MultiSymbolRecorder(db, symbols)
        self.session_name = session_name
        self._recordings: Dict[str, RecordingInfo] = {}

    async def __aenter__(self) -> "StreamingSession":
        self._recordings = await self.recorder.start(self.session_name)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.recorder.stop()

    def subscribe_signals(self) -> asyncio.Queue:
        """Subscribe to real-time signals."""
        return self.recorder.subscribe_signals()

    @property
    def recordings(self) -> Dict[str, RecordingInfo]:
        """Get recording info for all symbols."""
        return self._recordings

    @property
    def symbols(self) -> List[str]:
        """Get list of active symbols."""
        return self.recorder.get_active_symbols()
