"""
Market Replay Client - Python client for the Market Recorder API.

Provides async streaming and batch access to recorded market data
for backtesting, analysis, and ML training.

Usage:
    # Streaming replay
    async with MarketReplayClient() as client:
        async for frame in client.stream_replay(recording_id):
            process_frame(frame)

    # Batch download for ML
    async with MarketReplayClient() as client:
        frames = await client.get_batch(recording_id, limit=10000)
        train_model(frames)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncGenerator, List, Optional, Protocol, runtime_checkable
from uuid import UUID

import aiohttp

logger = logging.getLogger(__name__)


# ============================================================
# Data Types
# ============================================================

@dataclass
class OrderBookData:
    """Order book snapshot data."""
    bids: List[List[float]]
    asks: List[List[float]]
    best_bid: Optional[dict] = None
    best_ask: Optional[dict] = None
    spread: Optional[float] = None
    last_update_id: Optional[int] = None

    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price from best bid/ask."""
        if self.best_bid and self.best_ask:
            bid = self.best_bid.get("price")
            ask = self.best_ask.get("price")
            if bid and ask:
                return (bid + ask) / 2
        return None

    @property
    def imbalance(self) -> Optional[float]:
        """Calculate order book imbalance at best levels."""
        if self.best_bid and self.best_ask:
            bid_qty = self.best_bid.get("qty", 0)
            ask_qty = self.best_ask.get("qty", 0)
            total = bid_qty + ask_qty
            if total > 0:
                return (bid_qty - ask_qty) / total
        return None


@dataclass
class TradeData:
    """Trade data."""
    price: float
    quantity: float
    timestamp: datetime
    is_buyer_maker: bool
    trade_id: Optional[int] = None


@dataclass
class ReplayFrame:
    """Single frame of replay data."""
    timestamp: datetime
    tick: int
    order_book: Optional[OrderBookData] = None
    trades: Optional[List[TradeData]] = None

    @classmethod
    def from_dict(cls, data: dict) -> "ReplayFrame":
        """Create frame from API response dict."""
        order_book = None
        if data.get("order_book"):
            ob = data["order_book"]
            order_book = OrderBookData(
                bids=ob.get("bids", []),
                asks=ob.get("asks", []),
                best_bid=ob.get("best_bid"),
                best_ask=ob.get("best_ask"),
                spread=ob.get("spread"),
                last_update_id=ob.get("last_update_id"),
            )

        trades = None
        if data.get("trades"):
            trades = [
                TradeData(
                    price=t["price"],
                    quantity=t["quantity"],
                    timestamp=datetime.fromisoformat(t["timestamp"].replace("Z", "+00:00")),
                    is_buyer_maker=t["is_buyer_maker"],
                    trade_id=t.get("trade_id"),
                )
                for t in data["trades"]
            ]

        timestamp = data["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        return cls(
            timestamp=timestamp,
            tick=data["tick"],
            order_book=order_book,
            trades=trades,
        )


@dataclass
class RecordingInfo:
    """Recording metadata."""
    id: UUID
    name: str
    symbol: str
    started_at: datetime
    stopped_at: Optional[datetime] = None
    is_active: bool = False
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "RecordingInfo":
        """Create from API response dict."""
        started_at = data["started_at"]
        if isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at.replace("Z", "+00:00"))

        stopped_at = data.get("stopped_at")
        if stopped_at and isinstance(stopped_at, str):
            stopped_at = datetime.fromisoformat(stopped_at.replace("Z", "+00:00"))

        return cls(
            id=UUID(data["id"]) if isinstance(data["id"], str) else data["id"],
            name=data["name"],
            symbol=data["symbol"],
            started_at=started_at,
            stopped_at=stopped_at,
            is_active=data.get("is_active", False),
            metadata=data.get("metadata", {}),
        )


# ============================================================
# Protocols
# ============================================================

@runtime_checkable
class CanStreamReplay(Protocol):
    """Protocol for streaming replay data."""

    async def stream_replay(
        self,
        recording_id: UUID,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        downsample_ms: int = 100,
        speed: float = 1.0,
    ) -> AsyncGenerator[ReplayFrame, None]:
        """Stream replay frames via SSE."""
        ...


@runtime_checkable
class CanGetBatch(Protocol):
    """Protocol for batch replay data."""

    async def get_batch(
        self,
        recording_id: UUID,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        downsample_ms: int = 1000,
        limit: int = 10000,
    ) -> List[ReplayFrame]:
        """Get batch of replay frames for ML training."""
        ...


@runtime_checkable
class CanListRecordings(Protocol):
    """Protocol for listing recordings."""

    async def list_recordings(
        self,
        symbol: Optional[str] = None,
        active_only: bool = False,
    ) -> List[RecordingInfo]:
        """List available recordings."""
        ...


# ============================================================
# Client Implementation
# ============================================================

class MarketReplayClient:
    """
    Async client for the Market Recorder API.

    Implements: CanStreamReplay, CanGetBatch, CanListRecordings

    Usage:
        async with MarketReplayClient() as client:
            recordings = await client.list_recordings()
            async for frame in client.stream_replay(recordings[0].id):
                print(frame.order_book.mid_price)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
    ):
        """
        Initialize the client.

        Args:
            base_url: Market Recorder API base URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "MarketReplayClient":
        """Enter async context."""
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context."""
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        """Get or create session."""
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def close(self) -> None:
        """Close the client session."""
        if self._session:
            await self._session.close()
            self._session = None

    # --------------------------------------------------------
    # Recordings API
    # --------------------------------------------------------

    async def list_recordings(
        self,
        symbol: Optional[str] = None,
        active_only: bool = False,
    ) -> List[RecordingInfo]:
        """
        List available recordings.

        Args:
            symbol: Filter by trading symbol
            active_only: Only return active recordings

        Returns:
            List of RecordingInfo objects
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
        if active_only:
            params["active_only"] = "true"

        async with self.session.get(
            f"{self.base_url}/recordings/",
            params=params,
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            # API returns {"recordings": [...], "total": N, ...}
            recordings_list = data.get("recordings", data) if isinstance(data, dict) else data
            return [RecordingInfo.from_dict(r) for r in recordings_list]

    async def get_recording(self, recording_id: UUID) -> Optional[RecordingInfo]:
        """
        Get a specific recording by ID.

        Args:
            recording_id: Recording UUID

        Returns:
            RecordingInfo or None if not found
        """
        async with self.session.get(
            f"{self.base_url}/recordings/{recording_id}",
        ) as resp:
            if resp.status == 404:
                return None
            resp.raise_for_status()
            data = await resp.json()
            return RecordingInfo.from_dict(data)

    # --------------------------------------------------------
    # Replay API
    # --------------------------------------------------------

    async def stream_replay(
        self,
        recording_id: UUID,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        downsample_ms: int = 100,
        speed: float = 1.0,
        include_order_book: bool = True,
        include_trades: bool = True,
    ) -> AsyncGenerator[ReplayFrame, None]:
        """
        Stream replay frames via Server-Sent Events.

        Args:
            recording_id: Recording to replay
            start_time: Start of replay window
            end_time: End of replay window
            downsample_ms: Time bucket size in milliseconds
            speed: Playback speed multiplier (1.0 = realtime)
            include_order_book: Include order book data
            include_trades: Include trade data

        Yields:
            ReplayFrame objects
        """
        params = {
            "downsample_ms": downsample_ms,
            "speed": speed,
            "include_order_book": str(include_order_book).lower(),
            "include_trades": str(include_trades).lower(),
        }

        if start_time:
            params["start_time"] = start_time.isoformat()
        if end_time:
            params["end_time"] = end_time.isoformat()

        # Use longer timeout for streaming
        stream_timeout = aiohttp.ClientTimeout(total=None, sock_read=60)

        async with self.session.get(
            f"{self.base_url}/replay/{recording_id}/stream",
            params=params,
            timeout=stream_timeout,
        ) as resp:
            resp.raise_for_status()

            async for line in resp.content:
                line = line.decode("utf-8").strip()

                if not line:
                    continue

                if line.startswith("event: end"):
                    break

                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix
                    try:
                        data = json.loads(data_str)
                        # Skip end event data
                        if "frames" in data and "timestamp" not in data:
                            continue
                        yield ReplayFrame.from_dict(data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse frame: {e}")
                        continue

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
        Get batch of replay frames for ML training.

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
        params = {
            "downsample_ms": downsample_ms,
            "limit": limit,
            "include_order_book": str(include_order_book).lower(),
            "include_trades": str(include_trades).lower(),
        }

        if start_time:
            params["start_time"] = start_time.isoformat()
        if end_time:
            params["end_time"] = end_time.isoformat()

        async with self.session.get(
            f"{self.base_url}/replay/{recording_id}/batch",
            params=params,
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

            return [ReplayFrame.from_dict(f) for f in data.get("frames", [])]

    # --------------------------------------------------------
    # Convenience Methods
    # --------------------------------------------------------

    async def replay_to_list(
        self,
        recording_id: UUID,
        max_frames: Optional[int] = None,
        **kwargs,
    ) -> List[ReplayFrame]:
        """
        Stream replay and collect into a list.

        Args:
            recording_id: Recording to replay
            max_frames: Maximum frames to collect (None = all)
            **kwargs: Arguments passed to stream_replay

        Returns:
            List of ReplayFrame objects
        """
        frames = []
        async for frame in self.stream_replay(recording_id, **kwargs):
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break
        return frames

    async def get_latest_recording(
        self,
        symbol: Optional[str] = None,
    ) -> Optional[RecordingInfo]:
        """
        Get the most recent completed recording.

        Args:
            symbol: Filter by trading symbol

        Returns:
            Most recent RecordingInfo or None
        """
        recordings = await self.list_recordings(symbol=symbol)
        completed = [r for r in recordings if not r.is_active and r.stopped_at]
        if not completed:
            return None
        return max(completed, key=lambda r: r.stopped_at)


# ============================================================
# Sync Wrapper (for non-async code)
# ============================================================

class SyncMarketReplayClient:
    """
    Synchronous wrapper for MarketReplayClient.

    Usage:
        client = SyncMarketReplayClient()
        recordings = client.list_recordings()
        frames = client.get_batch(recordings[0].id)
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the sync client."""
        self.base_url = base_url
        self._async_client: Optional[MarketReplayClient] = None

    def _get_client(self) -> MarketReplayClient:
        """Get or create async client."""
        if self._async_client is None:
            self._async_client = MarketReplayClient(self.base_url)
        return self._async_client

    def _run(self, coro):
        """Run async coroutine synchronously."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None:
            return asyncio.run(coro)
        else:
            # Already in an async context - use nest_asyncio or similar
            raise RuntimeError(
                "Cannot use SyncMarketReplayClient from async context. "
                "Use MarketReplayClient instead."
            )

    def list_recordings(
        self,
        symbol: Optional[str] = None,
        active_only: bool = False,
    ) -> List[RecordingInfo]:
        """List available recordings."""
        client = self._get_client()
        return self._run(client.list_recordings(symbol, active_only))

    def get_batch(
        self,
        recording_id: UUID,
        downsample_ms: int = 1000,
        limit: int = 10000,
        **kwargs,
    ) -> List[ReplayFrame]:
        """Get batch of replay frames."""
        client = self._get_client()
        return self._run(client.get_batch(recording_id, downsample_ms=downsample_ms, limit=limit, **kwargs))

    def close(self) -> None:
        """Close the client."""
        if self._async_client:
            self._run(self._async_client.close())
            self._async_client = None
