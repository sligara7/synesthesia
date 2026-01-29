"""
API endpoints for multi-symbol streaming and real-time signals.

Provides endpoints for:
- Starting/stopping multi-symbol streaming sessions
- Subscribing to real-time signals via SSE
- Querying historical signals
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..database import get_session
from ..models import PenetrationSignal, SignalType
from ..services.multi_symbol_recorder import MultiSymbolRecorder, StreamingSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/streaming", tags=["streaming"])

# Global streaming session (singleton for this service)
_streaming_session: Optional[MultiSymbolRecorder] = None


# ============================================================
# Pydantic Schemas
# ============================================================


class StreamingStartRequest(BaseModel):
    """Request to start streaming session."""

    session_name: str = Field(default="Top5 Crypto Stream", description="Name for the session")
    symbols: Optional[List[str]] = Field(
        default=None,
        description="Symbols to stream (default: top 5 cryptos)",
    )


class StreamingStartResponse(BaseModel):
    """Response after starting streaming."""

    status: str
    session_name: str
    symbols: List[str]
    recordings: dict  # symbol -> recording_id


class StreamingStopResponse(BaseModel):
    """Response after stopping streaming."""

    status: str
    symbols: List[str]
    recordings: dict  # symbol -> recording summary


class StreamingStatusResponse(BaseModel):
    """Current streaming status."""

    running: bool
    symbols: List[str]
    recordings: dict  # symbol -> recording_id


class SignalResponse(BaseModel):
    """Single signal response."""

    time: datetime
    symbol: str
    signal_type: str
    positions: dict
    prices: dict
    volume: dict
    confidence: float
    recording_id: Optional[str]


class SignalListResponse(BaseModel):
    """List of signals response."""

    signals: List[SignalResponse]
    count: int
    has_more: bool


# ============================================================
# Streaming Endpoints
# ============================================================


@router.post("/start", response_model=StreamingStartResponse)
async def start_streaming(
    request: StreamingStartRequest,
    db: AsyncSession = Depends(get_session),
):
    """
    Start multi-symbol streaming session.

    Begins recording order book and trade data for all configured symbols,
    with real-time S/R penetration signal detection.
    """
    global _streaming_session

    if _streaming_session is not None and _streaming_session.is_running():
        raise HTTPException(
            status_code=400,
            detail="Streaming session already running. Stop it first.",
        )

    settings = get_settings()
    symbols = request.symbols or settings.symbols_list

    _streaming_session = MultiSymbolRecorder(db, symbols)

    try:
        recordings = await _streaming_session.start(request.session_name)

        return StreamingStartResponse(
            status="started",
            session_name=request.session_name,
            symbols=symbols,
            recordings={sym: str(info.id) for sym, info in recordings.items()},
        )
    except Exception as e:
        logger.error(f"Failed to start streaming: {e}")
        _streaming_session = None
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop", response_model=StreamingStopResponse)
async def stop_streaming():
    """
    Stop the current streaming session.

    Gracefully stops all streams, flushes buffered data, and marks recordings as complete.
    """
    global _streaming_session

    if _streaming_session is None or not _streaming_session.is_running():
        raise HTTPException(
            status_code=400,
            detail="No streaming session is running.",
        )

    try:
        recordings = await _streaming_session.stop()

        result = StreamingStopResponse(
            status="stopped",
            symbols=list(recordings.keys()),
            recordings={
                sym: {
                    "id": str(info.id),
                    "started_at": info.started_at.isoformat(),
                    "stopped_at": info.stopped_at.isoformat() if info.stopped_at else None,
                }
                for sym, info in recordings.items()
            },
        )

        _streaming_session = None
        return result

    except Exception as e:
        logger.error(f"Failed to stop streaming: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=StreamingStatusResponse)
async def get_streaming_status():
    """Get current streaming status."""
    global _streaming_session

    if _streaming_session is None or not _streaming_session.is_running():
        return StreamingStatusResponse(
            running=False,
            symbols=[],
            recordings={},
        )

    return StreamingStatusResponse(
        running=True,
        symbols=_streaming_session.get_active_symbols(),
        recordings={
            sym: str(rec_id)
            for sym, rec_id in _streaming_session.get_recording_ids().items()
        },
    )


@router.get("/signals/stream")
async def stream_signals():
    """
    Subscribe to real-time signals via Server-Sent Events (SSE).

    Returns a stream of signals as they are detected during live recording.
    """
    global _streaming_session

    if _streaming_session is None or not _streaming_session.is_running():
        raise HTTPException(
            status_code=400,
            detail="No streaming session is running. Start streaming first.",
        )

    queue = _streaming_session.subscribe_signals()

    async def event_generator():
        try:
            # Send initial connection message
            yield f"data: {json.dumps({'type': 'connected', 'symbols': _streaming_session.get_active_symbols()})}\n\n"

            while True:
                try:
                    # Wait for signal with timeout (for keepalive)
                    signal = await asyncio.wait_for(queue.get(), timeout=30.0)

                    event_data = {
                        "type": "signal",
                        "time": signal.timestamp.isoformat(),
                        "symbol": signal.symbol,
                        "signal_type": signal.signal_type.value,
                        "positions": {
                            "support": signal.support_position,
                            "resistance": signal.resistance_position,
                            "last_price": signal.last_price_position,
                        },
                        "prices": {
                            "bid_com": signal.bid_com_price,
                            "ask_com": signal.ask_com_price,
                            "support": signal.trend_support,
                            "resistance": signal.trend_resistance,
                            "last_trade": signal.last_trade_price,
                        },
                        "volume": {
                            "bid_active_qty": signal.bid_active_qty,
                            "ask_active_qty": signal.ask_active_qty,
                            "bulls_bears_ratio": signal.bulls_bears_ratio,
                        },
                        "confidence": signal.confidence,
                    }

                    yield f"data: {json.dumps(event_data)}\n\n"

                except asyncio.TimeoutError:
                    # Send keepalive
                    yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"

        except asyncio.CancelledError:
            pass
        finally:
            if _streaming_session:
                _streaming_session.unsubscribe_signals(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# Signal Query Endpoints
# ============================================================


@router.get("/signals", response_model=SignalListResponse)
async def list_signals(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    signal_type: Optional[SignalType] = Query(None, description="Filter by signal type"),
    since: Optional[datetime] = Query(None, description="Signals after this time"),
    until: Optional[datetime] = Query(None, description="Signals before this time"),
    min_confidence: Optional[float] = Query(None, ge=0, le=1, description="Minimum confidence"),
    limit: int = Query(100, ge=1, le=1000, description="Max signals to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: AsyncSession = Depends(get_session),
):
    """
    Query historical signals with filters.

    Supports filtering by symbol, signal type, time range, and confidence.
    """
    stmt = select(PenetrationSignal).order_by(desc(PenetrationSignal.time))

    if symbol:
        stmt = stmt.where(PenetrationSignal.symbol == symbol.upper())

    if signal_type:
        stmt = stmt.where(PenetrationSignal.signal_type == signal_type)

    if since:
        stmt = stmt.where(PenetrationSignal.time >= since)

    if until:
        stmt = stmt.where(PenetrationSignal.time <= until)

    if min_confidence is not None:
        stmt = stmt.where(PenetrationSignal.confidence >= min_confidence)

    # Get one extra to check if there are more
    stmt = stmt.limit(limit + 1).offset(offset)

    result = await db.execute(stmt)
    signals = result.scalars().all()

    has_more = len(signals) > limit
    signals = signals[:limit]

    return SignalListResponse(
        signals=[
            SignalResponse(
                time=s.time,
                symbol=s.symbol,
                signal_type=s.signal_type.value,
                positions={
                    "support": float(s.support_position) if s.support_position else None,
                    "resistance": float(s.resistance_position) if s.resistance_position else None,
                    "last_price": float(s.last_price_position) if s.last_price_position else None,
                },
                prices={
                    "bid_com": float(s.bid_com_price) if s.bid_com_price else None,
                    "ask_com": float(s.ask_com_price) if s.ask_com_price else None,
                    "support": float(s.trend_support) if s.trend_support else None,
                    "resistance": float(s.trend_resistance) if s.trend_resistance else None,
                    "last_trade": float(s.last_trade_price) if s.last_trade_price else None,
                },
                volume={
                    "bid_active_qty": float(s.bid_active_qty) if s.bid_active_qty else None,
                    "ask_active_qty": float(s.ask_active_qty) if s.ask_active_qty else None,
                    "bulls_bears_ratio": float(s.bulls_bears_ratio) if s.bulls_bears_ratio else None,
                },
                confidence=float(s.confidence) if s.confidence else 0.0,
                recording_id=str(s.recording_id) if s.recording_id else None,
            )
            for s in signals
        ],
        count=len(signals),
        has_more=has_more,
    )


@router.get("/signals/recent")
async def get_recent_signals(
    minutes: int = Query(5, ge=1, le=60, description="Minutes of history"),
    db: AsyncSession = Depends(get_session),
):
    """
    Get signals from the last N minutes.

    Quick endpoint for dashboard updates.
    """
    since = datetime.now(timezone.utc) - timedelta(minutes=minutes)

    stmt = (
        select(PenetrationSignal)
        .where(PenetrationSignal.time >= since)
        .order_by(desc(PenetrationSignal.time))
    )

    result = await db.execute(stmt)
    signals = result.scalars().all()

    # Group by symbol
    by_symbol: dict = {}
    for s in signals:
        if s.symbol not in by_symbol:
            by_symbol[s.symbol] = []
        by_symbol[s.symbol].append({
            "time": s.time.isoformat(),
            "signal_type": s.signal_type.value,
            "confidence": float(s.confidence) if s.confidence else 0.0,
            "last_price": float(s.last_trade_price) if s.last_trade_price else None,
        })

    return {
        "since": since.isoformat(),
        "total_signals": len(signals),
        "by_symbol": by_symbol,
    }


@router.get("/signals/summary")
async def get_signal_summary(
    hours: int = Query(24, ge=1, le=168, description="Hours of history"),
    db: AsyncSession = Depends(get_session),
):
    """
    Get signal summary statistics.

    Returns counts by symbol and signal type for the specified time period.
    """
    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    stmt = (
        select(PenetrationSignal)
        .where(PenetrationSignal.time >= since)
    )

    result = await db.execute(stmt)
    signals = result.scalars().all()

    # Aggregate stats
    by_symbol: dict = {}
    by_type: dict = {}

    for s in signals:
        # By symbol
        if s.symbol not in by_symbol:
            by_symbol[s.symbol] = {"total": 0, "by_type": {}}
        by_symbol[s.symbol]["total"] += 1
        signal_type = s.signal_type.value
        by_symbol[s.symbol]["by_type"][signal_type] = (
            by_symbol[s.symbol]["by_type"].get(signal_type, 0) + 1
        )

        # By type
        by_type[signal_type] = by_type.get(signal_type, 0) + 1

    return {
        "period_hours": hours,
        "since": since.isoformat(),
        "total_signals": len(signals),
        "by_symbol": by_symbol,
        "by_type": by_type,
    }
