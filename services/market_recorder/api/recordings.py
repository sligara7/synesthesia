"""
API endpoints for recording management.

Provides REST endpoints for starting, stopping, and listing recordings.
"""

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..models import Recording, OrderBookSnapshot, Trade
from ..services.recorder import RecordingService
from ..schemas.recording import (
    RecordingCreate,
    RecordingResponse,
    RecordingList,
    RecordingStats,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recordings", tags=["recordings"])

# Store active recording service (singleton per app)
_recording_service: Optional[RecordingService] = None


def get_recording_service(db: AsyncSession = Depends(get_session)) -> RecordingService:
    """Get or create recording service instance."""
    global _recording_service
    if _recording_service is None:
        _recording_service = RecordingService(db)
    else:
        # Update the db session
        _recording_service.db = db
    return _recording_service


@router.post("/", response_model=RecordingResponse, status_code=201)
async def start_recording(
    request: RecordingCreate,
    service: RecordingService = Depends(get_recording_service),
):
    """
    Start a new recording session.

    Creates a new recording and begins capturing order book and trade data
    from Binance WebSocket streams.
    """
    try:
        recording = await service.start_recording(
            name=request.name,
            symbol=request.symbol,
            metadata=request.metadata,
        )

        return RecordingResponse(
            id=recording.id,
            name=recording.name,
            symbol=recording.symbol,
            started_at=recording.started_at,
            stopped_at=recording.stopped_at,
            is_active=recording.is_active,
            metadata=recording.metadata,
        )

    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting recording: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start recording: {e}")


@router.post("/{recording_id}/stop", response_model=RecordingResponse)
async def stop_recording(
    recording_id: UUID,
    service: RecordingService = Depends(get_recording_service),
):
    """
    Stop an active recording session.

    Stops data capture and finalizes the recording.
    """
    # Check if recording exists
    recording = await service.get_recording(recording_id)
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")

    if not recording.is_active:
        raise HTTPException(status_code=400, detail="Recording is not active")

    try:
        recording = await service.stop_recording(recording_id)

        return RecordingResponse(
            id=recording.id,
            name=recording.name,
            symbol=recording.symbol,
            started_at=recording.started_at,
            stopped_at=recording.stopped_at,
            is_active=recording.is_active,
            metadata=recording.metadata,
        )

    except Exception as e:
        logger.error(f"Error stopping recording: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop recording: {e}")


@router.get("/", response_model=RecordingList)
async def list_recordings(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    active_only: bool = Query(False, description="Only show active recordings"),
    limit: int = Query(50, ge=1, le=100, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    service: RecordingService = Depends(get_recording_service),
):
    """
    List all recordings with optional filters.

    Returns paginated list of recordings, optionally filtered by symbol
    or active status.
    """
    recordings = await service.list_recordings(
        symbol=symbol,
        active_only=active_only,
        limit=limit,
        offset=offset,
    )

    return RecordingList(
        recordings=[
            RecordingResponse(
                id=r.id,
                name=r.name,
                symbol=r.symbol,
                started_at=r.started_at,
                stopped_at=r.stopped_at,
                is_active=r.is_active,
                metadata=r.metadata,
            )
            for r in recordings
        ],
        total=len(recordings),  # TODO: Add total count query
        limit=limit,
        offset=offset,
    )


@router.get("/{recording_id}", response_model=RecordingResponse)
async def get_recording(
    recording_id: UUID,
    service: RecordingService = Depends(get_recording_service),
    db: AsyncSession = Depends(get_session),
):
    """
    Get recording details by ID.

    Returns recording metadata and basic statistics.
    """
    recording = await service.get_recording(recording_id)
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")

    # Get counts
    snapshot_count = await db.scalar(
        select(func.count()).select_from(OrderBookSnapshot).where(
            OrderBookSnapshot.recording_id == recording_id
        )
    )
    trade_count = await db.scalar(
        select(func.count()).select_from(Trade).where(
            Trade.recording_id == recording_id
        )
    )

    # Calculate duration
    duration = None
    if recording.started_at:
        end_time = recording.stopped_at or recording.started_at
        duration = (end_time - recording.started_at).total_seconds()

    return RecordingResponse(
        id=recording.id,
        name=recording.name,
        symbol=recording.symbol,
        started_at=recording.started_at,
        stopped_at=recording.stopped_at,
        is_active=recording.is_active,
        metadata=recording.metadata,
        duration_seconds=duration,
        snapshot_count=snapshot_count,
        trade_count=trade_count,
    )


@router.get("/{recording_id}/stats", response_model=RecordingStats)
async def get_recording_stats(
    recording_id: UUID,
    db: AsyncSession = Depends(get_session),
    service: RecordingService = Depends(get_recording_service),
):
    """
    Get detailed statistics for a recording.

    Returns counts, timestamps, and aggregate metrics.
    """
    recording = await service.get_recording(recording_id)
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")

    # Order book stats
    obs_stats = await db.execute(
        select(
            func.count().label("count"),
            func.min(OrderBookSnapshot.time).label("first_time"),
            func.max(OrderBookSnapshot.time).label("last_time"),
            func.avg(OrderBookSnapshot.best_ask_price - OrderBookSnapshot.best_bid_price).label("avg_spread"),
        ).where(OrderBookSnapshot.recording_id == recording_id)
    )
    obs_row = obs_stats.first()

    # Trade stats
    trade_stats = await db.execute(
        select(
            func.count().label("count"),
            func.sum(Trade.quantity).label("total_volume"),
            (func.sum(Trade.price * Trade.quantity) / func.nullif(func.sum(Trade.quantity), 0)).label("vwap"),
        ).where(Trade.recording_id == recording_id)
    )
    trade_row = trade_stats.first()

    # Calculate duration
    duration = 0.0
    first_time = obs_row.first_time if obs_row else None
    last_time = obs_row.last_time if obs_row else None
    if first_time and last_time:
        duration = (last_time - first_time).total_seconds()

    return RecordingStats(
        recording_id=recording_id,
        duration_seconds=duration,
        order_book_snapshots=obs_row.count if obs_row else 0,
        trades=trade_row.count if trade_row else 0,
        first_timestamp=first_time,
        last_timestamp=last_time,
        avg_spread=float(obs_row.avg_spread) if obs_row and obs_row.avg_spread else None,
        total_volume=float(trade_row.total_volume) if trade_row and trade_row.total_volume else None,
        vwap=float(trade_row.vwap) if trade_row and trade_row.vwap else None,
    )


@router.delete("/{recording_id}", status_code=204)
async def delete_recording(
    recording_id: UUID,
    db: AsyncSession = Depends(get_session),
    service: RecordingService = Depends(get_recording_service),
):
    """
    Delete a recording and all its data.

    This permanently removes the recording and all associated
    order book snapshots and trades.
    """
    recording = await service.get_recording(recording_id)
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")

    if recording.is_active:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete an active recording. Stop it first."
        )

    # Delete recording (cascades to snapshots and trades)
    stmt = select(Recording).where(Recording.id == recording_id)
    result = await db.execute(stmt)
    rec = result.scalar_one()
    await db.delete(rec)
    await db.commit()

    logger.info(f"Deleted recording {recording_id}")
