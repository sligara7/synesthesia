"""
API endpoints for replaying recorded market data.

Provides SSE streaming for real-time replay and batch export for ML training.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import orjson

from ..database import get_session
from ..services.replayer import ReplayService
from ..schemas.replay import (
    ReplayFrame,
    ReplayBatchResponse,
    ReplayStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/replay", tags=["replay"])


def get_replay_service(db: AsyncSession = Depends(get_session)) -> ReplayService:
    """Get replay service instance."""
    return ReplayService(db)


@router.get("/{recording_id}/stream")
async def replay_stream(
    recording_id: UUID,
    start_time: Optional[datetime] = Query(None, description="Start time for replay"),
    end_time: Optional[datetime] = Query(None, description="End time for replay"),
    downsample_ms: int = Query(100, ge=10, le=60000, description="Time resolution in ms"),
    speed: float = Query(1.0, ge=0.1, le=100.0, description="Playback speed (1.0 = realtime)"),
    include_order_book: bool = Query(True, description="Include order book data"),
    include_trades: bool = Query(True, description="Include trade data"),
    service: ReplayService = Depends(get_replay_service),
):
    """
    Stream replay data as Server-Sent Events (SSE).

    Connect to this endpoint to receive real-time replay of recorded data.
    The stream will respect the playback speed setting for realistic replay.

    Example usage with JavaScript:
    ```javascript
    const eventSource = new EventSource('/replay/{id}/stream?speed=2.0');
    eventSource.onmessage = (event) => {
        const frame = JSON.parse(event.data);
        console.log(frame);
    };
    ```
    """
    # Verify recording exists
    recording = await service.get_recording(recording_id)
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")

    async def event_stream():
        """Generate SSE events."""
        last_time = None
        frame_count = 0

        try:
            async for frame in service.replay(
                recording_id=recording_id,
                start_time=start_time,
                end_time=end_time,
                downsample_ms=downsample_ms,
                include_order_book=include_order_book,
                include_trades=include_trades,
            ):
                # Apply speed factor for realistic replay timing
                if last_time is not None and speed < 100:
                    time_diff = (frame.timestamp - last_time).total_seconds()
                    delay = time_diff / speed
                    if delay > 0:
                        # Cap delay at 2 seconds to prevent long waits
                        await asyncio.sleep(min(delay, 2.0))

                last_time = frame.timestamp
                frame_count += 1

                # Serialize frame to JSON
                frame_json = orjson.dumps(frame.model_dump(mode="json")).decode()

                # SSE format: data: {json}\n\n
                yield f"data: {frame_json}\n\n"

            # Send end event
            yield f"event: end\ndata: {{\"frames\": {frame_count}}}\n\n"

        except asyncio.CancelledError:
            logger.info(f"Replay stream cancelled for {recording_id}")
        except Exception as e:
            logger.error(f"Replay stream error: {e}")
            yield f"event: error\ndata: {{\"error\": \"{str(e)}\"}}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@router.get("/{recording_id}/batch", response_model=ReplayBatchResponse)
async def replay_batch(
    recording_id: UUID,
    start_time: Optional[datetime] = Query(None, description="Start time for replay"),
    end_time: Optional[datetime] = Query(None, description="End time for replay"),
    downsample_ms: int = Query(1000, ge=100, le=60000, description="Time resolution in ms"),
    limit: int = Query(10000, ge=1, le=100000, description="Maximum frames to return"),
    include_order_book: bool = Query(True, description="Include order book data"),
    include_trades: bool = Query(True, description="Include trade data"),
    service: ReplayService = Depends(get_replay_service),
):
    """
    Get batch of replay data for ML training or analysis.

    Returns all frames in the specified time range, downsampled to the
    requested resolution. Use this endpoint for bulk data export.

    For large datasets, consider using smaller time ranges or higher
    downsample values to reduce response size.
    """
    # Verify recording exists
    recording = await service.get_recording(recording_id)
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")

    frames = await service.get_batch(
        recording_id=recording_id,
        start_time=start_time,
        end_time=end_time,
        downsample_ms=downsample_ms,
        limit=limit,
        include_order_book=include_order_book,
        include_trades=include_trades,
    )

    # Get actual time range from frames
    actual_start = frames[0].timestamp if frames else None
    actual_end = frames[-1].timestamp if frames else None

    return ReplayBatchResponse(
        recording_id=recording_id,
        frames=frames,
        total_frames=len(frames),
        start_time=actual_start,
        end_time=actual_end,
        downsample_ms=downsample_ms,
    )


@router.get("/{recording_id}/info", response_model=ReplayStatus)
async def replay_info(
    recording_id: UUID,
    downsample_ms: int = Query(100, ge=10, le=60000, description="Time resolution for frame count"),
    service: ReplayService = Depends(get_replay_service),
):
    """
    Get information about a recording for replay planning.

    Returns the time range and estimated frame count at the specified
    resolution. Use this to plan replay parameters.
    """
    # Verify recording exists
    recording = await service.get_recording(recording_id)
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")

    first_time, last_time = await service.get_recording_time_range(recording_id)
    frame_count = await service.get_frame_count(recording_id, downsample_ms=downsample_ms)

    return ReplayStatus(
        recording_id=recording_id,
        is_playing=False,
        current_time=first_time,
        progress_percent=0.0,
        frames_sent=frame_count,  # Using as total frames
        speed=1.0,
    )


@router.get("/{recording_id}/frames/{frame_number}", response_model=ReplayFrame)
async def get_single_frame(
    recording_id: UUID,
    frame_number: int,
    downsample_ms: int = Query(100, ge=10, le=60000, description="Time resolution in ms"),
    service: ReplayService = Depends(get_replay_service),
):
    """
    Get a single frame by frame number.

    Useful for random access to specific points in the recording.
    Frame numbers are 1-indexed.
    """
    # Verify recording exists
    recording = await service.get_recording(recording_id)
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")

    if frame_number < 1:
        raise HTTPException(status_code=400, detail="Frame number must be >= 1")

    # Stream through frames until we reach the requested one
    current_frame = 0
    async for frame in service.replay(
        recording_id=recording_id,
        downsample_ms=downsample_ms,
    ):
        current_frame += 1
        if current_frame == frame_number:
            return frame

    raise HTTPException(
        status_code=404,
        detail=f"Frame {frame_number} not found (recording has {current_frame} frames)"
    )
