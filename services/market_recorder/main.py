"""
FastAPI application entry point for the Market Recorder service.

Run with:
    uvicorn services.market_recorder.main:app --reload

Or as a module:
    python -m services.market_recorder.main
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .database import init_database, close_database


def setup_logging():
    """Configure logging for the application."""
    settings = get_settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Reduce noise from libraries
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    setup_logging()
    logger = logging.getLogger(__name__)

    # Startup
    settings = get_settings()
    logger.info(f"Starting Market Recorder on {settings.host}:{settings.port}")
    logger.info(f"Database: {settings.database_url.split('@')[-1]}")  # Hide credentials
    logger.info(f"Default symbols: {settings.symbols_list}")

    await init_database()
    logger.info("Database connection established")

    yield

    # Shutdown
    logger.info("Shutting down Market Recorder...")

    # Stop any active streaming session
    from .api.streaming import _streaming_session
    if _streaming_session is not None and _streaming_session.is_running():
        logger.info("Stopping active streaming session...")
        await _streaming_session.stop()

    await close_database()
    logger.info("Database connection closed")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Market Recorder API",
        description=(
            "Record and replay Binance market data with real-time S/R penetration signals.\n\n"
            "## Features\n"
            "- Multi-symbol concurrent streaming (default: top 5 cryptos)\n"
            "- Real-time S/R penetration signal detection\n"
            "- TimescaleDB storage for efficient time-series queries\n"
            "- SSE streaming for live signal updates\n"
            "- Enriched replay with computed analytics\n"
        ),
        version="0.2.0",
        lifespan=lifespan,
        debug=settings.debug,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check endpoint
    @app.get("/health", tags=["system"])
    async def health_check():
        """Health check endpoint for load balancers and monitoring."""
        return {"status": "healthy", "service": "market-recorder", "version": "0.2.0"}

    # Root endpoint
    @app.get("/", tags=["system"])
    async def root():
        """Root endpoint with API info."""
        return {
            "service": "market-recorder",
            "version": "0.2.0",
            "docs": "/docs",
            "endpoints": {
                "recordings": "/recordings",
                "replay": "/replay",
                "streaming": "/streaming",
            },
            "default_symbols": settings.symbols_list,
        }

    # Register routers
    from .api import recordings, replay, streaming
    app.include_router(recordings.router)
    app.include_router(replay.router)
    app.include_router(streaming.router)

    return app


# Application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "services.market_recorder.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
