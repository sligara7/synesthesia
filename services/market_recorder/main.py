"""
FastAPI application entry point for the Market Recorder service.

Run with:
    uvicorn services.market_recorder.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .database import init_database, close_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    settings = get_settings()
    print(f"Starting Market Recorder on {settings.host}:{settings.port}")
    print(f"Database: {settings.database_url.split('@')[-1]}")  # Hide credentials

    await init_database()
    print("Database connection established")

    yield

    # Shutdown
    await close_database()
    print("Database connection closed")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Market Recorder API",
        description="Record and replay Binance market data for backtesting and analysis",
        version="0.1.0",
        lifespan=lifespan,
        debug=settings.debug,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "market-recorder"}

    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API info."""
        return {
            "service": "market-recorder",
            "version": "0.1.0",
            "docs": "/docs",
            "endpoints": {
                "recordings": "/recordings",
                "replay": "/replay",
                "websocket": "/ws",
            }
        }

    # Register routers
    from .api import recordings, replay
    app.include_router(recordings.router)
    app.include_router(replay.router)
    # WebSocket routes will be added later
    # from .api import websocket
    # app.include_router(websocket.router)

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
