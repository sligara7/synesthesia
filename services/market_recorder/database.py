"""
Database setup and session management.

Uses SQLAlchemy 2.0 async API with asyncpg driver for PostgreSQL/TimescaleDB.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from .config import get_settings


def create_engine(database_url: str | None = None):
    """Create async SQLAlchemy engine."""
    url = database_url or get_settings().database_url

    return create_async_engine(
        url,
        echo=get_settings().debug,
        poolclass=NullPool,  # Better for async with connection pooling at DB level
    )


def create_session_factory(engine) -> async_sessionmaker[AsyncSession]:
    """Create async session factory."""
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )


# Global engine and session factory (initialized on startup)
_engine = None
_session_factory = None


async def init_database(database_url: str | None = None):
    """Initialize database connection."""
    global _engine, _session_factory

    _engine = create_engine(database_url)
    _session_factory = create_session_factory(_engine)


async def close_database():
    """Close database connection."""
    global _engine

    if _engine:
        await _engine.dispose()
        _engine = None


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency injection for database sessions.

    Usage:
        @router.get("/")
        async def endpoint(db: AsyncSession = Depends(get_session)):
            ...
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_session_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions (for use outside FastAPI).

    Usage:
        async with get_session_context() as db:
            ...
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
