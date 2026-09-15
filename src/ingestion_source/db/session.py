"""Async database engine and session management for LegalTrace.

Supports PostgreSQL via asyncpg and SQLite via aiosqlite (for lightweight testing).
Uses SQLAlchemy 2.0 DeclarativeBase and async_sessionmaker patterns.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base declarative class for all SQLAlchemy ORM models."""
    pass


_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def init_db(
    database_url: str,
    echo: bool = False,
    pool_size: int = 10,
    max_overflow: int = 20,
) -> AsyncEngine:
    """Initialize the async SQLAlchemy engine and sessionmaker.

    Args:
        database_url: Async connection string (e.g. postgresql+asyncpg://... or sqlite+aiosqlite://...).
        echo: Whether to log SQL statements.
        pool_size: Connection pool size (ignored for SQLite).
        max_overflow: Max overflow connections beyond pool_size.

    Returns:
        The initialized AsyncEngine instance.
    """
    global _engine, _session_factory

    engine_kwargs: dict = {"echo": echo}

    # SQLite does not support pool_size or max_overflow
    if "sqlite" not in database_url.lower():
        engine_kwargs.update({
            "pool_size": pool_size,
            "max_overflow": max_overflow,
            "pool_pre_ping": True,
        })

    _engine = create_async_engine(database_url, **engine_kwargs)
    _session_factory = async_sessionmaker(
        bind=_engine,
        expire_on_commit=False,
        class_=AsyncSession,
    )
    logger.info("Initialized database engine with URL: %s", database_url.split("@")[-1] if "@" in database_url else database_url)
    return _engine


def get_engine() -> AsyncEngine:
    """Get the current global AsyncEngine instance, raising RuntimeError if not initialized."""
    if _engine is None:
        raise RuntimeError("Database engine has not been initialized. Call init_db() first.")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get the global async_sessionmaker instance."""
    if _session_factory is None:
        raise RuntimeError("Session factory has not been initialized. Call init_db() first.")
    return _session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for obtaining an isolated AsyncSession.

    Yields:
        An active AsyncSession instance. Commits on successful exit, rollbacks on exception.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_tables(engine: Optional[AsyncEngine] = None) -> None:
    """Create all tables registered with Base.metadata.

    Args:
        engine: Optional explicit engine. Uses global engine if omitted.
    """
    target_engine = engine or get_engine()
    async with target_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables verified/created successfully.")


async def drop_tables(engine: Optional[AsyncEngine] = None) -> None:
    """Drop all tables registered with Base.metadata (useful for test teardown)."""
    target_engine = engine or get_engine()
    async with target_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.warning("All database tables dropped.")


async def close_db() -> None:
    """Dispose of the global database engine connection pool."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database engine disposed.")
