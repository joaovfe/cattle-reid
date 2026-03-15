"""
Async SQLAlchemy engine and session for PostgreSQL.
"""
from __future__ import annotations

import os
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from core.database.models import Base

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://cattle:cattle@localhost:5432/cattle_reid",
)


def get_engine(url: str | None = None):
    return create_async_engine(
        url or DATABASE_URL,
        echo=os.getenv("SQL_ECHO", "false").lower() == "true",
        poolclass=NullPool,
    )


def get_session_maker(engine=None):
    engine = engine or get_engine()
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )


async_session_factory = get_session_maker()


async def async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Create tables (for dev). In production use Alembic migrations."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
