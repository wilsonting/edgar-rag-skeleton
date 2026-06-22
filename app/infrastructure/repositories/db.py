import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pgvector.psycopg import register_vector_async

_pool: AsyncConnectionPool | None = None

def _connection_string() -> str:
    url = os.environ["POSTGRES_DATABASE_URL"]
    # Normalize: psycopg accepts postgresql:// directly
    if url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql://", 1)
    return url

async def init_pool() -> AsyncConnectionPool:
    """Open the global connection pool. Call once at app startup."""
    global _pool
    if _pool is not None:
        return _pool

    async def _configure(conn: psycopg.AsyncConnection) -> None:
        await register_vector_async(conn)

    _pool = AsyncConnectionPool(
        conninfo=_connection_string(),
        min_size=2,
        max_size=10,
        kwargs={"row_factory": dict_row},
        configure=_configure,
        open=False,
    )
    await _pool.open()

    return _pool

async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None

@asynccontextmanager
async def get_connection() -> AsyncIterator[psycopg.AsyncConnection]:
    """Yield a pooled connection. Caller controls transaction boundaries."""
    if _pool is None:
        await init_pool()
    assert _pool is not None
    async with _pool.connection() as conn:
        yield conn