"""Checkpointer factory for the conversation graph.

Selects the LangGraph state-persistence backend at startup: a durable Neon
Postgres ``AsyncPostgresSaver`` when ``DATABASE_URL`` is configured, otherwise an
in-memory ``MemorySaver`` (local development plus the hermetic test suite, which
set no ``DATABASE_URL``).

The Postgres path opens a small ``AsyncConnectionPool`` (kept for the whole
process lifetime, closed cleanly on shutdown) rather than a single connection, so
the checkpointer survives the database dropping idle connections: serverless
Postgres (Neon) auto-suspends after a few minutes idle and terminates open
connections. The pool's ``check`` validates and recycles a dead connection before
handing it out, so the next turn transparently reconnects instead of 500ing.
``setup()`` performs idempotent table creation, so no separate migration step is
required.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from app.config import Settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def open_checkpointer(settings: Settings) -> AsyncIterator[BaseCheckpointSaver]:
    """Yield the graph checkpointer for the process lifetime.

    Parameters
    ----------
    settings : Settings
        Application settings.  ``settings.database_url`` decides the backend:
        Postgres when set, in-memory otherwise.

    Yields
    ------
    BaseCheckpointSaver
        An ``AsyncPostgresSaver`` (durable, survives cold starts and scales
        across replicas) when ``DATABASE_URL`` is configured, else a
        ``MemorySaver`` (single-process, lost on restart).
    """
    if settings.database_url:
        # Imported lazily so the postgres driver is only required in deployments
        # that actually configure a database (keeps local dev / tests zero-dep).
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg_pool import AsyncConnectionPool

        dsn = settings.database_url.get_secret_value()
        logger.info("Opening Postgres checkpointer (AsyncPostgresSaver over a pool)")
        # A reconnecting pool — NOT a single long-lived connection — so the
        # checkpointer survives Neon's idle auto-suspend (which terminates open
        # connections). ``check`` validates each connection on checkout and the
        # pool transparently replaces a dead one; ``max_idle`` retires extras
        # before Neon would. AsyncPostgresSaver needs autocommit; prepare_threshold=0
        # keeps it safe if the DSN is ever pointed at Neon's pooled (pgbouncer) host.
        pool = AsyncConnectionPool(
            conninfo=dsn,
            min_size=1,
            max_size=4,
            max_idle=120,
            open=False,
            check=AsyncConnectionPool.check_connection,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        await pool.open(wait=True, timeout=10)
        try:
            saver = AsyncPostgresSaver(pool)
            await saver.setup()  # idempotent DDL — safe to run on every boot
            yield saver
        finally:
            await pool.close()
    else:
        logger.info("No DATABASE_URL set — using in-memory MemorySaver checkpointer")
        yield MemorySaver()
