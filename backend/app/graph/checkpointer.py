"""Checkpointer factory for the conversation graph.

Selects the LangGraph state-persistence backend at startup: a durable Neon
Postgres ``AsyncPostgresSaver`` when ``DATABASE_URL`` is configured, otherwise an
in-memory ``MemorySaver`` (local development plus the hermetic test suite, which
set no ``DATABASE_URL``).

The Postgres path is entered as an async context manager so the connection lives
for the whole process lifetime and is closed cleanly on shutdown; ``setup()``
performs idempotent table creation, so no separate migration step is required.
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

        dsn = settings.database_url.get_secret_value()
        logger.info("Opening Postgres checkpointer (AsyncPostgresSaver)")
        async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
            await saver.setup()  # idempotent DDL — safe to run on every boot
            yield saver
    else:
        logger.info("No DATABASE_URL set — using in-memory MemorySaver checkpointer")
        yield MemorySaver()
