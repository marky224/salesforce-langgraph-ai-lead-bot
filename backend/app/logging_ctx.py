"""
Request-scoped logging context.

Two contextvars — ``request_id`` and ``thread_id`` — are populated per HTTP
request (and per conversation turn) and injected into every log record by
``RequestContextFilter``, so a single request / conversation is traceable
end-to-end in the logs.  Contextvars are async-safe: each request the event
loop handles sees its own values with no cross-talk.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar

# "-" is the at-rest sentinel so columns stay aligned before a request sets a value.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
thread_id_var: ContextVar[str] = ContextVar("thread_id", default="-")


class RequestContextFilter(logging.Filter):
    """Inject the current request_id / thread_id onto every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        record.thread_id = thread_id_var.get()
        return True
