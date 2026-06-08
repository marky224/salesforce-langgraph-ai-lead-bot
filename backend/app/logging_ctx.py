"""
Request-scoped logging context.

Two contextvars — ``request_id`` and ``thread_id`` — are populated per HTTP
request (and per conversation turn) and injected into every log record by
``RequestContextFilter``, so a single request / conversation is traceable
end-to-end in the logs.  Contextvars are async-safe: each request the event
loop handles sees its own values with no cross-talk.
"""

from __future__ import annotations

import json
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


class JsonLogFormatter(logging.Formatter):
    """Emit one JSON object per log record (structured logs for Log Analytics)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
            "thread_id": getattr(record, "thread_id", "-"),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)
