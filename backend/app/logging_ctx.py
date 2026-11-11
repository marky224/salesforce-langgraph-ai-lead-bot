"""
Request-scoped logging context.

Two contextvars — ``request_id`` and ``thread_id`` — are populated per HTTP
request (and per conversation turn) and injected into every log record by
``RequestContextFilter``, so a single request / conversation is traceable
end-to-end in the logs.  Contextvars are async-safe: each request the event
loop handles sees its own values with no cross-talk.

The formatters mask emails, phone numbers and a SOQL query string in a
record's traceback (see ``PiiMaskingFormatter``).
"""

from __future__ import annotations

import json
import logging
import re
from contextvars import ContextVar
from typing import Any

# "-" is the at-rest sentinel so columns stay aligned before a request sets a value.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
thread_id_var: ContextVar[str] = ContextVar("thread_id", default="-")

# A Salesforce SOQL query in a URL, as simple_salesforce sends it (.../query/?q=...).
# The lead lookup's query holds the visitor's email.
_SOQL_QUERY_RE = re.compile(r"(/query(?:All)?/\?q=)[^\s'\"]+")


class RequestContextFilter(logging.Filter):
    """Inject the current request_id / thread_id onto every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        record.thread_id = thread_id_var.get()
        return True


def mask_pii(text: str) -> str:
    """
    Mask emails and phone numbers in *text* with the LangSmith tracer's patterns,
    and drop a SOQL query string.

    The Salesforce lead lookup carries the email in its query URL, and a failed
    request's error quotes that URL, so a SOQL query string is dropped whole
    (an email the pattern doesn't match would otherwise stay). ``%40`` is then
    read as ``@``, for an email percent-encoded anywhere else. Best-effort,
    like the tracer: names, companies and chat text don't match.
    """
    # Imported here: app.tracing imports app.config, which imports this module.
    from app.tracing import _EMAIL_RE, _PHONE_RE

    text = _SOQL_QUERY_RE.sub(r"\1[QUERY]", text)
    text = _EMAIL_RE.sub("[EMAIL]", text.replace("%40", "@"))
    return _PHONE_RE.sub("[PHONE]", text)


class PiiMaskingFormatter(logging.Formatter):
    """
    ``logging.Formatter`` with the traceback masked by ``mask_pii``.

    An exception's text comes from Salesforce, the LLM provider or a parser and
    can quote lead fields, and the traceback is what a failure is debugged from,
    so it stays, masked. The record's message is not touched.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Another handler's formatter may have cached the traceback unmasked.
        record.exc_text = None
        return super().format(record)

    def formatException(self, ei: Any) -> str:
        return mask_pii(super().formatException(ei))


class JsonLogFormatter(PiiMaskingFormatter):
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
        # Optional structured telemetry attached via logger(..., extra={"llm_call": {...}}).
        llm_call = getattr(record, "llm_call", None)
        if llm_call is not None:
            payload["llm_call"] = llm_call
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)
