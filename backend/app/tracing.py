"""
LangSmith tracing wiring — a PII-masking tracer (emails + phone numbers), built only when enabled.

Tracing stays vendor-neutral and opt-in. When enabled we attach ONE explicit
``LangChainTracer`` whose client carries a PII anonymizer, passed via each
run's config callbacks (see ``server._run_config``). LangChain's callback
manager only auto-adds an env-based tracer when no ``LangChainTracer`` is already
present, so our masked tracer suppresses the un-masked env one.

Masking is best-effort and covers run inputs and outputs only: emails, plus common
North American and "+"-prefixed phone numbers. Not masked: 7-digit local numbers,
international numbers without a leading "+" (e.g. "020 7946 0958"), and a run's
error text, which LangSmith uploads as-is.

No new dependency: ``langsmith`` is already transitive via ``langchain-core``.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.config import Settings

logger = logging.getLogger(__name__)

# Visitor emails are real PII — mask them before any run I/O leaves the process.
# Bounded parts (RFC 5321 limits) keep the scan linear: an unbounded local part made
# the regex quadratic on long text with no "@", and this runs on the event loop.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]{1,255}\.[A-Za-z]{2,24}")

# Phone numbers (optional lead field, and visitors type them into chat). Two shapes:
#  1. North American: optional leading 1, area code bare or in parens, 3-3-4 digits.
#     A separator is a dot, dash, en/em dash or nothing, with an optional space on
#     either side — "(210) 555-0123", "(210)-555-0123", "210 - 555 - 0123",
#     "1-800-555-0123", "2105550123".
#  2. International, only with a leading "+": "+44 20 7946 0958", "+1 (210) 555-0123".
# Either may end in an extension ("x45", "ext. 12"), masked with the number.
# Boundaries keep it off longer digit runs, decimals and IDs; dates (4-2-2) and UUID
# segments don't fit 3-3-4. Seven-digit local numbers ("555-0123") are not masked.
# Bounded quantifiers only, so no catastrophic backtracking on visitor text.
_PHONE_SEP = r"\s?[.\-\u2013\u2014]?\s?"
_PHONE_EXT = r"(?:\s?(?i:x|ext\.?)\s?\d{1,6})?"
_PHONE_RE = re.compile(
    r"(?<![\w.+])(?:"
    r"\+\d[\d\s().\-\u2013\u2014]{6,18}\d"
    rf"|(?:1{_PHONE_SEP})?(?:\(\d{{3}}\)|\d{{3}}){_PHONE_SEP}\d{{3}}{_PHONE_SEP}\d{{4}}"
    rf"){_PHONE_EXT}(?![\w]|\.\d)"
)


def build_email_anonymizer() -> Any:
    """Return a LangSmith anonymizer callable that masks emails and phone numbers in run I/O.

    Emails go first, so an email-shaped address like ``2105550123@txt.example`` becomes
    ``[EMAIL]`` rather than a half-masked ``[PHONE]@...``.
    """
    from langsmith.anonymizer import create_anonymizer

    return create_anonymizer(
        [
            {"pattern": _EMAIL_RE, "replace": "[EMAIL]"},
            {"pattern": _PHONE_RE, "replace": "[PHONE]"},
        ]
    )


def build_tracer(settings: Settings) -> Any | None:
    """
    Return a PII-masking ``LangChainTracer`` when tracing is enabled, else ``None``.

    Enabled via ``LANGSMITH_TRACING`` (canonical) or the legacy
    ``LANGCHAIN_TRACING_V2`` (see ``Settings.tracing_enabled``). The tracer's
    ``Client`` reads ``LANGSMITH_API_KEY`` / endpoint from the environment;
    ``LANGSMITH_PROJECT`` names the project.
    """
    if not settings.tracing_enabled:
        return None

    try:
        from langchain_core.tracers import LangChainTracer
        from langsmith import Client
    except ImportError:
        logger.warning("Tracing enabled but the LangSmith tracer is unavailable; skipping")
        return None

    # Pass the configured key explicitly: pydantic loads it into Settings (from the
    # environment OR the .env file), but a bare Client() only reads os.environ — which
    # the .env file never populates. Without this, a key set only in .env silently
    # fails to upload (and the None check below wouldn't fire). See 13-observability.md.
    api_key = (
        settings.langsmith_api_key.get_secret_value()
        if settings.langsmith_api_key is not None
        else None
    )
    if api_key is None:
        logger.warning("Tracing enabled but LANGSMITH_API_KEY is unset — traces won't upload")

    client = Client(api_key=api_key, anonymizer=build_email_anonymizer())
    tracer = LangChainTracer(client=client, project_name=settings.langsmith_project)
    logger.info(
        "LangSmith tracing enabled (project=%s, email + phone masking on)",
        settings.langsmith_project or "default",
    )
    return tracer
