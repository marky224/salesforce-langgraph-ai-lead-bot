"""
LangSmith tracing wiring — an email-masking tracer, built only when enabled.

Tracing stays vendor-neutral and opt-in. When enabled we attach ONE explicit
``LangChainTracer`` whose client carries an email anonymizer, passed via each
run's config callbacks (see ``server._run_config``). LangChain's callback
manager only auto-adds an env-based tracer when no ``LangChainTracer`` is already
present, so our masked tracer suppresses the un-masked env one — visitor emails
(real PII) never reach LangSmith.

No new dependency: ``langsmith`` is already transitive via ``langchain-core``.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.config import Settings

logger = logging.getLogger(__name__)

# Visitor emails are real PII — mask them before any run I/O leaves the process.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def build_email_anonymizer() -> Any:
    """Return a LangSmith anonymizer callable that masks email addresses in run I/O."""
    from langsmith.anonymizer import create_anonymizer

    return create_anonymizer([{"pattern": _EMAIL_RE, "replace": "[EMAIL]"}])


def build_tracer(settings: Settings) -> Any | None:
    """
    Return an email-masking ``LangChainTracer`` when tracing is enabled, else ``None``.

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
        "LangSmith tracing enabled (project=%s, email masking on)",
        settings.langsmith_project or "default",
    )
    return tracer
