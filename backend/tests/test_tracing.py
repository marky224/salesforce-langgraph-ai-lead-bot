"""Tests for LangSmith tracing wiring (PR A2b)."""

from __future__ import annotations

from langchain_core.tracers import LangChainTracer

from app.config import Settings
from app.tracing import build_email_anonymizer, build_tracer


def test_email_anonymizer_masks_addresses():
    anon = build_email_anonymizer()
    assert anon("reach me at jane.doe@acme.com please") == "reach me at [EMAIL] please"
    # walks nested structures (run inputs/outputs are dicts/lists)
    assert anon({"lead": {"email": "x@y.io"}, "note": "no email"}) == {
        "lead": {"email": "[EMAIL]"},
        "note": "no email",
    }


def test_build_tracer_none_when_disabled():
    # Init kwargs override any local .env, so this is hermetic.
    settings = Settings(langsmith_tracing=False, langchain_tracing_v2=False)
    assert build_tracer(settings) is None


def test_build_tracer_built_when_enabled(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-not-real")
    tracer = build_tracer(Settings(langsmith_tracing=True))
    assert isinstance(tracer, LangChainTracer)
