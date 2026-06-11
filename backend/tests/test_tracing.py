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


def test_build_tracer_uses_settings_api_key(monkeypatch):
    """A key present only in Settings/.env (not os.environ) still reaches the Client.

    A bare ``Client()`` reads only ``os.environ``, so without passing the key
    explicitly this would be ``None``. Clearing the env vars proves the value
    came from ``Settings``, not the implicit environment lookup.
    """
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    tracer = build_tracer(Settings(langsmith_tracing=True, langsmith_api_key="ls-from-settings"))
    assert isinstance(tracer, LangChainTracer)
    assert tracer.client.api_key == "ls-from-settings"
