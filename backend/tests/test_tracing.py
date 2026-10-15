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


def test_anonymizer_masks_phone_numbers():
    anon = build_email_anonymizer()
    for phone in (
        "(210) 555-0123",
        "210-555-0123",
        "210.555.0123",
        "210 555 0123",
        "2105550123",
        "1-800-555-0123",
        "+1 (210) 555-0123",
        "+1 210 555 0123",
        "+44 20 7946 0958",
    ):
        assert anon(f"call me at {phone} today") == "call me at [PHONE] today", phone
    # the lead's phone field, nested like run I/O
    assert anon({"lead_data": {"phone": "210-555-0123", "email": "x@y.io"}}) == {
        "lead_data": {"phone": "[PHONE]", "email": "[EMAIL]"}
    }
    # an email made of digits stays one [EMAIL], not [PHONE]@...
    assert anon("text 2105550123@txt.example.com") == "text [EMAIL]"


def test_anonymizer_leaves_non_phone_numbers():
    anon = build_email_anonymizer()
    for text in (
        "budget 50K-100K, timeline 2026-10-14",
        "score 87 of 100, 60 messages max",
        "cost_usd 0.001206",
        "thread 3f2a9c4e-1b7d-4e8a-9c3f-2d1e0b9a8c7f",
        "lead 00Q5g00000ABcdEFGH",
        "order 12345678901234",
        "call 555-0123",
    ):
        assert anon(text) == text, text


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
