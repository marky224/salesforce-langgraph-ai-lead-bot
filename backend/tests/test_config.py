"""
Config / LLM-factory tests (PR 3, C1 — LLM timeout + bounded retry).

These exercise the real `get_llm()` under `LLM_PROVIDER=openai` (the only provider
package installed); `ChatOpenAI` constructs without any network call, so the tests
stay hermetic.
"""

from __future__ import annotations


def test_llm_builder_applies_timeout_and_retries(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")

    from app.config import get_llm, get_settings

    get_settings.cache_clear()
    llm = get_llm()

    # `timeout` is the constructor alias for ChatOpenAI's `request_timeout`.
    assert llm.request_timeout == 60.0
    assert llm.max_retries == 2


def test_llm_timeout_and_retries_env_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("LLM_MAX_RETRIES", "5")

    from app.config import get_llm, get_settings

    get_settings.cache_clear()
    llm = get_llm()

    assert llm.request_timeout == 12.5
    assert llm.max_retries == 5


# ---------------------------------------------------------------------------
# Observability: JSON logs + tracing flag (PR 4, C3)
# ---------------------------------------------------------------------------

def test_build_formatter_selects_json_vs_text():
    from app.config import _build_formatter
    from app.logging_ctx import JsonLogFormatter

    assert isinstance(_build_formatter("json"), JsonLogFormatter)
    assert isinstance(_build_formatter("JSON"), JsonLogFormatter)  # case-insensitive
    assert not isinstance(_build_formatter("text"), JsonLogFormatter)


def test_json_formatter_emits_structured_fields():
    import json
    import logging

    from app.logging_ctx import JsonLogFormatter

    record = logging.LogRecord("app.x", logging.INFO, __file__, 1, "hello", None, None)
    record.request_id = "req-1"
    record.thread_id = "thread-9"
    out = json.loads(JsonLogFormatter().format(record))

    assert out["message"] == "hello"
    assert out["level"] == "INFO"
    assert out["logger"] == "app.x"
    assert out["request_id"] == "req-1"
    assert out["thread_id"] == "thread-9"


def test_tracing_flag_parsed_from_env(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

    from app.config import get_settings

    get_settings.cache_clear()
    assert get_settings().langchain_tracing_v2 is True
