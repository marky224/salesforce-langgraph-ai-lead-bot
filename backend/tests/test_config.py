"""
Config / LLM-factory tests (PR 3, C1 — LLM timeout + bounded retry).

These exercise the real `get_llm()` with dummy API keys; the chat models construct
without any network call, so the tests stay hermetic.
"""

from __future__ import annotations

import importlib

import pytest


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


def test_default_llm_provider_is_xai(monkeypatch):
    """With LLM_PROVIDER unset (and no .env), the bot runs xAI Grok, like prod."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    from app.config import LLMProvider, Settings

    assert Settings(_env_file=None).llm_provider is LLMProvider.XAI


# ---------------------------------------------------------------------------
# Every LLM_PROVIDER builds (provider packages bundled in requirements.txt)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("provider", "key_var", "module", "class_name", "model_attr", "model"),
    [
        ("anthropic", "ANTHROPIC_API_KEY", "langchain_anthropic", "ChatAnthropic", "model", "claude-sonnet-4-6"),
        ("openai", "OPENAI_API_KEY", "langchain_openai", "ChatOpenAI", "model_name", "gpt-4o"),
        ("groq", "GROQ_API_KEY", "langchain_groq", "ChatGroq", "model_name", "llama-3.3-70b-versatile"),
        ("xai", "XAI_API_KEY", "langchain_openai", "ChatOpenAI", "model_name", "grok-4.3"),
    ],
)
def test_every_provider_builds_its_default_model(
    monkeypatch, provider, key_var, module, class_name, model_attr, model
):
    """Each provider's package is installed and get_llm() builds its default model.

    CI installs requirements-dev.txt (-> requirements.txt), so a provider package
    dropped from requirements.txt fails here instead of at app startup.
    """
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv(key_var, "test-key-not-real")
    monkeypatch.setenv("LLM_MODEL", "")  # beat an LLM_MODEL from a local .env

    from app.config import get_llm, get_settings

    get_settings.cache_clear()
    llm = get_llm()

    assert type(llm) is getattr(importlib.import_module(module), class_name)
    assert getattr(llm, model_attr) == model


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
