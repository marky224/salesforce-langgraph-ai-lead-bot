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


def test_configure_logging_quiets_httpx2_loggers():
    """The anthropic SDK logs through httpx2/httpcore2; quiet them like httpx/httpcore."""
    import logging

    from app.config import configure_logging

    for name in ("httpx2", "httpcore2"):
        logging.getLogger(name).setLevel(logging.NOTSET)  # don't rely on an earlier test

    configure_logging()

    assert logging.getLogger("httpx2").level == logging.WARNING
    assert logging.getLogger("httpcore2").level == logging.WARNING


def test_tracing_flag_parsed_from_env(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

    from app.config import get_settings

    get_settings.cache_clear()
    assert get_settings().langchain_tracing_v2 is True


# ---------------------------------------------------------------------------
# Site index settings (TARS site questions)
# ---------------------------------------------------------------------------

_SITE_ENV = ("SITE_INDEX_URL", "SITE_INDEX_REFRESH_SECONDS", "SITE_SEARCH_MIN_SCORE")


def test_site_index_defaults(monkeypatch):
    """Unset, the bot loads the published index hourly with the calibrated threshold."""
    for var in _SITE_ENV:
        monkeypatch.delenv(var, raising=False)

    from app.config import Settings

    settings = Settings(_env_file=None)
    assert settings.site_index_url == "https://www.markandrewmarquez.com/site-index.json"
    assert settings.site_index_refresh_seconds == 3600
    assert settings.site_search_min_score == 0.4


def test_site_index_env_overrides(monkeypatch):
    monkeypatch.setenv("SITE_INDEX_URL", "https://example.org/index.json")
    monkeypatch.setenv("SITE_INDEX_REFRESH_SECONDS", "600")
    monkeypatch.setenv("SITE_SEARCH_MIN_SCORE", "0.55")

    from app.config import Settings

    settings = Settings(_env_file=None)
    assert settings.site_index_url == "https://example.org/index.json"
    assert settings.site_index_refresh_seconds == 600
    assert settings.site_search_min_score == 0.55


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_site_index_url_turns_retrieval_off(monkeypatch, value):
    monkeypatch.setenv("SITE_INDEX_URL", value)

    from app.config import Settings

    assert Settings(_env_file=None).site_index_url == ""


@pytest.mark.parametrize(
    ("var", "value"),
    [
        ("SITE_INDEX_URL", "http://www.markandrewmarquez.com/site-index.json"),
        ("SITE_INDEX_URL", "file:///etc/hosts"),
        ("SITE_INDEX_URL", "https://"),
        ("SITE_INDEX_URL", "www.markandrewmarquez.com/site-index.json"),
        ("SITE_INDEX_REFRESH_SECONDS", "30"),
        ("SITE_SEARCH_MIN_SCORE", "1.5"),
        ("SITE_SEARCH_MIN_SCORE", "-0.1"),
    ],
)
def test_site_index_settings_reject_bad_values(monkeypatch, var, value):
    monkeypatch.setenv(var, value)

    from pydantic import ValidationError

    from app.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_test_suite_never_points_at_a_real_site_index():
    """conftest blanks SITE_INDEX_URL, so no test lifespan fetches the index."""
    from app.config import get_settings

    assert get_settings().site_index_url == ""
