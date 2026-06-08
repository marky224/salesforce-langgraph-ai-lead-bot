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
