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


def test_mask_pii_masks_emails_also_percent_encoded_and_phones():
    """What both formatters apply to a traceback (the LangSmith tracer's patterns)."""
    from app.logging_ctx import mask_pii

    masked = mask_pii(
        "to=ada%40acme-corp.example for ada@acme-corp.example, phone (210) 555-0123"
    )

    assert "acme-corp.example" not in masked
    assert "555-0123" not in masked
    assert masked.count("[EMAIL]") == 2
    assert "[PHONE]" in masked


def test_mask_pii_drops_the_lead_lookups_query():
    """
    The lookup's URL carries the email in its SOQL, and an email the pattern doesn't
    match (here one with no top-level domain) would stay, so the whole query goes.
    """
    from app.logging_ctx import mask_pii

    masked = mask_pii(
        "Max retries exceeded with url: /services/data/v59.0/query/?q=SELECT+Id+FROM+Lead+WHERE"
        "+Email+%3D+%27augusta%40analytical-engines%27+ORDER+BY+CreatedDate+DESC+LIMIT+1 "
        "(Caused by NameResolutionError)"
    )

    assert masked == (
        "Max retries exceeded with url: /services/data/v59.0/query/?q=[QUERY] "
        "(Caused by NameResolutionError)"
    )


def test_configure_logging_quiets_httpx2_loggers():
    """The anthropic SDK logs through httpx2/httpcore2; quiet them like httpx/httpcore."""
    import logging

    from app.config import configure_logging

    for name in ("httpx2", "httpcore2"):
        logging.getLogger(name).setLevel(logging.NOTSET)  # don't rely on an earlier test

    configure_logging()

    assert logging.getLogger("httpx2").level == logging.WARNING
    assert logging.getLogger("httpcore2").level == logging.WARNING


# LOG_LEVEL=DEBUG must not make a provider SDK log its requests: each SDK's
# ``<sdk>._base_client`` logger writes "Request options: ..." at DEBUG, the whole
# request body, so the prompt, the transcript and the visitor's message. Its INFO
# line, "Retrying request to <path> in N seconds", stays at LOG_LEVEL=INFO: it is
# the only record of a model call that was retried. Above INFO, LOG_LEVEL quiets
# it like any other INFO line.

# A chat completion as the mocked provider returns it.
_CHAT_COMPLETION = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 0,
    "model": "grok-4.3",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


@pytest.fixture
def restore_logging():
    """configure_logging() replaces the root logger's handlers and level: put them back."""
    import logging

    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.mark.parametrize(
    ("log_level", "sdk_level"),
    [("DEBUG", "INFO"), ("INFO", "INFO"), ("WARNING", "WARNING"), ("ERROR", "ERROR")],
)
@pytest.mark.parametrize("sdk", ["openai", "anthropic", "groq"])
def test_configure_logging_keeps_the_provider_sdk_loggers_at_info_or_above(
    sdk, log_level, sdk_level, monkeypatch, restore_logging
):
    """
    Held at INFO when LOG_LEVEL=DEBUG (no request dump; the retry line stays),
    and following LOG_LEVEL above INFO.
    """
    import logging

    from app.config import configure_logging, get_settings

    monkeypatch.setenv("LOG_LEVEL", log_level)
    get_settings.cache_clear()
    logging.getLogger(sdk).setLevel(logging.NOTSET)  # don't rely on an earlier test

    configure_logging()

    assert logging.getLevelName(logging.getLogger().level) == log_level
    assert logging.getLevelName(logging.getLogger(sdk).level) == sdk_level
    assert not logging.getLogger(f"{sdk}._base_client").isEnabledFor(logging.DEBUG)
    for name in ("httpx", "httpcore", "httpx2", "httpcore2", "urllib3", "asyncio"):
        assert logging.getLogger(name).level == logging.WARNING, name


@pytest.mark.asyncio
async def test_debug_logging_never_holds_the_llm_request(monkeypatch, restore_logging):
    """
    LOG_LEVEL=DEBUG, then a real ChatOpenAI (the xai provider's client) over a
    mocked transport, no network: no log record holds the visitor's message.
    """
    import logging

    import httpx
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    from app.config import configure_logging, get_settings

    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    get_settings.cache_clear()
    for name in ("openai", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.NOTSET)  # don't rely on an earlier test
    configure_logging()
    assert logging.getLogger().level == logging.DEBUG

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_CHAT_COMPLETION)

    capture = _Capture(level=logging.DEBUG)
    logging.getLogger().addHandler(capture)
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            llm = ChatOpenAI(
                model="grok-4.3",
                api_key="test-key",
                base_url="https://api.x.ai/v1",
                max_retries=0,
                http_async_client=client,
            )
            reply = await llm.ainvoke(
                [HumanMessage(content="I'm Augusta Lovelace of Analytical Engines")]
            )
    finally:
        logging.getLogger().removeHandler(capture)

    assert reply.content == "ok"  # the request did go through the SDK
    logged = "\n".join(record.getMessage() for record in records)
    assert "Augusta Lovelace" not in logged
    assert "Analytical Engines" not in logged


async def _retried_chat_call(sdk: str, log_level: str, monkeypatch) -> tuple[list[str], str]:
    """
    configure_logging() at *log_level*, then a real chat model over a mocked
    transport, no network: a 429, then a 200. Returns the messages the SDK's
    ``_base_client`` logged and every message logged. ChatAnthropic takes no
    HTTP client, so the anthropic logger is covered by the level test only.
    """
    import logging

    import httpx
    from langchain_core.messages import HumanMessage
    from langchain_groq import ChatGroq
    from langchain_openai import ChatOpenAI

    from app.config import configure_logging, get_settings

    monkeypatch.setenv("LOG_LEVEL", log_level)
    get_settings.cache_clear()
    logging.getLogger(sdk).setLevel(logging.NOTSET)  # don't rely on an earlier test
    configure_logging()

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    sent: list[httpx.Request] = []
    replies = [
        # retry-after-ms sets the SDK's wait before the retry: 1 ms.
        httpx.Response(429, headers={"retry-after-ms": "1"}, json={"error": {"message": "slow down"}}),
        httpx.Response(200, json=_CHAT_COMPLETION),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return replies[len(sent) - 1]

    capture = _Capture(level=logging.DEBUG)
    logging.getLogger().addHandler(capture)
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            if sdk == "openai":  # the xai provider's client
                llm = ChatOpenAI(
                    model="grok-4.3",
                    api_key="test-key",
                    base_url="https://api.x.ai/v1",
                    max_retries=1,
                    http_async_client=client,
                )
            else:
                llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    api_key="test-key",
                    max_retries=1,
                    http_async_client=client,
                )
            reply = await llm.ainvoke(
                [HumanMessage(content="I'm Augusta Lovelace of Analytical Engines")]
            )
    finally:
        logging.getLogger().removeHandler(capture)

    assert reply.content == "ok"
    assert len(sent) == 2  # the 429, then the retry
    sdk_messages = [r.getMessage() for r in records if r.name == f"{sdk}._base_client"]
    return sdk_messages, "\n".join(record.getMessage() for record in records)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sdk", "path"), [("openai", "/chat/completions"), ("groq", "/openai/v1/chat/completions")]
)
async def test_a_retried_llm_call_logs_the_sdks_retry_line_at_info(
    sdk, path, monkeypatch, restore_logging
):
    """
    At LOG_LEVEL=INFO (the default) the SDK's retry line is the only record of
    the retry (httpx's request line is quiet), and it holds the path and the
    delay, not the request.
    """
    sdk_messages, logged = await _retried_chat_call(sdk, "INFO", monkeypatch)

    assert sdk_messages == [f"Retrying request to {path} in 0.001000 seconds"]
    assert "Augusta Lovelace" not in logged


@pytest.mark.asyncio
@pytest.mark.parametrize("sdk", ["openai", "groq"])
async def test_a_retried_llm_call_logs_no_retry_line_at_log_level_warning(
    sdk, monkeypatch, restore_logging
):
    """LOG_LEVEL=WARNING quiets the SDK's INFO retry line, as it does the app's own."""
    sdk_messages, logged = await _retried_chat_call(sdk, "WARNING", monkeypatch)

    assert sdk_messages == []
    assert "Retrying request to" not in logged


def test_tracing_flag_parsed_from_env(monkeypatch):
    from langsmith import utils as ls_utils

    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    ls_utils.get_env_var.cache_clear()  # langsmith caches env reads

    from app.config import get_settings

    get_settings.cache_clear()
    try:
        assert get_settings().langchain_tracing_v2 is True
    finally:
        ls_utils.get_env_var.cache_clear()  # don't leave "tracing on" cached for later tests


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
