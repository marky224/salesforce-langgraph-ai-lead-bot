"""
Shared pytest configuration and fixtures.
"""

from __future__ import annotations

import os
import sys
import time
import weakref

import pytest

# Ensure the backend app is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set test environment variables before any app imports
os.environ.setdefault("LLM_PROVIDER", "anthropic")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-not-real")
os.environ.setdefault("LOG_LEVEL", "WARNING")
# Never fetch the real site index: an empty URL turns retrieval off. Forced, not
# setdefault, so a shell or .env value can't make the suite go online; tests
# that exercise the loader set their own URL and fake the fetch.
os.environ["SITE_INDEX_URL"] = ""
# A test's LangSmith client gets empty server info (langsmith_offline below), which
# has no compression flag, so its background thread would warn about compression.
os.environ["LANGSMITH_DISABLE_RUN_COMPRESSION"] = "true"


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear the settings LRU cache between tests."""
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def clear_sf_cache():
    """Clear the Salesforce client cache between tests."""
    try:
        from app.tools.salesforce import reset_sf_client

        reset_sf_client()
    except ImportError:
        pass
    yield
    try:
        from app.tools.salesforce import reset_sf_client

        reset_sf_client()
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def langsmith_offline(monkeypatch):
    """
    Every langsmith Client starts with empty server info, so none fetches GET /info.

    A Client's background thread reads ``Client.info`` as soon as it starts, which
    sends GET /info unless the info was passed in. Passing it at construction, not
    patching the property, leaves no window for a late read after the test.
    """
    from langsmith import Client
    from langsmith.schemas import LangSmithInfo

    init = Client.__init__

    def _init(self, *args, info=None, **kwargs):
        init(self, *args, info=LangSmithInfo() if info is None else info, **kwargs)

    monkeypatch.setattr(Client, "__init__", _init)


# Every langsmith Client a test built that is still alive. Kept across tests: a Client
# held in module state (the server's tracer after a lifespan test) can queue a run in
# a later test, so each teardown flushes all of them, not only its own test's.
_langsmith_clients: weakref.WeakSet = weakref.WeakSet()


@pytest.fixture(autouse=True)
def no_langsmith_requests(monkeypatch):
    """
    Fail a test whose langsmith Client tries an HTTP request, including runs it
    queued for upload; none is sent.

    GET /info and the run uploads go through ``Client.request_with_retries``, so
    that records the call and raises instead. Each Client's background thread
    reads its server info on start and uploads queued runs later, so teardown
    waits for that read, then flushes, while the patch holds, every live Client
    any test built (``_langsmith_clients``) plus LangChain's cached client
    (``langsmith.run_trees._CLIENT``, the one LangChain's env tracer uses), each once.
    """
    from langsmith import Client, run_trees

    calls: list[str] = []
    init = Client.__init__

    def _init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        _langsmith_clients.add(self)

    def _request(self, method, pathname, **_kwargs):
        calls.append(f"{method} {pathname}")
        raise ConnectionError("tests never call LangSmith")

    monkeypatch.setattr(Client, "__init__", _init)
    monkeypatch.setattr(Client, "request_with_retries", _request)
    yield
    clients = list(_langsmith_clients)
    deadline = time.monotonic() + 5
    while any(client._info is None for client in clients) and time.monotonic() < deadline:
        time.sleep(0.01)
    cached = run_trees._CLIENT
    if cached is not None and all(client is not cached for client in clients):
        clients.append(cached)
    for client in clients:
        client.flush(timeout=5)
    assert not calls, f"the test called LangSmith: {calls}"
