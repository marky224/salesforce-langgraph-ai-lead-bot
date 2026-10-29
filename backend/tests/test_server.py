"""
FastAPI-layer tests for abuse protection (PR 2, C1 — rate limiting).

These drive the real app through ``TestClient``.  The lifespan builds the graph
with an in-memory ``MemorySaver`` and a ``ChatOpenAI`` client (constructed,
never called over the network); the test then overrides the LLM with an async
mock so endpoint bodies return without hitting an API.

``LLM_PROVIDER=openai`` keeps the lifespan's model a known ``ChatOpenAI`` (one
test asserts that provider in the trace metadata).  ``RATE_LIMIT`` is set
tiny so the limit is reached in a few calls.  Rate-limit buckets are keyed per
client IP, so each test uses a distinct ``X-Forwarded-For`` value to stay
isolated from the others.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import threading
import time
import urllib.error
from contextlib import asynccontextmanager
from email.message import Message
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT", "3/minute")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")

    from app.config import get_settings

    get_settings.cache_clear()

    from app import server
    from app.graph.nodes import set_llm

    # Keep the FastAPI-layer tests hermetic even when a real DATABASE_URL is
    # present in a local .env: force the in-memory saver so the lifespan doesn't
    # connect to Postgres.
    @asynccontextmanager
    async def _memory_checkpointer(_settings):
        yield MemorySaver()

    monkeypatch.setattr(server, "open_checkpointer", _memory_checkpointer)

    with TestClient(server.app) as test_client:
        # Replace the real LLM the lifespan installed with a fast async mock so
        # endpoint bodies don't reach out to a provider.
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="hello"))
        set_llm(mock_llm)
        yield test_client


def test_init_under_limit_ok(client):
    headers = {"X-Forwarded-For": "203.0.113.10"}
    for _ in range(3):
        assert client.post("/chat/init", headers=headers).status_code == 200


def test_init_over_limit_returns_429(client):
    headers = {"X-Forwarded-For": "203.0.113.11"}
    for _ in range(3):
        assert client.post("/chat/init", headers=headers).status_code == 200

    resp = client.post("/chat/init", headers=headers)
    assert resp.status_code == 429
    assert "too quickly" in resp.json()["detail"].lower()


def test_limit_is_per_ip(client):
    over = {"X-Forwarded-For": "203.0.113.12"}
    for _ in range(3):
        client.post("/chat/init", headers=over)
    # This IP is now over the limit...
    assert client.post("/chat/init", headers=over).status_code == 429
    # ...but a different IP has its own bucket.
    fresh = {"X-Forwarded-For": "203.0.113.13"}
    assert client.post("/chat/init", headers=fresh).status_code == 200


# ---------------------------------------------------------------------------
# Request-ID correlation (PR 4, C1)
# ---------------------------------------------------------------------------

def test_response_carries_generated_request_id(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    # No inbound id → the middleware mints one and echoes it back.
    assert resp.headers.get("X-Request-ID")


def test_supplied_request_id_is_echoed(client):
    rid = "req-abc-123"
    resp = client.get("/health", headers={"X-Request-ID": rid})
    assert resp.headers.get("X-Request-ID") == rid


# ---------------------------------------------------------------------------
# Liveness vs readiness (PR 4, C2)
# ---------------------------------------------------------------------------

def test_readiness_ready_when_graph_and_llm_present(client):
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["checks"]["graph"] is True
    assert body["checks"]["llm"] is True


def test_readiness_degraded_when_graph_missing(client, monkeypatch):
    from app import server

    monkeypatch.setattr(server, "_graph", None)
    resp = client.get("/health/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["checks"]["graph"] is False


# ---------------------------------------------------------------------------
# Chat endpoints — happy path, thread continuity, SSE shape, 500s (PR 5, C3)
# ---------------------------------------------------------------------------
# The `client` fixture installs a plain AsyncMock LLM (every reply is "hello").
# A first /chat turn deterministically routes extraction → discovery (the
# GREETING stage short-circuits past the router), so `stage` is predictable.


def _parse_sse(text: str) -> list[dict]:
    """Parse an SSE response body into the list of JSON `data:` payloads."""
    events = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            events.append(json.loads(line[len("data:"):].strip()))
    return events


def test_chat_returns_reply_and_thread(client):
    headers = {"X-Forwarded-For": "203.0.113.20"}
    resp = client.post("/chat", json={"message": "hi"}, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["reply"] == "hello"          # the mocked LLM's content
    assert body["thread_id"]                 # a fresh uuid was minted
    assert body["stage"] == "discovery"      # first turn always lands in discovery
    assert body["is_complete"] is False
    assert body["lead_id"] is None


def test_chat_continues_same_thread(client):
    headers = {"X-Forwarded-For": "203.0.113.21"}
    first = client.post("/chat", json={"message": "hi"}, headers=headers)
    thread_id = first.json()["thread_id"]

    second = client.post(
        "/chat", json={"message": "tell me more", "thread_id": thread_id}, headers=headers
    )
    assert second.status_code == 200
    assert second.json()["thread_id"] == thread_id


def test_chat_stream_emits_tokens_and_done_event(client):
    from app.graph.nodes import set_llm

    # A streaming-capable fake so astream_events emits on_chat_model_stream; an
    # infinite cycle feeds the multiple per-turn LLM calls (extraction + node).
    set_llm(
        GenericFakeChatModel(
            messages=itertools.cycle([AIMessage(content="Hello there friend")])
        )
    )

    resp = client.post(
        "/chat/stream", json={"message": "hi"}, headers={"X-Forwarded-For": "203.0.113.22"}
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(resp.text)
    token_events = [e for e in events if "token" in e]
    done_events = [e for e in events if e.get("done")]

    # The server forwards tokens only from conversational nodes (discovery here),
    # so the streamed text is the node's full reply.
    assert token_events, "expected at least one token event"
    assert "".join(e["token"] for e in token_events) == "Hello there friend"
    assert not any("error" in e for e in events)

    # Exactly one terminal frame, carrying the full SSE 'done' contract (CLAUDE.md #2).
    assert len(done_events) == 1
    done = done_events[0]
    for key in ("thread_id", "stage", "is_complete", "lead_id", "lead_data"):
        assert key in done
    assert done["stage"] == "discovery"
    assert done["is_complete"] is False


def test_chat_returns_500_when_graph_errors(client, monkeypatch):
    from app import server

    broken = MagicMock()
    broken.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(server, "_graph", broken)

    resp = client.post(
        "/chat", json={"message": "hi"}, headers={"X-Forwarded-For": "203.0.113.23"}
    )
    assert resp.status_code == 500
    assert "error occurred" in resp.json()["detail"].lower()


def test_chat_init_returns_500_when_graph_errors(client, monkeypatch):
    from app import server

    broken = MagicMock()
    broken.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(server, "_graph", broken)

    resp = client.post("/chat/init", headers={"X-Forwarded-For": "203.0.113.24"})
    assert resp.status_code == 500
    assert "failed to start" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Trace metadata on graph runs (PR A2a)
# ---------------------------------------------------------------------------


def test_run_config_carries_trace_metadata(client):
    from app import server

    cfg = server._run_config("thread-xyz")
    assert cfg["configurable"]["thread_id"] == "thread-xyz"
    assert cfg["metadata"]["session_id"] == "thread-xyz"
    assert cfg["metadata"]["provider"] == "openai"  # client fixture sets LLM_PROVIDER=openai
    # model is whatever's configured (env/.env LLM_MODEL, else provider default) — assert it's
    # carried, not its exact value, so the test stays hermetic across environments.
    assert isinstance(cfg["metadata"]["model"], str) and cfg["metadata"]["model"]
    assert cfg["tags"] == ["tars"]
    assert "callbacks" not in cfg  # tracing off in tests → no tracer attached


def test_lifespan_flushes_tracer_on_shutdown(monkeypatch):
    """In-flight traces are flushed on shutdown when tracing is enabled.

    Background uploads can be dropped if the process exits mid-flight, so the
    lifespan calls ``wait_for_all_tracers()`` — but only when a tracer is active.
    """
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-not-real")

    from app.config import get_settings

    get_settings.cache_clear()

    from app import server

    @asynccontextmanager
    async def _memory_checkpointer(_settings):
        yield MemorySaver()

    monkeypatch.setattr(server, "open_checkpointer", _memory_checkpointer)

    flushed: list[int] = []
    monkeypatch.setattr(server, "wait_for_all_tracers", lambda: flushed.append(1))

    # Startup builds a real (offline) tracer; the context exit runs the shutdown path.
    with TestClient(server.app):
        pass

    assert flushed == [1]  # flushed exactly once, because a tracer was active


# ---------------------------------------------------------------------------
# Site index: loaded in the background by the lifespan, never blocking startup
# ---------------------------------------------------------------------------

SITE_INDEX_URL = "https://www.markandrewmarquez.com/site-index.json"
SITE_INDEX_FIXTURE = Path(__file__).parent / "fixtures" / "site_index_v1.json"


class _FakeIndexResponse:
    """The slice of an HTTP response that site_index.fetch_site_index reads."""

    def __init__(self, body: bytes):
        self._body = body
        self.headers = Message()
        self.headers["Content-Type"] = "application/json"

    def read(self, size: int = -1) -> bytes:
        return self._body if size < 0 else self._body[:size]

    def geturl(self) -> str:
        return SITE_INDEX_URL

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


@pytest.fixture
def lifespan_env(monkeypatch):
    """Env for driving the real lifespan offline; resets the in-memory index."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")

    from app.config import get_settings

    get_settings.cache_clear()

    from app import server
    from app.tools import site_index

    @asynccontextmanager
    async def _memory_checkpointer(_settings):
        yield MemorySaver()

    monkeypatch.setattr(server, "open_checkpointer", _memory_checkpointer)
    site_index.set_site_index(None)
    yield server, site_index
    site_index.set_site_index(None)


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def test_lifespan_never_fetches_the_site_index_when_url_is_blank(lifespan_env, monkeypatch):
    server, site_index = lifespan_env
    fetches: list[object] = []
    started: list[tuple] = []

    async def _record_refresh(*args):
        started.append(args)

    monkeypatch.setattr(site_index, "_urlopen", lambda request, timeout: fetches.append(request))
    monkeypatch.setattr(server, "run_site_index_refresh", _record_refresh)

    with TestClient(server.app) as test_client:  # conftest leaves SITE_INDEX_URL empty
        assert test_client.get("/health").status_code == 200

    assert started == []
    assert fetches == []
    assert site_index.get_site_index() is None


def test_lifespan_loads_the_site_index_in_the_background(lifespan_env, monkeypatch):
    server, site_index = lifespan_env
    monkeypatch.setenv("SITE_INDEX_URL", SITE_INDEX_URL)
    from app.config import get_settings

    get_settings.cache_clear()

    calls: list[tuple] = []

    def _fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        return _FakeIndexResponse(SITE_INDEX_FIXTURE.read_bytes())

    monkeypatch.setattr(site_index, "_urlopen", _fake_urlopen)

    with TestClient(server.app):
        assert _wait_for(lambda: site_index.get_site_index() is not None)

    assert calls[0] == (SITE_INDEX_URL, 10.0)
    index = site_index.get_site_index()
    assert index is not None
    assert len(index.chunks) == 13


def test_startup_does_not_wait_for_or_fail_on_the_site_index(lifespan_env, monkeypatch):
    """A hung, then failing, fetch neither delays startup nor degrades readiness."""
    server, site_index = lifespan_env
    monkeypatch.setenv("SITE_INDEX_URL", SITE_INDEX_URL)
    from app.config import get_settings

    get_settings.cache_clear()

    entered = threading.Event()
    release = threading.Event()
    done = threading.Event()

    def _hanging_urlopen(request, timeout):
        entered.set()
        release.wait(5)
        done.set()
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(site_index, "_urlopen", _hanging_urlopen)

    started = time.monotonic()
    with TestClient(server.app) as test_client:
        # Startup returned at once (it normally takes a few hundredths of a second),
        # so it didn't wait on the fetch even briefly...
        assert time.monotonic() - started < 1.0
        # ...and the first fetch is still in flight: had the lifespan awaited it,
        # startup would only return once release.wait timed out.
        assert entered.wait(5)
        assert not done.is_set()
        assert test_client.get("/health").status_code == 200
        assert test_client.get("/health/ready").status_code == 200
        release.set()
        assert test_client.get("/health").status_code == 200

    assert site_index.get_site_index() is None


def test_shutdown_cancels_the_site_index_refresh(lifespan_env, monkeypatch):
    """The lifespan cancels the refresh task and waits for it before shutting down."""
    server, _ = lifespan_env
    monkeypatch.setenv("SITE_INDEX_URL", SITE_INDEX_URL)
    from app.config import get_settings

    get_settings.cache_clear()

    events: list[str] = []

    async def _refresh_until_cancelled(url, interval_seconds):
        events.append("started")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            events.append("cancelled")
            raise

    def _record_shutdown(msg, *args, **kwargs):
        if msg.startswith("Shutting down"):
            events.append("shutdown")

    monkeypatch.setattr(server, "run_site_index_refresh", _refresh_until_cancelled)
    monkeypatch.setattr(server.logger, "info", _record_shutdown)

    with TestClient(server.app):
        assert _wait_for(lambda: "started" in events)

    # Cancelled by the lifespan itself, before its shutdown log line, not by the
    # event loop's teardown afterwards.
    assert events == ["started", "cancelled", "shutdown"]
