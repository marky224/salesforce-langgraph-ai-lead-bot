"""
FastAPI-layer tests for abuse protection (PR 2, C1 — rate limiting).

These drive the real app through ``TestClient``.  The lifespan builds the graph
with an in-memory ``MemorySaver`` and a ``ChatOpenAI`` client (constructed,
never called over the network); the test then overrides the LLM with an async
mock so endpoint bodies return without hitting an API.

``LLM_PROVIDER=openai`` is used because ``langchain-anthropic`` is not installed
in the test environment (only ``langchain-openai`` is).  ``RATE_LIMIT`` is set
tiny so the limit is reached in a few calls.  Rate-limit buckets are keyed per
client IP, so each test uses a distinct ``X-Forwarded-For`` value to stay
isolated from the others.
"""

from __future__ import annotations

import itertools
import json
from contextlib import asynccontextmanager
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
