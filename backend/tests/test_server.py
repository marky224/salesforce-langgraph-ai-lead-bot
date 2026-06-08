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

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
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
