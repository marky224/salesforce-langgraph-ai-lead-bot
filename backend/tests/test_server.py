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

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT", "3/minute")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")

    from app.config import get_settings

    get_settings.cache_clear()

    from app import server
    from app.graph.nodes import set_llm

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
