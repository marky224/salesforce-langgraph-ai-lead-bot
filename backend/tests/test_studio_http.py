"""
Tests for the ``langgraph dev`` request guard (``app/graph/studio_http.py``).

A tiny Starlette app stands in for langgraph-api's routes, wrapped in
``StudioGuardMiddleware`` as langgraph-api wraps its own; no dev server needed.
"""

from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.graph.studio_http import StudioGuardMiddleware
from app.graph.studio_http import app as studio_app

LOCAL = "http://127.0.0.1:2024"


@pytest.fixture
def calls():
    return []


@pytest.fixture
def client(calls):
    async def threads(request: Request) -> JSONResponse:
        calls.append((request.method, await request.body()))
        return JSONResponse({"ok": True})

    app = Starlette(
        routes=[Route("/threads", threads, methods=["GET", "POST", "OPTIONS"])],
        middleware=[Middleware(StudioGuardMiddleware)],
    )
    return TestClient(app, base_url=LOCAL)


def test_studio_app_carries_the_guard():
    """langgraph-api picks up the custom app's middleware from ``user_middleware``."""
    assert [m.cls for m in studio_app.user_middleware] == [StudioGuardMiddleware]


class TestHost:
    @pytest.mark.parametrize(
        "host", ["127.0.0.1:2024", "127.0.0.1", "localhost:2024", "LOCALHOST", "[::1]:2024", "[::1]"]
    )
    def test_loopback_host_passes(self, client, calls, host):
        assert client.get("/threads", headers={"host": host}).status_code == 200
        assert calls == [("GET", b"")]

    @pytest.mark.parametrize(
        "host", ["rebind.attacker.example:2024", "localhost.attacker.example", "127.0.0.2:2024", "0.0.0.0:2024"]
    )
    def test_other_host_gets_403(self, client, calls, host):
        response = client.get("/threads", headers={"host": host})

        assert response.status_code == 403
        assert calls == []

    def test_preflight_from_another_host_gets_403(self, client, calls):
        response = client.options("/threads", headers={"host": "rebind.attacker.example:2024"})

        assert response.status_code == 403
        assert calls == []


class TestPostContentType:
    @pytest.mark.parametrize(
        "content_type",
        [
            "text/plain;charset=UTF-8",
            "application/x-www-form-urlencoded",
            "multipart/form-data; boundary=x",
        ],
    )
    def test_simple_request_body_gets_415(self, client, calls, content_type):
        response = client.post(
            "/threads", content=b'{"metadata": {"src": "cross-site"}}', headers={"content-type": content_type}
        )

        assert response.status_code == 415
        assert calls == []

    def test_body_without_content_type_gets_415(self, client, calls):
        response = client.post("/threads", content=b"{}", headers={"content-type": ""})

        assert response.status_code == 415
        assert calls == []

    @pytest.mark.parametrize("content_type", ["application/json", "application/json; charset=utf-8", "Application/JSON"])
    def test_json_body_passes(self, client, calls, content_type):
        response = client.post("/threads", content=b"{}", headers={"content-type": content_type})

        assert response.status_code == 200
        assert calls == [("POST", b"{}")]

    def test_post_without_body_passes(self, client, calls):
        assert client.post("/threads").status_code == 200
        assert calls == [("POST", b"")]

    def test_options_passes_unchanged(self, client, calls):
        response = client.options("/threads", headers={"content-type": "text/plain"})

        assert response.status_code == 200
        assert calls == [("OPTIONS", b"")]
