"""
Tests for the ``langgraph dev`` request guard (``app/graph/studio_http.py``).

A tiny Starlette app stands in for langgraph-api's routes (an HTTP route and a
WebSocket like ``/threads/{thread_id}/stream/events``), wrapped in
``StudioGuardMiddleware`` as langgraph-api wraps its own; no dev server needed.
"""

from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect

from app.graph.studio_http import StudioGuardMiddleware
from app.graph.studio_http import app as studio_app

LOCAL = "http://127.0.0.1:2024"
STUDIO = "https://smith.langchain.com"
CROSS_SITE = "https://unrelated-site.example"
# A page from another local server: same machine, another origin.
OTHER_PORT = "http://localhost:5173"


@pytest.fixture
def calls():
    return []


@pytest.fixture
def client(calls):
    async def threads(request: Request) -> JSONResponse:
        calls.append((request.method, await request.body()))
        return JSONResponse({"ok": True})

    async def events(websocket: WebSocket) -> None:
        await websocket.accept()
        calls.append(("WS", await websocket.receive_text()))
        await websocket.send_text("ok")
        await websocket.close()

    app = Starlette(
        routes=[
            Route("/threads", threads, methods=["GET", "POST", "OPTIONS"]),
            WebSocketRoute("/threads/t1/stream/events", events),
        ],
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


class TestOrigin:
    @pytest.mark.parametrize(
        "origin",
        [
            CROSS_SITE,
            "null",
            "http://localhost.attacker.example",
            "https://127.0.0.1:2024",
            OTHER_PORT,
            "http://127.0.0.1:5173",
            # Same port, another loopback name: a different origin from Host 127.0.0.1:2024.
            "http://localhost:2024",
            "http://[::1]:2024",
        ],
    )
    def test_foreign_origin_json_post_gets_403(self, client, calls, origin):
        response = client.post(
            "/threads", content=b"{}", headers={"content-type": "application/json", "origin": origin}
        )

        assert response.status_code == 403
        assert calls == []

    @pytest.mark.parametrize("origin", [CROSS_SITE, OTHER_PORT])
    def test_foreign_origin_bodyless_post_gets_403(self, client, calls, origin):
        # What fetch(url, {method: "POST", mode: "no-cors"}) sends, e.g. to /internal/truncate.
        response = client.post("/threads", headers={"origin": origin, "content-length": "0"})

        assert response.status_code == 403
        assert calls == []

    @pytest.mark.parametrize(
        ("origin", "host"),
        [
            (STUDIO, "127.0.0.1:2024"),
            # The server's own origin (its /docs page): http:// plus Host, any case.
            ("http://127.0.0.1:2024", "127.0.0.1:2024"),
            ("http://localhost:2024", "localhost:2024"),
            ("HTTP://LOCALHOST:2024", "localhost:2024"),
            ("http://[::1]:2024", "[::1]:2024"),
        ],
    )
    def test_allowed_origin_json_post_passes(self, client, calls, origin, host):
        response = client.post(
            "/threads",
            content=b"{}",
            headers={"content-type": "application/json", "origin": origin, "host": host},
        )

        assert response.status_code == 200
        assert calls == [("POST", b"{}")]

    @pytest.mark.parametrize("method", ["GET", "OPTIONS"])
    def test_foreign_origin_safe_method_passes(self, client, calls, method):
        # CORS keeps the reply from the page; the request changes nothing.
        response = client.request(method, "/threads", headers={"origin": CROSS_SITE})

        assert response.status_code == 200
        assert calls == [(method, b"")]


class TestWebSocket:
    @staticmethod
    def _connect(client, **headers):
        # A full URL: websocket_connect ignores base_url (it would send Host: testserver).
        with client.websocket_connect("ws://127.0.0.1:2024/threads/t1/stream/events", headers=headers) as ws:
            ws.send_text('{"id": 1, "method": "run.start"}')
            return ws.receive_text()

    @pytest.mark.parametrize(
        "headers",
        [
            {"origin": CROSS_SITE},
            {"origin": "null"},
            {"origin": OTHER_PORT},
            {"origin": "http://localhost:2024"},
            {"host": "rebind.attacker.example:2024"},
            {"host": "rebind.attacker.example:2024", "origin": STUDIO},
        ],
        ids=[
            "cross-site-origin",
            "null-origin",
            "other-port-origin",
            "other-loopback-name-origin",
            "rebind-host",
            "rebind-host-studio-origin",
        ],
    )
    def test_refused_before_accept(self, client, calls, headers):
        with pytest.raises(WebSocketDisconnect) as err:
            self._connect(client, **headers)

        assert err.value.code == 1008
        assert calls == []

    @pytest.mark.parametrize(
        "headers",
        [
            {"origin": STUDIO},
            {"origin": "http://127.0.0.1:2024"},
            {"origin": "http://localhost:2024", "host": "localhost:2024"},
            {},
        ],
        ids=["studio", "own-origin-ip", "own-origin-name", "no-origin"],
    )
    def test_allowed_connects(self, client, calls, headers):
        assert self._connect(client, **headers) == "ok"
        assert calls == [("WS", '{"id": 1, "method": "run.start"}')]
