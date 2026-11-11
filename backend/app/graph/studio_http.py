"""
Request guard for ``langgraph dev`` (LangGraph Studio), dev only.

``langgraph.json`` mounts ``app`` below as the dev server's custom HTTP app
(``http.app``).  langgraph-api puts a custom app's middleware outside its own,
so ``StudioGuardMiddleware`` sees every HTTP request and every WebSocket
handshake to the built-in routes (threads, runs, assistants, and the v2
``/threads/{thread_id}/stream/events`` socket) before they run.  Nothing in the
app imports this module and the production image never runs it.

The dev server has no auth, and the graph it serves holds the real LLM key and
Salesforce writer from ``.env``.  The CORS block in ``langgraph.json`` stops
other origins reading HTTP replies and sending preflighted (JSON) requests, but
it doesn't apply to WebSockets, a browser sends a "simple" request (a
``text/plain`` or form POST, or a POST with no body) with no preflight, and
langgraph-api parses any body as JSON.  So the guard refuses:

- any HTTP request or WebSocket whose ``Host`` is not ``127.0.0.1``,
  ``localhost`` or ``[::1]`` (any port), which stops DNS rebinding (a hostile
  name re-pointed at 127.0.0.1 is still that name in ``Host``);
- any WebSocket, and any request whose method isn't GET, HEAD or OPTIONS, that
  carries an ``Origin`` other than ``https://smith.langchain.com`` or the dev
  server's own origin (``http://`` plus the request's ``Host``, so the server's
  own ``/docs`` page passes).  Current browsers send ``Origin`` on those, so a
  page from another origin, another local port included (``http://localhost:5173``
  against a server on ``127.0.0.1:2024``), can't open the event socket or change
  state; a client that sends no ``Origin`` (curl, the SDK) passes;
- any POST with a body whose ``Content-Type`` is not ``application/json``
  (parameters such as ``charset`` allowed), so a body must be one CORS would
  have preflighted.

An HTTP request is refused with 403 (415 for the content type); a WebSocket is
closed before it is accepted, which the server sends as a 403 handshake reply.
Everything else (OPTIONS preflights, GETs from another origin, whose replies
CORS keeps from the page) passes through unchanged.  This is not auth: anything that can
reach the port without a browser still can, so stop the server when done.
"""

from __future__ import annotations

from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.middleware import Middleware
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocketClose

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]"})
_STUDIO_ORIGIN = "https://smith.langchain.com"
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def _host_name(host: str) -> str:
    """``Host`` header without its port, lower-cased (``[::1]:2024`` → ``[::1]``)."""
    host = host.strip().lower()
    if host.startswith("["):
        return host.split("]", 1)[0] + "]"
    return host.split(":", 1)[0]


def _has_body(headers: Headers) -> bool:
    """True when the request declares a body (a non-zero length or chunked)."""
    length = headers.get("content-length")
    if length is not None:
        return length.strip() != "0"
    return "transfer-encoding" in headers


def _is_json(content_type: str) -> bool:
    return content_type.split(";", 1)[0].strip().lower() == "application/json"


def _origin_allowed(origin: str, host: str) -> bool:
    """True for Studio's origin or the server's own (``http://`` + ``Host``).

    Only called once ``Host`` is known to be loopback, so the server's own
    origin is a loopback one; another port on the same host is a different
    origin and is refused.
    """
    origin = origin.strip().lower()
    return origin in (_STUDIO_ORIGIN, "http://" + host.strip().lower())


class StudioGuardMiddleware:
    """Refuse a non-loopback ``Host`` or a foreign ``Origin`` (403), and non-JSON POST bodies (415)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            headers = Headers(scope=scope)
            websocket = scope["type"] == "websocket"
            host = headers.get("host", "")
            if _host_name(host) not in _LOOPBACK_HOSTS:
                await _refuse(scope, receive, send, "Host must be 127.0.0.1, localhost or [::1].")
                return
            origin = headers.get("origin")
            if (
                origin is not None
                and (websocket or scope["method"] not in _SAFE_METHODS)
                and not _origin_allowed(origin, host)
            ):
                await _refuse(
                    scope, receive, send, f"Origin must be {_STUDIO_ORIGIN} or this server's own."
                )
                return
            if (
                not websocket
                and scope["method"] == "POST"
                and _has_body(headers)
                and not _is_json(headers.get("content-type", ""))
            ):
                response = PlainTextResponse(
                    "POST bodies must be application/json.", status_code=415
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


async def _refuse(scope: Scope, receive: Receive, send: Send, reason: str) -> None:
    """403 an HTTP request; close a WebSocket before accepting it (a 403 handshake)."""
    if scope["type"] == "websocket":
        await WebSocketClose(code=1008, reason=reason)(scope, receive, send)
    else:
        await PlainTextResponse(reason, status_code=403)(scope, receive, send)


app = Starlette(middleware=[Middleware(StudioGuardMiddleware)])
