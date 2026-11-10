"""
Request guard for ``langgraph dev`` (LangGraph Studio), dev only.

``langgraph.json`` mounts ``app`` below as the dev server's custom HTTP app
(``http.app``).  langgraph-api puts a custom app's middleware outside its own,
so ``StudioGuardMiddleware`` sees every request to the built-in routes
(threads, runs, assistants) before they run.  Nothing in the app imports this
module and the production image never runs it.

The dev server has no auth, and the graph it serves holds the real LLM key and
Salesforce writer from ``.env``.  The CORS block in ``langgraph.json`` stops
other origins reading replies and sending preflighted (JSON) requests, but a
browser sends a "simple" request (a ``text/plain`` or form POST) with no
preflight, and langgraph-api parses any body as JSON.  So the guard answers:

- 403 to any request whose ``Host`` is not ``127.0.0.1``, ``localhost`` or
  ``[::1]`` (any port), which stops DNS rebinding (a hostile name re-pointed at
  127.0.0.1 is still that name in ``Host``);
- 415 to any POST with a body whose ``Content-Type`` is not
  ``application/json`` (parameters such as ``charset`` allowed), so a cross-site
  page can't create threads or start runs without a preflight CORS refuses.

Every other request (OPTIONS preflights included) passes through unchanged.
"""

from __future__ import annotations

from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.middleware import Middleware
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]"})


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


class StudioGuardMiddleware:
    """Refuse non-loopback ``Host`` headers (403) and non-JSON POST bodies (415)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = Headers(scope=scope)
            if _host_name(headers.get("host", "")) not in _LOOPBACK_HOSTS:
                response = PlainTextResponse(
                    "Host must be 127.0.0.1, localhost or [::1].", status_code=403
                )
                await response(scope, receive, send)
                return
            if (
                scope["method"] == "POST"
                and _has_body(headers)
                and not _is_json(headers.get("content-type", ""))
            ):
                response = PlainTextResponse(
                    "POST bodies must be application/json.", status_code=415
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


app = Starlette(middleware=[Middleware(StudioGuardMiddleware)])
