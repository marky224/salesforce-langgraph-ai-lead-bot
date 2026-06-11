"""
FastAPI server entrypoint for the AI Sales Lead Bot.

Exposes three primary endpoints:

- ``POST /chat`` — synchronous chat: accepts a message + thread_id,
  runs the full graph turn, returns the assistant's reply.
- ``POST /chat/stream`` — streaming chat: same input, but streams
  the assistant's reply token-by-token via Server-Sent Events (SSE).
- ``GET /health`` — liveness / readiness check with Salesforce
  connection status.

The server initialises the LLM provider and compiles the LangGraph
at startup.  Each conversation is identified by a ``thread_id``
(UUID) which maps to a checkpointed graph state.

Run locally::

    uvicorn app.server:app --reload --port 8000

Or via Docker::

    docker run -p 8000:8000 --env-file .env salesforce-langgraph-ai-lead-bot
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tracers.langchain import wait_for_all_tracers
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config import configure_logging, get_llm, get_settings
from app.graph.checkpointer import open_checkpointer
from app.graph.graph import build_graph
from app.graph.nodes import is_llm_ready, set_llm
from app.logging_ctx import request_id_var, thread_id_var
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationStage,
    HealthResponse,
)
from app.tracing import build_tracer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate limiting (abuse protection)
# ---------------------------------------------------------------------------
# Behind Azure Container Apps ingress, request.client.host is the proxy, so we
# key off the first hop of X-Forwarded-For, falling back to the socket peer for
# direct/local calls.

def _client_ip(request: Request) -> str:
    """Return the real client IP, honouring the X-Forwarded-For proxy header."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


def _rate_limit() -> str:
    """Current per-IP limit, read live so env/tests can override it."""
    return get_settings().rate_limit


limiter = Limiter(key_func=_client_ip)


async def _rate_limit_exceeded_handler(
    request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    """Polite 429 in place of slowapi's default plain-text response."""
    return JSONResponse(
        status_code=429,
        content={
            "detail": (
                "You're sending messages a little too quickly. "
                "Please wait a moment and try again."
            )
        },
    )


# ---------------------------------------------------------------------------
# Request context (correlation IDs for tracing)
# ---------------------------------------------------------------------------

class RequestContextMiddleware:
    """
    Stamp each request with a correlation id.

    Reads an inbound ``X-Request-ID`` (or mints one), publishes it plus a reset
    ``thread_id`` into the logging contextvars, and echoes ``X-Request-ID`` on
    the response.  Implemented as raw ASGI (not ``BaseHTTPMiddleware``) so it
    runs in the request's own context — the contextvars stay visible to the
    route and the SSE generator — and never buffers the streaming response body.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = Headers(scope=scope).get("x-request-id") or str(uuid.uuid4())
        request_id_var.set(request_id)
        thread_id_var.set("-")  # cleared per request; chat handlers set the real thread

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        await self.app(scope, receive, send_with_request_id)


# ---------------------------------------------------------------------------
# Application state (populated at startup)
# ---------------------------------------------------------------------------

_graph = None
_tracer: Any | None = None
_trace_base: dict[str, Any] = {"metadata": {}, "tags": ["tars"]}


def _get_graph():
    """Return the compiled graph, raising if not initialised."""
    if _graph is None:
        raise RuntimeError("Graph not initialised — server startup failed.")
    return _graph


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.

    On startup:
    1. Configure logging.
    2. Instantiate the LLM provider.
    3. Inject the LLM into the graph nodes.
    4. Compile the LangGraph with a checkpointer.

    On shutdown:
    - Log a clean shutdown message.
    """
    global _graph, _trace_base, _tracer  # noqa: PLW0603

    # --- Startup ---
    configure_logging()
    settings = get_settings()

    logger.info(
        "Starting AI Sales Lead Bot — provider=%s, version=%s",
        settings.llm_provider.value,
        settings.app_version,
    )

    # Initialise LLM
    try:
        llm = get_llm()
        set_llm(llm)
        logger.info("LLM initialised: %s", type(llm).__name__)
    except Exception:
        logger.exception("Failed to initialise LLM — chat will not work")
        raise

    # Static trace metadata stamped on every graph run (session_id added per
    # request in _run_config). Resolved from the live model so it matches reality.
    model = getattr(llm, "model_name", None) or settings.llm_model or "unknown"
    _trace_base = {
        "metadata": {"provider": settings.llm_provider.value, "model": model},
        "tags": ["tars"],
    }
    _tracer = build_tracer(settings)

    # Open the checkpointer (Postgres if DATABASE_URL set, else MemorySaver) and
    # keep it open for the whole process so the DB connection lives for the
    # app's lifetime.
    async with open_checkpointer(settings) as checkpointer:
        _graph = build_graph(checkpointer=checkpointer)
        logger.info("LangGraph compiled and ready")

        yield

    # --- Shutdown ---
    if _tracer is not None:
        # Trace uploads run on a background thread; flush so a deploy/restart
        # (SIGTERM) doesn't drop in-flight runs.
        wait_for_all_tracers()
    logger.info("Shutting down AI Sales Lead Bot")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Sales Lead Bot",
    description=(
        "Stateful conversational sales chatbot powered by LangGraph. "
        "Qualifies leads through natural conversation and creates "
        "records in Salesforce."
    ),
    version=get_settings().app_version,
    lifespan=lifespan,
)

# --- Rate limiting ---
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- CORS ---
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,  # widget sends no cookies/credentials; the API has no auth
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("CORS origins: %s", settings.cors_origin_list)

# --- Request context (added last → outermost, so every request is stamped) ---
app.add_middleware(RequestContextMiddleware)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    """
    Liveness / readiness probe.

    Returns the server version and timestamp.  Optionally checks the
    Salesforce connection if credentials are configured.
    """
    return HealthResponse()  # version sourced from Settings.app_version (single source)


@app.get("/health/ready", tags=["system"])
async def readiness_check() -> JSONResponse:
    """
    Readiness probe (distinct from ``/health`` liveness).

    Reports whether the app can actually serve a turn: graph compiled, LLM
    injected, and — only when a durable checkpointer is configured — the DB
    reachable.  The DB probe just reads checkpoint state for a sentinel thread;
    it makes no LLM call, so it costs no tokens.  200 ready / 503 degraded.
    """
    settings = get_settings()
    checks: dict[str, bool] = {
        "graph": _graph is not None,
        "llm": is_llm_ready(),
    }

    if settings.database_url:
        db_ok = False
        if _graph is not None:
            try:
                await _graph.aget_state(
                    {"configurable": {"thread_id": "health-probe"}}
                )
                db_ok = True
            except Exception:
                logger.warning("Readiness DB probe failed", exc_info=True)
        checks["database"] = db_ok

    ready = all(checks.values())
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ready" if ready else "degraded", "checks": checks},
    )


@app.get("/health/salesforce", tags=["system"])
async def salesforce_health() -> dict[str, Any]:
    """
    Check Salesforce connectivity and API usage.

    Returns connection status, instance URL, and remaining API calls.
    """
    try:
        from app.tools.salesforce import verify_connection

        return await verify_connection()
    except ImportError:
        return {"connected": False, "error": "simple_salesforce not installed"}
    except Exception as exc:
        return {"connected": False, "error": str(exc)}


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
@limiter.limit(_rate_limit)
async def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    """
    Synchronous chat endpoint.

    Accepts a user message and optional thread_id.  Runs one full
    graph turn (extraction → routing → conversation node) and returns
    the assistant's reply.

    If no ``thread_id`` is provided, a new conversation is started.
    """
    graph = _get_graph()
    thread_id = payload.thread_id or str(uuid.uuid4())
    thread_id_var.set(thread_id)

    logger.info(
        "Chat request: thread=%s, message=%.80s",
        thread_id,
        payload.message,
    )

    config = _run_config(thread_id)

    # Build input — add the new human message
    graph_input = {
        "messages": [HumanMessage(content=payload.message)],
    }

    try:
        # Run the graph for one full turn
        result = await graph.ainvoke(graph_input, config=config)
    except Exception as err:
        logger.exception("Graph invocation failed for thread %s", thread_id)
        raise HTTPException(
            status_code=500,
            detail="An error occurred processing your message. Please try again.",
        ) from err

    # Extract the latest AI message
    reply = _extract_latest_ai_reply(result)
    stage = result.get("stage", ConversationStage.GREETING)
    is_complete = stage == ConversationStage.COMPLETE
    lead_id = result.get("salesforce_lead_id")

    logger.info(
        "Chat response: thread=%s, stage=%s, complete=%s, lead_id=%s",
        thread_id,
        stage.value if isinstance(stage, ConversationStage) else stage,
        is_complete,
        lead_id,
    )

    return ChatResponse(
        reply=reply,
        thread_id=thread_id,
        stage=stage,
        is_complete=is_complete,
        lead_id=lead_id,
    )


@app.post("/chat/stream", tags=["chat"])
@limiter.limit(_rate_limit)
async def chat_stream(request: Request, payload: ChatRequest) -> StreamingResponse:
    """
    Streaming chat endpoint via Server-Sent Events (SSE).

    Same input as ``/chat``, but streams the assistant's reply
    token-by-token.  The frontend (nlux) connects to this endpoint
    for a real-time typing effect.

    SSE event format::

        data: {"token": "Hello"}
        data: {"token": " there"}
        data: {"token": "!"}
        data: {"done": true, "thread_id": "abc", "stage": "discovery"}
    """
    graph = _get_graph()
    thread_id = payload.thread_id or str(uuid.uuid4())
    thread_id_var.set(thread_id)

    logger.info(
        "Stream request: thread=%s, message=%.80s",
        thread_id,
        payload.message,
    )

    config = _run_config(thread_id)
    graph_input = {
        "messages": [HumanMessage(content=payload.message)],
    }

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events from the graph stream."""
        collected_reply = ""
        final_stage = ConversationStage.GREETING
        lead_id = None

        # Only stream tokens from these nodes (not extraction/router/scoring)
        _CONVERSATIONAL_NODES = {
            "greeting", "discovery", "qualification",
            "objection_handling", "lead_capture", "confirmation", "error",
        }

        try:
            async for event in graph.astream_events(
                graph_input,
                config=config,
                version="v2",
            ):
                kind = event.get("event", "")

                # Stream LLM tokens only from conversational nodes
                if kind == "on_chat_model_stream":
                    # Check which graph node this LLM call belongs to
                    node_name = event.get("metadata", {}).get("langgraph_node", "")
                    if node_name not in _CONVERSATIONAL_NODES:
                        continue

                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        token = chunk.content
                        collected_reply += token
                        yield f"data: {_sse_json({'token': token})}\n\n"

                # Capture final state when the graph completes
                elif kind == "on_chain_end":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        if "stage" in output:
                            raw_stage = output["stage"]
                            if isinstance(raw_stage, ConversationStage):
                                final_stage = raw_stage
                            elif isinstance(raw_stage, str):
                                with suppress(ValueError):
                                    final_stage = ConversationStage(raw_stage)
                        if "salesforce_lead_id" in output and output["salesforce_lead_id"]:
                            lead_id = output["salesforce_lead_id"]

        except Exception:
            logger.exception("Stream failed for thread %s", thread_id)
            yield f"data: {_sse_json({'error': 'Stream interrupted. Please try again.'})}\n\n"

        # Fetch the final checkpointed state so we can return the current
        # lead_data snapshot — the frontend uses this to render a live
        # "contact info collected so far" panel during lead capture.
        lead_data: dict[str, Any] = {}
        try:
            snapshot = await graph.aget_state(config)
            if snapshot and snapshot.values:
                lead_data = snapshot.values.get("lead_data", {}) or {}
        except Exception:
            logger.warning("Failed to fetch final state for thread %s", thread_id, exc_info=True)

        # Send completion event
        is_complete = final_stage == ConversationStage.COMPLETE
        yield (
            f"data: {_sse_json({'done': True, 'thread_id': thread_id, 'stage': final_stage.value, 'is_complete': is_complete, 'lead_id': lead_id, 'lead_data': lead_data})}\n\n"
        )

        logger.info(
            "Stream complete: thread=%s, stage=%s, tokens=%d",
            thread_id,
            final_stage.value,
            len(collected_reply),
        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/chat/init", tags=["chat"])
@limiter.limit(_rate_limit)
async def chat_init(request: Request) -> dict[str, Any]:
    """
    Initialise a new conversation and return the greeting.

    Creates a fresh thread, runs the greeting node, and returns the
    AI's opening message along with the thread_id for subsequent
    requests.  The frontend calls this once when the chat widget opens.
    """
    graph = _get_graph()
    thread_id = str(uuid.uuid4())
    thread_id_var.set(thread_id)

    logger.info("Initialising new conversation: thread=%s", thread_id)

    config = _run_config(thread_id)

    # Invoke with empty messages — the entry point router will
    # detect no human messages and route to the greeting node.
    try:
        result = await graph.ainvoke(
            {"messages": []},
            config=config,
        )
    except Exception as err:
        logger.exception("Greeting generation failed for thread %s", thread_id)
        raise HTTPException(
            status_code=500,
            detail="Failed to start conversation. Please try again.",
        ) from err

    greeting = _extract_latest_ai_reply(result)

    return {
        "thread_id": thread_id,
        "greeting": greeting,
        "stage": ConversationStage.GREETING.value,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_config(thread_id: str) -> dict[str, Any]:
    """
    Build the per-run RunnableConfig: the checkpointer thread id plus trace
    metadata/tags. ``session_id`` groups a conversation in LangSmith's Threads
    view; provider/model/tags make runs filterable. All keys propagate to every
    child run (nodes, LLM calls).
    """
    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
        "metadata": {**_trace_base["metadata"], "session_id": thread_id},
        "tags": _trace_base["tags"],
    }
    if _tracer is not None:
        config["callbacks"] = [_tracer]
    return config


def _extract_latest_ai_reply(result: dict) -> str:
    """
    Extract the most recent AIMessage content from the graph result.

    Falls back to a generic message if no AI reply is found (shouldn't
    happen in normal operation).
    """
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content

    logger.warning("No AIMessage found in graph result")
    return "I'm sorry, I wasn't able to generate a response. Could you try again?"


def _sse_json(data: dict) -> str:
    """Serialise a dict to a JSON string for SSE events."""
    import json

    return json.dumps(data, default=str)
