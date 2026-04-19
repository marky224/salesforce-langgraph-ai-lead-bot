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
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from app.config import configure_logging, get_llm, get_settings
from app.graph.graph import build_graph
from app.graph.nodes import set_llm
from app.graph.state import create_initial_state
from app.models.schemas import ChatRequest, ChatResponse, ConversationStage, HealthResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state (populated at startup)
# ---------------------------------------------------------------------------

_graph = None


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
    global _graph  # noqa: PLW0603

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

    # Compile graph
    _graph = build_graph()
    logger.info("LangGraph compiled and ready")

    yield

    # --- Shutdown ---
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

# --- CORS ---
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("CORS origins: %s", settings.cors_origin_list)


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
    return HealthResponse(version=get_settings().app_version)


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
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Synchronous chat endpoint.

    Accepts a user message and optional thread_id.  Runs one full
    graph turn (extraction → routing → conversation node) and returns
    the assistant's reply.

    If no ``thread_id`` is provided, a new conversation is started.
    """
    graph = _get_graph()
    thread_id = request.thread_id or str(uuid.uuid4())

    logger.info(
        "Chat request: thread=%s, message=%.80s",
        thread_id,
        request.message,
    )

    config = {"configurable": {"thread_id": thread_id}}

    # Build input — add the new human message
    graph_input = {
        "messages": [HumanMessage(content=request.message)],
    }

    try:
        # Run the graph for one full turn
        result = await graph.ainvoke(graph_input, config=config)
    except Exception:
        logger.exception("Graph invocation failed for thread %s", thread_id)
        raise HTTPException(
            status_code=500,
            detail="An error occurred processing your message. Please try again.",
        )

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
async def chat_stream(request: ChatRequest) -> StreamingResponse:
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
    thread_id = request.thread_id or str(uuid.uuid4())

    logger.info(
        "Stream request: thread=%s, message=%.80s",
        thread_id,
        request.message,
    )

    config = {"configurable": {"thread_id": thread_id}}
    graph_input = {
        "messages": [HumanMessage(content=request.message)],
    }

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events from the graph stream."""
        collected_reply = ""
        final_stage = ConversationStage.GREETING
        lead_id = None

        try:
            async for event in graph.astream_events(
                graph_input,
                config=config,
                version="v2",
            ):
                kind = event.get("event", "")

                # Stream LLM tokens as they arrive
                if kind == "on_chat_model_stream":
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
                                try:
                                    final_stage = ConversationStage(raw_stage)
                                except ValueError:
                                    pass
                        if "salesforce_lead_id" in output and output["salesforce_lead_id"]:
                            lead_id = output["salesforce_lead_id"]

        except Exception:
            logger.exception("Stream failed for thread %s", thread_id)
            yield f"data: {_sse_json({'error': 'Stream interrupted. Please try again.'})}\n\n"

        # Send completion event
        is_complete = final_stage == ConversationStage.COMPLETE
        yield (
            f"data: {_sse_json({'done': True, 'thread_id': thread_id, 'stage': final_stage.value, 'is_complete': is_complete, 'lead_id': lead_id})}\n\n"
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
async def chat_init() -> dict[str, Any]:
    """
    Initialise a new conversation and return the greeting.

    Creates a fresh thread, runs the greeting node, and returns the
    AI's opening message along with the thread_id for subsequent
    requests.  The frontend calls this once when the chat widget opens.
    """
    graph = _get_graph()
    thread_id = str(uuid.uuid4())

    logger.info("Initialising new conversation: thread=%s", thread_id)

    config = {"configurable": {"thread_id": thread_id}}

    # Invoke with empty messages — the entry point router will
    # detect no human messages and route to the greeting node.
    try:
        result = await graph.ainvoke(
            {"messages": []},
            config=config,
        )
    except Exception:
        logger.exception("Greeting generation failed for thread %s", thread_id)
        raise HTTPException(
            status_code=500,
            detail="Failed to start conversation. Please try again.",
        )

    greeting = _extract_latest_ai_reply(result)

    return {
        "thread_id": thread_id,
        "greeting": greeting,
        "stage": ConversationStage.GREETING.value,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
