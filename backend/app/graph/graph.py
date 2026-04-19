"""
LangGraph graph builder and compilation for the AI Sales Lead Bot.

This module is the single place where the conversation graph is assembled.
It registers every node, wires the conditional edges, attaches a checkpointer
for multi-turn state persistence, and exposes the compiled graph as a
module-level callable.

Usage::

    from app.graph.graph import build_graph

    # At application startup (after LLM is configured):
    graph = build_graph()

    # Per-request invocation with thread-level memory:
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Hi there")]},
        config={"configurable": {"thread_id": "abc-123"}},
    )

The graph supports both ``ainvoke`` (full run, returns final state) and
``astream`` (token-level streaming via LangGraph's streaming interface).
"""

from __future__ import annotations

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.graph.edges import (
    NODE_CONFIRMATION,
    NODE_DISCOVERY,
    NODE_ERROR,
    NODE_EXTRACTION,
    NODE_GREETING,
    NODE_LEAD_CAPTURE,
    NODE_OBJECTION_HANDLING,
    NODE_QUALIFICATION,
    NODE_ROUTER,
    NODE_SALESFORCE,
    NODE_SCORING,
    route_after_conversation_node,
    route_after_error,
    route_after_extraction,
    route_after_router,
    route_after_salesforce,
    route_after_scoring,
    route_entry_point,
)
from app.graph.nodes import (
    confirmation_node,
    discovery_node,
    error_node,
    extraction_node,
    greeting_node,
    lead_capture_node,
    objection_handler_node,
    qualification_node,
    router_node,
    salesforce_node,
    scoring_node,
)
from app.graph.state import GraphState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(checkpointer: MemorySaver | None = None) -> StateGraph:
    """
    Construct and compile the full conversation graph.

    Parameters
    ----------
    checkpointer : MemorySaver | None
        State persistence backend.  When ``None`` a default in-memory
        ``MemorySaver`` is created — suitable for single-process
        deployments and local development.  For production with multiple
        replicas, pass a database-backed checkpointer (e.g.
        ``AsyncSqliteSaver`` or ``PostgresSaver``).

    Returns
    -------
    CompiledGraph
        A compiled LangGraph ready for ``ainvoke`` / ``astream`` calls.
        Each invocation requires a ``config`` dict with
        ``{"configurable": {"thread_id": "<unique-id>"}}`` so the
        checkpointer can persist and restore per-conversation state.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()
        logger.info("Using default in-memory MemorySaver checkpointer")

    graph = StateGraph(GraphState)

    # ------------------------------------------------------------------
    # 1. Register nodes
    # ------------------------------------------------------------------

    # Conversational nodes (produce AI replies)
    graph.add_node(NODE_GREETING, greeting_node)
    graph.add_node(NODE_DISCOVERY, discovery_node)
    graph.add_node(NODE_QUALIFICATION, qualification_node)
    graph.add_node(NODE_OBJECTION_HANDLING, objection_handler_node)
    graph.add_node(NODE_LEAD_CAPTURE, lead_capture_node)
    graph.add_node(NODE_CONFIRMATION, confirmation_node)

    # Data processing nodes
    graph.add_node(NODE_EXTRACTION, extraction_node)
    graph.add_node(NODE_ROUTER, router_node)
    graph.add_node(NODE_SCORING, scoring_node)

    # Integration nodes
    graph.add_node(NODE_SALESFORCE, salesforce_node)
    graph.add_node(NODE_ERROR, error_node)

    logger.debug("Registered %d nodes", 11)

    # ------------------------------------------------------------------
    # 2. Set conditional entry point
    # ------------------------------------------------------------------
    # When the graph is invoked, ``route_entry_point`` inspects the
    # incoming state to decide whether this is a brand-new conversation
    # (→ greeting) or a continuation (→ extraction).

    graph.set_conditional_entry_point(
        route_entry_point,
        {
            NODE_GREETING: NODE_GREETING,
            NODE_EXTRACTION: NODE_EXTRACTION,
        },
    )

    # ------------------------------------------------------------------
    # 3. Wire edges from each node
    # ------------------------------------------------------------------

    # --- Greeting → wait for human input ---
    graph.add_conditional_edges(
        NODE_GREETING,
        route_after_conversation_node,
        {
            NODE_SCORING: NODE_SCORING,
            END: END,
        },
    )

    # --- Extraction → deterministic fast-path or LLM router ---
    graph.add_conditional_edges(
        NODE_EXTRACTION,
        route_after_extraction,
        {
            NODE_ERROR: NODE_ERROR,
            NODE_SCORING: NODE_SCORING,
            NODE_DISCOVERY: NODE_DISCOVERY,
            NODE_ROUTER: NODE_ROUTER,
        },
    )

    # --- LLM Router → dispatch to the chosen conversation node ---
    graph.add_conditional_edges(
        NODE_ROUTER,
        route_after_router,
        {
            NODE_GREETING: NODE_GREETING,
            NODE_DISCOVERY: NODE_DISCOVERY,
            NODE_QUALIFICATION: NODE_QUALIFICATION,
            NODE_OBJECTION_HANDLING: NODE_OBJECTION_HANDLING,
            NODE_LEAD_CAPTURE: NODE_LEAD_CAPTURE,
            NODE_CONFIRMATION: NODE_CONFIRMATION,
            NODE_SCORING: NODE_SCORING,
        },
    )

    # --- Conversation nodes → wait for next human message ---
    for node_name in [
        NODE_DISCOVERY,
        NODE_QUALIFICATION,
        NODE_OBJECTION_HANDLING,
        NODE_LEAD_CAPTURE,
        NODE_CONFIRMATION,
    ]:
        graph.add_conditional_edges(
            node_name,
            route_after_conversation_node,
            {
                NODE_SCORING: NODE_SCORING,
                END: END,
            },
        )

    # --- Scoring → Salesforce ---
    graph.add_conditional_edges(
        NODE_SCORING,
        route_after_scoring,
        {
            NODE_ERROR: NODE_ERROR,
            NODE_SALESFORCE: NODE_SALESFORCE,
        },
    )

    # --- Salesforce → end or error ---
    graph.add_conditional_edges(
        NODE_SALESFORCE,
        route_after_salesforce,
        {
            NODE_ERROR: NODE_ERROR,
            END: END,
        },
    )

    # --- Error → end (allows conversation to continue on next message) ---
    graph.add_conditional_edges(
        NODE_ERROR,
        route_after_error,
        {
            END: END,
        },
    )

    # ------------------------------------------------------------------
    # 4. Compile with checkpointer
    # ------------------------------------------------------------------

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("Graph compiled successfully with %d nodes", 11)

    return compiled
