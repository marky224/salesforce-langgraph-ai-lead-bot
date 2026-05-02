"""
Conditional edge logic for the AI Sales Lead Bot graph.

Edges in LangGraph are functions that examine the current ``GraphState`` and
return the **name of the next node** to execute.  This module provides two
kinds of routing:

1. **Post-extraction router** (``route_after_extraction``) — called after the
   extraction node processes a new human message.  Delegates to the LLM-based
   ``router_node`` or short-circuits to deterministic destinations when the
   answer is obvious (e.g. error recovery, conversation complete).

2. **Post-router dispatcher** (``route_after_router``) — called after the
   ``router_node`` has set ``state["stage"]`` to the next stage.  Maps the
   stage enum to the corresponding node name.

3. **Post-conversation check** (``route_after_conversation_node``) — called
   after any conversational node.  Decides whether to proceed to Salesforce
   submission or wait for the next human message.

The graph wiring in ``graph.py`` will reference these functions by name when
adding conditional edges.

Design notes:
- Deterministic routing is preferred where possible — the LLM router is only
  invoked when the next stage is genuinely ambiguous.
- Every routing function returns a ``str`` node name matching the node keys
  registered in the ``StateGraph``.
"""

from __future__ import annotations

import logging

from app.graph.prompts import (
    get_missing_contact_fields,
    get_missing_qualification_fields,
)
from app.graph.state import GraphState
from app.models.schemas import ConversationStage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node name constants (must match keys used in graph.py add_node calls)
# ---------------------------------------------------------------------------

NODE_GREETING = "greeting"
NODE_DISCOVERY = "discovery"
NODE_QUALIFICATION = "qualification"
NODE_OBJECTION_HANDLING = "objection_handling"
NODE_LEAD_CAPTURE = "lead_capture"
NODE_CONFIRMATION = "confirmation"
NODE_EXTRACTION = "extraction"
NODE_ROUTER = "router"
NODE_SCORING = "scoring"
NODE_SALESFORCE = "salesforce"
NODE_ERROR = "error"

# LangGraph special targets
END = "__end__"


# ---------------------------------------------------------------------------
# Stage → node mapping
# ---------------------------------------------------------------------------

_STAGE_TO_NODE: dict[ConversationStage, str] = {
    ConversationStage.GREETING: NODE_GREETING,
    ConversationStage.DISCOVERY: NODE_DISCOVERY,
    ConversationStage.QUALIFICATION: NODE_QUALIFICATION,
    ConversationStage.OBJECTION_HANDLING: NODE_OBJECTION_HANDLING,
    ConversationStage.LEAD_CAPTURE: NODE_LEAD_CAPTURE,
    ConversationStage.CONFIRMATION: NODE_CONFIRMATION,
    ConversationStage.COMPLETE: NODE_SCORING,  # score → salesforce → end
}


# ---------------------------------------------------------------------------
# Edge 1: After extraction — should we use the LLM router or short-circuit?
# ---------------------------------------------------------------------------

def route_after_extraction(state: GraphState) -> str:
    """
    Called after the extraction node has processed the latest human message.

    Fast-path rules (no LLM call needed):
    - If there's an error → error node.
    - If stage is COMPLETE → scoring node.
    - If stage is GREETING → always go to discovery next (first real turn).
    - If stage is CONFIRMATION → scoring (visitor confirmed, submit to SF).

    Otherwise → delegate to the LLM-based router node for a contextual
    decision.
    """
    # Error recovery takes priority
    if state.get("error"):
        logger.info("Edge: extraction → error (error present)")
        return NODE_ERROR

    stage = state.get("stage", ConversationStage.GREETING)

    # Conversation already finished
    if stage == ConversationStage.COMPLETE:
        logger.info("Edge: extraction → scoring (stage is COMPLETE)")
        return NODE_SCORING

    # First turn after greeting — always move to discovery
    if stage == ConversationStage.GREETING:
        logger.info("Edge: extraction → discovery (post-greeting)")
        return NODE_DISCOVERY

    # Visitor confirmed details — proceed to scoring + Salesforce
    if stage == ConversationStage.CONFIRMATION:
        # Check if confirmation was accepted (not an objection)
        is_affirmative = _latest_message_is_affirmative(state)

        # --- Debug: log the exact message and check result ---
        from langchain_core.messages import HumanMessage as _HM
        _latest = ""
        for _m in reversed(state.get("messages", [])):
            if isinstance(_m, _HM):
                _latest = _m.content
                break
        logger.info(
            "CONFIRMATION check: is_affirmative=%s, word_count=%d, message='%.120s'",
            is_affirmative,
            len(_latest.split()),
            _latest,
        )
        # --- End debug ---

        if is_affirmative:
            logger.info("Edge: extraction → scoring (confirmation accepted)")
            return NODE_SCORING
        else:
            # They want to correct something — route back through LLM router
            logger.info("Edge: extraction → router (confirmation correction)")
            return NODE_ROUTER

    # Default: let the LLM router decide
    logger.info("Edge: extraction → router (contextual decision needed)")
    return NODE_ROUTER


# ---------------------------------------------------------------------------
# Edge 2: After router — dispatch to the chosen stage's node
# ---------------------------------------------------------------------------

def route_after_router(state: GraphState) -> str:
    """
    Called after the router node has updated ``state["stage"]``.

    Maps the stage enum to the corresponding node name.  If the stage
    is unrecognised, falls back to discovery.
    """
    stage = state.get("stage", ConversationStage.DISCOVERY)

    # Normalise string to enum if needed
    if isinstance(stage, str):
        try:
            stage = ConversationStage(stage)
        except ValueError:
            logger.warning("Unrecognised stage '%s', falling back to discovery", stage)
            return NODE_DISCOVERY

    # Hard guard: if we have the visitor's name but a required field is
    # missing, don't let the LLM router skip to confirmation — go back to
    # lead_capture to ask for whatever's missing.  Only applies when name is
    # present (conversation progressed enough to ask); early-exit flows with
    # only an email are allowed to proceed as-is.
    if stage == ConversationStage.CONFIRMATION:
        lead_data = state.get("lead_data", {})
        has_name = lead_data.get("first_name") or lead_data.get("last_name")
        title_ok = _title_satisfied(state)
        if has_name and (
            not lead_data.get("email")
            or not lead_data.get("company")
            or not title_ok
        ):
            missing = []
            if not lead_data.get("email"):
                missing.append("email")
            if not lead_data.get("company"):
                missing.append("company")
            if not title_ok:
                missing.append("title")
            logger.info(
                "Edge: router wanted confirmation but %s missing → lead_capture",
                "+".join(missing),
            )
            return NODE_LEAD_CAPTURE

    # Hard guard: if the router picks lead_capture but name + title + email +
    # company are already captured (or title has had its one explicit ask),
    # force confirmation.  Phone stays optional, and title is bounded by the
    # title_capture_attempts counter so a visitor who declines doesn't trap
    # the conversation in lead_capture forever.
    if stage == ConversationStage.LEAD_CAPTURE:
        lead_data = state.get("lead_data", {})
        has_name = lead_data.get("first_name") or lead_data.get("last_name")
        if (
            has_name
            and lead_data.get("email")
            and lead_data.get("company")
            and _title_satisfied(state)
        ):
            logger.info(
                "Edge: router picked lead_capture but required fields (name+title+email+company) satisfied → confirmation"
            )
            return NODE_CONFIRMATION

    node = _STAGE_TO_NODE.get(stage, NODE_DISCOVERY)
    logger.info("Edge: router → %s (stage=%s)", node, stage.value)
    return node


# ---------------------------------------------------------------------------
# Edge 3: After a conversational node — wait for input or proceed?
# ---------------------------------------------------------------------------

def route_after_conversation_node(state: GraphState) -> str:
    """
    Called after any conversational node (greeting, discovery, etc.)
    has produced an AI response.

    The graph pauses here to wait for the next human message.  In
    LangGraph this means we route to ``__end__`` — the graph will
    resume at the entry point when the next ``HumanMessage`` arrives.

    Exception: after the confirmation node, if we already have enough
    data, we can proceed directly to scoring without waiting.
    """
    stage = state.get("stage", ConversationStage.GREETING)

    if stage == ConversationStage.COMPLETE:
        logger.info("Edge: conversation → scoring (auto-proceed)")
        return NODE_SCORING

    # Default: pause and wait for next human message
    logger.info("Edge: conversation → __end__ (awaiting human input)")
    return END


# ---------------------------------------------------------------------------
# Edge 4: After scoring — proceed to Salesforce
# ---------------------------------------------------------------------------

def route_after_scoring(state: GraphState) -> str:
    """
    Called after the scoring node.  Always proceeds to the Salesforce node
    to create the Lead and Task records.

    If there's an error from scoring, routes to error node instead.
    """
    if state.get("error"):
        logger.info("Edge: scoring → error")
        return NODE_ERROR

    logger.info("Edge: scoring → salesforce")
    return NODE_SALESFORCE


# ---------------------------------------------------------------------------
# Edge 5: After Salesforce — end the graph
# ---------------------------------------------------------------------------

def route_after_salesforce(state: GraphState) -> str:
    """
    Called after the Salesforce node.  If an error occurred, routes to the
    error node to deliver a graceful fallback message.  Otherwise ends.
    """
    if state.get("error"):
        logger.info("Edge: salesforce → error (SF failure)")
        return NODE_ERROR

    logger.info("Edge: salesforce → __end__ (complete)")
    return END


# ---------------------------------------------------------------------------
# Edge 6: After error — end the graph
# ---------------------------------------------------------------------------

def route_after_error(state: GraphState) -> str:
    """
    After the error node delivers its fallback message, end the graph turn.

    The conversation can still continue if the visitor sends another message
    (the graph will re-enter at extraction).
    """
    logger.info("Edge: error → __end__")
    return END


# ---------------------------------------------------------------------------
# Entry point: first node when a new human message arrives
# ---------------------------------------------------------------------------

def route_entry_point(state: GraphState) -> str:
    """
    Determine the first node to run when a new human message enters the graph.

    - If this is the very first interaction (no messages yet or only a system
      greeting), start with the greeting node.
    - Otherwise, run extraction on the new human message.
    """
    messages = state.get("messages", [])

    if not messages:
        logger.info("Edge: entry → greeting (no messages yet)")
        return NODE_GREETING

    # Check if we have any human messages yet
    from langchain_core.messages import HumanMessage
    has_human = any(isinstance(m, HumanMessage) for m in messages)

    if not has_human:
        logger.info("Edge: entry → greeting (no human messages yet)")
        return NODE_GREETING

    logger.info("Edge: entry → extraction (processing new human message)")
    return NODE_EXTRACTION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _title_satisfied(state: GraphState) -> bool:
    """
    Title is satisfied when either:
    - the visitor has provided one (``lead_data.title`` is set), or
    - the lead_capture node has already made one explicit ask for title
      (``title_capture_attempts >= 1``).

    The counter mechanism prevents a visitor who declines to share their
    title from trapping the conversation in lead_capture forever — title
    is required, but bounded by a one-shot retry.
    """
    lead_data = state.get("lead_data", {})
    if lead_data.get("title"):
        return True
    return state.get("title_capture_attempts", 0) >= 1


def _latest_message_is_affirmative(state: GraphState) -> bool:
    """
    Quick heuristic check: does the latest human message look like a
    confirmation / agreement?

    Used by ``route_after_extraction`` to decide whether the visitor
    accepted the confirmation summary or wants to make corrections.

    The check is intentionally generous — it's better to proceed to
    scoring on an ambiguous message than to loop the visitor back
    through another round of questions.  If the visitor genuinely
    wants to correct something, they'll say so explicitly (e.g.
    "actually my email is wrong" or "no, change the company name").
    """
    from langchain_core.messages import HumanMessage

    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            text = msg.content.lower().strip()

            # --- Negative signals: visitor wants to correct something ---
            negatives = {
                "no", "nope", "wrong", "incorrect", "change", "update",
                "fix", "actually", "wait", "hold on", "not right",
                "that's wrong", "thats wrong", "mistake",
            }
            # If the message is short and clearly negative, reject
            if len(text.split()) <= 10 and any(n in text for n in negatives):
                # But "no problem" / "no worries" are actually affirmative
                false_negatives = {"no problem", "no worries", "no issues"}
                if not any(fn in text for fn in false_negatives):
                    return False

            # --- Positive signals ---
            affirmatives = {
                "yes", "yeah", "yep", "yup", "correct", "that's right",
                "thats right", "looks good", "looks great", "perfect",
                "confirmed", "confirm", "all good", "good to go", "sure",
                "ok", "okay", "sounds good", "right", "absolutely",
                "exactly", "spot on", "you got it", "that works",
                "great", "awesome", "thanks", "thank you", "lgtm",
                "go ahead", "proceed", "submit", "send it", "ship it",
                "all correct", "looks correct", "that's correct",
            }
            # Generous check: up to 20 words and contains any affirmative
            if len(text.split()) <= 20 and any(a in text for a in affirmatives):
                return True

            # Fallback: if the message is very short (1-3 words) and
            # doesn't contain a negative, treat it as affirmative.
            # Covers things like "cool", "nice", "done", thumbs-up text.
            if len(text.split()) <= 3 and not any(n in text for n in negatives):
                return True

            return False

    return False


def get_all_route_destinations() -> dict[str, list[str]]:
    """
    Return a mapping of each routing function's possible destinations.

    Used by ``graph.py`` when registering conditional edges — LangGraph
    requires the set of possible target nodes upfront for validation.
    """
    return {
        "route_entry_point": [NODE_GREETING, NODE_EXTRACTION],
        "route_after_extraction": [
            NODE_ERROR,
            NODE_SCORING,
            NODE_DISCOVERY,
            NODE_ROUTER,
        ],
        "route_after_router": [
            NODE_GREETING,
            NODE_DISCOVERY,
            NODE_QUALIFICATION,
            NODE_OBJECTION_HANDLING,
            NODE_LEAD_CAPTURE,
            NODE_CONFIRMATION,
            NODE_SCORING,
        ],
        "route_after_conversation_node": [NODE_SCORING, END],
        "route_after_scoring": [NODE_ERROR, NODE_SALESFORCE],
        "route_after_salesforce": [NODE_ERROR, END],
        "route_after_error": [END],
    }
