"""
LangGraph node functions for the AI Sales Lead Bot.

Each node is a plain function with signature ``(state: GraphState) -> dict``
that returns a *partial* state update.  LangGraph merges the returned dict
into the existing state (using reducers for annotated fields like ``messages``).

Node categories:
- **Conversational nodes** — call the LLM with a stage-specific prompt and
  return the assistant reply as a new ``AIMessage``.
- **Extraction node** — parses the latest visitor message for structured data
  (lead info, qualification signals, objections) and patches state dicts.
- **Scoring node** — computes a 0-100 lead score from qualification data.
- **Salesforce node** — creates Lead + Task records via the Salesforce API.
- **Error node** — produces a graceful fallback reply when something breaks.

All LLM calls go through a shared ``_invoke_llm`` helper so the provider
(Anthropic / OpenAI / Groq / xAI) is swappable from ``config.py``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.graph.prompts import (
    CONFIRMATION_PROMPT,
    DISCOVERY_PROMPT,
    EXTRACTION_PROMPT,
    GREETING_PROMPT,
    LEAD_CAPTURE_PROMPT,
    OBJECTION_HANDLING_PROMPT,
    PERSONA,
    QUALIFICATION_PROMPT,
    ROUTER_PROMPT,
    SCORING_PROMPT,
    TRANSCRIPT_SUMMARY_PROMPT,
    format_known_info,
    format_transcript,
    get_missing_contact_fields,
    get_missing_qualification_fields,
)
from app.graph.state import GraphState
from app.models.schemas import ConversationStage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Sentinel — replaced at app startup by ``config.get_llm()``
_llm = None


def set_llm(llm_instance: Any) -> None:
    """
    Inject the configured LLM instance at application startup.

    Called once from ``server.py`` after ``config.get_llm()`` resolves
    the provider.  Every node uses this shared instance.
    """
    global _llm  # noqa: PLW0603
    _llm = llm_instance
    logger.info("LLM instance set: %s", type(llm_instance).__name__)


def _get_llm() -> Any:
    """Return the configured LLM, raising early if not initialised."""
    if _llm is None:
        raise RuntimeError(
            "LLM not initialised — call nodes.set_llm() at startup."
        )
    return _llm


# Default values for state keys that might not exist after checkpoint restore
_STATE_DEFAULTS: dict[str, Any] = {
    "lead_data": {},
    "qualification_data": {
        "budget_range": "Unknown",
        "timeline": "Unknown",
        "company_size": "Unknown",
        "pain_points": [],
        "decision_maker": None,
        "current_solution": None,
        "goals": [],
    },
    "lead_score": 0,
    "lead_score_breakdown": {},
    "objections": [],
    "transcript_summary": "",
    "salesforce_lead_id": None,
    "salesforce_task_id": None,
    "retry_count": 0,
    "error": None,
}


def _gs(state: GraphState, key: str) -> Any:
    """Get a state value with a safe default if the key is missing."""
    return state.get(key, _STATE_DEFAULTS.get(key))


async def _invoke_llm(system_prompt: str, messages: list) -> str:
    """
    Send a system prompt + message history to the LLM and return the
    assistant's reply as a plain string.

    Works with any LangChain chat model (Anthropic, OpenAI, Groq, xAI).
    """
    llm = _get_llm()
    full_messages = [SystemMessage(content=system_prompt)] + messages

    try:
        response = await llm.ainvoke(full_messages)
        return response.content
    except Exception:
        logger.exception("LLM invocation failed")
        raise


def _safe_parse_json(text: str) -> dict:
    """
    Parse a JSON string returned by the LLM, stripping markdown fences,
    leading/trailing whitespace, and any preamble text before the JSON.

    Handles common LLM quirks:
    - Markdown fences: ```json ... ```
    - Preamble text: "Assistant: {..." or "Here is the JSON:\n{..."
    - Trailing text after the closing brace

    Returns ``{}`` on failure.
    """
    cleaned = text.strip()

    # Strip ```json ... ``` fencing if present
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]  # remove first line
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    # Strip any preamble before the first '{' (e.g. "Assistant: {...")
    # This handles xAI/Grok's tendency to prefix JSON with "Assistant:"
    brace_pos = cleaned.find("{")
    if brace_pos > 0:
        preamble = cleaned[:brace_pos].strip()
        if preamble:
            logger.debug("Stripping JSON preamble: '%.80s'", preamble)
        cleaned = cleaned[brace_pos:]

    # Strip any trailing text after the last '}'
    last_brace = cleaned.rfind("}")
    if last_brace >= 0:
        cleaned = cleaned[: last_brace + 1]

    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        logger.warning("Failed to parse LLM JSON output: %.200s", text)
        return {}

def _merge_dict(base: dict, updates: dict) -> dict:
    """
    Shallow-merge *updates* into *base*, skipping ``None`` values and
    appending to lists rather than replacing them.

    Used to incrementally patch ``lead_data`` and ``qualification_data``.
    """
    merged = dict(base)
    for key, value in updates.items():
        if value is None:
            continue
        if isinstance(value, list) and isinstance(merged.get(key), list):
            # Append new items, dedup
            existing = set(merged[key])
            merged[key] = merged[key] + [v for v in value if v not in existing]
        else:
            merged[key] = value
    return merged


# ---------------------------------------------------------------------------
# Conversational nodes
# ---------------------------------------------------------------------------

async def greeting_node(state: GraphState) -> dict:
    """
    Generate the opening greeting message.

    This node fires once at the start of the conversation.  It does NOT
    expect a prior human message — the graph can invoke it immediately
    to proactively greet the visitor.
    """
    logger.info("Node: greeting")

    prompt = GREETING_PROMPT.format(persona=PERSONA)
    reply = await _invoke_llm(prompt, list(state.get("messages", [])))

    return {
        "messages": [AIMessage(content=reply)],
        "stage": ConversationStage.GREETING,
    }


async def discovery_node(state: GraphState) -> dict:
    """
    Explore the visitor's pain points, current solutions, and goals.

    Injected context:
    - ``transcript``: full conversation so far
    - ``known_info``: what we've already captured (to avoid repeats)
    """
    logger.info("Node: discovery")

    transcript = format_transcript(state.get("messages", []))
    known_info = format_known_info(_gs(state, "lead_data"), _gs(state, "qualification_data"))

    prompt = DISCOVERY_PROMPT.format(
        persona=PERSONA,
        transcript=transcript,
        known_info=known_info,
    )
    reply = await _invoke_llm(prompt, list(state.get("messages", [])))

    return {
        "messages": [AIMessage(content=reply)],
        "stage": ConversationStage.DISCOVERY,
    }


async def qualification_node(state: GraphState) -> dict:
    """
    Ask about one missing qualification field (budget, timeline, etc.).

    Only asks about fields not yet captured, using
    ``get_missing_qualification_fields`` to determine the gap list.
    """
    logger.info("Node: qualification")

    transcript = format_transcript(state.get("messages", []))
    known_info = format_known_info(_gs(state, "lead_data"), _gs(state, "qualification_data"))
    missing = get_missing_qualification_fields(_gs(state, "qualification_data"))

    prompt = QUALIFICATION_PROMPT.format(
        persona=PERSONA,
        transcript=transcript,
        known_info=known_info,
        missing_fields=", ".join(missing) if missing else "All fields captured.",
    )
    reply = await _invoke_llm(prompt, list(state.get("messages", [])))

    return {
        "messages": [AIMessage(content=reply)],
        "stage": ConversationStage.QUALIFICATION,
    }


async def objection_handler_node(state: GraphState) -> dict:
    """
    Address a concern or objection raised in the visitor's latest message.

    The latest human message is passed as the ``objection`` context so the
    LLM focuses its response on the specific concern.
    """
    logger.info("Node: objection_handling")

    transcript = format_transcript(state.get("messages", []))

    # Find the latest human message to use as the objection text
    latest_human = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            latest_human = msg.content
            break

    prompt = OBJECTION_HANDLING_PROMPT.format(
        persona=PERSONA,
        transcript=transcript,
        objection=latest_human,
    )
    reply = await _invoke_llm(prompt, list(state.get("messages", [])))

    return {
        "messages": [AIMessage(content=reply)],
        "stage": ConversationStage.OBJECTION_HANDLING,
    }


async def lead_capture_node(state: GraphState) -> dict:
    """
    Collect one missing contact field (name, email, company, phone).

    Uses ``get_missing_contact_fields`` to determine what's still needed
    and asks for one field at a time.
    """
    logger.info("Node: lead_capture")

    transcript = format_transcript(state.get("messages", []))
    known_info = format_known_info(_gs(state, "lead_data"), _gs(state, "qualification_data"))
    missing = get_missing_contact_fields(_gs(state, "lead_data"))

    prompt = LEAD_CAPTURE_PROMPT.format(
        persona=PERSONA,
        transcript=transcript,
        known_info=known_info,
        missing_contact_fields=", ".join(missing) if missing else "All contact info captured.",
    )
    reply = await _invoke_llm(prompt, list(state.get("messages", [])))

    return {
        "messages": [AIMessage(content=reply)],
        "stage": ConversationStage.LEAD_CAPTURE,
    }


async def confirmation_node(state: GraphState) -> dict:
    """
    Summarise the conversation, confirm captured details, and set
    next-step expectations before handing off to Salesforce.
    """
    logger.info("Node: confirmation")

    transcript = format_transcript(state.get("messages", []))

    # Only pass user-facing contact fields — not internal qualification fields
    # like decision_maker, company_size, or budget enumerations.
    ld = _gs(state, "lead_data")
    contact_parts: list[str] = []
    name = " ".join(p for p in (ld.get("first_name"), ld.get("last_name")) if p)
    if name:
        contact_parts.append(f"Name: {name}")
    if ld.get("title"):
        contact_parts.append(f"Title: {ld['title']}")
    if ld.get("email"):
        contact_parts.append(f"Email: {ld['email']}")
    if ld.get("company"):
        contact_parts.append(f"Company: {ld['company']}")
    if ld.get("phone"):
        contact_parts.append(f"Phone: {ld['phone']}")
    contact_summary = "\n".join(contact_parts) if contact_parts else "No contact info captured yet."

    # Qualification summary: only human-readable fields
    qd = _gs(state, "qualification_data")
    qual_parts = []
    if qd.get("pain_points"):
        qual_parts.append(f"Key challenges: {'; '.join(qd['pain_points'])}")
    if qd.get("goals"):
        qual_parts.append(f"Goals: {'; '.join(qd['goals'])}")
    if qd.get("budget_range") and qd["budget_range"] != "Unknown":
        qual_parts.append(f"Budget: {qd['budget_range']}")
    if qd.get("timeline") and qd["timeline"] != "Unknown":
        qual_parts.append(f"Timeline: {qd['timeline']}")

    prompt = CONFIRMATION_PROMPT.format(
        persona=PERSONA,
        transcript=transcript,
        lead_summary=contact_summary,
        qualification_summary="\n".join(qual_parts) if qual_parts else "Limited info collected.",
    )
    reply = await _invoke_llm(prompt, list(state.get("messages", [])))

    return {
        "messages": [AIMessage(content=reply)],
        "stage": ConversationStage.CONFIRMATION,
    }


# ---------------------------------------------------------------------------
# Extraction node
# ---------------------------------------------------------------------------

async def extraction_node(state: GraphState) -> dict:
    """
    Parse the latest visitor message for new lead / qualification data.

    Runs after every human message.  Uses a dedicated extraction prompt
    that returns structured JSON.  Merges extracted values into existing
    state dicts without overwriting previously captured data.

    Returns a state patch with updated ``lead_data``,
    ``qualification_data``, and ``objections``.
    """
    logger.info("Node: extraction")

    transcript = format_transcript(state.get("messages", []))
    current_data = json.dumps(
        {
            "lead_data": _gs(state, "lead_data"),
            "qualification_data": _gs(state, "qualification_data"),
        },
        indent=2,
    )

    prompt = EXTRACTION_PROMPT.format(
        transcript=transcript,
        current_data=current_data,
    )
    raw = await _invoke_llm(prompt, [])
    extracted = _safe_parse_json(raw)

    if not extracted:
        logger.debug("Extraction returned empty — no new data in latest message.")
        return {}

    result: dict[str, Any] = {}

    # Merge lead data
    if "lead_data" in extracted and isinstance(extracted["lead_data"], dict):
        result["lead_data"] = _merge_dict(
            _gs(state, "lead_data"), extracted["lead_data"]
        )

    # Merge qualification data
    if "qualification_data" in extracted and isinstance(extracted["qualification_data"], dict):
        result["qualification_data"] = _merge_dict(
            _gs(state, "qualification_data"), extracted["qualification_data"]
        )

    # Append objections (uses operator.add reducer)
    if "objections" in extracted and isinstance(extracted["objections"], list):
        new_objections = [o for o in extracted["objections"] if o]
        if new_objections:
            result["objections"] = new_objections

    return result


# ---------------------------------------------------------------------------
# Scoring node
# ---------------------------------------------------------------------------

async def scoring_node(state: GraphState) -> dict:
    """
    Compute a 0-100 lead quality score from qualification + contact data,
    and generate the transcript summary for the Salesforce Description field.

    The transcript summary is generated here (not in confirmation_node) so
    that it never gets streamed to the frontend — this node is excluded from
    the streaming conversational node list.
    """
    logger.info("Node: scoring")

    prompt = SCORING_PROMPT.format(
        qualification_json=json.dumps(_gs(state, "qualification_data"), indent=2),
        lead_json=json.dumps(_gs(state, "lead_data"), indent=2),
    )
    raw = await _invoke_llm(prompt, [])
    parsed = _safe_parse_json(raw)

    score = parsed.get("score", 0)
    breakdown = parsed.get("breakdown", {})

    # Clamp score to valid range
    score = max(0, min(100, int(score)))

    # Generate transcript summary for Salesforce Description field.
    # Done here so it is never streamed to the frontend.
    transcript = format_transcript(state.get("messages", []))
    summary_prompt = TRANSCRIPT_SUMMARY_PROMPT.format(transcript=transcript)
    summary = await _invoke_llm(summary_prompt, [])

    return {
        "lead_score": score,
        "lead_score_breakdown": breakdown,
        "transcript_summary": summary,
    }


# ---------------------------------------------------------------------------
# Salesforce node
# ---------------------------------------------------------------------------

async def salesforce_node(state: GraphState) -> dict:
    """
    Create a Lead and attach a Task (conversation transcript) in Salesforce.

    Imports the Salesforce tool functions at call time to avoid circular
    imports and to keep this module testable with mocks.

    On success, populates ``salesforce_lead_id`` and ``salesforce_task_id``.
    On failure, sets ``error`` so the error node can respond.
    """
    logger.info("Node: salesforce")

    try:
        # Late import — the tools module depends on config which may not
        # be available during testing / import time.
        from app.tools.salesforce import create_lead, create_transcript_task

        # Build the full transcript text
        transcript_text = format_transcript(state.get("messages", []))

        # Create Lead
        lead_id = await create_lead(
            lead_data=_gs(state, "lead_data"),
            qualification_data=_gs(state, "qualification_data"),
            lead_score=_gs(state, "lead_score"),
            description=state.get("transcript_summary", ""),
        )
        logger.info("Salesforce Lead created: %s", lead_id)

        # Create Task with transcript
        task_id = await create_transcript_task(
            lead_id=lead_id,
            transcript=transcript_text,
        )
        logger.info("Salesforce Task created: %s", task_id)

        return {
            "salesforce_lead_id": lead_id,
            "salesforce_task_id": task_id,
            "stage": ConversationStage.COMPLETE,
        }

    except Exception as exc:
        logger.exception("Salesforce integration failed")
        return {
            "error": f"Salesforce error: {exc}",
            "stage": ConversationStage.COMPLETE,
        }


# ---------------------------------------------------------------------------
# Router node
# ---------------------------------------------------------------------------

async def router_node(state: GraphState) -> dict:
    """
    Decide which conversation stage should come next.

    Uses the ``ROUTER_PROMPT`` with an LLM call to make a contextual
    routing decision.  Returns the updated ``stage`` and optionally
    increments ``retry_count`` if the visitor tried to leave early.

    This node does NOT produce a user-visible message — it only updates
    the ``stage`` field for the conditional edge to read.
    """
    logger.info("Node: router")

    # Find latest human message
    latest_message = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            latest_message = msg.content
            break

    lead_summary = format_known_info(_gs(state, "lead_data"), {})
    qual_summary = format_known_info({}, _gs(state, "qualification_data"))

    prompt = ROUTER_PROMPT.format(
        current_stage=state["stage"].value if isinstance(state["stage"], ConversationStage) else state["stage"],
        lead_data_summary=lead_summary,
        qualification_data_summary=qual_summary,
        latest_message=latest_message,
        retry_count=state.get("retry_count", 0),
    )
    raw = await _invoke_llm(prompt, [])
    parsed = _safe_parse_json(raw)

    next_stage_str = parsed.get("next_stage", "discovery")
    reasoning = parsed.get("reasoning", "")
    logger.info("Router decision: %s — %s", next_stage_str, reasoning)

    # Map string to enum (fallback to discovery if unrecognised)
    try:
        next_stage = ConversationStage(next_stage_str)
    except ValueError:
        logger.warning("Unrecognised stage '%s', defaulting to discovery", next_stage_str)
        next_stage = ConversationStage.DISCOVERY

    result: dict[str, Any] = {"stage": next_stage}

    # Detect early-exit attempt: if visitor seems to be leaving and we
    # haven't retried yet, the router may keep the current stage.
    # We track retry_count so the router prompt can decide appropriately.
    exit_signals = {"bye", "no thanks", "not interested", "gotta go", "leave", "stop"}
    if any(signal in latest_message.lower() for signal in exit_signals):
        current_retry = state.get("retry_count", 0)
        if current_retry == 0:
            result["retry_count"] = 1

    return result


# ---------------------------------------------------------------------------
# Error node
# ---------------------------------------------------------------------------

async def error_node(state: GraphState) -> dict:
    """
    Produce a graceful fallback message when an error has occurred.

    Clears the ``error`` field after handling so the graph doesn't
    get stuck in an error loop.
    """
    logger.warning("Node: error — %s", state.get("error", "Unknown error"))

    error_msg = state.get("error", "")

    if "Salesforce" in error_msg:
        reply = (
            "I've captured all your information — thank you! I ran into a "
            "small technical hiccup saving your details, but don't worry — "
            "our team has been notified and someone will reach out to you "
            "shortly. Thanks for your patience!"
        )
    else:
        reply = (
            "I'm sorry, I hit a small snag on my end. Could you try sending "
            "that again? If the problem persists, you can always reach our "
            "team directly at the contact info on this page."
        )

    return {
        "messages": [AIMessage(content=reply)],
        "error": None,
    }
