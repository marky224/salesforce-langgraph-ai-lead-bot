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
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import get_settings
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
    TRANSCRIPT_SUMMARY_PROMPT,
    format_known_info,
    format_transcript,
    get_missing_contact_fields,
    get_missing_qualification_fields,
)
from app.graph.state import GraphState
from app.models.schemas import ConversationStage
from app.tools.qualification import compute_lead_score

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


def is_llm_ready() -> bool:
    """True once set_llm() has injected a model — used by the readiness probe."""
    return _llm is not None


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


# ---------------------------------------------------------------------------
# Structured-output schemas (opt-in via settings.llm_structured_output)
# ---------------------------------------------------------------------------
# Lenient ``str`` fields on purpose: a provider quirk shouldn't trip Pydantic
# validation (which would force a fallback). Downstream code already normalizes
# enums (e.g. the scorer's table lookups, the router's ConversationStage cast).

class _ExtractedLead(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None
    company: str | None = None
    phone: str | None = None
    title: str | None = None


class _ExtractedQualification(BaseModel):
    budget_range: str | None = None
    timeline: str | None = None
    company_size: str | None = None
    pain_points: list[str] = Field(default_factory=list)
    decision_maker: bool | None = None
    current_solution: str | None = None
    goals: list[str] = Field(default_factory=list)


class _ExtractionResult(BaseModel):
    lead_data: _ExtractedLead | None = None
    qualification_data: _ExtractedQualification | None = None
    objections: list[str] = Field(default_factory=list)


class _RouterDecision(BaseModel):
    next_stage: str
    reasoning: str = ""


# Lightweight parse-failure metric (PR 4 will wire real observability).
_parse_failure_count = 0


def get_parse_failure_count() -> int:
    """Return the running count of LLM JSON parse failures."""
    return _parse_failure_count


# ---------------------------------------------------------------------------
# LLM call telemetry (token / cost / latency capture)
# ---------------------------------------------------------------------------

# USD per 1M tokens, keyed by a model-name substring. Only models we actually
# run are listed; an unknown model logs tokens with ``cost_usd=None``. Extend
# when PR B's A/B picks a model.
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "grok-4.20-0309-reasoning": (1.25, 2.50),  # (input_per_1m, output_per_1m)
}


def _compute_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """USD cost for one call, or ``None`` if the model isn't in the price table."""
    for key, (in_price, out_price) in _MODEL_PRICING.items():
        if key in model:
            return round(input_tokens / 1e6 * in_price + output_tokens / 1e6 * out_price, 6)
    return None


def _log_llm_call(node: str, response: Any, latency_ms: int) -> None:
    """
    Emit one structured ``llm_call`` event: which node called the model, the
    model name, latency, token counts, and computed USD cost.

    Defensive: ``usage_metadata`` is ``None`` on a streamed call unless the model
    was built with ``stream_usage=True`` (see config.py), and a test fake may omit
    it — either way we log zero tokens / no cost rather than raise. The JSON log
    formatter surfaces the ``llm_call`` payload as queryable fields (Log
    Analytics); text mode renders the readable one-liner.
    """
    usage = getattr(response, "usage_metadata", None)
    if not isinstance(usage, dict):
        usage = {}
    meta = getattr(response, "response_metadata", None)
    model = meta.get("model_name", "unknown") if isinstance(meta, dict) else "unknown"

    input_tokens = usage.get("input_tokens") or 0
    output_tokens = usage.get("output_tokens") or 0
    total_tokens = usage.get("total_tokens") or (input_tokens + output_tokens)
    cost_usd = _compute_cost_usd(model, input_tokens, output_tokens)

    logger.info(
        "llm_call node=%s model=%s latency_ms=%d in=%d out=%d cost_usd=%s",
        node,
        model,
        latency_ms,
        input_tokens,
        output_tokens,
        cost_usd,
        extra={
            "llm_call": {
                "node": node,
                "model": model,
                "latency_ms": latency_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost_usd,
            }
        },
    )


async def _invoke_llm(system_prompt: str, messages: list, *, node: str) -> str:
    """
    Send a system prompt + message history to the LLM and return the
    assistant's reply as a plain string.

    Works with any LangChain chat model (Anthropic, OpenAI, Groq, xAI).
    Per-call timeout and bounded retries are enforced at the model layer
    (see ``config.get_llm`` — ``llm_timeout_seconds`` / ``llm_max_retries``),
    so every node call funnelling through here inherits them.  Every call emits
    an ``llm_call`` telemetry event (token / cost / latency) tagged with ``node``.
    """
    llm = _get_llm()
    full_messages = [SystemMessage(content=system_prompt)] + messages

    start = time.perf_counter()
    try:
        response = await llm.ainvoke(full_messages)
    except Exception:
        logger.exception("LLM invocation failed")
        raise
    _log_llm_call(node, response, int((time.perf_counter() - start) * 1000))
    return response.content


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
        global _parse_failure_count  # noqa: PLW0603
        _parse_failure_count += 1
        logger.warning(
            "Failed to parse LLM JSON output (count=%d): %.200s",
            _parse_failure_count,
            text,
        )
        return {}


async def _invoke_structured(
    system_prompt: str, messages: list, schema: type[BaseModel], *, node: str
) -> dict:
    """
    Structured LLM call with a graceful fallback.

    When ``settings.llm_structured_output`` is enabled, use the provider's
    ``with_structured_output(schema)``; on ANY error fall back to the plain-text
    ``_invoke_llm`` + ``_safe_parse_json`` path.  Returns a plain dict either way,
    so callers don't need to know which path produced it.
    """
    if get_settings().llm_structured_output:
        try:
            structured = _get_llm().with_structured_output(schema)
            result = await structured.ainvoke(
                [SystemMessage(content=system_prompt)] + messages
            )
            if isinstance(result, BaseModel):
                return result.model_dump(exclude_none=True)
            if isinstance(result, dict):
                return result
        except Exception:
            logger.warning(
                "Structured output failed; falling back to JSON parse", exc_info=True
            )

    raw = await _invoke_llm(system_prompt, messages, node=node)
    return _safe_parse_json(raw)

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
    reply = await _invoke_llm(prompt, list(state.get("messages", [])), node="greeting")

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
    reply = await _invoke_llm(prompt, list(state.get("messages", [])), node="discovery")

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
    reply = await _invoke_llm(prompt, list(state.get("messages", [])), node="qualification")

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
    reply = await _invoke_llm(prompt, list(state.get("messages", [])), node="objection_handling")

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
    reply = await _invoke_llm(prompt, list(state.get("messages", [])), node="lead_capture")

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
    reply = await _invoke_llm(prompt, list(state.get("messages", [])), node="confirmation")

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
    extracted = await _invoke_structured(prompt, [], _ExtractionResult, node="extraction")

    if not extracted:
        logger.debug("Extraction returned empty — no new data in latest message.")
        return {}

    result: dict[str, Any] = {}

    # Merge lead data
    if extracted.get("lead_data") and isinstance(extracted["lead_data"], dict):
        result["lead_data"] = _merge_dict(
            _gs(state, "lead_data"), extracted["lead_data"]
        )

    # Merge qualification data
    if extracted.get("qualification_data") and isinstance(extracted["qualification_data"], dict):
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

    Scoring is deterministic — see compute_lead_score in tools.qualification.
    The transcript summary is LLM-generated and produced here (not in
    confirmation_node) so it never streams to the frontend.
    """
    logger.info("Node: scoring")

    score_result = compute_lead_score(
        qualification_data=_gs(state, "qualification_data"),
        lead_data=_gs(state, "lead_data"),
    )

    transcript = format_transcript(state.get("messages", []))
    summary_prompt = TRANSCRIPT_SUMMARY_PROMPT.format(transcript=transcript)
    summary = await _invoke_llm(summary_prompt, [], node="scoring")

    return {
        "lead_score": score_result["score"],
        "lead_score_breakdown": score_result["breakdown"],
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
        from app.tools.salesforce import (
            create_lead,
            create_transcript_task,
            find_lead_by_email,
        )

        # Build the full transcript text
        transcript_text = format_transcript(state.get("messages", []))

        # Dedup: reuse an existing Lead for this email rather than creating a
        # duplicate (repeat visitors / retries shouldn't spam the CRM). The
        # transcript Task still attaches; the org side owns Rating/derivation.
        lead_data = _gs(state, "lead_data")
        existing_id = await find_lead_by_email(lead_data.get("email"))
        if existing_id:
            lead_id = existing_id
            logger.info("Reusing existing Salesforce Lead %s (dedup by email)", lead_id)
        else:
            lead_id = await create_lead(
                lead_data=lead_data,
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
    parsed = await _invoke_structured(prompt, [], _RouterDecision, node="router")

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

    error_msg = state.get("error") or ""

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


# ---------------------------------------------------------------------------
# Turn-cap node (abuse guard)
# ---------------------------------------------------------------------------

async def turn_cap_node(state: GraphState) -> dict:
    """
    Politely end a runaway thread.

    Fires from the entry point when a thread exceeds
    ``settings.max_thread_messages``.  Appends one fixed message and makes no
    LLM call, so a hammered thread can't keep driving model usage.  Leaves
    ``stage`` unchanged, so it re-fires on each further message.
    """
    logger.info("Node: turn_cap (thread message cap reached)")

    reply = (
        "Thanks for the great conversation! We've covered a lot here, so I'm "
        "going to wrap up this chat for now. Our team will follow up with you "
        "soon — and you're always welcome to start a fresh chat anytime."
    )
    return {"messages": [AIMessage(content=reply)]}
