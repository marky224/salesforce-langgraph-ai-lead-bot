"""
LangGraph state schema for the AI Sales Lead Bot.

Defines the central TypedDict that flows through every node in the graph.
LangGraph persists and patches this state between node executions, so every
field the graph needs to read or write must be declared here.

Key design decisions:
- Uses `Annotated[list, operator.add]` for messages so that each node can
  *append* messages rather than replacing the entire history.
- Domain objects (LeadData, QualificationData, LeadScore) are stored as
  serialisable dicts and reconstituted via Pydantic when nodes need
  validation — this keeps the state checkpoint-friendly.
- `stage` drives conditional routing; the router examines it plus the
  completeness of lead_data / qualification_data to pick the next node.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import AnyMessage

from app.models.schemas import ConversationStage


# ---------------------------------------------------------------------------
# Reducer helpers
# ---------------------------------------------------------------------------
# `operator.add` tells LangGraph to *append* new items returned by a node
# rather than overwriting the existing list.  This is critical for messages
# and objections which accumulate over the conversation.


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class GraphState(TypedDict, total=False):
    """
    Central state object threaded through every LangGraph node.

    Attributes
    ----------
    messages : list[AnyMessage]
        Full conversation history (HumanMessage / AIMessage).  Uses the
        ``operator.add`` reducer so nodes only need to return *new* messages.

    stage : ConversationStage
        Current phase of the sales conversation.  Updated by the router
        after each conversational turn.

    lead_data : dict
        Incrementally populated contact info.  Serialised form of
        ``LeadData``; validated via Pydantic when writing to Salesforce.
        Keys: first_name, last_name, email, company, phone, title.

    qualification_data : dict
        Structured qualification signals.  Serialised form of
        ``QualificationData``.
        Keys: budget_range, timeline, company_size, pain_points,
              decision_maker, current_solution, goals.

    lead_score : int
        Numeric quality score (0–100) computed by the scoring node.

    lead_score_breakdown : dict[str, int]
        Per-dimension point breakdown so downstream consumers
        (Agentforce) understand the scoring rationale.

    objections : list[str]
        Objections or concerns raised by the prospect, accumulated
        across the conversation via ``operator.add``.

    transcript_summary : str
        Running plain-text summary of the conversation, refreshed
        by the confirmation node before Salesforce submission.

    salesforce_lead_id : str | None
        Populated by the Salesforce node after a Lead record is
        successfully created.

    salesforce_task_id : str | None
        Populated by the Salesforce node after the transcript Task
        is attached to the Lead.

    retry_count : int
        Number of soft re-engagement attempts when the user tries
        to exit early.  Capped at 1 to avoid being pushy.

    error : str | None
        Human-readable error message if something goes wrong in an
        external call (LLM, Salesforce).  Checked by the error node.
    """

    # --- Conversation history (append-only via reducer) --------------------
    messages: Annotated[list[AnyMessage], operator.add]

    # --- Conversation flow -------------------------------------------------
    stage: ConversationStage

    # --- Lead contact info (accumulated incrementally) ---------------------
    lead_data: dict[str, Any]

    # --- Qualification signals ---------------------------------------------
    qualification_data: dict[str, Any]

    # --- Scoring -----------------------------------------------------------
    lead_score: int
    lead_score_breakdown: dict[str, int]

    # --- Objections (append-only via reducer) ------------------------------
    objections: Annotated[list[str], operator.add]

    # --- Transcript --------------------------------------------------------
    transcript_summary: str

    # --- Salesforce integration --------------------------------------------
    salesforce_lead_id: Optional[str]
    salesforce_task_id: Optional[str]

    # --- Control flow helpers ----------------------------------------------
    retry_count: int
    error: Optional[str]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_initial_state() -> GraphState:
    """
    Return a clean initial state for a new conversation.

    Used when a new thread_id is created or when no checkpoint
    exists for the given thread.
    """
    return GraphState(
        messages=[],
        stage=ConversationStage.GREETING,
        lead_data={},
        qualification_data={
            "budget_range": "Unknown",
            "timeline": "Unknown",
            "company_size": "Unknown",
            "pain_points": [],
            "decision_maker": None,
            "current_solution": None,
            "goals": [],
        },
        lead_score=0,
        lead_score_breakdown={},
        objections=[],
        transcript_summary="",
        salesforce_lead_id=None,
        salesforce_task_id=None,
        retry_count=0,
        error=None,
    )
