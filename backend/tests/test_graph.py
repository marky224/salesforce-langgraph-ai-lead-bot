"""
Unit tests for LangGraph nodes.

All LLM calls are mocked so these tests run without API keys or
network access.  Each test verifies that the node:
- Returns the correct state keys
- Produces the expected message type (AIMessage)
- Updates stage and data fields appropriately
- Handles edge cases (empty state, missing data)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.graph.nodes import (
    _merge_dict,
    _safe_parse_json,
    confirmation_node,
    discovery_node,
    error_node,
    extraction_node,
    greeting_node,
    lead_capture_node,
    objection_handler_node,
    qualification_node,
    router_node,
    scoring_node,
    set_llm,
)
from app.graph.state import create_initial_state
from app.models.schemas import ConversationStage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_llm():
    """Inject a mock LLM that returns configurable responses."""
    mock = AsyncMock()
    mock.ainvoke = AsyncMock(return_value=MagicMock(content="Mock response"))
    set_llm(mock)
    yield mock


@pytest.fixture
def base_state():
    """Return a clean initial state for tests."""
    return create_initial_state()


@pytest.fixture
def conversation_state():
    """Return a state with some conversation history."""
    state = create_initial_state()
    state["messages"] = [
        AIMessage(content="Hi! I'm Alex. What brings you by?"),
        HumanMessage(content="We're struggling with manual data entry."),
    ]
    state["stage"] = ConversationStage.DISCOVERY
    return state


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------

class TestSafeParseJson:
    """Tests for _safe_parse_json."""

    def test_valid_json(self):
        assert _safe_parse_json('{"key": "value"}') == {"key": "value"}

    def test_json_with_markdown_fences(self):
        assert _safe_parse_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_invalid_json_returns_empty(self):
        assert _safe_parse_json("not json at all") == {}

    def test_empty_string_returns_empty(self):
        assert _safe_parse_json("") == {}

    def test_nested_json(self):
        result = _safe_parse_json('{"lead_data": {"email": "a@b.com"}}')
        assert result["lead_data"]["email"] == "a@b.com"


class TestMergeDict:
    """Tests for _merge_dict."""

    def test_overwrites_scalar(self):
        result = _merge_dict({"a": "old"}, {"a": "new"})
        assert result["a"] == "new"

    def test_skips_none(self):
        result = _merge_dict({"a": "keep"}, {"a": None})
        assert result["a"] == "keep"

    def test_appends_lists(self):
        result = _merge_dict(
            {"items": ["x"]},
            {"items": ["y", "z"]},
        )
        assert result["items"] == ["x", "y", "z"]

    def test_deduplicates_lists(self):
        result = _merge_dict(
            {"items": ["x", "y"]},
            {"items": ["y", "z"]},
        )
        assert result["items"] == ["x", "y", "z"]

    def test_adds_new_keys(self):
        result = _merge_dict({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_preserves_original(self):
        original = {"a": 1}
        _merge_dict(original, {"b": 2})
        assert "b" not in original  # original not mutated


# ---------------------------------------------------------------------------
# Conversational node tests
# ---------------------------------------------------------------------------

class TestGreetingNode:
    """Tests for greeting_node."""

    @pytest.mark.asyncio
    async def test_returns_ai_message(self, base_state, mock_llm):
        mock_llm.ainvoke.return_value = MagicMock(
            content="Hey there! I'm Alex. What brings you by?"
        )

        result = await greeting_node(base_state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert "Alex" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_sets_greeting_stage(self, base_state):
        result = await greeting_node(base_state)
        assert result["stage"] == ConversationStage.GREETING

    @pytest.mark.asyncio
    async def test_calls_llm(self, base_state, mock_llm):
        await greeting_node(base_state)
        mock_llm.ainvoke.assert_called_once()


class TestDiscoveryNode:
    """Tests for discovery_node."""

    @pytest.mark.asyncio
    async def test_returns_ai_message(self, conversation_state):
        result = await discovery_node(conversation_state)
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)

    @pytest.mark.asyncio
    async def test_sets_discovery_stage(self, conversation_state):
        result = await discovery_node(conversation_state)
        assert result["stage"] == ConversationStage.DISCOVERY


class TestQualificationNode:
    """Tests for qualification_node."""

    @pytest.mark.asyncio
    async def test_returns_ai_message(self, conversation_state):
        result = await qualification_node(conversation_state)
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)

    @pytest.mark.asyncio
    async def test_sets_qualification_stage(self, conversation_state):
        result = await qualification_node(conversation_state)
        assert result["stage"] == ConversationStage.QUALIFICATION


class TestObjectionHandlerNode:
    """Tests for objection_handler_node."""

    @pytest.mark.asyncio
    async def test_returns_ai_message(self, conversation_state):
        conversation_state["messages"].append(
            HumanMessage(content="This sounds really expensive.")
        )
        result = await objection_handler_node(conversation_state)
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)

    @pytest.mark.asyncio
    async def test_sets_objection_stage(self, conversation_state):
        result = await objection_handler_node(conversation_state)
        assert result["stage"] == ConversationStage.OBJECTION_HANDLING


class TestLeadCaptureNode:
    """Tests for lead_capture_node."""

    @pytest.mark.asyncio
    async def test_returns_ai_message(self, conversation_state):
        result = await lead_capture_node(conversation_state)
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)

    @pytest.mark.asyncio
    async def test_sets_lead_capture_stage(self, conversation_state):
        result = await lead_capture_node(conversation_state)
        assert result["stage"] == ConversationStage.LEAD_CAPTURE


class TestConfirmationNode:
    """Tests for confirmation_node."""

    @pytest.mark.asyncio
    async def test_returns_ai_message(self, conversation_state, mock_llm):
        mock_llm.ainvoke.return_value = MagicMock(
            content="It was great chatting! Does that all look correct?"
        )

        result = await confirmation_node(conversation_state)

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert result["stage"] == ConversationStage.CONFIRMATION

    @pytest.mark.asyncio
    async def test_makes_one_llm_call(self, conversation_state, mock_llm):
        # Transcript summary is now generated in scoring_node, not here.
        await confirmation_node(conversation_state)
        assert mock_llm.ainvoke.call_count == 1


# ---------------------------------------------------------------------------
# Extraction node tests
# ---------------------------------------------------------------------------

class TestExtractionNode:
    """Tests for extraction_node."""

    @pytest.mark.asyncio
    async def test_extracts_lead_data(self, conversation_state, mock_llm):
        mock_llm.ainvoke.return_value = MagicMock(
            content=json.dumps({
                "lead_data": {"email": "sarah@acme.com", "company": "Acme"},
                "qualification_data": {"pain_points": ["manual data entry"]},
            })
        )

        result = await extraction_node(conversation_state)

        assert result["lead_data"]["email"] == "sarah@acme.com"
        assert result["lead_data"]["company"] == "Acme"

    @pytest.mark.asyncio
    async def test_extracts_qualification_data(self, conversation_state, mock_llm):
        mock_llm.ainvoke.return_value = MagicMock(
            content=json.dumps({
                "qualification_data": {
                    "budget_range": "$10K-$50K",
                    "pain_points": ["slow CRM"],
                },
            })
        )

        result = await extraction_node(conversation_state)

        assert "slow CRM" in result["qualification_data"]["pain_points"]

    @pytest.mark.asyncio
    async def test_extracts_objections(self, conversation_state, mock_llm):
        mock_llm.ainvoke.return_value = MagicMock(
            content=json.dumps({
                "objections": ["Concerned about migration complexity"],
            })
        )

        result = await extraction_node(conversation_state)
        assert "Concerned about migration complexity" in result["objections"]

    @pytest.mark.asyncio
    async def test_empty_extraction_returns_empty(self, conversation_state, mock_llm):
        mock_llm.ainvoke.return_value = MagicMock(content="{}")

        result = await extraction_node(conversation_state)
        assert result == {}

    @pytest.mark.asyncio
    async def test_merges_with_existing_data(self, conversation_state, mock_llm):
        conversation_state["lead_data"] = {"first_name": "Sarah"}
        conversation_state["qualification_data"]["pain_points"] = ["slow CRM"]

        mock_llm.ainvoke.return_value = MagicMock(
            content=json.dumps({
                "lead_data": {"last_name": "Chen"},
                "qualification_data": {"pain_points": ["manual reporting"]},
            })
        )

        result = await extraction_node(conversation_state)

        # Existing data preserved, new data merged
        assert result["lead_data"]["first_name"] == "Sarah"
        assert result["lead_data"]["last_name"] == "Chen"
        assert "slow CRM" in result["qualification_data"]["pain_points"]
        assert "manual reporting" in result["qualification_data"]["pain_points"]


# ---------------------------------------------------------------------------
# Scoring node tests
# ---------------------------------------------------------------------------

class TestScoringNode:
    """scoring_node — deterministic score + LLM transcript summary."""

    @pytest.mark.asyncio
    async def test_returns_zero_for_empty_data(self, conversation_state, mock_llm):
        mock_llm.ainvoke.return_value = MagicMock(content="Summary prose.")

        result = await scoring_node(conversation_state)

        assert result["lead_score"] == 0
        assert "budget" in result["lead_score_breakdown"]

    @pytest.mark.asyncio
    async def test_scores_perfect_lead(self, conversation_state, mock_llm):
        mock_llm.ainvoke.return_value = MagicMock(content="Summary prose.")
        conversation_state["qualification_data"] = {
            "budget_range": "$100K+",
            "timeline": "Immediate",
            "company_size": "1000+",
            "decision_maker": True,
            "pain_points": ["a", "b", "c"],
        }
        conversation_state["lead_data"] = {
            "first_name": "S",
            "last_name": "C",
            "email": "s@a.com",
            "company": "A",
            "phone": "1",
        }

        result = await scoring_node(conversation_state)

        assert result["lead_score"] == 100

    @pytest.mark.asyncio
    async def test_generates_transcript_summary(self, conversation_state, mock_llm):
        mock_llm.ainvoke.return_value = MagicMock(content="Summary prose.")

        result = await scoring_node(conversation_state)

        assert result["transcript_summary"] == "Summary prose."

    @pytest.mark.asyncio
    async def test_makes_single_llm_call(self, conversation_state, mock_llm):
        mock_llm.ainvoke.return_value = MagicMock(content="Summary prose.")

        await scoring_node(conversation_state)

        # Score is deterministic; only the transcript summary uses the LLM.
        assert mock_llm.ainvoke.call_count == 1


# ---------------------------------------------------------------------------
# Router node tests
# ---------------------------------------------------------------------------

class TestRouterNode:
    """Tests for router_node."""

    @pytest.mark.asyncio
    async def test_returns_valid_stage(self, conversation_state, mock_llm):
        mock_llm.ainvoke.return_value = MagicMock(
            content=json.dumps({
                "next_stage": "qualification",
                "reasoning": "Need budget info.",
            })
        )

        result = await router_node(conversation_state)
        assert result["stage"] == ConversationStage.QUALIFICATION

    @pytest.mark.asyncio
    async def test_defaults_to_discovery_for_invalid_stage(
        self, conversation_state, mock_llm
    ):
        mock_llm.ainvoke.return_value = MagicMock(
            content=json.dumps({"next_stage": "invalid_stage"})
        )

        result = await router_node(conversation_state)
        assert result["stage"] == ConversationStage.DISCOVERY

    @pytest.mark.asyncio
    async def test_increments_retry_on_exit_signal(
        self, conversation_state, mock_llm
    ):
        conversation_state["messages"].append(
            HumanMessage(content="No thanks, gotta go")
        )
        conversation_state["retry_count"] = 0

        mock_llm.ainvoke.return_value = MagicMock(
            content=json.dumps({"next_stage": "lead_capture"})
        )

        result = await router_node(conversation_state)
        assert result.get("retry_count") == 1


# ---------------------------------------------------------------------------
# Error node tests
# ---------------------------------------------------------------------------

class TestErrorNode:
    """Tests for error_node."""

    @pytest.mark.asyncio
    async def test_salesforce_error_message(self, base_state):
        base_state["error"] = "Salesforce error: connection refused"

        result = await error_node(base_state)

        assert len(result["messages"]) == 1
        assert "technical hiccup" in result["messages"][0].content
        assert result["error"] is None  # error cleared

    @pytest.mark.asyncio
    async def test_generic_error_message(self, base_state):
        base_state["error"] = "LLM timeout"

        result = await error_node(base_state)

        assert "try sending" in result["messages"][0].content.lower() or \
               "try again" in result["messages"][0].content.lower()
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_clears_error_field(self, base_state):
        base_state["error"] = "Some error"

        result = await error_node(base_state)
        assert result["error"] is None
