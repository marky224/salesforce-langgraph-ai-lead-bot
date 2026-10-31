"""
End-to-end conversation tests.

Uses a content-based mock LLM that inspects the system prompt to decide
what type of response to return (extraction JSON, router JSON, or
conversational text).  This is resilient to changes in graph routing
order — the mock doesn't care about call sequence, only about what the
graph is asking for.

Scoring is deterministic (see app.tools.qualification.compute_lead_score),
so the mock does not produce scores — assertions exercise the real scorer.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.graph import build_graph
from app.graph.nodes import set_llm
from app.models.schemas import ConversationStage

# ---------------------------------------------------------------------------
# Content-based mock LLM
# ---------------------------------------------------------------------------

class ContentBasedMockLLM:
    """
    Mock LLM that inspects the system prompt to determine the response type.

    For extraction prompts → returns extraction JSON
    For router prompts → returns routing JSON
    For conversational prompts → returns appropriate text

    The ``extraction_data`` dict is consumed in order — each extraction
    call pops the next item from the list.
    """

    def __init__(
        self,
        extraction_responses: list[dict] | None = None,
        router_responses: list[dict] | None = None,
    ):
        self._extraction_queue = list(extraction_responses or [])
        self._router_queue = list(router_responses or [])
        self._call_count = 0

    async def ainvoke(self, messages, **kwargs):
        self._call_count += 1
        system_prompt = ""
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt = msg.content
                break

        content = self._route_response(system_prompt)
        return MagicMock(content=content)

    def _route_response(self, prompt: str) -> str:
        prompt_lower = prompt.lower()

        # Extraction prompt
        if "data extraction" in prompt_lower and "return only a valid json" in prompt_lower:
            if self._extraction_queue:
                return json.dumps(self._extraction_queue.pop(0))
            return "{}"

        # Router prompt
        if "conversation stage router" in prompt_lower:
            if self._router_queue:
                return json.dumps(self._router_queue.pop(0))
            return json.dumps({"next_stage": "discovery", "reasoning": "default"})

        # Transcript summary prompt
        if "summarise the following sales chat" in prompt_lower:
            return "Test prospect discussed automation needs."

        # Site answer prompt (a detour that quotes site excerpts)
        if "answering a question about mark's website content" in prompt_lower:
            return "It runs ten security checks every week. Anything about timing I should know?"

        # Conversational prompts — return generic but appropriate text
        if "opening of the conversation" in prompt_lower:
            return "Hey there! I'm Alex. What brings you by today?"
        if "discovery phase" in prompt_lower:
            return "Tell me more about the challenges you're facing."
        if "qualification phase" in prompt_lower:
            return "Do you have a rough budget range in mind?"
        if "concern or objection" in prompt_lower:
            return "That's a fair concern. Let me address that."
        if "collect contact details" in prompt_lower:
            return "I'd love to connect you with our team. What's your name?"
        if "summarise what was discussed" in prompt_lower:
            return "Great — I've got everything. We'll be in touch!"

        # Fallback
        return "Thank you for chatting!"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_salesforce():
    """Mock all Salesforce API calls."""
    with patch("app.tools.salesforce._get_sf_client") as mock_client:
        sf = MagicMock()
        sf.query.return_value = {"totalSize": 0, "records": []}  # dedup miss → create
        sf.Lead.create.return_value = {"success": True, "id": "00Q000TEST00001"}
        sf.Task.create.return_value = {"success": True, "id": "00T000TEST00001"}
        sf.Lead.update.return_value = None
        mock_client.return_value = sf
        yield sf


# ---------------------------------------------------------------------------
# E2E: Full conversation flow
# ---------------------------------------------------------------------------

class TestFullConversationFlow:
    """Walk through a complete conversation from greeting to Salesforce."""

    @pytest.mark.asyncio
    async def test_greeting_to_completion(self, mock_salesforce):
        mock_llm = ContentBasedMockLLM(
            extraction_responses=[
                # Turn 1: user shares pain point
                {"qualification_data": {"pain_points": ["manual data entry"], "current_solution": "spreadsheets"}},
                # Turn 2: user shares team size
                {"qualification_data": {"company_size": "11-50"}},
                # Turn 3: user shares budget + timeline
                {"qualification_data": {"budget_range": "$10K-$50K", "timeline": "1-3 months", "decision_maker": True}},
                # Turn 4: user shares name
                {"lead_data": {"first_name": "Sarah", "last_name": "Chen"}},
                # Turn 5: user shares email + company
                {"lead_data": {"email": "sarah@acme.com", "company": "Acme Corp"}},
                # Turn 6: user confirms (empty extraction)
                {},
            ],
            router_responses=[
                # After turn 1 (discovery stage, fast-path skips router for greeting→discovery)
                # Turn 2: discovery → qualification
                {"next_stage": "qualification", "reasoning": "Have pain points, need budget"},
                # Turn 3: qualification → lead_capture
                {"next_stage": "lead_capture", "reasoning": "Qual complete"},
                # Turn 4: lead_capture → lead_capture (need email)
                {"next_stage": "lead_capture", "reasoning": "Need email"},
                # Turn 5: lead_capture → confirmation
                {"next_stage": "confirmation", "reasoning": "Have all contact info"},
            ],
        )
        set_llm(mock_llm)

        graph = build_graph()
        config = {"configurable": {"thread_id": "e2e-full-001"}}

        # Turn 0: Proactive greeting
        result = await graph.ainvoke({"messages": []}, config=config)
        assert result.get("stage") == ConversationStage.GREETING

        # Turn 1: Pain point
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="We have manual data entry problems")]},
            config=config,
        )

        # Turn 2: Team size
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="About 15 hours a week, team of 50")]},
            config=config,
        )

        # Turn 3: Budget + timeline
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="$10K-$50K budget, 1-3 months, I make the decisions")]},
            config=config,
        )

        # Turn 4: Name
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Sarah Chen")]},
            config=config,
        )

        # Turn 5: Email + company
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="sarah@acme.com, Acme Corp")]},
            config=config,
        )

        # Turn 6: Confirm
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Yes, looks good!")]},
            config=config,
        )

        # Verify final state
        assert result.get("salesforce_lead_id") == "00Q000TEST00001"
        assert result.get("salesforce_task_id") == "00T000TEST00001"
        assert result.get("stage") == ConversationStage.COMPLETE
        # Deterministic score: 18 (budget $10K-$50K) + 18 (timeline 1-3mo)
        # + 8 (size 11-50) + 15 (decision maker) + 5 (1 pain point)
        # + 7 (name+email+company, no phone) = 71
        assert result.get("lead_score") == 71

        # Verify Salesforce was called
        mock_salesforce.Lead.create.assert_called_once()
        lead_payload = mock_salesforce.Lead.create.call_args[0][0]
        assert lead_payload["LeadSource"] == "Web Chat"
        assert lead_payload["Lead_Score__c"] == 71


# ---------------------------------------------------------------------------
# E2E: Site question detour (answer from the site index, then back)
# ---------------------------------------------------------------------------

SITE_FIXTURE = Path(__file__).parent / "fixtures" / "site_index_v1.json"
M365_LINK = (
    "Read more: [M365 Guardian An LLM-Powered Microsoft 365 Administration Chatbot]"
    "(https://www.markandrewmarquez.com/portfolio/m365-guardian/)"
)


@pytest.fixture
def site_index():
    from app.tools.site_index import parse_site_index, set_site_index

    set_site_index(parse_site_index(
        SITE_FIXTURE.read_bytes(),
        content_type="application/json",
        final_url="https://www.markandrewmarquez.com/site-index.json",
        allowed_netloc="www.markandrewmarquez.com",
    ))
    yield
    set_site_index(None)


class TestSiteDetour:
    """A site question gets an answer with the page link; the funnel carries on."""

    @pytest.mark.asyncio
    async def test_site_question_mid_qualification_returns_to_qualification(
        self, mock_salesforce, site_index
    ):
        mock_llm = ContentBasedMockLLM(
            extraction_responses=[
                {"qualification_data": {"pain_points": ["manual data entry"], "current_solution": "spreadsheets"}},
                {"qualification_data": {"company_size": "11-50"}},
                {},  # the site question carries no lead data
                {"qualification_data": {"budget_range": "$10K-$50K", "timeline": "1-3 months", "decision_maker": True}},
                {"lead_data": {"first_name": "Sarah", "last_name": "Chen"}},
                {"lead_data": {"email": "sarah@acme.com", "company": "Acme Corp"}},
                {},
            ],
            router_responses=[
                {"next_stage": "qualification", "reasoning": "Have pain points"},
                {"next_stage": "site_question", "reasoning": "Asks about an article"},
                {"next_stage": "lead_capture", "reasoning": "Qual complete"},
                {"next_stage": "lead_capture", "reasoning": "Need email"},
                {"next_stage": "confirmation", "reasoning": "Have all contact info"},
            ],
        )
        set_llm(mock_llm)
        graph = build_graph()
        config = {"configurable": {"thread_id": "e2e-site-001"}}

        async def say(text):
            return await graph.ainvoke({"messages": [HumanMessage(content=text)]}, config=config)

        await graph.ainvoke({"messages": []}, config=config)
        await say("We have manual data entry problems")
        result = await say("About 15 hours a week, team of 50")
        assert result["stage"] == ConversationStage.QUALIFICATION

        # The detour: answered with the page link, stage left as it was.
        result = await say("What does M365 Guardian's weekly security report check?")
        assert result["stage"] == ConversationStage.QUALIFICATION
        assert result["site_question"] is False
        assert result["messages"][-1].content == (
            "It runs ten security checks every week. Anything about timing I should know?"
            f"\n\n{M365_LINK}"
        )

        # The funnel carries on from qualification to a completed lead.
        await say("$10K-$50K budget, 1-3 months, I make the decisions")
        await say("Sarah Chen")
        await say("sarah@acme.com, Acme Corp")
        result = await say("Yes, looks good!")

        assert result["stage"] == ConversationStage.COMPLETE
        assert result["salesforce_lead_id"] == "00Q000TEST00001"
        assert result["lead_score"] == 71  # same data as the full flow, same score

    @pytest.mark.asyncio
    async def test_a_first_message_about_the_site_gets_answered_then_discovery(self, site_index):
        mock_llm = ContentBasedMockLLM(
            extraction_responses=[{}, {"qualification_data": {"pain_points": ["manual data entry"]}}],
            router_responses=[{"next_stage": "site_question", "reasoning": "Asks about a project"}],
        )
        set_llm(mock_llm)
        graph = build_graph()
        config = {"configurable": {"thread_id": "e2e-site-002"}}

        await graph.ainvoke({"messages": []}, config=config)
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=(
                "I just read Mark's post on M365 Guardian. What does its weekly security report check?"
            ))]},
            config=config,
        )
        assert result["stage"] == ConversationStage.GREETING
        assert result["messages"][-1].content.endswith(M365_LINK)

        # Next turn: no site cue, so the greeting fast path goes to discovery.
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="We have manual data entry problems")]},
            config=config,
        )
        assert result["stage"] == ConversationStage.DISCOVERY


# ---------------------------------------------------------------------------
# E2E: Early exit flow
# ---------------------------------------------------------------------------

class TestEarlyExitFlow:
    """User leaves early — bot captures what it can."""

    @pytest.mark.asyncio
    async def test_partial_data_still_creates_lead(self, mock_salesforce):
        mock_llm = ContentBasedMockLLM(
            extraction_responses=[
                {"qualification_data": {"pain_points": ["CRM is slow"]}},
                {},  # "gotta go" — no new data
                {"lead_data": {"email": "quick@test.com"}},
                {},  # "ok" confirmation
            ],
            router_responses=[
                {"next_stage": "lead_capture", "reasoning": "User leaving"},
                {"next_stage": "confirmation", "reasoning": "Got email"},
            ],
        )
        set_llm(mock_llm)

        graph = build_graph()
        config = {"configurable": {"thread_id": "e2e-exit-001"}}

        await graph.ainvoke({"messages": []}, config=config)
        await graph.ainvoke(
            {"messages": [HumanMessage(content="Our CRM is really slow")]},
            config=config,
        )
        await graph.ainvoke(
            {"messages": [HumanMessage(content="Sorry gotta go")]},
            config=config,
        )
        await graph.ainvoke(
            {"messages": [HumanMessage(content="quick@test.com")]},
            config=config,
        )
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="ok")]},
            config=config,
        )

        assert result.get("salesforce_lead_id") is not None
        assert result.get("lead_score") == 7


# ---------------------------------------------------------------------------
# E2E: Salesforce failure
# ---------------------------------------------------------------------------

class TestSalesforceFailure:
    """Salesforce errors produce a friendly message, not a crash."""

    @pytest.mark.asyncio
    async def test_graceful_error_on_sf_failure(self):
        mock_llm = ContentBasedMockLLM(
            extraction_responses=[
                {"lead_data": {"email": "a@b.com", "last_name": "Test", "company": "TestCo"}},
                {},
            ],
            router_responses=[
                {"next_stage": "confirmation", "reasoning": "Have contact info"},
            ],
        )
        set_llm(mock_llm)

        with patch("app.tools.salesforce._get_sf_client") as mock_client:
            sf = MagicMock()
            sf.query.return_value = {"records": []}  # dedup miss → create is reached
            sf.Lead.create.side_effect = ConnectionError("Salesforce unreachable")
            mock_client.return_value = sf

            graph = build_graph()
            config = {"configurable": {"thread_id": "e2e-sffail-001"}}

            await graph.ainvoke({"messages": []}, config=config)
            await graph.ainvoke(
                {"messages": [HumanMessage(content="I'm a@b.com at TestCo, last name Test")]},
                config=config,
            )
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content="Yes confirmed")]},
                config=config,
            )

            # Should have an error-node message, not a crash
            messages = result.get("messages", [])
            last_ai = [m for m in messages if isinstance(m, type(messages[0])) and hasattr(m, 'content')]
            assert len(last_ai) > 0
            # The error node should have fired
            assert result.get("error") is None or "Salesforce" in str(result.get("error", ""))

    @pytest.mark.asyncio
    async def test_retry_after_a_partial_write_creates_no_second_lead(self):
        """
        The Lead is saved but its Task fails; the visitor's next message retries
        the write. With no email there is no dedup lookup, so only the Lead Id the
        failed turn kept stops a second Lead.
        """
        set_llm(ContentBasedMockLLM())

        with patch("app.tools.salesforce._get_sf_client") as mock_client:
            sf = MagicMock()
            sf.Lead.create.return_value = {"success": True, "id": "00Q000TEST00001"}
            sf.Task.create.side_effect = [
                ConnectionError("Salesforce unreachable"),
                {"success": True, "id": "00T000TEST00001"},
            ]
            mock_client.return_value = sf

            graph = build_graph()
            config = {"configurable": {"thread_id": "e2e-sfpartial-001"}}

            await graph.ainvoke({"messages": []}, config=config)
            # A confirmation with a company but no email (no name, so no guard asks for one).
            await graph.aupdate_state(
                config,
                {"stage": ConversationStage.CONFIRMATION, "lead_data": {"company": "TestCo"}},
                as_node="greeting",
            )
            failed = await graph.ainvoke(
                {"messages": [HumanMessage(content="Yes, that's correct")]}, config=config
            )
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content="ok thanks")]}, config=config
            )

        assert failed.get("salesforce_task_id") is None
        sf.query.assert_not_called()  # no email, no dedup lookup
        assert sf.Lead.create.call_count == 1
        assert sf.Task.create.call_count == 2
        assert sf.Task.create.call_args[0][0]["WhoId"] == "00Q000TEST00001"
        assert result["salesforce_lead_id"] == "00Q000TEST00001"
        assert result["salesforce_task_id"] == "00T000TEST00001"
