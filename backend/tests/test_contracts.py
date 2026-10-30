"""
Contract guard tests.

These assert *cross-cutting contracts* that span multiple modules and would
otherwise break silently. See CLAUDE.md ("Cross-cutting contracts") and
_private/docs/build/04-lead-scoring.md.

Contract: the deterministic scorer's per-dimension score tables must stay in
lock-step with the qualification enums. If a BudgetRange / Timeline /
CompanySize value is added or renamed without updating
``app.tools.qualification``, that value would silently score 0 (the scorer
falls back to ``.get(value, 0)``). This test fails loudly instead.
"""

from __future__ import annotations

from app.models.schemas import BudgetRange, CompanySize, Timeline
from app.tools.qualification import (
    _BUDGET_SCORES,
    _COMPANY_SIZE_SCORES,
    _TIMELINE_SCORES,
)


class TestEnumScoreTableContract:
    """Score tables and their enums must stay in lock-step (no missing/extra keys)."""

    def test_budget_table_matches_enum(self):
        assert set(_BUDGET_SCORES) == {b.value for b in BudgetRange}

    def test_timeline_table_matches_enum(self):
        assert set(_TIMELINE_SCORES) == {t.value for t in Timeline}

    def test_company_size_table_matches_enum(self):
        assert set(_COMPANY_SIZE_SCORES) == {c.value for c in CompanySize}


class TestSiteQuestionContract:
    """
    The router's ``site_question`` answer is a detour, not a stage. If it ever
    became a ConversationStage value it would enter the shared .value contract
    (scorer, prompts, Salesforce, the widget's ConversationStage typedef) and
    reach the SSE ``done`` frame.
    """

    def test_site_question_is_not_a_conversation_stage(self):
        from app.graph.nodes import SITE_QUESTION
        from app.models.schemas import ConversationStage

        assert SITE_QUESTION == "site_question"
        assert SITE_QUESTION not in {stage.value for stage in ConversationStage}

    def test_site_answer_is_a_router_destination(self):
        from app.graph.edges import NODE_SITE_ANSWER, get_all_route_destinations

        assert NODE_SITE_ANSWER in get_all_route_destinations()["route_after_router"]
