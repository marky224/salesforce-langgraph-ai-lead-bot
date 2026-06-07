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
