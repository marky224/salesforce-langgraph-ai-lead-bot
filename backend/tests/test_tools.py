"""
Tests for tools: qualification scoring and Salesforce integration.

Qualification tests run without mocks (pure deterministic logic).
Salesforce tests mock the simple_salesforce client.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.tools.qualification import (
    assess_qualification_completeness,
    compute_lead_score,
    generate_qualification_summary,
)


# ---------------------------------------------------------------------------
# Qualification scoring tests
# ---------------------------------------------------------------------------

class TestComputeLeadScore:
    """Tests for the deterministic scoring engine."""

    def test_perfect_score(self):
        result = compute_lead_score(
            qualification_data={
                "budget_range": "$100K+",
                "timeline": "Immediate",
                "company_size": "1000+",
                "decision_maker": True,
                "pain_points": ["a", "b", "c"],
            },
            lead_data={
                "first_name": "Sarah",
                "last_name": "Chen",
                "email": "s@acme.com",
                "company": "Acme",
                "phone": "555-0100",
            },
        )
        assert result["score"] == 100
        assert result["priority"] == "High"

    def test_minimum_score(self):
        result = compute_lead_score(
            qualification_data={
                "budget_range": "Unknown",
                "timeline": "Unknown",
                "company_size": "Unknown",
                "decision_maker": None,
                "pain_points": [],
            },
            lead_data={},
        )
        assert result["score"] == 0
        assert result["priority"] == "Low"

    def test_medium_priority(self):
        """A mid-tier mix should land in the Medium priority band (40-69)."""
        result = compute_lead_score(
            qualification_data={
                "budget_range": "$10K-$50K",  # 18
                "timeline": "Immediate",       # 20
                "company_size": "Unknown",
                "decision_maker": False,       # 5
                "pain_points": [],
            },
            lead_data={},
        )
        assert result["score"] == 43
        assert result["priority"] == "Medium"

    def test_high_priority_boundary(self):
        """Score of exactly 70 should be High."""
        result = compute_lead_score(
            qualification_data={
                "budget_range": "$100K+",      # 25
                "timeline": "Immediate",       # 20
                "company_size": "1000+",       # 15
                "decision_maker": None,        # 0
                "pain_points": ["one", "two"], # 10
            },
            lead_data={},
        )
        assert result["score"] == 70
        assert result["priority"] == "High"

    def test_each_budget_tier(self):
        for budget, expected in [
            ("Under $10K", 10),
            ("$10K-$50K", 18),
            ("$50K-$100K", 25),
            ("$100K+", 25),
            ("Unknown", 0),
        ]:
            result = compute_lead_score(
                {"budget_range": budget, "timeline": "Unknown",
                 "company_size": "Unknown", "decision_maker": None,
                 "pain_points": []},
                {},
            )
            assert result["breakdown"]["budget"] == expected, \
                f"Budget '{budget}' should score {expected}"

    def test_each_timeline_tier(self):
        for timeline, expected in [
            ("Immediate", 20),
            ("1-3 months", 18),
            ("3-6 months", 14),
            ("6+ months", 8),
            ("Just exploring", 3),
            ("Unknown", 0),
        ]:
            result = compute_lead_score(
                {"budget_range": "Unknown", "timeline": timeline,
                 "company_size": "Unknown", "decision_maker": None,
                 "pain_points": []},
                {},
            )
            assert result["breakdown"]["timeline"] == expected

    def test_contact_completeness_tiers(self):
        # email + name + company + phone = 10
        r1 = compute_lead_score(
            {"budget_range": "Unknown", "timeline": "Unknown",
             "company_size": "Unknown", "decision_maker": None,
             "pain_points": []},
            {"email": "a@b.com", "first_name": "A", "company": "B", "phone": "555"},
        )
        assert r1["breakdown"]["contact_completeness"] == 10

        # email + name + company = 7
        r2 = compute_lead_score(
            {"budget_range": "Unknown", "timeline": "Unknown",
             "company_size": "Unknown", "decision_maker": None,
             "pain_points": []},
            {"email": "a@b.com", "first_name": "A", "company": "B"},
        )
        assert r2["breakdown"]["contact_completeness"] == 7

        # email only = 2
        r3 = compute_lead_score(
            {"budget_range": "Unknown", "timeline": "Unknown",
             "company_size": "Unknown", "decision_maker": None,
             "pain_points": []},
            {"email": "a@b.com"},
        )
        assert r3["breakdown"]["contact_completeness"] == 2

    def test_pain_point_scaling(self):
        for count, expected in [(0, 0), (1, 5), (2, 10), (3, 15), (5, 15)]:
            points = [f"point_{i}" for i in range(count)]
            result = compute_lead_score(
                {"budget_range": "Unknown", "timeline": "Unknown",
                 "company_size": "Unknown", "decision_maker": None,
                 "pain_points": points},
                {},
            )
            assert result["breakdown"]["pain_points"] == expected

    def test_rationale_contains_details(self):
        result = compute_lead_score(
            {"budget_range": "$50K-$100K", "timeline": "1-3 months",
             "company_size": "Unknown", "decision_maker": True,
             "pain_points": ["slow CRM"]},
            {"email": "a@b.com", "last_name": "Chen", "company": "Acme"},
        )
        assert "budget" in result["rationale"].lower()
        assert "timeline" in result["rationale"].lower()
        assert "decision maker" in result["rationale"].lower()


# ---------------------------------------------------------------------------
# Qualification completeness tests
# ---------------------------------------------------------------------------

class TestAssessQualificationCompleteness:
    """Tests for assess_qualification_completeness."""

    def test_fully_complete(self):
        result = assess_qualification_completeness(
            {"budget_range": "$10K-$50K", "timeline": "Immediate",
             "company_size": "51-200", "decision_maker": True,
             "pain_points": ["x"], "current_solution": "Excel"},
            {"first_name": "A", "email": "a@b.com", "company": "B", "phone": "555"},
        )
        assert result["ready"] is True
        assert result["completeness_pct"] == 100
        assert len(result["missing_fields"]) == 0

    def test_not_ready_without_email(self):
        result = assess_qualification_completeness(
            {"budget_range": "$10K-$50K", "timeline": "Immediate",
             "company_size": "Unknown", "decision_maker": None,
             "pain_points": ["x"], "current_solution": None},
            {"first_name": "A"},
        )
        assert result["ready"] is False

    def test_not_ready_with_insufficient_qual_fields(self):
        """Need at least 2 qualification fields to be ready."""
        result = assess_qualification_completeness(
            {"budget_range": "$10K-$50K", "timeline": "Unknown",
             "company_size": "Unknown", "decision_maker": None,
             "pain_points": [], "current_solution": None},
            {"first_name": "A", "email": "a@b.com", "company": "B"},
        )
        assert result["ready"] is False
        assert result["qual_fields_captured"] == 1

    def test_ready_with_minimum_viable(self):
        """name + email + company + 2 qual fields = ready."""
        result = assess_qualification_completeness(
            {"budget_range": "$10K-$50K", "timeline": "1-3 months",
             "company_size": "Unknown", "decision_maker": None,
             "pain_points": [], "current_solution": None},
            {"last_name": "Chen", "email": "a@b.com", "company": "Acme"},
        )
        assert result["ready"] is True
        assert result["qual_fields_captured"] == 2


# ---------------------------------------------------------------------------
# Summary generation tests
# ---------------------------------------------------------------------------

class TestGenerateQualificationSummary:
    """Tests for generate_qualification_summary."""

    def test_includes_name_and_company(self):
        summary = generate_qualification_summary(
            {"budget_range": "Unknown", "timeline": "Unknown",
             "company_size": "Unknown", "decision_maker": None,
             "pain_points": [], "current_solution": None, "goals": []},
            {"first_name": "Sarah", "last_name": "Chen", "company": "Acme"},
        )
        assert "Sarah Chen" in summary
        assert "Acme" in summary

    def test_includes_pain_points(self):
        summary = generate_qualification_summary(
            {"budget_range": "Unknown", "timeline": "Unknown",
             "company_size": "Unknown", "decision_maker": None,
             "pain_points": ["slow CRM", "manual reporting"],
             "current_solution": None, "goals": []},
            {"last_name": "Test"},
        )
        assert "slow CRM" in summary
        assert "manual reporting" in summary

    def test_includes_score(self):
        summary = generate_qualification_summary(
            {"budget_range": "$50K-$100K", "timeline": "1-3 months",
             "company_size": "Unknown", "decision_maker": None,
             "pain_points": ["x"], "current_solution": None, "goals": []},
            {"email": "a@b.com", "last_name": "Test", "company": "TestCo"},
        )
        assert "/100" in summary
        assert "priority" in summary.lower()


# ---------------------------------------------------------------------------
# Salesforce tool tests (mocked)
# ---------------------------------------------------------------------------

class TestSalesforceTools:
    """Tests for Salesforce create_lead and create_transcript_task with mocked API."""

    @pytest.mark.asyncio
    async def test_create_lead_success(self):
        mock_sf = MagicMock()
        mock_sf.Lead.create.return_value = {"success": True, "id": "00Q000000000001"}

        with patch("app.tools.salesforce._get_sf_client", return_value=mock_sf):
            from app.tools.salesforce import create_lead

            lead_id = await create_lead(
                lead_data={
                    "first_name": "Sarah",
                    "last_name": "Chen",
                    "email": "sarah@acme.com",
                    "company": "Acme Corp",
                },
                qualification_data={
                    "budget_range": "$50K-$100K",
                    "pain_points": ["slow CRM", "manual reporting"],
                },
                lead_score=72,
                description="Test lead",
            )

        assert lead_id == "00Q000000000001"

        # Verify the payload sent to Salesforce
        call_args = mock_sf.Lead.create.call_args[0][0]
        assert call_args["FirstName"] == "Sarah"
        assert call_args["LastName"] == "Chen"
        assert call_args["Email"] == "sarah@acme.com"
        assert call_args["LeadSource"] == "Web Chat"
        assert call_args["Lead_Score__c"] == 72
        assert call_args["Budget_Range__c"] == "$50K-$100K"
        assert "slow CRM" in call_args["Pain_Points__c"]

    @pytest.mark.asyncio
    async def test_create_lead_defaults_last_name(self):
        mock_sf = MagicMock()
        mock_sf.Lead.create.return_value = {"success": True, "id": "00Q000000000002"}

        with patch("app.tools.salesforce._get_sf_client", return_value=mock_sf):
            from app.tools.salesforce import create_lead

            await create_lead(
                lead_data={"email": "test@test.com"},
                qualification_data={},
            )

        call_args = mock_sf.Lead.create.call_args[0][0]
        assert call_args["LastName"] == "Unknown"
        assert call_args["Company"] == "Unknown"

    @pytest.mark.asyncio
    async def test_create_lead_failure_raises(self):
        mock_sf = MagicMock()
        mock_sf.Lead.create.return_value = {
            "success": False,
            "errors": [{"message": "Required field missing"}],
        }

        with patch("app.tools.salesforce._get_sf_client", return_value=mock_sf):
            from app.tools.salesforce import create_lead

            with pytest.raises(RuntimeError, match="Salesforce Lead creation failed"):
                await create_lead(
                    lead_data={},
                    qualification_data={},
                )

    @pytest.mark.asyncio
    async def test_create_transcript_task_success(self):
        mock_sf = MagicMock()
        mock_sf.Task.create.return_value = {"success": True, "id": "00T000000000001"}
        mock_sf.Lead.update.return_value = None

        with patch("app.tools.salesforce._get_sf_client", return_value=mock_sf):
            from app.tools.salesforce import create_transcript_task

            task_id = await create_transcript_task(
                lead_id="00Q000000000001",
                transcript="Visitor: Hi\nAlex: Welcome!",
            )

        assert task_id == "00T000000000001"

        call_args = mock_sf.Task.create.call_args[0][0]
        assert call_args["WhoId"] == "00Q000000000001"
        assert "AI Chat Transcript" in call_args["Subject"]
        assert call_args["Status"] == "Completed"

        # Verify lead was updated with transcript ID
        mock_sf.Lead.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_task_failure_raises(self):
        mock_sf = MagicMock()
        mock_sf.Task.create.return_value = {
            "success": False,
            "errors": [{"message": "Error"}],
        }

        with patch("app.tools.salesforce._get_sf_client", return_value=mock_sf):
            from app.tools.salesforce import create_transcript_task

            with pytest.raises(RuntimeError, match="Task creation failed"):
                await create_transcript_task(
                    lead_id="00Q000000000001",
                    transcript="test",
                )
