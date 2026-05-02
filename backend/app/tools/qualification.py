"""
Lead scoring and qualification tools for the AI Sales Lead Bot.

Provides the canonical deterministic scoring engine used by the graph's
scoring node. The rubric awards up to 100 points across six dimensions:
budget (25), timeline (20), company size (15), decision-maker (15),
pain points (15), and contact completeness (10).

Usage::

    from app.tools.qualification import compute_lead_score, generate_summary

    result = compute_lead_score(
        qualification_data={"budget_range": "$50K-$100K", "timeline": "1-3 months", ...},
        lead_data={"email": "s@acme.com", "last_name": "Chen", "company": "Acme"},
    )
    print(result)  # {"score": 72, "breakdown": {...}, "priority": "High", "rationale": "..."}
"""

from __future__ import annotations

import logging
from typing import Any

from app.models.schemas import (
    BudgetRange,
    CompanySize,
    LeadPriority,
    LeadScore,
    Timeline,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scoring tables — canonical rubric. High tiers saturate (mid-market scores
# the same as enterprise) which fits the consulting-lead use case.
# ---------------------------------------------------------------------------

_BUDGET_SCORES: dict[str, int] = {
    BudgetRange.OVER_100K.value: 25,
    BudgetRange.FIFTY_TO_HUNDRED_K.value: 25,
    BudgetRange.TEN_TO_FIFTY_K.value: 18,
    BudgetRange.UNDER_10K.value: 10,
    BudgetRange.UNKNOWN.value: 0,
}

_TIMELINE_SCORES: dict[str, int] = {
    Timeline.IMMEDIATE.value: 20,
    Timeline.ONE_TO_THREE_MONTHS.value: 18,
    Timeline.THREE_TO_SIX_MONTHS.value: 14,
    Timeline.SIX_PLUS_MONTHS.value: 8,
    Timeline.JUST_EXPLORING.value: 3,
    Timeline.UNKNOWN.value: 0,
}

_COMPANY_SIZE_SCORES: dict[str, int] = {
    CompanySize.ENTERPRISE.value: 15,
    CompanySize.LARGE.value: 15,
    CompanySize.MEDIUM.value: 12,
    CompanySize.SMALL.value: 8,
    CompanySize.MICRO.value: 4,
    CompanySize.UNKNOWN.value: 0,
}


# ---------------------------------------------------------------------------
# Deterministic scorer
# ---------------------------------------------------------------------------

def compute_lead_score(
    qualification_data: dict[str, Any],
    lead_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute a lead quality score using the deterministic rubric.

    Parameters
    ----------
    qualification_data : dict
        Qualification signals with keys: budget_range, timeline,
        company_size, decision_maker, pain_points, current_solution, goals.
    lead_data : dict
        Contact info with keys: first_name, last_name, email, company,
        phone, title.

    Returns
    -------
    dict
        Contains: score (int 0-100), breakdown (dict[str, int]),
        priority (str), rationale (str).
    """
    breakdown: dict[str, int] = {}
    rationale_parts: list[str] = []

    # --- 1. Budget (0-25) ---
    budget = qualification_data.get("budget_range", "Unknown")
    budget_pts = _BUDGET_SCORES.get(budget, 0)
    breakdown["budget"] = budget_pts
    if budget_pts > 0:
        rationale_parts.append(f"budget {budget}")

    # --- 2. Timeline (0-20) ---
    timeline = qualification_data.get("timeline", "Unknown")
    timeline_pts = _TIMELINE_SCORES.get(timeline, 0)
    breakdown["timeline"] = timeline_pts
    if timeline_pts > 0:
        rationale_parts.append(f"timeline {timeline}")

    # --- 3. Company size (0-15) ---
    company_size = qualification_data.get("company_size", "Unknown")
    size_pts = _COMPANY_SIZE_SCORES.get(company_size, 0)
    breakdown["company_size"] = size_pts
    if size_pts > 0:
        rationale_parts.append(f"company size {company_size}")

    # --- 4. Decision maker (0-15) ---
    dm = qualification_data.get("decision_maker")
    if dm is True:
        dm_pts = 15
        rationale_parts.append("is decision maker")
    elif dm is False:
        dm_pts = 5
        rationale_parts.append("not decision maker")
    else:
        dm_pts = 0
    breakdown["decision_maker"] = dm_pts

    # --- 5. Pain points (0-15) ---
    pain_points = qualification_data.get("pain_points", [])
    if not isinstance(pain_points, list):
        pain_points = []
    pain_count = len(pain_points)

    if pain_count >= 3:
        pain_pts = 15
    elif pain_count == 2:
        pain_pts = 10
    elif pain_count == 1:
        pain_pts = 5
    else:
        pain_pts = 0
    breakdown["pain_points"] = pain_pts
    if pain_pts > 0:
        rationale_parts.append(f"{pain_count} pain point(s)")

    # --- 6. Contact completeness (0-10) ---
    has_email = bool(lead_data.get("email"))
    has_name = bool(lead_data.get("first_name") or lead_data.get("last_name"))
    has_company = bool(lead_data.get("company"))
    has_phone = bool(lead_data.get("phone"))

    if has_email and has_name and has_company and has_phone:
        contact_pts = 10
    elif has_email and has_name and has_company:
        contact_pts = 7
    elif has_email and has_name:
        contact_pts = 4
    elif has_email:
        contact_pts = 2
    else:
        contact_pts = 0
    breakdown["contact_completeness"] = contact_pts

    # --- Total ---
    score = sum(breakdown.values())
    score = max(0, min(100, score))

    # --- Priority ---
    if score >= 70:
        priority = LeadPriority.HIGH
    elif score >= 40:
        priority = LeadPriority.MEDIUM
    else:
        priority = LeadPriority.LOW

    # --- Rationale ---
    if rationale_parts:
        rationale = (
            f"Score {score}/100 ({priority.value} priority): "
            + ", ".join(rationale_parts)
            + f". Contact completeness: {contact_pts}/10."
        )
    else:
        rationale = f"Score {score}/100 ({priority.value} priority): limited data collected."

    logger.info("Lead scored: %d (%s)", score, priority.value)
    logger.debug("Score breakdown: %s", breakdown)

    return {
        "score": score,
        "breakdown": breakdown,
        "priority": priority.value,
        "rationale": rationale,
    }


# ---------------------------------------------------------------------------
# Qualification completeness assessment
# ---------------------------------------------------------------------------

def assess_qualification_completeness(
    qualification_data: dict[str, Any],
    lead_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Assess how complete the qualification and contact data is.

    Returns a summary of captured vs missing fields with a readiness
    flag indicating whether the lead is ready for Salesforce submission.

    Parameters
    ----------
    qualification_data : dict
        Current qualification data.
    lead_data : dict
        Current lead contact data.

    Returns
    -------
    dict
        Contains: ready (bool), captured_fields (list[str]),
        missing_fields (list[str]), completeness_pct (int).
    """
    captured: list[str] = []
    missing: list[str] = []

    # Qualification fields
    qd = qualification_data

    if qd.get("budget_range") and qd["budget_range"] != "Unknown":
        captured.append("budget_range")
    else:
        missing.append("budget_range")

    if qd.get("timeline") and qd["timeline"] != "Unknown":
        captured.append("timeline")
    else:
        missing.append("timeline")

    if qd.get("company_size") and qd["company_size"] != "Unknown":
        captured.append("company_size")
    else:
        missing.append("company_size")

    if qd.get("decision_maker") is not None:
        captured.append("decision_maker")
    else:
        missing.append("decision_maker")

    if qd.get("pain_points"):
        captured.append("pain_points")
    else:
        missing.append("pain_points")

    if qd.get("current_solution"):
        captured.append("current_solution")
    else:
        missing.append("current_solution")

    # Contact fields
    ld = lead_data

    if ld.get("first_name") or ld.get("last_name"):
        captured.append("name")
    else:
        missing.append("name")

    if ld.get("email"):
        captured.append("email")
    else:
        missing.append("email")

    if ld.get("company"):
        captured.append("company")
    else:
        missing.append("company")

    if ld.get("phone"):
        captured.append("phone")
    else:
        missing.append("phone")

    total = len(captured) + len(missing)
    pct = int((len(captured) / total) * 100) if total > 0 else 0

    # Ready = minimum viable lead: name + email + company + at least 2 qual fields
    has_min_contact = all(
        f in captured for f in ("name", "email", "company")
    )
    qual_field_count = sum(
        1 for f in captured
        if f in ("budget_range", "timeline", "company_size", "decision_maker", "pain_points")
    )
    ready = has_min_contact and qual_field_count >= 2

    return {
        "ready": ready,
        "captured_fields": captured,
        "missing_fields": missing,
        "completeness_pct": pct,
        "qual_fields_captured": qual_field_count,
    }


# ---------------------------------------------------------------------------
# Summary generation (deterministic, no LLM)
# ---------------------------------------------------------------------------

def generate_qualification_summary(
    qualification_data: dict[str, Any],
    lead_data: dict[str, Any],
    score_result: dict[str, Any] | None = None,
) -> str:
    """
    Generate a plain-text summary of the lead qualification for the
    Salesforce Lead Description field.

    This is a deterministic fallback if the LLM-generated transcript
    summary is unavailable.

    Parameters
    ----------
    qualification_data : dict
        Qualification signals.
    lead_data : dict
        Contact info.
    score_result : dict | None
        Output from ``compute_lead_score``.  If not provided, score
        is computed on the fly.

    Returns
    -------
    str
        A concise paragraph suitable for the Description field.
    """
    if score_result is None:
        score_result = compute_lead_score(qualification_data, lead_data)

    parts: list[str] = []

    # Contact context
    name = " ".join(
        p for p in (lead_data.get("first_name"), lead_data.get("last_name")) if p
    )
    company = lead_data.get("company", "")
    if name and company:
        parts.append(f"Lead: {name} at {company}.")
    elif name:
        parts.append(f"Lead: {name}.")

    # Qualification details
    qd = qualification_data

    if qd.get("pain_points"):
        points = ", ".join(qd["pain_points"])
        parts.append(f"Key challenges: {points}.")

    if qd.get("current_solution"):
        parts.append(f"Currently using: {qd['current_solution']}.")

    if qd.get("goals"):
        goals = ", ".join(qd["goals"])
        parts.append(f"Goals: {goals}.")

    details: list[str] = []
    if qd.get("budget_range") and qd["budget_range"] != "Unknown":
        details.append(f"budget {qd['budget_range']}")
    if qd.get("timeline") and qd["timeline"] != "Unknown":
        details.append(f"timeline {qd['timeline']}")
    if qd.get("company_size") and qd["company_size"] != "Unknown":
        details.append(f"team size {qd['company_size']}")
    if qd.get("decision_maker") is True:
        details.append("decision maker")
    elif qd.get("decision_maker") is False:
        details.append("not the decision maker")

    if details:
        parts.append(f"Qualification: {', '.join(details)}.")

    # Score
    parts.append(
        f"Lead score: {score_result['score']}/100 ({score_result['priority']} priority)."
    )

    return " ".join(parts)
