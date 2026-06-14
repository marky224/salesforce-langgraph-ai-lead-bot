"""
Offline unit tests for the simulated-user scorecard evaluators (``scorecard.py``).

Pure and synthetic — no network, no LLM, no graph, no Salesforce — so the scorecard
*logic* rides the existing ``eval_recorded`` CI gate even though the live simulation
(``test_simulation_live.py``, PR C2) is report-only ``eval_live``. Mirrors how
``test_extraction_recorded.py`` bundles the offline ``_safe_parse_json`` check.
"""

from __future__ import annotations

import pytest

from app.models.schemas import ConversationStage
from app.tools.qualification import compute_lead_score
from evals.scorecard import (
    Trajectory,
    brevity_ok,
    effective_score,
    markdown_leak,
    must_capture_check,
    no_stage_loop,
    reached_complete,
    reask_check,
    score_in_band,
    score_persona,
)

pytestmark = pytest.mark.eval_recorded


# --- reached_complete: str-enum compares equal as enum *and* as "complete" string ---

def test_reached_complete_enum_and_string():
    assert reached_complete({"stage": ConversationStage.COMPLETE})
    assert reached_complete({"stage": "complete"})            # post-Postgres-serde form
    assert not reached_complete({"stage": ConversationStage.QUALIFICATION})
    assert not reached_complete({})


# --- effective_score: trust state when COMPLETE, recompute otherwise ---

def test_effective_score_complete_trusts_state():
    fs = {"stage": ConversationStage.COMPLETE, "lead_score": 95,
          "qualification_data": {}, "lead_data": {}}
    assert effective_score(fs) == 95


def test_effective_score_incomplete_recomputes_and_locks_enterprise_band():
    qual = {"budget_range": "$100K+", "timeline": "1-3 months", "company_size": "1000+",
            "decision_maker": True, "pain_points": ["p1", "p2", "p3"]}
    lead = {"first_name": "Dana", "last_name": "Whitfield",
            "email": "dana.whitfield@northwind.example", "company": "Northwind Logistics"}
    fs = {"stage": ConversationStage.QUALIFICATION, "qualification_data": qual, "lead_data": lead}
    assert effective_score(fs) == compute_lead_score(qual, lead)["score"] == 95  # 25+18+15+15+15+7


def test_score_in_band():
    fs = {"stage": "complete", "lead_score": 66, "qualification_data": {}, "lead_data": {}}
    assert score_in_band(fs, [40, 69])
    assert not score_in_band(fs, [70, 100])


# --- must_capture: presence for lead_data, "not the Unknown/None default" for quals ---

def test_must_capture_check():
    fs = {
        "lead_data": {"email": "a@b.example", "company": "Acme"},
        "qualification_data": {"budget_range": "$10K-$50K", "timeline": "3-6 months",
                               "company_size": "Unknown", "decision_maker": None,
                               "pain_points": []},
    }
    ok = must_capture_check(fs, {"lead_data": ["email", "company"],
                                 "qualification_data": ["budget_range", "timeline"]})
    assert ok["ok"] and ok["missing"] == []

    bad = must_capture_check(fs, {"qualification_data": ["company_size", "decision_maker", "pain_points"]})
    assert not bad["ok"]
    assert set(bad["missing"]) == {
        "qualification_data.company_size",
        "qualification_data.decision_maker",
        "qualification_data.pain_points",
    }

    assert must_capture_check(fs, {})["ok"]  # evasive persona — nothing required


# --- no_stage_loop: consecutive-run threshold ---

def test_no_stage_loop():
    # The bot asks one field per turn, so a thorough qualification / lead_capture
    # stage legitimately repeats ~4-5 times; only a genuinely stuck run (6+) loops.
    healthy = Trajectory(stages=["greeting", "discovery"] + ["qualification"] * 5
                         + ["lead_capture", "confirmation", "complete"])
    assert no_stage_loop(healthy)                        # qualification x5 == max_repeat -> ok
    stuck = Trajectory(stages=["greeting", "discovery"] + ["qualification"] * 6)
    assert not no_stage_loop(stuck)                      # x6 consecutive -> stuck loop
    assert no_stage_loop(Trajectory(stages=[]))          # empty -> vacuously ok


# --- voice guardrails ---

def test_markdown_leak():
    assert not markdown_leak("Manual data entry — the closest most of us get to a digital detox.")
    assert markdown_leak("Here's the **plan**.")
    assert markdown_leak("Run `npm install` first.")
    assert markdown_leak("See [our docs](https://x.example).")
    assert markdown_leak("# Summary\nYou're set.")
    assert not markdown_leak("Issue #42 is the budget one, no rush.")  # bare # mid-line, not a header


def test_brevity_ok():
    assert brevity_ok("Short and dry. Two sentences.")
    assert not brevity_ok(" ".join(f"Sentence {i}." for i in range(8)))
    assert brevity_ok("I'll reach out at sarah@acme.example within a day.")  # email period must not split


# --- score_persona: integration over the dimensions ---

def _enterprise_complete_state():
    return {
        "stage": ConversationStage.COMPLETE,
        "lead_score": 95,
        "lead_data": {"first_name": "Dana", "last_name": "Whitfield",
                      "email": "dana.whitfield@northwind.example", "company": "Northwind Logistics"},
        "qualification_data": {"budget_range": "$100K+", "timeline": "1-3 months",
                               "company_size": "1000+", "decision_maker": True,
                               "pain_points": ["p1", "p2", "p3"]},
    }


def test_score_persona_pass():
    persona = {"id": "enterprise_dm_urgent", "expected": {
        "reaches_complete": True, "score_band": [70, 100],
        "must_capture": {"lead_data": ["email", "company"],
                         "qualification_data": ["budget_range", "timeline", "decision_maker"]},
        "max_turns": 12}}
    traj = Trajectory(
        turns=[("TARS", "Hi, I'm TARS. What brings you by?"),
               ("Visitor", "Manual reconciliation is eating us alive."),
               ("TARS", "How many hours a week is that costing the team?"),
               ("Visitor", "About thirty.")],
        stages=["greeting", "discovery", "qualification", "lead_capture", "confirmation", "complete"],
        final_state=_enterprise_complete_state(),
    )
    row = score_persona(traj, persona)
    assert row["overall_pass"]
    assert row["reached_complete"]["pass"] is True
    assert row["score"]["pass"] and row["must_capture"]["ok"]
    assert row["no_stage_loop"] and row["voice"]["markdown_clean"]
    assert row["turns"]["to_complete"] == 5


def test_score_persona_evasive_null_completion():
    persona = {"id": "evasive_tirekicker", "expected": {
        "reaches_complete": None, "score_band": [0, 39], "must_capture": {}, "max_turns": 12}}
    traj = Trajectory(
        turns=[("TARS", "What brings you by?"), ("Visitor", "Just looking, honestly.")],
        stages=["greeting"] + ["discovery"] * 12,    # stonewalled -> stuck in discovery by design
        final_state={"stage": ConversationStage.DISCOVERY,
                     "qualification_data": {"timeline": "Just exploring"}, "lead_data": {}},
    )
    row = score_persona(traj, persona)
    assert row["reached_complete"]["pass"] is None   # reported, not asserted
    assert row["score"]["value"] <= 39
    assert row["no_stage_loop"] is False             # the stall is real and reported...
    assert row["overall_pass"]                        # ...but loop is advisory for a non-converter


def test_score_persona_fails_when_expected_complete_but_not():
    persona = {"id": "enterprise_dm_urgent", "expected": {
        "reaches_complete": True, "score_band": [70, 100],
        "must_capture": {"lead_data": ["email"]}, "max_turns": 12}}
    traj = Trajectory(
        turns=[("Visitor", "hi")],
        stages=["greeting", "qualification"],
        final_state={"stage": ConversationStage.QUALIFICATION, "qualification_data": {}, "lead_data": {}},
    )
    row = score_persona(traj, persona)
    assert not row["overall_pass"]
    assert row["reached_complete"]["pass"] is False


def test_score_persona_converger_loop_fails():
    # A persona expected to converge IS still loop-checked: a genuinely stuck run
    # (6+) fails overall_pass even when completion, score, and capture are fine.
    persona = {"id": "enterprise_dm_urgent", "expected": {
        "reaches_complete": True, "score_band": [70, 100],
        "must_capture": {}, "max_turns": 12}}
    traj = Trajectory(
        turns=[("Visitor", f"answer {i}") for i in range(6)],
        stages=["greeting"] + ["qualification"] * 6,   # stuck re-asking -> real loop
        final_state=_enterprise_complete_state(),
    )
    row = score_persona(traj, persona)
    assert row["no_stage_loop"] is False
    assert not row["overall_pass"]                       # loop folded in for a converger


# --- reask_check: flag asking for an already-captured field (advisory, heuristic) ---

def test_reask_check_flags_already_captured_email():
    # Visitor gives email in turn 1; TARS asks for it again in the reply to that turn.
    # That TARS reply pairs with snapshots[1] (post-extraction state), where email is set.
    traj = Trajectory(
        turns=[
            ("TARS", "Hi, I'm TARS. What brings you by?"),
            ("Visitor", "Manual entry is eating us alive — I'm at sarah@acme.example."),
            ("TARS", "Got it. What's your email so I can follow up?"),
        ],
        stages=["greeting", "lead_capture"],
        snapshots=[
            {"lead_data": {}, "qualification_data": {}},
            {"lead_data": {"email": "sarah@acme.example"}, "qualification_data": {}},
        ],
    )
    result = reask_check(traj)
    assert not result["ok"]
    assert result["reasks"] == [{"turn": 1, "field": "lead_data.email"}]


def test_reask_check_ok_when_asking_for_uncaptured_field():
    # Email not yet known when TARS asks for it -> legitimate, not a re-ask.
    traj = Trajectory(
        turns=[
            ("TARS", "What brings you by?"),
            ("Visitor", "Reporting runs two days behind."),
            ("TARS", "That's rough. What's your email so I can send a summary?"),
        ],
        stages=["greeting", "lead_capture"],
        snapshots=[
            {"lead_data": {}, "qualification_data": {}},
            {"lead_data": {}, "qualification_data": {}},
        ],
    )
    assert reask_check(traj)["ok"]


def test_reask_check_ignores_confirmation_recap():
    # The confirmation summary restates a captured email but isn't *asking* — skipped.
    traj = Trajectory(
        turns=[
            ("TARS", "What brings you by?"),
            ("Visitor", "Here's my email: sarah@acme.example."),
            ("TARS", "So I've got your email as sarah@acme.example and we're set — confirm?"),
        ],
        stages=["greeting", "confirmation"],
        snapshots=[
            {"lead_data": {}, "qualification_data": {}},
            {"lead_data": {"email": "sarah@acme.example"}, "qualification_data": {}},
        ],
    )
    assert reask_check(traj)["ok"]


def test_reask_check_company_name_excludes_company_size():
    # "company size?" asks a different field — must not trip the company-name re-ask.
    traj = Trajectory(
        turns=[
            ("TARS", "What brings you by?"),
            ("Visitor", "I'm at Northwind Logistics."),
            ("TARS", "Thanks. Roughly what's your company size?"),
        ],
        stages=["greeting", "qualification"],
        snapshots=[
            {"lead_data": {}, "qualification_data": {}},
            {"lead_data": {"company": "Northwind Logistics"}, "qualification_data": {}},
        ],
    )
    assert reask_check(traj)["ok"]


def test_reask_check_empty_snapshots_trivially_ok():
    # Hand-built trajectories (no snapshots) never raise and never flag.
    traj = Trajectory(
        turns=[("TARS", "What's your email?"), ("Visitor", "later")],
        stages=["greeting", "lead_capture"],
    )
    assert reask_check(traj)["ok"]
