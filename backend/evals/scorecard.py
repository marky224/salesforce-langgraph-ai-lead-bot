"""
Whole-conversation (simulated-user) scorecard evaluators — pure, no I/O.

PR B's offline gate (``evaluators.py``) scores extraction/routing **per call, in
isolation**. It cannot see whether a *whole conversation* reaches COMPLETE, lands
in the right deterministic score band, captures the required fields, and doesn't
loop — that is what these evaluators measure, over a ``Trajectory`` produced by the
simulated-user loop (``simulate.py``, PR C2) driving the real compiled graph.

Everything here is pure (no network, no LLM, no graph, no Salesforce), so the
*logic* is unit-tested offline under the existing ``eval_recorded`` gate
(``test_scorecard.py``) even though the live simulation itself is report-only
(nondeterministic → ``eval_live``, never per-commit).

Design notes:
- ``Trajectory`` lives here, with its consumer, not in ``simulate.py`` — so the
  scorecard and its tests are self-contained and never import the live loop.
- Bands mirror the deterministic rubric in
  ``app.tools.qualification.compute_lead_score`` (HIGH >=70, MEDIUM 40-69, LOW <40)
  — contract #1: the same ``.value`` strings are scorer keys + SF picklist values.
- ``effective_score`` recomputes the deterministic score from captured data when a
  conversation did not COMPLETE, so an incomplete run still gets an honest band.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.models.schemas import ConversationStage
from app.tools.qualification import compute_lead_score

# ---------------------------------------------------------------------------
# Trajectory — the output of one simulated conversation (produced by simulate.py)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Trajectory:
    """
    One full simulated conversation.

    turns : ordered ``(speaker, text)`` pairs; speaker is ``"TARS"`` / ``"Visitor"``.
    stages : ``ConversationStage`` ``.value`` after each ``graph.ainvoke`` (i.e. after
        each TARS turn), oldest first. Read by ``no_stage_loop`` / ``_turns_to_complete``.
    final_state : the last ``graph.ainvoke`` result — the final graph state. Every
        final-state evaluator reads from this.
    """

    turns: list[tuple[str, str]] = field(default_factory=list)
    stages: list[str] = field(default_factory=list)
    final_state: dict[str, Any] = field(default_factory=dict)

    @property
    def tars_turns(self) -> list[str]:
        """Just TARS's messages, in order — what the voice guardrails read."""
        return [text for who, text in self.turns if who == "TARS"]

    @property
    def num_visitor_turns(self) -> int:
        """How many messages the simulated visitor sent (a turns-to-complete proxy)."""
        return sum(1 for who, _ in self.turns if who == "Visitor")


# ---------------------------------------------------------------------------
# Final-state outcome evaluators (deterministic)
# ---------------------------------------------------------------------------

def reached_complete(final_state: dict[str, Any]) -> bool:
    """
    True when the conversation reached the terminal COMPLETE stage.

    ``ConversationStage`` is a ``str`` enum, so this compares equal whether ``stage``
    is the enum member or its ``"complete"`` ``.value`` string (the Postgres-serde-safe
    invariant, CLAUDE.md contract #4).
    """
    return final_state.get("stage") == ConversationStage.COMPLETE


# Per-field "never captured" sentinels — mirrors evaluators._QUAL_SCALAR_DEFAULTS
# and qualification.assess_qualification_completeness.
_QUAL_UNCOLLECTED: dict[str, Any] = {
    "budget_range": "Unknown",
    "timeline": "Unknown",
    "company_size": "Unknown",
    "decision_maker": None,
    "current_solution": None,
}


def effective_score(final_state: dict[str, Any]) -> int:
    """
    The deterministic lead score for this conversation.

    When the run COMPLETEd, the score the graph actually computed (and sent to
    Salesforce) sits in ``final_state["lead_score"]`` — use it. When it did **not**
    complete (e.g. an evasive visitor who bailed), recompute from whatever was
    captured so the band check is still honest. Both paths are the same deterministic
    ``compute_lead_score`` rubric.
    """
    score = final_state.get("lead_score")
    if reached_complete(final_state) and isinstance(score, int):
        return score
    return compute_lead_score(
        qualification_data=final_state.get("qualification_data") or {},
        lead_data=final_state.get("lead_data") or {},
    )["score"]


def score_in_band(final_state: dict[str, Any], band: tuple[int, int] | list[int]) -> bool:
    """True when ``effective_score`` falls within the inclusive ``[min, max]`` band."""
    low, high = band
    return low <= effective_score(final_state) <= high


def _captured(section: str, name: str, data: dict[str, Any]) -> bool:
    """Was *name* actually captured (not left at its not-collected default)?"""
    value = data.get(name)
    if section == "qualification_data":
        if name in ("pain_points", "goals"):
            return bool(value)
        return value is not None and value != _QUAL_UNCOLLECTED.get(name)
    return bool(value)  # lead_data: any non-empty value counts


def must_capture_check(final_state: dict[str, Any], spec: dict[str, list[str]]) -> dict[str, Any]:
    """
    Check that every required field in *spec* was captured.

    ``spec`` maps a state section (``"lead_data"`` / ``"qualification_data"``) to the
    field names that MUST be present for this persona's lead to be usable. Returns
    ``{"ok": bool, "missing": ["section.field", ...]}``. An empty spec is trivially ok.
    """
    missing: list[str] = []
    for section, names in (spec or {}).items():
        data = final_state.get(section) or {}
        missing += [f"{section}.{name}" for name in names if not _captured(section, name, data)]
    return {"ok": not missing, "missing": missing}


# ---------------------------------------------------------------------------
# Trajectory-level evaluators
# ---------------------------------------------------------------------------

def no_stage_loop(trajectory: Trajectory, max_repeat: int = 3) -> bool:
    """
    True when no stage repeats *consecutively* more than ``max_repeat`` times.

    A coarse "the bot keeps making progress" check. A healthy run walks
    greeting -> discovery -> qualification (<=3, one per missing field) -> lead_capture
    (<=3) -> confirmation -> complete; a run stuck re-asking in one phase shows that
    stage repeating turn after turn. ``max_repeat=3`` tolerates a legitimately
    multi-field qualification / lead_capture stretch but flags 4+ identical
    consecutive stages. (``max_turns`` is the backstop for runaway *length*; this
    catches the stuck-in-place mode.)
    """
    longest = run = 1
    prev: str | None = None
    for stage in trajectory.stages:
        run = run + 1 if stage == prev else 1
        longest = max(longest, run)
        prev = stage
    return longest <= max_repeat


def _turns_to_complete(trajectory: Trajectory) -> int | None:
    """Visitor turns until COMPLETE first appears in the stage sequence, else ``None``."""
    target = ConversationStage.COMPLETE.value
    for i, stage in enumerate(trajectory.stages):
        if stage == target:
            return i  # stages[0] is the greeting -> 0 visitor turns
    return None


# ---------------------------------------------------------------------------
# Voice guardrails (deterministic slice — subjective tone/wit is a PR D judge)
# ---------------------------------------------------------------------------

# The widget renders TARS replies as raw text (prompts.py PERSONA: "Do NOT use
# markdown ... plain sentences only"), so any markdown is a real bug. Match the
# documented set: **bold**, `code`, [label](url) links, ATX (#) headers at line
# start. Em-dashes — which TARS leans on heavily — are not markdown and not flagged.
_MD_PATTERNS = (
    re.compile(r"\*\*"),                  # **bold**
    re.compile(r"`"),                     # `inline` / ``` fenced
    re.compile(r"\[[^\]]+\]\([^)]+\)"),   # [text](url)
    re.compile(r"(?m)^\s{0,3}#{1,6}\s"),  # # header at line start
)


def markdown_leak(text: str) -> bool:
    """True if *text* contains markdown formatting (a leak into the raw-text widget)."""
    return any(pattern.search(text) for pattern in _MD_PATTERNS)


def _sentence_count(text: str) -> int:
    """
    Approximate sentence count: split on terminal ``. ! ?`` followed by whitespace/end.

    The whitespace-or-end boundary keeps ``sarah@acme.example`` and decimals from
    inflating the count. Heuristic by nature (report-only) — good enough to flag a
    wall-of-text against the 2-4 sentence persona.
    """
    parts = [p for p in re.split(r"[.!?]+(?:\s|$)", text.strip()) if p.strip()]
    return len(parts)


def brevity_ok(text: str, max_sentences: int = 5) -> bool:
    """
    True when *text* stays within ``max_sentences``.

    The persona is 2-4 sentences; the confirmation turn is allowed 3-5, so the
    ceiling is 5. Fuzzy by design (see ``_sentence_count``) — a scorecard signal,
    not a gate.
    """
    return _sentence_count(text) <= max_sentences


# ---------------------------------------------------------------------------
# Per-persona scorecard
# ---------------------------------------------------------------------------

def score_persona(trajectory: Trajectory, persona: dict[str, Any]) -> dict[str, Any]:
    """
    Combine every dimension into one per-persona scorecard row.

    ``persona["expected"]`` carries the assertions: ``reaches_complete``
    (true/false/null — null = report, don't assert), ``score_band`` ``[min, max]``,
    ``must_capture`` ``{section: [field, ...]}``, ``max_turns``.

    ``overall_pass`` ANDs the *outcome* dimensions (completion when asserted, score
    band, must-capture, no stage loop, within turn cap). The voice guardrails are
    reported as advisory quality signals (deterministic but fuzzy) and are **not**
    folded into ``overall_pass``; the whole scorecard is report-only.
    """
    expected = persona.get("expected") or {}
    final_state = trajectory.final_state

    # completion — None when the persona deliberately doesn't assert it
    expected_complete = expected.get("reaches_complete")
    complete = reached_complete(final_state)
    complete_pass = None if expected_complete is None else (complete == expected_complete)

    # score band
    low, high = expected.get("score_band", [0, 100])
    score_value = effective_score(final_state)
    in_band = low <= score_value <= high

    capture = must_capture_check(final_state, expected.get("must_capture") or {})
    loop_ok = no_stage_loop(trajectory)

    max_turns = expected.get("max_turns")
    turns_taken = trajectory.num_visitor_turns
    within_cap = max_turns is None or turns_taken <= max_turns

    # voice — advisory, not part of overall_pass
    sentence_counts = [_sentence_count(t) for t in trajectory.tars_turns]
    md_offenders = [i for i, t in enumerate(trajectory.tars_turns) if markdown_leak(t)]
    brevity_offenders = [i for i, c in enumerate(sentence_counts) if c > 5]

    outcome_checks = [in_band, capture["ok"], loop_ok, within_cap]
    if complete_pass is not None:
        outcome_checks.append(complete_pass)

    return {
        "id": persona.get("id"),
        "overall_pass": all(outcome_checks),
        "reached_complete": {"value": complete, "expected": expected_complete, "pass": complete_pass},
        "score": {"value": score_value, "band": [low, high], "pass": in_band},
        "must_capture": capture,
        "turns": {
            "taken": turns_taken,
            "to_complete": _turns_to_complete(trajectory),
            "max": max_turns,
            "within_cap": within_cap,
        },
        "no_stage_loop": loop_ok,
        "voice": {
            "markdown_clean": not md_offenders,
            "markdown_offender_turns": md_offenders,
            "brevity_ok": not brevity_offenders,
            "max_sentences_seen": max(sentence_counts, default=0),
        },
    }
