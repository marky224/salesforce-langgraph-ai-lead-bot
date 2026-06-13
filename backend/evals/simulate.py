"""
Simulated-user loop (PR C2) — drive the *real* compiled graph against a persona.

A second LLM role-plays the website visitor (decision #4: reuse xAI grok-4.3) while
the graph runs the real grok-4.3 system-under-test, with Salesforce mocked at the
``app.tools.salesforce._get_sf_client`` seam (exactly as ``tests/test_e2e.py`` does)
so **zero** real CRM writes happen. The loop stops on the real terminal signal —
graph state ``stage == COMPLETE`` — or the persona's ``max_turns`` cap.

This is the *producer* of the ``Trajectory`` that ``scorecard.py`` consumes. It is
nondeterministic (two live models talking), so it is exercised only by the
report-only ``test_simulation_live.py`` (``eval_live``) and the ``run.py --dimension
sim --mode live`` CLI — never the offline gate. Roll-our-own (decision #1): ~40 lines,
zero new deps, and it stops on graph *state*, which a message-stream library can't
naturally see.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.models.schemas import ConversationStage
from evals.scorecard import Trajectory

# Generic rules for the simulated visitor; the persona's own profile is appended.
SIM_USER_PREAMBLE = (
    "You are role-playing a website visitor in a live chat with TARS, an AI sales "
    "advisor on a tech-consulting site. Stay fully in character as the person "
    "described below.\n"
    "Rules:\n"
    "- Send ONE short message at a time, like a real person typing in a chat — a "
    "sentence or two at most.\n"
    "- Answer only what TARS actually asked; don't dump your whole profile at once.\n"
    "- Invent nothing beyond your profile. If TARS asks something it doesn't cover, "
    "give a brief, natural non-answer.\n"
    "- When TARS summarizes everything and asks you to confirm, confirm briefly "
    "unless your character is meant to bail.\n"
    "- Never break character, never say you are an AI or a simulation, and do not "
    "use markdown.\n\n"
    "Your character:"
)


@contextmanager
def _patched_salesforce():
    """
    Patch the Salesforce client seam for the duration of one simulation.

    Mirrors ``tests/test_e2e.py::mock_salesforce``: dedup query misses (→ create),
    Lead/Task create return fake ids, update is a no-op. ``simulate`` always runs
    inside this, so the sim can never touch a real org (constraint #3).
    """
    with patch("app.tools.salesforce._get_sf_client") as mock_client:
        sf = MagicMock()
        sf.query.return_value = {"totalSize": 0, "records": []}
        sf.Lead.create.return_value = {"success": True, "id": "00Q000SIM000001"}
        sf.Task.create.return_value = {"success": True, "id": "00T000SIM000001"}
        sf.Lead.update.return_value = None
        mock_client.return_value = sf
        yield sf


def _latest_ai(state: dict[str, Any]) -> str:
    """Content of the most recent AIMessage in the graph state, or ``""``."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage):
            return msg.content
    return ""


def _stage_of(state: dict[str, Any]) -> str:
    """The state's ``stage`` as a ``.value`` string."""
    stage = state.get("stage")
    return stage.value if isinstance(stage, ConversationStage) else str(stage)


def _flip(turns: list[tuple[str, str]]) -> list:
    """
    Role-flip the transcript for the simulated user: from the visitor's seat, TARS's
    lines are the incoming (Human) messages and its own prior lines are its (AI) replies.
    """
    return [
        HumanMessage(content=text) if who == "TARS" else AIMessage(content=text)
        for who, text in turns
    ]


async def _user_reply(user_llm: Any, persona: dict[str, Any], turns: list[tuple[str, str]]) -> str:
    """Ask the simulated-user model for the visitor's next message."""
    system = SystemMessage(content=f"{SIM_USER_PREAMBLE}\n{persona['system']}")
    response = await user_llm.ainvoke([system, *_flip(turns)])
    return response.content.strip()


async def simulate(
    graph: Any, persona: dict[str, Any], user_llm: Any, *, max_turns: int = 12
) -> Trajectory:
    """
    Run one full persona conversation against ``graph`` and return its ``Trajectory``.

    ``graph`` is the real compiled graph with the system-under-test LLM already
    injected (``nodes.set_llm`` + ``build_graph`` by the caller); ``user_llm`` drives
    the visitor. Turn 0 is the proactive greeting (``ainvoke({"messages": []})``, what
    ``/chat/init`` does); each subsequent turn feeds the visitor's reply. Stops on
    state ``stage == COMPLETE`` or after ``max_turns`` visitor messages.
    """
    config = {"configurable": {"thread_id": f"sim-{persona['id']}"}}
    turns: list[tuple[str, str]] = []
    stages: list[str] = []

    with _patched_salesforce():
        state = await graph.ainvoke({"messages": []}, config=config)  # greeting
        stages.append(_stage_of(state))
        prev_ai = _latest_ai(state)
        turns.append(("TARS", prev_ai))

        for _ in range(max_turns):
            user_msg = await _user_reply(user_llm, persona, turns)
            state = await graph.ainvoke(
                {"messages": [HumanMessage(content=user_msg)]}, config=config
            )
            turns.append(("Visitor", user_msg))
            ai = _latest_ai(state)
            if ai and ai != prev_ai:  # the confirmation-accepted turn appends no new reply
                turns.append(("TARS", ai))
                prev_ai = ai
            stage = _stage_of(state)
            stages.append(stage)
            if stage == ConversationStage.COMPLETE.value:
                break

    return Trajectory(turns=turns, stages=stages, final_state=state)
