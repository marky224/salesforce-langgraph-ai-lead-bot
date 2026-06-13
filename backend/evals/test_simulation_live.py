"""
Live smoke test for the simulated-user loop (PR C2) — report-only, eval_live.

Hits the real provider (grok-4.3) for a single persona end-to-end with Salesforce
mocked. Asserts only *structure* (the loop runs, produces a trajectory, the scorecard
renders) — never the nondeterministic outcome (band / completion), which is what makes
it safe to run nightly/manually but never on the per-commit gate. The deterministic
scorecard *logic* is covered offline by ``test_scorecard.py``.

    pytest backend/evals -m eval_live      # needs a real XAI_API_KEY; spends tokens
"""

from __future__ import annotations

import pytest

from app.graph.graph import build_graph
from app.graph.nodes import set_llm
from app.models.schemas import ConversationStage
from evals.loader import load_dataset
from evals.scorecard import Trajectory, score_persona
from evals.simulate import simulate
from evals.targets import build_eval_llm

pytestmark = pytest.mark.eval_live


@pytest.mark.asyncio
async def test_one_persona_runs_end_to_end():
    persona = next(p for p in load_dataset("personas.jsonl") if p["id"] == "enterprise_dm_urgent")

    set_llm(build_eval_llm())
    graph = build_graph()
    user_llm = build_eval_llm()

    traj = await simulate(graph, persona, user_llm, max_turns=persona["expected"]["max_turns"])

    # Structure only — never assert the nondeterministic band/completion here.
    assert isinstance(traj, Trajectory)
    assert traj.turns and traj.turns[0][0] == "TARS" and traj.turns[0][1]   # greeting first
    assert traj.stages[0] == ConversationStage.GREETING.value
    assert traj.num_visitor_turns >= 1
    assert isinstance(traj.final_state.get("stage"), ConversationStage)

    row = score_persona(traj, persona)
    assert row["id"] == "enterprise_dm_urgent"
    assert {"overall_pass", "reached_complete", "score", "must_capture", "turns", "voice"} <= set(row)
