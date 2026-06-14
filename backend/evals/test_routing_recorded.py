"""
Recorded routing eval — the deterministic CI gate for the stage router.

Same two-tier design as the extraction gate: replays committed cassettes offline,
fails closed when a reworded ROUTER_PROMPT anchor changes the request body. Scored
against an ``acceptable_next`` set per case (discovery vs qualification is
legitimately ambiguous). Floor is the recorded baseline for
``grok-4.3``; refresh when the cassette is re-recorded.
"""

from __future__ import annotations

import pytest

from evals.evaluators import confusion_matrix, routing_accuracy, routing_match
from evals.loader import load_dataset
from evals.targets import build_eval_llm, run_routing

pytestmark = pytest.mark.eval_recorded

# Recorded baseline for grok-4.3 is 1.0 (every case hit its
# acceptable set). The floor keeps headroom for one genuinely-ambiguous case to
# drift on a future re-record, while still failing on a real router regression
# (2+ misses across 16 cases). See _private/docs/build/14-evals.md.
ROUTING_ACCURACY_FLOOR = 0.90


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.default_cassette("routing")
async def test_routing_recorded():
    rows = load_dataset("routing.jsonl")
    llm = build_eval_llm()

    results: list[dict] = []
    by_id: dict[str, str] = {}
    for row in rows:
        predicted = await run_routing(row, llm)
        by_id[row["id"]] = predicted
        results.append(routing_match(predicted, row["acceptable_next"]))

    agg = routing_accuracy(results)
    print(f"\n[routing] accuracy={agg['accuracy']} misses={agg['misses']}")  # noqa: T201
    print(f"[routing] confusion={confusion_matrix(results)}")  # noqa: T201

    assert agg["accuracy"] >= ROUTING_ACCURACY_FLOOR, agg

    # Hard contract: never skip greeting -> lead_capture, even when the visitor
    # volunteers contact info up front (prompts.py:464).
    assert by_id["no-skip-greeting-to-leadcapture"] != "lead_capture", by_id
