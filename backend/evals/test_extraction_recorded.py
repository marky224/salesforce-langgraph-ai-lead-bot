"""
Recorded extraction eval — the deterministic CI gate.

Replays committed VCR cassettes (no network, no key), so it fails closed on the
prompt-contract regressions the e2e mock can't catch:

- a reworded prompt anchor changes the request body -> cassette miss -> hard error
  (forcing a deliberate re-record), and
- an ``extraction_node`` / evaluator regression drops field-F1 below the recorded
  baseline.

It does NOT re-judge the live model — that's the nightly ``run.py --mode live``
path + the model A/B this harness enables. Floors below are the recorded baseline
for ``grok-4.3``; refresh them when the cassette is re-recorded.
"""

from __future__ import annotations

import pytest

from app.graph.nodes import _safe_parse_json
from evals.evaluators import aggregate_field_metrics, field_accuracy, parse_success
from evals.loader import load_dataset
from evals.targets import build_eval_llm, run_extraction

pytestmark = pytest.mark.eval_recorded

# Recorded baseline (grok-4.3, see _private/docs/build/14-evals.md). Scalars (the
# enum-.value contract) recorded a perfect 1.0 — the gate fails on ANY scalar
# regression in node/evaluator code. The list-count rate (0.9815 recorded) gets a
# small margin since list cardinality is inherently fuzzier. Floors travel with the
# cassette: re-recording for the model A/B resets them to the new model's baseline.
SCALAR_F1_FLOOR = 1.0
LIST_COUNT_MATCH_FLOOR = 0.90


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.default_cassette("extraction")
async def test_extraction_recorded():
    rows = load_dataset("extraction.jsonl")
    llm = build_eval_llm()

    per_row: list[dict] = []
    parse_failures: list[str] = []
    by_id: dict[str, dict] = {}

    for row in rows:
        predicted = await run_extraction(row, llm)
        by_id[row["id"]] = predicted
        per_row.append(field_accuracy(predicted, row["expected"]))
        ok = parse_success(predicted, row["expected"])
        if ok is False:
            parse_failures.append(row["id"])

    agg = aggregate_field_metrics(per_row)
    print(f"\n[extraction] {agg}")  # noqa: T201 - surfaced with `pytest -s` / on failure

    # No silent ``_safe_parse_json -> {}`` drops on rows that should yield data.
    assert not parse_failures, f"extraction returned empty for {parse_failures}; agg={agg}"
    assert agg["scalar_f1"] >= SCALAR_F1_FLOOR, agg
    assert agg["list_count_match_rate"] >= LIST_COUNT_MATCH_FLOOR, agg

    # Marquee contract: one multi-impact sentence -> distinct pain points, not one
    # consolidated entry (prompts.py:393).
    pains = (by_id["multi-pain-split"].get("qualification_data") or {}).get("pain_points") or []
    assert len(pains) >= 3, by_id["multi-pain-split"]


def test_safe_parse_json_survives_llm_quirks():
    """
    ``_safe_parse_json`` must recover JSON from the wrappers real models emit, so a
    well-formed extraction is never silently dropped to ``{}``. Pure/offline.
    """
    payload = {"qualification_data": {"timeline": "1-3 months"}}
    fenced = '```json\n{"qualification_data": {"timeline": "1-3 months"}}\n```'
    prefixed = 'Assistant: {"qualification_data": {"timeline": "1-3 months"}}'
    trailing = '{"qualification_data": {"timeline": "1-3 months"}}\n\nLet me know if that helps!'

    assert _safe_parse_json(fenced) == payload
    assert _safe_parse_json(prefixed) == payload
    assert _safe_parse_json(trailing) == payload
