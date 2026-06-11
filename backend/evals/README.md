# backend/evals — offline extraction & routing evals

An **opt-in** eval suite that measures extraction + routing against the **real prompts** and
**recorded real-model responses** — the gap the mock-based `backend/tests` suite can't cover. It is
never collected by `pytest tests/`, never installed into the prod image, and its deps live in
`backend/requirements-eval.txt`.

## Two tiers

| Tier | What | When |
|------|------|------|
| **recorded** (the gate) | Replays committed VCR cassettes — no key, no network, deterministic. | Every PR (CI), local. |
| **live** | Hits the real provider. Drives the reasoning→non-reasoning **model A/B**. Costs tokens. | Manual / nightly. |

The gate **fails closed**: extraction/router invoke the LLM with `messages=[]`, so the request body is
deterministic per case and cassettes are body-matched. A reworded prompt anchor changes the body, misses
the cassette, and errors — instead of silently replaying a stale response.

## Run

```bash
pip install -r backend/requirements-eval.txt

# the gate (offline replay — what CI runs):
pytest backend/evals -m eval_recorded --record-mode=none

# readable scorecard, no asserts:
cd backend && python -m evals.run --dimension all --report text

# model A/B (live, needs a real XAI_API_KEY; spends tokens):
cd backend && python -m evals.run --mode live --model grok-4.3 --report md
```

## Add a case

Append one JSON line to `datasets/extraction.jsonl` or `datasets/routing.jsonl`. Extraction `expected`
**must use the exact enum `.value` strings** from `app/models/schemas.py` (contract #1). Then record the
new case's response:

```bash
cd backend && python -m pytest evals/test_extraction_recorded.py -s --record-mode=once
```

## Refresh cassettes (prompt changed / model swap)

```bash
cd backend && rm evals/cassettes/<dim>.yaml
python -m pytest evals/test_<dim>_recorded.py -s --record-mode=once   # re-records against live Grok
```

Review the YAML diff (it's the "model behaviour changed" artifact), update the floors in the test to the
new baseline, and commit. Cassettes have auth/cookie headers filtered — no secrets are stored.

Deep dive (local-only): `_private/docs/build/14-evals.md`.
