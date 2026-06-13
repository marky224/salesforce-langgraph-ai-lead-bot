# backend/evals — offline extraction/routing gate + simulated-user scorecard

An **opt-in** eval suite that measures extraction + routing against the **real prompts** and
**recorded real-model responses** — the gap the mock-based `backend/tests` suite can't cover. It also
runs a **simulated-user scorecard** (PR C): whole conversations driven against the real graph by a
second LLM role-playing personas. It is never collected by `pytest tests/`, never installed into the
prod image, and its deps live in `backend/requirements-eval.txt`.

## Two tiers

| Tier | What | When |
|------|------|------|
| **recorded** (the gate) | Replays committed VCR cassettes — no key, no network, deterministic. | Every PR (CI), local. |
| **live** | Hits the real provider. Drives the **model A/B** and the **simulated-user scorecard**. Costs tokens. | Manual / nightly. |

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

# simulated-user scorecard (live, needs a real XAI_API_KEY; spends tokens):
cd backend && python -m evals.run --dimension sim --mode live --report md
```

## Simulated-user scorecard (live, report-only)

A second LLM role-plays a website visitor (3 personas in `datasets/personas.jsonl`) and drives the
**real compiled graph** + real grok-4.3 end-to-end, with Salesforce mocked (zero CRM writes). Each
conversation is scored on deterministic final-state outcomes — reached `COMPLETE`, deterministic
`compute_lead_score` in the persona's band, `must_capture` fields present, no stage loop,
turns-to-complete — plus advisory voice guardrails (markdown leak, brevity). Output is a per-persona
markdown scorecard.

Nondeterministic by nature → **report-only, bands not exact, manual/nightly, never the per-commit
gate.** The pure scorecard *evaluators* (`scorecard.py`) are unit-tested offline under `eval_recorded`
(`test_scorecard.py`), so the logic is CI-covered without any live call; the live smoke
(`test_simulation_live.py`) is `eval_live`. CI: a guarded `workflow_dispatch` job in `evals.yml` (needs
the `XAI_API_KEY` Actions secret) uploads the scorecard artifact — no auto-spend.

```bash
cd backend && python -m evals.run --dimension sim --mode live --report md --output scorecard.md
```

Add a persona: append one line to `datasets/personas.jsonl` (`id`, `system` = visitor profile,
`expected` = `{reaches_complete, score_band, must_capture, max_turns}`) using the exact enum `.value`
strings (contract #1) and synthetic `*.example` identities (no PII).

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
