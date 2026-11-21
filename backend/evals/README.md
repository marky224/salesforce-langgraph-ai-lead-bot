# backend/evals — offline extraction/routing/site-answer gate + simulated-user scorecard

An **opt-in** eval suite that measures extraction, routing and site answers against the **real prompts** and
**recorded real-model responses** — the gap the mock-based `backend/tests` suite can't cover. It also
runs a **simulated-user scorecard** (PR C): whole conversations driven against the real graph by a
second LLM role-playing personas. And it has an **online eval** (PR D4): a manual command that scores
recent production turns from LangSmith traces and writes the scores back as LangSmith feedback. It is
never collected by `pytest tests/`, never installed into the prod image, and its deps live in
`backend/requirements-eval.txt`.

## Two tiers

| Tier | What | When |
|------|------|------|
| **recorded** (the gate) | Replays committed VCR cassettes — no key, no network, deterministic. | Every PR (CI), local. |
| **live** | Hits the real provider. Drives the **model A/B** and the **simulated-user scorecard**. Costs tokens. | Manual / nightly. |

The live pytest tests (`eval_live`: `test_judges_live.py`, `test_simulation_live.py`) run only with
`--run-live`. A plain `pytest`, or `-m eval_live` alone, skips them, so real keys in the shell or `.env`
can't spend tokens by accident (`test_live_opt_in.py` checks this offline).

The gate **fails closed**: extraction/router invoke the LLM with `messages=[]`, so the request body is
deterministic per case and cassettes are body-matched. A reworded prompt anchor changes the body, misses
the cassette, and errors — instead of silently replaying a stale response. The site-answer gate passes
the visitor's question as the only message, which is just as deterministic.

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

# live pytest smoke tests (need real XAI_API_KEY + ANTHROPIC_API_KEY; spend tokens).
# --run-live is required. Pytest knows it only when the run starts in backend/evals or names it
# (or a file in it); a bare `pytest --run-live` from the repo root or backend/ is a usage error
# and nothing runs:
pytest backend/evals -m eval_live --run-live
```

## Site answers (recorded)

`test_site_answer_recorded.py` drives the `site_answer` node directly over the committed fixture index
(`backend/tests/fixtures/site_index_v1.json`, public page text plus synthetic edge chunks) with grok-4.3
replayed from `cassettes/site_answer.yaml`. Each row of `datasets/site_answer.jsonl` expects a `link`
(answered, ending with that page's `Read more` link), `no_answer` (the fixed no-match reply, from the
score threshold or the model's `NO_ANSWER`) or `either`. Every reply must carry no URL except its own
page link, and nothing from the fixture's injected line. Rows under the threshold make no model call
and have no cassette entry.

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
(`test_simulation_live.py`) is `eval_live` and needs `--run-live`. CI: a guarded `workflow_dispatch` job in `evals.yml` (needs
the `XAI_API_KEY` Actions secret) uploads the scorecard artifact — no auto-spend.

```bash
cd backend && python -m evals.run --dimension sim --mode live --report md --output scorecard.md
```

Add a persona: append one line to `datasets/personas.jsonl` (`id`, `system` = visitor profile,
`expected` = `{reaches_complete, score_band, must_capture, max_turns}`) using the exact enum `.value`
strings (contract #1) and synthetic `*.example` identities (no PII).

## Calibrated LLM judges (live, report-only)

Two LLM-as-judge quality checks the deterministic scorecard can't measure, graded by a frontier
`claude-sonnet-4-6` (a *different* model family from the grok-4.3 SUT, to avoid correlated blind spots):
**persona adherence** (did TARS hold its deadpan persona?) and **summary faithfulness** (is the Salesforce
`transcript_summary` grounded in the chat?). A judge counts toward the scorecard only after it clears a
documented **Cohen's κ** bar against human labels — that is the difference between *adding* a judge and
*validating* one.

```bash
# calibrate the judges against the 20 human-confirmed labels (live; needs ANTHROPIC_API_KEY):
cd backend && python -m evals.calibration

# run the sim scorecard WITH the calibrated judges attached (live; extra cost):
cd backend && python -m evals.run --dimension sim --mode live --judges --report md
```

`KAPPA_BAR = 0.6` (Landis–Koch "substantial"); the recorded result lives in `calibration.py::CALIBRATED_KAPPA`
and travels with the labels + judge prompts (re-run + update on any change). A judge below the bar is still
shown but flagged **advisory** and never folded into `overall_pass`. Recorded 2026-11-21, a re-run under
langchain-anthropic 1.7.5: `persona_adherence` κ=0.61 (validated, by a thin margin: 3 of its 12 labels
split; the first recording, 2026-06-15, was 0.74); `summary_faithfulness` κ=0.25, as before (advisory — the
generic groundedness rubric over-flags the summary's asked-for sentiment gloss; a task-specific prompt is
the follow-up). A longer write-up is in the maintainer's local build notes, which are not published.

## Online eval (live, manual)

`python -m evals.online` scores recent **production turns**. It reads conversational turns from one
LangSmith project, runs two code checks and the validated persona judge on each, and writes the scores
onto the turn's trace as LangSmith feedback. You run it by hand: nothing schedules it, no CI job runs it,
and there is no `eval_live` pytest for it, so `pytest backend/evals -m eval_live --run-live` never writes
to a project.

```bash
# dry run (the default): selects and reads turns, calls no judge, writes nothing
cd backend && python -m evals.online --project <project>

# judge up to 20 unscored turns and write the feedback rows (needs ANTHROPIC_API_KEY; spends tokens)
cd backend && python -m evals.online --project <project> --write
```

| Option | Default | What it does |
|--------|---------|--------------|
| `--project` | none (required) | The LangSmith project to read turns from and write feedback to. |
| `--limit` | 20 | Most turns one run scores. Hard maximum 100: a larger value is refused before any LangSmith call. |
| `--since-hours` | 168 | How far back to look for turns. |
| `--write` | off | Call the judge and write feedback. Without it the run is a dry run. |

**What one score covers.** One turn is one trace: exactly one of the five conversational nodes
(`discovery`, `qualification`, `objection_handling`, `lead_capture`, `confirmation`) ran in it and finished
without error, and it carries the `tars` tag. That is one visitor message and TARS's reply to it, the
shape the persona judge's κ was measured on (the 12 persona labels are single exchanges). Greetings, site
answers and the fixed replies are not judged. Turns are taken newest first, and a turn that already has
every key is skipped, so a second run over the same traces calls no judge and writes nothing; a run that
died halfway is finished by the next one, which writes only the missing keys.

**What is written.** Up to three feedback rows on the trace's root run:

| Key | Score | Source |
|-----|-------|--------|
| `markdown_clean` | 1 = no markdown in the reply, 0 = markdown found (`scorecard.markdown_leak`) | code |
| `brevity_ok` | 1 = at most 5 sentences, 0 = more (`scorecard.brevity_ok`) | code |
| `persona_adherence` | 1 = on-voice, 0.5 = slightly off, 0 = off-persona | the `claude-sonnet-4-6` judge |

Each comment is fixed text built in code. The judge's reasoning is never stored or printed, because
LangSmith does not mask feedback and the reasoning quotes the turn; to see why a turn scored 0, open its
trace. `persona_adherence` is written only while its recorded κ clears the bar (`calibration.is_validated`).
`summary_faithfulness` (κ below the bar) is not part of the online eval. The two code checks are written before the
judge is called, so a key that can't write feedback costs no judge call.

**What the judge provider sees.** The judge prompt and one exchange: `Visitor: <message>` / `TARS: <reply>`.
Before the call, the text goes through the tracer's email and phone masking again, which covers a trace
that was stored unmasked. Then the values captured in that trace's `lead_data` are replaced: `first_name`,
`last_name` and the full name with `[NAME]`, `company` with `[COMPANY]`, `title` with `[TITLE]`, and the
captured `email` and `phone` with `[EMAIL]` and `[PHONE]`. The order matters: a placeholder written next
to an `@` or after an area code would stop the tracer's patterns from matching the rest of that email or
number. Each captured value is looked for as stored and in title case, and each of those two forms also
as the tracer's masking rewrites it. The plain forms catch a captured phone that the tracer's pattern skips
in the text (right after a `.` or a `+`, or before `.<digit>`, as in `Ph.210-555-0188`); the rewritten ones
catch a captured email that the pattern only half matches, such as one with a non-ASCII letter in it,
whether the visitor typed it as captured or capitalised. Nothing else from the trace is sent: no earlier
turns, no lead data, no ids. The judge calls themselves are never traced.

**Limits.**
- The masking is best-effort. A name typed in another case or misspelled, another person's name, a phone
  number in a shape the patterns miss, and everything else the visitor typed in that one message still
  reach the judge provider. The judge was not calibrated on text that holds the `[NAME]`, `[COMPANY]` and
  `[TITLE]` placeholders.
- Visitor text goes into the judge prompt, so a visitor can steer a score (prompt injection). The damage
  stops at the score: it is always 1, 0.5 or 0, and nothing the judge writes is stored.
- Two runs at the same time can write a key twice. Run one at a time.
- One run reads at most 1,000 node runs from the listing. It sends LangSmith no `limit`, because the
  server refuses one over 100; it reads the listing page by page and stops at 1,000 itself. When it stops
  there it prints `scan cap reached`, and the turns past that point were not seen.
- A feedback write that fails is tried 3 times, then the run stops (exit 1). How long that takes depends on
  how the server fails. When it answers with an error (a 500 or a 404), the waits add up to about 3
  seconds. On a rate limit (429) they follow the server's `Retry-After`, about 12 times its value in all,
  or about 90 seconds when the server sends none. When the server stops answering, the client's HTTP
  adapter retries each attempt itself (4 tries, each up to the 60-second read timeout), so one write can
  wait about 12 minutes before it raises. A refused key (401 or 403) stops the run at once. The command
  prints nothing while it waits.
- A phone number that the tracer itself misses is sent as it was stored. The tracer's phone pattern skips
  a number right after a `.` or a `+`, or before `.<digit>` (`Ph.210-555-0188`). In a normal masked upload
  such a number is stored raw while the captured `phone` is stored as `[PHONE]`, so the online eval has
  nothing to find it with. Fixing that takes a change to the app's tracing patterns (`app/tracing.py`),
  which the online eval leaves alone.

**Keys.** `LANGSMITH_API_KEY` always, and `ANTHROPIC_API_KEY` with `--write`, read from the environment or
`.env`. An org-scoped LangSmith key also needs `LANGSMITH_WORKSPACE_ID` **exported in the shell**: the
LangSmith client reads it from the process environment, so a value that is only in `.env` never reaches it.

**Exit status.**

| Status | Meaning |
|--------|---------|
| 0 | A dry run, nothing to do, or every selected turn fully written. |
| 1 | A selected turn couldn't be parsed, a judge call failed twice, or a feedback write failed. |
| 2 | A missing key, a `--limit` outside 1–100, or a LangSmith list or read call that failed (a refused key or project included) before any write. |

Two cases where the rules meet: a dry run that found an unparsed turn exits 1, not 0; and a refusal (401
or 403) on a feedback write is a failed write, so it exits 1, not 2. Output goes to standard output only
(counts, trace ids, scores, judge call and token totals). An error prints its exception class, never its text.

**If the dry run finds no eligible turn** (`eligible=0`), stop and tell the maintainer. Don't start a chat
with the bot to have something to score: that spends the bot's tokens, adds a synthetic trace to the
project, and leaves the eval scoring its own traffic. Whether to chat with the bot first is the
maintainer's decision, and a result that rests on such a chat should say so.

**Cost.** One judge call per turn, plus one retry when a call fails. A default run (20 turns) is an
estimated $0.15–0.20 at the judge model's list price ($3 / $15 per 1M input / output tokens); the run
prints its real call and token totals.

**Tests.** `test_online_recorded.py` rides the recorded gate (`eval_recorded`): an in-memory LangSmith fake
and a stub judge, no key and no network.

## Add a case

Append one JSON line to `datasets/extraction.jsonl`, `datasets/routing.jsonl` or `datasets/site_answer.jsonl`.
Extraction `expected`
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

Deep dive: the maintainer's local build notes, which are not published.
