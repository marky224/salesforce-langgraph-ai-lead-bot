"""
Online eval (PR D4) — score recent production turns and write the scores to LangSmith.

``python -m evals.online --project <project>`` lists the recent conversational turns of
one LangSmith project, skips the ones that already carry this command's feedback keys,
and for up to ``--limit`` of the rest reads the trace's root run, runs two code checks
and the validated persona judge on that one exchange, and writes each score onto the
trace as LangSmith feedback. It is a manual, live command: no pytest gate runs it, and
nothing under ``app/`` imports it.

Dry run by default
    Without ``--write`` it selects the turns, reads them, checks that each one parses and
    prints counts and trace ids. It calls no judge and writes nothing.

What one score covers
    One turn is one trace: exactly one of the five conversational nodes (discovery,
    qualification, objection_handling, lead_capture, confirmation) ran in it and finished
    without error, it carries the ``tars`` tag, and the node run started inside
    ``--since-hours``. That is the shape the persona judge's kappa was measured on (one
    visitor message and one reply). Greetings, site answers and the fixed replies are
    not judged. Newest first; a turn that already has every key is skipped, so a second
    run over the same traces calls no judge and writes nothing.

What the judge provider receives
    The judge prompt and two strings: ``Visitor: <message>\\nTARS: <reply>`` and the reply
    alone. Before the call the values captured in that trace's ``lead_data`` are replaced:
    ``first_name``, ``last_name`` and the two joined with ``[NAME]``, ``company`` with
    ``[COMPANY]``, ``title`` with ``[TITLE]``, and the captured ``email`` and ``phone``
    with ``[EMAIL]`` and ``[PHONE]``. Then the text goes through the tracer's email and
    phone masking again. Nothing else from the trace is sent: no earlier turns, no lead
    data, no ids. Best-effort: a name typed in another case or misspelled, someone else's
    name and a phone shape the patterns miss still reach the judge, as does everything
    else the visitor typed in that message. Visitor text sits in the judge prompt, so it
    can steer a score; the damage stops at the score (1, 0.5 or 0), because the judge's
    reasoning is never stored or printed.

What is written
    Up to three feedback rows on the trace's root run: ``markdown_clean`` (1 or 0),
    ``brevity_ok`` (1 or 0) and ``persona_adherence`` (1, 0.5 or 0). Each comment is fixed
    text built here. ``persona_adherence`` is written only while its recorded kappa clears
    the bar (``calibration.is_validated``); under the bar the judge isn't called. Writes
    are synchronous: the client has no upload queue, so a failed write raises and stops
    the run. The two code checks are written before the judge is called, so a key that
    can't write costs no judge call.

Keys
    ``LANGSMITH_API_KEY`` always and ``ANTHROPIC_API_KEY`` with ``--write``, both through
    the app's Settings (the environment or ``.env``). An org-scoped LangSmith key also
    needs ``LANGSMITH_WORKSPACE_ID`` exported in the shell: the client reads it from the
    process environment, and a value that is only in ``.env`` never reaches it.

Exit status
    0  a dry run, nothing to do, or every selected turn fully written.
    1  a selected turn couldn't be parsed, a judge call failed twice, or a feedback write
       failed.
    2  a missing key, a ``--limit`` outside 1-100, or a LangSmith list or read call that
       failed (a refused key or project included) before any write.
    Precedence: a dry run that met an unparsed turn exits 1, not 0. A refusal (401 or
    403) on a feedback write is a failed write: exit 1, not 2.

Output
    Standard output only: counts, trace ids, scores, judge call and token totals. An
    error prints its exception class, never its text, and the ``langsmith`` logger is set
    above ERROR so the client's own error logs (a traceback with the response body, part
    of the key on a connection error) stay off stderr. No eval call is traced.

Run from ``backend/``:  ``python -m evals.online --project <project> [--write]``
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from langchain_core.callbacks import get_usage_metadata_callback
from langsmith import Client

# Make ``app`` / ``evals`` importable however this is invoked (mirrors run.py).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import config as app_config  # noqa: E402
from app.graph.edges import (  # noqa: E402
    NODE_CONFIRMATION,
    NODE_DISCOVERY,
    NODE_LEAD_CAPTURE,
    NODE_OBJECTION_HANDLING,
    NODE_QUALIFICATION,
)
from app.tracing import build_email_anonymizer  # noqa: E402
from evals import calibration, judges  # noqa: E402
from evals._config import force_tracing_off  # noqa: E402
from evals.scorecard import brevity_ok, markdown_leak  # noqa: E402

EVALUATOR = "evals.online"

PERSONA_KEY = "persona_adherence"
MARKDOWN_KEY = "markdown_clean"
BREVITY_KEY = "brevity_ok"

DEFAULT_LIMIT = 20
MAX_LIMIT = 100  # hard maximum: a typo such as 2000 must not become 2,000 judge calls
DEFAULT_SINCE_HOURS = 168
SCAN_CAP = 1000  # most node runs one listing reads
_FEEDBACK_READ_CHUNK = 50  # trace ids per feedback read

# The nodes whose reply TARS writes in its persona. One of them runs per chat turn.
CONVERSATIONAL_NODES = (
    NODE_DISCOVERY,
    NODE_QUALIFICATION,
    NODE_OBJECTION_HANDLING,
    NODE_LEAD_CAPTURE,
    NODE_CONFIRMATION,
)
# The tag the server puts on every graph run (``server._trace_base``); a replay by
# ``app.resend_lead`` doesn't carry it.
TRACE_TAG = "tars"

_RUN_FILTER = (
    f'and(has(tags, "{TRACE_TAG}"), or('
    + ", ".join(f'eq(name, "{node}")' for node in CONVERSATIONAL_NODES)
    + "))"
)
# ``run_type`` is there because the client builds a ``Run`` from each row and that
# schema requires it; ``tags`` and ``extra`` feed the local checks in ``_eligible_trace_ids``.
_RUN_SELECT = ["id", "name", "run_type", "trace_id", "start_time", "end_time", "error", "extra", "tags"]

# key -> (the check: True scores 1, the scorecard function it reports, the fixed comment)
_CHECKS: dict[str, tuple[Callable[[str], bool], str, str]] = {
    MARKDOWN_KEY: (
        lambda reply: not markdown_leak(reply),
        "markdown_leak",
        "evals.online: scorecard.markdown_leak on TARS's reply. 1 = no markdown, 0 = markdown found.",
    ),
    BREVITY_KEY: (
        brevity_ok,
        "brevity_ok",
        "evals.online: scorecard.brevity_ok on TARS's reply. 1 = at most 5 sentences, 0 = more.",
    ),
}

# lead_data value -> the placeholder the judge sees in its place.
_LEAD_PLACEHOLDERS = (
    ("company", "[COMPANY]"),
    ("title", "[TITLE]"),
    ("email", "[EMAIL]"),
    ("phone", "[PHONE]"),
)


@dataclass(frozen=True)
class Turn:
    """One visitor message and TARS's reply to it, as read from a trace's root run."""

    trace_id: str
    visitor: str
    reply: str
    lead_data: dict[str, Any]


@dataclass
class _Tally:
    """What a write run did, for its closing lines."""

    rows: int = 0
    traces: set[str] = field(default_factory=set)
    judge_calls: int = 0
    judge_errors: int = 0


class _WriteFailed(Exception):
    """A feedback write raised. Carries the original exception's class name only."""


def _out(line: str) -> None:
    print(line)  # noqa: T201


# ---------------------------------------------------------------------------
# Selecting turns
# ---------------------------------------------------------------------------

def _feedback_keys() -> list[str]:
    """The keys this run writes, in write order: the code checks, then the judge's."""
    keys = [MARKDOWN_KEY, BREVITY_KEY]
    if calibration.is_validated(PERSONA_KEY):
        keys.append(PERSONA_KEY)
    return keys


def _eligible_trace_ids(runs: Sequence[Any]) -> list[str]:
    """
    Trace ids of the scoreable turns among the listed node runs, newest first.

    A trace counts when exactly one conversational node ran in it, tagged ``tars``, and
    that run finished without error. The name and ``langgraph_node`` must agree, the same
    test ``/chat/stream`` uses to tell a node's own run from a run nested inside it. The
    server filter asks for the same thing; checking again here keeps the selection right
    if the server reads that filter differently.
    """
    by_trace: dict[str, list[Any]] = {}
    for run in sorted(runs, key=lambda r: r.start_time, reverse=True):
        metadata = (run.extra or {}).get("metadata") or {}
        if (
            run.name in CONVERSATIONAL_NODES
            and metadata.get("langgraph_node") == run.name
            and TRACE_TAG in (run.tags or [])
        ):
            by_trace.setdefault(str(run.trace_id), []).append(run)
    return [
        trace_id
        for trace_id, node_runs in by_trace.items()
        if len(node_runs) == 1 and node_runs[0].end_time is not None and not node_runs[0].error
    ]


def _existing_keys(client: Any, trace_ids: Sequence[str], keys: Sequence[str]) -> set[tuple[str, str]]:
    """The ``(trace id, key)`` pairs that already have a feedback row."""
    found: set[tuple[str, str]] = set()
    for start in range(0, len(trace_ids), _FEEDBACK_READ_CHUNK):
        chunk = list(trace_ids[start : start + _FEEDBACK_READ_CHUNK])
        for row in client.list_feedback(run_ids=chunk, feedback_key=list(keys)):
            found.add((str(row.run_id), row.key))
    return found


# ---------------------------------------------------------------------------
# Reading a turn from the stored root run
# ---------------------------------------------------------------------------

def _message_text(content: Any) -> str:
    """A stored message's text. A list of content blocks has its text parts joined."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


def _last_text(messages: Any, message_type: str) -> str:
    """The text of the last stored message of ``message_type`` (``human`` or ``ai``)."""
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("type") == message_type:
            return _message_text(message.get("content")).strip()
    return ""


def _parse_turn(trace_id: str, run: Any) -> Turn | None:
    """
    Read the turn out of a trace's root run, or ``None`` when it can't be read.

    The root run's inputs are the request's one visitor message; its outputs are the
    whole graph state after the turn, whose last ``ai`` message is the reply.
    """
    inputs, outputs = run.inputs, run.outputs
    if run.error or not isinstance(inputs, dict) or not isinstance(outputs, dict):
        return None
    visitor = _last_text(inputs.get("messages"), "human")
    reply = _last_text(outputs.get("messages"), "ai")
    if not visitor or not reply:
        return None
    lead_data = outputs.get("lead_data")
    return Turn(trace_id, visitor, reply, lead_data if isinstance(lead_data, dict) else {})


# ---------------------------------------------------------------------------
# What the judge sees
# ---------------------------------------------------------------------------

def _lead_values(lead_data: dict[str, Any]) -> dict[str, str]:
    """Each captured lead value worth masking, mapped to its placeholder."""
    first, last = lead_data.get("first_name"), lead_data.get("last_name")
    full_name = " ".join(part for part in (first, last) if isinstance(part, str) and part.strip())
    pairs = [(full_name, "[NAME]"), (first, "[NAME]"), (last, "[NAME]")]
    pairs += [(lead_data.get(key), placeholder) for key, placeholder in _LEAD_PLACEHOLDERS]

    values: dict[str, str] = {}
    for value, placeholder in pairs:
        if not isinstance(value, str) or len(value.strip()) < 2:
            continue
        # As stored and in title case: extraction may keep "dana" as the visitor typed it.
        for form in (value.strip(), value.strip().title()):
            values.setdefault(form, placeholder)
    return values


def _mask_lead_values(text: str, lead_data: dict[str, Any]) -> str:
    """
    Replace the lead values captured in this trace with their placeholders.

    The values are visitor text that came through the extraction model, so they are
    untrusted: each one is matched literally (``re.escape``), never as a pattern.
    ``(?<!\\w)`` / ``(?!\\w)`` instead of ``\\b``, which never matches a value that ends in
    a non-word character ("Yahoo!", "C++ Labs"). One pass, longest value first, so a
    full name wins over its parts and a placeholder is never matched again.
    """
    values = _lead_values(lead_data)
    if not values:
        return text
    alternatives = "|".join(re.escape(value) for value in sorted(values, key=len, reverse=True))
    return re.sub(rf"(?<!\w)(?:{alternatives})(?!\w)", lambda match: values[match.group(0)], text)


def _judge_inputs(turn: Turn) -> tuple[str, str]:
    """
    The two strings the judge gets: the exchange in the calibration labels' format, and
    the reply alone as the text to score.

    The captured lead values go first, then the tracer's own email and phone masking, so
    no raw email or phone shape leaves here even for a trace that was stored unmasked.
    """
    masked = build_email_anonymizer()(
        {
            "visitor": _mask_lead_values(turn.visitor, turn.lead_data),
            "reply": _mask_lead_values(turn.reply, turn.lead_data),
        }
    )
    return f"Visitor: {masked['visitor']}\nTARS: {masked['reply']}", masked["reply"]


# ---------------------------------------------------------------------------
# Judging and writing
# ---------------------------------------------------------------------------

def _persona_comment() -> str:
    kappa = calibration.CALIBRATED_KAPPA[PERSONA_KEY]
    return (
        f"evals.online: {judges.JUDGE_MODEL} judged one visitor message and TARS's reply against "
        "the TARS persona. 1 = on-voice, 0.5 = slightly off, 0 = off-persona. "
        f"Judge kappa {kappa:.2f} against human labels (bar {calibration.KAPPA_BAR}, validated)."
    )


def _persona_source() -> dict[str, Any]:
    return {
        "evaluator": EVALUATOR,
        "judge_model": judges.JUDGE_MODEL,
        "kappa": calibration.CALIBRATED_KAPPA[PERSONA_KEY],
        "kappa_bar": calibration.KAPPA_BAR,
        "validated": True,
    }


def _write_row(
    client: Any,
    tally: _Tally,
    trace_id: str,
    key: str,
    score: float,
    comment: str,
    source_type: str,
    source_info: dict[str, Any],
) -> None:
    """
    Write one feedback row on the trace's root run.

    ``run_id`` only, never ``trace_id=``: with a trace id a client that has an upload
    queue only queues the row, and a failed upload would not raise.
    """
    try:
        client.create_feedback(
            run_id=trace_id,
            key=key,
            score=score,
            comment=comment,
            feedback_source_type=source_type,
            source_info=source_info,
        )
    except Exception as exc:
        # The class only: a LangSmith error's text holds the URL and the response body.
        raise _WriteFailed(type(exc).__name__) from None
    tally.rows += 1
    tally.traces.add(trace_id)


async def _judge_score(judge: Any, turn: Turn, tally: _Tally) -> tuple[float | None, str]:
    """
    Ask the persona judge about one masked exchange, with one retry.

    Returns ``(score, "")``, or ``(None, <exception class>)`` after two failures. The
    exception's text is dropped: a parse error can quote the judge's output, which
    repeats the visitor. The score is snapped the way the calibration snaps it.
    """
    inputs, outputs = _judge_inputs(turn)
    error = ""
    for _ in range(2):
        tally.judge_calls += 1
        try:
            result = await judge(inputs=inputs, outputs=outputs)
            return calibration._persona_class(calibration._score(result)), ""
        except Exception as exc:
            tally.judge_errors += 1
            error = type(exc).__name__
    return None, error


async def _score_turn(client: Any, judge: Any, turn: Turn, missing: Sequence[str], tally: _Tally) -> bool:
    """
    Write this turn's missing rows and print its line. False when the judge failed twice.

    The code checks run on the reply as stored (they are local, nothing leaves) and are
    written before the judge is called.
    """
    fields: list[str] = []
    rows_before = tally.rows
    for key, (check, check_name, comment) in _CHECKS.items():
        if key in missing:
            score = int(check(turn.reply))
            _write_row(
                client, tally, turn.trace_id, key, score, comment,
                "api", {"evaluator": EVALUATOR, "check": check_name},
            )
            fields.append(f"{key}={score}")

    judged = True
    if PERSONA_KEY in missing:
        persona, error = await _judge_score(judge, turn, tally)
        if persona is None:
            judged = False
            fields.append(f"{PERSONA_KEY}=error({error})")
        else:
            _write_row(
                client, tally, turn.trace_id, PERSONA_KEY, persona, _persona_comment(),
                "model", _persona_source(),
            )
            fields.append(f"{PERSONA_KEY}={persona}")

    _out(f"trace {turn.trace_id} {' '.join(fields)} written={tally.rows - rows_before}")
    return judged


async def run_online(
    client: Any,
    judge: Any | None,
    *,
    project: str,
    limit: int,
    since_hours: int,
    write: bool,
    now: datetime,
) -> int:
    """
    Select, read and (with ``write``) score the turns of ``project``; return the exit status.

    ``client`` is a LangSmith client without an upload queue and ``judge`` the persona
    evaluator (``judges.persona_adherence_judge()``); both are arguments so the offline
    tests pass doubles. ``judge`` may be ``None`` in a dry run and while the persona
    judge is under the kappa bar. Every list and read call happens before the first
    write. Exit status: the module docstring.
    """
    keys = _feedback_keys()
    _out(
        f"online eval: project={project} window={since_hours}h limit={limit} "
        f"mode={'write' if write else 'dry-run'}"
    )

    try:
        runs = list(
            client.list_runs(
                project_name=project,
                run_type="chain",
                start_time=now - timedelta(hours=since_hours),
                filter=_RUN_FILTER,
                select=_RUN_SELECT,
                limit=SCAN_CAP,
            )
        )
        eligible = _eligible_trace_ids(runs)
        existing = _existing_keys(client, eligible, keys)
        missing = {
            trace_id: [key for key in keys if (trace_id, key) not in existing]
            for trace_id in eligible
        }
        unscored = [trace_id for trace_id in eligible if missing[trace_id]]
        selected = unscored[:limit]
        turns = [(trace_id, _parse_turn(trace_id, client.read_run(trace_id))) for trace_id in selected]
    except Exception as exc:
        # The class only: a LangSmith error's text holds the URL and the response body,
        # and a schema error would quote the stored run.
        _out(f"error: {type(exc).__name__} while listing or reading LangSmith; nothing was written")
        return 2

    unparsed = sum(1 for _, turn in turns if turn is None)
    _out(
        f"scanned={len(runs)} eligible={len(eligible)} "
        f"already_scored={len(eligible) - len(unscored)} selected={len(selected)} "
        f"remaining={len(unscored) - len(selected)} unparsed={unparsed}"
    )
    if len(runs) >= SCAN_CAP:
        _out(f"scan cap reached: the listing stopped at {SCAN_CAP} node runs; older turns were not seen")
    status = 1 if unparsed else 0

    if not write:
        for trace_id, turn in turns:
            if turn is None:
                _out(f"trace {trace_id} unparsed")
            else:
                _out(f"trace {trace_id} would_write={','.join(missing[trace_id])}")
        return status

    tally = _Tally()
    with get_usage_metadata_callback() as usage:
        for trace_id, turn in turns:
            if turn is None:
                _out(f"trace {trace_id} unparsed")
                continue
            try:
                if not await _score_turn(client, judge, turn, missing[trace_id], tally):
                    status = 1
            except _WriteFailed as failed:
                _out(f"error: {failed} on a feedback write for trace {trace_id}; run stopped")
                status = 1
                break

    if PERSONA_KEY in keys:
        totals = list(usage.usage_metadata.values())
        _out(
            f"judge: model={judges.JUDGE_MODEL} calls={tally.judge_calls} errors={tally.judge_errors} "
            f"input_tokens={sum(t.get('input_tokens', 0) for t in totals)} "
            f"output_tokens={sum(t.get('output_tokens', 0) for t in totals)}"
        )
    else:
        _out(f"judge: not run ({PERSONA_KEY} is under the kappa bar {calibration.KAPPA_BAR})")
    _out(f"written={tally.rows} rows on {len(tally.traces)} traces")
    return status


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evals.online",
        description="Score recent conversational turns of a LangSmith project and write the "
                    "scores back as feedback. A dry run unless --write is given.",
    )
    parser.add_argument(
        "--project", required=True,
        help="LangSmith project to read turns from and write feedback to (no default).",
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help=f"Most turns one run scores (default {DEFAULT_LIMIT}, maximum {MAX_LIMIT}).",
    )
    parser.add_argument(
        "--since-hours", type=int, default=DEFAULT_SINCE_HOURS,
        help=f"How far back to look for turns (default {DEFAULT_SINCE_HOURS}).",
    )
    parser.add_argument(
        "--write", action="store_true",
        help="Call the judge (needs ANTHROPIC_API_KEY, spends tokens) and write feedback rows. "
             "Without it nothing is judged or written.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not 1 <= args.limit <= MAX_LIMIT:
        _out(f"error: --limit must be between 1 and {MAX_LIMIT}")
        return 2

    # No eval call is traced, whatever the shell or .env says, and the Anthropic SDK's
    # own log switch is removed before the SDK is imported: at debug it prints the whole
    # request, which here holds a visitor's message.
    force_tracing_off()
    os.environ.pop("ANTHROPIC_LOG", None)
    # langsmith logs a failed request itself: a traceback with the URL and response body,
    # and on a connection error the first 5 and last 2 characters of the key.
    logging.getLogger("langsmith").setLevel(logging.CRITICAL)

    settings = app_config.get_settings()
    if settings.langsmith_api_key is None:
        _out("error: LANGSMITH_API_KEY is not set")
        return 2
    use_judge = args.write and calibration.is_validated(PERSONA_KEY)
    if use_judge and settings.anthropic_api_key is None:
        _out("error: ANTHROPIC_API_KEY is not set (the judge needs it with --write)")
        return 2

    # The key is passed explicitly: a bare Client() reads only the process environment,
    # which a key set in .env never reaches. No upload queue, so a feedback write is a
    # plain POST that raises when it fails.
    client = Client(
        api_key=settings.langsmith_api_key.get_secret_value(),
        auto_batch_tracing=False,
    )
    judge = judges.persona_adherence_judge() if use_judge else None
    return asyncio.run(
        run_online(
            client,
            judge,
            project=args.project,
            limit=args.limit,
            since_hours=args.since_hours,
            write=args.write,
            now=datetime.now(UTC),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
