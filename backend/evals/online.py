"""
Online eval (PR D4) — pick the recent production turns to score, from LangSmith.

``python -m evals.online --project <project>`` lists the recent conversational turns of
one LangSmith project, skips the ones that already carry the online eval's feedback
keys, reads the trace's root run for up to ``--limit`` of the rest, checks that each one
parses and prints counts and trace ids. It is a dry run: it calls no judge and writes
nothing. It is a manual, live command: no pytest gate runs it, and nothing under ``app/``
imports it.

What one turn is
    One turn is one trace: exactly one of the five conversational nodes (discovery,
    qualification, objection_handling, lead_capture, confirmation) ran in it and finished
    without error, it carries the ``tars`` tag, and the node run started inside
    ``--since-hours``. That is the shape the persona judge's kappa was measured on (one
    visitor message and one reply). Greetings, site answers and the fixed replies are
    not turns to judge. Newest first; a turn that already has every key is skipped.

What a judge would receive
    ``_judge_inputs`` builds the two strings for one turn: ``Visitor: <message>\\nTARS:
    <reply>`` and the reply alone. The values captured in that trace's ``lead_data`` are
    replaced first: ``first_name``, ``last_name`` and the two joined with ``[NAME]``,
    ``company`` with ``[COMPANY]``, ``title`` with ``[TITLE]``, and the captured ``email``
    and ``phone`` with ``[EMAIL]`` and ``[PHONE]``. Then the text goes through the
    tracer's email and phone masking again. Nothing else from the trace is used: no
    earlier turns, no lead data, no ids. Best-effort: a name typed in another case or
    misspelled, someone else's name and a phone shape the patterns miss stay in the text.

Keys
    ``LANGSMITH_API_KEY`` through the app's Settings (the environment or ``.env``). An
    org-scoped LangSmith key also needs ``LANGSMITH_WORKSPACE_ID`` exported in the shell:
    the client reads it from the process environment, and a value that is only in
    ``.env`` never reaches it.

Exit status
    0  the turns were selected and every selected one parsed, or there was nothing to do.
    1  a selected turn couldn't be parsed.
    2  a missing key, a ``--limit`` outside 1-100, or a LangSmith list or read call that
       failed (a refused key or project included).

Output
    Standard output only: counts and trace ids. An error prints its exception class,
    never its text, and the ``langsmith`` logger is set above ERROR so the client's own
    error logs (a traceback with the response body, part of the key on a connection
    error) stay off stderr. Nothing the command does is traced.

Run from ``backend/``:  ``python -m evals.online --project <project>``
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

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
from evals import calibration  # noqa: E402
from evals._config import force_tracing_off  # noqa: E402

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


def _out(line: str) -> None:
    print(line)  # noqa: T201


# ---------------------------------------------------------------------------
# Selecting turns
# ---------------------------------------------------------------------------

def _feedback_keys() -> list[str]:
    """The keys the online eval writes, in write order: the code checks, then the judge's."""
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
# The run
# ---------------------------------------------------------------------------

async def run_online(
    client: Any,
    *,
    project: str,
    limit: int,
    since_hours: int,
    now: datetime,
) -> int:
    """
    Select and read the turns of ``project``; return the exit status.

    ``client`` is a LangSmith client; it is an argument so the offline tests pass a
    double. Exit status: the module docstring.
    """
    keys = _feedback_keys()
    _out(f"online eval: project={project} window={since_hours}h limit={limit} mode=dry-run")

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

    for trace_id, turn in turns:
        if turn is None:
            _out(f"trace {trace_id} unparsed")
        else:
            _out(f"trace {trace_id} would_write={','.join(missing[trace_id])}")
    return 1 if unparsed else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evals.online",
        description="List the recent conversational turns of a LangSmith project that the "
                    "online eval would score. A dry run: nothing is judged or written.",
    )
    parser.add_argument(
        "--project", required=True,
        help="LangSmith project to read turns from (no default).",
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help=f"Most turns one run selects (default {DEFAULT_LIMIT}, maximum {MAX_LIMIT}).",
    )
    parser.add_argument(
        "--since-hours", type=int, default=DEFAULT_SINCE_HOURS,
        help=f"How far back to look for turns (default {DEFAULT_SINCE_HOURS}).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not 1 <= args.limit <= MAX_LIMIT:
        _out(f"error: --limit must be between 1 and {MAX_LIMIT}")
        return 2

    # Nothing here is traced, whatever the shell or .env says. The Anthropic SDK's own
    # log switch goes too: at debug the SDK prints whole requests, and a judge request
    # holds a visitor's message.
    force_tracing_off()
    os.environ.pop("ANTHROPIC_LOG", None)
    # langsmith logs a failed request itself: a traceback with the URL and response body,
    # and on a connection error the first 5 and last 2 characters of the key.
    logging.getLogger("langsmith").setLevel(logging.CRITICAL)

    settings = app_config.get_settings()
    if settings.langsmith_api_key is None:
        _out("error: LANGSMITH_API_KEY is not set")
        return 2

    # The key is passed explicitly: a bare Client() reads only the process environment,
    # which a key set in .env never reaches. No upload queue: every call is a plain
    # request that raises when it fails.
    client = Client(
        api_key=settings.langsmith_api_key.get_secret_value(),
        auto_batch_tracing=False,
    )
    return asyncio.run(
        run_online(
            client,
            project=args.project,
            limit=args.limit,
            since_hours=args.since_hours,
            now=datetime.now(UTC),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
