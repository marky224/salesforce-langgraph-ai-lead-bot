"""
Offline tests for the online eval (``online.py``) — eval_recorded.

No network, no key, no model call. LangSmith is ``FakeLangSmith``, an in-memory object
with the client calls the command makes. The live path (real LangSmith) is the command
itself, run by hand.

Two things keep these tests honest:

- The fake's runs are not hand-written JSON. Real ``HumanMessage`` / ``AIMessage``
  objects and a state dict go through langsmith's own serializer and the tracer's
  anonymizer (``_stored``), the two steps the client applies before an upload.
- Two autouse fixtures. ``no_live_calls`` records any LangSmith request or any build of
  the real judge model and fails the test at teardown (recorded, not only raised: the
  client's ``info`` property swallows the exception). ``no_env_keys`` replaces
  ``app.config.get_settings``, so no test reads a key from the shell or a ``.env``.

Rides the ``eval_recorded`` CI gate, like ``test_scorecard.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langsmith import Client
from langsmith import schemas as ls_schemas
from langsmith import utils as ls_utils
from langsmith._internal._serde import dumps_json

from app import config as app_config
from app.config import Settings
from app.models.schemas import LeadData
from app.tracing import build_email_anonymizer
from evals import judges, online

pytestmark = pytest.mark.eval_recorded

NOW = datetime(2026, 11, 18, 18, 0, tzinfo=UTC)
PROJECT = "test-project"
_TRACING_FLAGS = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING")


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------

def _stored(value: Any, *, masked: bool = True) -> Any:
    """
    What LangSmith holds for a run's inputs or outputs: the client's JSON serializer,
    then the tracer's anonymizer. ``masked=False`` skips the second step, as if the run
    had been uploaded unmasked.
    """
    data = json.loads(dumps_json(value))
    return build_email_anonymizer()(data) if masked else data


class FakeLangSmith:
    """In-memory stand-in for the LangSmith client calls the online eval makes."""

    def __init__(self) -> None:
        self.node_runs: list[ls_schemas.Run] = []
        self.root_runs: dict[str, ls_schemas.Run] = {}
        self.feedback: list[tuple[str, str]] = []  # (run id, key) of every row that exists
        self.created: list[dict[str, Any]] = []  # each create_feedback call, as it was given
        self.write_attempts = 0
        self.list_runs_calls: list[dict[str, Any]] = []
        self.feedback_reads: list[int] = []  # how many trace ids each list_feedback call asked for
        self.read_runs: list[str] = []

    # --- building what the project holds ---

    def add_node_run(
        self,
        trace_id: str,
        node: str,
        *,
        minutes_ago: int = 1,
        langgraph_node: str | None = None,
        tags: tuple[str, ...] = ("tars",),
        error: str | None = None,
        finished: bool = True,
    ) -> None:
        started = NOW - timedelta(minutes=minutes_ago)
        self.node_runs.append(
            ls_schemas.Run(
                id=uuid.uuid4(),
                trace_id=trace_id,
                name=node,
                run_type="chain",
                start_time=started,
                end_time=started + timedelta(seconds=2) if finished else None,
                error=error,
                tags=list(tags),
                extra={"metadata": {"langgraph_node": langgraph_node or node}},
            )
        )

    def add_turn(
        self,
        *,
        node: str = "discovery",
        visitor: str | None = "We lose a day a week to manual reporting.",
        reply: Any = "That adds up fast. Which report hurts most?",
        lead_data: dict[str, Any] | None = None,
        history: tuple[Any, ...] = (),
        masked: bool = True,
        outputs: bool = True,
        root_error: str | None = None,
        **node_run: Any,
    ) -> str:
        """One chat turn as the server traces it: a root run and the node run inside it."""
        trace_id = str(uuid.uuid4())
        asked = [HumanMessage(content=visitor)] if visitor is not None else []
        state = {
            "messages": [*history, *asked, AIMessage(content=reply)],
            "stage": node,
            "lead_data": lead_data or {},
        }
        self.root_runs[trace_id] = ls_schemas.Run(
            id=trace_id,
            trace_id=trace_id,
            name="LangGraph",
            run_type="chain",
            start_time=NOW,
            end_time=NOW,
            error=root_error,
            inputs=_stored({"messages": asked}, masked=masked),
            outputs=_stored(state, masked=masked) if outputs else None,
        )
        self.add_node_run(trace_id, node, **node_run)
        return trace_id

    # --- the client calls ---

    def list_runs(self, **kwargs: Any) -> Any:
        self.list_runs_calls.append(kwargs)
        return iter(self.node_runs[: kwargs["limit"]])

    def list_feedback(self, *, run_ids: list[str], feedback_key: list[str]) -> Any:
        self.feedback_reads.append(len(run_ids))
        return [
            ls_schemas.Feedback(
                id=uuid.uuid4(), created_at=NOW, modified_at=NOW, run_id=run_id, trace_id=run_id, key=key
            )
            for run_id, key in self.feedback
            if run_id in run_ids and key in feedback_key
        ]

    def read_run(self, run_id: str) -> ls_schemas.Run:
        self.read_runs.append(str(run_id))
        return self.root_runs[str(run_id)]

    def create_feedback(self, run_id: Any = None, key: str = "unnamed", **kwargs: Any) -> None:
        self.write_attempts += 1
        self.created.append({"run_id": run_id, "key": key, **kwargs})
        self.feedback.append((str(run_id), key))


def _settings(langsmith: str | None = None, anthropic: str | None = None) -> Settings:
    """Settings that read no ``.env`` file; the two keys come from the arguments only."""
    return Settings(_env_file=None, langsmith_api_key=langsmith, anthropic_api_key=anthropic)


def _run(fake: FakeLangSmith, *, limit: int = 20) -> int:
    return asyncio.run(
        online.run_online(fake, project=PROJECT, limit=limit, since_hours=168, now=NOW)
    )


def _trace_ids(out: str) -> list[str]:
    """The trace ids of the output's ``trace`` lines, in order."""
    return [line.split()[1] for line in out.splitlines() if line.startswith("trace ")]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def no_live_calls(monkeypatch):
    """
    Fail a test that sends a LangSmith request or builds the real judge model.

    Both are recorded and checked at teardown. A raise alone isn't enough: langsmith's
    ``Client.info`` catches ``BaseException`` and only warns.
    """
    calls: list[str] = []

    def _request(self, method, pathname, **_kwargs):
        calls.append(f"LangSmith {method} {pathname}")
        raise ConnectionError("the online eval tests never call LangSmith")

    def _build_judge_llm(*_args, **_kwargs):
        calls.append("judges.build_judge_llm")
        raise RuntimeError("the online eval tests never build the judge model")

    monkeypatch.setattr(Client, "request_with_retries", _request)
    monkeypatch.setattr(judges, "build_judge_llm", _build_judge_llm)
    yield
    assert not calls, f"the test made a live call: {calls}"


@pytest.fixture(autouse=True)
def no_env_keys(monkeypatch):
    """
    No test reads a key from the shell or a ``.env``.

    Settings loads ``../.env`` and ``.env`` relative to the working directory, so an
    "unset" key depends on where pytest was started. Every test gets key-less Settings
    built with no env file; a test of ``main()`` swaps in the keys it needs (``_wire_main``).
    """
    monkeypatch.setattr(app_config, "get_settings", lambda: _settings())


@pytest.fixture
def main_env(monkeypatch):
    """Put back what ``online.main`` changes in the process: env names and a logger level."""
    for flag in _TRACING_FLAGS:
        monkeypatch.setenv(flag, "false")
    monkeypatch.delenv("ANTHROPIC_LOG", raising=False)
    logger = logging.getLogger("langsmith")
    level = logger.level
    yield
    logger.setLevel(level)


class _MainWiring:
    def __init__(self) -> None:
        self.clients: list[dict[str, Any]] = []  # the kwargs of each Client main() built


def _wire_main(
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeLangSmith,
    *,
    langsmith: str | None = "langsmith-test-key",
) -> _MainWiring:
    """Point ``online.main`` at the doubles: Settings with the given key, and the fake."""
    wiring = _MainWiring()

    def _client(**kwargs):
        wiring.clients.append(kwargs)
        return fake

    monkeypatch.setattr(app_config, "get_settings", lambda: _settings(langsmith))
    monkeypatch.setattr(online, "Client", _client)
    return wiring


# ---------------------------------------------------------------------------
# Selecting turns
# ---------------------------------------------------------------------------

def test_selects_only_conversational_turns(capsys):
    fake = FakeLangSmith()
    conversational = ("discovery", "qualification", "objection_handling", "lead_capture", "confirmation")
    eligible = [fake.add_turn(node=node, minutes_ago=i + 1) for i, node in enumerate(conversational)]

    for node in ("greeting", "site_answer", "after_save", "turn_cap", "error", "scoring"):
        fake.add_turn(node=node)
    fake.add_turn(error="RuntimeError('the node failed')")
    fake.add_turn(finished=False)  # a run that is still open
    fake.add_turn(langgraph_node="qualification")  # a run nested inside another node
    fake.add_turn(tags=())  # not the server's run: no tars tag
    two_nodes = fake.add_turn()
    fake.add_node_run(two_nodes, "qualification")
    fake.node_runs.reverse()  # the listing's order must not matter

    assert _run(fake) == 0
    out = capsys.readouterr().out

    assert _trace_ids(out) == eligible  # newest first
    assert (
        f"scanned={len(fake.node_runs)} eligible=5 already_scored=0 selected=5 remaining=0 unparsed=0"
        in out
    )

    (query,) = fake.list_runs_calls
    assert query["project_name"] == PROJECT
    assert query["run_type"] == "chain"
    assert query["start_time"] == NOW - timedelta(hours=168)
    assert query["limit"] == 1000
    assert query["filter"] == (
        'and(has(tags, "tars"), or(eq(name, "discovery"), eq(name, "qualification"), '
        'eq(name, "objection_handling"), eq(name, "lead_capture"), eq(name, "confirmation")))'
    )
    # Every field the local checks read, and the ones langsmith's Run schema requires.
    assert {"id", "name", "run_type", "trace_id", "start_time", "end_time", "error", "extra", "tags"} <= set(
        query["select"]
    )


def test_dry_run_writes_nothing(capsys):
    fake = FakeLangSmith()
    trace_id = fake.add_turn()
    scored = fake.add_turn(minutes_ago=2)
    fake.feedback += [(scored, key) for key in ("markdown_clean", "brevity_ok", "persona_adherence")]

    assert _run(fake) == 0
    out = capsys.readouterr().out

    assert fake.write_attempts == 0
    assert f"online eval: project={PROJECT} window=168h limit=20 mode=dry-run" in out
    assert "eligible=2 already_scored=1 selected=1 remaining=0 unparsed=0" in out
    assert f"trace {trace_id} would_write=markdown_clean,brevity_ok,persona_adherence" in out
    assert fake.read_runs == [trace_id]  # a scored turn's root run isn't read again


def test_limit_caps_the_selected_turns(capsys):
    fake = FakeLangSmith()
    traces = [fake.add_turn(minutes_ago=minutes) for minutes in range(30, 0, -1)]  # oldest first

    assert _run(fake, limit=20) == 0
    out = capsys.readouterr().out

    assert "eligible=30 already_scored=0 selected=20 remaining=10 unparsed=0" in out
    assert _trace_ids(out) == traces[:-21:-1]  # the 20 newest, newest first


def test_feedback_is_read_50_traces_at_a_time():
    fake = FakeLangSmith()
    for minutes in range(120):
        fake.add_turn(minutes_ago=minutes + 1)

    assert _run(fake) == 0

    assert fake.feedback_reads == [50, 50, 20]


def test_the_scan_cap_is_reported(monkeypatch, capsys):
    monkeypatch.setattr(online, "SCAN_CAP", 3)
    fake = FakeLangSmith()
    for minutes in range(5):
        fake.add_turn(minutes_ago=minutes + 1)

    assert _run(fake) == 0
    out = capsys.readouterr().out

    assert "scanned=3 eligible=3" in out
    assert "scan cap reached: the listing stopped at 3 node runs" in out


# ---------------------------------------------------------------------------
# Reading and masking a turn
# ---------------------------------------------------------------------------

def test_reads_the_turn_from_the_stored_run():
    fake = FakeLangSmith()
    earlier = (HumanMessage(content="hi"), AIMessage(content="Hello. What brings you here?"))
    plain = fake.add_turn(
        visitor="We lose a day a week to manual reporting.",
        reply="That adds up fast. Which report hurts most?",
        history=earlier,
    )
    turn = online._parse_turn(plain, fake.root_runs[plain])
    assert turn is not None
    assert turn.visitor == "We lose a day a week to manual reporting."
    assert turn.reply == "That adds up fast. Which report hurts most?"

    blocks = fake.add_turn(
        reply=[
            {"type": "text", "text": "That adds up fast. "},
            {"type": "text", "text": "Which report hurts most?"},
        ]
    )
    turn = online._parse_turn(blocks, fake.root_runs[blocks])
    assert turn is not None
    assert turn.reply == "That adds up fast. Which report hurts most?"


def test_judge_gets_one_masked_turn():
    # LeadData documents the keys of the state's lead_data dict: building it from the
    # model means a wrong key in online.py (there is no "name" field) fails here.
    lead_data = LeadData(
        first_name="Dana",
        last_name="Whitfield",
        email="dana.whitfield@cpplabs.example",
        company="C++ Labs (US)",  # a regex metacharacter, and it ends in a non-word character
        phone="555-0142",  # seven digits: the tracer's phone pattern doesn't match it
        title="VP of Operations",
    ).model_dump()
    earlier = (
        HumanMessage(content="Our forklift bookings are a mess."),
        AIMessage(content="Bookings on paper, or in a spreadsheet?"),
        HumanMessage(content="A spreadsheet nobody trusts."),
        AIMessage(content="Understood. How many warehouses share it?"),
        HumanMessage(content="Three warehouses."),
        AIMessage(content="Got it. Who should the team follow up with?"),
    )
    fake = FakeLangSmith()
    trace_id = fake.add_turn(
        node="lead_capture",
        visitor=(
            "I'm Dana Whitfield, VP of Operations at C++ Labs (US). Reach me at "
            "dana.whitfield@cpplabs.example or 555-0142; my cell is (210) 555-0188 and billing "
            "is ap@cpplabs.example."
        ),
        reply=(
            "Thanks, Dana. I have you as VP of Operations at C++ Labs (US), email "
            "dana.whitfield@cpplabs.example, phone 555-0142. Does that look right, Ms. Whitfield?"
        ),
        lead_data=lead_data,
        history=earlier,
        masked=False,  # as if the tracer's masking had failed at upload
    )
    turn = online._parse_turn(trace_id, fake.root_runs[trace_id])
    assert turn is not None

    inputs, outputs = online._judge_inputs(turn)

    reply = (
        "Thanks, [NAME]. I have you as [TITLE] at [COMPANY], email [EMAIL], phone [PHONE]. "
        "Does that look right, Ms. [NAME]?"
    )
    # The calibration labels' format: one visitor line, one TARS line.
    assert inputs == (
        "Visitor: I'm [NAME], [TITLE] at [COMPANY]. Reach me at [EMAIL] or [PHONE]; "
        "my cell is [PHONE] and billing is [EMAIL].\n"
        f"TARS: {reply}"
    )
    assert outputs == reply
    for earlier_word in ("forklift", "spreadsheet", "warehouses"):
        assert earlier_word not in inputs + outputs
    for value in ("Dana", "Whitfield", "cpplabs", "C++", "555", "VP of Operations"):
        assert value not in inputs + outputs


# ---------------------------------------------------------------------------
# Failures and exit status
# ---------------------------------------------------------------------------

def test_an_unparsed_turn_is_counted_and_exits_1(capsys):
    fake = FakeLangSmith()
    good = fake.add_turn(minutes_ago=1)
    unreadable = [
        fake.add_turn(minutes_ago=2, outputs=False),
        fake.add_turn(minutes_ago=3, visitor=None),
        fake.add_turn(minutes_ago=4, root_error="RuntimeError('the graph failed')"),
    ]

    # Exit 1 although it is a dry run: the stored shape isn't what the command assumes.
    assert _run(fake) == 1
    out = capsys.readouterr().out

    assert "selected=4 remaining=0 unparsed=3" in out
    for trace_id in unreadable:
        assert f"trace {trace_id} unparsed" in out
    assert f"trace {good} would_write=" in out


@pytest.mark.parametrize("call", ["list_runs", "list_feedback", "read_run"])
def test_a_langsmith_error_while_listing_or_reading_exits_2(call, monkeypatch, capsys):
    fake = FakeLangSmith()
    fake.add_turn()

    def _refuse(*_args, **_kwargs):
        raise ls_utils.LangSmithAuthError(
            "Authentication failed for /runs/query. HTTPError('401 Client Error: Unauthorized')"
        )

    monkeypatch.setattr(fake, call, _refuse)

    assert _run(fake) == 2
    out = capsys.readouterr().out

    assert "error: LangSmithAuthError while listing or reading LangSmith; nothing was written" in out
    for text in ("/runs/query", "HTTPError", "Unauthorized"):
        assert text not in out
    assert fake.write_attempts == 0


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def test_main_turns_tracing_off_and_drops_anthropic_log(monkeypatch, main_env):
    for flag in _TRACING_FLAGS:
        monkeypatch.setenv(flag, "true")
    monkeypatch.setenv("ANTHROPIC_LOG", "debug")
    logging.getLogger("langsmith").setLevel(logging.NOTSET)
    fake = FakeLangSmith()
    fake.add_turn()
    _wire_main(monkeypatch, fake)

    ls_utils.get_env_var.cache_clear()  # langsmith caches env reads
    try:
        assert online.main(["--project", PROJECT]) == 0
        assert ls_utils.tracing_is_enabled() is False
    finally:
        ls_utils.get_env_var.cache_clear()  # don't leave this test's env cached for later tests

    assert {name: os.environ.get(name) for name in (*_TRACING_FLAGS, "ANTHROPIC_LOG")} == {
        "LANGSMITH_TRACING": "false",
        "LANGCHAIN_TRACING_V2": "false",
        "LANGSMITH_TRACING_V2": "false",
        "LANGCHAIN_TRACING": "false",
        "ANTHROPIC_LOG": None,
    }
    # The client logs a failed request with its traceback and response body at ERROR.
    assert logging.getLogger("langsmith.client").getEffectiveLevel() > logging.ERROR


def test_missing_keys_exit_2(monkeypatch, main_env, capsys):
    # A key in the shell must not change the result: main() reads Settings, and these
    # Settings are built with no .env file and no key.
    monkeypatch.setenv("LANGSMITH_API_KEY", "a-key-from-the-shell")
    fake = FakeLangSmith()
    wiring = _wire_main(monkeypatch, fake, langsmith=None)

    assert online.main(["--project", PROJECT]) == 2

    assert "error: LANGSMITH_API_KEY is not set" in capsys.readouterr().out
    assert wiring.clients == []  # no client was built, so nothing reached LangSmith


def test_limit_has_a_hard_maximum(monkeypatch, main_env, capsys):
    fake = FakeLangSmith()
    wiring = _wire_main(monkeypatch, fake)

    for refused in ("101", "0"):
        assert online.main(["--project", PROJECT, "--limit", refused]) == 2
        assert "error: --limit must be between 1 and 100" in capsys.readouterr().out
    assert wiring.clients == []  # refused before any LangSmith call
    assert fake.list_runs_calls == []

    assert online.main(["--project", PROJECT, "--limit", "100"]) == 0
    assert "limit=100 mode=dry-run" in capsys.readouterr().out
    assert wiring.clients == [{"api_key": "langsmith-test-key", "auto_batch_tracing": False}]
    assert len(fake.list_runs_calls) == 1
