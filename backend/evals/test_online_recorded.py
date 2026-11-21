"""
Offline tests for the online eval (``online.py``) — eval_recorded.

No network, no key, no model call. LangSmith is ``FakeLangSmith``, an in-memory object
with the four client calls the command makes, and the judge is ``StubJudge``, which
returns a canned result and records what it was given. The live path (real LangSmith,
real judge) is the command itself, run by hand.

Two things keep these tests honest:

- The fake's runs are not hand-written JSON. Real ``HumanMessage`` / ``AIMessage``
  objects and a state dict go through langsmith's own serializer and the tracer's
  anonymizer (``_stored``), the two steps the client applies before an upload.
- Two autouse fixtures. ``no_live_calls`` records any LangSmith request or any build of
  the real judge model and fails the test at teardown (recorded, not only raised: the
  client's ``info`` property swallows the exception). ``no_env_keys`` replaces
  ``app.config.get_settings``, so no test reads a key from the shell or a ``.env``.

One test goes a layer down: ``test_a_write_that_keeps_failing_is_tried_3_times`` writes
through a real client whose HTTP session is replaced, so langsmith's own retry loop runs
(and still nothing leaves the process).

Rides the ``eval_recorded`` CI gate, like ``test_scorecard.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import requests
from langchain_core.messages import AIMessage, HumanMessage
from langsmith import Client
from langsmith import schemas as ls_schemas
from langsmith import utils as ls_utils
from langsmith._internal._serde import dumps_json

from app import config as app_config
from app.config import Settings
from app.models.schemas import LeadData
from app.tracing import build_email_anonymizer
from evals import calibration, judges, online

pytestmark = pytest.mark.eval_recorded

NOW = datetime(2026, 11, 18, 18, 0, tzinfo=UTC)
PROJECT = "test-project"
ALL_KEYS = ["markdown_clean", "brevity_ok", "persona_adherence"]
_TRACING_FLAGS = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING")

MARKDOWN_COMMENT = (
    "evals.online: scorecard.markdown_leak on TARS's reply. 1 = no markdown, 0 = markdown found."
)
BREVITY_COMMENT = (
    "evals.online: scorecard.brevity_ok on TARS's reply. 1 = at most 5 sentences, 0 = more."
)
# langsmith's retry loop, kept here because ``no_live_calls`` replaces it in every test.
_REQUEST_WITH_RETRIES = Client.request_with_retries


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
    """In-memory stand-in for the four LangSmith client calls the command makes."""

    def __init__(self) -> None:
        self.node_runs: list[ls_schemas.Run] = []
        self.root_runs: dict[str, ls_schemas.Run] = {}
        self.feedback: list[tuple[str, str]] = []  # (run id, key) of every row that exists
        self.created: list[dict[str, Any]] = []  # each create_feedback call, as it was given
        self.write_attempts = 0
        self.write_error: Exception | None = None
        self.list_runs_calls: list[dict[str, Any]] = []
        self.runs_read = 0  # how many node runs the command took from a listing
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
        return self._one_at_a_time()

    def _one_at_a_time(self) -> Any:
        """Every node run, handed over as it is asked for: the client pages lazily too."""
        for run in self.node_runs:
            self.runs_read += 1
            yield run

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
        if self.write_error is not None:
            raise self.write_error
        self.created.append({"run_id": run_id, "key": key, **kwargs})
        self.feedback.append((str(run_id), key))


class StubJudge:
    """The persona evaluator's stand-in: canned results, and a record of what it was given."""

    def __init__(self, score: float = 1.0, comment: str = "On-voice.", errors: tuple[Exception, ...] = ()) -> None:
        self.score = score
        self.comment = comment
        self.errors = list(errors)  # raised by the next calls, in order
        self.calls: list[dict[str, str]] = []

    async def __call__(self, *, inputs: str, outputs: str) -> dict[str, Any]:
        self.calls.append({"inputs": inputs, "outputs": outputs})
        if self.errors:
            raise self.errors.pop(0)
        return {"key": "persona_adherence", "score": self.score, "comment": self.comment}


def _settings(langsmith: str | None = None, anthropic: str | None = None) -> Settings:
    """Settings that read no ``.env`` file; the two keys come from the arguments only."""
    return Settings(_env_file=None, langsmith_api_key=langsmith, anthropic_api_key=anthropic)


def _run(fake: FakeLangSmith, judge: Any = None, *, write: bool = False, limit: int = 20) -> int:
    return asyncio.run(
        online.run_online(
            fake, judge, project=PROJECT, limit=limit, since_hours=168, write=write, now=NOW
        )
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
        self.judge_builds = 0


def _wire_main(
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeLangSmith,
    judge: Any = None,
    *,
    langsmith: str | None = "langsmith-test-key",
    anthropic: str | None = "anthropic-test-key",
) -> _MainWiring:
    """Point ``online.main`` at the doubles: Settings with the given keys, the fake, the stub."""
    wiring = _MainWiring()

    def _client(**kwargs):
        wiring.clients.append(kwargs)
        return fake

    def _judge():
        wiring.judge_builds += 1
        return judge

    monkeypatch.setattr(app_config, "get_settings", lambda: _settings(langsmith, anthropic))
    monkeypatch.setattr(online, "Client", _client)
    monkeypatch.setattr(judges, "persona_adherence_judge", _judge)
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
    assert query["filter"] == (
        'and(has(tags, "tars"), or(eq(name, "discovery"), eq(name, "qualification"), '
        'eq(name, "objection_handling"), eq(name, "lead_capture"), eq(name, "confirmation")))'
    )
    # Every field the local checks read, and the ones langsmith's Run schema requires.
    assert {"id", "name", "run_type", "trace_id", "start_time", "end_time", "error", "extra", "tags"} <= set(
        query["select"]
    )


def test_dry_run_calls_no_judge_and_writes_nothing(capsys):
    fake, judge = FakeLangSmith(), StubJudge()
    trace_id = fake.add_turn()
    scored = fake.add_turn(minutes_ago=2)
    fake.feedback += [(scored, key) for key in ALL_KEYS]

    assert _run(fake, judge) == 0
    out = capsys.readouterr().out

    assert judge.calls == []
    assert fake.write_attempts == 0
    assert f"online eval: project={PROJECT} window=168h limit=20 mode=dry-run" in out
    assert "eligible=2 already_scored=1 selected=1 remaining=0 unparsed=0" in out
    assert f"trace {trace_id} would_write=markdown_clean,brevity_ok,persona_adherence" in out
    assert fake.read_runs == [trace_id]  # a scored turn's root run isn't read again
    assert "judge:" not in out
    assert "written=" not in out


def test_limit_caps_the_judge_calls(capsys):
    fake, judge = FakeLangSmith(), StubJudge()
    traces = [fake.add_turn(minutes_ago=minutes) for minutes in range(30, 0, -1)]  # oldest first

    assert _run(fake, judge, write=True, limit=20) == 0
    out = capsys.readouterr().out

    assert len(judge.calls) == 20
    assert "eligible=30 already_scored=0 selected=20 remaining=10 unparsed=0" in out
    assert _trace_ids(out) == traces[:-21:-1]  # the 20 newest, newest first
    assert {row["run_id"] for row in fake.created} == set(traces[-20:])
    assert "written=60 rows on 20 traces" in out


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
    assert fake.runs_read == 3  # the command stops reading there: the fake held 5


def test_the_listing_sends_no_limit():
    # LangSmith answers 400 to a /runs/query limit over 100 and the client sends the
    # argument as it is, so the command sends none and applies the scan cap itself.
    fake = FakeLangSmith()
    fake.add_turn()

    assert _run(fake) == 0

    (query,) = fake.list_runs_calls
    assert "limit" not in query


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


# Test 7's turns. Each is stored unmasked, as if the tracer's masking had failed at upload:
# (lead_data, the visitor's message, TARS's reply, the two as the judge must get them,
# and text that must not reach the judge in any form).
_NOTED = "Noted. What should the team look at first?"
_UNMASKED_TURNS = [
    pytest.param(
        {
            "first_name": "Dana",
            "last_name": "Whitfield",
            "email": "dana.whitfield@cpplabs.example",
            "company": "C++ Labs (US)",  # a regex metacharacter, and it ends in a non-word character
            "phone": "555-0142",  # seven digits: the tracer's phone pattern doesn't match it
            "title": "VP of Operations",
        },
        "I'm Dana Whitfield, VP of Operations at C++ Labs (US). Reach me at "
        "dana.whitfield@cpplabs.example or 555-0142; my cell is (210) 555-0188 and billing "
        "is ap@cpplabs.example.",
        "Thanks, Dana. I have you as VP of Operations at C++ Labs (US), email "
        "dana.whitfield@cpplabs.example, phone 555-0142. Does that look right, Ms. Whitfield?",
        "I'm [NAME], [TITLE] at [COMPANY]. Reach me at [EMAIL] or [PHONE]; "
        "my cell is [PHONE] and billing is [EMAIL].",
        "Thanks, [NAME]. I have you as [TITLE] at [COMPANY], email [EMAIL], phone [PHONE]. "
        "Does that look right, Ms. [NAME]?",
        ("Dana", "Whitfield", "cpplabs", "C++", "555", "VP of Operations"),
        id="every-captured-value",
    ),
    # A captured value inside another email, or at the end of a longer number. The whole
    # email or number must become its placeholder: with the lead values replaced first,
    # a placeholder next to "@" or after the area code stops the tracer's patterns and
    # the rest gets through ("cc [NAME]@kestrelfreight.io", "(210) [PHONE]").
    pytest.param(
        {"first_name": "dana"},  # stored as the visitor typed it
        "i'm dana. cc dana@kestrelfreight.io on it",
        _NOTED,
        "i'm [NAME]. cc [EMAIL] on it",
        _NOTED,
        ("dana", "kestrelfreight"),
        id="first-name-in-another-email",
    ),
    pytest.param(
        {"company": "Kestrel"},
        "We're Kestrel. Quotes go to sales@Kestrel.io",
        _NOTED,
        "We're [COMPANY]. Quotes go to [EMAIL]",
        _NOTED,
        ("Kestrel", "sales"),
        id="company-in-another-email",
    ),
    pytest.param(
        {"last_name": "whitfield"},
        "whitfield here, whitfield@cpplabs.example",
        _NOTED,
        "[NAME] here, [EMAIL]",
        _NOTED,
        ("whitfield", "cpplabs"),
        id="last-name-in-another-email",
    ),
    pytest.param(
        {"phone": "555-0188"},
        "Call 555-0188, or (210) 555-0188 from outside the city",
        _NOTED,
        "Call [PHONE], or [PHONE] from outside the city",
        _NOTED,
        ("210", "555", "0188"),
        id="captured-phone-inside-a-longer-number",
    ),
    # A captured phone that the tracer's pattern matches on its own but skips in the text,
    # right after a ".": only the value as stored still finds it there (the tracer's
    # masking turns the captured value itself into "[PHONE]"). No digit may get through.
    pytest.param(
        {"phone": "210-555-0188"},
        "Ph.210-555-0188 is my cell",
        _NOTED,
        "Ph.[PHONE] is my cell",
        _NOTED,
        ("0", "1", "2", "5", "8"),
        id="captured-phone-right-after-a-dot",
    ),
    pytest.param(
        {"phone": "2105550188"},
        "tel.2105550188 any time",
        _NOTED,
        "tel.[PHONE] any time",
        _NOTED,
        ("0", "1", "2", "5", "8"),
        id="captured-phone-digits-right-after-a-dot",
    ),
    # A captured first name that starts a captured email the pattern only half matches
    # (it stops at the "ñ" and leaves "ines.muñ[EMAIL]"). Both forms of the captured
    # values go in one pass, longest first: replacing the name first would cut the
    # remainder off the email and let ".muñ" through.
    pytest.param(
        {"first_name": "ines", "email": "ines.muñoz@cpplabs.example"},
        "ines here, ines.muñoz@cpplabs.example",
        _NOTED,
        "[NAME] here, [EMAIL]",
        _NOTED,
        ("ines", "muñ", "cpplabs"),
        id="captured-name-starts-a-half-matched-email",
    ),
    # A half-matched captured email that the visitor typed capitalised. The tracer's
    # masking leaves "Ines.Muñ[EMAIL]" in the text, which only the title-case form put
    # through that same masking equals: the title case of the already rewritten value is
    # "Ines.Muñ[Email]" and matches nothing.
    pytest.param(
        {"email": "ines.muñoz@cpplabs.example"},
        "Send it to Ines.Muñoz@Cpplabs.Example please",
        _NOTED,
        "Send it to [EMAIL] please",
        _NOTED,
        ("Ines", "Muñ", "Cpplabs"),
        id="half-matched-email-typed-in-title-case",
    ),
    pytest.param(
        {"email": "dána@cpplabs.example"},
        "Send it to Dána@Cpplabs.Example please",
        _NOTED,
        "Send it to [EMAIL] please",
        _NOTED,
        ("Dá", "Cpplabs"),
        id="tail-matched-email-typed-in-title-case",
    ),
]


@pytest.mark.parametrize(
    ("lead", "visitor", "reply", "visitor_sent", "reply_sent", "never_sent"), _UNMASKED_TURNS
)
def test_judge_gets_one_masked_turn(lead, visitor, reply, visitor_sent, reply_sent, never_sent, capsys):
    # LeadData documents the keys of the state's lead_data dict: building it from the
    # model means a wrong key in online.py (there is no "name" field) fails here.
    lead_data = LeadData(**lead).model_dump()
    earlier = (
        HumanMessage(content="Our forklift bookings are a mess."),
        AIMessage(content="Bookings on paper, or in a spreadsheet?"),
        HumanMessage(content="A spreadsheet nobody trusts."),
        AIMessage(content="Understood. How many warehouses share it?"),
        HumanMessage(content="Three warehouses."),
        AIMessage(content="Got it. Who should the team follow up with?"),
    )
    fake, judge = FakeLangSmith(), StubJudge()
    fake.add_turn(
        node="lead_capture",
        visitor=visitor,
        reply=reply,
        lead_data=lead_data,
        history=earlier,
        masked=False,  # as if the tracer's masking had failed at upload
    )

    assert _run(fake, judge, write=True) == 0
    out = capsys.readouterr().out

    assert judge.calls == [
        {
            # The calibration labels' format: one visitor line, one TARS line.
            "inputs": f"Visitor: {visitor_sent}\nTARS: {reply_sent}",
            "outputs": reply_sent,
        }
    ]
    sent = judge.calls[0]["inputs"] + judge.calls[0]["outputs"]
    for earlier_word in ("forklift", "spreadsheet", "warehouses"):
        assert earlier_word not in sent
    for value in never_sent:
        assert value not in sent
    # The output holds ids and numbers only. (Digit strings aren't checked here: a trace
    # id is random hex and could hold one.)
    for value in (*never_sent, "[NAME]"):
        assert value.isdigit() or value not in out


def test_a_name_stored_in_lowercase_is_masked_in_title_case_too():
    # Extraction may keep a value as the visitor typed it; TARS then writes it capitalised.
    fake, judge = FakeLangSmith(), StubJudge()
    fake.add_turn(
        node="lead_capture",
        visitor="it's dana whitfield from kestrel freight",
        reply="Thanks, Dana. Is Kestrel Freight the registered name, Ms. Whitfield?",
        lead_data=LeadData(first_name="dana", last_name="whitfield", company="kestrel freight").model_dump(),
    )

    assert _run(fake, judge, write=True) == 0

    reply = "Thanks, [NAME]. Is [COMPANY] the registered name, Ms. [NAME]?"
    assert judge.calls == [
        {"inputs": f"Visitor: it's [NAME] from [COMPANY]\nTARS: {reply}", "outputs": reply}
    ]


@pytest.mark.parametrize("masked", [True, False], ids=["stored-masked", "stored-unmasked"])
@pytest.mark.parametrize(
    "email",
    [
        "andré@cpplabs.example",  # the pattern finds nothing: no ASCII character before the "@"
        "dána@cpplabs.example",  # the pattern finds only "na@cpplabs.example"
    ],
    ids=["pattern-finds-nothing", "pattern-finds-the-tail"],
)
def test_a_captured_email_the_pattern_misses_is_masked_whole(email, masked):
    # The tracer's email pattern is ASCII-only, so the captured value is the only cover
    # for an address like these. A masked upload stores the second one as "dá[EMAIL]",
    # in the text and in lead_data alike.
    fake, judge = FakeLangSmith(), StubJudge()
    fake.add_turn(
        node="lead_capture",
        visitor=f"Send it to {email} please.",
        reply=f"Will do: {email}.",
        lead_data=LeadData(email=email).model_dump(),
        masked=masked,
    )

    assert _run(fake, judge, write=True) == 0

    assert judge.calls == [
        {
            "inputs": "Visitor: Send it to [EMAIL] please.\nTARS: Will do: [EMAIL].",
            "outputs": "Will do: [EMAIL].",
        }
    ]


# ---------------------------------------------------------------------------
# Writing feedback
# ---------------------------------------------------------------------------

def test_second_run_writes_nothing(capsys):
    fake, judge = FakeLangSmith(), StubJudge()
    for minutes in (1, 2, 3):
        fake.add_turn(minutes_ago=minutes)

    assert _run(fake, judge, write=True) == 0
    first = capsys.readouterr().out
    assert "written=9 rows on 3 traces" in first
    assert (len(judge.calls), len(fake.created), len(fake.read_runs)) == (3, 9, 3)

    assert _run(fake, judge, write=True) == 0
    second = capsys.readouterr().out
    assert (len(judge.calls), len(fake.created), len(fake.read_runs)) == (3, 9, 3)
    assert "eligible=3 already_scored=3 selected=0 remaining=0 unparsed=0" in second
    assert "written=0 rows on 0 traces" in second


def test_writes_only_the_missing_keys(capsys):
    fake, judge = FakeLangSmith(), StubJudge(score=0.5)
    trace_id = fake.add_turn()
    fake.feedback.append((trace_id, "markdown_clean"))  # an earlier run wrote this row, then died

    assert _run(fake, judge, write=True) == 0
    out = capsys.readouterr().out

    assert [row["key"] for row in fake.created] == ["brevity_ok", "persona_adherence"]
    assert f"trace {trace_id} brevity_ok=1 persona_adherence=0.5 written=2" in out


def test_feedback_holds_no_visitor_text_and_no_judge_reasoning(capsys):
    fake = FakeLangSmith()
    judge = StubJudge(
        score=0.0,
        comment="The visitor said forklift scheduling is chaos; TARS mocked the night shift: sarcasm.",
    )
    fake.add_turn(
        visitor="Our forklift scheduling is chaos and the night shift keeps double-booking.",
        reply="Sounds like your night shift can't count. **Bold** strategy.",
    )

    assert _run(fake, judge, write=True) == 0
    out = capsys.readouterr().out

    kappa = calibration.CALIBRATED_KAPPA["persona_adherence"]
    assert {row["key"]: row["comment"] for row in fake.created} == {
        "markdown_clean": MARKDOWN_COMMENT,
        "brevity_ok": BREVITY_COMMENT,
        "persona_adherence": (
            f"evals.online: {judges.JUDGE_MODEL} judged one visitor message and TARS's reply "
            "against the TARS persona. 1 = on-voice, 0.5 = slightly off, 0 = off-persona. "
            f"Judge kappa {kappa:.2f} against human labels (bar 0.6, validated)."
        ),
    }
    assert {row["key"]: row["score"] for row in fake.created} == {
        "markdown_clean": 0,  # the reply has **bold**
        "brevity_ok": 1,
        "persona_adherence": 0.0,
    }
    for row in fake.created:  # no value, correction or extra: nothing free-form is sent
        assert set(row) == {
            "run_id", "key", "score", "comment", "feedback_source_type", "source_info", "stop_after_attempt",
        }
    everything = json.dumps(fake.created, default=str) + out
    for words in ("forklift", "night shift", "double-booking", "mocked", "sarcasm", "Bold"):
        assert words not in everything


def test_persona_key_needs_the_recorded_kappa(monkeypatch, capsys):
    fake, judge = FakeLangSmith(), StubJudge()
    fake.add_turn()

    under_the_bar = {"persona_adherence": calibration.KAPPA_BAR - 0.01}
    monkeypatch.setattr(calibration, "CALIBRATED_KAPPA", under_the_bar)
    assert _run(fake, judge, write=True) == 0
    out = capsys.readouterr().out
    assert judge.calls == []
    assert [row["key"] for row in fake.created] == ["markdown_clean", "brevity_ok"]
    assert "judge: not run (persona_adherence is under the kappa bar 0.6)" in out

    # Exactly at the bar the judge counts, and the same trace now lacks its row.
    monkeypatch.setattr(calibration, "CALIBRATED_KAPPA", {"persona_adherence": calibration.KAPPA_BAR})
    assert _run(fake, judge, write=True) == 0
    assert len(judge.calls) == 1
    assert [row["key"] for row in fake.created] == ALL_KEYS
    assert fake.created[-1]["source_info"] == {
        "evaluator": "evals.online",
        "judge_model": judges.JUDGE_MODEL,
        "kappa": calibration.KAPPA_BAR,
        "kappa_bar": calibration.KAPPA_BAR,
        "validated": True,
    }


def test_a_judge_error_is_retried_once(capsys):
    fake = FakeLangSmith()
    judge = StubJudge(errors=(KeyError("score"),))
    fake.add_turn()

    assert _run(fake, judge, write=True) == 0
    out = capsys.readouterr().out

    assert len(judge.calls) == 2
    assert [row["key"] for row in fake.created] == ALL_KEYS
    assert "calls=2 errors=1" in out


def test_a_judge_error_keeps_the_checks_and_exits_1(capsys):
    visitor = "We need the Kestrel rollout fixed before the audit."
    fake = FakeLangSmith()
    judge = StubJudge(
        errors=(ValueError(f"could not parse: {visitor}"), ValueError(f"could not parse: {visitor}"))
    )
    trace_id = fake.add_turn(visitor=visitor)

    assert _run(fake, judge, write=True) == 1
    out = capsys.readouterr().out

    assert len(judge.calls) == 2  # the call and its one retry
    assert [row["key"] for row in fake.created] == ["markdown_clean", "brevity_ok"]
    assert (
        f"trace {trace_id} markdown_clean=1 brevity_ok=1 persona_adherence=error(ValueError) written=2"
        in out
    )
    assert "calls=2 errors=2" in out
    assert "Kestrel" not in out
    assert "could not parse" not in out

    # The next run retries the judge for that turn and writes only its row.
    assert _run(fake, judge, write=True) == 0
    assert len(judge.calls) == 3
    assert [row["key"] for row in fake.created] == ALL_KEYS


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("POST /feedback failed: HTTPError('the response body')"),
        ls_utils.LangSmithAuthError("Authentication failed for /feedback. HTTPError('401 Unauthorized')"),
        ls_utils.LangSmithError("Failed to POST /feedback in LangSmith API. HTTPError('403 Forbidden')"),
    ],
    ids=["any-error", "refused-401", "refused-403"],
)
def test_a_failed_write_stops_before_the_judge(error, capsys):
    fake, judge = FakeLangSmith(), StubJudge()
    newest = fake.add_turn(minutes_ago=1)
    fake.add_turn(minutes_ago=2)
    fake.write_error = error

    # A refused write is a failed write: exit 1, not the exit 2 of a refused listing.
    assert _run(fake, judge, write=True) == 1
    out = capsys.readouterr().out

    assert judge.calls == []
    assert fake.write_attempts == 1  # the first cheap row; the run stopped there
    assert f"error: {type(error).__name__} on a feedback write for trace {newest}; run stopped" in out
    assert "written=0 rows on 0 traces" in out
    # Words, not status digits: a trace id is random hex and could hold "401".
    for text in ("/feedback", "HTTPError", "Unauthorized", "Forbidden", "response body"):
        assert text not in out


@pytest.mark.parametrize(
    ("status", "error", "longest_wait"),
    [(500, "LangSmithAPIError", 5), (429, "LangSmithRateLimitError", 95)],
    ids=["server-error", "rate-limit"],
)
def test_a_write_that_keeps_failing_is_tried_3_times(status, error, longest_wait, monkeypatch, capsys):
    # langsmith's own retry loop runs here, on a real client whose HTTP session and sleeps
    # are replaced. With the client's default of 10 attempts the same write sleeps for
    # about 8.5 minutes before it raises, and for hours on a rate limit.
    fake, judge = FakeLangSmith(), StubJudge()
    newest = fake.add_turn(minutes_ago=1)
    fake.add_turn(minutes_ago=2)
    posts: list[str] = []
    waits: list[float] = []

    def _answer(method, url, **_kwargs):
        posts.append(f"{method} {url}")
        response = requests.Response()
        response.status_code = status
        response._content = b'{"detail": "the response body"}'
        return response

    client = Client(
        api_key="langsmith-test-key",
        api_url="https://langsmith.invalid",  # resolves nowhere, should the session stub ever be missed
        auto_batch_tracing=False,
        info=ls_schemas.LangSmithInfo(),  # given here, or the client asks the server for it
    )
    monkeypatch.setattr(client.session, "request", _answer)
    monkeypatch.setattr(Client, "request_with_retries", _REQUEST_WITH_RETRIES)
    monkeypatch.setattr(time, "sleep", waits.append)
    monkeypatch.setattr(fake, "create_feedback", client.create_feedback)

    assert _run(fake, judge, write=True) == 1
    out = capsys.readouterr().out

    assert posts == ["POST https://langsmith.invalid/feedback"] * 3
    assert len(waits) == 2
    assert sum(waits) < longest_wait  # seconds
    assert judge.calls == []
    assert f"error: {error} on a feedback write for trace {newest}; run stopped" in out
    assert "written=0 rows on 0 traces" in out
    assert "the response body" not in out


def test_feedback_goes_to_the_root_run_synchronously(monkeypatch, main_env):
    fake, judge = FakeLangSmith(), StubJudge()
    trace_id = fake.add_turn()

    assert _run(fake, judge, write=True) == 0

    # run_id is the trace id (the root run's id) and there is no trace_id argument: with
    # one, a client that has an upload queue only queues the row and a failure can't raise.
    assert [row["run_id"] for row in fake.created] == [trace_id] * 3
    assert all("trace_id" not in row for row in fake.created)
    assert [(row["key"], row["feedback_source_type"]) for row in fake.created] == [
        ("markdown_clean", "api"),
        ("brevity_ok", "api"),
        ("persona_adherence", "model"),
    ]
    assert fake.created[0]["source_info"] == {"evaluator": "evals.online", "check": "markdown_leak"}
    assert fake.created[1]["source_info"] == {"evaluator": "evals.online", "check": "brevity_ok"}

    # And the client main() builds has no upload queue at all.
    built: dict[str, Any] = {}

    async def _capture(client, judge, **_kwargs):
        built["client"] = client
        return 0

    monkeypatch.setattr(app_config, "get_settings", lambda: _settings(langsmith="langsmith-test-key"))
    monkeypatch.setattr(online, "run_online", _capture)
    assert online.main(["--project", PROJECT]) == 0
    client = built["client"]
    assert isinstance(client, Client)
    assert client.tracing_queue is None
    assert client.compressed_traces is None
    assert client.api_key == "langsmith-test-key"  # from Settings, passed explicitly


# ---------------------------------------------------------------------------
# Failures and exit status
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("write", [False, True], ids=["dry-run", "write"])
def test_an_unparsed_turn_is_counted_and_exits_1(write, capsys):
    fake, judge = FakeLangSmith(), StubJudge()
    good = fake.add_turn(minutes_ago=1)
    unreadable = [
        fake.add_turn(minutes_ago=2, outputs=False),
        fake.add_turn(minutes_ago=3, visitor=None),
        fake.add_turn(minutes_ago=4, root_error="RuntimeError('the graph failed')"),
    ]

    # Exit 1 in a dry run too: the stored shape isn't what the command assumes.
    assert _run(fake, judge, write=write) == 1
    out = capsys.readouterr().out

    assert "selected=4 remaining=0 unparsed=3" in out
    for trace_id in unreadable:
        assert f"trace {trace_id} unparsed" in out
    assert {row["run_id"] for row in fake.created} == ({good} if write else set())
    assert len(judge.calls) == (1 if write else 0)


@pytest.mark.parametrize("call", ["list_runs", "list_feedback", "read_run"])
def test_a_langsmith_error_while_listing_or_reading_exits_2(call, monkeypatch, capsys):
    fake, judge = FakeLangSmith(), StubJudge()
    fake.add_turn()

    def _refuse(*_args, **_kwargs):
        raise ls_utils.LangSmithAuthError(
            "Authentication failed for /runs/query. HTTPError('401 Client Error: Unauthorized')"
        )

    monkeypatch.setattr(fake, call, _refuse)

    assert _run(fake, judge, write=True) == 2
    out = capsys.readouterr().out

    assert "error: LangSmithAuthError while listing or reading LangSmith; nothing was written" in out
    for text in ("/runs/query", "HTTPError", "Unauthorized"):
        assert text not in out
    assert judge.calls == []
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

    at_judge_build: dict[str, Any] = {}

    def _build_judge():
        # What the process looks like at the moment the judge model would be built.
        at_judge_build.update({name: os.environ.get(name) for name in (*_TRACING_FLAGS, "ANTHROPIC_LOG")})
        at_judge_build["langsmith would trace"] = ls_utils.tracing_is_enabled()
        return StubJudge()

    monkeypatch.setattr(judges, "persona_adherence_judge", _build_judge)
    ls_utils.get_env_var.cache_clear()  # langsmith caches env reads
    try:
        assert online.main(["--project", PROJECT, "--write"]) == 0
    finally:
        ls_utils.get_env_var.cache_clear()  # don't leave this test's env cached for later tests

    assert at_judge_build == {
        "LANGSMITH_TRACING": "false",
        "LANGCHAIN_TRACING_V2": "false",
        "LANGSMITH_TRACING_V2": "false",
        "LANGCHAIN_TRACING": "false",
        "ANTHROPIC_LOG": None,
        "langsmith would trace": False,
    }
    # The client logs a failed request with its traceback and response body at ERROR.
    assert logging.getLogger("langsmith.client").getEffectiveLevel() > logging.ERROR


@pytest.mark.parametrize(
    ("extra_args", "keys", "missing"),
    [
        ([], {}, "LANGSMITH_API_KEY"),
        (["--write"], {"langsmith": "langsmith-test-key"}, "ANTHROPIC_API_KEY"),
    ],
    ids=["no-langsmith-key", "no-anthropic-key-with-write"],
)
def test_missing_keys_exit_2(extra_args, keys, missing, monkeypatch, main_env, capsys):
    # Keys in the shell must not change the result: main() reads Settings, and these
    # Settings are built with no .env file and the two keys given here.
    monkeypatch.setenv("LANGSMITH_API_KEY", "a-key-from-the-shell")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-key-from-the-shell")
    fake = FakeLangSmith()
    wiring = _wire_main(monkeypatch, fake, StubJudge(), langsmith=keys.get("langsmith"), anthropic=None)

    assert online.main(["--project", PROJECT, *extra_args]) == 2

    assert f"error: {missing} is not set" in capsys.readouterr().out
    assert wiring.clients == []  # no client was built, so nothing reached LangSmith
    assert wiring.judge_builds == 0


def test_limit_has_a_hard_maximum(monkeypatch, main_env, capsys):
    fake = FakeLangSmith()
    wiring = _wire_main(monkeypatch, fake, anthropic=None)  # a dry run needs no judge key

    for refused in ("101", "0"):
        assert online.main(["--project", PROJECT, "--limit", refused]) == 2
        assert "error: --limit must be between 1 and 100" in capsys.readouterr().out
    assert wiring.clients == []  # refused before any LangSmith call
    assert fake.list_runs_calls == []

    assert online.main(["--project", PROJECT, "--limit", "100"]) == 0
    assert "limit=100 mode=dry-run" in capsys.readouterr().out
    assert wiring.clients == [{"api_key": "langsmith-test-key", "auto_batch_tracing": False}]
    assert len(fake.list_runs_calls) == 1
