"""
Tests for ``python -m app.resend_lead``: re-send a failed Salesforce write.

The command reads a thread from the durable checkpointer. Here a ``MemorySaver``
stands in for Neon (``open_checkpointer`` is patched) and the Salesforce client
is mocked, so nothing reaches a database or the network. The LLM raises if it is
called: a re-send makes no LLM call.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app import resend_lead
from app.config import get_settings
from app.graph.graph import build_graph
from app.graph.nodes import set_llm
from app.models.schemas import ConversationStage

THREAD = "thread-with-a-failed-write"
CONFIG = {"configurable": {"thread_id": THREAD}}
LEAD = {"first_name": "Ada", "last_name": "Lovelace", "email": "ada@example.com", "company": "Acme"}


@pytest.fixture
def memory(monkeypatch):
    """A MemorySaver standing in for Neon, with DATABASE_URL set so the command runs."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused.invalid/leadbot")
    get_settings.cache_clear()
    saver = MemorySaver()

    @asynccontextmanager
    async def _open(_settings):
        yield saver

    monkeypatch.setattr(resend_lead, "open_checkpointer", _open)
    llm = AsyncMock()
    llm.ainvoke.side_effect = AssertionError("a re-send must not call the LLM")
    set_llm(llm)
    return saver


@pytest.fixture
def sf():
    """The Salesforce client: a dedup miss, then a Lead and a Task are created."""
    client = MagicMock()
    client.query.return_value = {"records": []}
    client.Lead.create.return_value = {"success": True, "id": "00Q000TEST00001"}
    client.Task.create.return_value = {"success": True, "id": "00T000TEST00001"}
    with patch("app.tools.salesforce._get_sf_client", return_value=client):
        yield client


def _seed(memory: MemorySaver, **values) -> None:
    """Write the thread as a failed confirmation turn leaves it (the error node ran last)."""
    state = {
        "messages": [
            AIMessage(content="Hi, I'm TARS."),
            HumanMessage(content="Yes, that's correct"),
            AIMessage(content="I've got all your details, thank you! I hit a technical hiccup..."),
        ],
        "stage": ConversationStage.COMPLETE,
        "lead_data": LEAD,
        "lead_score": 42,
        "transcript_summary": "Ada from Acme wants a demo.",
        **values,
    }
    build_graph(memory).update_state(CONFIG, state, as_node="error")


def _values(memory: MemorySaver) -> dict:
    return build_graph(memory).get_state(CONFIG).values


def test_dry_run_shows_the_contact_and_writes_nothing(memory, sf, capsys):
    _seed(memory)

    assert resend_lead.main([THREAD]) == 0

    out = capsys.readouterr().out
    assert "email: ada@example.com" in out
    assert "stage: complete" in out
    assert "Dry run" in out
    sf.Lead.create.assert_not_called()
    sf.Task.create.assert_not_called()
    assert "salesforce_task_id" not in _values(memory)


@pytest.mark.asyncio
async def test_send_writes_the_lead_and_task_and_records_their_ids(memory, sf):
    _seed(memory)
    messages_before = len(_values(memory)["messages"])

    assert await resend_lead.resend(THREAD, send=True) == 0

    values = _values(memory)
    assert values["salesforce_lead_id"] == "00Q000TEST00001"
    assert values["salesforce_task_id"] == "00T000TEST00001"
    sf.Lead.create.assert_called_once()
    assert sf.Lead.create.call_args[0][0]["Email"] == "ada@example.com"
    sf.Task.create.assert_called_once()
    assert len(values["messages"]) == messages_before  # no new message in the thread


@pytest.mark.asyncio
async def test_send_after_a_partial_write_adds_only_the_task(memory, sf):
    """The Lead was saved before the Task failed; with no email, only its kept Id stops a second Lead."""
    _seed(memory, lead_data={"company": "Acme"}, salesforce_lead_id="00QKEPT")

    assert await resend_lead.resend(THREAD, send=True) == 0

    sf.Lead.create.assert_not_called()
    assert sf.Task.create.call_args[0][0]["WhoId"] == "00QKEPT"
    assert _values(memory)["salesforce_task_id"] == "00T000TEST00001"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("seed", "status"),
    [
        pytest.param(None, 1, id="unknown-thread"),
        pytest.param(
            {"salesforce_lead_id": "00QSAVED", "salesforce_task_id": "00TSAVED"}, 0, id="already-saved"
        ),
        pytest.param({"stage": ConversationStage.LEAD_CAPTURE}, 1, id="not-confirmed-yet"),
    ],
)
async def test_nothing_to_send(memory, sf, seed, status):
    if seed is not None:
        _seed(memory, **seed)

    assert await resend_lead.resend(THREAD, send=True) == status

    sf.query.assert_not_called()
    sf.Lead.create.assert_not_called()
    sf.Task.create.assert_not_called()


@pytest.mark.asyncio
async def test_refuses_without_database_url(monkeypatch, sf):
    """Without DATABASE_URL the app would read an empty in-memory saver: say so instead."""
    monkeypatch.setenv("DATABASE_URL", "")  # an env value beats any root .env value
    get_settings.cache_clear()
    opened = MagicMock()
    monkeypatch.setattr(resend_lead, "open_checkpointer", opened)

    assert await resend_lead.resend(THREAD, send=True) == 2

    opened.assert_not_called()
    sf.Task.create.assert_not_called()


@pytest.fixture
def recorded_runs(monkeypatch):
    """
    LangSmith tracing exported in the shell, with a fake key, but nothing is sent:
    the tracer's two persist hooks are replaced by recorders that keep each run
    and the client that would have uploaded it.
    """
    from langchain_core.tracers.langchain import LangChainTracer
    from langsmith import utils as ls_utils

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-not-real")
    get_settings.cache_clear()
    ls_utils.get_env_var.cache_clear()  # langsmith caches env reads
    posted: list = []
    patched: list = []
    clients: dict = {}

    def _post(self, run):
        clients[run.id] = self.client
        posted.append((self.client, run))

    def _patch(run):
        patched.append((clients.get(run.id), run))

    monkeypatch.setattr(LangChainTracer, "_persist_run_single", _post)
    monkeypatch.setattr(LangChainTracer, "_update_run_single", staticmethod(_patch))
    yield posted, patched
    ls_utils.get_env_var.cache_clear()  # don't leave "tracing on" cached for later tests


@pytest.mark.asyncio
async def test_send_with_tracing_on_masks_every_run(memory, sf, recorded_runs):
    """
    With LANGSMITH_TRACING exported, every run of the replay goes through the
    server's masked tracer (build_tracer), never LangChain's unmasked env tracer,
    so the stored email and phone never leave the process.
    """
    posted, patched = recorded_runs
    _seed(memory, lead_data={**LEAD, "phone": "210-555-0123"})
    posted.clear()  # the seeding above isn't the command's run; only the replay counts
    patched.clear()

    assert await resend_lead.resend(THREAD, send=True) == 0

    assert "salesforce" in {run.name for _, run in posted}  # the replay was traced
    for client, run in posted:
        assert client._anonymizer is not None, f"unmasked tracer for run {run.name!r}"
        masked = str(client._hide_run_inputs(run.inputs))
        assert "ada@example.com" not in masked and "210-555-0123" not in masked, run.name
    for client, run in patched:
        assert client is not None and client._anonymizer is not None, run.name
        masked = str(client._hide_run_outputs(run.outputs or {}))
        assert "ada@example.com" not in masked and "210-555-0123" not in masked, run.name
