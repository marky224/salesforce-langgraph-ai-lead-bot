"""
Re-send a lead whose Salesforce write failed, from the thread's checkpoint.

When ``salesforce_node`` fails, the visitor's data stays in the thread's
latest checkpoint: stage ``complete``, ``lead_data``, the score, the summary
and the transcript, but no ``salesforce_task_id`` (a partial write keeps
``salesforce_lead_id``).  The "Salesforce integration failed" log line carries
the thread id, and the failed-write alert in Azure Monitor emails it.

This command reads that state from the durable checkpointer and shows it.
With ``--send`` it replays only the ``salesforce`` node on the thread: the
same Lead reuse, dedup and writes as the live turn, with no LLM call and no
new message.  If the write fails again the command exits 1 with "The write
failed again": that is the signal.  The failure line is logged to this
terminal only, not to the Container Apps logs, so no alert fires.  The error
node also appends its reply to the thread, which the visitor doesn't see.

With LangSmith tracing on (``LANGSMITH_TRACING`` or ``LANGCHAIN_TRACING_V2``),
the replay runs with the server's masked tracer (``app.tracing.build_tracer``),
so emails and phone numbers are masked before any run leaves the process.

Run from ``backend/`` with ``DATABASE_URL`` and the Salesforce settings in the
environment or the root ``.env``::

    ../.venv/bin/python -m app.resend_lead <thread_id>          # show it, write nothing
    ../.venv/bin/python -m app.resend_lead <thread_id> --send   # write it to Salesforce

Exit status: 0 when sent, already saved, or a dry run; 1 for an unknown
thread, a lead not confirmed yet, or a write that failed again; 2 without
``DATABASE_URL``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from app.config import configure_logging, get_settings
from app.graph.checkpointer import open_checkpointer
from app.graph.graph import build_graph
from app.models.schemas import ConversationStage
from app.tracing import build_tracer

_CONTACT_FIELDS = ("first_name", "last_name", "email", "phone", "company", "title")


def _show(values: dict[str, Any], created_at: str | None) -> None:
    """Print what the thread holds (contact details go to this terminal only)."""
    stage = values.get("stage")
    lead_data = values.get("lead_data") or {}
    print(f"last checkpoint: {created_at}")
    print(f"stage: {stage.value if isinstance(stage, ConversationStage) else stage}")
    for field in _CONTACT_FIELDS:
        print(f"{field}: {lead_data.get(field) or '-'}")
    print(f"lead_score: {values.get('lead_score')}")
    print(f"salesforce_lead_id: {values.get('salesforce_lead_id') or '-'}")
    print(f"salesforce_task_id: {values.get('salesforce_task_id') or '-'}")


async def resend(thread_id: str, send: bool) -> int:
    """Show the thread's lead and, if ``send``, write it to Salesforce. Returns the exit status."""
    settings = get_settings()
    if not settings.database_url:
        print("DATABASE_URL is not set, so there are no stored threads to read.", file=sys.stderr)
        return 2

    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    # The server's masked tracer, attached as server._run_config does: with tracing
    # on, emails and phones are masked before any run leaves the process, and
    # LangChain adds no unmasked tracer of its own from the environment.
    tracer = build_tracer(settings)
    if tracer is not None:
        config["callbacks"] = [tracer]
    async with open_checkpointer(settings) as checkpointer:
        graph = build_graph(checkpointer)
        snapshot = await graph.aget_state(config)
        if not snapshot.values:
            print(f"No thread {thread_id}.", file=sys.stderr)
            return 1
        _show(snapshot.values, snapshot.created_at)

        if snapshot.values.get("salesforce_task_id"):
            print("Already saved in Salesforce: nothing to send.")
            return 0
        if snapshot.values.get("stage") != ConversationStage.COMPLETE:
            print("The visitor hasn't confirmed their details: nothing to send.", file=sys.stderr)
            return 1
        if not send:
            print("Dry run: add --send to write this lead to Salesforce.")
            return 0

        # Replay only the salesforce node: record "scoring" as the last step, so
        # its edge (route_after_scoring) runs salesforce next, then resume.
        await graph.aupdate_state(config, None, as_node="scoring")
        await graph.ainvoke(None, config)
        after = (await graph.aget_state(config)).values

    lead_id, task_id = after.get("salesforce_lead_id"), after.get("salesforce_task_id")
    if task_id:
        print(f"Saved: Lead {lead_id}, Task {task_id}.")
        return 0
    print(f"The write failed again (Lead {lead_id or '-'}); see the log above.", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.resend_lead",
        description="Re-send a lead whose Salesforce write failed, from its checkpoint.",
    )
    parser.add_argument("thread_id", help="the thread id from the failed-write alert")
    parser.add_argument(
        "--send", action="store_true", help="write the lead to Salesforce (default: show it only)"
    )
    args = parser.parse_args(argv)
    configure_logging()
    return asyncio.run(resend(args.thread_id, args.send))


if __name__ == "__main__":
    sys.exit(main())
