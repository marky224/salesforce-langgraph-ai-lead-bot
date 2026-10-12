"""
Strict msgpack checkpoint deserialization (GHSA-g48c-2wqr-h844).

langgraph-checkpoint 4.x only blocks unregistered types in stored msgpack when
``LANGGRAPH_STRICT_MSGPACK=true``; by default it revives them with a warning.
In strict mode ``StateGraph.compile()`` builds the allow-list from the state
schema, so ``ConversationStage`` and the LangChain message classes still load.

The variable is read once, at ``langgraph`` import time, so each case runs in a
fresh subprocess with the variable set (or unset) explicitly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]

# Writes a state holding ConversationStage + Human/AI messages through the real
# compiled graph (MemorySaver serializes to msgpack), reads it back, then decodes
# an unregistered enum with the compiled graph's own serializer.
_SCRIPT = textwrap.dedent(
    """
    import asyncio
    import enum
    import json

    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.memory import MemorySaver

    from app.graph.edges import NODE_GREETING
    from app.graph.graph import build_graph
    from app.models.schemas import ConversationStage


    class Unregistered(enum.Enum):
        RED = "red"


    async def main():
        graph = build_graph(MemorySaver())
        config = {"configurable": {"thread_id": "serde-probe"}}
        await graph.aupdate_state(
            config,
            {
                "stage": ConversationStage.QUALIFICATION,
                "messages": [HumanMessage(content="hi"), AIMessage(content="hello")],
            },
            as_node=NODE_GREETING,
        )
        values = (await graph.aget_state(config)).values

        serde = graph.checkpointer.serde
        revived = serde.loads_typed(serde.dumps_typed(Unregistered.RED))
        print(json.dumps({
            "stage_type": type(values["stage"]).__name__,
            "stage_value": values["stage"].value,
            "message_types": [type(m).__name__ for m in values["messages"]],
            "unregistered_revived": isinstance(revived, Unregistered),
        }))


    asyncio.run(main())
    """
)


def _run_probe(strict: bool) -> dict:
    env = dict(os.environ)
    env.pop("LANGGRAPH_STRICT_MSGPACK", None)
    if strict:
        env["LANGGRAPH_STRICT_MSGPACK"] = "true"
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        cwd=BACKEND_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestStrictMsgpack:
    def test_strict_mode_resumes_state_and_blocks_unregistered_types(self):
        out = _run_probe(strict=True)
        assert out["stage_type"] == "ConversationStage"
        assert out["stage_value"] == "qualification"
        assert out["message_types"] == ["HumanMessage", "AIMessage"]
        assert out["unregistered_revived"] is False

    def test_default_mode_revives_unregistered_types(self):
        """Control: without the variable the same payload is revived (the advisory's gap)."""
        out = _run_probe(strict=False)
        assert out["stage_type"] == "ConversationStage"
        assert out["unregistered_revived"] is True

    def test_dockerfile_enables_strict_msgpack(self):
        """The production image must set the variable — it's what closes the advisory."""
        dockerfile = (BACKEND_DIR / "Dockerfile").read_text()
        assert "ENV LANGGRAPH_STRICT_MSGPACK=true" in dockerfile
