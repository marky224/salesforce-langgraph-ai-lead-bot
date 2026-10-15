"""
Strict msgpack checkpoint deserialization (GHSA-g48c-2wqr-h844).

langgraph-checkpoint 4.x only blocks unregistered types in stored msgpack when
``LANGGRAPH_STRICT_MSGPACK=true``; by default it revives them with a warning.
In strict mode ``StateGraph.compile()`` builds the allow-list from the state
schema, so ``ConversationStage`` and the LangChain message classes still load.

The variable is read once, at ``langgraph`` import time, so each case runs in a
fresh subprocess. The strict case takes its environment from the ``ENV`` lines of
the Dockerfile's runtime stage (after the last ``FROM``), so it fails unless the
production image really sets the variable. The parser forgets ``ENV`` at every
``FROM``; if the runtime stage is ever built ``FROM`` an earlier stage, Docker keeps
that stage's ``ENV`` but the parser does not, so the test can only fail as a false
alarm, never pass by mistake.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import textwrap
from importlib import metadata
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
DOCKERFILE = BACKEND_DIR / "Dockerfile"

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


def _runtime_stage_env(dockerfile: Path) -> dict[str, str]:
    """
    Return the variables set by ``ENV`` in the Dockerfile's final (runtime) stage.

    Comment lines are skipped and ``\\`` continuations joined; both
    ``ENV KEY=value ...`` and the legacy ``ENV KEY value`` form are handled.

    ``ENV`` set before the last ``FROM`` is dropped. That matches Docker only for
    ``FROM <image>``: if the runtime stage is ever built from an earlier stage (e.g.
    ``FROM builder AS runtime``), Docker keeps that stage's ``ENV`` but this parser
    drops it. The strict test then fails as a false alarm; it can never pass by
    mistake.
    """
    instructions: list[str] = []
    pending = ""
    for raw in dockerfile.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        instructions.append(pending + line)
        pending = ""

    env: dict[str, str] = {}
    for instruction in instructions:
        keyword, _, args = instruction.partition(" ")
        if keyword.upper() == "FROM":
            env = {}
        elif keyword.upper() == "ENV":
            tokens = shlex.split(args)
            if tokens and "=" in tokens[0]:
                env.update(token.split("=", 1) for token in tokens)
            elif tokens:
                env[tokens[0]] = " ".join(tokens[1:])
    return env


def _run_probe(extra_env: dict[str, str]) -> dict:
    env = dict(os.environ)
    env.pop("LANGGRAPH_STRICT_MSGPACK", None)
    env.update(extra_env)
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
    def test_image_env_resumes_state_and_blocks_unregistered_types(self):
        """Under the production image's ENV, app state resumes and unknown types are blocked."""
        out = _run_probe(_runtime_stage_env(DOCKERFILE))
        assert out["stage_type"] == "ConversationStage"
        assert out["stage_value"] == "qualification"
        assert out["message_types"] == ["HumanMessage", "AIMessage"]
        assert out["unregistered_revived"] is False, (
            "an unregistered type was revived under the image's ENV, so strict msgpack "
            "is off. Likely causes: the Dockerfile runtime stage no longer sets "
            "LANGGRAPH_STRICT_MSGPACK=true, or the installed langgraph-checkpoint "
            f"({metadata.version('langgraph-checkpoint')}) is older than 4.x."
        )

    def test_default_mode_revives_unregistered_types(self):
        """Control: without the variable the same payload is revived (the advisory's gap)."""
        out = _run_probe({})
        assert out["stage_type"] == "ConversationStage"
        assert out["unregistered_revived"] is True


class TestRuntimeStageEnv:
    """The Dockerfile parser only counts live ENV lines in the final stage."""

    def _parse(self, tmp_path: Path, body: str) -> dict[str, str]:
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text(textwrap.dedent(body))
        return _runtime_stage_env(dockerfile)

    def test_both_env_forms_in_runtime_stage(self, tmp_path):
        env = self._parse(tmp_path, """
            FROM python:3.12-slim AS builder
            FROM python:3.12-slim
            ENV A=1 B="two words"
            ENV C three words
            """)
        assert env == {"A": "1", "B": "two words", "C": "three words"}

    def test_commented_and_builder_stage_env_are_ignored(self, tmp_path):
        env = self._parse(tmp_path, """
            FROM python:3.12-slim AS builder
            ENV LANGGRAPH_STRICT_MSGPACK=true
            FROM python:3.12-slim
            # ENV LANGGRAPH_STRICT_MSGPACK=true
            """)
        assert env == {}
