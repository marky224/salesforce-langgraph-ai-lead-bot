"""
Eval targets — thin adapters that drive the real graph nodes from a dataset row.

The offline evals call the node functions **directly** (``extraction_node`` /
``router_node``) on a constructed ``GraphState`` rather than running the whole
graph. Both nodes invoke the LLM with ``messages=[]`` — the entire transcript is
baked into the system prompt — so the request body is fully deterministic per row,
which is what makes VCR body-matching stable.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from app.config import LLMProvider, get_llm
from app.graph import nodes
from evals._config import RECORD_MODEL, RECORD_TEMPERATURE


def build_eval_llm(model: str | None = None, temperature: float | None = None) -> Any:
    """
    Build the xAI chat model the evals run against.

    Pinned explicitly (not read from ``LLM_MODEL``) so the request body — and thus
    the cassette — is stable regardless of the developer's ``.env``. The API key
    still comes from settings: the real key from ``.env`` when recording, the
    conftest dummy when replaying.
    """
    return get_llm(
        provider=LLMProvider.XAI,
        model=model or RECORD_MODEL,
        temperature=RECORD_TEMPERATURE if temperature is None else temperature,
    )


def _to_messages(turns: list[dict[str, str]]) -> list:
    """Convert dataset ``{role, content}`` turns into LangChain messages."""
    out: list = []
    for turn in turns:
        if turn["role"] == "user":
            out.append(HumanMessage(content=turn["content"]))
        else:
            out.append(AIMessage(content=turn["content"]))
    return out


async def run_extraction(row: dict[str, Any], llm: Any) -> dict[str, Any]:
    """
    Run ``extraction_node`` on one dataset row and return its state patch.

    ``current_data`` defaults to empty dicts, so the returned ``lead_data`` /
    ``qualification_data`` are exactly the *delta* extracted from the latest
    message — which is what the row's ``expected`` encodes.
    """
    nodes.set_llm(llm)
    current = row.get("current_data") or {}
    state: dict[str, Any] = {
        "messages": _to_messages(row["messages"]),
        "lead_data": current.get("lead_data", {}),
        "qualification_data": current.get("qualification_data", {}),
    }
    return await nodes.extraction_node(state)  # type: ignore[arg-type]
