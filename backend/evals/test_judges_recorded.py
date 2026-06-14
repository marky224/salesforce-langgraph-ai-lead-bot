"""
Offline plumbing tests for the calibrated LLM judges (``judges.py``) — eval_recorded.

These verify the openevals wiring *constructs* (prompt template accepted, async
evaluator built) with **zero model calls and no API key**:
``create_async_llm_as_judge`` only invokes the judge inside its async closure, so
passing a fake LangChain chat model exercises construction — including embedding the
real PERSONA into the str.format template — entirely offline.  Importing this module
also smoke-tests that the eval-only deps (openevals, langchain) are installed.

The real *graded* calls and human-agreement calibration (Cohen's κ) are ``eval_live``
(``test_judges_live.py``, a later PR D commit).  Mirrors how ``test_scorecard.py``
keeps the deterministic slice on the per-commit gate without a live call.
"""

from __future__ import annotations

import inspect

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from evals.judges import persona_adherence_judge, summary_faithfulness_judge

pytestmark = pytest.mark.eval_recorded


def _fake_judge() -> GenericFakeChatModel:
    """A real BaseChatModel that is never actually invoked (construction is call-free)."""
    return GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))


def test_persona_judge_constructs_async_without_model_call():
    # Builds the persona prompt (embedding PERSONA via str.format) + async evaluator.
    ev = persona_adherence_judge(judge=_fake_judge())
    assert inspect.iscoroutinefunction(ev)


def test_faithfulness_judge_constructs_async_without_model_call():
    ev = summary_faithfulness_judge(judge=_fake_judge())
    assert inspect.iscoroutinefunction(ev)
