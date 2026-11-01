"""
Offline check that the eval suite's tracing switch (``_config.force_tracing_off``)
beats every env name LangSmith traces on from, so no eval run is uploaded.

No network, no LLM: it sets env vars and asks LangSmith whether it would trace.
Rides the ``eval_recorded`` CI gate, like ``test_scorecard.py``.
"""

from __future__ import annotations

import pytest

from evals._config import force_tracing_off

pytestmark = pytest.mark.eval_recorded

_TRACING_FLAGS = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING")


@pytest.mark.parametrize("flag", _TRACING_FLAGS)
def test_force_tracing_off_beats_every_flag(monkeypatch, flag):
    """A tracing name exported as "true" in the shell is off once the helper runs."""
    from langsmith import utils as ls_utils

    for name in _TRACING_FLAGS:
        monkeypatch.delenv(name, raising=False)  # restored after the test
    monkeypatch.setenv(flag, "true")
    force_tracing_off()
    ls_utils.get_env_var.cache_clear()  # langsmith caches env reads
    try:
        assert ls_utils.tracing_is_enabled() is False
    finally:
        ls_utils.get_env_var.cache_clear()  # don't leave this test's env cached for later tests
