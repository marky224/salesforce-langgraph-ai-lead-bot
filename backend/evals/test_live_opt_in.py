"""
Offline check that the live evals are opt-in (``conftest.pytest_collection_modifyitems``):
an ``eval_live`` test is skipped unless the run passes ``--run-live``, so a plain
``pytest`` with real keys in the shell or ``.env`` can't call a provider and spend tokens.

No network, no LLM: each case copies this package's real ``conftest.py`` next to a
stub ``eval_live`` test in a temp dir and runs pytest on that dir in a subprocess.
Rides the ``eval_recorded`` CI gate, like ``test_scorecard.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

pytestmark = pytest.mark.eval_recorded

_EVALS_DIR = Path(__file__).resolve().parent
_SKIP_LINE = "SKIPPED*live eval: pass --run-live (calls the real provider and spends tokens)"

_STUB_TESTS = """
import pytest


@pytest.mark.eval_live
def test_live_stub():
    pass


def test_offline_stub():
    pass
"""


@pytest.fixture
def stub_suite(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> pytest.Pytester:
    """A temp ``suite/`` dir: the real evals conftest, one stub live test, one offline test."""
    suite = pytester.mkdir("suite")
    (suite / "conftest.py").write_text((_EVALS_DIR / "conftest.py").read_text())
    (suite / "test_stub.py").write_text(_STUB_TESTS)
    # The conftest imports ``evals._config`` and ``app.config``; both live under backend/.
    monkeypatch.setenv("PYTHONPATH", str(_EVALS_DIR.parent))
    return pytester


def test_plain_run_skips_live_tests(stub_suite: pytest.Pytester):
    """No path, no marker, no flag (a plain ``pytest`` from a parent dir): live is skipped."""
    result = stub_suite.runpytest_subprocess("-rs", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=1, skipped=1)
    result.stdout.fnmatch_lines([_SKIP_LINE])


def test_marker_alone_does_not_opt_in(stub_suite: pytest.Pytester):
    """``-m eval_live`` selects the live tests but doesn't run them: only the flag does."""
    result = stub_suite.runpytest_subprocess("-m", "eval_live", "-rs", "-p", "no:cacheprovider")
    result.assert_outcomes(skipped=1, deselected=1)
    result.stdout.fnmatch_lines([_SKIP_LINE])


def test_run_live_flag_runs_live_tests(stub_suite: pytest.Pytester):
    """``--run-live`` with the suite's path runs the live tests unskipped."""
    result = stub_suite.runpytest_subprocess(
        "suite", "-m", "eval_live", "--run-live", "-p", "no:cacheprovider"
    )
    result.assert_outcomes(passed=1, deselected=1)
