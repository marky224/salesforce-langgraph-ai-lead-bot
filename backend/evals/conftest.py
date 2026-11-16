"""
Pytest configuration for the opt-in eval suite (``backend/evals``).

Kept *local* to this package on purpose: the markers + ``vcr_config`` live here,
not in the root ``pyproject.toml``. There is currently no ``[tool.pytest.ini_options]``
anywhere in the repo, so adding one would move pytest's rootdir for the hermetic
``pytest tests/`` run — an avoidable perturbation. A conftest is only loaded when
``backend/evals`` is actually collected, so it touches nothing else.

Live evals are opt-in: every ``eval_live`` test is skipped unless the run passes
``--run-live`` (``pytest_collection_modifyitems`` below), so a plain ``pytest`` with
real keys in the shell or ``.env`` can't call a provider. ``-m eval_live`` alone
selects them but doesn't opt in. Pytest only knows ``--run-live`` when ``backend/evals``
(or a file in it) is on the command line; without a path the flag is a usage error
and nothing runs.
"""

from __future__ import annotations

import os
import sys

import pytest

# Make ``app`` and ``evals`` importable when run from the repo root or backend/
# (mirrors backend/tests/conftest.py).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evals._config import force_tracing_off  # noqa: E402

# Pin the eval pipeline to what prod runs: xAI provider, JSON-parse extraction.
# LLM_STRUCTURED_OUTPUT and tracing are *forced* off (not setdefault) so a stray
# .env value can't change the request wire format (with_structured_output adds
# tools/response_format -> a different body -> cassette mismatch) or upload traces.
os.environ.setdefault("LLM_PROVIDER", "xai")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ["LLM_STRUCTURED_OUTPUT"] = "false"
force_tracing_off()
# No site-index fetch from evals: tests that need the index set it directly.
os.environ["SITE_INDEX_URL"] = ""


LIVE_SKIP_REASON = "live eval: pass --run-live (calls the real provider and spends tokens)"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add ``--run-live``, the only way to run the ``eval_live`` tests."""
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="run the eval_live tests (they call the real provider and spend tokens)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip every ``eval_live`` test unless the run passes ``--run-live``."""
    if config.getoption("--run-live"):
        return
    skip_live = pytest.mark.skip(reason=LIVE_SKIP_REASON)
    for item in items:
        if item.get_closest_marker("eval_live"):
            item.add_marker(skip_live)


def pytest_configure(config: pytest.Config) -> None:
    """Register markers and, in replay-only mode, supply a throwaway API key."""
    config.addinivalue_line(
        "markers", "eval_recorded: offline VCR-replay eval — the deterministic CI gate."
    )
    config.addinivalue_line(
        "markers",
        "eval_live: hits the real provider — nightly / manual, report-only; skipped unless --run-live.",
    )

    mode = config.getoption("--record-mode", default="none") or "none"
    if mode == "none":
        # Replay intercepts every request before it leaves the process, so no real
        # secret is needed — a dummy lets model construction succeed offline (CI).
        # When recording (mode != none) this is skipped, so the real key from
        # .env / the environment is used instead.
        os.environ.setdefault("XAI_API_KEY", "xai-dummy-for-replay")


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Rebuild Settings per test so env tweaks (e.g. the dummy key) take effect."""
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def vcr_config() -> dict:
    """Shared VCR settings (see evals._config.VCR_RECORD_CONFIG)."""
    from evals._config import VCR_RECORD_CONFIG

    return VCR_RECORD_CONFIG


@pytest.fixture(scope="module")
def vcr_cassette_dir() -> str:
    """
    Flatten cassettes into one ``backend/evals/cassettes/`` dir (the plugin default
    nests a per-module subdir). Combined with ``@pytest.mark.default_cassette("…")``
    per test, this yields stable names (``extraction.yaml`` / ``routing.yaml``) the
    run.py CLI can reference without coupling to pytest test names.
    """
    from evals._config import CASSETTE_DIR

    return str(CASSETTE_DIR)
