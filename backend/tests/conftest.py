"""
Shared pytest configuration and fixtures.
"""

from __future__ import annotations

import os
import sys

import pytest

# Ensure the backend app is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set test environment variables before any app imports
os.environ.setdefault("LLM_PROVIDER", "anthropic")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-not-real")
os.environ.setdefault("LOG_LEVEL", "WARNING")


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear the settings LRU cache between tests."""
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def clear_sf_cache():
    """Clear the Salesforce client cache between tests."""
    try:
        from app.tools.salesforce import reset_sf_client

        reset_sf_client()
    except ImportError:
        pass
    yield
    try:
        from app.tools.salesforce import reset_sf_client

        reset_sf_client()
    except ImportError:
        pass
