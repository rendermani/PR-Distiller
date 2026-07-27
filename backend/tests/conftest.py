"""
Shared pytest fixtures for the PR-Distiller backend test suite.

All fixtures here are available automatically to every test file in this directory.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def _short_db_proxy_timeout(monkeypatch):
    """Fail fast when a test reaches an unbound LazyDbProxy.

    The production default waits minutes so a cold embedding-model download is
    not cut short. Under test nothing calls db.bind() unless a test does it
    explicitly, so that default turns a wiring mistake into a multi-minute
    stall per request. One second is long enough for a test that does bind
    (binding is synchronous) and short enough that a test which forgot gets a
    prompt, readable TimeoutError naming the attribute.
    """
    import db_proxy

    monkeypatch.setattr(db_proxy, "DEFAULT_READY_TIMEOUT_SECONDS", 1.0)


@pytest.fixture
def mock_db():
    """Mock LightRAGManager for tests that do not need a real DB connection."""
    db = MagicMock()
    db.collection = MagicMock()
    return db
