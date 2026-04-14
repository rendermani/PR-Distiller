"""
Shared pytest fixtures for the PR-Distiller backend test suite.

All fixtures here are available automatically to every test file in this directory.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_db():
    """Mock LightRAGManager for tests that do not need a real DB connection."""
    db = MagicMock()
    db.collection = MagicMock()
    return db
