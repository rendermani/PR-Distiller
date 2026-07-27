"""POST /api/config must reject bad registry input at the boundary, not 500.

Two gaps:

- The sentinel-restore step indexed stored entries with e["id"], so one legacy
  or hand-edited entry lacking "id" turned any POST carrying a masked
  api_key_override into an unhandled KeyError -> HTTP 500.
- llm_models_active was never cross-checked against llm_models, so a payload
  naming a nonexistent model returned 200 and deferred the failure to job
  runtime, where it surfaced as a per-model preflight error instead of a
  rejected request.
"""
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import settings


def _entry(**overrides):
    base = {
        "id": "good",
        "label": "Good",
        "model": "ollama/qwen3:8b",
        "api_base": "http://localhost:11434/v1",
        "api_key_override": "",
        "enabled": True,
    }
    base.update(overrides)
    return base


def _make_client(tmp_dir: str):
    """A TestClient over a freshly reloaded api module rooted at tmp_dir."""
    from importlib import reload
    from fastapi.testclient import TestClient

    mock_db = MagicMock()
    with patch.object(settings, "DATA_DIR", tmp_dir), \
         patch("db.lightrag_manager.LightRAGManager", return_value=mock_db), \
         patch("pipeline.job_orchestrator.JobOrchestrator", return_value=MagicMock()):
        import api as api_module
        reload(api_module)

    api_module.db.bind(mock_db)
    return TestClient(api_module.app), api_module


class TestActiveIdsAreValidated(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.client, _ = _make_client(self.tmp.name)

    def test_active_id_not_in_registry_is_rejected(self):
        res = self.client.post("/api/config", json={
            "llm_models": [_entry(id="real")],
            "llm_models_active": ["does-not-exist"],
        })

        self.assertEqual(res.status_code, 400)
        self.assertIn("does-not-exist", res.text)

    def test_matching_active_id_is_accepted(self):
        res = self.client.post("/api/config", json={
            "llm_models": [_entry(id="real")],
            "llm_models_active": ["real"],
        })

        self.assertEqual(res.status_code, 200)

    def test_empty_active_list_is_accepted(self):
        res = self.client.post("/api/config", json={
            "llm_models": [_entry(id="real")],
            "llm_models_active": [],
        })

        self.assertEqual(res.status_code, 200)

    def test_duplicate_registry_ids_are_rejected(self):
        res = self.client.post("/api/config", json={
            "llm_models": [_entry(id="dup"), _entry(id="dup", label="Shadow")],
            "llm_models_active": ["dup"],
        })

        self.assertEqual(res.status_code, 400)

    def test_malformed_entry_is_rejected_with_400_not_500(self):
        res = self.client.post("/api/config", json={
            "llm_models": [_entry(model=None)],
        })

        self.assertEqual(res.status_code, 400)


class TestSentinelRestoreSurvivesLegacyEntries(unittest.TestCase):
    """A stored entry without "id" must not crash the sentinel-restore step."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_stored_entry_missing_id_does_not_500(self):
        from pipeline.config_manager import ConfigManager

        # Persist a registry entry with no "id", as a hand-edited or
        # pre-validation config.json could contain.
        with patch.object(settings, "DATA_DIR", self.tmp.name):
            mgr = ConfigManager()
            mgr.save_config({"llm_models": [{"label": "legacy", "model": "ollama/x"}]})

        client, _ = _make_client(self.tmp.name)

        res = client.post("/api/config", json={
            "llm_models": [_entry(api_key_override="***")],
            "llm_models_active": ["good"],
        })

        self.assertNotEqual(res.status_code, 500)
        self.assertEqual(res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
