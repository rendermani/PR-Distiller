"""
Integration tests for api.py FastAPI endpoints.

Strategy:
- Import api.py once with the heavy singletons (LightRAGManager,
  ConfigManager, JobOrchestrator) patched out at class construction time.
- After import, replace api.db, api.conf_manager, and api.orchestrator
  with fresh MagicMock instances for each test method.
- Use FastAPI TestClient to send HTTP requests.

Run with:
    cd /home/mani/Projects/PR-Distiller/backend
    ./venv/bin/python -m pytest tests/test_api_endpoints.py -v --tb=short
"""
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Patch construction-time singletons before the module is imported.
with patch("db.lightrag_manager.LightRAGManager"), \
     patch("pipeline.config_manager.ConfigManager"), \
     patch("pipeline.job_orchestrator.JobOrchestrator"):
    import api
    from fastapi.testclient import TestClient


def _make_db_mock():
    db = MagicMock()
    db.collection = MagicMock()
    return db


def _make_conf_mock(config: dict = None):
    conf = MagicMock()
    conf.load_config.return_value = config or {
        "github_token": "",
        "llm_api_key": "",
        "llm_provider": "ollama",
        "llm_api_base": "http://localhost:11434/v1",
        "llm_model": "ollama/qwen2.5-coder:7b-instruct",
        "embedding_model": "BAAI/bge-base-en-v1.5",
        "repos": {},
        "provider_models": {},
        "crawl_cursors": {},
    }
    conf.save_config.side_effect = lambda p: p
    return conf


def _make_orchestrator_mock(job_id: str = "job-123"):
    orch = MagicMock()
    orch.trigger_job.return_value = job_id
    orch.request_cancel.return_value = True
    orch.get_status.return_value = {"status": "running", "progress": 50}
    return orch


class TestConfigEndpoints(unittest.TestCase):
    def setUp(self):
        api.db = _make_db_mock()
        api.conf_manager = _make_conf_mock()
        api.orchestrator = _make_orchestrator_mock()
        self.client = TestClient(api.app)

    def test_get_config_returns_200(self):
        response = self.client.get("/api/config")
        self.assertEqual(response.status_code, 200)

    def test_get_config_redacts_github_token_when_set(self):
        api.conf_manager = _make_conf_mock(
            config={"github_token": "secret", "llm_api_key": ""}
        )
        response = self.client.get("/api/config")
        self.assertEqual(response.json()["github_token"], "***")

    def test_get_config_redacts_llm_api_key_when_set(self):
        api.conf_manager = _make_conf_mock(
            config={"github_token": "", "llm_api_key": "sk-hidden"}
        )
        response = self.client.get("/api/config")
        self.assertEqual(response.json()["llm_api_key"], "***")

    def test_get_config_shows_empty_string_when_token_absent(self):
        response = self.client.get("/api/config")
        self.assertEqual(response.json()["github_token"], "")

    def test_post_config_returns_200(self):
        response = self.client.post("/api/config", json={"llm_model": "new-model"})
        self.assertEqual(response.status_code, 200)

    def test_post_config_calls_save_config(self):
        self.client.post("/api/config", json={"llm_model": "updated"})
        api.conf_manager.save_config.assert_called_once_with({"llm_model": "updated"})


class TestRulesEndpoints(unittest.TestCase):
    def setUp(self):
        api.db = _make_db_mock()
        api.conf_manager = _make_conf_mock()
        api.orchestrator = _make_orchestrator_mock()
        self.client = TestClient(api.app)

    def test_get_rules_returns_200(self):
        api.db.collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        response = self.client.get("/api/rules")
        self.assertEqual(response.status_code, 200)

    def test_get_rules_returns_empty_list_when_no_rules(self):
        api.db.collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        response = self.client.get("/api/rules")
        self.assertEqual(response.json(), {"rules": []})

    def test_get_rules_returns_rules_sorted_by_occurrence_count(self):
        api.db.collection.get.return_value = {
            "ids": ["r1", "r2"],
            "documents": ["doc1", "doc2"],
            "metadatas": [
                {"occurrence_count": 1, "status": "active"},
                {"occurrence_count": 5, "status": "active"},
            ],
        }
        response = self.client.get("/api/rules")
        rules = response.json()["rules"]
        self.assertEqual(rules[0]["id"], "r2")  # Higher count first.

    def test_post_rules_creates_rule_and_returns_201_or_200(self):
        api.db.add_rule.return_value = "new-rule-id"
        payload = {
            "repo": "acme/app",
            "title": "Use type hints",
            "description": "All functions must have type hints",
            "enforcement": "required",
        }
        response = self.client.post("/api/rules", json=payload)
        self.assertIn(response.status_code, (200, 201))
        self.assertEqual(response.json()["rule_id"], "new-rule-id")

    def test_approve_rule_returns_approved_state(self):
        response = self.client.post("/api/rules/rule-42/approve")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["state"], "approved")
        self.assertEqual(body["rule_id"], "rule-42")
        api.db.approve_rule.assert_called_once_with("rule-42")

    def test_block_rule_returns_blocked_state(self):
        response = self.client.post("/api/rules/rule-99/block")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["state"], "blocked")
        api.db.block_rule.assert_called_once_with("rule-99")

    def test_delete_rule_returns_deleted_status(self):
        response = self.client.delete("/api/rules/rule-delete-me")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "deleted")
        api.db.collection.delete.assert_called_once_with(ids=["rule-delete-me"])


class TestJobEndpoints(unittest.TestCase):
    def setUp(self):
        api.db = _make_db_mock()
        api.conf_manager = _make_conf_mock()
        api.orchestrator = _make_orchestrator_mock(job_id="job-abc")
        self.client = TestClient(api.app)

    def test_start_job_returns_job_id_and_started_status(self):
        payload = {"repo": "acme/app", "months": 3, "threshold": 0.45, "use_cache": False}
        response = self.client.post("/api/jobs/start", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "started")
        self.assertEqual(body["job_id"], "job-abc")

    def test_cancel_job_returns_cancelled_status_when_found(self):
        api.orchestrator.request_cancel.return_value = True
        response = self.client.post("/api/jobs/cancel/job-abc")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "cancelled")

    def test_cancel_job_returns_not_found_when_absent(self):
        api.orchestrator.request_cancel.return_value = False
        response = self.client.post("/api/jobs/cancel/nonexistent")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "not_found")

    def test_get_job_status_returns_status_dict(self):
        api.orchestrator.get_status.return_value = {"status": "running", "progress": 75}
        response = self.client.get("/api/jobs/status/job-abc")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "running")
        self.assertEqual(response.json()["progress"], 75)


class TestExportEndpoint(unittest.TestCase):
    def setUp(self):
        api.db = _make_db_mock()
        api.conf_manager = _make_conf_mock()
        api.orchestrator = _make_orchestrator_mock()
        self.client = TestClient(api.app)

    def test_export_returns_content_string(self):
        api.db.collection.get.return_value = {
            "documents": ["Use type hints everywhere."],
            "metadatas": [{"status": "active"}],
        }
        with patch("pipeline.skill_synthesizer.SkillSynthesizer.export", return_value="# rules") as mock_export:
            response = self.client.get("/api/export?repo=acme/app&type=prompt&arch=openai")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["content"], "# rules")

    def test_export_returns_500_on_synthesizer_exception(self):
        api.db.collection.get.return_value = {
            "documents": ["rule"],
            "metadatas": [{"status": "active"}],
        }
        with patch("pipeline.skill_synthesizer.SkillSynthesizer.export", side_effect=RuntimeError("boom")):
            response = self.client.get("/api/export?repo=acme/app&type=prompt&arch=openai")
        self.assertEqual(response.status_code, 500)


class TestCacheInfoEndpoint(unittest.TestCase):
    def setUp(self):
        api.db = _make_db_mock()
        api.conf_manager = _make_conf_mock()
        api.orchestrator = _make_orchestrator_mock()
        self.client = TestClient(api.app)

    def test_cache_info_returns_cached_false_when_no_cache(self):
        with patch("pipeline.dev_cache.cache_info", return_value=None):
            response = self.client.get("/api/cache/acme/myrepo")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"cached": False})

    def test_cache_info_returns_cached_true_with_metadata(self):
        fake_info = {"repo": "acme/myrepo", "count": 42, "updated_at": "2026-04-13T00:00:00Z"}
        with patch("pipeline.dev_cache.cache_info", return_value=fake_info):
            response = self.client.get("/api/cache/acme/myrepo")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["cached"])
        self.assertEqual(body["count"], 42)
        self.assertEqual(body["repo"], "acme/myrepo")


if __name__ == "__main__":
    unittest.main()
