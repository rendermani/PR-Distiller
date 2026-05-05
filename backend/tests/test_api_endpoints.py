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

# Ensure submodules are importable before patching.
import pipeline.job_orchestrator  # noqa: F401

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
        "llm_model": "ollama/qwen3:8b",
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

    def test_post_config_rejects_unknown_top_level_keys(self):
        """Unknown payload keys must produce a 422 instead of being silently persisted."""
        response = self.client.post(
            "/api/config", json={"llm_model": "x", "totally_made_up_field": "evil"}
        )
        self.assertEqual(response.status_code, 422)

    def test_post_config_strips_redaction_sentinel_for_provider_api_keys(self):
        """'***' values inside provider_api_keys must be filtered before save_config."""
        self.client.post(
            "/api/config",
            json={"provider_api_keys": {"openai": "***", "anthropic": "sk-real"}},
        )
        saved = api.conf_manager.save_config.call_args[0][0]
        # '***' filtered; real key preserved.
        self.assertNotIn("openai", saved.get("provider_api_keys", {}))
        self.assertEqual(saved["provider_api_keys"]["anthropic"], "sk-real")

    def test_get_config_redacts_provider_api_keys(self):
        api.conf_manager = _make_conf_mock(
            config={
                "github_token": "",
                "llm_api_key": "",
                "provider_api_keys": {"openai": "sk-real", "anthropic": ""},
            }
        )
        body = self.client.get("/api/config").json()
        self.assertEqual(body["provider_api_keys"]["openai"], "***")
        self.assertEqual(body["provider_api_keys"]["anthropic"], "")

    def test_get_config_redacts_huggingface_and_webhook_secrets(self):
        api.conf_manager = _make_conf_mock(config={
            "huggingface_token": "hf_real_secret",
            "github_webhook_secret": "wh_real_secret",
        })
        self.client = TestClient(api.app)
        r = self.client.get("/api/config")
        body = r.json()
        self.assertNotEqual(body.get("huggingface_token"), "hf_real_secret")
        self.assertNotEqual(body.get("github_webhook_secret"), "wh_real_secret")
        # Either redaction sentinel or empty string — matching existing convention.
        self.assertIn(body.get("huggingface_token"), ("***", "", None))
        self.assertIn(body.get("github_webhook_secret"), ("***", "", None))

    def test_get_config_requires_auth_when_token_set(self):
        with patch.object(api.settings, "API_AUTH_TOKEN", "secret"):
            r = self.client.get("/api/config")
        self.assertEqual(r.status_code, 401)


class TestHealthEndpoints(unittest.TestCase):
    def setUp(self):
        api.db = _make_db_mock()
        api.conf_manager = _make_conf_mock()
        api.orchestrator = _make_orchestrator_mock()
        self.client = TestClient(api.app)

    @patch("pipeline.github_client.requests.get")
    def test_validate_github_token_valid(self, mock_get):
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"login": "octocat"}
        mock_resp.headers = {"X-OAuth-Scopes": "repo, read:user"}
        mock_get.return_value = mock_resp
        r = self.client.post("/api/github/validate", json={"token": "ghp_abc"})
        body = r.json()
        self.assertTrue(body["valid"])
        self.assertEqual(body["login"], "octocat")
        self.assertIn("repo", body["scopes"])

    @patch("pipeline.github_client.requests.get")
    def test_validate_github_token_401(self, mock_get):
        mock_resp = MagicMock(status_code=401)
        mock_resp.json.return_value = {"message": "Bad credentials"}
        mock_get.return_value = mock_resp
        r = self.client.post("/api/github/validate", json={"token": "ghp_bad"})
        body = r.json()
        self.assertFalse(body["valid"])
        self.assertIn("401", body["error"])

    def test_validate_github_token_empty(self):
        api.conf_manager = _make_conf_mock(config={"github_token": ""})
        r = self.client.post("/api/github/validate", json={"token": ""})
        body = r.json()
        self.assertFalse(body["valid"])
        self.assertIn("No token", body["error"])

    def test_health_endpoint_returns_200_without_touching_db_or_auth(self):
        """The /api/health ping must respond OK regardless of DB state and never require auth."""
        # Make any DB access throw so we'd notice if health touched it.
        api.db.collection.get.side_effect = RuntimeError("db should not be touched")
        with patch.object(api.settings, "API_AUTH_TOKEN", "any-token"):
            r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json().get("status"), "ok")

    @patch("api._requests" if False else "requests.get")  # patch requests.get
    def test_llm_health_reachable(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        r = self.client.get("/api/health/llm")
        body = r.json()
        self.assertTrue(body["reachable"])

    @patch("requests.get")
    def test_llm_health_unreachable(self, mock_get):
        import requests as _r
        mock_get.side_effect = _r.ConnectionError("refused")
        r = self.client.get("/api/health/llm")
        body = r.json()
        self.assertFalse(body["reachable"])
        self.assertIn("Could not reach", body["error"])


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

    def test_delete_rules_for_repo_returns_count_and_calls_db(self):
        api.db.delete_repo_rules.return_value = 7
        response = self.client.delete("/api/rules?repo=acme/app")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "deleted")
        self.assertEqual(body["repo"], "acme/app")
        self.assertEqual(body["removed"], 7)
        api.db.delete_repo_rules.assert_called_once_with("acme/app")

    def test_delete_rules_for_repo_returns_zero_when_no_rules(self):
        api.db.delete_repo_rules.return_value = 0
        response = self.client.delete("/api/rules?repo=acme/empty")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["removed"], 0)


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


class TestConfigEnvOverridesField(unittest.TestCase):
    def setUp(self):
        api.db = _make_db_mock()
        api.conf_manager = _make_conf_mock()
        api.orchestrator = _make_orchestrator_mock()
        self.client = TestClient(api.app)

    def test_get_config_includes_env_overrides(self):
        api.conf_manager.env_overrides.return_value = {
            "github_token": True,
            "huggingface_token": False,
        }
        r = self.client.get("/api/config")
        body = r.json()
        self.assertEqual(body["env_overrides"], {
            "github_token": True,
            "huggingface_token": False,
        })

    def test_get_config_includes_webhook_url(self):
        api.conf_manager.env_overrides.return_value = {}
        with patch.object(api.settings, "WEBHOOK_PUBLIC_URL", "https://example.com"):
            r = self.client.get("/api/config")
        self.assertEqual(r.json()["webhook_url"], "https://example.com/api/webhooks/github")

    def test_webhook_url_falls_back_to_request_origin(self):
        api.conf_manager.env_overrides.return_value = {}
        with patch.object(api.settings, "WEBHOOK_PUBLIC_URL", ""):
            r = self.client.get("/api/config", headers={"host": "fallback.local:8923"})
        self.assertIn("/api/webhooks/github", r.json()["webhook_url"])


if __name__ == "__main__":
    unittest.main()
