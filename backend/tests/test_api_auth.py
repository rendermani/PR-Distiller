"""Tests for the optional API_AUTH_TOKEN bearer-auth gate.

When API_AUTH_TOKEN is unset, all endpoints stay open (backwards-compatible
local-dev default). When set, mutating endpoints (POST, DELETE) require
`Authorization: Bearer <token>`. Public-by-design endpoints (health, webhook,
MCP query) remain unauthenticated.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pipeline.job_orchestrator  # noqa: F401 — ensure submodule import order

with patch("db.lightrag_manager.LightRAGManager"), \
     patch("pipeline.config_manager.ConfigManager"), \
     patch("pipeline.job_orchestrator.JobOrchestrator"):
    import api
    from fastapi.testclient import TestClient


def _reset_singletons():
    api.db = MagicMock()
    api.db.collection = MagicMock()
    api.conf_manager = MagicMock()
    api.conf_manager.load_config.return_value = {}
    api.conf_manager.save_config.side_effect = lambda p: p
    api.orchestrator = MagicMock()
    api.orchestrator.trigger_job.return_value = "job-x"
    api.orchestrator.request_cancel.return_value = True


class TestAuthDisabledByDefault(unittest.TestCase):
    """When API_AUTH_TOKEN is empty, all endpoints accept unauthenticated calls."""

    def setUp(self):
        _reset_singletons()
        self.client = TestClient(api.app)

    def test_post_config_succeeds_without_auth_when_token_unset(self):
        with patch.object(api.settings, "API_AUTH_TOKEN", ""):
            r = self.client.post("/api/config", json={"llm_model": "x"})
        self.assertEqual(r.status_code, 200)

    def test_delete_repo_rules_succeeds_without_auth_when_token_unset(self):
        api.db.delete_repo_rules.return_value = 0
        with patch.object(api.settings, "API_AUTH_TOKEN", ""):
            r = self.client.delete("/api/rules?repo=acme/app")
        self.assertEqual(r.status_code, 200)


class TestAuthEnforced(unittest.TestCase):
    """When API_AUTH_TOKEN is set, mutating endpoints require Bearer token."""

    def setUp(self):
        _reset_singletons()
        self.client = TestClient(api.app)
        self.token = "secret-token-xyz"

    def test_post_config_returns_401_without_auth_header(self):
        with patch.object(api.settings, "API_AUTH_TOKEN", self.token):
            r = self.client.post("/api/config", json={"llm_model": "x"})
        self.assertEqual(r.status_code, 401)

    def test_post_config_returns_401_with_wrong_token(self):
        with patch.object(api.settings, "API_AUTH_TOKEN", self.token):
            r = self.client.post(
                "/api/config",
                json={"llm_model": "x"},
                headers={"Authorization": "Bearer wrong"},
            )
        self.assertEqual(r.status_code, 401)

    def test_post_config_succeeds_with_correct_bearer_token(self):
        with patch.object(api.settings, "API_AUTH_TOKEN", self.token):
            r = self.client.post(
                "/api/config",
                json={"llm_model": "x"},
                headers={"Authorization": f"Bearer {self.token}"},
            )
        self.assertEqual(r.status_code, 200)

    def test_delete_repo_rules_requires_auth(self):
        api.db.delete_repo_rules.return_value = 0
        with patch.object(api.settings, "API_AUTH_TOKEN", self.token):
            r = self.client.delete("/api/rules?repo=acme/app")
        self.assertEqual(r.status_code, 401)

    def test_delete_single_rule_requires_auth(self):
        with patch.object(api.settings, "API_AUTH_TOKEN", self.token):
            r = self.client.delete("/api/rules/some-rule")
        self.assertEqual(r.status_code, 401)

    def test_admin_reindex_requires_auth(self):
        with patch.object(api.settings, "API_AUTH_TOKEN", self.token):
            r = self.client.post("/api/admin/reindex")
        self.assertEqual(r.status_code, 401)

    def test_jobs_start_requires_auth(self):
        with patch.object(api.settings, "API_AUTH_TOKEN", self.token):
            r = self.client.post(
                "/api/jobs/start",
                json={"repo": "acme/app", "months": 2, "threshold": 0.45, "use_cache": False},
            )
        self.assertEqual(r.status_code, 401)


class TestPublicByDesignEndpoints(unittest.TestCase):
    """Health, webhooks, and MCP query stay open even when auth is enabled."""

    def setUp(self):
        _reset_singletons()
        self.client = TestClient(api.app)

    def test_health_llm_does_not_require_auth(self):
        with patch.object(api.settings, "API_AUTH_TOKEN", "any-token"), \
             patch("requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            api.conf_manager.load_config.return_value = {"llm_api_base": "http://localhost:11434/v1"}
            r = self.client.get("/api/health/llm")
        self.assertEqual(r.status_code, 200)

    def test_mcp_query_does_not_require_auth(self):
        api.db.get_contextual_rules.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]]}
        with patch.object(api.settings, "API_AUTH_TOKEN", "any-token"):
            r = self.client.post("/api/mcp/query", json={"code_diff": "x"})
        self.assertEqual(r.status_code, 200)


if __name__ == "__main__":
    unittest.main()
