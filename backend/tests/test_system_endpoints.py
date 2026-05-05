"""Tests for /api/system/status, /api/system/events, /api/system/embedding/retry."""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pipeline.job_orchestrator  # noqa: F401

with patch("db.lightrag_manager.LightRAGManager"), \
     patch("pipeline.config_manager.ConfigManager"), \
     patch("pipeline.job_orchestrator.JobOrchestrator"):
    import api
    from fastapi.testclient import TestClient


class TestSystemStatusEndpoint(unittest.TestCase):
    def setUp(self):
        api.db = MagicMock()
        api.conf_manager = MagicMock()
        api.orchestrator = MagicMock()
        self.client = TestClient(api.app)

    def test_status_endpoint_returns_initial_idle_state(self):
        from system_status import SystemStatus
        with patch("api.SYSTEM_STATUS", SystemStatus()):
            r = self.client.get("/api/system/status")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["embedding"]["state"], "idle")
        self.assertEqual(body["queued_jobs"], [])

    def test_status_endpoint_does_not_require_auth(self):
        from system_status import SystemStatus
        with patch.object(api.settings, "API_AUTH_TOKEN", "secret-token"), \
             patch("api.SYSTEM_STATUS", SystemStatus()):
            r = self.client.get("/api/system/status")
        self.assertEqual(r.status_code, 200)

    def test_status_endpoint_reflects_singleton_updates(self):
        from system_status import SystemStatus
        s = SystemStatus()
        s.update(state="downloading", model_name="bge-base")
        with patch("api.SYSTEM_STATUS", s):
            r = self.client.get("/api/system/status")
        self.assertEqual(r.json()["embedding"]["state"], "downloading")
        self.assertEqual(r.json()["embedding"]["model_name"], "bge-base")


if __name__ == "__main__":
    unittest.main()
