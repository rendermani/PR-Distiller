"""POST /api/config must redact secrets in its response, as GET already does.

save_config returns the merged config and api.py returned it verbatim, so every
Settings or repo save echoed the plaintext GitHub token (and every other stored
secret) back to the browser. GET redacted them to "***", so the leak was easy to
miss: the same field was safe on read and unsafe on write.

Exposure: browser devtools, any logging proxy, and anything that records
response bodies.
"""
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import settings

SENTINEL = "***"
FAKE_GITHUB_TOKEN = "ghp_FAKEfakeFAKEfake0123456789abcdef"
FAKE_HF_TOKEN = "hf_FAKEfakeFAKEfake0123456789"
FAKE_PROVIDER_KEY = "sk-FAKEfake0123456789"


def _make_client(tmp_dir: str):
    from importlib import reload
    from fastapi.testclient import TestClient

    mock_db = MagicMock()
    with patch.object(settings, "DATA_DIR", tmp_dir), \
         patch("db.lightrag_manager.LightRAGManager", return_value=mock_db), \
         patch("pipeline.job_orchestrator.JobOrchestrator", return_value=MagicMock()):
        import api as api_module
        reload(api_module)

    api_module.db.bind(mock_db)
    return TestClient(api_module.app)


class TestPostConfigRedactsSecrets(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.client = _make_client(self.tmp.name)
        # Seed real-looking secrets through the API.
        res = self.client.post("/api/config", json={
            "github_token": FAKE_GITHUB_TOKEN,
            "huggingface_token": FAKE_HF_TOKEN,
            "provider_api_keys": {"openai": FAKE_PROVIDER_KEY},
        })
        self.assertEqual(res.status_code, 200)

    def test_post_response_does_not_contain_github_token(self):
        res = self.client.post("/api/config", json={"embedding_model": "BAAI/bge-base-en-v1.5"})

        self.assertEqual(res.status_code, 200)
        self.assertNotIn(FAKE_GITHUB_TOKEN, res.text)
        self.assertEqual(res.json().get("github_token"), SENTINEL)

    def test_post_response_does_not_contain_huggingface_token(self):
        res = self.client.post("/api/config", json={"embedding_model": "BAAI/bge-base-en-v1.5"})

        self.assertNotIn(FAKE_HF_TOKEN, res.text)

    def test_post_response_does_not_contain_provider_api_keys(self):
        res = self.client.post("/api/config", json={"embedding_model": "BAAI/bge-base-en-v1.5"})

        self.assertNotIn(FAKE_PROVIDER_KEY, res.text)

    def test_repo_save_does_not_leak_the_token(self):
        """The exact flow that leaked: adding a repo from the UI."""
        res = self.client.post("/api/config", json={
            "repos": {"owner/repo": {"months": 4, "threshold": 0.45}},
        })

        self.assertEqual(res.status_code, 200)
        self.assertNotIn(FAKE_GITHUB_TOKEN, res.text)

    def test_post_and_get_agree_on_redaction(self):
        post_token = self.client.post("/api/config", json={"embedding_model": "x"}).json().get("github_token")
        get_token = self.client.get("/api/config").json().get("github_token")

        self.assertEqual(post_token, get_token)

    def test_secrets_are_still_stored_despite_redacted_response(self):
        """Redaction must not have dropped the value on the way in."""
        self.client.post("/api/config", json={"repos": {"a/b": {"months": 1, "threshold": 0.4}}})

        from pipeline.config_manager import ConfigManager
        with patch.object(settings, "DATA_DIR", self.tmp.name):
            stored = ConfigManager().load_config()

        self.assertEqual(stored["github_token"], FAKE_GITHUB_TOKEN)

    def test_api_key_override_is_not_echoed(self):
        self.client.post("/api/config", json={
            "llm_models": [{
                "id": "m", "label": "M", "model": "openai/gpt-4o",
                "api_base": "https://api.openai.com/v1",
                "api_key_override": FAKE_PROVIDER_KEY, "enabled": True,
            }],
            "llm_models_active": ["m"],
        })

        res = self.client.post("/api/config", json={"embedding_model": "y"})

        self.assertNotIn(FAKE_PROVIDER_KEY, res.text)


if __name__ == "__main__":
    unittest.main()
