"""Tests that ConfigManager round-trips huggingface_token and
github_webhook_secret with Fernet encryption and the same redaction sentinel
behaviour as github_token / provider_api_keys."""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _make_manager(tmp_dir: str):
    from pipeline.config_manager import ConfigManager
    from cryptography.fernet import Fernet
    m = ConfigManager.__new__(ConfigManager)
    m.config_path = os.path.join(tmp_dir, "config.json")
    m.key_path = os.path.join(tmp_dir, ".secret_key")
    m.cipher = Fernet(m._resolve_key())
    m._ensure_default_config()
    return m


class TestHuggingFaceTokenRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.m = _make_manager(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_huggingface_token_persists_encrypted(self):
        self.m.save_config({"huggingface_token": "hf_abc123"})
        loaded = self.m.load_config()
        self.assertEqual(loaded["huggingface_token"], "hf_abc123")
        # On-disk value should be encrypted, not plaintext.
        with open(self.m.config_path) as f:
            raw = f.read()
        self.assertNotIn("hf_abc123", raw)
        self.assertIn("huggingface_token_enc", raw)

    def test_redaction_sentinel_for_huggingface_token_does_not_overwrite(self):
        self.m.save_config({"huggingface_token": "hf_real"})
        self.m.save_config({"huggingface_token": "***"})
        self.assertEqual(self.m.load_config()["huggingface_token"], "hf_real")


class TestWebhookSecretRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.m = _make_manager(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_webhook_secret_persists_encrypted(self):
        self.m.save_config({"github_webhook_secret": "wh_abc"})
        self.assertEqual(self.m.load_config()["github_webhook_secret"], "wh_abc")
        with open(self.m.config_path) as f:
            raw = f.read()
        self.assertNotIn("wh_abc", raw)
        self.assertIn("github_webhook_secret_enc", raw)


if __name__ == "__main__":
    unittest.main()
