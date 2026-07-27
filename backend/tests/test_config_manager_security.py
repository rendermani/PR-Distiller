"""Tests for ConfigManager security and correctness behaviour.

Covers:
- FERNET_KEY env override and helpful error on malformed value.
- save_config refuses to persist the redaction sentinel '***'.
- provider_api_keys merge across calls preserves keys for unspecified providers.
"""
import os
import sys
import shutil
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _make_manager(tmp_dir: str):
    from pipeline.config_manager import ConfigManager
    from cryptography.fernet import Fernet

    manager = ConfigManager.__new__(ConfigManager)
    manager.config_path = os.path.join(tmp_dir, "config.json")
    manager.key_path = os.path.join(tmp_dir, ".secret_key")
    manager.cipher = Fernet(manager._resolve_key())
    manager._ensure_default_config()
    return manager


class TestFernetKeyResolution(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_malformed_fernet_key_raises_helpful_error(self):
        """A malformed FERNET_KEY env var must produce an operator-friendly error."""
        from pipeline.config_manager import ConfigManager
        with patch("settings.FERNET_KEY", "not-a-valid-fernet-key"):
            manager = ConfigManager.__new__(ConfigManager)
            manager.config_path = os.path.join(self.tmp_dir, "config.json")
            manager.key_path = os.path.join(self.tmp_dir, ".secret_key")
            with self.assertRaises(ValueError) as ctx:
                manager._build_cipher(manager._resolve_key())
            self.assertIn("FERNET_KEY", str(ctx.exception))


class TestSaveConfigSentinelFilter(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.manager = _make_manager(self.tmp_dir)
        self.manager.save_config({
            "github_token": "real-gh-token",
            "llm_provider": "openai",
            "provider_api_keys": {"openai": "sk-real"},
        })

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save_config_does_not_persist_redaction_sentinel_for_github_token(self):
        """If '***' arrives for github_token, the existing real token must be preserved."""
        self.manager.save_config({"github_token": "***"})
        self.assertEqual(self.manager.load_config()["github_token"], "real-gh-token")

    def test_save_config_does_not_persist_sentinel_in_provider_api_keys(self):
        """'***' values inside provider_api_keys must be dropped, real keys preserved."""
        self.manager.save_config({"provider_api_keys": {"openai": "***"}})
        self.assertEqual(
            self.manager.load_config()["provider_api_keys"]["openai"], "sk-real"
        )

class TestProviderApiKeysMerge(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.manager = _make_manager(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_switching_provider_preserves_other_provider_keys(self):
        """Saving only the active provider's key must not wipe keys for others."""
        self.manager.save_config({
            "llm_provider": "openai",
            "provider_api_keys": {"openai": "sk-openai"},
        })
        self.manager.save_config({
            "llm_provider": "anthropic",
            "provider_api_keys": {"anthropic": "sk-anthropic"},
        })
        loaded = self.manager.load_config()
        self.assertEqual(loaded["provider_api_keys"]["openai"], "sk-openai")
        self.assertEqual(loaded["provider_api_keys"]["anthropic"], "sk-anthropic")


if __name__ == "__main__":
    unittest.main()
