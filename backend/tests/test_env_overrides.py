"""ConfigManager.env_overrides reports which secret fields are set via env vars."""
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _make_manager(tmp_dir):
    from pipeline.config_manager import ConfigManager
    from cryptography.fernet import Fernet
    m = ConfigManager.__new__(ConfigManager)
    m.config_path = os.path.join(tmp_dir, "config.json")
    m.key_path = os.path.join(tmp_dir, ".secret_key")
    m.cipher = Fernet(m._resolve_key())
    m._ensure_default_config()
    return m


class TestEnvOverrides(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.m = _make_manager(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_no_env_vars_set_returns_all_false(self):
        with patch("settings.GITHUB_TOKEN", ""), \
             patch("settings.HUGGINGFACE_HUB_TOKEN", ""), \
             patch("settings.OPENAI_API_KEY", ""), \
             patch("settings.ANTHROPIC_API_KEY", ""), \
             patch("settings.GOOGLE_API_KEY", ""), \
             patch("settings.OPENROUTER_API_KEY", ""), \
             patch("settings.GITHUB_WEBHOOK_SECRET", ""):
            self.assertEqual(
                self.m.env_overrides(),
                {
                    "github_token": False,
                    "huggingface_token": False,
                    "github_webhook_secret": False,
                    "provider_api_keys.openai": False,
                    "provider_api_keys.anthropic": False,
                    "provider_api_keys.google": False,
                    "provider_api_keys.openrouter": False,
                },
            )

    def test_huggingface_env_var_marks_true(self):
        with patch("settings.HUGGINGFACE_HUB_TOKEN", "hf_xxx"):
            self.assertTrue(self.m.env_overrides()["huggingface_token"])

    def test_provider_api_key_env_var_marks_per_provider(self):
        with patch("settings.OPENAI_API_KEY", "sk-x"), \
             patch("settings.ANTHROPIC_API_KEY", ""):
            o = self.m.env_overrides()
            self.assertTrue(o["provider_api_keys.openai"])
            self.assertFalse(o["provider_api_keys.anthropic"])


if __name__ == "__main__":
    unittest.main()
