"""
Unit tests for settings.py — centralized environment variable configuration.

Tests verify:
- Default values are correct when env vars are absent.
- Env var overrides replace defaults correctly.
- Derived paths (CHROMA_DIR, DEV_CACHE_DIR, CONFIG_PATH) are built from DATA_DIR.
- Type conversions: API_PORT is int, float fields are float.

Run with:
    cd /home/mani/Projects/PR-Distiller/backend
    ./venv/bin/python -m pytest tests/test_settings.py -v --tb=short
"""
import sys
import os
import importlib
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import settings as _settings_module


def _reload_with_env(env_overrides: dict):
    """Reload the settings module under the given environment overrides."""
    with patch.dict(os.environ, env_overrides, clear=False):
        return importlib.reload(_settings_module)


class TestSettingsDefaults(unittest.TestCase):
    """Verify every default value matches the documented expectation."""

    def setUp(self):
        # Strip all relevant env vars so defaults are exercised.
        self._env_keys = [
            "LLM_PROVIDER", "LLM_API_BASE", "LLM_API_KEY", "LLM_MODEL",
            "EMBEDDING_MODEL", "API_HOST", "API_PORT", "CORS_ORIGINS",
            "GITHUB_TOKEN", "GITHUB_WEBHOOK_SECRET", "FERNET_KEY",
            "API_AUTH_TOKEN", "DATA_DIR", "DEDUP_DISTANCE_THRESHOLD",
            "AUTO_APPROVE_CONFIDENCE", "WEBHOOK_MIN_INTERVAL",
        ]
        self._original = {k: os.environ.pop(k) for k in self._env_keys if k in os.environ}
        self._settings = importlib.reload(_settings_module)

    def tearDown(self):
        os.environ.update(self._original)
        importlib.reload(_settings_module)

    def test_llm_provider_default(self):
        self.assertEqual(self._settings.LLM_PROVIDER, "ollama")

    def test_llm_api_base_default(self):
        self.assertEqual(self._settings.LLM_API_BASE, "http://localhost:11434/v1")

    def test_llm_api_key_default_is_empty_string(self):
        self.assertEqual(self._settings.LLM_API_KEY, "")

    def test_llm_model_default(self):
        self.assertEqual(self._settings.LLM_MODEL, "ollama/qwen3:8b")

    def test_embedding_model_default(self):
        self.assertEqual(self._settings.EMBEDDING_MODEL, "BAAI/bge-base-en-v1.5")

    def test_api_host_default(self):
        self.assertEqual(self._settings.API_HOST, "0.0.0.0")

    def test_api_port_default_is_int_8923(self):
        self.assertIsInstance(self._settings.API_PORT, int)
        self.assertEqual(self._settings.API_PORT, 8923)

    def test_cors_origins_default_is_list(self):
        self.assertIsInstance(self._settings.CORS_ORIGINS, list)
        self.assertIn("http://localhost:4096", self._settings.CORS_ORIGINS)
        self.assertIn("http://localhost:3000", self._settings.CORS_ORIGINS)

    def test_dedup_distance_threshold_default_is_float(self):
        self.assertIsInstance(self._settings.DEDUP_DISTANCE_THRESHOLD, float)
        self.assertAlmostEqual(self._settings.DEDUP_DISTANCE_THRESHOLD, 0.32)

    def test_auto_approve_confidence_default_is_float(self):
        self.assertIsInstance(self._settings.AUTO_APPROVE_CONFIDENCE, float)
        self.assertAlmostEqual(self._settings.AUTO_APPROVE_CONFIDENCE, 0.8)

    def test_webhook_min_interval_default_is_int(self):
        self.assertIsInstance(self._settings.WEBHOOK_MIN_INTERVAL, int)
        self.assertEqual(self._settings.WEBHOOK_MIN_INTERVAL, 60)

    def test_github_token_default_is_empty_string(self):
        self.assertEqual(self._settings.GITHUB_TOKEN, "")

    def test_fernet_key_default_is_empty_string(self):
        self.assertEqual(self._settings.FERNET_KEY, "")

    def test_api_auth_token_default_is_empty_string(self):
        self.assertEqual(self._settings.API_AUTH_TOKEN, "")


class TestSettingsEnvOverrides(unittest.TestCase):
    """Verify env vars override defaults after module reload."""

    def tearDown(self):
        importlib.reload(_settings_module)

    def test_llm_api_base_override(self):
        s = _reload_with_env({"LLM_API_BASE": "http://myserver:9999/v1"})
        self.assertEqual(s.LLM_API_BASE, "http://myserver:9999/v1")

    def test_api_port_override_is_int(self):
        s = _reload_with_env({"API_PORT": "9000"})
        self.assertIsInstance(s.API_PORT, int)
        self.assertEqual(s.API_PORT, 9000)

    def test_dedup_distance_threshold_override_is_float(self):
        s = _reload_with_env({"DEDUP_DISTANCE_THRESHOLD": "0.55"})
        self.assertAlmostEqual(s.DEDUP_DISTANCE_THRESHOLD, 0.55)

    def test_cors_origins_override_splits_on_comma(self):
        s = _reload_with_env({"CORS_ORIGINS": "http://a.com,http://b.com,http://c.com"})
        self.assertEqual(s.CORS_ORIGINS, ["http://a.com", "http://b.com", "http://c.com"])

    def test_github_token_override(self):
        s = _reload_with_env({"GITHUB_TOKEN": "ghp_test_token"})
        self.assertEqual(s.GITHUB_TOKEN, "ghp_test_token")


class TestSettingsDerivedPaths(unittest.TestCase):
    """Verify CHROMA_DIR, DEV_CACHE_DIR, and CONFIG_PATH derive from DATA_DIR."""

    def tearDown(self):
        importlib.reload(_settings_module)

    def test_chroma_dir_is_under_data_dir(self):
        s = _reload_with_env({"DATA_DIR": "/custom/data"})
        self.assertEqual(s.CHROMA_DIR, "/custom/data/chroma_db")

    def test_dev_cache_dir_is_under_data_dir(self):
        s = _reload_with_env({"DATA_DIR": "/custom/data"})
        self.assertEqual(s.DEV_CACHE_DIR, "/custom/data/dev_cache")

    def test_config_path_is_under_data_dir(self):
        s = _reload_with_env({"DATA_DIR": "/custom/data"})
        self.assertEqual(s.CONFIG_PATH, "/custom/data/config.json")


if __name__ == "__main__":
    unittest.main()
