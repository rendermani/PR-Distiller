"""An empty llm_models registry is seeded from the LLM_* env vars.

The multi-model refactor made `llm_models` the source of truth for inference
config but shipped no backfill, so every install predating it loaded with an
empty registry: jobs had no model to run and /api/health/llm reported
"No active LLM models configured" without ever probing the server.

Seeding happens at config-creation time rather than as a read-time fallback:
a fallback inside the health probe would mask a genuinely empty registry.
"""
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import settings
from pipeline.model_registry import validate_entry


def _fresh_manager(tmp_dir: str):
    """Build a ConfigManager rooted at tmp_dir, exercising real init."""
    from pipeline.config_manager import ConfigManager

    with patch.object(settings, "DATA_DIR", tmp_dir):
        return ConfigManager()


class TestRegistrySeededFromEnv(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_default_config_seeds_one_model_from_env(self):
        with patch.object(settings, "LLM_MODEL", "ollama/qwen3:8b"), \
             patch.object(settings, "LLM_API_BASE", "http://host.docker.internal:11434/v1"):
            mgr = _fresh_manager(self.tmp.name)
            cfg = mgr.load_config()

        self.assertEqual(len(cfg["llm_models"]), 1)
        entry = cfg["llm_models"][0]
        self.assertEqual(entry["model"], "ollama/qwen3:8b")
        self.assertEqual(entry["api_base"], "http://host.docker.internal:11434/v1")
        self.assertTrue(entry["enabled"])

    def test_seeded_entry_is_active(self):
        """A seeded model must be selected, or the pipeline still has nothing to run."""
        with patch.object(settings, "LLM_MODEL", "ollama/qwen3:8b"), \
             patch.object(settings, "LLM_API_BASE", "http://localhost:11434/v1"):
            mgr = _fresh_manager(self.tmp.name)
            cfg = mgr.load_config()

        self.assertEqual(len(cfg["llm_models_active"]), 1)
        self.assertEqual(cfg["llm_models_active"][0], cfg["llm_models"][0]["id"])

    def test_seeded_entry_passes_registry_validation(self):
        with patch.object(settings, "LLM_MODEL", "ollama/qwen3:8b"), \
             patch.object(settings, "LLM_API_BASE", "http://localhost:11434/v1"):
            mgr = _fresh_manager(self.tmp.name)
            cfg = mgr.load_config()

        validate_entry(cfg["llm_models"][0])  # raises ModelRegistryError on failure

    def test_api_key_from_env_is_carried_into_entry(self):
        with patch.object(settings, "LLM_MODEL", "openai/gpt-4o"), \
             patch.object(settings, "LLM_API_BASE", "https://api.openai.com/v1"), \
             patch.object(settings, "LLM_API_KEY", "sk-test-key"):
            mgr = _fresh_manager(self.tmp.name)
            cfg = mgr.load_config()

        self.assertEqual(cfg["llm_models"][0]["api_key_override"], "sk-test-key")

    def test_no_seed_when_env_model_is_blank(self):
        """Without LLM_MODEL there is nothing to seed — stay empty, don't invent one."""
        with patch.object(settings, "LLM_MODEL", ""), \
             patch.object(settings, "LLM_API_BASE", "http://localhost:11434/v1"):
            mgr = _fresh_manager(self.tmp.name)
            cfg = mgr.load_config()

        self.assertEqual(cfg["llm_models"], [])
        self.assertEqual(cfg["llm_models_active"], [])


class TestExistingConfigIsMigratedOnce(unittest.TestCase):
    """A config written before the registry existed must be back-filled.

    Seeding only ran when config.json was absent, so every install that already
    had a config file — i.e. every existing install — loaded with an empty
    registry and no runnable model. The marker distinguishes "never seeded" from
    "the operator removed every model", so migration happens exactly once.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _write_pre_registry_config(self):
        """A config.json shaped like one written before seeding existed."""
        import json

        path = os.path.join(self.tmp.name, "config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "github_token_enc": "",
                "llm_models": [],
                "llm_models_active": [],
                "provider_api_keys_enc": {},
                "embedding_model": "BAAI/bge-base-en-v1.5",
                "repos": {},
            }, f)

    def test_pre_existing_empty_registry_is_seeded_from_env(self):
        self._write_pre_registry_config()

        with patch.object(settings, "LLM_MODEL", "ollama/qwen3:8b"), \
             patch.object(settings, "LLM_API_BASE", "http://host.docker.internal:11434/v1"):
            cfg = _fresh_manager(self.tmp.name).load_config()

        self.assertEqual(len(cfg["llm_models"]), 1)
        self.assertEqual(cfg["llm_models"][0]["model"], "ollama/qwen3:8b")
        self.assertEqual(cfg["llm_models_active"], [cfg["llm_models"][0]["id"]])

    def test_migration_does_not_repeat_after_operator_empties_registry(self):
        """Once migrated, removing every model must stick."""
        self._write_pre_registry_config()

        with patch.object(settings, "LLM_MODEL", "ollama/qwen3:8b"), \
             patch.object(settings, "LLM_API_BASE", "http://localhost:11434/v1"):
            mgr = _fresh_manager(self.tmp.name)
            self.assertEqual(len(mgr.load_config()["llm_models"]), 1)

            cfg = mgr.load_config()
            cfg["llm_models"] = []
            cfg["llm_models_active"] = []
            mgr.save_config(cfg)

            reloaded = _fresh_manager(self.tmp.name).load_config()

        self.assertEqual(reloaded["llm_models"], [])

    def test_no_migration_when_env_model_is_blank(self):
        self._write_pre_registry_config()

        with patch.object(settings, "LLM_MODEL", ""):
            cfg = _fresh_manager(self.tmp.name).load_config()

        self.assertEqual(cfg["llm_models"], [])


class TestSeedingDoesNotClobberExistingConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_existing_registry_is_left_untouched(self):
        """Seeding must not overwrite a registry the operator already curated."""
        with patch.object(settings, "LLM_MODEL", "ollama/qwen3:8b"), \
             patch.object(settings, "LLM_API_BASE", "http://localhost:11434/v1"):
            mgr = _fresh_manager(self.tmp.name)

            cfg = mgr.load_config()
            cfg["llm_models"] = [{
                "id": "my-model", "label": "Curated",
                "model": "openai/gpt-4o", "api_base": "https://api.openai.com/v1",
                "api_key_override": "", "enabled": True,
            }]
            cfg["llm_models_active"] = ["my-model"]
            mgr.save_config(cfg)

            reloaded = _fresh_manager(self.tmp.name).load_config()

        self.assertEqual(len(reloaded["llm_models"]), 1)
        self.assertEqual(reloaded["llm_models"][0]["id"], "my-model")
        self.assertEqual(reloaded["llm_models_active"], ["my-model"])

    def test_deliberately_emptied_registry_is_not_reseeded(self):
        """An operator who removes every model must not have one silently restored."""
        with patch.object(settings, "LLM_MODEL", "ollama/qwen3:8b"), \
             patch.object(settings, "LLM_API_BASE", "http://localhost:11434/v1"):
            mgr = _fresh_manager(self.tmp.name)

            cfg = mgr.load_config()
            cfg["llm_models"] = []
            cfg["llm_models_active"] = []
            mgr.save_config(cfg)

            reloaded = _fresh_manager(self.tmp.name).load_config()

        self.assertEqual(reloaded["llm_models"], [])


if __name__ == "__main__":
    unittest.main()
