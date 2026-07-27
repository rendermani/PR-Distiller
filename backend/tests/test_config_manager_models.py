"""Tests for the llm_models registry + llm_models_active fields in ConfigManager."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import settings


def _make_manager(tmp_dir: str):
    """Construct a ConfigManager isolated to *tmp_dir*.

    Mirrors the pattern in test_config_manager_security.py: bypass __init__
    so the real backend/data/config.json is never touched, then wire up
    paths and cipher manually before letting the manager seed defaults.
    """
    from pipeline.config_manager import ConfigManager
    from cryptography.fernet import Fernet

    manager = ConfigManager.__new__(ConfigManager)
    manager.config_path = os.path.join(tmp_dir, "config.json")
    manager.key_path = os.path.join(tmp_dir, ".secret_key")
    manager.cipher = Fernet(Fernet.generate_key())
    manager._ensure_default_config()
    return manager


class TestConfigManagerModels(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cm = _make_manager(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_config_seeds_models_from_env(self):
        """A fresh config seeds one active model from the LLM_* env vars.

        Superseded an assertion that the registry starts empty: that left every
        install with no runnable model, since nothing else populates it.
        See tests/test_llm_registry_seeding.py for the seeding contract.
        """
        cfg = self.cm.load_config()
        self.assertEqual(len(cfg["llm_models"]), 1)
        self.assertEqual(cfg["llm_models"][0]["model"], settings.LLM_MODEL)
        self.assertEqual(cfg["llm_models_active"], [cfg["llm_models"][0]["id"]])

    def test_save_and_reload_round_trips_models(self):
        models = [
            {
                "id": "qwen-local",
                "label": "Qwen Local",
                "model": "ollama/qwen3:8b",
                "api_base": "http://localhost:11434/v1",
                "api_key_override": "",
                "enabled": True,
            },
            {
                "id": "gemma-local",
                "label": "Gemma Local",
                "model": "ollama/gemma4:e4b",
                "api_base": "http://localhost:11434/v1",
                "api_key_override": "",
                "enabled": True,
            },
        ]
        self.cm.save_config({
            "llm_models": models,
            "llm_models_active": ["qwen-local", "gemma-local"],
            "provider_api_keys": {},
            "repos": {},
        })

        reloaded = self.cm.load_config()
        # The reloaded entries should have api_key_override as plaintext "" (empty).
        for i, entry in enumerate(reloaded["llm_models"]):
            self.assertEqual(entry["id"], models[i]["id"])
            self.assertEqual(entry["label"], models[i]["label"])
            self.assertEqual(entry["model"], models[i]["model"])
            self.assertEqual(entry["api_base"], models[i]["api_base"])
            self.assertEqual(entry["api_key_override"], "")
            self.assertEqual(entry["enabled"], True)
        self.assertEqual(reloaded["llm_models_active"], ["qwen-local", "gemma-local"])

    def test_api_key_override_is_encrypted_on_disk(self):
        """Per-entry overrides must be Fernet-encrypted in config.json the same
        way provider_api_keys are."""
        self.cm.save_config({
            "llm_models": [{
                "id": "gpt4",
                "label": "GPT-4",
                "model": "openai/gpt-4o",
                "api_base": "https://api.openai.com/v1",
                "api_key_override": "sk-secret-XYZ",
                "enabled": True,
            }],
            "llm_models_active": ["gpt4"],
            "provider_api_keys": {},
            "repos": {},
        })

        # Raw file must not contain the plaintext key.
        with open(self.cm.config_path, "r") as f:
            raw = f.read()
        self.assertNotIn("sk-secret-XYZ", raw)

        # Loaded config decrypts it back.
        reloaded = self.cm.load_config()
        self.assertEqual(reloaded["llm_models"][0]["api_key_override"], "sk-secret-XYZ")

    def test_inactive_ids_in_llm_models_active_are_kept(self):
        """The active list can reference disabled or non-existent ids; the
        orchestrator decides what to do with them. Config layer just stores."""
        self.cm.save_config({
            "llm_models": [],
            "llm_models_active": ["ghost", "another-ghost"],
            "provider_api_keys": {},
            "repos": {},
        })
        reloaded = self.cm.load_config()
        self.assertEqual(reloaded["llm_models_active"], ["ghost", "another-ghost"])

    def test_legacy_llm_model_fields_dropped_on_load(self):
        """Old `llm_model`/`llm_api_base`/`llm_provider`/`llm_api_key` fields
        must not appear in loaded config — the new schema replaces them."""
        # Manually write an old-style config file.
        os.makedirs(os.path.dirname(self.cm.config_path), exist_ok=True)
        with open(self.cm.config_path, "w") as f:
            json.dump({
                "llm_model": "ollama/old-model",
                "llm_provider": "ollama",
                "llm_api_base": "http://old:11434/v1",
                "repos": {},
            }, f)

        reloaded = self.cm.load_config()
        self.assertNotIn("llm_model", reloaded)
        self.assertNotIn("llm_provider", reloaded)
        self.assertNotIn("llm_api_base", reloaded)
        self.assertNotIn("llm_api_key", reloaded)
        # New schema fields appear with empty defaults.
        self.assertEqual(reloaded["llm_models"], [])
        self.assertEqual(reloaded["llm_models_active"], [])

    def test_masked_override_becomes_empty_at_config_manager_level(self):
        """At the config_manager layer, an incoming api_key_override='***'
        is stripped to ''. This is the layer contract: do not persist the
        UI mask sentinel as a literal value. The API layer (Task 3) handles
        the round-trip preservation by reading current state before PUT.
        """
        # Seed a real key.
        self.cm.save_config({
            "llm_models": [{
                "id": "gpt4",
                "label": "GPT-4",
                "model": "openai/gpt-4o",
                "api_base": "https://api.openai.com/v1",
                "api_key_override": "sk-real",
                "enabled": True,
            }],
            "llm_models_active": [],
            "provider_api_keys": {},
            "repos": {},
        })
        # Now save with '***' as the override (simulating UI re-PUT of masked GET).
        self.cm.save_config({
            "llm_models": [{
                "id": "gpt4",
                "label": "GPT-4",
                "model": "openai/gpt-4o",
                "api_base": "https://api.openai.com/v1",
                "api_key_override": "***",
                "enabled": True,
            }],
            "llm_models_active": [],
            "provider_api_keys": {},
            "repos": {},
        })
        # The sentinel-stripping turned '***' into '', so the second save's
        # entry effectively has no override. The stored value will be ''.
        # (Task 3's API layer is what preserves the original via merge.)
        reloaded = self.cm.load_config()
        self.assertEqual(reloaded["llm_models"][0]["api_key_override"], "")


if __name__ == "__main__":
    unittest.main()
