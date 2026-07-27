"""Provider keys must resolve for every provider the UI can configure.

Two defects made configured keys silently unusable:

1. LiteLLM's Gemini prefix is `gemini/`, but the built-in model list groups
   those models under the provider name `google`, env_overrides reports
   `provider_api_keys.google`, and the Vault page renders a `google` input.
   resolve_model looked up `provider_keys["gemini"]`, so a key pasted into the
   Google field was never found.

2. Keys supplied via the documented env vars (OPENAI_API_KEY etc.) never
   reached provider_api_keys at all: load_config built that map purely from the
   encrypted on-disk values, while env_overrides told the UI the field was
   env-sourced and locked the input. The key was advertised as active,
   uneditable, and ignored.
"""
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import settings
from pipeline.model_registry import resolve_model


def _entry(model: str, **overrides):
    base = {
        "id": "m",
        "label": "M",
        "model": model,
        "api_base": "https://api.example.com/v1",
        "api_key_override": "",
        "enabled": True,
    }
    base.update(overrides)
    return base


class TestGeminiKeyResolvesFromGoogleField(unittest.TestCase):
    def test_gemini_model_finds_key_stored_under_google(self):
        resolved = resolve_model("m", [_entry("gemini/gemini-2.5-flash")], {"google": "g-key"})

        self.assertEqual(resolved["api_key"], "g-key")

    def test_gemini_key_name_still_works_if_stored_directly(self):
        """Don't break anyone who already stored it under 'gemini'."""
        resolved = resolve_model("m", [_entry("gemini/gemini-2.5-pro")], {"gemini": "direct"})

        self.assertEqual(resolved["api_key"], "direct")

    def test_entry_override_still_wins_over_provider_key(self):
        resolved = resolve_model(
            "m",
            [_entry("gemini/gemini-2.5-flash", api_key_override="per-entry")],
            {"google": "g-key"},
        )

        self.assertEqual(resolved["api_key"], "per-entry")

    def test_unprefixed_model_resolves_as_openai(self):
        resolved = resolve_model("m", [_entry("gpt-4o")], {"openai": "o-key"})

        self.assertEqual(resolved["api_key"], "o-key")


class TestEnvProviderKeysReachConfig(unittest.TestCase):
    """load_config must expose env-supplied provider keys, not just on-disk ones."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _load(self):
        from pipeline.config_manager import ConfigManager

        with patch.object(settings, "DATA_DIR", self.tmp.name):
            return ConfigManager().load_config()

    def test_openai_env_key_appears_in_provider_api_keys(self):
        with patch.object(settings, "OPENAI_API_KEY", "sk-from-env"):
            cfg = self._load()

        self.assertEqual(cfg["provider_api_keys"].get("openai"), "sk-from-env")

    def test_google_env_key_appears_in_provider_api_keys(self):
        with patch.object(settings, "GOOGLE_API_KEY", "g-from-env"):
            cfg = self._load()

        self.assertEqual(cfg["provider_api_keys"].get("google"), "g-from-env")

    def test_env_key_resolves_end_to_end_for_a_gemini_model(self):
        """The whole chain: env var -> config -> resolve_model."""
        with patch.object(settings, "GOOGLE_API_KEY", "g-from-env"):
            cfg = self._load()

        resolved = resolve_model(
            "m", [_entry("gemini/gemini-2.5-flash")], cfg["provider_api_keys"]
        )

        self.assertEqual(resolved["api_key"], "g-from-env")

    def test_stored_key_takes_precedence_over_env(self):
        """An operator who typed a key in the UI must not be overridden by env."""
        from pipeline.config_manager import ConfigManager

        with patch.object(settings, "DATA_DIR", self.tmp.name), \
             patch.object(settings, "OPENAI_API_KEY", "sk-from-env"):
            mgr = ConfigManager()
            mgr.save_config({"provider_api_keys": {"openai": "sk-typed"}})
            cfg = mgr.load_config()

        self.assertEqual(cfg["provider_api_keys"]["openai"], "sk-typed")

    def test_no_env_key_leaves_provider_absent(self):
        with patch.object(settings, "OPENAI_API_KEY", ""), \
             patch.object(settings, "GOOGLE_API_KEY", ""), \
             patch.object(settings, "ANTHROPIC_API_KEY", ""), \
             patch.object(settings, "OPENROUTER_API_KEY", ""):
            cfg = self._load()

        self.assertEqual(cfg["provider_api_keys"], {})


if __name__ == "__main__":
    unittest.main()
