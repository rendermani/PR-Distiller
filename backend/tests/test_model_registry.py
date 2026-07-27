"""Tests for the model registry: validation + id-to-config resolution."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.model_registry import (
    ModelRegistryError,
    resolve_model,
    validate_entry,
)


class TestResolveModel(unittest.TestCase):
    def test_resolves_ollama_entry_with_no_key(self):
        registry = [
            {
                "id": "qwen-local",
                "label": "Qwen Local",
                "model": "ollama/qwen2.5-coder:7b-instruct",
                "api_base": "http://host.docker.internal:11434/v1",
                "api_key_override": "",
                "enabled": True,
            }
        ]
        provider_keys = {}
        resolved = resolve_model("qwen-local", registry, provider_keys)
        self.assertEqual(resolved["model"], "ollama/qwen2.5-coder:7b-instruct")
        self.assertEqual(resolved["api_base"], "http://host.docker.internal:11434/v1")
        # Ollama gets the existing 'unused' sentinel when no key is configured.
        self.assertEqual(resolved["api_key"], "unused")

    def test_resolves_openai_entry_via_provider_key(self):
        registry = [
            {
                "id": "gpt4",
                "label": "GPT-4 Omni",
                "model": "openai/gpt-4o",
                "api_base": "https://api.openai.com/v1",
                "api_key_override": "",
                "enabled": True,
            }
        ]
        provider_keys = {"openai": "sk-test-123"}
        resolved = resolve_model("gpt4", registry, provider_keys)
        self.assertEqual(resolved["api_key"], "sk-test-123")

    def test_api_key_override_wins(self):
        registry = [
            {
                "id": "gpt4-personal",
                "label": "GPT-4 Personal",
                "model": "openai/gpt-4o",
                "api_base": "https://api.openai.com/v1",
                "api_key_override": "sk-personal-456",
                "enabled": True,
            }
        ]
        provider_keys = {"openai": "sk-team-default"}
        resolved = resolve_model("gpt4-personal", registry, provider_keys)
        self.assertEqual(resolved["api_key"], "sk-personal-456")

    def test_unknown_id_raises(self):
        with self.assertRaises(ModelRegistryError):
            resolve_model("does-not-exist", [], {})

    def test_disabled_entry_raises(self):
        registry = [
            {
                "id": "qwen-local",
                "label": "Qwen Local",
                "model": "ollama/qwen3:8b",
                "api_base": "http://localhost:11434/v1",
                "api_key_override": "",
                "enabled": False,
            }
        ]
        with self.assertRaises(ModelRegistryError) as ctx:
            resolve_model("qwen-local", registry, {})
        self.assertIn("disabled", str(ctx.exception).lower())


class TestValidateEntry(unittest.TestCase):
    def test_minimal_valid_entry(self):
        entry = {
            "id": "qwen-local",
            "label": "Qwen Local",
            "model": "ollama/qwen3:8b",
            "api_base": "http://localhost:11434/v1",
            "api_key_override": "",
            "enabled": True,
        }
        # Returns None on success (raises on failure).
        self.assertIsNone(validate_entry(entry))

    def test_missing_required_field_raises(self):
        entry = {"id": "qwen-local", "label": "Qwen Local"}
        with self.assertRaises(ModelRegistryError):
            validate_entry(entry)

    def test_empty_id_raises(self):
        entry = {
            "id": "",
            "label": "x",
            "model": "ollama/x",
            "api_base": "http://x",
            "api_key_override": "",
            "enabled": True,
        }
        with self.assertRaises(ModelRegistryError):
            validate_entry(entry)

    def test_invalid_id_chars_raises(self):
        entry = {
            "id": "has spaces",
            "label": "x",
            "model": "ollama/x",
            "api_base": "http://x",
            "api_key_override": "",
            "enabled": True,
        }
        with self.assertRaises(ModelRegistryError):
            validate_entry(entry)

    def test_non_bool_enabled_raises(self):
        entry = {
            "id": "qwen-local",
            "label": "Qwen Local",
            "model": "ollama/qwen3:8b",
            "api_base": "http://localhost:11434/v1",
            "api_key_override": "",
            "enabled": "yes",  # string, not bool
        }
        with self.assertRaises(ModelRegistryError):
            validate_entry(entry)

    def test_non_string_id_raises(self):
        entry = {
            "id": 42,  # int, not string
            "label": "x",
            "model": "ollama/x",
            "api_base": "http://x",
            "api_key_override": "",
            "enabled": True,
        }
        with self.assertRaises(ModelRegistryError):
            validate_entry(entry)
