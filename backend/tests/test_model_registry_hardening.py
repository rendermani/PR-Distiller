"""A malformed registry entry must not escape as an undeclared exception type.

resolve_model() indexed entries with e["id"] and entry["model"] directly, so an
entry missing "id" raised KeyError and an entry with model=None raised
TypeError. job_orchestrator guards the per-model call with
`except ModelRegistryError`, so both escaped that filter and killed the whole
job: the remaining, well-formed models never ran.

validate_entry() also accepted those entries, since it checked field presence
but never the type or emptiness of `model` / `api_base`, and never id
uniqueness.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.model_registry import (
    ModelRegistryError,
    resolve_model,
    validate_entry,
    validate_registry,
)


def _entry(**overrides):
    base = {
        "id": "good",
        "label": "Good",
        "model": "ollama/qwen3:8b",
        "api_base": "http://localhost:11434/v1",
        "api_key_override": "",
        "enabled": True,
    }
    base.update(overrides)
    return base


class TestResolveModelRaisesOnlyRegistryError(unittest.TestCase):
    """Every malformed-input failure must surface as ModelRegistryError."""

    def test_entry_missing_id_does_not_raise_keyerror(self):
        registry = [{"label": "no id", "model": "ollama/x", "enabled": True}, _entry()]

        with self.assertRaises(ModelRegistryError):
            resolve_model("missing", registry, {})

    def test_sibling_entry_missing_id_does_not_break_a_good_lookup(self):
        """A broken neighbour must not prevent resolving a well-formed entry."""
        registry = [{"label": "no id"}, _entry()]

        resolved = resolve_model("good", registry, {})

        self.assertEqual(resolved["model"], "ollama/qwen3:8b")

    def test_none_model_raises_registry_error_not_typeerror(self):
        registry = [_entry(id="broken", model=None)]

        with self.assertRaises(ModelRegistryError):
            resolve_model("broken", registry, {})

    def test_empty_model_raises_registry_error(self):
        registry = [_entry(id="broken", model="")]

        with self.assertRaises(ModelRegistryError):
            resolve_model("broken", registry, {})

    def test_missing_enabled_key_raises_registry_error(self):
        registry = [{"id": "part", "label": "l", "model": "ollama/x", "api_base": "http://h"}]

        with self.assertRaises(ModelRegistryError):
            resolve_model("part", registry, {})


class TestValidateEntryChecksModelAndApiBase(unittest.TestCase):
    def test_none_model_is_rejected(self):
        with self.assertRaises(ModelRegistryError):
            validate_entry(_entry(model=None))

    def test_empty_model_is_rejected(self):
        with self.assertRaises(ModelRegistryError):
            validate_entry(_entry(model=""))

    def test_none_api_base_is_rejected(self):
        with self.assertRaises(ModelRegistryError):
            validate_entry(_entry(api_base=None))

    def test_empty_api_base_is_rejected(self):
        with self.assertRaises(ModelRegistryError):
            validate_entry(_entry(api_base=""))

    def test_well_formed_entry_still_passes(self):
        validate_entry(_entry())  # must not raise


class TestValidateRegistryRejectsDuplicateIds(unittest.TestCase):
    """Duplicate ids made the second entry dead config the UI still showed."""

    def test_duplicate_ids_are_rejected(self):
        registry = [_entry(id="dup"), _entry(id="dup", label="Shadowed")]

        with self.assertRaises(ModelRegistryError) as ctx:
            validate_registry(registry)

        self.assertIn("dup", str(ctx.exception))

    def test_distinct_ids_pass(self):
        validate_registry([_entry(id="a"), _entry(id="b")])

    def test_validate_registry_also_validates_each_entry(self):
        with self.assertRaises(ModelRegistryError):
            validate_registry([_entry(), _entry(id="bad", model="")])


if __name__ == "__main__":
    unittest.main()
