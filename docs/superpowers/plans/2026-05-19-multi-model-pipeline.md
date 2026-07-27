# Multi-Model Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-model `llm_model` config field with a per-job-selectable list of pre-configured models. A single distillation job runs the full extract pipeline once per active model, with the existing semantic fuser deduplicating across passes. Cancel and partial-failure are resilient.

**Architecture:** Backend-only change. Config gains a `llm_models` registry (list of named model entries, each with its own `model`/`api_base`/`api_key_override`/`enabled`) and a `llm_models_active` selection (list of registry ids the next job should run). The orchestrator's `_execute_distillation` runs the crawl/dedup/redact phases once, then loops over the active models running classify+extract per pass. Rule metadata gains `extracted_by_model` (id) and `merged_with_models` (list of ids) so provenance survives semantic merging. The UI is **not** in scope.

**Tech Stack:** Python 3.11, FastAPI, LiteLLM (Ollama/OpenAI/Anthropic), ChromaDB, pytest (`unittest.TestCase` style — see existing tests).

---

## Context for the engineer

### Background

PR-Distiller crawls GitHub PR review comments, classifies them with an LLM, extracts coding rules as JSON with the same LLM, and stores them in ChromaDB with semantic-similarity deduplication via a "fuser" that runs another LLM call to merge near-duplicates.

Today, a job uses **one** model identified by `llm_model` (e.g. `"ollama/qwen3:8b"`). We benchmarked 4 small models against the same 1,241-comment cache and found ~55% rule overlap between any two models — running multiple models in sequence and letting the fuser dedupe gives ~40% more unique rules per job at the cost of N× wall time.

### Glossary

| Term | Meaning |
|---|---|
| **registry entry** | One element of `llm_models` — a named, reusable model config (id + label + model string + base URL + optional key override + enabled flag) |
| **active model** | A registry id listed in `llm_models_active` for the next job |
| **pass** | One full classify+extract loop over the cached `cleaned_payloads`, using a single model |
| **provider** | LiteLLM provider prefix in the `model` string. `ollama/...` → Ollama, `openai/...` → OpenAI, etc. Determines which provider key in `provider_api_keys` to use. |

### Existing files you'll touch

| File | Today's responsibility |
|---|---|
| `backend/pipeline/config_manager.py` | Load/save config with Fernet-encrypted secrets. Currently has `llm_provider`/`llm_model`/`llm_api_base`/`llm_api_key` fields. |
| `backend/pipeline/job_orchestrator.py` | Async pipeline runner. Single-model path lives in `_execute_distillation`. |
| `backend/api.py` | FastAPI routes for `/api/config`, `/api/jobs/*`. |
| `backend/llm/extractor.py` | `LargeLLMExtractor` — takes model/base/key in constructor; `batch_extract` runs classify+extract. **No change needed**, already per-instance. |
| `backend/pipeline/semantic_fuser.py` | `process_and_fuse` — stores or merges. **Small change**: needs to write `extracted_by_model` and append to `merged_with_models`. |

### Files you'll create

| File | Responsibility |
|---|---|
| `backend/pipeline/model_registry.py` | Pure helpers: validate a registry entry, resolve a registry id to a full `(model, api_base, api_key)` triple using `provider_api_keys` as fallback for the key. |
| `backend/tests/test_model_registry.py` | Unit tests for the resolver. |
| `backend/tests/test_orchestrator_multi_model.py` | Integration-style tests for the multi-pass orchestrator using mocked `LargeLLMExtractor`. |
| `backend/tests/test_config_manager_models.py` | Tests for the new config fields' load/save round-trip. |

### Schema (final shape after this plan)

```json
{
  "llm_models": [
    {
      "id": "qwen-local",
      "label": "Qwen 2.5 Coder 7B",
      "model": "ollama/qwen2.5-coder:7b-instruct",
      "api_base": "http://host.docker.internal:11434/v1",
      "api_key_override": "",
      "enabled": true
    }
  ],
  "llm_models_active": ["qwen-local"],
  "provider_api_keys": {
    "openai": "sk-...",
    "anthropic": "..."
  }
}
```

Resolver behaviour:
- `model` starts with `ollama/` → provider = `"ollama"`
- Looks up `provider_api_keys["ollama"]` for the key
- If `api_key_override` is non-empty, that wins (per-entry override beats provider default)
- Ollama may still send `"unused"` as a sentinel when no key is set (existing convention)

### Non-goals for this plan

- No UI changes (handled in a follow-up)
- No `llm_model`/`llm_api_base`/`llm_api_key`/`llm_provider` backward-compatibility (user confirmed: no compat needed)
- No new providers — the LiteLLM model string carries that
- No automatic registry migration from old config — the user will rebuild config.json by hand or via a one-shot script (out of scope)

---

## Pre-flight

- [ ] **Step 0.1: Make sure tests currently pass before starting**

Run: `cd /Users/mlautenschlager/Documents/Development/PR-Distiller/backend && python3 -m pytest tests/ -x --tb=short -q 2>&1 | tail -30`
Expected: All pass, or note any pre-existing failures so we don't blame ourselves later.

- [ ] **Step 0.2: Create the worktree-or-branch for this work**

If you have a clean working tree, branch off `main`:
```bash
git checkout -b feat/multi-model-pipeline
```
Otherwise stash first.

- [ ] **Step 0.3: Verify Docker backend is running and healthy**

Run: `curl -s http://localhost:8923/api/health`
Expected: `{"status":"ok"}`

---

## Task 1: Add registry data model + resolver

**Files:**
- Create: `backend/pipeline/model_registry.py`
- Create: `backend/tests/test_model_registry.py`

The resolver is the heart of this PR — turning a registry id into the three things `LargeLLMExtractor` needs (model, api_base, api_key). Building it first means later tasks can lean on it.

- [ ] **Step 1.1: Write the failing test for happy-path resolution**

Create `backend/tests/test_model_registry.py`:

```python
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
```

- [ ] **Step 1.2: Run the tests to verify they fail**

Run: `cd backend && python3 -m pytest tests/test_model_registry.py -v`
Expected: `ModuleNotFoundError: No module named 'pipeline.model_registry'`

- [ ] **Step 1.3: Write the minimal implementation**

Create `backend/pipeline/model_registry.py`:

```python
"""Model registry helpers.

The registry is a list of named model entries stored in config.json under
`llm_models`. Each entry pairs a stable id with the full inference config:
model string (LiteLLM provider prefix + model name), api_base URL, and an
optional per-entry api_key_override.

`provider_api_keys` (existing config field) is the fallback source of keys,
keyed by LiteLLM provider name extracted from the model string prefix.
"""
import re

_VALID_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_REQUIRED_FIELDS = ("id", "label", "model", "api_base", "api_key_override", "enabled")


class ModelRegistryError(ValueError):
    """Raised when a registry entry is malformed or a lookup fails."""


def validate_entry(entry: dict) -> None:
    """Raise ModelRegistryError if the entry is malformed. Return None on success."""
    if not isinstance(entry, dict):
        raise ModelRegistryError(f"Entry must be a dict, got {type(entry).__name__}")
    for field in _REQUIRED_FIELDS:
        if field not in entry:
            raise ModelRegistryError(f"Entry missing required field: {field!r}")
    entry_id = entry["id"]
    if not isinstance(entry_id, str) or not entry_id:
        raise ModelRegistryError("Entry 'id' must be a non-empty string")
    if not _VALID_ID_RE.match(entry_id):
        raise ModelRegistryError(
            f"Entry 'id' must match {_VALID_ID_RE.pattern!r}, got {entry_id!r}"
        )
    if not isinstance(entry["enabled"], bool):
        raise ModelRegistryError("Entry 'enabled' must be a bool")


def _provider_from_model(model: str) -> str:
    """Extract LiteLLM provider prefix from a model string.

    `ollama/qwen3:8b` -> `ollama`
    `openai/gpt-4o` -> `openai`
    `gpt-4o` (no prefix) -> `openai` (LiteLLM default)
    """
    if "/" in model:
        return model.split("/", 1)[0]
    return "openai"


def resolve_model(model_id: str, registry: list, provider_keys: dict) -> dict:
    """Resolve a registry id to an inference config triple.

    Returns: {"model": str, "api_base": str, "api_key": str}
    Raises: ModelRegistryError if id is unknown or entry is disabled.
    """
    entry = next((e for e in registry if e.get("id") == model_id), None)
    if entry is None:
        raise ModelRegistryError(f"Unknown model id: {model_id!r}")
    if not entry.get("enabled", False):
        raise ModelRegistryError(f"Model {model_id!r} is disabled")

    api_key = entry.get("api_key_override") or ""
    if not api_key:
        provider = _provider_from_model(entry["model"])
        api_key = provider_keys.get(provider, "")
        if not api_key and provider == "ollama":
            # Ollama doesn't need auth, but LiteLLM rejects empty strings.
            api_key = "unused"

    return {
        "model": entry["model"],
        "api_base": entry["api_base"],
        "api_key": api_key,
    }
```

- [ ] **Step 1.4: Run tests to verify pass**

Run: `cd backend && python3 -m pytest tests/test_model_registry.py -v`
Expected: 7 passed.

- [ ] **Step 1.5: Commit**

```bash
git add backend/pipeline/model_registry.py backend/tests/test_model_registry.py
git commit -m "feat(pipeline): add model registry data model + resolver

The registry maps stable ids to full inference configs (model string,
api_base, api_key). Per-entry api_key_override beats the
provider_api_keys default. Validation enforces kebab-case ids and
required fields.

Used by the upcoming multi-model orchestrator."
```

---

## Task 2: Persist `llm_models` and `llm_models_active` in config_manager

**Files:**
- Modify: `backend/pipeline/config_manager.py`
- Create: `backend/tests/test_config_manager_models.py`

Add the new fields to load/save. Remove the old `llm_model`/`llm_api_base`/`llm_provider`/`llm_api_key` fields entirely (user confirmed: no backward compatibility). `provider_api_keys` stays untouched — it's still the keyring for cloud providers.

- [ ] **Step 2.1: Read the current config_manager structure**

Run: `cat backend/pipeline/config_manager.py | head -200`

Confirm: `_ensure_default_config` writes defaults, `load_config` reads, `save_config` writes. Note where `llm_provider/llm_model/llm_api_base/llm_api_key/llm_api_key_enc` references appear — you'll remove **all** of them.

- [ ] **Step 2.2: Write failing tests for the new fields**

Create `backend/tests/test_config_manager_models.py`:

```python
"""Tests for the llm_models registry + llm_models_active fields in ConfigManager."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.config_manager import ConfigManager


class TestConfigManagerModels(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.tmp.name, "config.json")
        # ConfigManager reads its path from env; set it up.
        os.environ["CONFIG_PATH"] = self.config_path
        # Also need a key path so Fernet doesn't fail.
        self.key_path = os.path.join(self.tmp.name, ".secret_key")
        os.environ["CONFIG_SECRET_KEY_PATH"] = self.key_path

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("CONFIG_PATH", None)
        os.environ.pop("CONFIG_SECRET_KEY_PATH", None)

    def test_default_config_has_empty_models_list(self):
        cm = ConfigManager()
        cfg = cm.load_config()
        self.assertEqual(cfg["llm_models"], [])
        self.assertEqual(cfg["llm_models_active"], [])

    def test_save_and_reload_round_trips_models(self):
        cm = ConfigManager()
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
        cm.save_config({
            "llm_models": models,
            "llm_models_active": ["qwen-local", "gemma-local"],
            "provider_api_keys": {},
            "repos": {},
        })

        reloaded = cm.load_config()
        self.assertEqual(reloaded["llm_models"], models)
        self.assertEqual(reloaded["llm_models_active"], ["qwen-local", "gemma-local"])

    def test_api_key_override_is_encrypted_on_disk(self):
        """Per-entry overrides must be Fernet-encrypted in config.json the same
        way provider_api_keys are."""
        cm = ConfigManager()
        cm.save_config({
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
        with open(self.config_path, "r") as f:
            raw = f.read()
        self.assertNotIn("sk-secret-XYZ", raw)

        # Loaded config decrypts it back.
        reloaded = cm.load_config()
        self.assertEqual(reloaded["llm_models"][0]["api_key_override"], "sk-secret-XYZ")

    def test_inactive_ids_in_llm_models_active_are_kept(self):
        """The active list can reference disabled or non-existent ids; the
        orchestrator decides what to do with them. Config layer just stores."""
        cm = ConfigManager()
        cm.save_config({
            "llm_models": [],
            "llm_models_active": ["ghost", "another-ghost"],
            "provider_api_keys": {},
            "repos": {},
        })
        reloaded = cm.load_config()
        self.assertEqual(reloaded["llm_models_active"], ["ghost", "another-ghost"])

    def test_legacy_llm_model_field_is_dropped(self):
        """Old `llm_model`/`llm_api_base`/`llm_provider`/`llm_api_key` fields
        must not appear in loaded config — the new schema replaces them."""
        cm = ConfigManager()
        # Manually write an old-style config file.
        with open(self.config_path, "w") as f:
            json.dump({
                "llm_model": "ollama/old-model",
                "llm_provider": "ollama",
                "llm_api_base": "http://old:11434/v1",
                "repos": {},
            }, f)

        reloaded = cm.load_config()
        self.assertNotIn("llm_model", reloaded)
        self.assertNotIn("llm_provider", reloaded)
        self.assertNotIn("llm_api_base", reloaded)
        self.assertNotIn("llm_api_key", reloaded)
        # New schema fields appear with empty defaults.
        self.assertEqual(reloaded["llm_models"], [])
        self.assertEqual(reloaded["llm_models_active"], [])
```

- [ ] **Step 2.3: Run tests to confirm failures**

Run: `cd backend && python3 -m pytest tests/test_config_manager_models.py -v`
Expected: 5 failures (`KeyError: 'llm_models'` or similar).

- [ ] **Step 2.4: Modify `config_manager.py` — remove old fields**

In `backend/pipeline/config_manager.py`, locate the sentinel-stripping loop in the `save_config` helper, the legacy migration in `_migrate_provider`, and the `_ensure_default_config` defaults. Edit as follows.

Replace the `_ensure_default_config` body so the default written to a fresh config.json has the new schema only:

```python
def _ensure_default_config(self):
    if not os.path.exists(self.config_path):
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        default_config = {
            "github_token": "",
            "github_webhook_secret": "",
            "huggingface_token": "",
            "llm_models": [],
            "llm_models_active": [],
            "provider_api_keys": {},
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "repos": {},
            "provider_models": self._get_default_providers(),
        }
        self.save_config(default_config)
```

Delete the `_migrate_provider` method entirely (no longer needed).

In `load_config`, remove every reference to `llm_provider`, `llm_model`, `llm_api_base`, `llm_api_key`, `llm_api_key_enc`, and the `_migrate_provider` call. Replace with reading the new fields:

```python
def load_config(self) -> dict:
    try:
        with open(self.config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        pm = raw.get("provider_models", {})
        if not pm:
            pm = self._get_default_providers()

        provider_keys_enc = raw.get("provider_api_keys_enc", {})
        provider_keys = {p: self._decrypt(v) for p, v in provider_keys_enc.items()}

        # Decrypt per-entry api_key_override fields.
        llm_models_raw = raw.get("llm_models", [])
        llm_models = []
        for entry in llm_models_raw:
            decrypted_entry = dict(entry)
            enc_key = entry.get("api_key_override_enc", "")
            decrypted_entry["api_key_override"] = self._decrypt(enc_key) if enc_key else ""
            decrypted_entry.pop("api_key_override_enc", None)
            llm_models.append(decrypted_entry)

        return {
            "github_token": self._decrypt(raw.get("github_token_enc", "")),
            "huggingface_token": self._decrypt(raw.get("huggingface_token_enc", "")),
            "github_webhook_secret": self._decrypt(raw.get("github_webhook_secret_enc", "")),
            "llm_models": llm_models,
            "llm_models_active": raw.get("llm_models_active", []),
            "provider_api_keys": provider_keys,
            "embedding_model": raw.get("embedding_model", "BAAI/bge-base-en-v1.5"),
            "repos": raw.get("repos", {}),
            "provider_models": pm,
            "crawl_cursors": raw.get("crawl_cursors", {}),
        }
    except FileNotFoundError:
        return {}
    except Exception:
        logger.exception("Failed to load config from %s", self.config_path)
        return {}
```

In `save_config`, remove every reference to the legacy single-model fields and add encryption of the per-entry overrides. Find the block that builds the on-disk dict and replace its model-related portion:

```python
# Encrypt per-entry api_key_override values.
llm_models_in = new_config.get("llm_models", [])
llm_models_enc = []
for entry in llm_models_in:
    enc_entry = dict(entry)
    plaintext_key = enc_entry.pop("api_key_override", "")
    enc_entry["api_key_override_enc"] = self._encrypt(plaintext_key) if plaintext_key else ""
    llm_models_enc.append(enc_entry)

on_disk = {
    "github_token_enc": self._encrypt(new_config.get("github_token", "")),
    "huggingface_token_enc": self._encrypt(new_config.get("huggingface_token", "")),
    "github_webhook_secret_enc": self._encrypt(new_config.get("github_webhook_secret", "")),
    "llm_models": llm_models_enc,
    "llm_models_active": new_config.get("llm_models_active", []),
    "provider_api_keys_enc": provider_keys_enc,
    "embedding_model": new_config.get("embedding_model", "BAAI/bge-base-en-v1.5"),
    "repos": new_config.get("repos", {}),
    "provider_models": new_config.get("provider_models", self._get_default_providers()),
    "crawl_cursors": new_config.get("crawl_cursors", {}),
}
```

Remove the legacy-key merge block (lines around `legacy_key = new_config.get("llm_api_key", "")` and the `provider_keys[active_provider] = legacy_key` assignment). It's no longer needed.

If `config_manager.py` references `settings.LLM_PROVIDER`/`settings.LLM_MODEL`/`settings.LLM_API_BASE`/`settings.LLM_API_KEY`, remove those imports/uses too — they're handled by the registry now.

- [ ] **Step 2.5: Run tests to verify pass**

Run: `cd backend && python3 -m pytest tests/test_config_manager_models.py tests/test_config_manager_security.py tests/test_config_manager_concurrency.py -v`
Expected: All pass.

Also run the broader test suite to catch fallout:
```bash
cd backend && python3 -m pytest tests/ -x --tb=short -q 2>&1 | tail -30
```
Some pre-existing tests for `llm_model`/`llm_provider` will fail. **These need fixing in Task 2.6** — don't move on until they pass.

- [ ] **Step 2.6: Update broken tests that referenced removed fields**

Search for any test that asserts on `llm_model`, `llm_api_base`, `llm_provider`, or `llm_api_key` in the loaded config dict:

```bash
cd backend && grep -rln "llm_model\|llm_provider\|llm_api_base\|llm_api_key" tests/
```

For each hit:
- If it's about the orchestrator running a single-model pass, defer to Task 4 (the orchestrator change). Skip those tests temporarily by adding `@unittest.skip("rewritten in Task 4")` with a TODO comment. Track them so they get un-skipped in Task 4.
- If it's about config CRUD specifically, rewrite it to use `llm_models`/`llm_models_active` (the patterns in `test_config_manager_models.py` are your guide).

- [ ] **Step 2.7: Commit**

```bash
git add backend/pipeline/config_manager.py backend/tests/test_config_manager_models.py backend/tests/
git commit -m "feat(config): replace llm_model with llm_models registry

config.json now stores a list of named model entries (id, label, model,
api_base, api_key_override, enabled) in 'llm_models' and a selection list
in 'llm_models_active'. Per-entry api_key_override is Fernet-encrypted on
disk. provider_api_keys remains the default keyring per LiteLLM provider.

The single-model fields llm_model/llm_provider/llm_api_base/llm_api_key
are removed entirely — there is no migration path."
```

---

## Task 3: Add /api/config CRUD for the registry

**Files:**
- Modify: `backend/api.py` (`/api/config` GET and PUT handlers, around lines 240-420)
- Add tests: `backend/tests/test_config_manager_models.py` (extend with API-level tests)

The existing `/api/config` route does field-by-field copy with sentinel masking. Replace its body with a pass-through that round-trips the new schema.

- [ ] **Step 3.1: Read the current /api/config GET and PUT handlers**

Run: `cd backend && grep -n '"/api/config"' api.py`

Read the surrounding 80 lines. Note:
- The PUT handler uses a Pydantic model `ConfigUpdate` with `llm_model`/`llm_api_base`/`llm_api_key` fields. Those go away.
- The GET handler masks secrets with `"***"`. Per-entry `api_key_override` needs the same masking.

- [ ] **Step 3.2: Write failing tests for the new CRUD shape**

Append to `backend/tests/test_config_manager_models.py`:

```python
from fastapi.testclient import TestClient


class TestConfigAPI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["CONFIG_PATH"] = os.path.join(self.tmp.name, "config.json")
        os.environ["CONFIG_SECRET_KEY_PATH"] = os.path.join(self.tmp.name, ".secret_key")
        os.environ["API_AUTH_TOKEN"] = "test-token"
        # Re-import api with the new env.
        import importlib
        import api
        importlib.reload(api)
        self.client = TestClient(api.app)
        self.headers = {"Authorization": "Bearer test-token"}

    def tearDown(self):
        self.tmp.cleanup()
        for k in ("CONFIG_PATH", "CONFIG_SECRET_KEY_PATH", "API_AUTH_TOKEN"):
            os.environ.pop(k, None)

    def test_get_returns_empty_models_initially(self):
        r = self.client.get("/api/config", headers=self.headers)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["llm_models"], [])
        self.assertEqual(r.json()["llm_models_active"], [])

    def test_put_creates_models_and_get_returns_them(self):
        body = {
            "llm_models": [{
                "id": "qwen-local",
                "label": "Qwen",
                "model": "ollama/qwen3:8b",
                "api_base": "http://localhost:11434/v1",
                "api_key_override": "",
                "enabled": True,
            }],
            "llm_models_active": ["qwen-local"],
            "provider_api_keys": {},
        }
        r = self.client.put("/api/config", json=body, headers=self.headers)
        self.assertEqual(r.status_code, 200)

        got = self.client.get("/api/config", headers=self.headers).json()
        self.assertEqual(len(got["llm_models"]), 1)
        self.assertEqual(got["llm_models"][0]["id"], "qwen-local")
        self.assertEqual(got["llm_models_active"], ["qwen-local"])

    def test_get_masks_api_key_override(self):
        self.client.put("/api/config", json={
            "llm_models": [{
                "id": "gpt4",
                "label": "GPT-4",
                "model": "openai/gpt-4o",
                "api_base": "https://api.openai.com/v1",
                "api_key_override": "sk-secret",
                "enabled": True,
            }],
            "llm_models_active": [],
            "provider_api_keys": {},
        }, headers=self.headers)

        got = self.client.get("/api/config", headers=self.headers).json()
        self.assertEqual(got["llm_models"][0]["api_key_override"], "***")

    def test_put_preserves_masked_api_key_override(self):
        """Round-trip a GET-then-PUT: if api_key_override comes back as '***',
        the server must keep the stored plaintext, not overwrite with '***'."""
        # Seed a key.
        self.client.put("/api/config", json={
            "llm_models": [{
                "id": "gpt4",
                "label": "GPT-4",
                "model": "openai/gpt-4o",
                "api_base": "https://api.openai.com/v1",
                "api_key_override": "sk-secret",
                "enabled": True,
            }],
            "llm_models_active": [],
            "provider_api_keys": {},
        }, headers=self.headers)

        # Read it back (with mask), then PUT unchanged.
        got = self.client.get("/api/config", headers=self.headers).json()
        self.client.put("/api/config", json=got, headers=self.headers)

        # Reload from disk directly to confirm plaintext is preserved.
        from pipeline.config_manager import ConfigManager
        cm = ConfigManager()
        cfg = cm.load_config()
        self.assertEqual(cfg["llm_models"][0]["api_key_override"], "sk-secret")

    def test_put_rejects_invalid_entry(self):
        r = self.client.put("/api/config", json={
            "llm_models": [{"id": "", "label": "bad"}],  # missing fields
            "llm_models_active": [],
            "provider_api_keys": {},
        }, headers=self.headers)
        self.assertEqual(r.status_code, 400)
```

- [ ] **Step 3.3: Run failing tests**

Run: `cd backend && python3 -m pytest tests/test_config_manager_models.py::TestConfigAPI -v`
Expected: All fail with field-validation or 422 errors.

- [ ] **Step 3.4: Update api.py**

Find the existing `ConfigUpdate` Pydantic model (around line 240) and replace it with a passthrough-style schema:

```python
from typing import Any, Dict, List
from pydantic import BaseModel

class ConfigUpdate(BaseModel):
    llm_models: List[Dict[str, Any]] | None = None
    llm_models_active: List[str] | None = None
    provider_api_keys: Dict[str, str] | None = None
    github_token: str | None = None
    huggingface_token: str | None = None
    github_webhook_secret: str | None = None
    embedding_model: str | None = None
    repos: Dict[str, Any] | None = None
    # Do NOT include llm_model, llm_api_base, llm_api_key, llm_provider.
```

Find the `/api/config` GET handler. Where it builds the response, mask each entry's `api_key_override` if non-empty:

```python
@app.get("/api/config", dependencies=[Depends(verify_token)])
async def get_config():
    cfg = config_manager.load_config()
    # Mask secrets.
    for field in ("github_token", "huggingface_token", "github_webhook_secret"):
        if cfg.get(field):
            cfg[field] = "***"
    cfg["provider_api_keys"] = {p: "***" for p, k in cfg.get("provider_api_keys", {}).items() if k}
    for entry in cfg.get("llm_models", []):
        if entry.get("api_key_override"):
            entry["api_key_override"] = "***"
    return cfg
```

Find the `/api/config` PUT handler. Validate registry entries via `validate_entry` before saving, and preserve masked values:

```python
@app.put("/api/config", dependencies=[Depends(verify_token)])
async def update_config(update: ConfigUpdate):
    from pipeline.model_registry import validate_entry, ModelRegistryError

    current = config_manager.load_config()
    new = current.copy()

    incoming = update.model_dump(exclude_unset=True)

    # Validate any incoming model entries before merging.
    if "llm_models" in incoming:
        try:
            for entry in incoming["llm_models"]:
                validate_entry(entry)
        except ModelRegistryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Preserve masked api_key_override values: if incoming entry has '***'
        # for its key, keep the currently-stored plaintext.
        current_by_id = {e["id"]: e for e in current.get("llm_models", [])}
        for entry in incoming["llm_models"]:
            if entry.get("api_key_override") == "***":
                entry["api_key_override"] = current_by_id.get(entry["id"], {}).get("api_key_override", "")
        new["llm_models"] = incoming["llm_models"]

    # Same '***' preservation for the other secret scalars.
    for field in ("github_token", "huggingface_token", "github_webhook_secret"):
        if field in incoming and incoming[field] != "***":
            new[field] = incoming[field]

    if "provider_api_keys" in incoming:
        # Strip '***' sentinels from incoming, keep stored values for those.
        merged = dict(current.get("provider_api_keys", {}))
        for provider, key in incoming["provider_api_keys"].items():
            if key != "***":
                merged[provider] = key
        new["provider_api_keys"] = merged

    for plain_field in ("llm_models_active", "embedding_model", "repos"):
        if plain_field in incoming:
            new[plain_field] = incoming[plain_field]

    config_manager.save_config(new)
    return {"status": "ok"}
```

- [ ] **Step 3.5: Run tests to verify pass**

Run: `cd backend && python3 -m pytest tests/test_config_manager_models.py::TestConfigAPI -v`
Expected: 5 passed.

Also re-run the whole suite:
```bash
cd backend && python3 -m pytest tests/ -x --tb=short -q 2>&1 | tail -30
```
Anything from the old `/api/config` tests that still references `llm_model` field needs to be updated or skipped per Task 2.6's pattern.

- [ ] **Step 3.6: Commit**

```bash
git add backend/api.py backend/tests/test_config_manager_models.py
git commit -m "feat(api): /api/config CRUD for llm_models registry

GET masks per-entry api_key_override. PUT validates each entry via the
model_registry validator and preserves stored plaintexts when the
incoming value is the '***' mask. Old llm_model/llm_api_base fields are
removed from the request/response schema."
```

---

## Task 4: Multi-pass orchestrator + per-rule provenance

**Files:**
- Modify: `backend/pipeline/job_orchestrator.py` (mostly `_execute_distillation`)
- Modify: `backend/pipeline/semantic_fuser.py` (`process_and_fuse` — add provenance fields)
- Create: `backend/tests/test_orchestrator_multi_model.py`

This is where the actual feature lands. The job now runs N passes over the same `cleaned_payloads`.

### Per-rule provenance

`semantic_fuser.process_and_fuse` is called once per extracted rule. Today it stamps `rule_dict["metadata"]["repo"]` and writes the dict. We need:
- The orchestrator to pass the current model's id and label to the extractor each pass (via the existing constructor args plus two new args).
- The extractor to stamp `rule_dict["metadata"]["extracted_by_model"]` and `rule_dict["metadata"]["extracted_by_label"]` before calling `fuser.process_and_fuse`.
- The fuser, when it detects a collision and merges, to **append** the incoming `extracted_by_model` to the **existing rule's** `merged_with_models` list.

### Per-pass result tracking

Each pass returns a count. The orchestrator collects `{model_id: {attempted, extracted, failed_reason?}}` so the final status string can say something like `"Completed: 312 unique rules (qwen-local: 280, gemma-local: 268)"`.

### Cancel + partial-failure semantics

- Cancellation between passes: check `cancel_flags[job_id]` after each pass; if set, stop the loop, mark the job aborted, **keep** whatever rules were saved.
- Per-model failure (e.g. preflight LLM call raises): catch the exception around the per-pass block, log it as `{model_id: {..., failed_reason: str(exc)}}`, and continue to the next model.

- [ ] **Step 4.1: Write a failing orchestrator test using mocks**

Create `backend/tests/test_orchestrator_multi_model.py`:

```python
"""Integration tests for the multi-model orchestrator.

Mocks LargeLLMExtractor + the crawler so we only test the loop & wiring.
"""
import asyncio
import os
import sys
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestMultiModelOrchestrator(unittest.TestCase):
    def setUp(self):
        from pipeline.job_orchestrator import JobOrchestrator
        self.db = MagicMock()
        self.db.delete_repo_rules = MagicMock(return_value=0)
        self.orchestrator = JobOrchestrator(self.db)

    def _config(self, model_ids):
        models = [
            {
                "id": mid,
                "label": mid.upper(),
                "model": f"ollama/{mid}:test",
                "api_base": "http://test:11434/v1",
                "api_key_override": "",
                "enabled": True,
            } for mid in model_ids
        ]
        return {
            "llm_models": models,
            "llm_models_active": model_ids,
            "provider_api_keys": {},
            "github_token": "",
        }

    def _payload(self):
        return {"repo": "tucowsinc/tdp-apis", "months": 1, "use_cache": True}

    @patch("pipeline.job_orchestrator.acompletion", new_callable=AsyncMock)
    @patch("pipeline.job_orchestrator.dev_cache")
    @patch("pipeline.job_orchestrator.LargeLLMExtractor")
    def test_runs_one_pass_per_active_model(self, mock_extractor_cls, mock_cache, mock_preflight):
        mock_cache.has_cache.return_value = True
        mock_cache.cache_info.return_value = {"count": 3, "updated_at": "2026-01-01T00:00:00Z"}
        mock_cache.load_crawl.return_value = [("comment-1", "diff-1")] * 3

        instances = []
        def make_mock(*args, **kwargs):
            inst = MagicMock()
            inst.batch_extract = AsyncMock(return_value=[{"id": "r1"}, {"id": "r2"}])
            instances.append((inst, kwargs))
            return inst
        mock_extractor_cls.side_effect = make_mock

        config = self._config(["qwen", "gemma"])
        job_id = self.orchestrator.trigger_job(self._payload(), config)
        # Run the loop synchronously by collecting the asyncio task.
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            # trigger_job uses asyncio.create_task; we need to await it.
            for task in asyncio.all_tasks(loop):
                pass
            # Drain pending tasks by running until idle.
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        finally:
            loop.close()

        self.assertEqual(len(instances), 2, "Extractor should be instantiated once per model")
        ids_used = [kwargs["model"] for _, kwargs in instances]
        self.assertEqual(ids_used, ["ollama/qwen:test", "ollama/gemma:test"])

    @patch("pipeline.job_orchestrator.acompletion", new_callable=AsyncMock)
    @patch("pipeline.job_orchestrator.dev_cache")
    @patch("pipeline.job_orchestrator.LargeLLMExtractor")
    def test_failed_model_does_not_abort_remaining_models(self, mock_extractor_cls, mock_cache, mock_preflight):
        mock_cache.has_cache.return_value = True
        mock_cache.cache_info.return_value = {"count": 3, "updated_at": "2026-01-01T00:00:00Z"}
        mock_cache.load_crawl.return_value = [("c", "d")] * 3

        # First instance raises during preflight; second succeeds.
        mock_preflight.side_effect = [Exception("LLM unreachable"), MagicMock(), MagicMock()]
        successful = MagicMock()
        successful.batch_extract = AsyncMock(return_value=[{"id": "r1"}])
        mock_extractor_cls.return_value = successful

        config = self._config(["bad-model", "good-model"])
        job_id = self.orchestrator.trigger_job(self._payload(), config)
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        finally:
            loop.close()

        status = self.orchestrator.get_status(job_id)
        self.assertEqual(status.get("progress"), 100)
        per_model = status.get("per_model_results", {})
        self.assertIn("bad-model", per_model)
        self.assertIn("good-model", per_model)
        self.assertIn("failed_reason", per_model["bad-model"])
        self.assertNotIn("failed_reason", per_model["good-model"])


class TestFuserProvenance(unittest.TestCase):
    """Provenance fields on rules: extracted_by_model + merged_with_models."""

    def test_first_insertion_stamps_extracted_by_model(self):
        from pipeline.semantic_fuser import SemanticFuser
        db = MagicMock()
        db.is_blocked.return_value = False
        db.find_similar_rule.return_value = (None, None, None)
        db.store_rule = MagicMock(return_value="stored-id")

        fuser = SemanticFuser(db, model="m", api_base="http://x", api_key="k")
        rule = {
            "rule_id": "r1",
            "content": {"title": "t", "description": "d", "enforcement_prompt": "e"},
            "metadata": {
                "repo": "owner/repo",
                "extracted_by_model": "qwen-local",
                "extracted_by_label": "Qwen Local",
            },
        }
        fuser.process_and_fuse(rule)

        # The fuser must have called store_rule with the provenance preserved.
        stored = db.store_rule.call_args[0][0]
        self.assertEqual(stored["metadata"]["extracted_by_model"], "qwen-local")
        self.assertEqual(stored["metadata"].get("merged_with_models", []), [])

    def test_collision_appends_to_merged_with_models(self):
        from pipeline.semantic_fuser import SemanticFuser
        db = MagicMock()
        db.is_blocked.return_value = False
        # First call -> collision against an existing rule.
        existing_meta = {
            "repo": "owner/repo",
            "extracted_by_model": "qwen-local",
            "merged_with_models": [],
            "occurrence_count": 1,
        }
        db.find_similar_rule.return_value = ("existing-id", "Rule: foo", existing_meta)
        db.store_rule = MagicMock(return_value="merged-id")

        # Mock the LLM merge call so we don't hit a real API.
        with patch("pipeline.semantic_fuser.completion") as mock_completion:
            mock_completion.return_value.choices = [MagicMock()]
            mock_completion.return_value.choices[0].message.content = (
                '{"rule_id": "r-merged", "category": "correctness", '
                '"confidence": 0.9, "content": {"title": "t", "description": "d", '
                '"enforcement_prompt": "e"}}'
            )

            fuser = SemanticFuser(db, model="m", api_base="http://x", api_key="k")
            rule = {
                "rule_id": "r2",
                "content": {"title": "t", "description": "d", "enforcement_prompt": "e"},
                "metadata": {
                    "repo": "owner/repo",
                    "extracted_by_model": "gemma-local",
                },
            }
            fuser.process_and_fuse(rule)

        stored = db.store_rule.call_args[0][0]
        # The new rule was from gemma. The pre-existing match was from qwen.
        # After merge, the lineage should reflect both.
        self.assertIn("gemma-local", stored["metadata"]["merged_with_models"])
        self.assertIn("qwen-local", stored["metadata"]["merged_with_models"])
```

- [ ] **Step 4.2: Run failing tests**

Run: `cd backend && python3 -m pytest tests/test_orchestrator_multi_model.py -v --tb=short`
Expected: All fail (orchestrator still single-model, fuser doesn't set provenance).

- [ ] **Step 4.3: Modify the extractor to stamp provenance**

In `backend/llm/extractor.py`, the `__init__` already takes `model`/`api_base`/`api_key`. Add two more constructor args: `model_id: str | None = None` and `model_label: str | None = None`. Store them as `self.model_id` and `self.model_label`.

Then, in `async_extract_rule` (and the sync `extract_rule` for symmetry), after `rule_dict` is parsed and `_apply_metadata` has run, but **before** `fuser.process_and_fuse` is called, stamp the fields:

```python
if self.model_id:
    rule_dict["metadata"]["extracted_by_model"] = self.model_id
if self.model_label:
    rule_dict["metadata"]["extracted_by_label"] = self.model_label
# Initialize the merge lineage if not already set.
rule_dict["metadata"].setdefault("merged_with_models", [])
```

- [ ] **Step 4.4: Modify the fuser to append provenance on merge**

In `backend/pipeline/semantic_fuser.py`, locate the merge branch (after the LLM merge call and just before `db.store_rule(fused_rule)`). After building `fused_rule["metadata"]`, append the new rule's `extracted_by_model` and the existing rule's `extracted_by_model` to a deduplicated list:

```python
existing_models = list(
    (matched_metadatas or {}).get("merged_with_models", [])
) if matched_metadatas else []
existing_origin = (matched_metadatas or {}).get("extracted_by_model")
new_origin = new_rule_json.get("metadata", {}).get("extracted_by_model")

lineage = list(existing_models)
for origin in (existing_origin, new_origin):
    if origin and origin not in lineage:
        lineage.append(origin)
fused_rule["metadata"]["merged_with_models"] = lineage
# Preserve the original first extractor as the canonical extracted_by.
fused_rule["metadata"]["extracted_by_model"] = (
    existing_origin or new_origin
)
```

In the non-collision branch (where `not matched_id`), preserve provenance straight from `new_rule_json` if present (no extra code needed beyond the existing `self.db.store_rule(new_rule_json)` call — the rule already has the fields).

- [ ] **Step 4.5: Modify the orchestrator to run N passes**

In `backend/pipeline/job_orchestrator.py` `_execute_distillation`, locate the block starting at `# 3. Extract — pass per-job LLM config via constructor args ...` (around line 180).

**Replace** the single-model resolution (`llm_api_base = config.get("llm_api_base", ...)` etc., plus the `extractor = LargeLLMExtractor(...)` instantiation, plus the preflight + `batch_extract` call) with the multi-pass loop:

```python
# 3. Resolve active models from the registry. Skip unknown/disabled ids.
from pipeline.model_registry import resolve_model, ModelRegistryError

registry = config.get("llm_models", [])
active_ids = config.get("llm_models_active", [])
provider_keys = config.get("provider_api_keys", {})

if not active_ids:
    raise RuntimeError("No models selected: llm_models_active is empty")

per_model_results: dict[str, dict] = {}
self.active_jobs[job_id]["per_model_results"] = per_model_results
total_attempted = len(cleaned_payloads)

# Per-pass progress: divide the 35→95% range across N models.
pct_per_model = max(1, (95 - 35) // max(len(active_ids), 1))

for idx, model_id in enumerate(active_ids):
    if check_cancel():
        self.active_jobs[job_id]["status"] = "Pipeline Aborted via User Interrupt"
        self.active_jobs[job_id]["progress"] = -1
        return

    entry = next((e for e in registry if e.get("id") == model_id), None)
    label = entry.get("label", model_id) if entry else model_id

    pass_pct_start = 35 + idx * pct_per_model
    pass_pct_end = 35 + (idx + 1) * pct_per_model

    self.active_jobs[job_id]["status"] = f"[{idx+1}/{len(active_ids)}] Preflighting {label}..."
    self.active_jobs[job_id]["progress"] = pass_pct_start

    try:
        resolved = resolve_model(model_id, registry, provider_keys)
    except ModelRegistryError as exc:
        per_model_results[model_id] = {
            "label": label,
            "attempted": total_attempted,
            "extracted": 0,
            "failed_reason": str(exc),
        }
        continue

    llm_model = resolved["model"]
    llm_api_base = resolved["api_base"]
    llm_api_key = resolved["api_key"]
    if llm_model.startswith("ollama/") and llm_api_base.endswith("/v1"):
        llm_api_base = llm_api_base[:-3]

    try:
        await acompletion(
            model=llm_model,
            messages=[{"role": "user", "content": "ping"}],
            api_base=llm_api_base,
            api_key=llm_api_key,
            max_tokens=1, temperature=0,
        )
    except Exception as exc:
        per_model_results[model_id] = {
            "label": label,
            "attempted": total_attempted,
            "extracted": 0,
            "failed_reason": f"Preflight failed: {exc}",
        }
        # Continue to next model.
        continue

    extractor = LargeLLMExtractor(
        self.db,
        model=llm_model,
        api_base=llm_api_base,
        api_key=llm_api_key,
        model_id=model_id,
        model_label=label,
    )

    def _progress(msg: str, pct_inner: int, _label=label, _start=pass_pct_start, _end=pass_pct_end):
        # Map the extractor's 35→95 internal range onto this pass's allocated slice.
        local = max(0, min(60, pct_inner - 35))
        scaled = _start + (local / 60.0) * (_end - _start)
        if job_id in self.active_jobs:
            self.active_jobs[job_id]["status"] = f"[{idx+1}/{len(active_ids)}] {_label}: {msg}"
            self.active_jobs[job_id]["progress"] = int(scaled)

    try:
        extracted = await extractor.batch_extract(
            cleaned_payloads, repo=repo, progress_callback=_progress
        )
        per_model_results[model_id] = {
            "label": label,
            "attempted": total_attempted,
            "extracted": len(extracted) if extracted else 0,
        }
    except Exception as exc:
        per_model_results[model_id] = {
            "label": label,
            "attempted": total_attempted,
            "extracted": 0,
            "failed_reason": str(exc),
        }
        continue

# Final summary.
total_extracted = sum(r.get("extracted", 0) for r in per_model_results.values())
summary_parts = [
    f"{r['label']}: {r['extracted']}" + (" (FAILED)" if r.get("failed_reason") else "")
    for r in per_model_results.values()
]
self.active_jobs[job_id]["status"] = (
    f"Completed: {total_extracted} extractions across {len(per_model_results)} model(s) "
    f"— " + ", ".join(summary_parts)
)
self.active_jobs[job_id]["progress"] = 100
```

Remove the old code that built `llm_model = config.get("llm_model", ...)`, the single `extractor = LargeLLMExtractor(...)`, the single preflight, and the single `batch_extract` call.

- [ ] **Step 4.6: Run multi-model tests**

Run: `cd backend && python3 -m pytest tests/test_orchestrator_multi_model.py -v --tb=short`
Expected: All pass.

Then the full suite:
```bash
cd backend && python3 -m pytest tests/ -x --tb=short -q 2>&1 | tail -30
```

Un-skip any tests you marked `@unittest.skip("rewritten in Task 4")` in Task 2.6 and update them to use the new multi-model schema.

- [ ] **Step 4.7: Commit**

```bash
git add backend/pipeline/job_orchestrator.py backend/pipeline/semantic_fuser.py backend/llm/extractor.py backend/tests/test_orchestrator_multi_model.py backend/tests/
git commit -m "feat(pipeline): run distillation once per active model

The orchestrator now iterates llm_models_active, running the full
classify+extract pass per model on the same cached payloads. The
semantic fuser deduplicates across passes and tracks lineage via a new
metadata.merged_with_models field. Per-model results are reported in
the job status so partial failures don't abort the run."
```

---

## Task 5: End-to-end smoke test with a real config

**Files:**
- No code changes
- Manual verification against the running Docker backend

This is a sanity check on a live system, not a unit test.

- [ ] **Step 5.1: Rebuild the container**

```bash
cd /Users/mlautenschlager/Documents/Development/PR-Distiller
docker compose build backend
docker compose up -d backend
until curl -s http://localhost:8923/api/health | grep -q ok; do sleep 2; done
```

- [ ] **Step 5.2: Seed a two-model registry via the API**

```bash
TOKEN=$(docker exec pr-distiller-backend-1 cat /app/data/.api_token)
curl -s -X PUT http://localhost:8923/api/config \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "llm_models": [
      {"id":"qwen-local","label":"Qwen 2.5 Coder","model":"ollama/qwen2.5-coder:7b-instruct","api_base":"http://host.docker.internal:11434/v1","api_key_override":"","enabled":true},
      {"id":"gemma-local","label":"Gemma 4 E4B","model":"ollama/gemma4:e4b","api_base":"http://host.docker.internal:11434/v1","api_key_override":"","enabled":true}
    ],
    "llm_models_active": ["qwen-local","gemma-local"],
    "provider_api_keys": {}
  }'
```

Expected: `{"status":"ok"}`. Then GET /api/config — both entries should appear, `api_key_override` masked or empty.

- [ ] **Step 5.3: Trigger a small job using the cached crawl**

```bash
curl -s -X POST http://localhost:8923/api/jobs/start \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"repo":"tucowsinc/tdp-apis","months":1,"use_cache":true}'
```

Expected: `{"job_id":"job-...","status":"started"}`.

- [ ] **Step 5.4: Poll status and verify two passes ran**

```bash
JOB=<job-id-from-previous-step>
curl -s "http://localhost:8923/api/jobs/status/$JOB" -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
```

Watch for status text like `[1/2] Qwen 2.5 Coder: Classifying X/Y...` then `[2/2] Gemma 4 E4B: ...`. When done, response should include `per_model_results` with both ids.

- [ ] **Step 5.5: Inspect rules for provenance fields**

```bash
curl -s "http://localhost:8923/api/rules" -H "Authorization: Bearer $TOKEN" \
  | python3 -c "
import sys, json
rules = json.load(sys.stdin).get('rules', [])
with_origin = [r for r in rules if r['metadata'].get('extracted_by_model')]
merged = [r for r in rules if r['metadata'].get('merged_with_models')]
print(f'{len(rules)} rules, {len(with_origin)} with extracted_by_model, {len(merged)} cross-model-merged')
for r in merged[:3]:
    print(f'  {r[\"metadata\"][\"title_slug\"]} merged from: {r[\"metadata\"][\"merged_with_models\"]}')
"
```

Expected: A non-zero number of rules show `merged_with_models` containing both `qwen-local` and `gemma-local`.

- [ ] **Step 5.6: Commit the plan as done**

```bash
git add docs/superpowers/plans/2026-05-19-multi-model-pipeline.md
git commit -m "docs(plan): mark multi-model pipeline plan as completed"
```

---

## Definition of Done

- All new tests pass (`pytest tests/ -q`)
- All pre-existing tests pass (or have been correctly updated to the new schema — none are skipped at end of plan)
- A two-model run produces rules with `extracted_by_model` and at least some with non-empty `merged_with_models`
- `curl /api/config` returns `llm_models` and `llm_models_active`, with `api_key_override` masked
- `curl /api/jobs/status/$JOB` returns `per_model_results` with per-model attempted/extracted counts
- One configured-but-broken model (e.g. wrong api_base) does not abort the job; its slot in `per_model_results` shows `failed_reason`
