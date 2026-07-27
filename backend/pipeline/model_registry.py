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
# LiteLLM rejects empty api_key strings; Ollama requires no auth, so we
# pass this sentinel value to satisfy LiteLLM while signaling "no auth needed."
_OLLAMA_NO_AUTH_SENTINEL = "unused"


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
    # `model` and `api_base` were previously checked for presence only, so
    # None or "" passed validation and then failed deep inside resolve_model
    # (TypeError) or produced an all-empty inference config.
    for field in ("model", "api_base"):
        value = entry[field]
        if not isinstance(value, str) or not value:
            raise ModelRegistryError(
                f"Entry {entry_id!r} field {field!r} must be a non-empty string, "
                f"got {value!r}"
            )
    if not isinstance(entry["enabled"], bool):
        raise ModelRegistryError("Entry 'enabled' must be a bool")


def validate_registry(registry: list) -> None:
    """Validate every entry and reject duplicate ids.

    Ids must be unique because llm_models_active refers to entries by id and
    resolve_model resolves by first match: a duplicate silently shadowed the
    later entry, which the UI still displayed as though it were live.
    """
    if not isinstance(registry, list):
        raise ModelRegistryError(
            f"Registry must be a list, got {type(registry).__name__}"
        )
    seen = set()
    for entry in registry:
        validate_entry(entry)
        entry_id = entry["id"]
        if entry_id in seen:
            raise ModelRegistryError(f"Duplicate model id in registry: {entry_id!r}")
        seen.add(entry_id)


def _provider_from_model(model: str) -> str:
    """Extract LiteLLM provider prefix from a model string.

    `ollama/qwen3:8b` -> `ollama`
    `openai/gpt-4o` -> `openai`
    `gpt-4o` (no prefix) -> `openai` (LiteLLM default)
    """
    if "/" in model:
        return model.split("/", 1)[0]
    return "openai"


# LiteLLM's model prefix and the provider name this project stores keys under
# are not always the same word. Gemini models carry the `gemini/` prefix, while
# the built-in model list, env_overrides, the .env var (GOOGLE_API_KEY) and the
# Vault page all use `google` — so a key pasted into the Google field was looked
# up under `gemini` and never found.
_PROVIDER_KEY_ALIASES = {
    "gemini": ("gemini", "google"),
}


def _lookup_provider_key(provider: str, provider_keys: dict) -> str:
    """First non-empty key among the provider's accepted aliases."""
    for name in _PROVIDER_KEY_ALIASES.get(provider, (provider,)):
        key = provider_keys.get(name, "")
        if key:
            return key
    return ""


def api_key_for_call(model: str | None, api_key: str | None) -> str:
    """The api_key to hand LiteLLM for *model*.

    Ollama needs no auth but LiteLLM rejects an empty api_key, hence the
    sentinel. For every other provider an empty key is a configuration gap and
    must stay empty so the failure names the missing credential rather than
    surfacing as an opaque upstream 401.
    """
    if api_key:
        return api_key
    if (model or "").startswith("ollama/"):
        return _OLLAMA_NO_AUTH_SENTINEL
    return ""


def resolve_model(model_id: str, registry: list, provider_keys: dict) -> dict:
    """Resolve a registry id to an inference config triple.

    Returns: {"model": str, "api_base": str, "api_key": str}
    Raises: ModelRegistryError if the id is unknown, the entry is disabled, or
        the entry is malformed.

    Every failure raises ModelRegistryError. Callers that run several models in
    sequence (job_orchestrator) filter on that type to isolate one bad model
    from the rest of the job; leaking a KeyError or TypeError from here escaped
    that filter and aborted the whole run.
    """
    # .get() rather than e["id"]: a legacy or hand-edited entry missing "id"
    # must not raise KeyError while searching for an unrelated model.
    entry = next((e for e in registry if isinstance(e, dict) and e.get("id") == model_id), None)
    if entry is None:
        raise ModelRegistryError(f"Unknown model id: {model_id!r}")

    # Validate before dereferencing, so a malformed entry produces a precise
    # message instead of a TypeError from deeper in the call.
    validate_entry(entry)

    if not entry["enabled"]:
        raise ModelRegistryError(f"Model {model_id!r} is disabled")

    api_key = entry["api_key_override"]
    if not api_key:
        provider = _provider_from_model(entry["model"])
        api_key = _lookup_provider_key(provider, provider_keys)
        if not api_key and provider == "ollama":
            api_key = _OLLAMA_NO_AUTH_SENTINEL

    return {
        "model": entry["model"],
        "api_base": entry["api_base"],
        "api_key": api_key,
    }
