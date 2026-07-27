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
    entry = next((e for e in registry if e["id"] == model_id), None)
    if entry is None:
        raise ModelRegistryError(f"Unknown model id: {model_id!r}")
    if not entry["enabled"]:
        raise ModelRegistryError(f"Model {model_id!r} is disabled")

    api_key = entry["api_key_override"]
    if not api_key:
        provider = _provider_from_model(entry["model"])
        api_key = provider_keys.get(provider, "")
        if not api_key and provider == "ollama":
            api_key = _OLLAMA_NO_AUTH_SENTINEL

    return {
        "model": entry["model"],
        "api_base": entry["api_base"],
        "api_key": api_key,
    }
