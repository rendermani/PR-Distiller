import contextlib
import fcntl
import os
import json
import logging
from cryptography.fernet import Fernet
import settings

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Centralizes settings securely.
    Writes persistently to `backend/data/config.json`.
    Automatically encrypts and decrypts sensitive tokens at-rest using a locally bootstrapped symmetric key.
    """
    REDACTION_SENTINEL = "***"

    def __init__(self):
        # Paths come from settings.DATA_DIR so that containerized and native
        # runs share one store. Deriving them from __file__ made the location
        # depend on where the code sits, silently splitting the two modes.
        self.config_path = os.path.join(settings.DATA_DIR, "config.json")
        self.key_path = os.path.join(settings.DATA_DIR, ".secret_key")
        self.cipher = self._build_cipher(self._resolve_key())
        self._ensure_default_config()

    @staticmethod
    def _build_cipher(key: bytes) -> Fernet:
        """Construct a Fernet from *key*, re-raising init errors with the env var name."""
        try:
            return Fernet(key)
        except Exception as exc:
            raise ValueError(
                "Failed to initialise encryption cipher. "
                "Check the FERNET_KEY environment variable is a 32-byte url-safe "
                f"base64-encoded value, or unset it to use the on-disk key file. "
                f"Underlying error: {exc}"
            ) from exc

    def _resolve_key(self) -> bytes:
        env_key = settings.FERNET_KEY
        if env_key:
            return env_key.encode() if isinstance(env_key, str) else env_key
        # EAFP read: avoids a TOCTOU between exists() and open().
        try:
            with open(self.key_path, "rb") as key_file:
                return key_file.read()
        except FileNotFoundError:
            os.makedirs(os.path.dirname(self.key_path), exist_ok=True)
            key = Fernet.generate_key()
            # O_EXCL so concurrent inits don't both clobber. Loser re-reads.
            try:
                fd = os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "wb") as key_file:
                    key_file.write(key)
                return key
            except FileExistsError:
                with open(self.key_path, "rb") as key_file:
                    return key_file.read()

    def _encrypt(self, text: str) -> str:
        if not text: return ""
        return self.cipher.encrypt(text.encode()).decode()

    def _decrypt(self, token: str) -> str:
        if not token: return ""
        try:
            return self.cipher.decrypt(token.encode()).decode()
        except Exception:
            return ""  # Gracefully fail if encryption key was rotated or corrupted

    @classmethod
    def _strip_redaction_sentinels(cls, payload: dict) -> dict:
        """Return *payload* with REDACTION_SENTINEL stripped from secret fields."""
        cleaned = dict(payload)
        sentinel = cls.REDACTION_SENTINEL
        for field in ("github_token", "huggingface_token", "github_webhook_secret"):
            if cleaned.get(field) == sentinel:
                cleaned.pop(field)
        if "provider_api_keys" in cleaned and isinstance(cleaned["provider_api_keys"], dict):
            cleaned["provider_api_keys"] = {
                p: k for p, k in cleaned["provider_api_keys"].items() if k != sentinel
            }
        if "llm_models" in cleaned and isinstance(cleaned["llm_models"], list):
            # Strip the sentinel from each entry's api_key_override so a UI that
            # round-trips a masked value doesn't overwrite the stored plaintext.
            # We replace with empty string; the API layer (Task 3) restores from
            # storage. At the config_manager level, empty means "no override".
            sanitized = []
            for entry in cleaned["llm_models"]:
                new_entry = dict(entry)
                if new_entry.get("api_key_override") == sentinel:
                    new_entry["api_key_override"] = ""
                sanitized.append(new_entry)
            cleaned["llm_models"] = sanitized
        return cleaned

    def _get_default_providers(self):
        return {
            "ollama": [
                {"id": "openai/google/gemma-4-E4B-it", "label": "Gemma 4 E4B (recommended, ~8GB, best Pareto per benchmark)"},
                {"id": "openai/Qwen/Qwen3-Coder-Next-FP8", "label": "Qwen3-Coder-Next FP8 (~75GB, highest quality)"},
                {"id": "openai/Qwen/Qwen3-Coder-30B-A3B-Instruct", "label": "Qwen3-Coder 30B MoE (3B active, ~60GB)"},
                {"id": "openai/google/gemma-4-26B-A4B-it", "label": "Gemma 4 26B-A4B MoE (4B active, ~52GB)"},
                {"id": "openai/Qwen/Qwen3-8B", "label": "Qwen 3 8B dense (~15GB)"},
                {"id": "ollama/qwen3:8b", "label": "Qwen 3 8B via Ollama (~5.2GB)"},
                {"id": "ollama/qwen3:4b", "label": "Qwen 3 4B via Ollama (~2.5GB, 256K ctx)"},
            ],
            "google": [
                {"id": "gemini/gemini-2.5-flash", "label": "Gemini 2.5 Flash"},
                {"id": "gemini/gemini-2.5-pro", "label": "Gemini 2.5 Pro"}
            ],
            "anthropic": [
                {"id": "anthropic/claude-3-haiku", "label": "Claude 3 Haiku"},
                {"id": "anthropic/claude-3-5-sonnet", "label": "Claude 3.5 Sonnet"}
            ],
            "openrouter": [
                {"id": "openrouter/auto", "label": "OpenRouter Auto Component"},
                {"id": "openrouter/anthropic/claude-3-opus", "label": "Claude 3 Opus"},
                {"id": "openrouter/meta-llama/llama-4-70b-instruct", "label": "Llama-4 70B Fast"},
                {"id": "openrouter/qwen/qwen-3.5-72b-instruct", "label": "Qwen 3.5 72B"}
            ],
            "openai": [
                {"id": "openai/gpt-4o", "label": "GPT-4 Omni"},
                {"id": "openai/gpt-4-turbo", "label": "GPT-4 Turbo"},
                {"id": "openai/gpt-3.5-turbo", "label": "GPT-3.5 Legacy"}
            ]
        }

    @staticmethod
    def _seed_models_from_env() -> list:
        """Build the initial llm_models registry from the LLM_* env vars.

        The multi-model refactor made `llm_models` the source of truth without a
        backfill path, leaving fresh and pre-refactor installs with an empty
        registry and no way to run a job. Seeding from the documented env vars
        reproduces the single-model behaviour operators configured via .env.

        Returns an empty list when LLM_MODEL is unset — there is nothing to
        seed, and inventing an endpoint would hide the misconfiguration.
        """
        if not settings.LLM_MODEL:
            return []
        return [{
            "id": "default",
            "label": settings.LLM_MODEL,
            "model": settings.LLM_MODEL,
            "api_base": settings.LLM_API_BASE,
            "api_key_override": settings.LLM_API_KEY,
            "enabled": True,
        }]

    def _ensure_default_config(self):
        if not os.path.exists(self.config_path):
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            seeded_models = self._seed_models_from_env()
            default_config = {
                "github_token": "",
                "github_webhook_secret": "",
                "huggingface_token": "",
                "llm_models": seeded_models,
                "llm_models_active": [e["id"] for e in seeded_models],
                "provider_api_keys": {},
                "embedding_model": "BAAI/bge-base-en-v1.5",
                "repos": {},
                "provider_models": self._get_default_providers(),
            }
            self.save_config(default_config)

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

    @contextlib.contextmanager
    def _file_lock(self):
        """Cross-process exclusive lock around the config file.

        Uses an fcntl flock on a side-car .lock file so we don't fight with
        the truncating write of the actual config. Threads in the same process
        share an fcntl lock, so this also serialises ThreadPoolExecutor calls.
        """
        lock_path = self.config_path + ".lock"
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _atomic_write_json(self, payload: dict) -> None:
        """Write *payload* to config_path via tmp + rename so a crash mid-write
        cannot leave a half-written or empty config file."""
        tmp_path = f"{self.config_path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.config_path)

    def save_config(self, new_config: dict):
        # Lock-load-mutate-write so concurrent callers don't lose updates and
        # the on-disk JSON never contains interleaved bytes from two writers.
        with self._file_lock():
            current = self.load_config()

            # Contract: strip the redaction sentinel '***' from any incoming
            # secret field so a programmatic caller (or a UI that round-trips
            # redacted values) cannot persist '***' as a real secret.
            new_config = self._strip_redaction_sentinels(new_config)

            # Deep-merge nested dicts so a partial update for one entry doesn't
            # wipe sibling entries written by a concurrent caller.
            existing_keys = dict(current.get("provider_api_keys", {}))
            existing_cursors = dict(current.get("crawl_cursors", {}))
            existing_repos = dict(current.get("repos", {}))
            incoming_keys = new_config.get("provider_api_keys")
            incoming_cursors = new_config.get("crawl_cursors")
            incoming_repos = new_config.get("repos")

            current.update(new_config)
            if incoming_keys is not None:
                current["provider_api_keys"] = {**existing_keys, **incoming_keys}
            if incoming_cursors is not None:
                current["crawl_cursors"] = {**existing_cursors, **incoming_cursors}
            if incoming_repos is not None:
                current["repos"] = {**existing_repos, **incoming_repos}

            provider_keys = current.get("provider_api_keys", {})
            provider_keys_enc = {p: self._encrypt(k) for p, k in provider_keys.items() if k}

            # Encrypt per-entry api_key_override values.
            llm_models_in = current.get("llm_models", [])
            llm_models_enc = []
            for entry in llm_models_in:
                enc_entry = dict(entry)
                plaintext_key = enc_entry.pop("api_key_override", "")
                enc_entry["api_key_override_enc"] = (
                    self._encrypt(plaintext_key) if plaintext_key else ""
                )
                llm_models_enc.append(enc_entry)

            encrypted_wrap = {
                "github_token_enc": self._encrypt(current.get("github_token", "")),
                "huggingface_token_enc": self._encrypt(current.get("huggingface_token", "")),
                "github_webhook_secret_enc": self._encrypt(current.get("github_webhook_secret", "")),
                "llm_models": llm_models_enc,
                "llm_models_active": current.get("llm_models_active", []),
                "provider_api_keys_enc": provider_keys_enc,
                "embedding_model": current.get("embedding_model", "BAAI/bge-base-en-v1.5"),
                "repos": current.get("repos", {}),
                "provider_models": current.get("provider_models", {}),
                "crawl_cursors": current.get("crawl_cursors", {}),
            }

            self._atomic_write_json(encrypted_wrap)

            return current

    def env_overrides(self) -> dict[str, bool]:
        """Return which secret fields are sourced from environment variables.

        UI uses this to disable inputs that would otherwise be ineffective
        (env wins over the encrypted-config value).
        """
        return {
            "github_token": bool(settings.GITHUB_TOKEN),
            "huggingface_token": bool(settings.HUGGINGFACE_HUB_TOKEN),
            "github_webhook_secret": bool(settings.GITHUB_WEBHOOK_SECRET),
            "provider_api_keys.openai": bool(settings.OPENAI_API_KEY),
            "provider_api_keys.anthropic": bool(settings.ANTHROPIC_API_KEY),
            "provider_api_keys.google": bool(settings.GOOGLE_API_KEY),
            "provider_api_keys.openrouter": bool(settings.OPENROUTER_API_KEY),
        }
