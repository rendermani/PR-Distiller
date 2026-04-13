import os
import json
from cryptography.fernet import Fernet
import settings

class ConfigManager:
    """
    Centralizes settings securely.
    Writes persistently to `backend/data/config.json`.
    Automatically encrypts and decrypts sensitive tokens at-rest using a locally bootstrapped symmetric key.
    """
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = os.path.join(self.base_dir, "data", "config.json")
        self.key_path = os.path.join(self.base_dir, "data", ".secret_key")
        self._ensure_encryption_key()
        self.cipher = Fernet(self._load_key())
        self._ensure_default_config()

    def _ensure_encryption_key(self):
        if not os.path.exists(self.key_path):
            os.makedirs(os.path.dirname(self.key_path), exist_ok=True)
            key = Fernet.generate_key()
            with open(self.key_path, "wb") as key_file:
                key_file.write(key)

    def _load_key(self):
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

    def _get_default_providers(self):
        return {
            "local": [
                {"id": "openai/Qwen/Qwen2.5-Coder-7B-Instruct", "label": "Qwen 2.5 Coder 7B"},
                {"id": "openai/Qwen/Qwen2.5-Coder-14B-Instruct", "label": "Qwen 2.5 Coder 14B"},
                {"id": "openai/meta-llama/Llama-4-8b-Instruct", "label": "Llama-4 8B"},
                {"id": "openai/google/gemma-4-9b-it", "label": "Gemma-4 9B"},
                {"id": "openai/google/gemma-4-27b-it", "label": "Gemma-4 27B (Q4)"},
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

    def _ensure_default_config(self):
        if not os.path.exists(self.config_path):
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            default_config = {
                "github_token": "",
                "llm_provider": settings.LLM_PROVIDER,
                "llm_api_base": settings.LLM_API_BASE,
                "llm_model": settings.LLM_MODEL,
                "llm_api_key": settings.LLM_API_KEY,
                "embedding_model": "BAAI/bge-base-en-v1.5",
                "repos": {},
                "provider_models": self._get_default_providers()
            }
            self.save_config(default_config)

    def load_config(self) -> dict:
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            pm = raw.get("provider_models", {})
            if not pm:
                pm = self._get_default_providers()

            return {
                "github_token": self._decrypt(raw.get("github_token_enc", "")),
                "llm_provider": raw.get("llm_provider", "local"),
                "llm_api_base": raw.get("llm_api_base", settings.LLM_API_BASE),
                "llm_model": raw.get("llm_model", ""),
                "llm_api_key": self._decrypt(raw.get("llm_api_key_enc", "")),
                "embedding_model": raw.get("embedding_model", "BAAI/bge-base-en-v1.5"),
                "repos": raw.get("repos", {}),
                "provider_models": pm,
                "crawl_cursors": raw.get("crawl_cursors", {})
            }
        except Exception:
            return {}

    def save_config(self, new_config: dict):
        current = self.load_config()
        current.update(new_config)

        # Preserve system configs while enforcing physical encryption
        encrypted_wrap = {
            "github_token_enc": self._encrypt(current.get("github_token", "")),
            "llm_provider": current.get("llm_provider", "local"),
            "llm_api_base": current.get("llm_api_base", ""),
            "llm_model": current.get("llm_model", ""),
            "llm_api_key_enc": self._encrypt(current.get("llm_api_key", "")),
            "embedding_model": current.get("embedding_model", "BAAI/bge-base-en-v1.5"),
            "repos": current.get("repos", {}),
            "provider_models": current.get("provider_models", {}),
            "crawl_cursors": current.get("crawl_cursors", {})
        }

        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(encrypted_wrap, f, indent=4)

        return current
