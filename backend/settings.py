"""Centralized configuration loaded from environment variables.

Every tunable in the system is defined here. Modules import from this
file instead of calling os.environ.get() with scattered defaults.
"""
import os

# --- LLM Inference ---
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
LLM_API_BASE = os.environ.get("LLM_API_BASE", "http://localhost:11434/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "ollama/qwen3:8b")

# --- Embedding ---
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

# --- API Server ---
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8923"))
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:4096,http://localhost:3000").split(",")

# --- GitHub ---
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_WEBHOOK_SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET", "")

# --- Security ---
FERNET_KEY = os.environ.get("FERNET_KEY", "")
API_AUTH_TOKEN = os.environ.get("API_AUTH_TOKEN", "")  # optional Bearer token for API auth

# --- Data Paths ---
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
DEV_CACHE_DIR = os.path.join(DATA_DIR, "dev_cache")
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")

# --- Pipeline ---
DEDUP_DISTANCE_THRESHOLD = float(os.environ.get("DEDUP_DISTANCE_THRESHOLD", "0.32"))
AUTO_APPROVE_CONFIDENCE = float(os.environ.get("AUTO_APPROVE_CONFIDENCE", "0.8"))
WEBHOOK_MIN_INTERVAL = int(os.environ.get("WEBHOOK_MIN_INTERVAL", "60"))
