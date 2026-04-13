# Configuration

PR-Analysis is configured through two complementary mechanisms: environment variables (read at
startup by `backend/settings.py`) and a persistent JSON file (`backend/data/config.json`)
managed by `ConfigManager`. Environment variables establish the baseline; `config.json` stores
runtime overrides and encrypted secrets set through the web UI.

---

## Environment Variables

All env vars are defined in `backend/settings.py`. Set them in a `.env` file at the project
root (loaded by Docker Compose via `env_file`) or export them in your shell before running the
backend directly.

| Variable                  | Type    | Default                                    | Description                                                                            |
|---------------------------|---------|--------------------------------------------|----------------------------------------------------------------------------------------|
| `LLM_PROVIDER`            | string  | `ollama`                                   | Active provider hint used to populate UI defaults. Does not change request routing.   |
| `LLM_API_BASE`            | string  | `http://localhost:11434/v1`                | OpenAI-compatible base URL for LLM inference calls.                                   |
| `LLM_API_KEY`             | string  | *(empty)*                                  | API key forwarded to the LLM endpoint. Leave empty for local Ollama.                  |
| `LLM_MODEL`               | string  | `ollama/qwen2.5-coder:7b-instruct`         | litellm model identifier used for extraction and semantic fusion.                     |
| `EMBEDDING_MODEL`         | string  | `BAAI/bge-base-en-v1.5`                    | SentenceTransformer model name for ChromaDB embeddings. Changing this requires reindex.|
| `API_HOST`                | string  | `0.0.0.0`                                  | Interface the FastAPI server binds to.                                                 |
| `API_PORT`                | integer | `8923`                                     | Port for the FastAPI backend.                                                          |
| `CORS_ORIGINS`            | string  | `http://localhost:4096,http://localhost:3000` | Comma-separated allowed CORS origins.                                               |
| `GITHUB_TOKEN`            | string  | *(empty)*                                  | Personal access token for GitHub API crawling. Required for private repos.            |
| `GITHUB_WEBHOOK_SECRET`   | string  | *(empty)*                                  | HMAC-SHA256 secret for validating incoming GitHub webhooks. Optional.                 |
| `FERNET_KEY`              | string  | *(empty)*                                  | Base64 Fernet key for encrypting secrets at rest. Auto-generated if absent.           |
| `API_AUTH_TOKEN`          | string  | *(empty)*                                  | Bearer token for protecting backend API endpoints. Optional.                          |
| `DATA_DIR`                | string  | `backend/data`                             | Root directory for ChromaDB storage, config.json, and dev cache.                     |
| `DEDUP_DISTANCE_THRESHOLD`| float   | `0.32`                                     | Cosine distance below which two rules are considered duplicates and merged.           |
| `AUTO_APPROVE_CONFIDENCE` | float   | `0.8`                                      | Confidence score above which newly extracted rules are auto-approved to `active`.     |
| `WEBHOOK_MIN_INTERVAL`    | integer | `60`                                       | Minimum seconds between webhook-triggered pipeline runs per repository.               |

---

## Config File (config.json)

`backend/data/config.json` is written and read exclusively by `ConfigManager`. It persists
settings across restarts and stores encrypted versions of sensitive tokens using a Fernet
symmetric key kept at `backend/data/.secret_key` (auto-generated on first run, never committed).

**Do not edit `config.json` by hand.** Use the web UI Settings panel or `POST /api/config`.

### Fields

| Field               | Type   | Description                                                                         |
|---------------------|--------|-------------------------------------------------------------------------------------|
| `github_token_enc`  | string | Fernet-encrypted GitHub personal access token.                                      |
| `llm_provider`      | string | Active provider: `local`, `openai`, `anthropic`, `google`, `openrouter`.           |
| `llm_api_base`      | string | OpenAI-compatible base URL for the active LLM.                                     |
| `llm_model`         | string | litellm model identifier (e.g. `openai/Qwen/Qwen2.5-Coder-7B-Instruct`).          |
| `llm_api_key_enc`   | string | Fernet-encrypted API key for cloud LLM providers.                                  |
| `embedding_model`   | string | SentenceTransformer model for embeddings (matches `EMBEDDING_MODEL` env var).      |
| `repos`             | object | Map of `"owner/repo"` → per-repo configuration metadata.                           |
| `provider_models`   | object | Provider-keyed lists of `{id, label}` model options shown in the UI dropdown.      |
| `crawl_cursors`     | object | Map of `"owner/repo"` → last processed comment ID for incremental crawling.        |

### Relationship to Environment Variables

Environment variables set the startup defaults when `config.json` does not exist yet. Once
the file is present, `ConfigManager.load_config()` reads from the file and the env vars are
only used for fields not present in the file. The `JobOrchestrator` always reads from
`config.json` at job start, so changes made via the UI take effect for the next pipeline run
without restarting the server.

---

## LLM Presets

The `LLM_API_BASE` and `LLM_MODEL` variables follow [litellm](https://docs.litellm.ai/)
naming conventions. The provider prefix in the model name determines which litellm adapter
is used.

### Ollama (Default — local)

```env
LLM_PROVIDER=ollama
LLM_API_BASE=http://localhost:11434/v1
LLM_API_KEY=
LLM_MODEL=ollama/qwen2.5-coder:7b-instruct
```

Pull the model first: `ollama pull qwen2.5-coder:7b-instruct`

### vLLM (Self-hosted OpenAI-compatible)

```env
LLM_PROVIDER=local
LLM_API_BASE=http://your-vllm-host:8080/v1
LLM_API_KEY=dummy-key
LLM_MODEL=openai/Qwen/Qwen2.5-Coder-7B-Instruct
```

The `openai/` prefix tells litellm to use the OpenAI-compatible adapter against `LLM_API_BASE`.

### OpenAI

```env
LLM_PROVIDER=openai
LLM_API_BASE=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=openai/gpt-4o
```

### Anthropic

```env
LLM_PROVIDER=anthropic
LLM_API_BASE=https://api.anthropic.com
LLM_API_KEY=sk-ant-...
LLM_MODEL=anthropic/claude-3-5-sonnet
```

---

## Pipeline Tuning

### DEDUP_DISTANCE_THRESHOLD (default: `0.32`)

Controls how aggressively the `SemanticFuser` merges rules. Cosine distance in ChromaDB ranges
from `0.0` (identical) to `2.0` (opposite). At `0.32`, rules must be substantially similar in
meaning before they trigger an LLM merge.

- **Lower (e.g. 0.15)**: only near-identical rules merge. More rules in the store, less merging.
- **Higher (e.g. 0.45)**: broader merging. Fewer rules, but risks over-merging distinct concepts.

Tune by inspecting merge events in the server log and reviewing rule histories in the dashboard.

### AUTO_APPROVE_CONFIDENCE (default: `0.8`)

The LLM extractor returns a `confidence` float (`0.0`–`1.0`) for each extracted rule. Rules
above this threshold are immediately set to `active`; rules at or below enter `needs_review`.

- **Higher (e.g. 0.9)**: stricter auto-approval; more rules need manual review.
- **Lower (e.g. 0.7)**: more rules auto-approve; reduces review queue but increases noise risk.

### WEBHOOK_MIN_INTERVAL (default: `60`)

Minimum seconds between webhook-triggered pipeline runs for the same repository. Prevents a
burst of merged PRs from queueing multiple overlapping jobs. Adjust upward for high-volume
repositories, or downward for low-traffic repos where faster response is acceptable.
