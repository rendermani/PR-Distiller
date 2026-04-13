# Compatibility

## Operating Systems

| OS | Status | Notes |
|---|---|---|
| Linux x86_64 | Full support | GPU acceleration with NVIDIA via `docker-compose.override.yml` |
| macOS Apple Silicon (M1/M2/M3/M4) | Full support | CPU inference via Ollama; remove `docker-compose.override.yml` |
| macOS Intel | Full support | CPU inference; remove `docker-compose.override.yml` |
| Windows (WSL2) | Supported | Docker Desktop with WSL2 backend required; NVIDIA GPU passthrough supported via WSL2 |

## Docker Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Docker Engine | 24.0 | Latest stable |
| Docker Compose | v2.20 | Latest stable |
| Available disk | 10 GB | 20 GB (model storage) |
| RAM | 8 GB | 16 GB |

## GPU Support

| GPU | Status | Notes |
|---|---|---|
| NVIDIA (CUDA 12+) | Supported | Requires `docker-compose.override.yml` (auto-loaded on Linux) |
| AMD (ROCm) | Not tested | May work with the `ollama/ollama:rocm` image; manual `docker-compose.override.yml` edit required |
| Apple Metal (M-series) | Supported | Ollama uses Metal natively inside Docker Desktop on macOS |
| CPU only | Supported | All features work; 7B model inference is 5–20 tokens/second depending on CPU |

To disable GPU (e.g., on a CPU-only Linux machine):

```bash
docker compose -f docker-compose.yml up -d
```

Or rename the override file so it is no longer auto-loaded:

```bash
mv docker-compose.override.yml docker-compose.gpu.yml
```

## LLM Compatibility

The backend uses an OpenAI-compatible `/v1/chat/completions` interface (`LLM_API_BASE`). Any model served by Ollama, LM Studio, vLLM, or a remote provider that exposes this interface is compatible.

| Model | Provider | Status | Notes |
|---|---|---|---|
| qwen2.5-coder:7b-instruct | Ollama | Tested, default | Best extraction quality at the 7B tier |
| qwen2.5-coder:14b-instruct | Ollama | Tested | Better reasoning, requires 16 GB RAM |
| llama3.2:3b | Ollama | Tested | Faster, lower quality extractions |
| llama3.1:8b | Ollama | Tested | Good general-purpose alternative |
| gemma3:9b | Ollama | Tested | Competitive with Llama 3.1 8B |
| gemini-2.5-flash | Google AI | Compatible | Set `LLM_PROVIDER=gemini`, `LLM_API_KEY` |
| claude-3-haiku | Anthropic | Compatible | Set `LLM_PROVIDER=anthropic`, `LLM_API_KEY` |
| gpt-4o | OpenAI | Compatible | Set `LLM_PROVIDER=openai`, `LLM_API_KEY` |

## Embedding Model Compatibility

The default embedding model is `BAAI/bge-base-en-v1.5` (768-dimensional vectors). Changing `EMBEDDING_MODEL` after rules have been stored requires running `POST /api/admin/reindex` to re-embed all existing rules. Mismatched dimensions will cause ChromaDB query errors.

## Language Support (AST Parsing)

The pipeline uses language-specific parsers to extract precise code context windows. When no parser is available, it falls back to a simple line-window approach (~20 lines around the diff).

| Language | Parser | Status |
|---|---|---|
| Python | stdlib `ast` | Full support — function and class boundary extraction |
| JavaScript / JSX | tree-sitter-javascript | Full support |
| TypeScript / TSX | tree-sitter-typescript | Full support |
| Go | tree-sitter-go | Full support |
| Rust | tree-sitter-rust | Full support |
| Ruby, Java, C, C++, others | Line-window fallback | Partial — ±20 line context only |

## Browser Support

The web UI is a standard Next.js application and works in all modern browsers:

- Chrome / Chromium 100+
- Firefox 100+
- Safari 15+
- Edge 100+

Internet Explorer is not supported.

## Python Version

The backend requires Python 3.11 or later. The Docker image pins to `python:3.11-slim`. Running outside Docker on Python 3.10 or earlier is not supported.
