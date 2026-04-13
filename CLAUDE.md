# PR-Analysis — Agent Instructions

## Project Overview

PR-Analysis extracts reusable coding rules from GitHub PR review comments and serves them to AI coding assistants via MCP (Model Context Protocol). The system uses a two-pass LLM extraction pipeline with semantic deduplication.

## Architecture

- **Backend** (`backend/`): Python FastAPI on port 8923. ChromaDB for vector storage. LiteLLM for inference.
- **Web UI** (`web-ui/`): Next.js dashboard on port 4096.
- **MCP Server** (`mcp-server/`): TypeScript stdio server for Cursor/Claude Code integration.
- **LLM**: Ollama with Qwen 2.5 Coder 7B (default). Configurable via `LLM_*` env vars.

## Key Files

| File | Purpose |
|------|---------|
| `backend/api.py` | FastAPI app — all REST endpoints |
| `backend/settings.py` | Centralized env var configuration |
| `backend/db/lightrag_manager.py` | ChromaDB wrapper (rules, vectors, queries) |
| `backend/llm/extractor.py` | Two-pass LLM extraction (classifier + extractor) |
| `backend/pipeline/semantic_fuser.py` | Deduplication via cosine distance + LLM merge |
| `backend/pipeline/job_orchestrator.py` | Async pipeline job management |
| `backend/pipeline/config_manager.py` | Config file + Fernet encryption |
| `backend/pipeline/ast_slicer.py` | Multi-language AST slicing (tree-sitter) |
| `backend/scripts/deep_crawler.py` | GitHub PR/issue comment crawler |
| `mcp-server/src/index.ts` | MCP tool definitions |

## Development Commands

```bash
make preflight      # Check prerequisites
make up             # Start all services
make run-model      # Pull LLM into Ollama
make test           # Run backend tests
make test-coverage  # Tests with coverage report
```

## Conventions

- All configuration via environment variables; defaults in `backend/settings.py`.
- No hardcoded IPs, hostnames, or API keys in source code.
- New env vars must be added to both `settings.py` and `.env.example`.
- Tests use `unittest.TestCase` with mocked external services.
- Test files go in `backend/tests/`.
- API endpoints need corresponding tests in `tests/test_api_endpoints.py`.

## Testing

```bash
cd backend && python -m pytest tests/ -v                    # all tests
cd backend && python -m pytest tests/test_specific.py -v    # single file
cd backend && python -m pytest tests/ --cov=. --cov-report=term-missing  # coverage
```

## Environment

Copy `.env.example` to `.env`. Key variables:
- `LLM_API_BASE`: LLM endpoint (default: `http://localhost:11434/v1` for Ollama)
- `LLM_MODEL`: Model identifier (default: `ollama/qwen2.5-coder:7b-instruct`)
- `GITHUB_TOKEN`: Required for crawling PR comments
- `API_PORT`: Backend port (default: 8923)
- `CORS_ORIGINS`: Allowed CORS origins (comma-separated)
