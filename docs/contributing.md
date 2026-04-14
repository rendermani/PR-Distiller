# Contributing

## Development Setup

**Backend (Python / FastAPI)**

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

The backend requires Python 3.11+. All environment variables are read from the shell; copy `.env.example` to `.env` at the repo root and export the vars into your shell, or prefix test commands with `env $(cat ../.env | xargs)`.

**MCP Server (TypeScript)**

```bash
cd mcp-server
npm install
npm run build          # Compiles TypeScript to dist/
npm run dev            # Watch mode — recompiles on change
```

**Web UI (Next.js)**

```bash
cd web-ui
npm install
npm run dev            # Dev server on http://localhost:3000
```

Set `NEXT_PUBLIC_API_URL=http://localhost:8923` in a `.env.local` file inside `web-ui/` when running the web UI outside Docker.

## Running Tests

```bash
# Full backend test suite
make test

# With line-by-line coverage report (HTML in backend/htmlcov/)
make test-coverage

# Single test file
cd backend && python -m pytest tests/test_issue_13_webhook.py -v

# Single test case
cd backend && python -m pytest tests/test_issue_13_webhook.py::TestWebhookSignature::test_valid_signature -v --tb=short
```

All tests use `unittest.TestCase`. External services (Ollama, ChromaDB, GitHub API) are mocked with `unittest.mock`. Tests must not make real network calls.

## Code Style

**Python**

- Follow the patterns in the existing codebase. No unnecessary abstractions.
- `settings.py` is the single source of truth for configuration. Every env var is defined there; modules import from `settings`, never call `os.environ.get()` directly.
- Type annotations on all public functions.
- No silent `except Exception: pass` blocks.

**TypeScript**

- Strict mode is enabled (`tsconfig.json`). All types must be explicit — no `any` unless unavoidable and commented.
- Follow the MCP SDK patterns established in `mcp-server/src/index.ts`.

**Tests**

- Use `unittest.TestCase`, not bare functions.
- Mock all external services. Tests must pass with no network access and no running Docker containers.
- Test file names follow the pattern `test_<subject>.py`.

## Architecture Guidelines

**Adding a new environment variable**

1. Add it to `backend/settings.py` with a clear comment.
2. Add it to `.env.example` with an empty value and a one-line comment describing its purpose.
3. Add it to the `environment:` block of the `backend` service in `docker-compose.yml`.

**Adding a new API endpoint**

1. Add the route to `backend/api.py`.
2. Write tests in `backend/tests/test_api_endpoints.py` (or a new file if the surface area is large).
3. Document the endpoint in the relevant docs page.

**Adding a new LLM call**

Use `settings.LLM_API_BASE` and `settings.LLM_MODEL` — never hardcode URLs or model names. Follow the pattern in `backend/llm/extractor.py`.

**Pipeline modules**

The modules in `backend/pipeline/` must remain independent — no circular imports. The dependency direction is: `job_orchestrator` → individual pipeline modules. Pipeline modules must not import from each other or from `api.py`.

## Pull Request Process

1. Fork the repository and create a feature branch from `main`:

```bash
git checkout -b feat/your-feature-name
```

2. Write tests for new functionality before writing the implementation (red → green → refactor).

3. Ensure all tests pass and coverage does not regress:

```bash
make test-coverage
```

4. Lint the backend:

```bash
make lint
```

5. Commit with a message that explains *why* the change is needed, not just what changed.

6. Open a pull request against `main`. Include:
   - What problem it solves
   - How to test it manually
   - Any new env vars introduced
   - Links to related issues

## Reporting Issues

Use [GitHub Issues](https://github.com/your-org/PR-Distiller/issues) with the following structure:

- **Steps to reproduce** — exact commands or UI actions
- **Expected behavior** — what should have happened
- **Actual behavior** — what happened instead, including full error messages and relevant log output
- **Environment** — OS, Docker version, model being used, contents of relevant `.env` vars (redact token values)
