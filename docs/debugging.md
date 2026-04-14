# Debugging

## Health Checks

Run these first to determine which component is unhealthy:

```bash
# Backend API — returns {"rules": [...]} if healthy
curl -s http://localhost:8923/api/rules | jq 'keys'

# Ollama — returns list of loaded models
curl -s http://localhost:11434/api/tags | jq '.models[].name'

# Web UI — should return 200
curl -o /dev/null -sw "HTTP %{http_code}\n" http://localhost:4096

# All container health states
docker compose ps
```

## Common Issues

### "Connection refused" on port 8923

The backend container is not running or not yet healthy.

```bash
# Check container states
docker compose ps

# Check backend startup logs
docker compose logs backend --tail=50
```

The backend waits for Ollama to pass its health check before starting. If Ollama is slow to start (e.g., on first run while pulling an image), the backend may restart a few times — this is expected. Wait 30–60 seconds and check again.

### Ollama model not found

The model must be explicitly pulled after the container starts. It is not bundled in the Docker image.

```bash
# Pull the default model
make run-model

# Or pull a specific model
make run-model MODEL=llama3.2:3b

# List models currently available in Ollama
docker compose exec ollama ollama list
```

### "No rules extracted" after a pipeline run

Check in order:

1. **LLM is reachable:** `curl -s http://localhost:11434/api/tags` should list at least one model.
2. **GitHub token is set:** `grep GITHUB_TOKEN .env` should return a non-empty value.
3. **Backend logs show the extraction pipeline running:**

```bash
docker compose logs backend | grep -E '\[Two-Pass\]|\[Extractor\]|\[Fuser\]'
```

If the log shows `[Two-Pass] 0 comments found`, the GitHub token may lack `pull_requests: read` permission, or the repository has no merged PRs in the selected time window.

### ChromaDB dimension mismatch

This error appears when `EMBEDDING_MODEL` is changed after rules have already been stored. The existing vectors were computed with a different embedding dimension and are incompatible with the new model.

Fix by re-embedding all stored rules:

```bash
curl -s -X POST http://localhost:8923/api/admin/reindex | jq .
```

This re-computes embeddings for every rule using the current `EMBEDDING_MODEL`. The operation may take several minutes if there are many rules.

### Web UI shows "Error fetching rules"

The web UI cannot reach the backend API.

1. Confirm the backend is running: `docker compose ps`
2. Check `NEXT_PUBLIC_API_URL` in `.env` — it must match the actual backend port:

```
NEXT_PUBLIC_API_URL=http://localhost:8923
```

3. If you changed the port via `API_PORT`, update `NEXT_PUBLIC_API_URL` to match and rebuild the web UI image:

```bash
docker compose build web-ui
docker compose up -d web-ui
```

4. Check CORS: `CORS_ORIGINS` must include the origin the browser is loading the UI from. Default is `http://localhost:4096,http://localhost:3000`.

### Webhook returns 401

`GITHUB_WEBHOOK_SECRET` is set but the signature does not match. Verify the secret value in `.env` exactly matches the secret configured in GitHub's webhook settings. Trailing spaces or newlines in `.env` are a common cause.

## Viewing Logs

```bash
# Follow all services
docker compose logs -f

# Follow a single service
docker compose logs -f backend
docker compose logs -f ollama
docker compose logs -f web-ui

# Last 100 lines from backend
docker compose logs backend --tail=100
```

## Running Tests

```bash
# Full test suite
make test

# With coverage report (HTML output in backend/htmlcov/)
make test-coverage

# Single test file
cd backend && python -m pytest tests/test_api_endpoints.py -v

# Single test case
cd backend && python -m pytest tests/test_issue_13_webhook.py::TestWebhookSignature -v --tb=short
```

## Resetting State

To wipe all extracted rules and start fresh:

```bash
# Stop services and remove volumes (this deletes all ChromaDB data and Ollama models)
make clean

# Or remove only the backend data volume, preserving Ollama models
docker compose down
docker volume rm pr-distiller_backend_data
docker compose up -d
```
