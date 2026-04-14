# Getting Started

## Prerequisites

- Docker Engine 24+ and Docker Compose v2.20+
- GitHub Personal Access Token — Settings → Developer Settings → Personal access tokens → Fine-grained tokens. Required scopes: `contents: read` and `pull_requests: read` on the target repositories.
- 8 GB RAM minimum. 16 GB recommended if running a GPU-accelerated model.

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-org/PR-Distiller.git
cd PR-Distiller
```

**2. Create your environment file**

```bash
cp .env.example .env
```

**3. Set your GitHub token**

Open `.env` and set:

```
GITHUB_TOKEN=github_pat_your_token_here
```

**4. Run the preflight check**

```bash
make preflight
```

This verifies Docker, Docker Compose, your `.env` file, `GITHUB_TOKEN`, GPU availability, and port availability (8923, 4096, 11434). Fix any reported failures before continuing.

**5. Start all services**

```bash
make up
```

Services started: `ollama` (LLM inference), `backend` (FastAPI on port 8923), `web-ui` (Next.js on port 4096).

**6. Pull the default LLM**

```bash
make run-model
```

This pulls `qwen2.5-coder:7b-instruct` into Ollama. The model is approximately 4.7 GB. Run this once; the model is stored in a Docker volume and persists across restarts.

To use a different model:

```bash
make run-model MODEL=llama3.2:3b
```

**7. Open the dashboard**

Navigate to [http://localhost:4096](http://localhost:4096).

## First Extraction

You can start a rule extraction job from the web UI or via the API directly.

**Via the web UI:** Open [http://localhost:4096](http://localhost:4096), enter a GitHub repository in `owner/repo` format (e.g. `torvalds/linux`), configure the number of months to look back, and click **Start Job**.

**Via curl:**

```bash
curl -s -X POST http://localhost:8923/api/jobs/start \
  -H "Content-Type: application/json" \
  -d '{"repo": "owner/repo", "months": 2, "threshold": 0.45}' | jq .
```

Poll job status:

```bash
curl -s http://localhost:8923/api/jobs/status/<job_id> | jq .
```

Once complete, view extracted rules:

```bash
curl -s http://localhost:8923/api/rules | jq '.rules | length'
```

## Verify Installation

Check each service is healthy:

```bash
# Backend API
curl -s http://localhost:8923/api/rules | jq .

# Ollama — list loaded models
curl -s http://localhost:11434/api/tags | jq '.models[].name'

# Web UI — should return HTTP 200
curl -o /dev/null -sw "%{http_code}\n" http://localhost:4096
```

Check container health status:

```bash
docker compose ps
```

All three services (`backend`, `web-ui`, `ollama`) should show `healthy` or `running`.

View logs for a specific service:

```bash
docker compose logs -f backend
docker compose logs -f ollama
```

## macOS Setup

Docker Desktop on macOS does not support the NVIDIA runtime. Remove or rename `docker-compose.override.yml` before running `make up`:

```bash
mv docker-compose.override.yml docker-compose.gpu.yml
make up
```

Ollama will run on CPU. Inference is slower (roughly 5-15 tokens/second on Apple Silicon vs 60+ tokens/second on an NVIDIA GPU), but fully functional. Apple Silicon Macs with 16 GB unified memory handle 7B models comfortably.

## Next Steps

- [configuration.md](configuration.md) — LLM providers, embedding models, pipeline tuning
- [mcp-integration.md](mcp-integration.md) — Connect Cursor, Claude Code, or other MCP-compatible IDEs
- [security.md](security.md) — API authentication, token encryption, CORS, webhook verification
- [debugging.md](debugging.md) — Common issues and how to diagnose them
