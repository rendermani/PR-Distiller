# Security

## Network Model

PR-Analysis is designed for local development and trusted-network use. The API server binds to `0.0.0.0:8923` by default and has no built-in authentication unless explicitly configured. Do not expose port 8923 to the public internet without enabling authentication and placing the service behind a TLS-terminating reverse proxy.

## Optional API Authentication

Set `API_AUTH_TOKEN` in `.env` to require a Bearer token on all API requests:

```
API_AUTH_TOKEN=your-secret-token-here
```

When set, every request to the backend must include:

```
Authorization: Bearer your-secret-token-here
```

Requests without a valid token receive `401 Unauthorized`. The MCP server reads `PR_DISTILLER_API_URL` and inherits the token from environment — set `PR_DISTILLER_API_TOKEN` in the MCP server's environment to match.

## GitHub Webhook Validation

When `GITHUB_WEBHOOK_SECRET` is set, the `/api/webhooks/github` endpoint validates the `X-Hub-Signature-256` header on every incoming payload using HMAC-SHA256:

```
GITHUB_WEBHOOK_SECRET=a-random-high-entropy-string
```

Requests with a missing or invalid signature are rejected with `401 Unauthorized`. To generate a suitable secret:

```bash
openssl rand -hex 32
```

Configure the same value in your GitHub repository's webhook settings under **Settings → Webhooks → Secret**.

## Token Storage

GitHub tokens and LLM API keys entered via the web UI settings panel are stored encrypted in `backend/data/config.json` using [Fernet](https://cryptography.io/en/latest/fernet/) symmetric encryption.

By default, the encryption key is auto-generated on first start and stored in `backend/data/.secret_key`. To supply your own key (recommended for production-like deployments):

```bash
# Generate a Fernet key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Set it in `.env`:

```
FERNET_KEY=your-base64-fernet-key-here
```

**Key rotation:** If `FERNET_KEY` changes, previously encrypted tokens in `config.json` become unreadable. Re-enter tokens via the settings panel after rotating the key.

## CORS Configuration

The `CORS_ORIGINS` variable controls which origins the backend accepts cross-origin requests from:

```
CORS_ORIGINS=http://localhost:4096,http://localhost:3000
```

Multiple origins are comma-separated. The default allows the bundled web UI on port 4096 and Next.js dev server on port 3000. Restrict this to only the origins that need access.

## Secrets Management

- Never commit `.env` to version control. The repository's `.gitignore` excludes it.
- Use `.env.example` as the canonical template. Add new variables there (with empty or example values) whenever a new env var is introduced in `settings.py`.
- The `backend/data/` directory contains `config.json` and `.secret_key`. Both are excluded from version control and should not be shared.
- GitHub tokens stored in `config.json` are encrypted at rest, but the encryption key in `.secret_key` (or `FERNET_KEY`) must be protected with the same diligence as the token itself.
- Rotate `FERNET_KEY` if the key file is compromised. After rotation, re-enter all tokens via the settings panel.

## Rate Limiting

The webhook endpoint enforces a minimum interval between pipeline triggers per repository (default: 60 seconds, configurable via `WEBHOOK_MIN_INTERVAL`). This prevents runaway extraction jobs if a repository receives many PRs in quick succession. The API does not currently rate-limit non-webhook endpoints — implement a reverse proxy (e.g., nginx with `limit_req`) if you expose the API beyond a trusted network.

## Reporting Vulnerabilities

Report security issues by opening a GitHub Issue marked **[Security]** in the title, or contact the repository owner directly via the email listed in their GitHub profile. Please do not include exploit details in public issues — describe the class of vulnerability and wait for acknowledgement before disclosing details.
