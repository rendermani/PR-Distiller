# API Reference

Base URL: `http://localhost:8923`

All endpoints return JSON. Errors are returned as `{"detail": "<message>"}` with an appropriate
HTTP status code. If `API_AUTH_TOKEN` is set, include `Authorization: Bearer <token>` on every
request.

---

## Rules

### GET /api/rules

Returns all extracted rules, sorted descending by `occurrence_count`. Optionally filter by repo.

**Query parameters**

| Parameter | Type   | Required | Description                           |
|-----------|--------|----------|---------------------------------------|
| `repo`    | string | No       | Filter to a single `owner/repo`.      |

**Response**

```json
{
  "rules": [
    {
      "id": "a1b2c3d4",
      "document": "Rule: Always validate JWT expiry — Context: ... Enforce: ...",
      "metadata": {
        "rule_id": "a1b2c3d4",
        "repo": "acme/backend",
        "status": "active",
        "confidence": 0.92,
        "category": "security",
        "occurrence_count": 7,
        "path_patterns": "**/auth/*.py"
      }
    }
  ]
}
```

```bash
curl http://localhost:8923/api/rules
curl "http://localhost:8923/api/rules?repo=acme/backend"
```

---

### POST /api/rules

Manually adds a new rule. The rule enters `needs_review` status.

**Request body**

```json
{
  "repo": "acme/backend",
  "title": "Validate input at service boundaries",
  "description": "All data crossing a service boundary must be validated before processing.",
  "enforcement": "Check that every public method validates its arguments with Pydantic or equivalent."
}
```

**Response**

```json
{ "status": "created", "rule_id": "e5f6a7b8" }
```

```bash
curl -X POST http://localhost:8923/api/rules \
  -H "Content-Type: application/json" \
  -d '{"repo":"acme/backend","title":"Validate at boundaries","description":"...","enforcement":"..."}'
```

---

### DELETE /api/rules/{rule_id}

Permanently deletes a rule from the vector store.

**Response**

```json
{ "status": "deleted" }
```

```bash
curl -X DELETE http://localhost:8923/api/rules/a1b2c3d4
```

---

### POST /api/rules/{rule_id}/approve

Transitions a rule from `needs_review` to `active`.

**Response**

```json
{ "status": "success", "rule_id": "a1b2c3d4", "state": "approved" }
```

```bash
curl -X POST http://localhost:8923/api/rules/a1b2c3d4/approve
```

---

### POST /api/rules/{rule_id}/block

Sets a rule to `blocked`. Blocked rules are excluded from MCP queries and exports. Future
extractions semantically similar to a blocked rule are discarded automatically.

**Response**

```json
{ "status": "success", "rule_id": "a1b2c3d4", "state": "blocked" }
```

```bash
curl -X POST http://localhost:8923/api/rules/a1b2c3d4/block
```

---

### GET /api/rules/{rule_id}/history

Returns the merge lineage for a rule: all rules that were merged into it by the Semantic Fuser.

**Response**

```json
{
  "history": [
    {
      "original_rule_id": "old123",
      "merged_into": "a1b2c3d4",
      "document": "Rule: ...",
      "repo": "acme/backend",
      "occurrence_count": 3
    }
  ]
}
```

```bash
curl http://localhost:8923/api/rules/a1b2c3d4/history
```

---

### POST /api/rules/{rule_id}/feedback

Records whether an IDE agent applied or dismissed a rule suggestion. Drives effectiveness
tracking and automatic promotion of `needs_review` rules.

**Request body**

```json
{ "action": "applied" }
```

`action` must be `"applied"` or `"dismissed"`.

**Response**

```json
{ "status": "recorded", "rule_id": "a1b2c3d4", "action": "applied" }
```

Returns `404` if the rule does not exist.

```bash
curl -X POST http://localhost:8923/api/rules/a1b2c3d4/feedback \
  -H "Content-Type: application/json" \
  -d '{"action":"applied"}'
```

---

### GET /api/rules/{rule_id}/effectiveness

Returns per-rule effectiveness statistics.

**Response**

```json
{
  "rule_id": "a1b2c3d4",
  "times_served": 12,
  "times_applied": 9,
  "times_dismissed": 3,
  "acceptance_rate": 0.75
}
```

Returns `404` if the rule does not exist.

```bash
curl http://localhost:8923/api/rules/a1b2c3d4/effectiveness
```

---

## Query (MCP)

### POST /api/mcp/query

Performs a semantic search over active rules for a given code diff. Called by the MCP server;
can also be called directly for testing.

**Request body**

```json
{
  "code_diff": "- token = request.headers.get('auth')\n+ token = request.headers.get('Authorization')",
  "top_k": 5,
  "categories": ["security", "correctness"],
  "file_path": "src/auth/middleware.py"
}
```

| Field        | Type          | Required | Description                                                   |
|--------------|---------------|----------|---------------------------------------------------------------|
| `code_diff`  | string        | Yes      | The code snippet or git diff to match against stored rules.   |
| `top_k`      | integer       | No       | Max rules to return (default: 5).                             |
| `categories` | string array  | No       | Restrict to these categories. Omit for all categories.        |
| `file_path`  | string        | No       | Current file path; triggers path_patterns scoping when set.  |

**Response**

```json
{
  "matched_rules": [
    {
      "document": "Rule: Validate JWT expiry — Context: ...",
      "metadata": { "rule_id": "a1b2c3d4", "category": "security", "occurrence_count": 7 }
    }
  ]
}
```

```bash
curl -X POST http://localhost:8923/api/mcp/query \
  -H "Content-Type: application/json" \
  -d '{"code_diff":"+ return token","top_k":3}'
```

---

## Jobs

### POST /api/jobs/start

Starts an extraction pipeline run for a repository. Runs asynchronously; returns a `job_id`
to poll for status.

**Request body**

```json
{
  "repo": "acme/backend",
  "months": 2,
  "threshold": 0.45,
  "use_cache": false
}
```

| Field       | Type    | Required | Description                                                      |
|-------------|---------|----------|------------------------------------------------------------------|
| `repo`      | string  | Yes      | Repository in `owner/repo` format.                              |
| `months`    | integer | No       | How many months back to crawl (default: 2).                     |
| `threshold` | float   | No       | Minimum similarity threshold for extraction filtering.           |
| `use_cache` | boolean | No       | Use the local dev cache if available (skips GitHub API calls).  |

**Response**

```json
{ "job_id": "job-1712345678", "status": "started" }
```

```bash
curl -X POST http://localhost:8923/api/jobs/start \
  -H "Content-Type: application/json" \
  -d '{"repo":"acme/backend","months":3}'
```

---

### GET /api/jobs/status/{job_id}

Returns the current status and progress of a running or completed job.

**Response**

```json
{
  "status": "Batch Processing 142 rule payloads with openai/gpt-4o",
  "progress": 35,
  "trigger_source": "manual"
}
```

`progress` is `0`–`100` while running, `100` on success, `-1` on failure or cancellation.

```bash
curl http://localhost:8923/api/jobs/status/job-1712345678
```

---

### POST /api/jobs/cancel/{job_id}

Requests cancellation of a running job. The pipeline checks the cancel flag between stages
and stops at the next safe checkpoint.

**Response**

```json
{ "status": "cancelled" }
```

Returns `{"status": "not_found"}` if the `job_id` is unknown.

```bash
curl -X POST http://localhost:8923/api/jobs/cancel/job-1712345678
```

---

## Config

### GET /api/config

Returns the current configuration. Sensitive fields (`github_token`, `llm_api_key`) are
redacted to `"***"` in the response.

```bash
curl http://localhost:8923/api/config
```

---

### POST /api/config

Updates one or more configuration fields. Accepts a partial object; fields not included are
preserved. Sensitive tokens are encrypted before being written to `config.json`.

```bash
curl -X POST http://localhost:8923/api/config \
  -H "Content-Type: application/json" \
  -d '{"llm_model":"openai/gpt-4o","llm_api_key":"sk-..."}'
```

---

## Webhooks

### POST /api/webhooks/github

Receives GitHub webhook payloads. Only `pull_request` events with `action: closed` and
`merged: true` trigger a pipeline run. Other events return `{"status": "ignored"}`.

When `GITHUB_WEBHOOK_SECRET` is set, the `X-Hub-Signature-256` header is validated. Requests
with a missing or invalid signature return `401`.

Rate-limited to one trigger per repository per `WEBHOOK_MIN_INTERVAL` seconds (default: 60).

```bash
# Configure in GitHub: Settings → Webhooks → Add webhook
# Payload URL: https://your-host/api/webhooks/github
# Content type: application/json
# Secret: value of GITHUB_WEBHOOK_SECRET
# Events: Pull requests
```

---

### GET /api/webhooks/stats

Returns per-repository webhook trigger counts and timestamps.

**Response**

```json
{
  "repos": {
    "acme/backend": {
      "trigger_count": 14,
      "last_trigger_time": 1712345678.123
    }
  }
}
```

```bash
curl http://localhost:8923/api/webhooks/stats
```

---

## Admin

### POST /api/admin/reindex

Re-embeds all rules using the current `EMBEDDING_MODEL`. Required after changing the embedding
model to prevent dimension mismatches. Returns the number of rules re-indexed.

**Response**

```json
{ "status": "success", "reindexed_count": 248 }
```

```bash
curl -X POST http://localhost:8923/api/admin/reindex
```

---

### GET /api/export

Exports active rules for a repository as an agent prompt or system instruction block.

**Query parameters**

| Parameter | Type   | Required | Description                                                    |
|-----------|--------|----------|----------------------------------------------------------------|
| `repo`    | string | Yes      | Repository in `owner/repo` format.                            |
| `type`    | string | No       | Export format: `prompt` (default) or `json`.                  |
| `arch`    | string | No       | Target architecture: `openai` (default), `anthropic`, `raw`.  |

**Response**

```json
{ "content": "<formatted rules as system prompt or JSON>" }
```

```bash
curl "http://localhost:8923/api/export?repo=acme/backend&type=prompt&arch=openai"
```
