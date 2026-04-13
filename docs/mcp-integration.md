# MCP Integration

## What is MCP?

The [Model Context Protocol](https://modelcontextprotocol.io/) is an open standard that allows AI coding assistants (Cursor, Claude Code, GitHub Copilot, and others) to query external tools during a conversation. PR-Analysis exposes an MCP server that surfaces coding rules extracted from your team's PR history directly into the IDE as the assistant is generating code.

## Available Tools

### `query_architectural_constraints`

Queries the ChromaDB vector store for rules that are semantically similar to the code currently being written.

**Inputs:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `code_diff` | string | Yes | The active code snippet or git diff block |
| `top_k` | number | No (default: 3) | Number of rules to retrieve |
| `file_path` | string | No | Path of the file being edited (e.g. `src/auth/login.py`). When provided, only rules whose path patterns match are returned |

**Output:** A formatted text block listing matched constraints, each tagged with its extraction frequency (how many PRs reinforced the rule).

### `report_rule_feedback`

Reports whether the developer applied or dismissed a rule that was presented by `query_architectural_constraints`. The backend uses this signal to auto-promote high-quality rules and surface them more frequently.

**Inputs:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `rule_id` | string | Yes | The `rule_id` from the matched rule's metadata |
| `action` | string | Yes | `"applied"` or `"dismissed"` |

**Output:** Confirmation message with the recorded action.

## How It Works

1. The developer writes or pastes code in their IDE.
2. The IDE sends the code diff to the MCP server over stdio.
3. The MCP server forwards the diff to the backend API (`POST /api/mcp/query`).
4. The backend embeds the diff and queries ChromaDB for the closest matching rules.
5. Matched rules are returned to the MCP server, which formats them and returns them to the IDE.
6. The IDE presents the constraints to the developer as assistant context.
7. Optionally, the IDE calls `report_rule_feedback` to record whether the suggestion was followed.

## Setup with Cursor

Build the MCP server:

```bash
cd mcp-server && npm install && npm run build
```

Add to `.cursor/mcp.json` in your project root (create if it does not exist):

```json
{
  "mcpServers": {
    "pr-analysis": {
      "command": "node",
      "args": ["/absolute/path/to/PR-Analysis/mcp-server/dist/index.js"],
      "env": {
        "PR_DISTILLER_API_URL": "http://localhost:8923"
      }
    }
  }
}
```

Restart Cursor. The `query_architectural_constraints` and `report_rule_feedback` tools will appear in the Cursor agent tool list.

## Setup with Claude Code

```bash
claude mcp add pr-analysis \
  --command "node" \
  --args "/absolute/path/to/PR-Analysis/mcp-server/dist/index.js" \
  --env "PR_DISTILLER_API_URL=http://localhost:8923"
```

Or add manually to `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "pr-analysis": {
      "command": "node",
      "args": ["/absolute/path/to/PR-Analysis/mcp-server/dist/index.js"],
      "env": {
        "PR_DISTILLER_API_URL": "http://localhost:8923"
      }
    }
  }
}
```

## Setup with Docker

The MCP server image is included in the Compose project but requires the `mcp` profile to start (because MCP uses stdio, not a persistent daemon):

```bash
docker compose run --rm mcp-server
```

When running the MCP server in Docker, the backend URL must use the internal Docker network hostname:

```bash
docker compose run --rm \
  -e PR_DISTILLER_API_URL=http://backend:8923 \
  mcp-server
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PR_DISTILLER_API_URL` | `http://127.0.0.1:8923` | URL of the backend API the MCP server connects to |

## Verifying the Integration

After setup, ask your AI assistant:

> "What are the architectural constraints for this codebase?"

The assistant should call `query_architectural_constraints` with a sample diff and return any matching rules from the database. If no rules have been extracted yet, run a pipeline job first (see [getting-started.md](getting-started.md)).
