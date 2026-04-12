import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({
    name: "PR-Distiller-Engine",
    version: "1.0.0"
});

// The Python backend URL (Can be passed via env if testing on random ports)
const API_URL = process.env.PR_DISTILLER_API_URL || "http://127.0.0.1:8923";

/**
 * Core MCP Tool: Dynamically query the historical PR anti-patterns based on a code snippet.
 */
server.tool(
    "query_architectural_constraints",
    "Queries the historical PR database for contextually similar anti-patterns mapping to the current code diff",
    {
        code_diff: z.string().describe("The active raw code snippet, or git diff block being edited."),
        top_k: z.number().optional().default(3).describe("Number of relevant historical constraints to retrieve."),
        file_path: z.string().optional().describe("Path of the file being edited (e.g. src/auth/login.py). When provided, only rules scoped to matching path patterns are returned.")
    },
    async ({ code_diff, top_k, file_path }) => {
        try {
            console.error(`[MCP] Querying Knowledge Matrix for snippet...`);

            const response = await fetch(`${API_URL}/api/mcp/query`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ code_diff, top_k, file_path: file_path ?? null })
            });

            if (!response.ok) {
                throw new Error(`Python API returned ${response.status}`);
            }

            const data = await response.json();
            const rules = data.matched_rules || [];

            if (rules.length === 0) {
                return {
                    content: [{ type: "text", text: "No relevant architectural constraints found for this codebase pattern." }]
                };
            }

            // Format the rules cleanly for Cursor / Copilot to ingest effortlessly
            let formattedContext = "CRITICAL ARCHITECTURAL CONSTRAINTS EXTRACTED FROM HISTORICAL CODE REVIEWS:\\n\\n";
            rules.forEach((rule: any, idx: number) => {
                const doc = rule.document || "";
                const triageDensity = rule.metadata?.occurrence_count || 1;
                formattedContext += `--- Constraint #${idx + 1} [Severity Density: ${triageDensity}x] ---\\n`;
                formattedContext += `${doc}\\n\\n`;
            });

            return {
                content: [{ type: "text", text: formattedContext }]
            };
        } catch (error: any) {
            console.error(`[MCP] Backend connection failed:`, error.message);
            return {
                content: [{ type: "text", text: `Error retrieving context: ${error.message}` }],
                isError: true
            };
        }
    }
);

/**
 * MCP Tool: Report whether a previously served rule was applied or dismissed.
 *
 * Call this after showing a constraint to the user so the backend can track
 * accept/dismiss ratios and auto-promote high-quality rules.
 */
server.tool(
    "report_rule_feedback",
    "Reports whether a rule suggestion was applied or dismissed by the developer. Call this after presenting a constraint from query_architectural_constraints.",
    {
        rule_id: z.string().describe("The rule_id from the matched rule's metadata."),
        action: z.enum(["applied", "dismissed"]).describe("'applied' if the developer followed the rule, 'dismissed' if they overrode or ignored it.")
    },
    async ({ rule_id, action }) => {
        try {
            console.error(`[MCP] Recording feedback: rule=${rule_id} action=${action}`);

            const response = await fetch(`${API_URL}/api/rules/${encodeURIComponent(rule_id)}/feedback`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ action })
            });

            if (!response.ok) {
                const detail = await response.text();
                throw new Error(`Backend returned ${response.status}: ${detail}`);
            }

            const data = await response.json();
            return {
                content: [{
                    type: "text",
                    text: `Feedback recorded: rule '${data.rule_id}' marked as '${data.action}'.`
                }]
            };
        } catch (error: any) {
            console.error(`[MCP] Feedback recording failed:`, error.message);
            return {
                content: [{ type: "text", text: `Error recording feedback: ${error.message}` }],
                isError: true
            };
        }
    }
);

// Bind the server natively to Standard IO interfaces (cursor native protocol)
async function run() {
    console.error("[*] Starting PR-Distiller MCP Server over STDIO...");
    const transport = new StdioServerTransport();
    await server.connect(transport);
}

run().catch(err => {
    console.error("Fatal MCP crash:", err);
    process.exit(1);
});
