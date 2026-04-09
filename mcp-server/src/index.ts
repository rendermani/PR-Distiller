import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { setupHandlers } from "./handlers.js";

const server = new McpServer({
    name: "Agile-Rule-Extractor",
    version: "1.0.0"
});

// Setup tools and resources
setupHandlers(server);

async function main() {
    const transport = new StdioServerTransport();
    await server.connect(transport);
    console.error("PR-Analysis Agile-Rule-Extractor MCP Server running on stdio");
}

main().catch(console.error);
