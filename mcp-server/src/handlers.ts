import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

export function setupHandlers(server: McpServer) {
    server.tool(
        "get_contextual_rules",
        "Retrieves extracted architectural and coding rules for a given file path based on historical PR comments.",
        {
            code_path: z.string().describe("The file path currently being edited (e.g. src/auth/login.tsx)"),
            semantic_query: z.string().optional().describe("A semantic description of what you are trying to do")
        },
        async (args) => {
            const { code_path } = args;
            const mockRule = {
                rule_id: "uuid-1234",
                title: "Authentication Hook Rule",
                enforcement_prompt: "Always use the custom useAuth hook instead of calling the Context directly.",
                examples: {
                    bad: "const { user } = useContext(AuthContext);",
                    good: "const user = useAuth();"
                }
            };

            return {
                content: [{
                    type: "text",
                    text: `Found applicable rules for ${code_path}:\\n\\n` + JSON.stringify(mockRule, null, 2)
                }]
            };
        }
    );

    server.tool(
        "report_rule_feedback",
        "Flags a rule as problematic or outdated so it can be reviewed in the Management UI.",
        {
            rule_id: z.string().describe("UUID of the rule being reported"),
            reason: z.string().describe("Explain why the rule is incorrect or needs review")
        },
        async (args) => {
            const { rule_id, reason } = args;
            return {
                content: [{
                    type: "text",
                    text: `Rule ${rule_id} has been flagged for review. Reason: ${reason}`
                }]
            };
        }
    );
}
