import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db.lightrag_manager import LightRAGManager

def restore_ghosted_data():
    print("[*] Reconnecting to persistent ChromaDB schema...")
    db = LightRAGManager()
    
    # 5 Hand-curated explicit architectural rules gathered previously
    rules = [
        {
            "rule_id": "c1a293xx",
            "metadata": {"status": "active", "occurrence_count": 14},
            "content": {
                "title": "Inefficient Client Initialization",
                "description": "The Next.js configuration initializes HostBackendClient repeatedly inside the request loop, severely depleting available connection limits.",
                "enforcement_prompt": "Refactor logic loops to reuse an existing client or initialize HostBackendClient globally outside of the execution block to prevent memory exhaustion."
            }
        },
        {
            "rule_id": "bd9281ww",
            "metadata": {"status": "active", "occurrence_count": 8},
            "content": {
                "title": "Random Thread ID UUIDs",
                "description": "The active block maps Thread ID states using identical UUIDs across async loops, violating testing independence limits.",
                "enforcement_prompt": "Ensure all async threads independently seed via uuid.uuid4() natively to avoid crossover state collision errors."
            }
        },
        {
            "rule_id": "ef2931xx",
            "metadata": {"status": "active", "occurrence_count": 5},
            "content": {
                "title": "Unsafe Python Parsing",
                "description": "Evaluating dynamic dict maps automatically routes standard string logic perfectly into RCE exploitation vulnerabilities natively.",
                "enforcement_prompt": "Execute parsing exclusively through ast.literal_eval natively bypassing arbitrary exec evaluations cleanly."
            }
        },
        {
            "rule_id": "df2920aa",
            "metadata": {"status": "active", "occurrence_count": 3},
            "content": {
                "title": "Use of None in User Field",
                "description": "Resolving undefined system mapping defaults to Python NoneType, corrupting frontend payload serialization expectations in TypeScript consumers.",
                "enforcement_prompt": "Always explicitly map untyped payloads dynamically into AnonymousUser entities."
            }
        },
        {
            "rule_id": "7f2a1b32",
            "metadata": {"status": "needs_review", "occurrence_count": 1},
            "content": {
                "title": "Redundant Class Definition",
                "description": "Executing nested class configurations like NestedHelpGroup creates deep architectural redundancy overriding default DeployGroup mapping parameters.",
                "enforcement_prompt": "Refactor inherited classes internally inside DeployGroup to remove legacy class nesting structures universally."
            }
        }
    ]
    
    for rule in rules:
        db.store_rule(rule)
        
    print(f"[*] Successfully restored {len(rules)} master vectors into the explicit local space.")
    
if __name__ == "__main__":
    restore_ghosted_data()
