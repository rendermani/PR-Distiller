import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db.lightrag_manager import LightRAGManager
from pipeline.semantic_fuser import SemanticFuser

def run_deduplication_audit():
    print("====================================")
    print("🧠 PHASE 2 DEMO: SEMANTIC LLM FUSING")
    print("====================================\\n")
    
    db = LightRAGManager()
    fuser = SemanticFuser(db)
    
    print("[1] Simulating first extraction hitting the pipeline...")
    # First, a common constraint written objectively
    rule_1 = {
        "rule_id": "test-rule-a",
        "metadata": {"status": "active"},
        "content": {
            "title": "Avoid Python eval()",
            "description": "Using eval() executes arbitrary strings which opens the system to critical RCE vulnerabilities.",
            "enforcement_prompt": "Refactor all generic eval(...) calls to ast.literal_eval() to ensure safe code execution."
        }
    }
    fuser.process_and_fuse(rule_1)

    print("\\n[2] Simulating a massive batch pulling a highly redundant identical constraint...")
    # A seemingly distinct rule written by a different developer, hitting the same root cause conceptually
    rule_2 = {
        "rule_id": "test-rule-b",
        "metadata": {"status": "active"},
        "content": {
            "title": "Unsafe python parsing",
            "description": "Never parse dict configurations using eval because hackers can inject commands.",
            "enforcement_prompt": "Please change the configuration eval block to use ast.literal_eval."
        }
    }
    
    # Process it natively - the fuser should catch the ChromaDB vector match immediately
    fuser.process_and_fuse(rule_2)
    
    print("\\n[3] Simulating a THIRD identical extraction from 2 months later...")
    rule_3 = {
        "rule_id": "test-rule-c",
        "metadata": {"status": "active"},
        "content": {
            "title": "Do not evaluate raw execution chains",
            "description": "I noticed we parse execution directly into eval. It creates high scale injection risks and fails enterprise standard audits.",
            "enforcement_prompt": "Always process logic configurations safely by converting evaluations directly to ast.literal_eval."
        }
    }
    
    final_rule = fuser.process_and_fuse(rule_3)
    
    print("\\n====================================")
    print("🏁 FINAL SYNTHESIZED ARCHITECTURAL RULE")
    print("====================================")
    print(f"Title: {final_rule['content'].get('title')}")
    print(f"Occurrences (Triage Density): {final_rule.get('metadata', {}).get('occurrence_count')}")
    print(f"Description: {final_rule['content'].get('description')}")
    print(f"Enforcement: {final_rule['content'].get('enforcement_prompt')}")

if __name__ == "__main__":
    run_deduplication_audit()
