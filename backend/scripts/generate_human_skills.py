import os
import sys
import json
import uuid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db.lightrag_manager import LightRAGManager
from pipeline.skill_synthesizer import SkillSynthesizer

def main():
    print("🚀 Initializing Human PR-to-Skill Pipeline...")
    
    # 1. Load the messy human comments
    data_path = os.path.join(os.path.dirname(__file__), '../data/human_rejected_dataset.json')
    with open(data_path, 'r') as f:
        datasets = json.load(f)

    db_manager = LightRAGManager()
    
    # We simulate the 70B extraction process since the DGX/Cloud API Keys are unavailable locally.
    # We parse the actual fetched human comments to highlight the structural extraction.
    for repo_data in datasets:
        repo_name = repo_data["repo"]
        comments = repo_data.get("human_comments", [])
        
        print(f"\\n[*] Extracting rules for {repo_name} ({len(comments)} human comments)...")
        for comment in comments:
            raw_text = comment.get("reviewer_comment", "")
            
            # Simulated 70B Extractor Logic (Bypassing API Key requirement)
            # The model translates the raw developer frustration into a strict JSON payload.
            extracted_rule = {
                "rule_id": str(uuid.uuid4())[:8],
                "metadata": {"status": "active"},
                "content": {
                    "title": f"Strict Constraint for {repo_name}",
                    "description": "Aggregated from core maintainer feedback to prevent recurring architecture violations.",
                    "enforcement_prompt": raw_text.strip().replace('\\n', ' ')
                }
            }
            
            # Immediately sink the rule into the persistent CodeRAG vector database
            db_manager.store_rule(extracted_rule)
            
    print("\\n==================================")
    print("🔥 TRIGGERING SKILL SYNTHESIZER 🔥")
    print("==================================")
    
    # 2. Engage the Synthesis Engine
    # Pulls all vectors from ChromaDB, deduplicates them, and burns them to Markdown Skill Prompts
    synthesizer = SkillSynthesizer(db_manager)
    generated_files = synthesizer.generate_combined_skills()
    
    print("\\n✅ Final PR-to-Skill Pipeline Complete!")
    if generated_files:
        for f in generated_files:
            with open(f, 'r') as file:
                print(f"\\n--- {os.path.basename(f)} ---\\n{file.read()}\\n")

if __name__ == "__main__":
    main()
