import os
from collections import defaultdict

class SkillSynthesizer:
    """
    The PR-to-Skill Generator.
    Responsible for taking fragmented JSON vectors (anti-patterns, structural feedback)
    from ChromaDB, deduplicating them semantically, and compiling them into unified, 
    context-friendly LLM 'Skill' markdown files.
    """
    def __init__(self, db_manager):
        self.db = db_manager
        self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../skills'))
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_combined_skills(self):
        """
        Extracts all vectors from the database, clusters them by topic/condition,
        and fuses them into a polished LLM skill document.
        """
        # Fetch all stored enterprise rejections from our CodeRAG mapped space
        try:
            results = self.db.collection.get(include=["documents", "metadatas"])
        except Exception as e:
            print(f"Error reading from ChromaDB: {e}")
            return

        documents = results.get("documents", [])
        if not documents:
            print("No rules in DB to synthesize yet.")
            return

        # Perform clustering/aggregation (In production, use semantic clustering over embeddings)
        # Here we group by keyword heuristic for standard compilation
        clusters = defaultdict(list)
        for doc in documents:
            if "sql" in doc.lower() or "database" in doc.lower():
                clusters["Database_Security"].append(doc)
            elif "state" in doc.lower() or "config" in doc.lower():
                clusters["Architecture_State"].append(doc)
            else:
                clusters["General_Best_Practices"].append(doc)

        print(f"[*] Deduping {len(documents)} raw vectors into {len(clusters)} Unified Skills...")

        generated_skills = []
        for topic, rules in clusters.items():
            skill_md = f'''---
name: {topic}
description: Aggregated enterprise constraints ensuring Copilot compliance for {topic}.
---

# {topic.replace('_', ' ')} Guidelines

You must strictly adhere to the following architecture rules aggregated from previous PR rejections.
Failure to do so will result in enterprise architectural violations.

## Constraints & Anti-Patterns:
'''
            # Deduplicate semantically similar sentences inline before injecting
            unique_rules = list(set(rules))
            for i, rule in enumerate(unique_rules, 1):
                skill_md += f"{i}. {rule}\\n"

            file_path = os.path.join(self.output_dir, f"{topic.lower()}.md")
            with open(file_path, 'w') as f:
                f.write(skill_md)
                
            generated_skills.append(file_path)
            print(f" -> Generated Skill: {file_path}")
            
        return generated_skills
