import os
import json
import uuid
from litellm import completion
from db.lightrag_manager import LightRAGManager

class SemanticFuser:
    """
    Phase 2 Big Data Deduplicator.
    Handles mathematical collisions in ChromaDB by feeding both conflicting vectors 
    back into the local Small LLM for an intelligent merge.
    """
    def __init__(self, db_manager: LightRAGManager):
        self.db = db_manager
        # We actively target the 7B node for fast text-merging tasks
        self.model = os.environ.get("EXTRACTOR_MODEL", "openai/Qwen/Qwen2.5-Coder-7B-Instruct")
        self.api_base = os.environ.get("EXTRACTOR_API_BASE", "http://gx10:8080/v1")
        self.api_key = os.environ.get("EXTRACTOR_API_KEY", "dgx-dummy-key")

    def process_and_fuse(self, new_rule_json: dict):
        """
        Takes a new LLM extraction and explicitly checks for vector-overlaps.
        If a collision occurs, the LLM fuses them and deletes the obsolete vector.
        """
        content = new_rule_json.get("content", {})
        document_text = f"Rule: {content.get('title')} - Context: {content.get('description')}. Enforce: {content.get('enforcement_prompt')}"
        
        # 1. Execute purely mathematical Vector Distance checking against the database
        matched_id, matched_doc, matched_metadatas = self.db.find_similar_rule(document_text, distance_threshold=0.45)

        if not matched_id:
            # Clean insertion if the vector is fundamentally distinct
            self.db.store_rule(new_rule_json)
            return new_rule_json

        print("\\n[Semantic Fuser] ⚠️ Detected Duplicate Vector Collisions! Initiating LLM Merging Phase...")

        # 2. Instruct the DGX Server to combine the two semantic strings intelligently
        system_prompt = (
            "You are a Senior AI Architect managing a technical knowledge base. "
            "You will be given two highly overlapping developer constraints. "
            "Merge them into a single, unified constraint without losing any unique technical nuances from either. "
            "Output ONLY valid JSON following this schema: "
            "{'rule_id': 'uuid', 'metadata': {'status': 'active'}, 'content': {'title': '...', 'description': '...', 'enforcement_prompt': '...'}}"
        )

        user_prompt = f'''
        <EXISTING_RULE>
        {matched_doc}
        </EXISTING_RULE>
        
        <NEW_REDUNDANT_RULE>
        {document_text}
        </NEW_REDUNDANT_RULE>
        '''

        try:
            response = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                api_base=self.api_base,
                api_key=self.api_key,
                temperature=0.1
            )
            raw_content = response.choices[0].message.content.strip()
            
            if raw_content.startswith("```json"):
                raw_content = raw_content.replace("```json", "").replace("```", "").strip()
                
            fused_rule = json.loads(raw_content)
            
            # Auto-assign a freshly merged unique ID
            fused_rule["rule_id"] = str(uuid.uuid4())[:8]
            
            # Mathematical Fusing: Combine total volumes observed of this architectural failure
            base_occurrences = matched_metadatas.get("occurrence_count", 1) if matched_metadatas else 1
            new_occurrences = new_rule_json.get("metadata", {}).get("occurrence_count", 1)
            
            if "metadata" not in fused_rule:
                fused_rule["metadata"] = {}
                
            fused_rule["metadata"]["occurrence_count"] = base_occurrences + new_occurrences
            
            # 3. Destructively purge the obsolete redundant string from vector space, then write the synthesized truth
            self.db.delete_rule(matched_id)
            self.db.store_rule(fused_rule)
            
            print(f"✅ Semantic Merge Complete! Replaced two messy concepts into single unified rule: {fused_rule['content']['title']}")
            return fused_rule
            
        except Exception as e:
            print(f"[Semantic Fuser Error]: Merging failed securely, retreating. Details: {e}")
            # Fallback to just storing it normally if the merging fails
            self.db.store_rule(new_rule_json)
            return new_rule_json
