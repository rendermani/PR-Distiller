import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

class LightRAGManager:
    """
    CodeRAG DB Receiver. 
    Implements the 'No-Training' solution by mapping extracted PR rejection vectors
    so they can be contextually injected into the student models during routing.
    """
    def __init__(self, db_path: str = "./data/chroma_db"):
        self.db_path = db_path
        os.makedirs(self.db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.db_path, settings=Settings(allow_reset=True))
        
        # Use a lightweight fast embedding model to vectorize the anti-pattern constraints
        self.embed_fn = embedding_functions.DefaultEmbeddingFunction()
        
        # We store anti-patterns explicitly
        self.collection = self.client.get_or_create_collection(
            name="enterprise_rejections",
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def store_rule(self, rule_json: dict):
        """
        Stores the high-quality rule extracted by the 70B Teacher / Gemini 3.1 Pro.
        """
        rule_id = rule_json.get("rule_id")
        content = rule_json.get("content", {})
        description = content.get("description", "")
        enforcement = content.get("enforcement_prompt", "")
        
        # The searchable vector string combines condition and what was rejected
        document_text = f"Rule: {content.get('title')} - Context: {description}. Enforce: {enforcement}"
        
        self.collection.add(
            documents=[document_text],
            metadatas=[{"rule_id": rule_id, "status": rule_json.get("metadata", {}).get("status", "active")}],
            ids=[rule_id]
        )
        print(f"[*] Stored Rule Vector [{rule_id}] into CodeRAG Receiver.")

    def get_contextual_rules(self, code_diff: str, top_k: int = 3):
        """
        Retrieves the exact past rejected rules that semantically match the current code diff constraint.
        This provides the explicit context for the Student model to execute perfectly.
        """
        results = self.collection.query(
            query_texts=[code_diff],
            n_results=top_k
        )
        
        if not results["documents"] or not results["documents"][0]:
            return []
            
        return results["documents"][0]
        
    def generate_rag_system_prompt(self, base_prompt: str, code_diff: str) -> str:
        """Helper to inject the contextual vectors directly into the student LLM instructions."""
        matched_vectors = self.get_contextual_rules(code_diff)
        if not matched_vectors:
            return base_prompt
            
        rag_context = "\\n\\n[DATABASE RAG CONTEXT - PREVIOUS ENTERPRISE REJECTIONS]:\\n"
        for i, vector in enumerate(matched_vectors, 1):
            rag_context += f"{i}. {vector}\\n"
            
        return base_prompt + rag_context
