import fnmatch
import os
from typing import Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

def _rule_matches_file_path(path_patterns_str: str, file_path: str) -> bool:
    """Return True if file_path matches any of the comma-separated glob patterns.

    An empty patterns string means the rule is unscoped and always matches.
    """
    if not path_patterns_str:
        return True
    patterns = path_patterns_str.split(",")
    return any(fnmatch.fnmatch(file_path, pattern) for pattern in patterns)


def _apply_path_scoping(results: dict, file_path: str) -> dict:
    """Post-filter a ChromaDB query result to rules that match file_path.

    Preserves the nested-list structure ChromaDB returns:
      results["documents"] == [[doc1, doc2, ...]]
      results["ids"]       == [[id1, id2, ...]]
      results["metadatas"] == [[meta1, meta2, ...]]

    Returns a results dict with the same shape but only matching entries.
    Rules whose metadata lacks 'path_patterns' are treated as unscoped.
    """
    docs = results["documents"][0]
    ids = results["ids"][0]
    metadatas = results.get("metadatas", [[]])[0]

    filtered_docs, filtered_ids, filtered_metas = [], [], []
    for doc, rule_id, meta in zip(docs, ids, metadatas):
        patterns_str = meta.get("path_patterns", "") if meta else ""
        if _rule_matches_file_path(patterns_str, file_path):
            filtered_docs.append(doc)
            filtered_ids.append(rule_id)
            filtered_metas.append(meta)

    return {
        **results,
        "documents": [filtered_docs],
        "ids": [filtered_ids],
        "metadatas": [filtered_metas],
    }


class LightRAGManager:
    """
    CodeRAG DB Receiver. 
    Implements the 'No-Training' solution by mapping extracted PR rejection vectors
    so they can be contextually injected into the student models during routing.
    """
    def __init__(self):
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(base_dir, "data", "code_rag_vectors")
        
        # We target the robust persistent Client specifically. 
        self.embed_fn = embedding_functions.DefaultEmbeddingFunction()
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="enterprise_rejections",
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"}
        )
        self.history_collection = self.client.get_or_create_collection(
            name="rule_history",
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def find_similar_rule(self, document_text: str, repo: str, distance_threshold: float = 0.20):
        """
        Determines mathematically if a new extraction is redundant.
        Distance threshold ~0.20 strongly implies identical architectural constraints.
        Returns the (rule_id, previous_text) if found, else None.
        """
        results = self.collection.query(
            query_texts=[document_text],
            n_results=1,
            where={"repo": repo},
            include=['documents', 'distances']
        )
        
        # Guard against empty results natively 
        if not results.get("distances") or not len(results["distances"][0]):
            return None, None, None
            
        closest_distance = results["distances"][0][0]
        if closest_distance < distance_threshold:
            print(f"[CodeRAG Filter] Caught similar vector locally! Distance: {closest_distance:.3f}")
            matched_id = results["ids"][0][0]
            matched_doc = results["documents"][0][0]
            matched_metadatas = results["metadatas"][0][0] if results.get("metadatas") else {}
            return matched_id, matched_doc, matched_metadatas
            
        return None, None, None

    def store_rule(self, rule_json: dict):
        """
        Stores the high-quality rule extracted by the 70B Teacher / Gemini 3.1 Pro.
        """
        rule_id = rule_json.get("rule_id", os.urandom(4).hex())
        content = rule_json.get("content", {})
        description = content.get("description", "")
        enforcement = content.get("enforcement_prompt", "")
        metadata = rule_json.get("metadata", {})

        # Pull occurrence_count, defaulting to 1 explicitly
        occurrences = metadata.get("occurrence_count", 1)
        repo = metadata.get("repo", "global")

        # The searchable vector string combines condition and what was rejected
        document_text = f"Rule: {content.get('title')} - Context: {description}. Enforce: {enforcement}"

        # Flatten path_patterns list to a comma-separated string.
        # ChromaDB metadata values must be str/int/float/bool — not lists.
        raw_patterns: list = rule_json.get("scoping", {}).get("path_patterns", [])
        path_patterns_str: str = ",".join(raw_patterns)

        chroma_metadata = {
            "rule_id": rule_id,
            "repo": repo,
            "status": metadata.get("status", "active"),
            "occurrence_count": occurrences,
            "path_patterns": path_patterns_str,
        }

        # Propagate optional fields from issues #6 and #7 when present
        if "confidence" in metadata:
            chroma_metadata["confidence"] = metadata["confidence"]
        if "category" in metadata:
            chroma_metadata["category"] = metadata["category"]

        self.collection.add(
            documents=[document_text],
            metadatas=[chroma_metadata],
            ids=[rule_id]
        )
        print(f"[*] Stored Rule Vector [{rule_id}] into CodeRAG Receiver.")

    def delete_rule(self, rule_id: str):
        """Removes an obsolete rule aggressively after Semantic LLM fusing merges it safely."""
        self.collection.delete(ids=[rule_id])
        print(f"[*] Purged overlapping Vector [{rule_id}] post-synthesis.")

    def archive_rule(self, rule_id: str, *, merged_into_id: str) -> None:
        """
        Copies a rule from enterprise_rejections into rule_history before deletion.

        The archived copy records:
        - original_rule_id: the ID of the rule being superseded
        - merged_into: the ID of the new fused rule that replaces it

        Raises ValueError if the rule does not exist in the main collection.
        """
        results = self.collection.get(
            ids=[rule_id],
            include=["documents", "metadatas"],
        )
        if not results.get("ids"):
            raise ValueError(f"Rule '{rule_id}' not found in enterprise_rejections")

        document = results["documents"][0]
        original_metadata = results["metadatas"][0] if results.get("metadatas") else {}

        archive_metadata = {
            **original_metadata,
            "original_rule_id": rule_id,
            "merged_into": merged_into_id,
        }

        import os as _os
        archive_id = _os.urandom(8).hex()
        self.history_collection.add(
            ids=[archive_id],
            documents=[document],
            metadatas=[archive_metadata],
        )
        print(f"[*] Archived rule [{rule_id}] -> history (merged_into={merged_into_id})")

    def get_rule_history(self, rule_id: str) -> list[dict]:
        """
        Returns all history entries for rules that were merged into the given rule_id.

        Each entry is a dict with at minimum: original_rule_id, merged_into, document.
        Returns an empty list when no history exists.
        """
        results = self.history_collection.get(
            where={"merged_into": rule_id},
            include=["documents", "metadatas"],
        )
        if not results.get("ids"):
            return []

        entries = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            entries.append({**meta, "document": doc})
        return entries

    def get_contextual_rules(
        self,
        code_diff: str,
        top_k: int = 3,
        categories: list[str] | None = None,
        file_path: Optional[str] = None,
    ):
        """
        Retrieves past rejected rules that semantically match the current code diff.

        When file_path is provided, applies post-filter scoping:
        - Rules with no path_patterns always match (unscoped rules).
        - Rules with path_patterns only match when any pattern matches file_path
          via fnmatch glob semantics.
        When file_path is None, all rules are returned (backward-compatible).

        Args:
            code_diff: The code diff to match against stored rules.
            top_k: Maximum number of rules to return.
            categories: Optional list of category values to restrict results to
                        (issue #7). When None, all categories are included.
            file_path: Optional path of the file being edited (issue #4).
        """
        if categories:
            where_clause = {"$and": [{"status": "active"}, {"category": {"$in": categories}}]}
        else:
            where_clause = {"status": "active"}

        results = self.collection.query(
            query_texts=[code_diff],
            n_results=top_k,
            where=where_clause
        )

        if not results["documents"] or not results["documents"][0]:
            return []

        if file_path is None:
            return results

        return _apply_path_scoping(results, file_path)

    def approve_rule(self, rule_id: str):
        """
        Allows Human-in-The-Loop reviewers to dynamically flip rule states to active permanently in DB.
        """
        results = self.collection.get(ids=[rule_id])
        if not results.get("ids"): raise ValueError("Rule not found")

        metadatas = results["metadatas"][0] if results.get("metadatas") else {}
        metadatas["status"] = "active"

        self.collection.update(
            ids=[rule_id],
            metadatas=[metadatas]
        )

    def block_rule(self, rule_id: str):
        """Blocks a rule permanently — future extractions matching this pattern will be skipped."""
        results = self.collection.get(ids=[rule_id])
        if not results.get("ids"): raise ValueError("Rule not found")

        metadatas = results["metadatas"][0] if results.get("metadatas") else {}
        metadatas["status"] = "blocked"

        self.collection.update(
            ids=[rule_id],
            metadatas=[metadatas]
        )

    def is_blocked(self, document_text: str, repo: str, distance_threshold: float = 0.20) -> bool:
        """Checks if a new extraction is semantically similar to any blocked rule."""
        blocked = self.collection.get(where={"$and": [{"repo": repo}, {"status": "blocked"}]})
        if not blocked.get("ids"):
            return False

        results = self.collection.query(
            query_texts=[document_text],
            n_results=1,
            where={"$and": [{"repo": repo}, {"status": "blocked"}]},
            include=['distances']
        )

        if not results.get("distances") or not results["distances"][0]:
            return False

        return results["distances"][0][0] < distance_threshold

    def add_rule(self, repo: str, title: str, description: str, enforcement: str):
        """Manually adds a rule from the UI."""
        rule_id = os.urandom(4).hex()
        document_text = f"Rule: {title} - Context: {description}. Enforce: {enforcement}"

        self.collection.add(
            documents=[document_text],
            metadatas=[{"rule_id": rule_id, "repo": repo, "status": "needs_review", "occurrence_count": 1}],
            ids=[rule_id]
        )
        return rule_id
        
    def generate_rag_system_prompt(self, base_prompt: str, code_diff: str) -> str:
        """Helper to inject the contextual vectors directly into the student LLM instructions."""
        results = self.get_contextual_rules(code_diff)
        if not results:
            return base_prompt

        docs = results["documents"][0] if results.get("documents") else []
        if not docs:
            return base_prompt

        rag_context = "\\n\\n[DATABASE RAG CONTEXT - PREVIOUS ENTERPRISE REJECTIONS]:\\n"
        for i, vector in enumerate(docs, 1):
            rag_context += f"{i}. {vector}\\n"
            
        return base_prompt + rag_context
