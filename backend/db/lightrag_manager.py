import fnmatch
import hashlib
import os
import re
from typing import Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

import settings as app_settings

# Sourced from settings so containerized and native runs share one store; a
# __file__-derived path made the location depend on where the code sits.
VECTOR_DB_PATH = app_settings.VECTOR_DB_DIR

_SLUG_CHARS = re.compile(r"[^a-z0-9_-]+")


def _build_unique_rule_id(slug: str, repo: str, document_text: str) -> str:
    """Construct a ChromaDB-unique id from an LLM slug, the repo, and content hash.

    The LLM is now prompted to emit a kebab-case slug (e.g.
    "use-parameterized-sql-queries"). Two reviews on the same topic legitimately
    produce the same slug; ChromaDB ids must be unique. We namespace by repo and
    append a content hash so:
    - same slug + same content + same repo  -> same id (idempotent)
    - same slug + different content         -> different id (no collision)
    - same slug + different repo            -> different id (no cross-repo clash)
    """
    safe_slug = _SLUG_CHARS.sub("-", (slug or "rule").strip().lower()).strip("-") or "rule"
    safe_repo = _SLUG_CHARS.sub("-", (repo or "global").lower()).strip("-") or "global"
    content_hash = hashlib.sha256(document_text.encode("utf-8")).hexdigest()[:6]
    return f"{safe_repo}__{safe_slug}__{content_hash}"

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


_EMBEDDING_MODEL_ENV_VAR = "EMBEDDING_MODEL"


def _resolve_embedding_function():
    """Return the configured ChromaDB embedding function.

    When the EMBEDDING_MODEL environment variable is set, uses
    SentenceTransformerEmbeddingFunction with that model name.
    When absent, falls back to DefaultEmbeddingFunction (all-MiniLM-L6-v2)
    for backward compatibility with existing vector stores.

    Backward-compat fallback explicitly approved per issue #8 spec.
    """
    model_name = os.environ.get(_EMBEDDING_MODEL_ENV_VAR)
    if model_name:
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    return embedding_functions.DefaultEmbeddingFunction()


class LightRAGManager:
    """
    CodeRAG DB Receiver.
    Implements the 'No-Training' solution by mapping extracted PR rejection vectors
    so they can be contextually injected into the student models during routing.
    """
    def __init__(self):
        self.embed_fn = _resolve_embedding_function()
        self.client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
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
            include=['documents', 'distances', 'metadatas']
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

    def store_rule(self, rule_json: dict) -> str:
        """Store the rule and return the ChromaDB id used.

        The returned id is the canonical reference (used by callers like
        SemanticFuser to populate `merged_into` lineage). Does not mutate
        `rule_json["rule_id"]` so calling this twice with the same input is
        idempotent (same id both times).
        """
        slug = rule_json.get("rule_id") or os.urandom(8).hex()
        content = rule_json.get("content", {})
        description = content.get("description", "")
        enforcement = content.get("enforcement_prompt", "")
        metadata = rule_json.get("metadata", {})

        # Pull occurrence_count, defaulting to 1 explicitly
        occurrences = metadata.get("occurrence_count", 1)
        repo = metadata.get("repo", "global")

        # The searchable vector string combines condition and what was rejected
        document_text = f"Rule: {content.get('title')} - Context: {description}. Enforce: {enforcement}"

        # Build a ChromaDB-unique id; same slug across rules legitimately
        # produces the same kebab-case label, so we namespace by repo and
        # append a content hash to disambiguate.
        rule_id = _build_unique_rule_id(slug, repo, document_text)

        # Flatten path_patterns list to a comma-separated string.
        # ChromaDB metadata values must be str/int/float/bool — not lists.
        raw_patterns: list = rule_json.get("scoping", {}).get("path_patterns", [])
        path_patterns_str: str = ",".join(raw_patterns)

        chroma_metadata = {
            "rule_id": rule_id,
            "title_slug": slug,
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

        # Provenance and versioning fields (Task 4 multi-model pipeline).
        # merged_with_models and merged_from are comma-joined strings to satisfy
        # ChromaDB's str/int/float/bool metadata constraint.
        if "extracted_by_model" in metadata:
            chroma_metadata["extracted_by_model"] = metadata["extracted_by_model"]
        if "extracted_by_label" in metadata:
            chroma_metadata["extracted_by_label"] = metadata["extracted_by_label"]
        if "merged_with_models" in metadata:
            val = metadata["merged_with_models"]
            # Normalise: the fuser stores a comma-joined string; the extractor
            # seeds it as an empty list via setdefault — flatten to string here.
            if isinstance(val, list):
                val = ",".join(str(v) for v in val)
            chroma_metadata["merged_with_models"] = val
        if "merge_count" in metadata:
            chroma_metadata["merge_count"] = metadata["merge_count"]
        if "merged_from" in metadata:
            chroma_metadata["merged_from"] = metadata["merged_from"]

        self.collection.add(
            documents=[document_text],
            metadatas=[chroma_metadata],
            ids=[rule_id]
        )
        print(f"[*] Stored Rule Vector [{rule_id}] into CodeRAG Receiver.")
        return rule_id

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

        archive_id = os.urandom(8).hex()
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

    def delete_repo_rules(self, repo: str) -> int:
        """Delete every rule for *repo*. Returns the number of rules removed.

        Used by dev-mode replay: when re-extracting from the crawl cache with
        a new model or prompt, the repo's existing rules must be cleared so
        fresh extractions stand on their own instead of being deduplicated
        into stale rules via semantic fusion.
        """
        existing = self.collection.get(where={"repo": repo})
        ids = existing.get("ids") or []
        if not ids:
            return 0
        self.collection.delete(ids=ids)
        return len(ids)

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

    def record_feedback(self, rule_id: str, action: str) -> None:
        """
        Records that a rule was either applied or dismissed by the IDE.

        Increments times_served plus the counter for the given action.
        When a 'needs_review' rule reaches >=5 serves with >70% acceptance rate,
        it is auto-promoted to 'active'.

        Args:
            rule_id: The ID of the rule that received feedback.
            action: Either 'applied' or 'dismissed'.

        Raises:
            ValueError: If rule_id is not found or action is not a valid value.
        """
        if action not in ("applied", "dismissed"):
            raise ValueError(
                f"Invalid feedback action '{action}'. Must be 'applied' or 'dismissed'."
            )

        results = self.collection.get(ids=[rule_id], include=["documents", "metadatas"])
        if not results.get("ids"):
            raise ValueError(f"Rule '{rule_id}' not found.")

        metadata = dict(results["metadatas"][0]) if results.get("metadatas") else {}

        times_served = metadata.get("times_served", 0) + 1
        times_applied = metadata.get("times_applied", 0) + (1 if action == "applied" else 0)
        times_dismissed = metadata.get("times_dismissed", 0) + (1 if action == "dismissed" else 0)

        metadata["times_served"] = times_served
        metadata["times_applied"] = times_applied
        metadata["times_dismissed"] = times_dismissed

        if metadata.get("status") == "needs_review":
            metadata["status"] = self._evaluate_promotion(times_served, times_applied)

        self.collection.update(ids=[rule_id], metadatas=[metadata])

    @staticmethod
    def _evaluate_promotion(times_served: int, times_applied: int) -> str:
        """
        Returns 'active' when auto-promotion criteria are met, else 'needs_review'.

        Criteria: served >= 5 times AND acceptance rate strictly > 70%.
        """
        _MIN_SERVES = 5
        _ACCEPTANCE_THRESHOLD = 0.70

        if times_served < _MIN_SERVES:
            return "needs_review"
        acceptance_rate = times_applied / times_served
        return "active" if acceptance_rate > _ACCEPTANCE_THRESHOLD else "needs_review"

    def get_rule_effectiveness(self, rule_id: str) -> dict:
        """
        Returns effectiveness stats for a single rule.

        Returns a dict with: rule_id, times_served, times_applied, times_dismissed,
        acceptance_rate.

        Raises:
            ValueError: If the rule is not found.
        """
        results = self.collection.get(ids=[rule_id], include=["documents", "metadatas"])
        if not results.get("ids"):
            raise ValueError(f"Rule '{rule_id}' not found.")

        metadata = results["metadatas"][0] if results.get("metadatas") else {}
        times_served = metadata.get("times_served", 0)
        times_applied = metadata.get("times_applied", 0)
        times_dismissed = metadata.get("times_dismissed", 0)
        acceptance_rate = times_applied / times_served if times_served > 0 else 0.0

        return {
            "rule_id": rule_id,
            "times_served": times_served,
            "times_applied": times_applied,
            "times_dismissed": times_dismissed,
            "acceptance_rate": acceptance_rate,
        }

    def get_all_effectiveness_stats(self) -> dict:
        """
        Returns aggregate effectiveness stats across every rule in the collection.

        Structure:
          {
            "rules": [ { rule_id, times_served, times_applied, times_dismissed,
                         acceptance_rate }, ... ],
            "totals": { total_served, total_applied, total_dismissed,
                        overall_acceptance_rate }
          }
        """
        results = self.collection.get(include=["documents", "metadatas"])

        ids = results.get("ids") or []
        metadatas = results.get("metadatas") or []

        rule_stats = []
        total_served = 0
        total_applied = 0
        total_dismissed = 0

        for rule_id, meta in zip(ids, metadatas):
            meta = meta or {}
            served = meta.get("times_served", 0)
            applied = meta.get("times_applied", 0)
            dismissed = meta.get("times_dismissed", 0)
            rate = applied / served if served > 0 else 0.0

            rule_stats.append({
                "rule_id": rule_id,
                "times_served": served,
                "times_applied": applied,
                "times_dismissed": dismissed,
                "acceptance_rate": rate,
            })
            total_served += served
            total_applied += applied
            total_dismissed += dismissed

        overall_rate = total_applied / total_served if total_served > 0 else 0.0

        return {
            "rules": rule_stats,
            "totals": {
                "total_served": total_served,
                "total_applied": total_applied,
                "total_dismissed": total_dismissed,
                "overall_acceptance_rate": overall_rate,
            },
        }

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

    def reindex_all_rules(self) -> int:
        """Re-embed all rules in enterprise_rejections using the current embed_fn.

        Deletes all existing vectors and re-adds them so the new embedding model
        is applied uniformly.  Returns the number of rules re-indexed.

        Must be called after changing EMBEDDING_MODEL to prevent dimension mismatches
        between old and new vectors in the same collection.
        """
        snapshot = self.collection.get(include=["documents", "metadatas"])

        rule_ids: list[str] = snapshot.get("ids", [])
        if not rule_ids:
            return 0

        documents: list[str] = snapshot["documents"]
        metadatas: list[dict] = snapshot["metadatas"]

        self.collection.delete(ids=rule_ids)
        self.collection.add(
            ids=rule_ids,
            documents=documents,
            metadatas=metadatas,
        )
        return len(rule_ids)
        
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
