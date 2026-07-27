import json
import uuid
from litellm import completion
from db.lightrag_manager import LightRAGManager


def _qwen_extra_body(model: str) -> dict:
    """Disable Qwen3 thinking for fast merge calls. Ollama's native /api/chat
    honours `think: false` to suppress the reasoning chain entirely, cutting
    latency from ~30s to ~3s per merge.
    """
    if "qwen" in (model or "").lower():
        return {"extra_body": {"think": False}}
    return {}


def _first_meaningful(*values):
    """Return the first value that is not None and not an empty string.

    Plain `a or b or c` would skip legitimate falsy values (0, 0.0, False).
    For category/confidence we want only None and "" to be considered missing.
    """
    for v in values:
        if v is None:
            continue
        if isinstance(v, str) and v == "":
            continue
        return v
    return None


class SemanticFuser:
    """
    Phase 2 Big Data Deduplicator.
    Handles mathematical collisions in ChromaDB by feeding both conflicting vectors
    back into the local Small LLM for an intelligent merge.
    """
    def __init__(
        self,
        db_manager: LightRAGManager,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
    ):
        self.db = db_manager
        # The extractor (and orchestrator behind it) are the single source of
        # truth for model config. If None is passed the LLM merge call will
        # fail loudly — that is intentional.
        self.model = model
        resolved_base = api_base
        # LiteLLM's native ollama/ provider must NOT receive a /v1 suffix.
        if self.model and self.model.startswith("ollama/") and resolved_base and resolved_base.endswith("/v1"):
            resolved_base = resolved_base[:-3]
        self.api_base = resolved_base
        self.api_key = api_key or "unused"

    def process_and_fuse(self, new_rule_json: dict):
        """
        Takes a new LLM extraction and explicitly checks for vector-overlaps.
        If a collision occurs, the LLM fuses them and deletes the obsolete vector.
        """
        content = new_rule_json.get("content", {})
        document_text = f"Rule: {content.get('title')} - Context: {content.get('description')}. Enforce: {content.get('enforcement_prompt')}"
        
        repo = new_rule_json.get("metadata", {}).get("repo", "global")

        # 0. Skip if this matches a blocked rule
        if self.db.is_blocked(document_text, repo):
            print(f"[Semantic Fuser] Skipped — matches a blocked rule in {repo}")
            return None

        # 1. Execute purely mathematical Vector Distance checking against the database explicitly isolated natively to the target Git Repository
        matched_id, matched_doc, matched_metadatas = self.db.find_similar_rule(document_text, repo=repo, distance_threshold=0.22)

        if not matched_id:
            # Clean insertion if the vector is fundamentally distinct
            self.db.store_rule(new_rule_json)
            return new_rule_json

        print("\\n[Semantic Fuser] ⚠️ Detected Duplicate Vector Collisions! Initiating LLM Merging Phase...")

        # 2. Instruct the DGX Server to combine the two semantic strings intelligently
        system_prompt = (
            "You merge two overlapping developer rules into one unified rule. "
            "Preserve all unique technical nuances from both. "
            "Output ONLY valid JSON:\n"
            "{\"rule_id\": \"descriptive-kebab-case-slug\", \"category\": \"security|performance|testing|code-style|architecture|correctness\", "
            "\"confidence\": 0.0, \"content\": {\"title\": \"...\", \"description\": \"...\", \"enforcement_prompt\": \"...\"}}\n"
            "rule_id must be a descriptive kebab-case slug (e.g. \"use-parameterized-sql-queries\"). NEVER use placeholders like \"uuid\"."
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
                temperature=0.1,
                **_qwen_extra_body(self.model),
            )
            msg = response.choices[0].message
            raw_content = (msg.content or "").strip()
            if not raw_content:
                raw_content = (getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or "").strip()
            if '</think>' in raw_content:
                raw_content = raw_content.split('</think>')[-1].strip()
            if raw_content.startswith("```json"):
                raw_content = raw_content.replace("```json", "").replace("```", "").strip()
            elif raw_content.startswith("```"):
                raw_content = raw_content.replace("```", "").strip()

            decoder = json.JSONDecoder()
            fused_rule, _ = decoder.raw_decode(raw_content.strip())

            # Keep the LLM's slug; store_rule namespaces it to a unique chroma
            # id (`<repo>__<slug>__<content-hash>`) and returns the canonical
            # id we use for the merged_into lineage reference. Falling back to
            # a random hex slug is fine — store_rule still namespaces it.
            if not fused_rule.get("rule_id"):
                fused_rule["rule_id"] = uuid.uuid4().hex[:8]

            # Mathematical Fusing: Combine total volumes observed of this architectural failure
            base_occurrences = matched_metadatas.get("occurrence_count", 1) if matched_metadatas else 1
            new_occurrences = new_rule_json.get("metadata", {}).get("occurrence_count", 1)

            if "metadata" not in fused_rule:
                fused_rule["metadata"] = {}

            fused_rule["metadata"]["occurrence_count"] = base_occurrences + new_occurrences
            fused_rule["metadata"]["repo"] = repo
            fused_rule["metadata"]["status"] = "needs_review"

            # `a or b or c` would skip legitimate falsy values like
            # confidence=0.0 or category="" — use explicit None checks.
            base_category = matched_metadatas.get("category") if matched_metadatas else None
            new_meta = new_rule_json.get("metadata", {})
            new_category = (
                new_meta.get("category")
                if new_meta.get("category") is not None
                else new_rule_json.get("category")
            )
            fused_rule["metadata"]["category"] = _first_meaningful(
                fused_rule.get("category"), new_category, base_category
            )
            base_conf = matched_metadatas.get("confidence") if matched_metadatas else None
            new_conf = (
                new_meta.get("confidence")
                if new_meta.get("confidence") is not None
                else new_rule_json.get("confidence")
            )
            fused_rule["metadata"]["confidence"] = _first_meaningful(
                fused_rule.get("confidence"), new_conf, base_conf
            )

            # Versioning: record which rules were merged to produce this one
            old_merge_count = matched_metadatas.get("merge_count", 0) if matched_metadatas else 0
            fused_rule["metadata"]["merge_count"] = old_merge_count + 1
            new_rule_id = new_rule_json.get("rule_id", "")
            fused_rule["metadata"]["merged_from"] = f"{matched_id},{new_rule_id}"

            # Provenance lineage: which models contributed to this fused rule.
            # ChromaDB metadata values must be str/int/float/bool — store as
            # comma-joined string, consistent with path_patterns convention.
            existing_models_raw = matched_metadatas.get("merged_with_models", "") if matched_metadatas else ""
            existing_origin = matched_metadatas.get("extracted_by_model") if matched_metadatas else None
            new_origin = new_rule_json.get("metadata", {}).get("extracted_by_model")

            # Build a deduped ordered list from existing comma-joined string + both origins.
            existing_models = [m for m in existing_models_raw.split(",") if m] if existing_models_raw else []
            lineage = list(existing_models)
            for origin in (existing_origin, new_origin):
                if origin and origin not in lineage:
                    lineage.append(origin)
            fused_rule["metadata"]["merged_with_models"] = ",".join(lineage)
            # Keep the first extractor as the canonical extracted_by; falls back
            # to new_origin if the existing rule pre-dates Task 4.
            if existing_origin:
                fused_rule["metadata"]["extracted_by_model"] = existing_origin
            elif new_origin:
                fused_rule["metadata"]["extracted_by_model"] = new_origin

            # 3. Store first so we have the canonical chroma id, then archive
            # the old rule pointing at it, then delete the old rule. archive
            # MUST run before delete so lineage isn't lost mid-operation.
            fused_chroma_id = self.db.store_rule(fused_rule)
            self.db.archive_rule(matched_id, merged_into_id=fused_chroma_id)
            self.db.delete_rule(matched_id)

            print(f"✅ Semantic Merge Complete! Replaced two messy concepts into single unified rule: {fused_rule['content']['title']}")
            return fused_rule
            
        except Exception as e:
            print(f"[Semantic Fuser Error]: Merging failed securely, retreating. Details: {e}")
            # Fallback to just storing it normally if the merging fails
            self.db.store_rule(new_rule_json)
            return new_rule_json
