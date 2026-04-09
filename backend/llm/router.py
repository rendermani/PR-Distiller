from typing import Dict, Any
from db.lightrag_manager import LightRAGManager

class SmallLLMRouter:
    """
    Stage 3 Small LLM Router.
    Pre-categorizes PR comments before expensive 70B Extraction.
    Uses the CodeRAG receiver to detect if an exact anti-pattern was already banned
    to bypass the large LLM extraction stage entirely!
    """
    def __init__(self, db_manager: LightRAGManager):
        self.db = db_manager
        
    def categorize_comment(self, pr_comment: str) -> Dict[str, Any]:
        """
        Routes the comment into one of the known pipelines.
        """
        # 1. Fast Vector Check - Has the AI done this exact mistake before?
        # A simple semantic check to bypass 70B invocation if we are 95%+ sure
        matched_vectors = self.db.get_contextual_rules(pr_comment, top_k=1)
        
        # In a real environment, you'd check chromadb distance metrics.
        # If we have a vector that strongly correlates to the comment, short-circuit!
        if matched_vectors:
            return {
                "route": "exact_match_found",
                "cached_rule": matched_vectors[0]
            }

        # 2. Basic Noise / Instruction Filter
        lower_comment = pr_comment.lower()
        if "metadata doesn’t match" in lower_comment or "bump version" in lower_comment:
            return {
                "route": "discard_noise"
            }
            
        if "migrate" in lower_comment or "deprecated" in lower_comment:
            return {
                "route": "needs_deep_dive",
                "label": "reason:migration_instruction"
            }
            
        return {
            "route": "extract_direct",
            "label": "topic:architecture_design"
        }
