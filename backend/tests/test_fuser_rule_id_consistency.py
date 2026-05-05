"""SemanticFuser must use the same collision-free rule_id strategy as
store_rule, instead of substituting `uuid.uuid4()[:8]` (which both ignores the
LLM slug and could collide on its own).
"""
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _completion_response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _make_db():
    from db.lightrag_manager import LightRAGManager
    db = LightRAGManager.__new__(LightRAGManager)
    db.collection = MagicMock()
    db.history_collection = MagicMock()
    return db


def _matched_setup(db):
    db.is_blocked = MagicMock(return_value=False)
    db.find_similar_rule = MagicMock(return_value=(
        "old-id", "old doc", {"occurrence_count": 1, "repo": "acme/app"},
    ))
    db.archive_rule = MagicMock()
    db.delete_rule = MagicMock()
    return db


class TestFusedRuleIdConsistency(unittest.TestCase):
    def test_fused_rule_id_uses_namespaced_helper_not_uuid(self):
        """A fused rule's stored chroma id must follow `<repo>__<slug>__<hash>`."""
        from pipeline.semantic_fuser import SemanticFuser

        db = _matched_setup(_make_db())
        fused_response = json.dumps({
            "rule_id": "use-eval-not-literal",
            "category": "security",
            "confidence": 0.9,
            "content": {
                "title": "Avoid eval",
                "description": "Use ast.literal_eval",
                "enforcement_prompt": "Reject eval()",
            },
        })

        from db.lightrag_manager import LightRAGManager
        db.store_rule = LightRAGManager.store_rule.__get__(db)

        new_rule = {
            "rule_id": "input-slug",
            "metadata": {"repo": "acme/app", "occurrence_count": 1},
            "content": {
                "title": "Avoid eval",
                "description": "Use ast.literal_eval",
                "enforcement_prompt": "Reject eval()",
            },
        }

        fuser = SemanticFuser(db)
        with patch("pipeline.semantic_fuser.completion") as mock:
            mock.return_value = _completion_response(fused_response)
            fuser.process_and_fuse(new_rule)

        stored_id = db.collection.add.call_args.kwargs["ids"][0]
        # Expected pattern: <repo>__<slug>__<6 hex>
        parts = stored_id.split("__")
        self.assertEqual(len(parts), 3, f"Expected namespaced id, got {stored_id!r}")
        self.assertEqual(parts[0], "acme-app")
        self.assertIn("use-eval-not-literal", parts[1])
        self.assertEqual(len(parts[2]), 6)

    def test_archive_rule_merged_into_id_matches_stored_chroma_id(self):
        """archive_rule(merged_into_id=...) must equal the actual chroma id of the new fused rule."""
        from pipeline.semantic_fuser import SemanticFuser

        db = _matched_setup(_make_db())
        from db.lightrag_manager import LightRAGManager
        db.store_rule = LightRAGManager.store_rule.__get__(db)

        new_rule = {
            "rule_id": "input-slug",
            "metadata": {"repo": "acme/app", "occurrence_count": 1},
            "content": {"title": "T", "description": "D", "enforcement_prompt": "E"},
        }
        fused_response = json.dumps({
            "rule_id": "merged-rule-slug",
            "content": {"title": "T", "description": "D", "enforcement_prompt": "E"},
        })

        fuser = SemanticFuser(db)
        with patch("pipeline.semantic_fuser.completion") as mock:
            mock.return_value = _completion_response(fused_response)
            fuser.process_and_fuse(new_rule)

        stored_id = db.collection.add.call_args.kwargs["ids"][0]
        merged_into_id = db.archive_rule.call_args.kwargs["merged_into_id"]
        self.assertEqual(merged_into_id, stored_id)


if __name__ == "__main__":
    unittest.main()
