"""Merging two rules must not silently widen or de-label the survivor.

process_and_fuse() rebuilt fused_rule["metadata"] field by field, copying only
a fixed set. Two things were dropped:

- `scoping.path_patterns`. store_rule reads rule_json["scoping"], so a merged
  rule was stored with path_patterns="" — and _rule_matches_file_path treats an
  empty pattern string as "unscoped, always matches". A rule correctly scoped
  to **/auth/*.py became global on its first merge.
- `extracted_by_label`, the human-readable model name, so the UI lost model
  attribution on exactly the rules a multi-model run produces most.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_fuser(matched_metadata: dict, matched_document: str = "old rule text"):
    """A SemanticFuser whose DB reports one near-duplicate match."""
    from pipeline.semantic_fuser import SemanticFuser

    db = MagicMock()
    # is_blocked must be explicitly False: a bare MagicMock is truthy, which
    # makes process_and_fuse short-circuit before it ever merges.
    db.is_blocked.return_value = False
    db.find_similar_rule.return_value = ("matched-id", matched_document, matched_metadata)
    db.store_rule.return_value = "fused-id"

    fuser = SemanticFuser(
        db, model="ollama/qwen3:8b", api_base="http://localhost:11434", api_key="unused"
    )
    return fuser, db


def _new_rule(**overrides):
    rule = {
        "rule_id": "new-id",
        "title": "Use parameterized SQL",
        "description": "Avoid string interpolation in queries.",
        "scoping": {"path_patterns": ["**/api/*.py"]},
        "metadata": {
            "occurrence_count": 1,
            "extracted_by_model": "gemma",
            "extracted_by_label": "Gemma 4 E4B",
        },
    }
    rule.update(overrides)
    return rule


class TestFuserPreservesScoping(unittest.TestCase):
    """Path scoping must survive a merge, or the rule silently goes global."""

    def test_new_rule_scoping_is_carried_into_the_fused_rule(self):
        fuser, db = _make_fuser({"path_patterns": "**/api/*.py", "occurrence_count": 1})
        fuser._call_llm_merge = MagicMock(
            return_value={"title": "Use parameterized SQL", "description": "merged"}
        )

        fuser.process_and_fuse(_new_rule())

        stored = db.store_rule.call_args[0][0]
        self.assertEqual(
            stored.get("scoping", {}).get("path_patterns"), ["**/api/*.py"]
        )

    def test_existing_rule_scoping_is_preserved_when_new_rule_is_unscoped(self):
        """A scoped rule must not be widened by merging an unscoped one into it."""
        fuser, db = _make_fuser({"path_patterns": "**/auth/*.py", "occurrence_count": 1})
        fuser._call_llm_merge = MagicMock(
            return_value={"title": "t", "description": "merged"}
        )

        fuser.process_and_fuse(_new_rule(scoping={"path_patterns": []}))

        stored = db.store_rule.call_args[0][0]
        self.assertEqual(
            stored.get("scoping", {}).get("path_patterns"), ["**/auth/*.py"]
        )

    def test_patterns_from_both_rules_are_unioned(self):
        fuser, db = _make_fuser({"path_patterns": "**/auth/*.py", "occurrence_count": 1})
        fuser._call_llm_merge = MagicMock(
            return_value={"title": "t", "description": "merged"}
        )

        fuser.process_and_fuse(_new_rule(scoping={"path_patterns": ["**/api/*.py"]}))

        stored = db.store_rule.call_args[0][0]
        patterns = stored.get("scoping", {}).get("path_patterns", [])
        self.assertIn("**/auth/*.py", patterns)
        self.assertIn("**/api/*.py", patterns)

    def test_no_scoping_anywhere_stays_unscoped(self):
        """Genuinely global rules must remain global — no invented patterns."""
        fuser, db = _make_fuser({"occurrence_count": 1})
        fuser._call_llm_merge = MagicMock(
            return_value={"title": "t", "description": "merged"}
        )

        fuser.process_and_fuse(_new_rule(scoping={"path_patterns": []}))

        stored = db.store_rule.call_args[0][0]
        self.assertEqual(stored.get("scoping", {}).get("path_patterns", []), [])


class TestFuserPreservesModelLabel(unittest.TestCase):
    def test_extracted_by_label_survives_the_merge(self):
        fuser, db = _make_fuser({"occurrence_count": 1})
        fuser._call_llm_merge = MagicMock(
            return_value={"title": "t", "description": "merged"}
        )

        fuser.process_and_fuse(_new_rule())

        stored = db.store_rule.call_args[0][0]
        self.assertEqual(
            stored["metadata"].get("extracted_by_label"), "Gemma 4 E4B"
        )

    def test_existing_label_wins_to_match_extracted_by_model(self):
        """extracted_by_model keeps the first extractor; the label must agree."""
        fuser, db = _make_fuser({
            "occurrence_count": 1,
            "extracted_by_model": "qwen",
            "extracted_by_label": "Qwen 2.5 Coder",
        })
        fuser._call_llm_merge = MagicMock(
            return_value={"title": "t", "description": "merged"}
        )

        fuser.process_and_fuse(_new_rule())

        stored = db.store_rule.call_args[0][0]
        self.assertEqual(stored["metadata"]["extracted_by_model"], "qwen")
        self.assertEqual(stored["metadata"]["extracted_by_label"], "Qwen 2.5 Coder")


if __name__ == "__main__":
    unittest.main()
