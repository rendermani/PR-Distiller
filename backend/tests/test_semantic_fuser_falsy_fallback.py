"""Tests for SemanticFuser falsy-fallback bug.

`a or b or c` skips legitimate falsy values like 0.0 and "". The fuser must use
explicit None checks so confidence=0.0 from the LLM is preserved instead of
falling through to the new/base rule's value.
"""
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _make_mock_db():
    db = MagicMock()
    db.is_blocked.return_value = False
    db.find_similar_rule.return_value = (
        "old-rule",
        "old doc",
        {"occurrence_count": 1, "repo": "org/repo", "confidence": 0.9, "category": "security"},
    )
    return db


def _new_rule(confidence=0.5, category="performance"):
    rule = {
        "rule_id": "new-rule",
        "metadata": {"repo": "org/repo", "occurrence_count": 1},
        "content": {
            "title": "Test rule",
            "description": "Description",
            "enforcement_prompt": "Enforce it",
        },
    }
    if confidence is not None:
        rule["metadata"]["confidence"] = confidence
    if category is not None:
        rule["metadata"]["category"] = category
    return rule


def _fused_response_with(confidence, category):
    body = {
        "rule_id": "ignored-by-fuser",
        "content": {
            "title": "Merged",
            "description": "Merged desc",
            "enforcement_prompt": "Merged enforce",
        },
    }
    if confidence is not None:
        body["confidence"] = confidence
    if category is not None:
        body["category"] = category
    return body


class TestFalsyFallback(unittest.TestCase):
    def _run_fuser(self, fused_response, new_rule):
        from pipeline.semantic_fuser import SemanticFuser

        db = _make_mock_db()
        stored = []
        db.store_rule.side_effect = lambda r: stored.append(r)
        fuser = SemanticFuser(db)
        with patch("pipeline.semantic_fuser.completion") as mock_completion:
            mock_response = MagicMock()
            mock_response.choices[0].message.content = json.dumps(fused_response)
            mock_completion.return_value = mock_response
            fuser.process_and_fuse(new_rule)
        return stored[0]

    def test_zero_confidence_from_fused_response_is_preserved(self):
        """confidence=0.0 from the LLM merge must be preserved, not replaced by new/base."""
        fused = self._run_fuser(
            _fused_response_with(confidence=0.0, category="security"),
            _new_rule(confidence=0.5),
        )
        self.assertEqual(fused["metadata"]["confidence"], 0.0)

    def test_empty_string_category_from_fused_response_falls_back(self):
        """An empty category from the fused response should fall back to new_rule's category."""
        # An empty string IS a missing value here — fall through is desired.
        fused = self._run_fuser(
            _fused_response_with(confidence=0.5, category=""),
            _new_rule(category="performance"),
        )
        self.assertEqual(fused["metadata"]["category"], "performance")

    def test_zero_confidence_from_new_rule_used_when_fused_omits_it(self):
        """When fused response omits confidence, new_rule.confidence=0.0 must be used."""
        fused = self._run_fuser(
            _fused_response_with(confidence=None, category="security"),
            _new_rule(confidence=0.0),
        )
        self.assertEqual(fused["metadata"]["confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
