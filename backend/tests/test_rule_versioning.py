"""
Tests for GitHub issue #11: Rule versioning and merge history.

Coverage:
  - LightRAGManager.archive_rule copies a rule to rule_history with merged_into metadata
  - LightRAGManager.get_rule_history returns archived rules for a given rule_id
  - SemanticFuser.process_and_fuse sets merged_from / merge_count, calls archive_rule
  - GET /api/rules/{rule_id}/history returns the merge lineage
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# LightRAGManager — archive_rule
# ---------------------------------------------------------------------------

class TestArchiveRule(unittest.TestCase):
    """archive_rule copies a rule into rule_history with merged_into metadata."""

    def _make_db(self):
        """Return a LightRAGManager whose ChromaDB client is fully mocked."""
        with patch("db.lightrag_manager.chromadb.PersistentClient") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client

            # Two distinct collection mocks: enterprise_rejections and rule_history
            mock_main_coll = MagicMock()
            mock_history_coll = MagicMock()
            mock_client.get_or_create_collection.side_effect = [
                mock_main_coll,
                mock_history_coll,
            ]

            from db.lightrag_manager import LightRAGManager
            db = LightRAGManager()

        db.collection = mock_main_coll
        db.history_collection = mock_history_coll
        return db, mock_main_coll, mock_history_coll

    def test_archive_rule_fetches_from_main_collection(self):
        db, mock_main, mock_history = self._make_db()

        mock_main.get.return_value = {
            "ids": ["rule-abc"],
            "documents": ["Rule: X - Context: Y. Enforce: Z"],
            "metadatas": [{"repo": "org/repo", "status": "active", "occurrence_count": 3}],
        }

        db.archive_rule("rule-abc", merged_into_id="rule-new")

        mock_main.get.assert_called_once_with(ids=["rule-abc"], include=["documents", "metadatas"])

    def test_archive_rule_adds_to_history_with_merged_into(self):
        db, mock_main, mock_history = self._make_db()

        original_metadata = {"repo": "org/repo", "status": "active", "occurrence_count": 3}
        mock_main.get.return_value = {
            "ids": ["rule-abc"],
            "documents": ["Rule: X - Context: Y. Enforce: Z"],
            "metadatas": [original_metadata],
        }

        db.archive_rule("rule-abc", merged_into_id="rule-new")

        mock_history.add.assert_called_once()
        call_kwargs = mock_history.add.call_args[1]
        stored_metadata = call_kwargs["metadatas"][0]

        self.assertEqual(stored_metadata["merged_into"], "rule-new")
        self.assertEqual(stored_metadata["original_rule_id"], "rule-abc")
        self.assertEqual(stored_metadata["repo"], "org/repo")
        self.assertEqual(stored_metadata["occurrence_count"], 3)

    def test_archive_rule_raises_when_rule_not_found(self):
        db, mock_main, mock_history = self._make_db()

        mock_main.get.return_value = {"ids": [], "documents": [], "metadatas": []}

        with self.assertRaises(ValueError):
            db.archive_rule("nonexistent-id", merged_into_id="rule-new")

        mock_history.add.assert_not_called()

    def test_archive_rule_preserves_document_text(self):
        db, mock_main, mock_history = self._make_db()

        document_text = "Rule: Never use eval - Context: security. Enforce: reject eval calls"
        mock_main.get.return_value = {
            "ids": ["rule-abc"],
            "documents": [document_text],
            "metadatas": [{"repo": "org/repo", "status": "active", "occurrence_count": 1}],
        }

        db.archive_rule("rule-abc", merged_into_id="rule-new")

        call_kwargs = mock_history.add.call_args[1]
        self.assertEqual(call_kwargs["documents"][0], document_text)


# ---------------------------------------------------------------------------
# LightRAGManager — get_rule_history
# ---------------------------------------------------------------------------

class TestGetRuleHistory(unittest.TestCase):
    """get_rule_history queries rule_history for entries pointing at a given rule_id."""

    def _make_db(self):
        with patch("db.lightrag_manager.chromadb.PersistentClient") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client

            mock_main_coll = MagicMock()
            mock_history_coll = MagicMock()
            mock_client.get_or_create_collection.side_effect = [
                mock_main_coll,
                mock_history_coll,
            ]

            from db.lightrag_manager import LightRAGManager
            db = LightRAGManager()

        db.collection = mock_main_coll
        db.history_collection = mock_history_coll
        return db, mock_main_coll, mock_history_coll

    def test_get_rule_history_queries_by_merged_into(self):
        db, _, mock_history = self._make_db()

        mock_history.get.return_value = {
            "ids": ["hist-1"],
            "documents": ["old document text"],
            "metadatas": [{"merged_into": "rule-new", "original_rule_id": "rule-abc"}],
        }

        results = db.get_rule_history("rule-new")

        mock_history.get.assert_called_once_with(
            where={"merged_into": "rule-new"},
            include=["documents", "metadatas"],
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["original_rule_id"], "rule-abc")

    def test_get_rule_history_returns_empty_list_when_no_history(self):
        db, _, mock_history = self._make_db()

        mock_history.get.return_value = {"ids": [], "documents": [], "metadatas": []}

        results = db.get_rule_history("rule-with-no-history")

        self.assertEqual(results, [])

    def test_get_rule_history_returns_all_matching_entries(self):
        db, _, mock_history = self._make_db()

        mock_history.get.return_value = {
            "ids": ["hist-1", "hist-2"],
            "documents": ["doc-1", "doc-2"],
            "metadatas": [
                {"merged_into": "rule-new", "original_rule_id": "rule-a"},
                {"merged_into": "rule-new", "original_rule_id": "rule-b"},
            ],
        }

        results = db.get_rule_history("rule-new")

        self.assertEqual(len(results), 2)
        original_ids = {r["original_rule_id"] for r in results}
        self.assertEqual(original_ids, {"rule-a", "rule-b"})


# ---------------------------------------------------------------------------
# SemanticFuser — merged_from, merge_count, archive_rule wiring
# ---------------------------------------------------------------------------

class TestSemanticFuserMergeHistory(unittest.TestCase):
    """
    SemanticFuser.process_and_fuse must:
    - call db.archive_rule(matched_id, merged_into_id=<new fused id>) before delete_rule
    - set fused_rule["metadata"]["merged_from"] = "<matched_id>,<new_rule_id>"
    - set fused_rule["metadata"]["merge_count"] incrementing the old rule's count
    """

    def _make_mock_db(self, matched_id="old-rule", matched_doc="old doc text",
                      matched_meta=None):
        if matched_meta is None:
            matched_meta = {"occurrence_count": 2, "merge_count": 0, "repo": "org/repo"}
        db = MagicMock()
        db.is_blocked.return_value = False
        db.find_similar_rule.return_value = (matched_id, matched_doc, matched_meta)
        return db

    def _make_new_rule(self, rule_id="new-input-rule"):
        return {
            "rule_id": rule_id,
            "metadata": {"repo": "org/repo", "occurrence_count": 1},
            "content": {
                "title": "Avoid eval",
                "description": "Never use eval in production",
                "enforcement_prompt": "Reject eval() calls",
            },
        }

    def _fused_response(self):
        return {
            "rule_id": "will-be-overwritten",
            "metadata": {"status": "active"},
            "content": {
                "title": "Avoid eval (merged)",
                "description": "Never use eval in production code",
                "enforcement_prompt": "Reject eval() calls strictly",
            },
        }

    def test_archive_rule_called_before_delete_rule(self):
        """archive_rule must be called with matched_id before delete_rule is called."""
        import json
        db = self._make_mock_db()
        new_rule = self._make_new_rule()

        call_order = []
        db.archive_rule.side_effect = lambda *a, **kw: call_order.append("archive")
        db.delete_rule.side_effect = lambda *a, **kw: call_order.append("delete")
        db.store_rule.side_effect = lambda *a, **kw: call_order.append("store")

        fused_json = json.dumps(self._fused_response())

        from pipeline.semantic_fuser import SemanticFuser
        fuser = SemanticFuser(db)

        with patch("pipeline.semantic_fuser.completion") as mock_completion:
            mock_response = MagicMock()
            mock_response.choices[0].message.content = fused_json
            mock_completion.return_value = mock_response

            fuser.process_and_fuse(new_rule)

        self.assertIn("archive", call_order)
        self.assertIn("delete", call_order)
        archive_pos = call_order.index("archive")
        delete_pos = call_order.index("delete")
        self.assertLess(archive_pos, delete_pos, "archive_rule must run before delete_rule")

    def test_archive_rule_called_with_correct_matched_id_and_fused_id(self):
        import json
        db = self._make_mock_db(matched_id="old-xyz")
        # store_rule returns the canonical chroma id, which is what
        # archive_rule must reference as merged_into.
        db.store_rule.return_value = "chroma-id-of-fused-rule"
        new_rule = self._make_new_rule()

        fused_json = json.dumps(self._fused_response())

        from pipeline.semantic_fuser import SemanticFuser
        fuser = SemanticFuser(db)

        with patch("pipeline.semantic_fuser.completion") as mock_completion:
            mock_response = MagicMock()
            mock_response.choices[0].message.content = fused_json
            mock_completion.return_value = mock_response

            fuser.process_and_fuse(new_rule)

        db.archive_rule.assert_called_once()
        call_kwargs = db.archive_rule.call_args
        # First positional arg must be the old matched_id
        self.assertEqual(call_kwargs[0][0], "old-xyz")
        # merged_into_id must equal the chroma id returned by store_rule.
        self.assertEqual(call_kwargs[1]["merged_into_id"], "chroma-id-of-fused-rule")

    def test_fused_rule_has_merged_from_metadata(self):
        import json
        db = self._make_mock_db(matched_id="old-abc")
        new_rule = self._make_new_rule(rule_id="input-xyz")

        fused_json = json.dumps(self._fused_response())

        from pipeline.semantic_fuser import SemanticFuser
        fuser = SemanticFuser(db)

        stored_rules = []
        db.store_rule.side_effect = lambda r: stored_rules.append(r)

        with patch("pipeline.semantic_fuser.completion") as mock_completion:
            mock_response = MagicMock()
            mock_response.choices[0].message.content = fused_json
            mock_completion.return_value = mock_response

            fuser.process_and_fuse(new_rule)

        self.assertEqual(len(stored_rules), 1)
        fused = stored_rules[0]
        merged_from = fused["metadata"].get("merged_from", "")
        self.assertIn("old-abc", merged_from)
        self.assertIn("input-xyz", merged_from)

    def test_fused_rule_merge_count_increments_from_old_rule(self):
        """merge_count must be old_rule.merge_count + 1."""
        import json
        db = self._make_mock_db(
            matched_meta={"occurrence_count": 5, "merge_count": 3, "repo": "org/repo"}
        )
        new_rule = self._make_new_rule()

        fused_json = json.dumps(self._fused_response())

        from pipeline.semantic_fuser import SemanticFuser
        fuser = SemanticFuser(db)

        stored_rules = []
        db.store_rule.side_effect = lambda r: stored_rules.append(r)

        with patch("pipeline.semantic_fuser.completion") as mock_completion:
            mock_response = MagicMock()
            mock_response.choices[0].message.content = fused_json
            mock_completion.return_value = mock_response

            fuser.process_and_fuse(new_rule)

        fused = stored_rules[0]
        self.assertEqual(fused["metadata"]["merge_count"], 4)

    def test_fused_rule_merge_count_starts_at_one_when_old_rule_has_no_count(self):
        """When old rule has no merge_count, the fused rule should have merge_count=1."""
        import json
        # matched_meta without merge_count key
        db = self._make_mock_db(
            matched_meta={"occurrence_count": 2, "repo": "org/repo"}
        )
        new_rule = self._make_new_rule()

        fused_json = json.dumps(self._fused_response())

        from pipeline.semantic_fuser import SemanticFuser
        fuser = SemanticFuser(db)

        stored_rules = []
        db.store_rule.side_effect = lambda r: stored_rules.append(r)

        with patch("pipeline.semantic_fuser.completion") as mock_completion:
            mock_response = MagicMock()
            mock_response.choices[0].message.content = fused_json
            mock_completion.return_value = mock_response

            fuser.process_and_fuse(new_rule)

        fused = stored_rules[0]
        self.assertEqual(fused["metadata"]["merge_count"], 1)


# ---------------------------------------------------------------------------
# API — GET /api/rules/{rule_id}/history
# ---------------------------------------------------------------------------

class TestRuleHistoryEndpoint(unittest.TestCase):
    """GET /api/rules/{rule_id}/history returns the merge lineage from rule_history."""

    def setUp(self):
        from fastapi.testclient import TestClient

        # Patch LightRAGManager so no real ChromaDB is constructed at import time
        with patch("db.lightrag_manager.chromadb.PersistentClient"):
            import api
            self.app = api.app
            self.db_mock = api.db

        self.client = TestClient(self.app)

    def test_history_endpoint_returns_lineage(self):
        history_entries = [
            {
                "original_rule_id": "rule-old",
                "document": "Rule: old text",
                "merged_into": "rule-new",
            }
        ]
        self.db_mock.get_rule_history = MagicMock(return_value=history_entries)

        response = self.client.get("/api/rules/rule-new/history")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("history", body)
        self.assertEqual(len(body["history"]), 1)
        self.assertEqual(body["history"][0]["original_rule_id"], "rule-old")

    def test_history_endpoint_returns_empty_list_when_no_history(self):
        self.db_mock.get_rule_history = MagicMock(return_value=[])

        response = self.client.get("/api/rules/rule-with-no-parents/history")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"history": []})

    def test_history_endpoint_calls_get_rule_history_with_correct_id(self):
        self.db_mock.get_rule_history = MagicMock(return_value=[])

        self.client.get("/api/rules/target-rule-id/history")

        self.db_mock.get_rule_history.assert_called_once_with("target-rule-id")


if __name__ == "__main__":
    unittest.main()
