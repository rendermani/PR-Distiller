"""
Tests for issue #4: file-path scoping in MCP queries.

Covers:
- store_rule() stores path_patterns as comma-separated string in ChromaDB metadata
- get_contextual_rules() with no file_path returns all rules (backward compat)
- get_contextual_rules() with file_path filters by pattern matching
- Unscoped rules always match regardless of file_path
- Rules with non-matching patterns are excluded
- Multiple patterns: any match is sufficient to include the rule
- Empty path_patterns list stored as empty string
"""
import os
import sys
import fnmatch
import unittest
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Helper: build a minimal rule dict as the extractor would produce
# ---------------------------------------------------------------------------

def _make_rule(rule_id: str, patterns: list, title: str = "No bare except") -> dict:
    return {
        "rule_id": rule_id,
        "scoping": {"path_patterns": patterns},
        "metadata": {"status": "needs_review", "repo": "test/repo", "occurrence_count": 1},
        "content": {
            "title": title,
            "description": "desc",
            "enforcement_prompt": "enforce",
            "code_examples": {"bad_code": "bad", "good_code": "good"},
        },
    }


# ---------------------------------------------------------------------------
# store_rule — path_patterns persisted to ChromaDB metadata
# ---------------------------------------------------------------------------

class TestStoreRulePathPatterns(unittest.TestCase):
    """store_rule must write path_patterns as a comma-separated string in metadata."""

    def _make_db_with_mock_collection(self):
        """Return a LightRAGManager whose ChromaDB collection is fully mocked."""
        from db.lightrag_manager import LightRAGManager

        db = LightRAGManager.__new__(LightRAGManager)
        db.collection = MagicMock()
        return db

    def test_stores_single_path_pattern_as_string(self):
        db = self._make_db_with_mock_collection()
        rule = _make_rule("r1", ["**/auth/*.py"])

        from db.lightrag_manager import LightRAGManager
        LightRAGManager.store_rule(db, rule)

        _, kwargs = db.collection.add.call_args
        stored_meta = kwargs["metadatas"][0]
        self.assertEqual(stored_meta["path_patterns"], "**/auth/*.py")

    def test_stores_multiple_path_patterns_comma_separated(self):
        db = self._make_db_with_mock_collection()
        rule = _make_rule("r2", ["**/auth/*.py", "**/security/**"])

        from db.lightrag_manager import LightRAGManager
        LightRAGManager.store_rule(db, rule)

        _, kwargs = db.collection.add.call_args
        stored_meta = kwargs["metadatas"][0]
        self.assertEqual(stored_meta["path_patterns"], "**/auth/*.py,**/security/**")

    def test_stores_empty_path_patterns_as_empty_string(self):
        db = self._make_db_with_mock_collection()
        rule = _make_rule("r3", [])

        from db.lightrag_manager import LightRAGManager
        LightRAGManager.store_rule(db, rule)

        _, kwargs = db.collection.add.call_args
        stored_meta = kwargs["metadatas"][0]
        self.assertEqual(stored_meta["path_patterns"], "")

    def test_stores_empty_string_when_scoping_key_absent(self):
        """Rules extracted without any scoping block default to unscoped."""
        db = self._make_db_with_mock_collection()
        rule = _make_rule("r4", [])
        del rule["scoping"]  # simulate missing key

        from db.lightrag_manager import LightRAGManager
        LightRAGManager.store_rule(db, rule)

        _, kwargs = db.collection.add.call_args
        stored_meta = kwargs["metadatas"][0]
        self.assertEqual(stored_meta["path_patterns"], "")

    def test_path_patterns_metadata_value_is_string_type(self):
        """ChromaDB only accepts str/int/float/bool — never a list."""
        db = self._make_db_with_mock_collection()
        rule = _make_rule("r5", ["**/*.ts"])

        from db.lightrag_manager import LightRAGManager
        LightRAGManager.store_rule(db, rule)

        _, kwargs = db.collection.add.call_args
        stored_meta = kwargs["metadatas"][0]
        self.assertIsInstance(stored_meta["path_patterns"], str)


# ---------------------------------------------------------------------------
# get_contextual_rules — no file_path → current behavior preserved
# ---------------------------------------------------------------------------

class TestGetContextualRulesNoFilePath(unittest.TestCase):
    """When file_path is omitted, all active rules are returned (no filtering)."""

    def _db_with_query_result(self, query_result: dict):
        from db.lightrag_manager import LightRAGManager

        db = LightRAGManager.__new__(LightRAGManager)
        db.collection = MagicMock()
        db.collection.query.return_value = query_result
        return db

    def _chromadb_result(self, docs: list, metas: list, ids: list) -> dict:
        return {
            "ids": [ids],
            "documents": [docs],
            "metadatas": [metas],
            "distances": [[0.1] * len(docs)],
        }

    def test_returns_all_results_when_no_file_path_given(self):
        metas = [
            {"rule_id": "r1", "status": "active", "path_patterns": "**/auth/*.py"},
            {"rule_id": "r2", "status": "active", "path_patterns": ""},
        ]
        result = self._chromadb_result(["doc1", "doc2"], metas, ["r1", "r2"])
        db = self._db_with_query_result(result)

        from db.lightrag_manager import LightRAGManager
        returned = LightRAGManager.get_contextual_rules(db, "some diff")

        # All documents pass through unfiltered
        self.assertEqual(len(returned["documents"][0]), 2)

    def test_returns_empty_list_when_collection_is_empty(self):
        result = {"documents": [[]], "ids": [[]], "metadatas": [[]], "distances": [[]]}
        db = self._db_with_query_result(result)

        from db.lightrag_manager import LightRAGManager
        returned = LightRAGManager.get_contextual_rules(db, "diff")

        self.assertEqual(returned, [])


# ---------------------------------------------------------------------------
# get_contextual_rules — with file_path → post-filter applied
# ---------------------------------------------------------------------------

class TestGetContextualRulesWithFilePath(unittest.TestCase):
    """When file_path is provided, only matching or unscoped rules are returned."""

    def _db_returning(self, docs, metas, ids):
        from db.lightrag_manager import LightRAGManager

        db = LightRAGManager.__new__(LightRAGManager)
        db.collection = MagicMock()
        db.collection.query.return_value = {
            "ids": [ids],
            "documents": [docs],
            "metadatas": [metas],
            "distances": [[0.1] * len(docs)],
        }
        return db

    def test_scoped_rule_matches_when_file_path_matches_pattern(self):
        db = self._db_returning(
            docs=["auth rule"],
            metas=[{"rule_id": "r1", "status": "active", "path_patterns": "**/auth/*.py"}],
            ids=["r1"],
        )

        from db.lightrag_manager import LightRAGManager
        result = LightRAGManager.get_contextual_rules(db, "diff", file_path="src/auth/login.py")

        self.assertEqual(len(result["documents"][0]), 1)
        self.assertEqual(result["documents"][0][0], "auth rule")

    def test_scoped_rule_excluded_when_file_path_does_not_match_pattern(self):
        db = self._db_returning(
            docs=["auth rule"],
            metas=[{"rule_id": "r1", "status": "active", "path_patterns": "**/auth/*.py"}],
            ids=["r1"],
        )

        from db.lightrag_manager import LightRAGManager
        result = LightRAGManager.get_contextual_rules(db, "diff", file_path="src/models/user.py")

        self.assertEqual(len(result["documents"][0]), 0)

    def test_unscoped_rule_always_matches(self):
        db = self._db_returning(
            docs=["general rule"],
            metas=[{"rule_id": "r2", "status": "active", "path_patterns": ""}],
            ids=["r2"],
        )

        from db.lightrag_manager import LightRAGManager
        result = LightRAGManager.get_contextual_rules(db, "diff", file_path="src/anything/foo.py")

        self.assertEqual(len(result["documents"][0]), 1)
        self.assertEqual(result["documents"][0][0], "general rule")

    def test_missing_path_patterns_key_treated_as_unscoped(self):
        """Older rules in the DB without path_patterns metadata key always match."""
        db = self._db_returning(
            docs=["legacy rule"],
            metas=[{"rule_id": "r3", "status": "active"}],
            ids=["r3"],
        )

        from db.lightrag_manager import LightRAGManager
        result = LightRAGManager.get_contextual_rules(db, "diff", file_path="anything.py")

        self.assertEqual(len(result["documents"][0]), 1)

    def test_rule_with_multiple_patterns_matches_on_any_pattern(self):
        db = self._db_returning(
            docs=["security rule"],
            metas=[{"rule_id": "r4", "status": "active", "path_patterns": "**/auth/*.py,**/security/**"}],
            ids=["r4"],
        )

        from db.lightrag_manager import LightRAGManager
        # Matches the second pattern
        result = LightRAGManager.get_contextual_rules(db, "diff", file_path="src/security/validator.py")

        self.assertEqual(len(result["documents"][0]), 1)

    def test_mixed_rules_only_matching_ones_returned(self):
        """Auth-scoped rule excluded; unscoped rule always included."""
        db = self._db_returning(
            docs=["auth rule", "general rule"],
            metas=[
                {"rule_id": "r1", "status": "active", "path_patterns": "**/auth/*.py"},
                {"rule_id": "r2", "status": "active", "path_patterns": ""},
            ],
            ids=["r1", "r2"],
        )

        from db.lightrag_manager import LightRAGManager
        result = LightRAGManager.get_contextual_rules(db, "diff", file_path="src/models/user.py")

        # Auth rule excluded, general rule included
        self.assertEqual(len(result["documents"][0]), 1)
        self.assertEqual(result["documents"][0][0], "general rule")

    def test_returns_empty_list_when_no_rules_match_file_path(self):
        db = self._db_returning(
            docs=["auth rule"],
            metas=[{"rule_id": "r1", "status": "active", "path_patterns": "**/auth/*.py"}],
            ids=["r1"],
        )

        from db.lightrag_manager import LightRAGManager
        result = LightRAGManager.get_contextual_rules(db, "diff", file_path="src/models/user.py")

        self.assertEqual(result["documents"][0], [])

    def test_preserves_ids_and_metadatas_after_filtering(self):
        """After filtering, ids and metadatas lists must stay aligned with documents."""
        db = self._db_returning(
            docs=["auth rule", "general rule"],
            metas=[
                {"rule_id": "r1", "status": "active", "path_patterns": "**/auth/*.py"},
                {"rule_id": "r2", "status": "active", "path_patterns": ""},
            ],
            ids=["r1", "r2"],
        )

        from db.lightrag_manager import LightRAGManager
        result = LightRAGManager.get_contextual_rules(db, "diff", file_path="src/models/user.py")

        self.assertEqual(result["ids"][0], ["r2"])
        self.assertEqual(result["metadatas"][0][0]["rule_id"], "r2")


# ---------------------------------------------------------------------------
# FastAPI endpoint — QueryRequest accepts optional file_path
# ---------------------------------------------------------------------------

class TestQueryRequestModel(unittest.TestCase):
    """QueryRequest Pydantic model must accept an optional file_path field."""

    def test_query_request_accepts_file_path(self):
        from api import QueryRequest

        req = QueryRequest(code_diff="some diff", top_k=3, file_path="src/auth/login.py")
        self.assertEqual(req.file_path, "src/auth/login.py")

    def test_query_request_file_path_defaults_to_none(self):
        from api import QueryRequest

        req = QueryRequest(code_diff="some diff")
        self.assertIsNone(req.file_path)


class TestMCPEndpointPassesFilePath(unittest.TestCase):
    """mcp_contextual_query must pass file_path through to get_contextual_rules."""

    def test_endpoint_passes_file_path_to_db(self):
        from unittest.mock import patch, MagicMock
        from fastapi.testclient import TestClient
        import api as api_module

        mock_db = MagicMock()
        mock_db.get_contextual_rules.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
        }

        with patch.object(api_module, "db", mock_db):
            from fastapi.testclient import TestClient
            client = TestClient(api_module.app)
            client.post(
                "/api/mcp/query",
                json={"code_diff": "def foo(): pass", "top_k": 3, "file_path": "src/auth/login.py"},
            )

        # categories defaults to None and is passed as the third positional arg
        mock_db.get_contextual_rules.assert_called_once_with(
            "def foo(): pass", 3, None, file_path="src/auth/login.py"
        )

    def test_endpoint_passes_none_file_path_when_omitted(self):
        from unittest.mock import patch, MagicMock
        import api as api_module

        mock_db = MagicMock()
        mock_db.get_contextual_rules.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
        }

        with patch.object(api_module, "db", mock_db):
            from fastapi.testclient import TestClient
            client = TestClient(api_module.app)
            client.post(
                "/api/mcp/query",
                json={"code_diff": "def foo(): pass", "top_k": 3},
            )

        # categories defaults to None and is passed as the third positional arg
        mock_db.get_contextual_rules.assert_called_once_with(
            "def foo(): pass", 3, None, file_path=None
        )


if __name__ == "__main__":
    unittest.main()
