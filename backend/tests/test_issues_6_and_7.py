"""
Tests for GitHub issue #6 (confidence scoring) and issue #7 (rule categories/tags).

Issue #6: LLM outputs a confidence float (0.0–1.0). If confidence > 0.8, the rule
          is auto-approved (status = "active"). Otherwise status = "needs_review".
          Confidence is stored in rule_dict["metadata"]["confidence"].

Issue #7: LLM outputs a category string from a fixed set of 6 values.
          Category is stored in rule_dict["metadata"]["category"].
          ChromaDB receives category in the metadata dict.
          get_contextual_rules() accepts an optional list of categories to filter by.
          QueryRequest accepts an optional categories list, passed through to the DB.
"""
import os
import sys
import json
import unittest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

VALID_CATEGORIES = {"security", "performance", "testing", "code-style", "architecture", "correctness"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_response(confidence: float, category: str, skip: bool = False) -> str:
    if skip:
        return json.dumps({"skip": True})
    return json.dumps({
        "rule_id": "abc12345",
        "metadata": {"status": "active"},
        "scoping": {"path_patterns": ["**/*.py"]},
        "content": {
            "title": "Avoid eval()",
            "description": "eval() is a security risk",
            "enforcement_prompt": "Never use eval()",
            "code_examples": {"bad_code": "eval(x)", "good_code": "ast.literal_eval(x)"},
        },
        "confidence": confidence,
        "category": category,
    })


def _make_mock_litellm_response(content: str):
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = content
    return mock_resp


# ---------------------------------------------------------------------------
# Issue #7 — SYSTEM_PROMPT schema
# ---------------------------------------------------------------------------

class TestSystemPromptSchema(unittest.TestCase):
    """SYSTEM_PROMPT must describe both new fields so the LLM knows to output them."""

    def test_system_prompt_mentions_confidence_field(self):
        from llm.extractor import LargeLLMExtractor
        self.assertIn("confidence", LargeLLMExtractor.SYSTEM_PROMPT)

    def test_system_prompt_mentions_category_field(self):
        from llm.extractor import LargeLLMExtractor
        self.assertIn("category", LargeLLMExtractor.SYSTEM_PROMPT)

    def test_system_prompt_lists_all_six_categories(self):
        from llm.extractor import LargeLLMExtractor
        for cat in VALID_CATEGORIES:
            self.assertIn(cat, LargeLLMExtractor.SYSTEM_PROMPT,
                          msg=f"Category '{cat}' missing from SYSTEM_PROMPT")


# ---------------------------------------------------------------------------
# Issue #6 — confidence propagation in extract_rule (sync)
# ---------------------------------------------------------------------------

class TestExtractRuleConfidenceSync(unittest.TestCase):

    def _make_extractor(self):
        mock_db = MagicMock()
        mock_fuser = MagicMock()
        with patch("llm.extractor.SemanticFuser", return_value=mock_fuser):
            from llm.extractor import LargeLLMExtractor
            extractor = LargeLLMExtractor(mock_db)
            extractor.fuser = mock_fuser
        return extractor

    def test_high_confidence_sets_status_active(self):
        """confidence > 0.8 must set status to 'active' (auto-approve)."""
        extractor = self._make_extractor()
        llm_json = _make_llm_response(confidence=0.92, category="security")

        with patch("llm.extractor.completion", return_value=_make_mock_litellm_response(llm_json)):
            result = extractor.extract_rule("do not eval", "diff", "owner/repo")

        self.assertIsNotNone(result)
        self.assertEqual(result["metadata"]["status"], "active")

    def test_low_confidence_keeps_status_needs_review(self):
        """confidence <= 0.8 must leave status as 'needs_review'."""
        extractor = self._make_extractor()
        llm_json = _make_llm_response(confidence=0.6, category="correctness")

        with patch("llm.extractor.completion", return_value=_make_mock_litellm_response(llm_json)):
            result = extractor.extract_rule("style note", "diff", "owner/repo")

        self.assertIsNotNone(result)
        self.assertEqual(result["metadata"]["status"], "needs_review")

    def test_boundary_confidence_0_8_keeps_needs_review(self):
        """Exactly 0.8 is not > 0.8, so status must remain 'needs_review'."""
        extractor = self._make_extractor()
        llm_json = _make_llm_response(confidence=0.8, category="performance")

        with patch("llm.extractor.completion", return_value=_make_mock_litellm_response(llm_json)):
            result = extractor.extract_rule("comment", "diff", "owner/repo")

        self.assertIsNotNone(result)
        self.assertEqual(result["metadata"]["status"], "needs_review")

    def test_confidence_stored_in_metadata(self):
        extractor = self._make_extractor()
        llm_json = _make_llm_response(confidence=0.75, category="architecture")

        with patch("llm.extractor.completion", return_value=_make_mock_litellm_response(llm_json)):
            result = extractor.extract_rule("comment", "diff", "owner/repo")

        self.assertAlmostEqual(result["metadata"]["confidence"], 0.75)

    def test_category_stored_in_metadata(self):
        extractor = self._make_extractor()
        llm_json = _make_llm_response(confidence=0.9, category="testing")

        with patch("llm.extractor.completion", return_value=_make_mock_litellm_response(llm_json)):
            result = extractor.extract_rule("comment", "diff", "owner/repo")

        self.assertEqual(result["metadata"]["category"], "testing")


# ---------------------------------------------------------------------------
# Issue #6 — confidence propagation in async_extract_rule
# ---------------------------------------------------------------------------

class TestExtractRuleConfidenceAsync(unittest.IsolatedAsyncioTestCase):

    def _make_extractor(self):
        mock_db = MagicMock()
        mock_fuser = MagicMock()
        with patch("llm.extractor.SemanticFuser", return_value=mock_fuser):
            from llm.extractor import LargeLLMExtractor
            extractor = LargeLLMExtractor(mock_db)
            extractor.fuser = mock_fuser
        return extractor

    async def test_async_high_confidence_sets_status_active(self):
        extractor = self._make_extractor()
        llm_json = _make_llm_response(confidence=0.95, category="security")

        with patch("llm.extractor.acompletion", return_value=_make_mock_litellm_response(llm_json)):
            result = await extractor.async_extract_rule("do not eval", "diff", "owner/repo")

        self.assertIsNotNone(result)
        self.assertEqual(result["metadata"]["status"], "active")

    async def test_async_low_confidence_keeps_needs_review(self):
        extractor = self._make_extractor()
        llm_json = _make_llm_response(confidence=0.5, category="code-style")

        with patch("llm.extractor.acompletion", return_value=_make_mock_litellm_response(llm_json)):
            result = await extractor.async_extract_rule("style note", "diff", "owner/repo")

        self.assertIsNotNone(result)
        self.assertEqual(result["metadata"]["status"], "needs_review")

    async def test_async_confidence_stored_in_metadata(self):
        extractor = self._make_extractor()
        llm_json = _make_llm_response(confidence=0.65, category="performance")

        with patch("llm.extractor.acompletion", return_value=_make_mock_litellm_response(llm_json)):
            result = await extractor.async_extract_rule("comment", "diff", "owner/repo")

        self.assertAlmostEqual(result["metadata"]["confidence"], 0.65)

    async def test_async_category_stored_in_metadata(self):
        extractor = self._make_extractor()
        llm_json = _make_llm_response(confidence=0.88, category="architecture")

        with patch("llm.extractor.acompletion", return_value=_make_mock_litellm_response(llm_json)):
            result = await extractor.async_extract_rule("comment", "diff", "owner/repo")

        self.assertEqual(result["metadata"]["category"], "architecture")


# ---------------------------------------------------------------------------
# Issue #7 — store_rule passes confidence and category to ChromaDB
# ---------------------------------------------------------------------------

class TestStoreRulePassesNewFields(unittest.TestCase):
    """
    store_rule() must include confidence and category in the metadatas dict
    passed to collection.add().
    """

    def _make_db(self):
        from db.lightrag_manager import LightRAGManager
        db = LightRAGManager.__new__(LightRAGManager)
        db.collection = MagicMock()
        return db

    def test_store_rule_includes_confidence_in_chroma_metadata(self):
        db = self._make_db()
        rule = {
            "rule_id": "rule01",
            "metadata": {
                "status": "active",
                "repo": "owner/repo",
                "occurrence_count": 1,
                "confidence": 0.91,
                "category": "security",
            },
            "content": {
                "title": "Avoid eval",
                "description": "Security risk",
                "enforcement_prompt": "Never use eval()",
            },
        }
        db.store_rule(rule)

        call_kwargs = db.collection.add.call_args
        stored_metadata = call_kwargs.kwargs["metadatas"][0]
        self.assertAlmostEqual(stored_metadata["confidence"], 0.91)

    def test_store_rule_includes_category_in_chroma_metadata(self):
        db = self._make_db()
        rule = {
            "rule_id": "rule02",
            "metadata": {
                "status": "needs_review",
                "repo": "owner/repo",
                "occurrence_count": 1,
                "confidence": 0.70,
                "category": "performance",
            },
            "content": {
                "title": "Avoid N+1 queries",
                "description": "Performance hit",
                "enforcement_prompt": "Use select_related()",
            },
        }
        db.store_rule(rule)

        call_kwargs = db.collection.add.call_args
        stored_metadata = call_kwargs.kwargs["metadatas"][0]
        self.assertEqual(stored_metadata["category"], "performance")

    def test_store_rule_without_confidence_omits_it_from_metadata(self):
        """Rules stored without confidence (e.g. manually added) must not crash."""
        db = self._make_db()
        rule = {
            "rule_id": "rule03",
            "metadata": {
                "status": "needs_review",
                "repo": "owner/repo",
                "occurrence_count": 1,
            },
            "content": {
                "title": "No magic strings",
                "description": "Use constants",
                "enforcement_prompt": "Extract strings",
            },
        }
        # Must not raise; confidence key simply absent from ChromaDB metadata
        db.store_rule(rule)
        db.collection.add.assert_called_once()


# ---------------------------------------------------------------------------
# Issue #7 — get_contextual_rules category filter
# ---------------------------------------------------------------------------

class TestGetContextualRulesCategoryFilter(unittest.TestCase):

    def _make_db(self):
        from db.lightrag_manager import LightRAGManager
        db = LightRAGManager.__new__(LightRAGManager)
        db.collection = MagicMock()
        db.collection.query.return_value = {"documents": [[]], "ids": [[]], "metadatas": [[]]}
        return db

    def test_no_category_filter_uses_only_active_status_filter(self):
        """Without a category filter, where clause must be {"status": "active"}."""
        db = self._make_db()
        db.get_contextual_rules("some diff", top_k=3)

        call_kwargs = db.collection.query.call_args.kwargs
        self.assertEqual(call_kwargs["where"], {"status": "active"})

    def test_single_category_filter_added_to_where_clause(self):
        """Passing categories=["security"] must add a category filter."""
        db = self._make_db()
        db.get_contextual_rules("some diff", top_k=3, categories=["security"])

        call_kwargs = db.collection.query.call_args.kwargs
        where = call_kwargs["where"]
        # Must filter active AND the given category
        self.assertIn("$and", where)
        conditions = where["$and"]
        self.assertIn({"status": "active"}, conditions)
        self.assertIn({"category": {"$in": ["security"]}}, conditions)

    def test_multiple_categories_use_dollar_in_operator(self):
        """Passing categories=["security", "performance"] must use $in."""
        db = self._make_db()
        db.get_contextual_rules("some diff", top_k=3, categories=["security", "performance"])

        call_kwargs = db.collection.query.call_args.kwargs
        where = call_kwargs["where"]
        self.assertIn("$and", where)
        conditions = where["$and"]
        self.assertIn({"category": {"$in": ["security", "performance"]}}, conditions)


# ---------------------------------------------------------------------------
# Issue #7 — QueryRequest model accepts categories field
# ---------------------------------------------------------------------------

class TestQueryRequestModel(unittest.TestCase):

    def test_query_request_has_categories_field_defaulting_to_none(self):
        import importlib
        import types

        # Import api module without running uvicorn startup side-effects is tricky;
        # patch the db-level imports so the module loads cleanly.
        mock_db = MagicMock()
        mock_conf = MagicMock()
        mock_orch = MagicMock()

        with patch("db.lightrag_manager.LightRAGManager", return_value=mock_db), \
             patch("pipeline.config_manager.ConfigManager", return_value=mock_conf), \
             patch("pipeline.job_orchestrator.JobOrchestrator", return_value=mock_orch):
            import api as api_module
            importlib.reload(api_module)

        QueryRequest = api_module.QueryRequest
        req = QueryRequest(code_diff="some diff")
        self.assertIsNone(req.categories)

    def test_query_request_accepts_categories_list(self):
        import importlib
        mock_db = MagicMock()
        mock_conf = MagicMock()
        mock_orch = MagicMock()

        with patch("db.lightrag_manager.LightRAGManager", return_value=mock_db), \
             patch("pipeline.config_manager.ConfigManager", return_value=mock_conf), \
             patch("pipeline.job_orchestrator.JobOrchestrator", return_value=mock_orch):
            import api as api_module
            importlib.reload(api_module)

        QueryRequest = api_module.QueryRequest
        req = QueryRequest(code_diff="some diff", categories=["security", "testing"])
        self.assertEqual(req.categories, ["security", "testing"])


if __name__ == "__main__":
    unittest.main()
