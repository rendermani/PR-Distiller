"""Integration tests for the multi-model orchestrator.

Mocks LargeLLMExtractor + the crawler so we only test the loop & wiring.
"""
import asyncio
import os
import sys
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _make_config(model_ids):
    """Build a multi-model config dict for the given model id list."""
    models = [
        {
            "id": mid,
            "label": mid.upper(),
            "model": f"ollama/{mid}:test",
            "api_base": "http://test:11434/v1",
            "api_key_override": "",
            "enabled": True,
        }
        for mid in model_ids
    ]
    return {
        "llm_models": models,
        "llm_models_active": model_ids,
        "provider_api_keys": {},
        "github_token": "",
    }


def _make_payload():
    return {"repo": "tucowsinc/tdp-apis", "months": 1, "use_cache": True}


class TestMultiModelOrchestrator(unittest.IsolatedAsyncioTestCase):

    @patch("pipeline.job_orchestrator.acompletion", new_callable=AsyncMock)
    @patch("pipeline.job_orchestrator.dev_cache")
    @patch("pipeline.job_orchestrator.LargeLLMExtractor")
    async def test_runs_one_pass_per_active_model(
        self, mock_extractor_cls, mock_cache, mock_preflight
    ):
        from pipeline.job_orchestrator import JobOrchestrator

        mock_cache.has_cache.return_value = True
        mock_cache.cache_info.return_value = {
            "count": 3,
            "updated_at": "2026-01-01T00:00:00Z",
        }
        mock_cache.load_crawl.return_value = [("comment-1", "diff-1")] * 3

        instances = []

        def make_mock(*args, **kwargs):
            inst = MagicMock()
            inst.batch_extract = AsyncMock(return_value=[{"id": "r1"}, {"id": "r2"}])
            instances.append((inst, kwargs))
            return inst

        mock_extractor_cls.side_effect = make_mock

        db = MagicMock()
        db.delete_repo_rules = MagicMock(return_value=0)
        orchestrator = JobOrchestrator(db)
        config = _make_config(["qwen", "gemma"])
        job_id = orchestrator.trigger_job(_make_payload(), config)

        # Gather all pending _execute_distillation tasks.
        pending = [
            t
            for t in asyncio.all_tasks()
            if not t.done() and t.get_coro().__name__ == "_execute_distillation"
        ]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        self.assertEqual(len(instances), 2, "Extractor should be instantiated once per model")
        ids_used = [kwargs.get("model_id") for _, kwargs in instances]
        self.assertEqual(ids_used, ["qwen", "gemma"])
        labels_used = [kwargs.get("model_label") for _, kwargs in instances]
        self.assertEqual(labels_used, ["QWEN", "GEMMA"])

    @patch("pipeline.job_orchestrator.acompletion", new_callable=AsyncMock)
    @patch("pipeline.job_orchestrator.dev_cache")
    @patch("pipeline.job_orchestrator.LargeLLMExtractor")
    async def test_failed_preflight_does_not_abort_remaining_models(
        self, mock_extractor_cls, mock_cache, mock_preflight
    ):
        from pipeline.job_orchestrator import JobOrchestrator

        mock_cache.has_cache.return_value = True
        mock_cache.cache_info.return_value = {
            "count": 3,
            "updated_at": "2026-01-01T00:00:00Z",
        }
        mock_cache.load_crawl.return_value = [("c", "d")] * 3

        # First preflight raises; second succeeds.
        mock_preflight.side_effect = [Exception("LLM unreachable"), MagicMock()]

        successful = MagicMock()
        successful.batch_extract = AsyncMock(return_value=[{"id": "r1"}])
        mock_extractor_cls.return_value = successful

        db = MagicMock()
        db.delete_repo_rules = MagicMock(return_value=0)
        orchestrator = JobOrchestrator(db)
        config = _make_config(["bad-model", "good-model"])
        job_id = orchestrator.trigger_job(_make_payload(), config)
        pending = [
            t
            for t in asyncio.all_tasks()
            if not t.done() and t.get_coro().__name__ == "_execute_distillation"
        ]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        status = orchestrator.get_status(job_id)
        self.assertEqual(status.get("progress"), 100)
        per_model = status.get("per_model_results", {})
        self.assertIn("bad-model", per_model)
        self.assertIn("good-model", per_model)
        self.assertIn("failed_reason", per_model["bad-model"])
        self.assertNotIn("failed_reason", per_model["good-model"])

    @patch("pipeline.job_orchestrator.acompletion", new_callable=AsyncMock)
    @patch("pipeline.job_orchestrator.dev_cache")
    @patch("pipeline.job_orchestrator.LargeLLMExtractor")
    async def test_empty_active_ids_raises_and_job_fails(
        self, mock_extractor_cls, mock_cache, mock_preflight
    ):
        from pipeline.job_orchestrator import JobOrchestrator

        mock_cache.has_cache.return_value = True
        mock_cache.cache_info.return_value = {
            "count": 1,
            "updated_at": "2026-01-01T00:00:00Z",
        }
        mock_cache.load_crawl.return_value = [("c", "d")]

        db = MagicMock()
        db.delete_repo_rules = MagicMock(return_value=0)
        orchestrator = JobOrchestrator(db)
        # Config with no active model ids.
        config = {
            "llm_models": [],
            "llm_models_active": [],
            "provider_api_keys": {},
            "github_token": "",
        }
        job_id = orchestrator.trigger_job(_make_payload(), config)
        pending = [
            t
            for t in asyncio.all_tasks()
            if not t.done() and t.get_coro().__name__ == "_execute_distillation"
        ]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        status = orchestrator.get_status(job_id)
        self.assertEqual(status.get("progress"), -1)
        self.assertIn("FAILED", status.get("status", ""))

    @patch("pipeline.job_orchestrator.acompletion", new_callable=AsyncMock)
    @patch("pipeline.job_orchestrator.dev_cache")
    @patch("pipeline.job_orchestrator.LargeLLMExtractor")
    async def test_per_model_results_recorded_in_job_status(
        self, mock_extractor_cls, mock_cache, mock_preflight
    ):
        from pipeline.job_orchestrator import JobOrchestrator

        mock_cache.has_cache.return_value = True
        mock_cache.cache_info.return_value = {
            "count": 2,
            "updated_at": "2026-01-01T00:00:00Z",
        }
        mock_cache.load_crawl.return_value = [("c", "d")] * 2

        inst = MagicMock()
        inst.batch_extract = AsyncMock(return_value=[{"id": "x"}])
        mock_extractor_cls.return_value = inst

        db = MagicMock()
        db.delete_repo_rules = MagicMock(return_value=0)
        orchestrator = JobOrchestrator(db)
        config = _make_config(["model-a"])
        job_id = orchestrator.trigger_job(_make_payload(), config)
        pending = [
            t
            for t in asyncio.all_tasks()
            if not t.done() and t.get_coro().__name__ == "_execute_distillation"
        ]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        status = orchestrator.get_status(job_id)
        per_model = status.get("per_model_results", {})
        self.assertIn("model-a", per_model)
        self.assertEqual(per_model["model-a"]["extracted"], 1)
        self.assertEqual(per_model["model-a"]["label"], "MODEL-A")


class TestFuserProvenance(unittest.TestCase):
    """Provenance fields on rules: extracted_by_model + merged_with_models."""

    def test_first_insertion_preserves_extracted_by_model(self):
        from pipeline.semantic_fuser import SemanticFuser

        db = MagicMock()
        db.is_blocked.return_value = False
        db.find_similar_rule.return_value = (None, None, None)
        db.store_rule = MagicMock(return_value="stored-id")

        fuser = SemanticFuser(db, model="m", api_base="http://x", api_key="k")
        rule = {
            "rule_id": "r1",
            "content": {
                "title": "t",
                "description": "d",
                "enforcement_prompt": "e",
            },
            "metadata": {
                "repo": "owner/repo",
                "extracted_by_model": "qwen-local",
            },
        }
        fuser.process_and_fuse(rule)

        # No fusion happened; store_rule received the original rule unchanged.
        stored = db.store_rule.call_args[0][0]
        self.assertEqual(stored["metadata"]["extracted_by_model"], "qwen-local")

    def test_collision_appends_to_merged_with_models(self):
        from pipeline.semantic_fuser import SemanticFuser

        db = MagicMock()
        db.is_blocked.return_value = False
        existing_meta = {
            "repo": "owner/repo",
            "extracted_by_model": "qwen-local",
            "merged_with_models": "",  # comma-joined string (ChromaDB convention)
            "occurrence_count": 1,
        }
        db.find_similar_rule.return_value = ("existing-id", "Rule: foo", existing_meta)
        db.store_rule = MagicMock(return_value="merged-id")

        with patch("pipeline.semantic_fuser.completion") as mock_completion:
            mock_completion.return_value.choices = [MagicMock()]
            mock_completion.return_value.choices[0].message.content = (
                '{"rule_id": "r-merged", "category": "correctness", '
                '"confidence": 0.9, "content": {"title": "t", "description": "d", '
                '"enforcement_prompt": "e"}}'
            )

            fuser = SemanticFuser(db, model="m", api_base="http://x", api_key="k")
            rule = {
                "rule_id": "r2",
                "content": {
                    "title": "t",
                    "description": "d",
                    "enforcement_prompt": "e",
                },
                "metadata": {
                    "repo": "owner/repo",
                    "extracted_by_model": "gemma-local",
                },
            }
            fuser.process_and_fuse(rule)

        stored = db.store_rule.call_args[0][0]
        lineage = stored["metadata"]["merged_with_models"]
        # Stored as a comma-joined string (ChromaDB metadata convention).
        self.assertIn("gemma-local", lineage)
        self.assertIn("qwen-local", lineage)
        # First extractor is canonical.
        self.assertEqual(stored["metadata"]["extracted_by_model"], "qwen-local")

    def test_collision_with_pre_task4_rule_uses_new_origin_as_fallback(self):
        """If the existing rule has no extracted_by_model, new origin becomes canonical."""
        from pipeline.semantic_fuser import SemanticFuser

        db = MagicMock()
        db.is_blocked.return_value = False
        # Existing rule has no provenance (pre-Task-4 rule).
        existing_meta = {
            "repo": "owner/repo",
            "occurrence_count": 1,
        }
        db.find_similar_rule.return_value = ("old-id", "Rule: bar", existing_meta)
        db.store_rule = MagicMock(return_value="new-id")

        with patch("pipeline.semantic_fuser.completion") as mock_completion:
            mock_completion.return_value.choices = [MagicMock()]
            mock_completion.return_value.choices[0].message.content = (
                '{"rule_id": "r-merged", "category": "correctness", '
                '"confidence": 0.8, "content": {"title": "t2", "description": "d2", '
                '"enforcement_prompt": "e2"}}'
            )

            fuser = SemanticFuser(db, model="m", api_base="http://x", api_key="k")
            rule = {
                "rule_id": "r-new",
                "content": {
                    "title": "t2",
                    "description": "d2",
                    "enforcement_prompt": "e2",
                },
                "metadata": {
                    "repo": "owner/repo",
                    "extracted_by_model": "gemma-local",
                },
            }
            fuser.process_and_fuse(rule)

        stored = db.store_rule.call_args[0][0]
        # Falls back to new_origin since existing has none.
        self.assertEqual(stored["metadata"]["extracted_by_model"], "gemma-local")

    def test_extractor_stamps_model_id_and_label(self):
        """LargeLLMExtractor must stamp extracted_by_model and extracted_by_label on rule."""
        from llm.extractor import LargeLLMExtractor

        db = MagicMock()
        fuser_mock = MagicMock()

        with patch("llm.extractor.SemanticFuser", return_value=fuser_mock), \
             patch("llm.extractor.completion") as mock_completion:

            mock_completion.return_value.choices = [MagicMock()]
            mock_completion.return_value.choices[0].message.content = (
                '{"rule_id": "r1", "confidence": 0.9, "category": "security", '
                '"metadata": {}, "content": {"title": "t", '
                '"description": "d", "enforcement_prompt": "e"}}'
            )
            mock_completion.return_value.choices[0].message.reasoning_content = None
            mock_completion.return_value.choices[0].message.reasoning = None

            extractor = LargeLLMExtractor(
                db,
                model="ollama/qwen:test",
                api_base="http://localhost:11434",
                api_key="unused",
                model_id="qwen-local",
                model_label="Qwen Local",
            )
            result = extractor.extract_rule("do not use eval", "diff here", "owner/repo")

        self.assertIsNotNone(result)
        self.assertEqual(result["metadata"]["extracted_by_model"], "qwen-local")
        self.assertEqual(result["metadata"]["extracted_by_label"], "Qwen Local")

    def test_extractor_without_model_id_does_not_stamp(self):
        """Extractor with no model_id must NOT add extracted_by_model to rule metadata."""
        from llm.extractor import LargeLLMExtractor

        db = MagicMock()
        fuser_mock = MagicMock()

        with patch("llm.extractor.SemanticFuser", return_value=fuser_mock), \
             patch("llm.extractor.completion") as mock_completion:

            mock_completion.return_value.choices = [MagicMock()]
            mock_completion.return_value.choices[0].message.content = (
                '{"rule_id": "r2", "confidence": 0.85, "category": "correctness", '
                '"metadata": {}, "content": {"title": "t2", '
                '"description": "d2", "enforcement_prompt": "e2"}}'
            )
            mock_completion.return_value.choices[0].message.reasoning_content = None
            mock_completion.return_value.choices[0].message.reasoning = None

            extractor = LargeLLMExtractor(
                db,
                model="ollama/some:model",
                api_base="http://localhost:11434",
                api_key="unused",
                # no model_id / model_label
            )
            result = extractor.extract_rule("use parameterized queries", "diff", "owner/repo")

        self.assertIsNotNone(result)
        self.assertNotIn("extracted_by_model", result["metadata"])
        self.assertNotIn("extracted_by_label", result["metadata"])


if __name__ == "__main__":
    unittest.main()
