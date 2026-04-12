"""
Tests for issue #9: missing repo parameter in run_production_extraction.py
Tests for issue #10: dedup + PII redaction missing from job_orchestrator pipeline
"""
import os
import sys
import asyncio
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Issue #9 — repo extraction from pull_request_url
# ---------------------------------------------------------------------------

class TestRepoSlugExtraction(unittest.TestCase):
    """
    The production script needs to derive a repo slug (e.g. 'langchain-ai/langchain')
    from a comment's pull_request_url field so it can pass repo to extract_rule.
    """

    def test_extracts_repo_slug_from_pull_request_url(self):
        from scripts.run_production_extraction import repo_slug_from_pr_url
        url = "https://api.github.com/repos/langchain-ai/langchain/pulls/36613"
        self.assertEqual(repo_slug_from_pr_url(url), "langchain-ai/langchain")

    def test_extracts_repo_slug_different_owner(self):
        from scripts.run_production_extraction import repo_slug_from_pr_url
        url = "https://api.github.com/repos/encode/starlette/pulls/99"
        self.assertEqual(repo_slug_from_pr_url(url), "encode/starlette")

    def test_returns_empty_string_for_malformed_url(self):
        from scripts.run_production_extraction import repo_slug_from_pr_url
        self.assertEqual(repo_slug_from_pr_url("not-a-url"), "")

    def test_returns_empty_string_for_empty_url(self):
        from scripts.run_production_extraction import repo_slug_from_pr_url
        self.assertEqual(repo_slug_from_pr_url(""), "")


class TestExtractRuleCalledWithRepo(unittest.TestCase):
    """
    Verify that the production script's per-comment processing passes the repo
    slug to extract_rule (the third positional argument).
    """

    def test_extract_rule_receives_repo_argument(self):
        """extract_rule must be called with (comment, ast_slice, repo_slug) — not 2 args."""
        from scripts.run_production_extraction import process_comment

        mock_extractor = MagicMock()
        mock_extractor.extract_rule.return_value = {"rule_id": "r1"}

        mock_redactor = MagicMock()
        mock_redactor.redact_text.return_value = "safe comment"

        mock_dedup = MagicMock()
        mock_dedup.is_duplicate.return_value = False

        mock_intent = MagicMock()
        mock_intent.check_intent.return_value = "CODE"

        mock_slicer = MagicMock()
        mock_slicer.get_node_at_line.return_value = "def foo(): pass"

        comment = {
            "body": "do not use eval()",
            "line": 42,
            "pull_request_url": "https://api.github.com/repos/encode/starlette/pulls/5",
        }
        pr = {"diff": "some diff content"}

        process_comment(
            comment=comment,
            pr=pr,
            extractor=mock_extractor,
            redactor=mock_redactor,
            dedup=mock_dedup,
            intent=mock_intent,
            slicer=mock_slicer,
        )

        mock_extractor.extract_rule.assert_called_once_with(
            "safe comment", "def foo(): pass", "encode/starlette"
        )


# ---------------------------------------------------------------------------
# Issue #10 — dedup + redaction in job_orchestrator before batch_extract
# ---------------------------------------------------------------------------

class TestOrchestratorPreProcessing(unittest.IsolatedAsyncioTestCase):
    """
    Verify that _execute_distillation applies hash dedup and PII redaction
    to crawled tuples before passing them to batch_extract.
    """

    async def test_duplicate_comments_are_dropped_before_extraction(self):
        """
        When the crawler returns two identical comment bodies, only the first
        should reach batch_extract after dedup.
        """
        from pipeline.job_orchestrator import JobOrchestrator
        from db.lightrag_manager import LightRAGManager

        mock_db = MagicMock(spec=LightRAGManager)
        orchestrator = JobOrchestrator(mock_db)

        job_id = "test-job-dedup"
        orchestrator.active_jobs[job_id] = {"status": "init", "progress": 0}
        orchestrator.cancel_flags[job_id] = False

        duplicate_comment = "avoid using eval() — security risk"
        crawled_tuples = [
            (duplicate_comment, "diff hunk A"),
            (duplicate_comment, "diff hunk B"),  # exact duplicate body
        ]

        captured_payloads = []

        async def fake_batch_extract(payloads, repo):
            captured_payloads.extend(payloads)
            return []

        with patch("pipeline.job_orchestrator.crawl_human_rejections", return_value=(crawled_tuples, {})), \
             patch("pipeline.job_orchestrator.LargeLLMExtractor") as MockExtractor:

            mock_extractor_instance = MagicMock()
            mock_extractor_instance.batch_extract = fake_batch_extract
            MockExtractor.return_value = mock_extractor_instance

            await orchestrator._execute_distillation(
                job_id,
                {"repo": "encode/starlette", "months": 1, "use_cache": False},
                {"github_token": "", "llm_api_base": "http://localhost/v1",
                 "llm_api_key": "key", "llm_model": "test-model"},
            )

        # Only one unique comment should have reached batch_extract
        self.assertEqual(len(captured_payloads), 1)
        self.assertEqual(captured_payloads[0][0], duplicate_comment)

    async def test_pii_redaction_applied_before_extraction(self):
        """
        Comment bodies must be passed through SecurityRedactor.redact_text
        before reaching batch_extract.
        """
        from pipeline.job_orchestrator import JobOrchestrator
        from db.lightrag_manager import LightRAGManager

        mock_db = MagicMock(spec=LightRAGManager)
        orchestrator = JobOrchestrator(mock_db)

        job_id = "test-job-redact"
        orchestrator.active_jobs[job_id] = {"status": "init", "progress": 0}
        orchestrator.cancel_flags[job_id] = False

        raw_comment = "contact user@example.com about the eval() bug"
        crawled_tuples = [(raw_comment, "diff hunk")]
        redacted_comment = "contact <EMAIL> about the eval() bug"

        captured_payloads = []

        async def fake_batch_extract(payloads, repo):
            captured_payloads.extend(payloads)
            return []

        mock_redactor = MagicMock()
        mock_redactor.redact_text.return_value = redacted_comment

        with patch("pipeline.job_orchestrator.crawl_human_rejections", return_value=(crawled_tuples, {})), \
             patch("pipeline.job_orchestrator.SecurityRedactor", return_value=mock_redactor), \
             patch("pipeline.job_orchestrator.LargeLLMExtractor") as MockExtractor:

            mock_extractor_instance = MagicMock()
            mock_extractor_instance.batch_extract = fake_batch_extract
            MockExtractor.return_value = mock_extractor_instance

            await orchestrator._execute_distillation(
                job_id,
                {"repo": "encode/starlette", "months": 1, "use_cache": False},
                {"github_token": "", "llm_api_base": "http://localhost/v1",
                 "llm_api_key": "key", "llm_model": "test-model"},
            )

        self.assertEqual(len(captured_payloads), 1)
        self.assertEqual(captured_payloads[0][0], redacted_comment)

    async def test_both_dedup_and_redaction_applied_in_order(self):
        """
        Dedup must run on raw text, redaction on unique comments.
        Two comments: one unique (to be redacted), one duplicate (to be dropped).
        """
        from pipeline.job_orchestrator import JobOrchestrator
        from db.lightrag_manager import LightRAGManager

        mock_db = MagicMock(spec=LightRAGManager)
        orchestrator = JobOrchestrator(mock_db)

        job_id = "test-job-both"
        orchestrator.active_jobs[job_id] = {"status": "init", "progress": 0}
        orchestrator.cancel_flags[job_id] = False

        comment_a = "use ast.literal_eval, not eval()"
        comment_b = "use ast.literal_eval, not eval()"  # duplicate of a
        crawled_tuples = [
            (comment_a, "diff A"),
            (comment_b, "diff B"),
        ]
        redacted_a = "use ast.literal_eval, not eval()_REDACTED"

        captured_payloads = []

        async def fake_batch_extract(payloads, repo):
            captured_payloads.extend(payloads)
            return []

        mock_redactor = MagicMock()
        mock_redactor.redact_text.return_value = redacted_a

        with patch("pipeline.job_orchestrator.crawl_human_rejections", return_value=(crawled_tuples, {})), \
             patch("pipeline.job_orchestrator.SecurityRedactor", return_value=mock_redactor), \
             patch("pipeline.job_orchestrator.LargeLLMExtractor") as MockExtractor:

            mock_extractor_instance = MagicMock()
            mock_extractor_instance.batch_extract = fake_batch_extract
            MockExtractor.return_value = mock_extractor_instance

            await orchestrator._execute_distillation(
                job_id,
                {"repo": "encode/starlette", "months": 1, "use_cache": False},
                {"github_token": "", "llm_api_base": "http://localhost/v1",
                 "llm_api_key": "key", "llm_model": "test-model"},
            )

        # Duplicate dropped → only 1 payload, and it must be the redacted version
        self.assertEqual(len(captured_payloads), 1)
        self.assertEqual(captured_payloads[0][0], redacted_a)
        # Redactor called exactly once (only for the unique comment)
        mock_redactor.redact_text.assert_called_once_with(comment_a)


if __name__ == "__main__":
    unittest.main()
