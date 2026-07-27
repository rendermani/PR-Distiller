"""batch_extract must observe cancellation mid-pass.

The orchestrator only consulted check_cancel() between model passes, and
batch_extract took no cancel hook at all. With a single active model there is no
between-pass checkpoint, so a user who cancelled a 5,000-comment job saw the
request acknowledged while both LLM passes ran to completion — potentially tens
of minutes of unwanted inference.
"""
import asyncio
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_extractor():
    from llm.extractor import LargeLLMExtractor

    with patch("llm.extractor.SemanticFuser"):
        return LargeLLMExtractor(
            db_manager=MagicMock(),
            model="ollama/qwen3:8b",
            api_base="http://localhost:11434",
            api_key="unused",
        )


class TestBatchExtractCancellation(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_during_classify_returns_no_rules(self):
        extractor = _make_extractor()
        payloads = [(f"comment {i}", f"diff {i}") for i in range(20)]

        classify_calls = 0

        async def fake_classify(comment):
            nonlocal classify_calls
            classify_calls += 1
            return True

        extractor._async_classify = fake_classify
        extractor.async_extract_rule = MagicMock()

        result = await extractor.batch_extract(
            payloads, repo="owner/repo", check_cancel=lambda: True
        )

        self.assertEqual(result, [])
        # The load-bearing guarantee: the expensive extract pass never begins.
        # (Classify tasks are all scheduled up front, so with instant mocks they
        # may all complete before the first cancel check is reached; the cancel
        # point that matters is the one before pass 2.)
        extractor.async_extract_rule.assert_not_called()

    async def test_cancel_during_extract_stops_pending_work(self):
        """Pending extract tasks are cancelled rather than run to completion.

        Extraction is slowed here so tasks are genuinely still in flight when
        cancellation is observed — with instant mocks every task would finish
        before the first check and there would be nothing left to cancel.
        """
        extractor = _make_extractor()
        payloads = [(f"comment {i}", f"diff {i}") for i in range(40)]

        async def fake_classify(comment):
            return True

        extract_calls = 0

        async def slow_extract(comment, diff, repo):
            nonlocal extract_calls
            extract_calls += 1
            await asyncio.sleep(0.02)
            return {"title": f"rule {extract_calls}", "metadata": {}}

        extractor._async_classify = fake_classify
        extractor.async_extract_rule = slow_extract

        # Cancel as soon as the extract pass has begun.
        state = {"extracting": False}

        def progress(msg, pct):
            if msg.startswith("Extracting"):
                state["extracting"] = True

        result = await extractor.batch_extract(
            payloads,
            repo="owner/repo",
            progress_callback=progress,
            check_cancel=lambda: state["extracting"],
        )

        # Stopped early: not every keeper was extracted.
        self.assertLess(extract_calls, len(payloads))
        # Whatever did finish is returned rather than discarded.
        self.assertGreater(len(result), 0)

    async def test_no_cancel_hook_runs_to_completion(self):
        """Absent a hook, behaviour is unchanged."""
        extractor = _make_extractor()
        payloads = [(f"comment {i}", f"diff {i}") for i in range(5)]

        async def fake_classify(comment):
            return True

        async def fake_extract(comment, diff, repo):
            return {"title": "rule", "metadata": {}}

        extractor._async_classify = fake_classify
        extractor.async_extract_rule = fake_extract

        result = await extractor.batch_extract(payloads, repo="owner/repo")

        self.assertEqual(len(result), 5)

    async def test_cancel_returning_false_does_not_interrupt(self):
        extractor = _make_extractor()
        payloads = [(f"comment {i}", f"diff {i}") for i in range(5)]

        async def fake_classify(comment):
            return True

        async def fake_extract(comment, diff, repo):
            return {"title": "rule", "metadata": {}}

        extractor._async_classify = fake_classify
        extractor.async_extract_rule = fake_extract

        result = await extractor.batch_extract(
            payloads, repo="owner/repo", check_cancel=lambda: False
        )

        self.assertEqual(len(result), 5)


if __name__ == "__main__":
    unittest.main()
