"""Tests that the Qwen-specific `enable_thinking=False` extra_body is only
sent to Qwen models. Other providers (GPT, Claude, Gemini) reject unknown
extra_body keys and would 4xx.
"""
import asyncio
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _make_extractor(model: str):
    from llm.extractor import LargeLLMExtractor
    extractor = LargeLLMExtractor.__new__(LargeLLMExtractor)
    extractor.model = model
    extractor.api_base = "http://localhost"
    extractor.api_key = "unused"
    extractor.db = MagicMock()
    extractor.fuser = MagicMock()
    return extractor


def _completion_response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


class TestQwenGating(unittest.TestCase):
    def test_extra_body_sent_for_qwen_model(self):
        # _async_classify uses _qwen_no_think (Ollama native think:false)
        # to skip Qwen3 reasoning chain — ~50x faster than the chat-template
        # `enable_thinking=False` route.
        extractor = _make_extractor("ollama/qwen3:8b")
        with patch("llm.extractor.acompletion", new=MagicMock()) as mock_acompletion:
            async def fake(*a, **kw):
                return _completion_response("EXTRACT")
            mock_acompletion.side_effect = fake
            asyncio.run(extractor._async_classify("comment"))
        kwargs = mock_acompletion.call_args.kwargs
        self.assertIn("extra_body", kwargs)
        self.assertEqual(kwargs["extra_body"], {"think": False})

    def test_extra_body_omitted_for_gpt_model(self):
        extractor = _make_extractor("openai/gpt-4o")
        with patch("llm.extractor.acompletion", new=MagicMock()) as mock_acompletion:
            async def fake(*a, **kw):
                return _completion_response("EXTRACT")
            mock_acompletion.side_effect = fake
            asyncio.run(extractor._async_classify("comment"))
        kwargs = mock_acompletion.call_args.kwargs
        self.assertNotIn("extra_body", kwargs)

    def test_extra_body_omitted_for_claude_model(self):
        extractor = _make_extractor("anthropic/claude-3-5-sonnet")
        with patch("llm.extractor.acompletion", new=MagicMock()) as mock_acompletion:
            async def fake(*a, **kw):
                return _completion_response("EXTRACT")
            mock_acompletion.side_effect = fake
            asyncio.run(extractor._async_classify("comment"))
        kwargs = mock_acompletion.call_args.kwargs
        self.assertNotIn("extra_body", kwargs)

    def test_extra_body_omitted_for_gemini_model(self):
        extractor = _make_extractor("gemini/gemini-2.5-flash")
        with patch("llm.extractor.acompletion", new=MagicMock()) as mock_acompletion:
            async def fake(*a, **kw):
                return _completion_response("EXTRACT")
            mock_acompletion.side_effect = fake
            asyncio.run(extractor._async_classify("comment"))
        kwargs = mock_acompletion.call_args.kwargs
        self.assertNotIn("extra_body", kwargs)


if __name__ == "__main__":
    unittest.main()
