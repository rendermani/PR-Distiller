"""Tests that LargeLLMExtractor and SemanticFuser accept config via the
constructor and do not leak across concurrent jobs through process-wide env vars.
"""
import asyncio
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


class TestExtractorConstructorInjection(unittest.TestCase):
    def test_constructor_args_override_env_vars(self):
        """When model/api_base/api_key are passed to ctor, env values must be ignored."""
        from llm.extractor import LargeLLMExtractor

        with patch.dict(os.environ, {
            "EXTRACTOR_MODEL": "stale-env-model",
            "EXTRACTOR_API_BASE": "http://stale-env-base",
            "EXTRACTOR_API_KEY": "stale-env-key",
        }):
            extractor = LargeLLMExtractor(
                MagicMock(),
                model="ctor-model",
                api_base="http://ctor-base",
                api_key="ctor-key",
            )

        self.assertEqual(extractor.model, "ctor-model")
        self.assertEqual(extractor.api_base, "http://ctor-base")
        self.assertEqual(extractor.api_key, "ctor-key")

    def test_two_extractors_with_different_ctor_args_do_not_leak(self):
        """Concurrent extractors with different configs must each use their own."""
        from llm.extractor import LargeLLMExtractor

        a = LargeLLMExtractor(MagicMock(), model="model-A", api_base="http://a", api_key="key-a")
        b = LargeLLMExtractor(MagicMock(), model="model-B", api_base="http://b", api_key="key-b")
        self.assertEqual(a.model, "model-A")
        self.assertEqual(b.model, "model-B")
        self.assertEqual(a.api_key, "key-a")
        self.assertEqual(b.api_key, "key-b")

    def test_classify_uses_ctor_model_not_env(self):
        """Async classify must call the LLM with the constructor-supplied model."""
        from llm.extractor import LargeLLMExtractor

        with patch.dict(os.environ, {"EXTRACTOR_MODEL": "stale-env-model"}):
            extractor = LargeLLMExtractor(
                MagicMock(), model="ctor-model", api_base="x", api_key="y",
            )
            with patch("llm.extractor.acompletion", new=MagicMock()) as mock_acompletion:
                async def fake(*a, **kw):
                    return _completion_response("EXTRACT")
                mock_acompletion.side_effect = fake
                asyncio.run(extractor._async_classify("c"))

        self.assertEqual(mock_acompletion.call_args.kwargs["model"], "ctor-model")


class TestSemanticFuserConstructorInjection(unittest.TestCase):
    def test_constructor_args_override_env_vars(self):
        from pipeline.semantic_fuser import SemanticFuser

        with patch.dict(os.environ, {
            "EXTRACTOR_MODEL": "stale-env-model",
            "EXTRACTOR_API_BASE": "http://stale-env-base",
            "EXTRACTOR_API_KEY": "stale-env-key",
        }):
            fuser = SemanticFuser(
                MagicMock(),
                model="ctor-model",
                api_base="http://ctor-base",
                api_key="ctor-key",
            )
        self.assertEqual(fuser.model, "ctor-model")
        self.assertEqual(fuser.api_base, "http://ctor-base")
        self.assertEqual(fuser.api_key, "ctor-key")


if __name__ == "__main__":
    unittest.main()
