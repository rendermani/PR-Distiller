"""
Unit tests for pipeline/dev_cache.py.

All filesystem operations are redirected to a pytest tmp_path so the tests
are fully isolated and leave no artefacts in the source tree.

The module exposes CACHE_DIR as a module-level variable; each test patches it
to a temporary directory via unittest.mock.patch.

Run with:
    cd /home/mani/Projects/PR-Analysis/backend
    ./venv/bin/python -m pytest tests/test_dev_cache.py -v --tb=short
"""
import sys
import os
import json
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pipeline.dev_cache as dev_cache

REPO = "acme/my-repo"
SAFE_REPO_NAME = "acme__my-repo"


def _patch_cache_dir(tmp_path):
    """Return a context manager that redirects dev_cache.CACHE_DIR to tmp_path."""
    return patch.object(dev_cache, "CACHE_DIR", str(tmp_path))


class TestCachePath(unittest.TestCase):
    def test_cache_path_replaces_slash_with_double_underscore(self):
        with patch.object(dev_cache, "CACHE_DIR", "/some/dir"):
            path = dev_cache.cache_path("owner/repo")
        self.assertTrue(path.endswith("owner__repo.json"))

    def test_cache_path_uses_cache_dir_as_prefix(self):
        with patch.object(dev_cache, "CACHE_DIR", "/my/cache"):
            path = dev_cache.cache_path("foo/bar")
        self.assertTrue(path.startswith("/my/cache"))


class TestHasCache(unittest.TestCase):
    def test_has_cache_returns_false_when_file_absent(self, tmp_path=None):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            with patch.object(dev_cache, "CACHE_DIR", td):
                self.assertFalse(dev_cache.has_cache(REPO))

    def test_has_cache_returns_true_after_save(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            with patch.object(dev_cache, "CACHE_DIR", td):
                dev_cache.save_crawl(REPO, [("comment", "diff")])
                self.assertTrue(dev_cache.has_cache(REPO))


class TestSaveAndLoadCrawl(unittest.TestCase):
    def test_save_creates_json_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            with patch.object(dev_cache, "CACHE_DIR", td):
                dev_cache.save_crawl(REPO, [("comment A", "diff A")])
                expected_path = dev_cache.cache_path(REPO)
                self.assertTrue(os.path.exists(expected_path))

    def test_load_returns_same_tuples_that_were_saved(self):
        import tempfile
        tuples = [("avoid bare except", "- except: pass"), ("use type hints", "+ def fn(x: int)")]
        with tempfile.TemporaryDirectory() as td:
            with patch.object(dev_cache, "CACHE_DIR", td):
                dev_cache.save_crawl(REPO, tuples)
                loaded = dev_cache.load_crawl(REPO)
        self.assertEqual(loaded, tuples)

    def test_load_returns_empty_list_when_no_cache_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            with patch.object(dev_cache, "CACHE_DIR", td):
                result = dev_cache.load_crawl(REPO)
        self.assertEqual(result, [])

    def test_save_appends_new_tuples_to_existing_cache(self):
        import tempfile
        first_batch = [("comment 1", "diff 1")]
        second_batch = [("comment 2", "diff 2")]
        with tempfile.TemporaryDirectory() as td:
            with patch.object(dev_cache, "CACHE_DIR", td):
                dev_cache.save_crawl(REPO, first_batch)
                dev_cache.save_crawl(REPO, second_batch)
                loaded = dev_cache.load_crawl(REPO)
        self.assertEqual(len(loaded), 2)
        self.assertIn(("comment 1", "diff 1"), loaded)
        self.assertIn(("comment 2", "diff 2"), loaded)

    def test_save_deduplicates_tuples_across_incremental_crawls(self):
        import tempfile
        tuples = [("same comment", "same diff")]
        with tempfile.TemporaryDirectory() as td:
            with patch.object(dev_cache, "CACHE_DIR", td):
                dev_cache.save_crawl(REPO, tuples)
                dev_cache.save_crawl(REPO, tuples)  # Same batch again.
                loaded = dev_cache.load_crawl(REPO)
        self.assertEqual(loaded, tuples)

    def test_load_returns_empty_list_on_corrupt_json(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            with patch.object(dev_cache, "CACHE_DIR", td):
                corrupt_path = dev_cache.cache_path(REPO)
                os.makedirs(td, exist_ok=True)
                with open(corrupt_path, "w") as f:
                    f.write("{not valid json")
                result = dev_cache.load_crawl(REPO)
        self.assertEqual(result, [])


class TestCacheInfo(unittest.TestCase):
    def test_cache_info_returns_none_when_no_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            with patch.object(dev_cache, "CACHE_DIR", td):
                info = dev_cache.cache_info(REPO)
        self.assertIsNone(info)

    def test_cache_info_returns_correct_count_and_repo(self):
        import tempfile
        tuples = [("c1", "d1"), ("c2", "d2"), ("c3", "d3")]
        with tempfile.TemporaryDirectory() as td:
            with patch.object(dev_cache, "CACHE_DIR", td):
                dev_cache.save_crawl(REPO, tuples)
                info = dev_cache.cache_info(REPO)
        self.assertEqual(info["repo"], REPO)
        self.assertEqual(info["count"], 3)

    def test_cache_info_includes_updated_at_timestamp(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            with patch.object(dev_cache, "CACHE_DIR", td):
                dev_cache.save_crawl(REPO, [("c", "d")])
                info = dev_cache.cache_info(REPO)
        self.assertIn("updated_at", info)
        self.assertIsNotNone(info["updated_at"])


if __name__ == "__main__":
    unittest.main()
