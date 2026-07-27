"""Persistence paths must derive from settings.DATA_DIR, not from __file__.

Regression guard for the split-store bug: ConfigManager, LightRAGManager and
dev_cache each derived their paths from os.path.dirname(__file__), ignoring
settings.DATA_DIR. Inside the container that coincidentally equalled the
mounted volume (/app/data); run natively it resolved to backend/data, so the
two modes wrote to two unrelated stores and data appeared to vanish.
"""
import importlib
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import settings


class TestConfigManagerHonoursDataDir(unittest.TestCase):
    def test_config_and_key_paths_live_under_data_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(settings, "DATA_DIR", tmp):
                from pipeline.config_manager import ConfigManager

                mgr = ConfigManager()

                self.assertEqual(mgr.config_path, os.path.join(tmp, "config.json"))
                self.assertEqual(mgr.key_path, os.path.join(tmp, ".secret_key"))

    def test_config_file_is_written_into_data_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(settings, "DATA_DIR", tmp):
                from pipeline.config_manager import ConfigManager

                ConfigManager()

                self.assertTrue(os.path.exists(os.path.join(tmp, "config.json")))

    def test_paths_do_not_fall_back_to_package_directory(self):
        """The backend/data directory must not be used when DATA_DIR is set."""
        package_data = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(settings, "DATA_DIR", tmp):
                from pipeline.config_manager import ConfigManager

                mgr = ConfigManager()

                self.assertNotIn(package_data, mgr.config_path)
                self.assertNotIn(package_data, mgr.key_path)


class TestDevCacheHonoursDataDir(unittest.TestCase):
    def test_cache_dir_matches_settings(self):
        from pipeline import dev_cache

        importlib.reload(dev_cache)

        self.assertEqual(dev_cache.CACHE_DIR, settings.DEV_CACHE_DIR)


class TestChromaPathHonoursDataDir(unittest.TestCase):
    def test_vector_db_path_derives_from_data_dir(self):
        """LightRAGManager must build its Chroma path from settings.DATA_DIR.

        Asserted against the module-level constant rather than by constructing
        the manager, which would load a real embedding model.
        """
        from db import lightrag_manager

        self.assertEqual(
            lightrag_manager.VECTOR_DB_PATH,
            os.path.join(settings.DATA_DIR, "code_rag_vectors"),
        )


if __name__ == "__main__":
    unittest.main()
