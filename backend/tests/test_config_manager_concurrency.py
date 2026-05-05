"""Concurrent save_config must not lose updates.

The original code: load → update → write, with no inter-process locking and no
deep-merge of nested fields like crawl_cursors. Two pipeline jobs writing
different repos' cursors at the same time will lose one job's update because
each writes the entire `crawl_cursors` dict verbatim.

Fix: an fcntl-based file lock plus deep-merge of crawl_cursors (the same
overlay-on-existing pattern already used for provider_api_keys).
"""
import os
import shutil
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _make_manager(tmp_dir: str):
    from pipeline.config_manager import ConfigManager
    from cryptography.fernet import Fernet

    manager = ConfigManager.__new__(ConfigManager)
    manager.base_dir = tmp_dir
    manager.config_path = os.path.join(tmp_dir, "config.json")
    manager.key_path = os.path.join(tmp_dir, ".secret_key")
    manager.cipher = Fernet(manager._resolve_key())
    manager._ensure_default_config()
    return manager


class TestConcurrentSaveConfig(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        # Bootstrap the config file once.
        _make_manager(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_concurrent_crawl_cursor_writes_all_survive(self):
        """20 threads each POST a unique cursor — every one must survive."""
        N = 20

        def saver(i: int) -> None:
            manager = _make_manager(self.tmp_dir)
            manager.save_config({"crawl_cursors": {f"repo-{i}": 1000 + i}})

        with ThreadPoolExecutor(max_workers=N) as ex:
            list(ex.map(saver, range(N)))

        final = _make_manager(self.tmp_dir).load_config()
        cursors = final.get("crawl_cursors", {})
        for i in range(N):
            self.assertIn(f"repo-{i}", cursors, f"Lost cursor for repo-{i}")
            self.assertEqual(cursors[f"repo-{i}"], 1000 + i)

    def test_concurrent_provider_api_keys_writes_all_survive(self):
        """provider_api_keys merging must also be atomic under concurrent writes."""
        providers = ["openai", "anthropic", "google", "openrouter", "ollama"]

        def saver(name: str) -> None:
            manager = _make_manager(self.tmp_dir)
            manager.save_config({"provider_api_keys": {name: f"key-for-{name}"}})

        with ThreadPoolExecutor(max_workers=len(providers)) as ex:
            list(ex.map(saver, providers))

        final = _make_manager(self.tmp_dir).load_config()
        keys = final.get("provider_api_keys", {})
        for name in providers:
            self.assertEqual(keys.get(name), f"key-for-{name}",
                             f"Lost provider key for {name}: got {keys}")


if __name__ == "__main__":
    unittest.main()
