"""Tests that capture_hf_progress installs a tqdm subclass that aggregates
byte progress across all live instances and pushes into SystemStatus."""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestProgressCapturingTqdm(unittest.TestCase):
    def setUp(self):
        from system_status import SystemStatus
        self.status = SystemStatus()

    def test_single_instance_pushes_aggregate_progress(self):
        from embedding_loader import _ProgressCapturingTqdm
        with patch("embedding_loader.SYSTEM_STATUS", self.status):
            t = _ProgressCapturingTqdm(total=1000)
            t.update(250)
            snap = self.status.snapshot()
            self.assertEqual(snap["embedding"]["bytes_downloaded"], 250)
            self.assertEqual(snap["embedding"]["bytes_total"], 1000)
            t.close()

    def test_multiple_instances_aggregate_across_them(self):
        from embedding_loader import _ProgressCapturingTqdm
        with patch("embedding_loader.SYSTEM_STATUS", self.status):
            a = _ProgressCapturingTqdm(total=1000)
            b = _ProgressCapturingTqdm(total=2000)
            a.update(500)
            b.update(800)
            snap = self.status.snapshot()
            self.assertEqual(snap["embedding"]["bytes_downloaded"], 1300)
            self.assertEqual(snap["embedding"]["bytes_total"], 3000)
            a.close()
            b.close()

    def test_close_removes_instance_from_aggregate(self):
        from embedding_loader import _ProgressCapturingTqdm
        with patch("embedding_loader.SYSTEM_STATUS", self.status):
            a = _ProgressCapturingTqdm(total=1000)
            b = _ProgressCapturingTqdm(total=2000)
            a.update(500)
            b.update(800)
            a.close()
            snap = self.status.snapshot()
            # Throttle may delay propagation, so call a no-op on b to flush.
            b.update(0)
            snap = self.status.snapshot()
            self.assertEqual(snap["embedding"]["bytes_downloaded"], 800)
            self.assertEqual(snap["embedding"]["bytes_total"], 2000)
            b.close()


class TestCaptureHfProgress(unittest.TestCase):
    # Use sys.modules to reach the actual tqdm submodule. We deliberately don't
    # use `import huggingface_hub.utils.tqdm as alias` because huggingface_hub's
    # __init__ re-exports the class as the package attribute, so that import
    # form binds to the *class*, not the module — and we need to assert against
    # the module attribute the loader patches.
    def _hf_tqdm_module(self):
        import sys
        import importlib
        importlib.import_module("huggingface_hub.utils.tqdm")
        return sys.modules["huggingface_hub.utils.tqdm"]

    def test_context_manager_restores_originals_on_exit(self):
        from embedding_loader import capture_hf_progress

        m = self._hf_tqdm_module()
        original = m.tqdm
        with capture_hf_progress():
            self.assertIsNot(m.tqdm, original)
        self.assertIs(m.tqdm, original)

    def test_context_manager_restores_originals_on_exception(self):
        from embedding_loader import capture_hf_progress

        m = self._hf_tqdm_module()
        original = m.tqdm
        try:
            with capture_hf_progress():
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        self.assertIs(m.tqdm, original)


if __name__ == "__main__":
    unittest.main()
