"""Tests for api.py lifespan startup hook and background embedding-load thread.

Covers:
- _start_background_embedding_load: thread spawning, SYSTEM_STATUS transitions,
  db.bind() call, and error handling.
- lifespan: attach_loop, ensure_api_token, and starter invocation.
- _embedding_load_starter: delegates to _start_background_embedding_load.

Import strategy: patches LightRAGManager at construction time (same as other
test files) so module import doesn't trigger a real download.
"""
import asyncio
import sys
import os
import threading
import unittest
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pipeline.job_orchestrator  # noqa: F401

with patch("db.lightrag_manager.LightRAGManager"), \
     patch("pipeline.config_manager.ConfigManager"), \
     patch("pipeline.job_orchestrator.JobOrchestrator"):
    import api


class TestStartBackgroundEmbeddingLoad(unittest.TestCase):
    """Unit tests for _start_background_embedding_load."""

    def setUp(self):
        # Fresh proxy and status for every test so tests don't bleed state.
        self.fake_db = MagicMock()
        self.fake_real_db = MagicMock()

        # Patch api.db so _runner calls self.fake_db.bind(...)
        api.db = self.fake_db

    def _run_in_foreground(self, patch_lightrag_return=None, patch_lightrag_raise=None):
        """Run _start_background_embedding_load and wait for the thread to finish.

        Patches LightRAGManager so no real download occurs.
        """
        done = threading.Event()

        original_start = threading.Thread.start

        def capturing_start(thread_self):
            original_target = thread_self._target
            original_args = thread_self._args
            original_kwargs = thread_self._kwargs

            def wrapped():
                try:
                    original_target(*original_args, **original_kwargs)
                finally:
                    done.set()

            thread_self._target = wrapped
            thread_self._args = ()
            thread_self._kwargs = {}
            original_start(thread_self)

        from system_status import SystemStatus
        fresh_status = SystemStatus()

        if patch_lightrag_raise is not None:
            lightrag_side_effect = patch_lightrag_raise
        else:
            real_db_instance = patch_lightrag_return or self.fake_real_db
            lightrag_side_effect = None

        with patch("api.SYSTEM_STATUS", fresh_status), \
             patch("threading.Thread.start", capturing_start), \
             patch("api.LightRAGManager",
                   side_effect=lightrag_side_effect,
                   return_value=(patch_lightrag_return or self.fake_real_db)):
            api._start_background_embedding_load()
            done.wait(timeout=5.0)

        return fresh_status

    def test_spawns_daemon_thread_named_embedding_loader(self):
        """_start_background_embedding_load must spawn a daemon thread."""
        spawned = []
        original_init = threading.Thread.__init__

        def capturing_init(thread_self, *args, **kwargs):
            original_init(thread_self, *args, **kwargs)
            spawned.append(thread_self)

        done = threading.Event()
        original_start = threading.Thread.start

        def capturing_start(thread_self):
            original_target = thread_self._target

            def wrapped():
                try:
                    original_target()
                finally:
                    done.set()

            thread_self._target = wrapped
            original_start(thread_self)

        from system_status import SystemStatus
        with patch("api.SYSTEM_STATUS", SystemStatus()), \
             patch("api.LightRAGManager", return_value=MagicMock()), \
             patch("threading.Thread.__init__", capturing_init), \
             patch("threading.Thread.start", capturing_start):
            api._start_background_embedding_load()
            done.wait(timeout=5.0)

        self.assertTrue(len(spawned) >= 1)
        loader_threads = [t for t in spawned if getattr(t, "name", "") == "embedding-loader"]
        self.assertEqual(len(loader_threads), 1)
        self.assertTrue(loader_threads[0].daemon)

    def test_sets_state_downloading_before_loading(self):
        """State must be 'downloading' before LightRAGManager() is called."""
        states_during_construction = []
        fresh_status_holder = [None]
        done = threading.Event()

        from system_status import SystemStatus
        fresh_status = SystemStatus()
        fresh_status_holder[0] = fresh_status

        original_lightrag_init = MagicMock

        def capturing_lightrag():
            # Called inside _runner — record state at the moment of construction.
            states_during_construction.append(
                fresh_status_holder[0].snapshot()["embedding"]["state"]
            )
            return MagicMock()

        original_start = threading.Thread.start

        def capturing_start(thread_self):
            original_target = thread_self._target

            def wrapped():
                try:
                    original_target()
                finally:
                    done.set()

            thread_self._target = wrapped
            original_start(thread_self)

        with patch("api.SYSTEM_STATUS", fresh_status), \
             patch("api.LightRAGManager", side_effect=capturing_lightrag), \
             patch("threading.Thread.start", capturing_start):
            api._start_background_embedding_load()
            done.wait(timeout=5.0)

        self.assertIn("downloading", states_during_construction)

    def test_binds_proxy_with_real_manager_on_success(self):
        """db.bind() must be called with the LightRAGManager instance on success."""
        real_db_instance = MagicMock(name="real_lightrag")
        done = threading.Event()

        original_start = threading.Thread.start

        def capturing_start(thread_self):
            original_target = thread_self._target

            def wrapped():
                try:
                    original_target()
                finally:
                    done.set()

            thread_self._target = wrapped
            original_start(thread_self)

        from system_status import SystemStatus
        with patch("api.SYSTEM_STATUS", SystemStatus()), \
             patch("api.LightRAGManager", return_value=real_db_instance), \
             patch("threading.Thread.start", capturing_start):
            api._start_background_embedding_load()
            done.wait(timeout=5.0)

        self.fake_db.bind.assert_called_once_with(real_db_instance)

    def test_sets_state_ready_after_successful_load(self):
        """Final state must be 'ready' after a successful load."""
        done = threading.Event()
        original_start = threading.Thread.start

        def capturing_start(thread_self):
            original_target = thread_self._target

            def wrapped():
                try:
                    original_target()
                finally:
                    done.set()

            thread_self._target = wrapped
            original_start(thread_self)

        from system_status import SystemStatus
        fresh_status = SystemStatus()
        with patch("api.SYSTEM_STATUS", fresh_status), \
             patch("api.LightRAGManager", return_value=MagicMock()), \
             patch("threading.Thread.start", capturing_start):
            api._start_background_embedding_load()
            done.wait(timeout=5.0)

        self.assertEqual(fresh_status.snapshot()["embedding"]["state"], "ready")

    def test_sets_state_error_when_lightrag_raises(self):
        """State must be 'error' and error field populated when LightRAGManager() raises."""
        done = threading.Event()
        original_start = threading.Thread.start

        def capturing_start(thread_self):
            original_target = thread_self._target

            def wrapped():
                try:
                    original_target()
                finally:
                    done.set()

            thread_self._target = wrapped
            original_start(thread_self)

        from system_status import SystemStatus
        fresh_status = SystemStatus()
        with patch("api.SYSTEM_STATUS", fresh_status), \
             patch("api.LightRAGManager", side_effect=RuntimeError("disk full")), \
             patch("threading.Thread.start", capturing_start):
            api._start_background_embedding_load()
            done.wait(timeout=5.0)

        snapshot = fresh_status.snapshot()
        self.assertEqual(snapshot["embedding"]["state"], "error")
        self.assertIn("RuntimeError", snapshot["embedding"]["error"])
        self.assertIn("disk full", snapshot["embedding"]["error"])


class TestEmbeddingLoadStarter(unittest.TestCase):
    """_embedding_load_starter must delegate to _start_background_embedding_load."""

    def test_calls_start_background_embedding_load(self):
        with patch("api._start_background_embedding_load") as mock_start:
            api._embedding_load_starter()
        mock_start.assert_called_once_with()


class TestLifespanHook(unittest.IsolatedAsyncioTestCase):
    """Lifespan hook must attach the loop, resolve the token, and start the loader.

    Each test saves and restores settings.API_AUTH_TOKEN so the lifespan's
    assignment does not leak into subsequent tests in the same process.
    """

    def setUp(self):
        import api as _api
        self._original_api_auth_token = _api.settings.API_AUTH_TOKEN

    def tearDown(self):
        import api as _api
        _api.settings.API_AUTH_TOKEN = self._original_api_auth_token

    async def test_lifespan_attaches_running_loop(self):
        """lifespan must call SYSTEM_STATUS.attach_loop with the running event loop."""
        from system_status import SystemStatus
        fresh_status = SystemStatus()
        loop_attached = []

        original_attach = SystemStatus.attach_loop

        def capturing_attach(status_self, loop):
            loop_attached.append(loop)
            original_attach(status_self, loop)

        with patch("api.SYSTEM_STATUS", fresh_status), \
             patch.object(type(fresh_status), "attach_loop", capturing_attach), \
             patch("api._start_background_embedding_load"), \
             patch("api.ensure_api_token", return_value="tok"):
            async with api.lifespan(api.app):
                pass

        self.assertEqual(len(loop_attached), 1)
        self.assertIsInstance(loop_attached[0], asyncio.AbstractEventLoop)

    async def test_lifespan_resolves_api_token(self):
        """lifespan must call ensure_api_token and assign result to settings.API_AUTH_TOKEN."""
        import api as _api
        from system_status import SystemStatus

        with patch("api.SYSTEM_STATUS", SystemStatus()), \
             patch("api._start_background_embedding_load"), \
             patch("api.ensure_api_token", return_value="resolved-token") as mock_ensure:
            async with _api.lifespan(_api.app):
                pass

        mock_ensure.assert_called_once()
        self.assertEqual(_api.settings.API_AUTH_TOKEN, "resolved-token")

    async def test_lifespan_starts_background_embedding_load(self):
        """lifespan must invoke _start_background_embedding_load exactly once."""
        from system_status import SystemStatus

        with patch("api.SYSTEM_STATUS", SystemStatus()), \
             patch("api._start_background_embedding_load") as mock_start, \
             patch("api.ensure_api_token", return_value="tok"):
            async with api.lifespan(api.app):
                pass

        mock_start.assert_called_once_with()


class TestStartBackgroundLoadIdempotency(unittest.TestCase):
    """_start_background_embedding_load must be idempotent while a thread is alive."""

    def setUp(self):
        # Reset loader-thread state before each test so tests are isolated.
        api._loader_thread = None

    def tearDown(self):
        # Clean up any live thread we started so it doesn't bleed into later tests.
        api._loader_thread = None

    def test_second_call_while_thread_alive_is_noop(self):
        """A second call while the loader thread is still alive must not spawn a new thread."""
        import time
        with patch("api._runner", side_effect=lambda: time.sleep(0.5)):
            api._start_background_embedding_load()
            first = api._loader_thread
            self.assertIsNotNone(first)
            api._start_background_embedding_load()
            second = api._loader_thread
            self.assertIs(first, second, "second call must not spawn a new thread")
            first.join(timeout=2.0)

    def test_third_call_after_thread_finished_spawns_new_one(self):
        """A call after the previous loader thread has finished must spawn a fresh thread."""
        with patch("api._runner", new=lambda: None):
            api._start_background_embedding_load()
            first = api._loader_thread
            first.join(timeout=2.0)
            self.assertFalse(first.is_alive())
            api._start_background_embedding_load()
            second = api._loader_thread
            self.assertIsNot(first, second, "after first finished, a new thread should spawn")
            second.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
