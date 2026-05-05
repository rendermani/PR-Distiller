import asyncio
import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestSystemStatusSnapshot(unittest.TestCase):
    def setUp(self):
        from system_status import SystemStatus
        self.status = SystemStatus()

    def test_initial_snapshot_has_idle_embedding_state(self):
        snap = self.status.snapshot()
        self.assertEqual(snap["embedding"]["state"], "idle")
        self.assertEqual(snap["embedding"]["bytes_downloaded"], 0)
        self.assertEqual(snap["embedding"]["bytes_total"], 0)
        self.assertEqual(snap["queued_jobs"], [])

    def test_update_state_is_visible_in_next_snapshot(self):
        self.status.update(state="downloading", model_name="bge-base")
        snap = self.status.snapshot()
        self.assertEqual(snap["embedding"]["state"], "downloading")
        self.assertEqual(snap["embedding"]["model_name"], "bge-base")

    def test_update_download_progress_writes_bytes(self):
        self.status.update_download_progress(bytes_downloaded=1024, bytes_total=4096)
        snap = self.status.snapshot()
        self.assertEqual(snap["embedding"]["bytes_downloaded"], 1024)
        self.assertEqual(snap["embedding"]["bytes_total"], 4096)


class TestSystemStatusSubscribers(unittest.IsolatedAsyncioTestCase):
    async def test_subscribe_returns_queue_without_initial_push(self):
        from system_status import SystemStatus
        status = SystemStatus()
        q = status.subscribe()
        # subscribe() must not push an initial snapshot itself; the SSE
        # endpoint pushes the snapshot once after subscribe(). We just
        # assert subscribe returned a queue.
        self.assertIsInstance(q, asyncio.Queue)
        status.unsubscribe(q)

    async def test_update_fans_out_to_all_subscribers(self):
        from system_status import SystemStatus
        status = SystemStatus()
        q1 = status.subscribe()
        q2 = status.subscribe()
        status.update(state="downloading")
        s1 = await asyncio.wait_for(q1.get(), timeout=1.0)
        s2 = await asyncio.wait_for(q2.get(), timeout=1.0)
        self.assertEqual(s1["embedding"]["state"], "downloading")
        self.assertEqual(s2["embedding"]["state"], "downloading")
        status.unsubscribe(q1)
        status.unsubscribe(q2)

    async def test_unsubscribe_stops_further_updates(self):
        from system_status import SystemStatus
        status = SystemStatus()
        q = status.subscribe()
        status.unsubscribe(q)
        status.update(state="ready")
        # No update should arrive on q.
        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(q.get(), timeout=0.2)

    async def test_safe_put_cross_thread_uses_call_soon_threadsafe(self):
        """update() called from a background thread fans out via call_soon_threadsafe."""
        from system_status import SystemStatus
        status = SystemStatus()
        loop = asyncio.get_running_loop()
        status.attach_loop(loop)
        q = status.subscribe()

        def _worker():
            status.update(state="downloading")

        t = threading.Thread(target=_worker)
        t.start()
        t.join()

        snap = await asyncio.wait_for(q.get(), timeout=1.0)
        self.assertEqual(snap["embedding"]["state"], "downloading")
        status.unsubscribe(q)

    async def test_queuefull_drops_silently(self):
        """Overflow beyond maxsize=64 is dropped without raising an exception."""
        from system_status import SystemStatus
        status = SystemStatus()
        q = status.subscribe()
        # Fire 70 updates without consuming; 6 should be silently dropped.
        for i in range(70):
            status.update(state="downloading", model_name=f"model-{i}")
        self.assertEqual(q.qsize(), 64)
        status.unsubscribe(q)

    async def test_update_download_progress_throttles_rapid_calls(self):
        """Rapid progress updates well under 1% and within 250 ms are throttled."""
        from system_status import SystemStatus
        status = SystemStatus()
        q = status.subscribe()
        # bytes 0..9 out of 1000: each step is 0.1%, well under the 1% threshold.
        # All 10 calls happen within milliseconds (inside 250 ms window).
        # The first call always passes (sentinel _last_progress_pct == -1.0).
        for i in range(10):
            status.update_download_progress(i, 1000)
        self.assertLess(q.qsize(), 10)
        status.unsubscribe(q)

    async def test_on_ready_fires_only_on_idle_to_ready_transition(self):
        """on_ready callback fires exactly once on the idle→ready transition."""
        from system_status import SystemStatus
        status = SystemStatus()
        calls: list[str] = []

        def _on_ready() -> None:
            calls.append("fired")

        # No loop attached — uses synchronous fallback path.
        status.set_on_ready(_on_ready)

        status.update(state="downloading")
        self.assertEqual(calls, [], "callback must not fire on non-ready transition")

        status.update(state="ready")
        self.assertEqual(calls, ["fired"], "callback must fire exactly once on idle→ready")

        status.update(state="ready")
        self.assertEqual(calls, ["fired"], "callback must not fire again on ready→ready")


if __name__ == "__main__":
    unittest.main()
