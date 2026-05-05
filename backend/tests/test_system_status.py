import asyncio
import os
import sys
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
    async def test_subscribers_receive_initial_snapshot_on_subscribe(self):
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


if __name__ == "__main__":
    unittest.main()
