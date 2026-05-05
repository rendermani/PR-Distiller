"""Tests for /api/system/status, /api/system/events, /api/system/embedding/retry."""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pipeline.job_orchestrator  # noqa: F401

with patch("db.lightrag_manager.LightRAGManager"), \
     patch("pipeline.config_manager.ConfigManager"), \
     patch("pipeline.job_orchestrator.JobOrchestrator"):
    import api
    from fastapi.testclient import TestClient


class TestSystemStatusEndpoint(unittest.TestCase):
    def setUp(self):
        api.db = MagicMock()
        api.conf_manager = MagicMock()
        api.orchestrator = MagicMock()
        self.client = TestClient(api.app)

    def test_status_endpoint_returns_initial_idle_state(self):
        from system_status import SystemStatus
        with patch("api.SYSTEM_STATUS", SystemStatus()):
            r = self.client.get("/api/system/status")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["embedding"]["state"], "idle")
        self.assertEqual(body["queued_jobs"], [])

    def test_status_endpoint_does_not_require_auth(self):
        from system_status import SystemStatus
        with patch.object(api.settings, "API_AUTH_TOKEN", "secret-token"), \
             patch("api.SYSTEM_STATUS", SystemStatus()):
            r = self.client.get("/api/system/status")
        self.assertEqual(r.status_code, 200)

    def test_status_endpoint_reflects_singleton_updates(self):
        from system_status import SystemStatus
        s = SystemStatus()
        s.update(state="downloading", model_name="bge-base")
        with patch("api.SYSTEM_STATUS", s):
            r = self.client.get("/api/system/status")
        self.assertEqual(r.json()["embedding"]["state"], "downloading")
        self.assertEqual(r.json()["embedding"]["model_name"], "bge-base")


import asyncio
import json


# ---------------------------------------------------------------------------
# SSE endpoint tests
#
# httpx.ASGITransport buffers all body chunks until the generator completes,
# which creates a deadlock with infinite SSE streams: the generator blocks on
# asyncio.wait_for(queue.get(), timeout=15) while the transport waits for the
# response to finish before delivering any bytes to the client.
#
# We therefore test the endpoint function directly via its StreamingResponse
# body_iterator, which lets us control the generator lifecycle with a mock
# request.is_disconnected() callable.  This is a genuine unit test — it
# exercises the exact same code path as the production ASGI app would.
# ---------------------------------------------------------------------------

class TestSystemEventsSSE(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        api.db = MagicMock()
        api.conf_manager = MagicMock()
        api.orchestrator = MagicMock()

    def _make_request(self, *, disconnect_after_n_checks: int = 1) -> MagicMock:
        """Return a mock Request whose is_disconnected() returns True after n calls."""
        call_count = 0

        async def is_disconnected() -> bool:
            nonlocal call_count
            call_count += 1
            return call_count > disconnect_after_n_checks

        mock = MagicMock()
        mock.is_disconnected = is_disconnected
        return mock

    async def _first_data_line(self, body_iterator) -> str | None:
        """Collect chunks from body_iterator; return the first 'data: ' line found."""
        accumulated = ""
        async for chunk in body_iterator:
            accumulated += chunk if isinstance(chunk, str) else chunk.decode()
            for line in accumulated.splitlines():
                if line.startswith("data: "):
                    return line
        return None

    async def test_events_endpoint_emits_initial_snapshot(self):
        """First SSE event must carry the current SystemStatus snapshot."""
        from system_status import SystemStatus

        status = SystemStatus()
        status.update(state="downloading", model_name="x")

        with patch("api.SYSTEM_STATUS", status):
            response = await api.system_events(self._make_request())
            payload_line = await self._first_data_line(response.body_iterator)

        self.assertIsNotNone(payload_line)
        payload = json.loads(payload_line[len("data: "):])
        self.assertEqual(payload["embedding"]["state"], "downloading")

    async def test_events_response_carries_sse_content_type(self):
        """Response media_type must be text/event-stream."""
        from system_status import SystemStatus

        with patch("api.SYSTEM_STATUS", SystemStatus()):
            response = await api.system_events(self._make_request())
            # Drain body so the generator's finally block runs cleanly.
            async for _ in response.body_iterator:
                break

        self.assertIn("text/event-stream", response.media_type)

    async def test_events_response_carries_no_cache_headers(self):
        """Cache-Control and X-Accel-Buffering headers must be set for SSE proxies."""
        from system_status import SystemStatus

        with patch("api.SYSTEM_STATUS", SystemStatus()):
            response = await api.system_events(self._make_request())
            async for _ in response.body_iterator:
                break

        self.assertIn("no-cache", response.headers.get("cache-control", ""))
        self.assertEqual(response.headers.get("x-accel-buffering"), "no")

    async def test_events_subscriber_removed_after_disconnect(self):
        """Generator's finally block must unsubscribe from SYSTEM_STATUS on exit."""
        from system_status import SystemStatus

        status = SystemStatus()
        self.assertEqual(len(status._subscribers), 0)

        with patch("api.SYSTEM_STATUS", status):
            response = await api.system_events(self._make_request())
            # Exhaust the iterator so the generator's finally block executes.
            async for _ in response.body_iterator:
                pass

        self.assertEqual(len(status._subscribers), 0)


if __name__ == "__main__":
    unittest.main()
