"""JobOrchestrator must enqueue when SystemStatus is not ready and drain
on the on-ready callback."""
import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestOrchestratorQueueing(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from pipeline.job_orchestrator import JobOrchestrator
        from system_status import SystemStatus
        self.status = SystemStatus()
        self.status.attach_loop(asyncio.get_event_loop())
        self.db = MagicMock()
        self.orchestrator = JobOrchestrator(self.db, system_status=self.status)

    async def test_trigger_when_ready_runs_immediately(self):
        self.status.update(state="ready")

        with patch.object(self.orchestrator, "_execute_distillation",
                          new=AsyncMock(return_value=None)):
            job_id = self.orchestrator.trigger_job(
                {"repo": "acme/app"}, {"github_token": ""}
            )
            # Allow the created task to complete before asserting.
            await asyncio.sleep(0)
        self.assertTrue(job_id.startswith("job-"))
        self.assertEqual(self.status.snapshot()["queued_jobs"], [])

    async def test_trigger_when_not_ready_enqueues(self):
        self.status.update(state="downloading")
        job_id = self.orchestrator.trigger_job(
            {"repo": "acme/app"}, {"github_token": ""}
        )
        snap = self.status.snapshot()
        self.assertEqual(len(snap["queued_jobs"]), 1)
        self.assertEqual(snap["queued_jobs"][0]["job_id"], job_id)
        self.assertEqual(
            self.orchestrator.active_jobs[job_id]["status"],
            "Queued — waiting for embedding model",
        )

    async def test_state_flip_to_ready_drains_queue(self):
        self.status.update(state="downloading")
        job_id = self.orchestrator.trigger_job(
            {"repo": "acme/app"}, {"github_token": ""}
        )

        executed = []

        async def fake_execute(jid, payload, config):
            executed.append((jid, payload, config))

        with patch.object(self.orchestrator, "_execute_distillation", side_effect=fake_execute):
            self.status.update(state="ready")
            # Give the call_soon_threadsafe + create_task a chance to run.
            await asyncio.sleep(0.05)

        self.assertEqual([e[0] for e in executed], [job_id])

    async def test_request_cancel_removes_queued_job(self):
        self.status.update(state="downloading")
        job_id = self.orchestrator.trigger_job(
            {"repo": "acme/app"}, {"github_token": ""}
        )
        self.assertTrue(self.orchestrator.request_cancel(job_id))
        self.assertEqual(self.status.snapshot()["queued_jobs"], [])
        self.assertEqual(self.orchestrator.active_jobs[job_id]["status"],
                         "Cancelled (was queued)")


if __name__ == "__main__":
    unittest.main()
