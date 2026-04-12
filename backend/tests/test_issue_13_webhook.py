"""
Tests for GitHub issue #13 — GitHub webhook for automatic incremental extraction.

Acceptance criteria:
  1. Webhook endpoint accepts PR merge/close events.
  2. Only processes pull_request events with action=closed and merged=true.
  3. Validates X-Hub-Signature-256 header when GITHUB_WEBHOOK_SECRET is set.
  4. Skips validation in dev mode (no secret configured).
  5. Rate-limits: minimum 60 seconds between triggers per repo.
  6. Triggers incremental job via orchestrator with trigger_source="webhook".
  7. Stats endpoint returns trigger counts and last trigger time per repo.
  8. trigger_job records trigger_source in active_jobs metadata.
"""
import hashlib
import hmac
import importlib
import json
import os
import time
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pr_merged_payload(repo_full_name: str = "owner/repo") -> dict:
    return {
        "action": "closed",
        "pull_request": {"merged": True, "number": 42},
        "repository": {"full_name": repo_full_name},
    }


def _make_pr_closed_not_merged_payload(repo_full_name: str = "owner/repo") -> dict:
    return {
        "action": "closed",
        "pull_request": {"merged": False, "number": 7},
        "repository": {"full_name": repo_full_name},
    }


def _sign_payload(body: bytes, secret: str) -> str:
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _load_api_module():
    """Load api module with all heavy dependencies mocked out."""
    mock_db = MagicMock()
    mock_conf = MagicMock()
    mock_conf.load_config.return_value = {
        "github_token": "",
        "llm_model": "test-model",
        "llm_api_base": "",
        "llm_api_key": "",
    }
    mock_orch = MagicMock()
    mock_orch.trigger_job.return_value = "job-111"

    with patch("db.lightrag_manager.LightRAGManager", return_value=mock_db), \
         patch("pipeline.config_manager.ConfigManager", return_value=mock_conf), \
         patch("pipeline.job_orchestrator.JobOrchestrator", return_value=mock_orch):
        import api as api_module
        importlib.reload(api_module)

    return api_module, mock_orch, mock_conf


# ---------------------------------------------------------------------------
# Webhook signature validation (unit — pure function)
# ---------------------------------------------------------------------------

class TestWebhookSignatureValidation(unittest.TestCase):
    """verify_github_signature must accept correct signatures and reject bad ones."""

    def _import_verifier(self):
        api_module, _, _ = _load_api_module()
        return api_module.verify_github_signature

    def test_valid_signature_returns_true(self):
        verify = self._import_verifier()
        body = b'{"action":"closed"}'
        secret = "my-webhook-secret"
        sig = _sign_payload(body, secret)
        self.assertTrue(verify(body, sig, secret))

    def test_tampered_body_returns_false(self):
        verify = self._import_verifier()
        body = b'{"action":"closed"}'
        secret = "my-webhook-secret"
        sig = _sign_payload(body, secret)
        self.assertFalse(verify(b'{"action":"opened"}', sig, secret))

    def test_wrong_secret_returns_false(self):
        verify = self._import_verifier()
        body = b'{"action":"closed"}'
        sig = _sign_payload(body, "correct-secret")
        self.assertFalse(verify(body, sig, "wrong-secret"))

    def test_missing_sha256_prefix_returns_false(self):
        verify = self._import_verifier()
        body = b'{"action":"closed"}'
        secret = "secret"
        raw_hex = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        # No "sha256=" prefix
        self.assertFalse(verify(body, raw_hex, secret))


# ---------------------------------------------------------------------------
# Event filtering (unit — pure function)
# ---------------------------------------------------------------------------

class TestIsMergedPullRequest(unittest.TestCase):
    """is_merged_pull_request must return True only for closed+merged PR events."""

    def _import_filter(self):
        api_module, _, _ = _load_api_module()
        return api_module.is_merged_pull_request

    def test_closed_and_merged_returns_true(self):
        is_merged = self._import_filter()
        self.assertTrue(is_merged(_make_pr_merged_payload()))

    def test_closed_but_not_merged_returns_false(self):
        is_merged = self._import_filter()
        self.assertFalse(is_merged(_make_pr_closed_not_merged_payload()))

    def test_opened_action_returns_false(self):
        is_merged = self._import_filter()
        payload = {
            "action": "opened",
            "pull_request": {"merged": False},
            "repository": {"full_name": "owner/repo"},
        }
        self.assertFalse(is_merged(payload))

    def test_non_pull_request_event_returns_false(self):
        is_merged = self._import_filter()
        payload = {"action": "created", "issue": {"number": 1}}
        self.assertFalse(is_merged(payload))


# ---------------------------------------------------------------------------
# Rate limiter (unit — pure class)
# ---------------------------------------------------------------------------

class TestWebhookRateLimiter(unittest.TestCase):
    """WebhookRateLimiter must enforce minimum 60 seconds between triggers per repo."""

    def _import_limiter(self):
        api_module, _, _ = _load_api_module()
        return api_module.WebhookRateLimiter

    def test_first_trigger_is_allowed(self):
        Limiter = self._import_limiter()
        limiter = Limiter(min_interval_seconds=60)
        self.assertTrue(limiter.is_allowed("owner/repo"))

    def test_second_trigger_within_interval_is_denied(self):
        Limiter = self._import_limiter()
        limiter = Limiter(min_interval_seconds=60)
        limiter.record_trigger("owner/repo")
        self.assertFalse(limiter.is_allowed("owner/repo"))

    def test_trigger_after_interval_elapses_is_allowed(self):
        Limiter = self._import_limiter()
        limiter = Limiter(min_interval_seconds=60)
        # Backdate the last trigger by 61 seconds
        limiter.record_trigger("owner/repo")
        limiter._last_trigger_times["owner/repo"] = time.time() - 61
        self.assertTrue(limiter.is_allowed("owner/repo"))

    def test_different_repos_are_rate_limited_independently(self):
        Limiter = self._import_limiter()
        limiter = Limiter(min_interval_seconds=60)
        limiter.record_trigger("owner/repo-a")
        # repo-b was not triggered, must be allowed
        self.assertTrue(limiter.is_allowed("owner/repo-b"))

    def test_record_trigger_stores_last_trigger_time(self):
        Limiter = self._import_limiter()
        limiter = Limiter(min_interval_seconds=60)
        before = time.time()
        limiter.record_trigger("owner/repo")
        after = time.time()
        recorded = limiter._last_trigger_times["owner/repo"]
        self.assertGreaterEqual(recorded, before)
        self.assertLessEqual(recorded, after)

    def test_stats_returns_trigger_count_and_last_time(self):
        Limiter = self._import_limiter()
        limiter = Limiter(min_interval_seconds=60)
        limiter.record_trigger("owner/repo")
        limiter._last_trigger_times["owner/repo"] = time.time() - 61
        limiter.record_trigger("owner/repo")
        stats = limiter.get_stats()
        self.assertEqual(stats["owner/repo"]["trigger_count"], 2)
        self.assertIn("last_trigger_time", stats["owner/repo"])


# ---------------------------------------------------------------------------
# Webhook HTTP endpoint (integration — via FastAPI TestClient)
# ---------------------------------------------------------------------------

class TestWebhookEndpoint(unittest.TestCase):
    """POST /api/webhooks/github must route merged PRs to the orchestrator."""

    def _make_client(self, env_overrides: dict | None = None):
        """
        Build a TestClient with mocked deps and optional env overrides.

        env_overrides are applied to os.environ for the lifetime of the test
        and cleaned up via addCleanup so request handlers see the correct values.
        """
        from fastapi.testclient import TestClient

        mock_db = MagicMock()
        mock_conf = MagicMock()
        mock_conf.load_config.return_value = {
            "github_token": "",
            "llm_model": "test-model",
            "llm_api_base": "",
            "llm_api_key": "",
        }
        mock_orch = MagicMock()
        mock_orch.trigger_job.return_value = "job-webhook-1"

        # Apply env overrides before reload so the module and request handlers
        # both see the correct values. Capture originals for cleanup.
        originals: dict[str, str | None] = {}
        for key, value in (env_overrides or {}).items():
            originals[key] = os.environ.get(key)
            if value == "":
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        def _restore_env():
            for key, original in originals.items():
                if original is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original

        self.addCleanup(_restore_env)

        with patch("db.lightrag_manager.LightRAGManager", return_value=mock_db), \
             patch("pipeline.config_manager.ConfigManager", return_value=mock_conf), \
             patch("pipeline.job_orchestrator.JobOrchestrator", return_value=mock_orch):
            import api as api_module
            importlib.reload(api_module)
            client = TestClient(api_module.app)

        return client, mock_orch, api_module

    def test_merged_pr_triggers_job_and_returns_200(self):
        client, mock_orch, _ = self._make_client()
        payload = _make_pr_merged_payload("owner/my-repo")
        body = json.dumps(payload).encode()

        resp = client.post(
            "/api/webhooks/github",
            content=body,
            headers={"X-GitHub-Event": "pull_request", "Content-Type": "application/json"},
        )

        self.assertEqual(resp.status_code, 200)
        mock_orch.trigger_job.assert_called_once()

    def test_non_merged_pr_returns_200_but_skips_job(self):
        client, mock_orch, _ = self._make_client()
        payload = _make_pr_closed_not_merged_payload()
        body = json.dumps(payload).encode()

        resp = client.post(
            "/api/webhooks/github",
            content=body,
            headers={"X-GitHub-Event": "pull_request", "Content-Type": "application/json"},
        )

        self.assertEqual(resp.status_code, 200)
        mock_orch.trigger_job.assert_not_called()

    def test_push_event_is_ignored(self):
        client, mock_orch, _ = self._make_client()

        resp = client.post(
            "/api/webhooks/github",
            content=b'{"ref": "refs/heads/main"}',
            headers={"X-GitHub-Event": "push", "Content-Type": "application/json"},
        )

        self.assertEqual(resp.status_code, 200)
        mock_orch.trigger_job.assert_not_called()

    def test_valid_signature_with_secret_allows_request(self):
        secret = "test-secret-abc"
        client, mock_orch, _ = self._make_client(
            env_overrides={"GITHUB_WEBHOOK_SECRET": secret}
        )
        payload = _make_pr_merged_payload()
        body = json.dumps(payload).encode()
        sig = _sign_payload(body, secret)

        resp = client.post(
            "/api/webhooks/github",
            content=body,
            headers={
                "X-GitHub-Event": "pull_request",
                "X-Hub-Signature-256": sig,
                "Content-Type": "application/json",
            },
        )

        self.assertEqual(resp.status_code, 200)
        mock_orch.trigger_job.assert_called_once()

    def test_invalid_signature_with_secret_returns_401(self):
        client, mock_orch, _ = self._make_client(
            env_overrides={"GITHUB_WEBHOOK_SECRET": "real-secret"}
        )
        payload = _make_pr_merged_payload()
        body = json.dumps(payload).encode()
        bad_sig = _sign_payload(body, "wrong-secret")

        resp = client.post(
            "/api/webhooks/github",
            content=body,
            headers={
                "X-GitHub-Event": "pull_request",
                "X-Hub-Signature-256": bad_sig,
                "Content-Type": "application/json",
            },
        )

        self.assertEqual(resp.status_code, 401)
        mock_orch.trigger_job.assert_not_called()

    def test_missing_signature_header_with_secret_returns_401(self):
        client, mock_orch, _ = self._make_client(
            env_overrides={"GITHUB_WEBHOOK_SECRET": "required-secret"}
        )
        payload = _make_pr_merged_payload()
        body = json.dumps(payload).encode()

        resp = client.post(
            "/api/webhooks/github",
            content=body,
            headers={"X-GitHub-Event": "pull_request", "Content-Type": "application/json"},
        )

        self.assertEqual(resp.status_code, 401)

    def test_no_secret_configured_skips_signature_validation(self):
        """Dev mode: no GITHUB_WEBHOOK_SECRET set, no signature header — must succeed."""
        client, mock_orch, _ = self._make_client(
            env_overrides={"GITHUB_WEBHOOK_SECRET": ""}
        )
        payload = _make_pr_merged_payload()
        body = json.dumps(payload).encode()

        resp = client.post(
            "/api/webhooks/github",
            content=body,
            headers={"X-GitHub-Event": "pull_request", "Content-Type": "application/json"},
        )

        self.assertEqual(resp.status_code, 200)

    def test_trigger_job_receives_trigger_source_webhook(self):
        client, mock_orch, _ = self._make_client()
        payload = _make_pr_merged_payload("owner/repo")
        body = json.dumps(payload).encode()

        client.post(
            "/api/webhooks/github",
            content=body,
            headers={"X-GitHub-Event": "pull_request", "Content-Type": "application/json"},
        )

        call_args = mock_orch.trigger_job.call_args
        job_payload = call_args[0][0]  # first positional arg
        self.assertEqual(job_payload.get("trigger_source"), "webhook")

    def test_trigger_job_receives_correct_repo(self):
        client, mock_orch, _ = self._make_client()
        payload = _make_pr_merged_payload("myorg/special-repo")
        body = json.dumps(payload).encode()

        client.post(
            "/api/webhooks/github",
            content=body,
            headers={"X-GitHub-Event": "pull_request", "Content-Type": "application/json"},
        )

        call_args = mock_orch.trigger_job.call_args
        job_payload = call_args[0][0]
        self.assertEqual(job_payload.get("repo"), "myorg/special-repo")

    def test_rate_limited_second_trigger_returns_429(self):
        client, mock_orch, _ = self._make_client()
        payload = _make_pr_merged_payload("owner/repo")
        body = json.dumps(payload).encode()
        headers = {"X-GitHub-Event": "pull_request", "Content-Type": "application/json"}

        resp1 = client.post("/api/webhooks/github", content=body, headers=headers)
        resp2 = client.post("/api/webhooks/github", content=body, headers=headers)

        self.assertEqual(resp1.status_code, 200)
        self.assertEqual(resp2.status_code, 429)
        # Orchestrator called exactly once (second was rate-limited)
        self.assertEqual(mock_orch.trigger_job.call_count, 1)

    def test_response_body_includes_job_id(self):
        client, _, _ = self._make_client()
        payload = _make_pr_merged_payload()
        body = json.dumps(payload).encode()

        resp = client.post(
            "/api/webhooks/github",
            content=body,
            headers={"X-GitHub-Event": "pull_request", "Content-Type": "application/json"},
        )

        data = resp.json()
        self.assertIn("job_id", data)
        self.assertEqual(data["job_id"], "job-webhook-1")


# ---------------------------------------------------------------------------
# Stats endpoint
# ---------------------------------------------------------------------------

class TestWebhookStatsEndpoint(unittest.TestCase):
    """GET /api/webhooks/stats must return trigger counts and last trigger times."""

    def _make_client(self):
        from fastapi.testclient import TestClient

        mock_db = MagicMock()
        mock_conf = MagicMock()
        mock_conf.load_config.return_value = {}
        mock_orch = MagicMock()
        mock_orch.trigger_job.return_value = "job-stats-1"

        with patch("db.lightrag_manager.LightRAGManager", return_value=mock_db), \
             patch("pipeline.config_manager.ConfigManager", return_value=mock_conf), \
             patch("pipeline.job_orchestrator.JobOrchestrator", return_value=mock_orch):
            import api as api_module
            importlib.reload(api_module)
            client = TestClient(api_module.app)

        return client, mock_orch

    def test_stats_endpoint_returns_200(self):
        client, _ = self._make_client()
        resp = client.get("/api/webhooks/stats")
        self.assertEqual(resp.status_code, 200)

    def test_stats_initially_empty(self):
        client, _ = self._make_client()
        resp = client.get("/api/webhooks/stats")
        data = resp.json()
        self.assertIn("repos", data)
        self.assertEqual(data["repos"], {})

    def test_stats_show_trigger_count_after_webhook_fires(self):
        client, _ = self._make_client()

        payload = _make_pr_merged_payload("owner/repo")
        body = json.dumps(payload).encode()
        client.post(
            "/api/webhooks/github",
            content=body,
            headers={"X-GitHub-Event": "pull_request", "Content-Type": "application/json"},
        )

        resp = client.get("/api/webhooks/stats")
        data = resp.json()
        self.assertEqual(data["repos"]["owner/repo"]["trigger_count"], 1)

    def test_stats_show_last_trigger_time_after_webhook_fires(self):
        client, _ = self._make_client()

        payload = _make_pr_merged_payload("owner/repo")
        body = json.dumps(payload).encode()
        client.post(
            "/api/webhooks/github",
            content=body,
            headers={"X-GitHub-Event": "pull_request", "Content-Type": "application/json"},
        )

        resp = client.get("/api/webhooks/stats")
        data = resp.json()
        self.assertIn("last_trigger_time", data["repos"]["owner/repo"])
        self.assertIsNotNone(data["repos"]["owner/repo"]["last_trigger_time"])


# ---------------------------------------------------------------------------
# JobOrchestrator trigger_source tracking
# ---------------------------------------------------------------------------

class TestJobOrchestratorTriggerSource(unittest.TestCase):
    """trigger_job must store trigger_source in the initial active_jobs entry."""

    def _make_orchestrator(self):
        mock_db = MagicMock()
        with patch("pipeline.job_orchestrator.asyncio.create_task"):
            from pipeline.job_orchestrator import JobOrchestrator
            orch = JobOrchestrator(mock_db)
        return orch

    def test_trigger_job_without_source_stores_manual(self):
        """Default trigger_source for jobs started from UI must be 'manual'."""
        orch = self._make_orchestrator()
        mock_config = {"llm_model": "test"}

        with patch("pipeline.job_orchestrator.asyncio.create_task"):
            job_id = orch.trigger_job({"repo": "owner/repo"}, mock_config)

        status = orch.get_status(job_id)
        self.assertEqual(status.get("trigger_source"), "manual")

    def test_trigger_job_with_webhook_source_stores_webhook(self):
        """Jobs started by webhook must record trigger_source='webhook'."""
        orch = self._make_orchestrator()
        mock_config = {"llm_model": "test"}

        with patch("pipeline.job_orchestrator.asyncio.create_task"):
            job_id = orch.trigger_job(
                {"repo": "owner/repo", "trigger_source": "webhook"}, mock_config
            )

        status = orch.get_status(job_id)
        self.assertEqual(status.get("trigger_source"), "webhook")


if __name__ == "__main__":
    unittest.main()
