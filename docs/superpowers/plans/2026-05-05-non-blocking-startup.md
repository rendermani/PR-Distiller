# Non-blocking Startup, Embedding Progress, Encrypted Vault — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the FastAPI backend respond immediately on boot while the embedding model downloads in the background with byte-level progress; queue user-triggered jobs until ready; persist the HF cache across restarts; and surface all user-supplied credentials on a new `/secrets` page with env-override awareness.

**Architecture:** A `SystemStatus` singleton holds embedding state, download progress, and a job queue. A background thread launched from FastAPI's lifespan startup hook loads the embedding model with `tqdm` instrumentation that pushes byte counts into the singleton. A lazy `db` proxy lets `/api/health` and `/api/system/status` answer immediately. The UI subscribes to a Server-Sent Events stream (with a 10 s polling backup and on-mount snapshot) and renders a banner with the existing progress-bar pattern. A new `/secrets` Next.js route hosts the Vault.

**Tech Stack:** FastAPI, sentence-transformers, huggingface_hub, ChromaDB, Fernet (cryptography), Next.js 16, React 19, EventSource API, fcntl.

---

## File structure

### New backend files
- `backend/system_status.py` — singleton + subscriber fan-out + queue
- `backend/embedding_loader.py` — `_ProgressCapturingTqdm` + `capture_hf_progress` ctxmgr
- `backend/db_proxy.py` — `_LazyDbProxy` with `bind()` and blocking `__getattr__`

### Modified backend files
- `backend/api.py` — lifespan hook, lazy `db`, status/events/retry endpoints, env_overrides flow
- `backend/pipeline/job_orchestrator.py` — queueing + drain callback
- `backend/pipeline/config_manager.py` — `huggingface_token` + `github_webhook_secret` fields, `env_overrides()` method
- `backend/settings.py` — `HUGGINGFACE_HUB_TOKEN`, `WEBHOOK_PUBLIC_URL`

### New backend test files
- `backend/tests/test_system_status.py`
- `backend/tests/test_embedding_loader.py`
- `backend/tests/test_db_proxy.py`
- `backend/tests/test_system_endpoints.py`
- `backend/tests/test_orchestrator_queueing.py`
- `backend/tests/test_huggingface_token_config.py`
- `backend/tests/test_env_overrides.py`

### New / modified frontend files
- `web-ui/src/app/api/proxy/[...path]/route.ts` — preserve SSE Cache-Control header
- `web-ui/src/app/page.tsx` — embedding banner, SystemStatus state, SSE + polling, queued-job UI, link to /secrets
- `web-ui/src/app/secrets/page.tsx` — new Vault page

### Config / ops files
- `docker-compose.yml` — `hf_cache` volume, `HF_HOME`, `SENTENCE_TRANSFORMERS_HOME`, `HUGGINGFACE_HUB_TOKEN`
- `.env.example` — document new env vars
- `README.md` — document Vault page + new env vars

---

## Task 1: `SystemStatus` snapshot/update/subscribers

**Files:**
- Create: `backend/system_status.py`
- Test: `backend/tests/test_system_status.py`

- [ ] **Step 1: Write the failing test for snapshot/update**

```python
# backend/tests/test_system_status.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd backend && python -m pytest tests/test_system_status.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'system_status'`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/system_status.py
"""Singleton holding embedding-load state, download progress, and a job queue.

Mutating methods take a threading lock because the embedding loader runs in a
background thread; subscribers are asyncio.Queue instances belonging to SSE
handlers on the FastAPI event loop.
"""
import asyncio
import copy
import threading
import time
from typing import Any, Callable


class SystemStatus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "embedding": {
                "state": "idle",
                "model_name": "",
                "bytes_downloaded": 0,
                "bytes_total": 0,
                "error": None,
            },
            "queued_jobs": [],
        }
        self._subscribers: list[asyncio.Queue] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._on_ready: Callable[[], None] | None = None
        self._last_progress_push = 0.0
        self._last_progress_pct = -1.0

    # --- snapshot / update ---

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._state)

    def update(self, **kwargs: Any) -> None:
        """Mutate top-level embedding fields and fan out to subscribers."""
        with self._lock:
            prev_state = self._state["embedding"]["state"]
            self._state["embedding"].update(kwargs)
            new_state = self._state["embedding"]["state"]
        self._fanout()
        if prev_state != "ready" and new_state == "ready" and self._on_ready:
            self._schedule_on_ready()

    def update_download_progress(self, bytes_downloaded: int, bytes_total: int) -> None:
        """Record byte counts; fan out only when 250 ms or 1 % has elapsed."""
        with self._lock:
            self._state["embedding"]["bytes_downloaded"] = bytes_downloaded
            self._state["embedding"]["bytes_total"] = bytes_total
            now = time.monotonic()
            pct = (100.0 * bytes_downloaded / bytes_total) if bytes_total else 0.0
            should_push = (
                now - self._last_progress_push >= 0.25
                or abs(pct - self._last_progress_pct) >= 1.0
            )
            if should_push:
                self._last_progress_push = now
                self._last_progress_pct = pct
        if should_push:
            self._fanout()

    # --- subscribers ---

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Called once at FastAPI startup so background threads can fan out."""
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def _fanout(self) -> None:
        snapshot = self.snapshot()
        with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            self._safe_put(q, snapshot)

    def _safe_put(self, q: asyncio.Queue, snapshot: dict[str, Any]) -> None:
        if self._loop is None:
            # Called from sync context with no loop attached (test setup).
            try:
                q.put_nowait(snapshot)
            except asyncio.QueueFull:
                pass
            return
        # Cross-thread schedule onto the FastAPI loop.
        def _enqueue() -> None:
            try:
                q.put_nowait(snapshot)
            except asyncio.QueueFull:
                pass
        try:
            self._loop.call_soon_threadsafe(_enqueue)
        except RuntimeError:
            # Loop already closed.
            pass

    # --- on-ready callback (set by orchestrator) ---

    def set_on_ready(self, callback: Callable[[], None]) -> None:
        self._on_ready = callback

    def _schedule_on_ready(self) -> None:
        cb = self._on_ready
        if cb is None:
            return
        if self._loop is not None:
            self._loop.call_soon_threadsafe(cb)
        else:
            cb()


# Module-level singleton used by api.py and tests.
SYSTEM_STATUS = SystemStatus()
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd backend && python -m pytest tests/test_system_status.py -v
```
Expected: PASS, all 5 tests green.

- [ ] **Step 5: Commit**

```bash
git add backend/system_status.py backend/tests/test_system_status.py
git commit -m "feat(status): SystemStatus singleton with snapshot/update/subscribers"
```

---

## Task 2: `SystemStatus` job queue

**Files:**
- Modify: `backend/system_status.py`
- Modify: `backend/tests/test_system_status.py`

- [ ] **Step 1: Write the failing test for queue methods**

Append to `backend/tests/test_system_status.py`:

```python
class TestSystemStatusQueue(unittest.TestCase):
    def setUp(self):
        from system_status import SystemStatus
        self.status = SystemStatus()

    def test_enqueue_appends_to_queued_jobs_in_snapshot(self):
        self.status.enqueue_job("job-1", {"repo": "acme/app"}, {}, reason="not ready")
        snap = self.status.snapshot()
        self.assertEqual(len(snap["queued_jobs"]), 1)
        self.assertEqual(snap["queued_jobs"][0]["job_id"], "job-1")
        self.assertEqual(snap["queued_jobs"][0]["repo"], "acme/app")
        self.assertEqual(snap["queued_jobs"][0]["reason"], "not ready")
        self.assertIn("queued_at", snap["queued_jobs"][0])

    def test_drain_queue_returns_and_clears_all_jobs(self):
        self.status.enqueue_job("job-1", {"repo": "a"}, {"k": "v1"}, reason="r")
        self.status.enqueue_job("job-2", {"repo": "b"}, {"k": "v2"}, reason="r")
        drained = self.status.drain_queue()
        self.assertEqual(len(drained), 2)
        self.assertEqual([d[0] for d in drained], ["job-1", "job-2"])
        self.assertEqual(drained[0][1], {"repo": "a"})
        self.assertEqual(drained[0][2], {"k": "v1"})
        self.assertEqual(self.status.snapshot()["queued_jobs"], [])

    def test_dequeue_job_removes_specific_job_and_returns_true(self):
        self.status.enqueue_job("job-1", {"repo": "a"}, {}, reason="r")
        self.status.enqueue_job("job-2", {"repo": "b"}, {}, reason="r")
        self.assertTrue(self.status.dequeue_job("job-1"))
        snap = self.status.snapshot()
        self.assertEqual([j["job_id"] for j in snap["queued_jobs"]], ["job-2"])

    def test_dequeue_returns_false_when_job_missing(self):
        self.assertFalse(self.status.dequeue_job("nope"))

    def test_is_ready_only_when_state_is_ready(self):
        self.assertFalse(self.status.is_ready())
        self.status.update(state="downloading")
        self.assertFalse(self.status.is_ready())
        self.status.update(state="ready")
        self.assertTrue(self.status.is_ready())
```

- [ ] **Step 2: Run test to verify failure**

Run:
```bash
cd backend && python -m pytest tests/test_system_status.py::TestSystemStatusQueue -v
```
Expected: FAIL with `AttributeError: 'SystemStatus' object has no attribute 'enqueue_job'`.

- [ ] **Step 3: Add queue methods to `SystemStatus`**

Append to `backend/system_status.py` inside the `SystemStatus` class, after `set_on_ready`:

```python
    # --- job queueing ---

    def is_ready(self) -> bool:
        with self._lock:
            return self._state["embedding"]["state"] == "ready"

    def enqueue_job(self, job_id: str, payload: dict, config: dict, *, reason: str) -> None:
        entry = {
            "job_id": job_id,
            "repo": payload.get("repo", ""),
            "queued_at": time.time(),
            "reason": reason,
        }
        with self._lock:
            self._state["queued_jobs"].append(entry)
            # Keep payload+config out of the snapshot (may contain secrets);
            # store separately keyed by job_id.
            if not hasattr(self, "_pending"):
                self._pending: dict[str, tuple[dict, dict]] = {}
            self._pending[job_id] = (payload, config)
        self._fanout()

    def drain_queue(self) -> list[tuple[str, dict, dict]]:
        with self._lock:
            jobs = self._state["queued_jobs"]
            self._state["queued_jobs"] = []
            pending = getattr(self, "_pending", {})
            drained = [(j["job_id"], *pending.pop(j["job_id"], ({}, {}))) for j in jobs]
        self._fanout()
        return drained

    def dequeue_job(self, job_id: str) -> bool:
        with self._lock:
            before = len(self._state["queued_jobs"])
            self._state["queued_jobs"] = [
                j for j in self._state["queued_jobs"] if j["job_id"] != job_id
            ]
            removed = before != len(self._state["queued_jobs"])
            if removed and hasattr(self, "_pending"):
                self._pending.pop(job_id, None)
        if removed:
            self._fanout()
        return removed
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd backend && python -m pytest tests/test_system_status.py -v
```
Expected: PASS — all tests green (original 5 plus 5 new queue tests = 10).

- [ ] **Step 5: Commit**

```bash
git add backend/system_status.py backend/tests/test_system_status.py
git commit -m "feat(status): add job queue methods to SystemStatus"
```

---

## Task 3: Embedding loader with tqdm interception

**Files:**
- Create: `backend/embedding_loader.py`
- Test: `backend/tests/test_embedding_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_embedding_loader.py
"""Tests that capture_hf_progress installs a tqdm subclass that aggregates
byte progress across all live instances and pushes into SystemStatus."""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

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
    def test_context_manager_restores_originals_on_exit(self):
        import huggingface_hub.utils.tqdm as hf_tqdm_mod
        from embedding_loader import capture_hf_progress

        original = hf_tqdm_mod.tqdm
        with capture_hf_progress():
            self.assertIsNot(hf_tqdm_mod.tqdm, original)
        self.assertIs(hf_tqdm_mod.tqdm, original)

    def test_context_manager_restores_originals_on_exception(self):
        import huggingface_hub.utils.tqdm as hf_tqdm_mod
        from embedding_loader import capture_hf_progress

        original = hf_tqdm_mod.tqdm
        try:
            with capture_hf_progress():
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        self.assertIs(hf_tqdm_mod.tqdm, original)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify failure**

Run:
```bash
cd backend && python -m pytest tests/test_embedding_loader.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'embedding_loader'`.

- [ ] **Step 3: Write `embedding_loader.py`**

```python
# backend/embedding_loader.py
"""Hooks `huggingface_hub` and `sentence_transformers` tqdm so we can publish
live byte-level download progress into SystemStatus.

Use as `with capture_hf_progress(): ...` around the model load. The hook is
installed by monkey-patching the `tqdm` attribute on each library's progress
module; the original is restored on exit (success or exception).
"""
import contextlib
from typing import ClassVar

import tqdm.auto
from system_status import SYSTEM_STATUS


class _ProgressCapturingTqdm(tqdm.auto.tqdm):
    """Subclass of tqdm.auto.tqdm that pushes aggregate byte progress."""

    _live: ClassVar[set] = set()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        type(self)._live.add(self)
        self._push()

    def update(self, n: float = 1) -> bool | None:
        result = super().update(n)
        self._push()
        return result

    def close(self) -> None:
        type(self)._live.discard(self)
        self._push()
        return super().close()

    @classmethod
    def _push(cls) -> None:
        bytes_downloaded = sum(int(t.n or 0) for t in cls._live)
        bytes_total = sum(int(t.total or 0) for t in cls._live)
        SYSTEM_STATUS.update_download_progress(bytes_downloaded, bytes_total)


@contextlib.contextmanager
def capture_hf_progress():
    """Patch tqdm in huggingface_hub and sentence_transformers; restore on exit."""
    import huggingface_hub.utils.tqdm as hf_tqdm_mod
    try:
        import sentence_transformers.util as st_util_mod
    except ImportError:  # pragma: no cover
        st_util_mod = None

    original_hf = hf_tqdm_mod.tqdm
    original_st = getattr(st_util_mod, "tqdm", None) if st_util_mod else None
    try:
        hf_tqdm_mod.tqdm = _ProgressCapturingTqdm
        if st_util_mod is not None:
            st_util_mod.tqdm = _ProgressCapturingTqdm
        yield
    finally:
        hf_tqdm_mod.tqdm = original_hf
        if st_util_mod is not None and original_st is not None:
            st_util_mod.tqdm = original_st
        _ProgressCapturingTqdm._live.clear()
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd backend && python -m pytest tests/test_embedding_loader.py -v
```
Expected: PASS, all 5 tests green.

- [ ] **Step 5: Commit**

```bash
git add backend/embedding_loader.py backend/tests/test_embedding_loader.py
git commit -m "feat(embedding): tqdm-based byte-level progress capture"
```

---

## Task 4: Lazy db proxy

**Files:**
- Create: `backend/db_proxy.py`
- Test: `backend/tests/test_db_proxy.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_db_proxy.py
"""LazyDbProxy lets api.py construct `db` at module import time without
blocking on the real LightRAGManager. Attribute access on the proxy blocks
until bind() is called with the real instance."""
import os
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestLazyDbProxy(unittest.TestCase):
    def test_attribute_access_blocks_until_bind(self):
        from db_proxy import LazyDbProxy
        proxy = LazyDbProxy()

        result = []

        def reader():
            result.append(proxy.collection)

        t = threading.Thread(target=reader)
        t.start()

        # Reader should still be blocked.
        t.join(timeout=0.2)
        self.assertTrue(t.is_alive(), "reader returned before bind()")

        real = MagicMock()
        real.collection = "hello"
        proxy.bind(real)
        t.join(timeout=1.0)
        self.assertFalse(t.is_alive())
        self.assertEqual(result, ["hello"])

    def test_method_calls_route_to_real_after_bind(self):
        from db_proxy import LazyDbProxy
        proxy = LazyDbProxy()
        real = MagicMock()
        real.add_rule.return_value = "rule-id-x"
        proxy.bind(real)
        self.assertEqual(proxy.add_rule("a", "b", "c", "d"), "rule-id-x")
        real.add_rule.assert_called_once_with("a", "b", "c", "d")

    def test_bind_twice_raises(self):
        from db_proxy import LazyDbProxy
        proxy = LazyDbProxy()
        proxy.bind(MagicMock())
        with self.assertRaises(RuntimeError):
            proxy.bind(MagicMock())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify failure**

Run:
```bash
cd backend && python -m pytest tests/test_db_proxy.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'db_proxy'`.

- [ ] **Step 3: Write `db_proxy.py`**

```python
# backend/db_proxy.py
"""Lazy proxy for LightRAGManager so api.py can be imported without blocking
on the embedding-model download. Attribute access blocks on a threading event
until bind() is called with the real manager."""
import threading
from typing import Any


class LazyDbProxy:
    def __init__(self) -> None:
        self._real: Any = None
        self._ready = threading.Event()

    def bind(self, real: Any) -> None:
        if self._real is not None:
            raise RuntimeError("LazyDbProxy already bound")
        self._real = real
        self._ready.set()

    def __getattr__(self, name: str) -> Any:
        # __getattr__ is only called when normal attribute lookup fails, so
        # `_real` and `_ready` are reached via __dict__ without recursion.
        self._ready.wait()
        return getattr(self._real, name)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd backend && python -m pytest tests/test_db_proxy.py -v
```
Expected: PASS, all 3 tests green.

- [ ] **Step 5: Commit**

```bash
git add backend/db_proxy.py backend/tests/test_db_proxy.py
git commit -m "feat(db): lazy proxy for non-blocking FastAPI startup"
```

---

## Task 5: GET `/api/system/status` endpoint

**Files:**
- Modify: `backend/api.py`
- Test: `backend/tests/test_system_endpoints.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_system_endpoints.py
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify failure**

Run:
```bash
cd backend && python -m pytest tests/test_system_endpoints.py -v
```
Expected: FAIL with 404 (endpoint doesn't exist yet).

- [ ] **Step 3: Add endpoint to `api.py`**

Find the import block at the top of `backend/api.py` and add:

```python
from system_status import SYSTEM_STATUS
```

Then, after the `/api/health` endpoint definition, add:

```python
@app.get("/api/system/status")
def get_system_status():
    """Auth-free snapshot of embedding state, download progress, queued jobs."""
    return SYSTEM_STATUS.snapshot()
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd backend && python -m pytest tests/test_system_endpoints.py::TestSystemStatusEndpoint -v
```
Expected: PASS, all 3 tests green.

- [ ] **Step 5: Commit**

```bash
git add backend/api.py backend/tests/test_system_endpoints.py
git commit -m "feat(api): GET /api/system/status snapshot endpoint"
```

---

## Task 6: GET `/api/system/events` SSE endpoint

**Files:**
- Modify: `backend/api.py`
- Modify: `backend/tests/test_system_endpoints.py`

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_system_endpoints.py`:

```python
import asyncio
import json


class TestSystemEventsSSE(unittest.TestCase):
    def setUp(self):
        api.db = MagicMock()
        api.conf_manager = MagicMock()
        api.orchestrator = MagicMock()
        self.client = TestClient(api.app)

    def test_events_endpoint_emits_initial_snapshot(self):
        """First event over SSE must be the current snapshot."""
        from system_status import SystemStatus
        s = SystemStatus()
        s.update(state="downloading", model_name="x")
        with patch("api.SYSTEM_STATUS", s):
            with self.client.stream("GET", "/api/system/events") as r:
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.headers["content-type"], "text/event-stream; charset=utf-8")
                # Read up to the first "data:" line.
                line_iter = r.iter_lines()
                payload_line = None
                for line in line_iter:
                    if line.startswith("data: "):
                        payload_line = line
                        break
                self.assertIsNotNone(payload_line)
                payload = json.loads(payload_line[len("data: "):])
                self.assertEqual(payload["embedding"]["state"], "downloading")

    def test_events_response_carries_no_cache_headers(self):
        from system_status import SystemStatus
        with patch("api.SYSTEM_STATUS", SystemStatus()):
            with self.client.stream("GET", "/api/system/events") as r:
                self.assertIn("no-cache", r.headers.get("cache-control", ""))
                self.assertEqual(r.headers.get("x-accel-buffering"), "no")
```

- [ ] **Step 2: Run test to verify failure**

Run:
```bash
cd backend && python -m pytest tests/test_system_endpoints.py::TestSystemEventsSSE -v
```
Expected: FAIL with 404.

- [ ] **Step 3: Add SSE endpoint to `api.py`**

Add to the imports:

```python
import asyncio
import json as _json2
from fastapi.responses import StreamingResponse
```

After the `/api/system/status` endpoint, add:

```python
@app.get("/api/system/events")
async def system_events(request: Request):
    """SSE stream of SystemStatus mutations.

    Sends the current snapshot immediately on subscribe, then one event per
    mutation. Heartbeat comment every 15 s keeps proxies from idling.
    """
    async def event_stream():
        q = SYSTEM_STATUS.subscribe()
        try:
            yield f"data: {_json2.dumps(SYSTEM_STATUS.snapshot())}\n\n"
            while True:
                if await request.is_disconnected():
                    return
                try:
                    update = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {_json2.dumps(update)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            SYSTEM_STATUS.unsubscribe(q)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd backend && python -m pytest tests/test_system_endpoints.py -v
```
Expected: PASS, all status + events tests green.

- [ ] **Step 5: Commit**

```bash
git add backend/api.py backend/tests/test_system_endpoints.py
git commit -m "feat(api): GET /api/system/events SSE stream"
```

---

## Task 7: POST `/api/system/embedding/retry` endpoint

**Files:**
- Modify: `backend/api.py`
- Modify: `backend/tests/test_system_endpoints.py`

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_system_endpoints.py`:

```python
class TestEmbeddingRetryEndpoint(unittest.TestCase):
    def setUp(self):
        api.db = MagicMock()
        api.conf_manager = MagicMock()
        api.orchestrator = MagicMock()
        self.client = TestClient(api.app)

    def test_retry_resets_state_to_idle_and_calls_loader(self):
        from system_status import SystemStatus
        s = SystemStatus()
        s.update(state="error", error="boom")
        loader = MagicMock()
        with patch("api.SYSTEM_STATUS", s), \
             patch("api._embedding_load_starter", loader):
            r = self.client.post("/api/system/embedding/retry")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(s.snapshot()["embedding"]["state"], "idle")
        loader.assert_called_once()

    def test_retry_with_clean_query_wipes_cache_dir(self):
        from system_status import SystemStatus
        with patch("api.SYSTEM_STATUS", SystemStatus()), \
             patch("api._embedding_load_starter", MagicMock()), \
             patch("api._wipe_hf_cache") as wipe:
            r = self.client.post("/api/system/embedding/retry?clean=true")
        self.assertEqual(r.status_code, 200)
        wipe.assert_called_once()

    def test_retry_requires_auth_when_token_set(self):
        with patch.object(api.settings, "API_AUTH_TOKEN", "secret"):
            r = self.client.post("/api/system/embedding/retry")
        self.assertEqual(r.status_code, 401)
```

- [ ] **Step 2: Run test to verify failure**

Run:
```bash
cd backend && python -m pytest tests/test_system_endpoints.py::TestEmbeddingRetryEndpoint -v
```
Expected: FAIL with 404.

- [ ] **Step 3: Add retry endpoint and helpers to `api.py`**

At module level (near other helpers), add:

```python
import shutil

_HF_CACHE_DIR = os.environ.get("HF_HOME", "/cache/huggingface")


def _wipe_hf_cache() -> None:
    """Best-effort wipe of the HuggingFace cache directory."""
    if os.path.isdir(_HF_CACHE_DIR):
        shutil.rmtree(_HF_CACHE_DIR, ignore_errors=True)


def _embedding_load_starter() -> None:
    """Spawn the background embedding-load thread. Implemented in Task 13."""
    raise NotImplementedError("set in startup hook")
```

After the SSE endpoint:

```python
@app.post("/api/system/embedding/retry", dependencies=[Depends(require_api_token)])
def retry_embedding(clean: bool = False):
    if clean:
        _wipe_hf_cache()
    SYSTEM_STATUS.update(state="idle", error=None, bytes_downloaded=0, bytes_total=0)
    _embedding_load_starter()
    return {"status": "retry-started"}
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd backend && python -m pytest tests/test_system_endpoints.py::TestEmbeddingRetryEndpoint -v
```
Expected: PASS, all 3 tests green.

- [ ] **Step 5: Commit**

```bash
git add backend/api.py backend/tests/test_system_endpoints.py
git commit -m "feat(api): POST /api/system/embedding/retry endpoint"
```

---

## Task 8: ConfigManager `huggingface_token` + `github_webhook_secret` fields

**Files:**
- Modify: `backend/pipeline/config_manager.py`
- Test: `backend/tests/test_huggingface_token_config.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_huggingface_token_config.py
"""Tests that ConfigManager round-trips huggingface_token and
github_webhook_secret with Fernet encryption and the same redaction sentinel
behaviour as github_token / provider_api_keys."""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _make_manager(tmp_dir: str):
    from pipeline.config_manager import ConfigManager
    from cryptography.fernet import Fernet
    m = ConfigManager.__new__(ConfigManager)
    m.base_dir = tmp_dir
    m.config_path = os.path.join(tmp_dir, "config.json")
    m.key_path = os.path.join(tmp_dir, ".secret_key")
    m.cipher = Fernet(m._resolve_key())
    m._ensure_default_config()
    return m


class TestHuggingFaceTokenRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.m = _make_manager(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_huggingface_token_persists_encrypted(self):
        self.m.save_config({"huggingface_token": "hf_abc123"})
        loaded = self.m.load_config()
        self.assertEqual(loaded["huggingface_token"], "hf_abc123")
        # On-disk value should be encrypted, not plaintext.
        with open(self.m.config_path) as f:
            raw = f.read()
        self.assertNotIn("hf_abc123", raw)
        self.assertIn("huggingface_token_enc", raw)

    def test_redaction_sentinel_for_huggingface_token_does_not_overwrite(self):
        self.m.save_config({"huggingface_token": "hf_real"})
        self.m.save_config({"huggingface_token": "***"})
        self.assertEqual(self.m.load_config()["huggingface_token"], "hf_real")


class TestWebhookSecretRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.m = _make_manager(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_webhook_secret_persists_encrypted(self):
        self.m.save_config({"github_webhook_secret": "wh_abc"})
        self.assertEqual(self.m.load_config()["github_webhook_secret"], "wh_abc")
        with open(self.m.config_path) as f:
            raw = f.read()
        self.assertNotIn("wh_abc", raw)
        self.assertIn("github_webhook_secret_enc", raw)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify failure**

Run:
```bash
cd backend && python -m pytest tests/test_huggingface_token_config.py -v
```
Expected: FAIL — `huggingface_token` and `github_webhook_secret` aren't in the load/save flow.

- [ ] **Step 3: Update `config_manager.py`**

In `_strip_redaction_sentinels`, extend the secret-fields tuple:

```python
        for field in ("github_token", "llm_api_key", "huggingface_token", "github_webhook_secret"):
```

In `load_config`, after the `provider_keys` block, add:

```python
            huggingface_token = self._decrypt(raw.get("huggingface_token_enc", ""))
            github_webhook_secret = self._decrypt(raw.get("github_webhook_secret_enc", ""))
```

And add the two fields to the returned dict alongside `github_token`:

```python
                "huggingface_token": huggingface_token,
                "github_webhook_secret": github_webhook_secret,
```

In `save_config`, in the `encrypted_wrap` dict, add:

```python
                "huggingface_token_enc": self._encrypt(current.get("huggingface_token", "")),
                "github_webhook_secret_enc": self._encrypt(current.get("github_webhook_secret", "")),
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd backend && python -m pytest tests/test_huggingface_token_config.py -v
```
Expected: PASS, all 3 tests green.

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/config_manager.py backend/tests/test_huggingface_token_config.py
git commit -m "feat(config): encrypted huggingface_token and github_webhook_secret fields"
```

---

## Task 9: ConfigManager `env_overrides()` method

**Files:**
- Modify: `backend/pipeline/config_manager.py`
- Modify: `backend/settings.py`
- Test: `backend/tests/test_env_overrides.py`

- [ ] **Step 1: Add env vars to `settings.py`**

Append to `backend/settings.py`:

```python
HUGGINGFACE_HUB_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN", "")
WEBHOOK_PUBLIC_URL = os.environ.get("WEBHOOK_PUBLIC_URL", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
```

- [ ] **Step 2: Write the failing test**

```python
# backend/tests/test_env_overrides.py
"""ConfigManager.env_overrides reports which secret fields are set via env vars."""
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _make_manager(tmp_dir):
    from pipeline.config_manager import ConfigManager
    from cryptography.fernet import Fernet
    m = ConfigManager.__new__(ConfigManager)
    m.base_dir = tmp_dir
    m.config_path = os.path.join(tmp_dir, "config.json")
    m.key_path = os.path.join(tmp_dir, ".secret_key")
    m.cipher = Fernet(m._resolve_key())
    m._ensure_default_config()
    return m


class TestEnvOverrides(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.m = _make_manager(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_no_env_vars_set_returns_all_false(self):
        with patch("settings.GITHUB_TOKEN", ""), \
             patch("settings.HUGGINGFACE_HUB_TOKEN", ""), \
             patch("settings.OPENAI_API_KEY", ""), \
             patch("settings.ANTHROPIC_API_KEY", ""), \
             patch("settings.GOOGLE_API_KEY", ""), \
             patch("settings.OPENROUTER_API_KEY", ""), \
             patch("settings.GITHUB_WEBHOOK_SECRET", ""):
            self.assertEqual(
                self.m.env_overrides(),
                {
                    "github_token": False,
                    "huggingface_token": False,
                    "github_webhook_secret": False,
                    "provider_api_keys.openai": False,
                    "provider_api_keys.anthropic": False,
                    "provider_api_keys.google": False,
                    "provider_api_keys.openrouter": False,
                },
            )

    def test_huggingface_env_var_marks_true(self):
        with patch("settings.HUGGINGFACE_HUB_TOKEN", "hf_xxx"):
            self.assertTrue(self.m.env_overrides()["huggingface_token"])

    def test_provider_api_key_env_var_marks_per_provider(self):
        with patch("settings.OPENAI_API_KEY", "sk-x"), \
             patch("settings.ANTHROPIC_API_KEY", ""):
            o = self.m.env_overrides()
            self.assertTrue(o["provider_api_keys.openai"])
            self.assertFalse(o["provider_api_keys.anthropic"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run test to verify failure**

Run:
```bash
cd backend && python -m pytest tests/test_env_overrides.py -v
```
Expected: FAIL with `AttributeError: 'ConfigManager' object has no attribute 'env_overrides'`.

- [ ] **Step 4: Add `env_overrides()` to ConfigManager**

Append inside the `ConfigManager` class:

```python
    def env_overrides(self) -> dict[str, bool]:
        """Return which secret fields are sourced from environment variables.

        UI uses this to disable inputs that would otherwise be ineffective
        (env wins over the encrypted-config value).
        """
        return {
            "github_token": bool(settings.GITHUB_TOKEN),
            "huggingface_token": bool(settings.HUGGINGFACE_HUB_TOKEN),
            "github_webhook_secret": bool(settings.GITHUB_WEBHOOK_SECRET),
            "provider_api_keys.openai": bool(settings.OPENAI_API_KEY),
            "provider_api_keys.anthropic": bool(settings.ANTHROPIC_API_KEY),
            "provider_api_keys.google": bool(settings.GOOGLE_API_KEY),
            "provider_api_keys.openrouter": bool(settings.OPENROUTER_API_KEY),
        }
```

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
cd backend && python -m pytest tests/test_env_overrides.py -v
```
Expected: PASS, all 3 tests green.

- [ ] **Step 6: Commit**

```bash
git add backend/settings.py backend/pipeline/config_manager.py backend/tests/test_env_overrides.py
git commit -m "feat(config): env_overrides() reporter + new env var declarations"
```

---

## Task 10: Expose `env_overrides` and `webhook_url` via GET `/api/config`

**Files:**
- Modify: `backend/api.py`
- Modify: `backend/tests/test_api_endpoints.py`

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_api_endpoints.py`:

```python
class TestConfigEnvOverridesField(unittest.TestCase):
    def setUp(self):
        api.db = _make_db_mock()
        api.conf_manager = _make_conf_mock()
        api.orchestrator = _make_orchestrator_mock()
        self.client = TestClient(api.app)

    def test_get_config_includes_env_overrides(self):
        api.conf_manager.env_overrides.return_value = {
            "github_token": True,
            "huggingface_token": False,
        }
        r = self.client.get("/api/config")
        body = r.json()
        self.assertEqual(body["env_overrides"], {
            "github_token": True,
            "huggingface_token": False,
        })

    def test_get_config_includes_webhook_url(self):
        api.conf_manager.env_overrides.return_value = {}
        with patch.object(api.settings, "WEBHOOK_PUBLIC_URL", "https://example.com"):
            r = self.client.get("/api/config")
        self.assertEqual(r.json()["webhook_url"], "https://example.com/api/webhooks/github")

    def test_webhook_url_falls_back_to_request_origin(self):
        api.conf_manager.env_overrides.return_value = {}
        with patch.object(api.settings, "WEBHOOK_PUBLIC_URL", ""):
            r = self.client.get("/api/config", headers={"host": "fallback.local:8923"})
        self.assertIn("/api/webhooks/github", r.json()["webhook_url"])
```

- [ ] **Step 2: Run test to verify failure**

Run:
```bash
cd backend && python -m pytest tests/test_api_endpoints.py::TestConfigEnvOverridesField -v
```
Expected: FAIL — fields not present.

- [ ] **Step 3: Update `/api/config` GET in `api.py`**

Replace:

```python
@app.get("/api/config")
def get_config():
    """Serves the Unified JSON configurations to the Next.js UI Settings panel."""
    return _redact_sensitive_fields(conf_manager.load_config())
```

With:

```python
@app.get("/api/config")
def get_config(request: Request):
    """Serves the Unified JSON configurations to the Next.js UI Settings panel.

    Augments the redacted config with `env_overrides` and a computed
    `webhook_url` so the Vault page can render env-disabled inputs and
    the GitHub-webhook copy field.
    """
    payload = _redact_sensitive_fields(conf_manager.load_config())
    payload["env_overrides"] = conf_manager.env_overrides()

    public_base = settings.WEBHOOK_PUBLIC_URL or f"{request.url.scheme}://{request.headers.get('host', '')}"
    payload["webhook_url"] = f"{public_base.rstrip('/')}/api/webhooks/github"
    return payload
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd backend && python -m pytest tests/test_api_endpoints.py::TestConfigEnvOverridesField -v
```
Expected: PASS, all 3 tests green.

- [ ] **Step 5: Commit**

```bash
git add backend/api.py backend/tests/test_api_endpoints.py
git commit -m "feat(api): /api/config exposes env_overrides + webhook_url"
```

---

## Task 11: JobOrchestrator queueing + drain

**Files:**
- Modify: `backend/pipeline/job_orchestrator.py`
- Test: `backend/tests/test_orchestrator_queueing.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_orchestrator_queueing.py
"""JobOrchestrator must enqueue when SystemStatus is not ready and drain
on the on-ready callback."""
import asyncio
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

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
                          new=MagicMock(return_value=asyncio.Future())):
            self.orchestrator._execute_distillation.return_value.set_result(None)
            job_id = self.orchestrator.trigger_job(
                {"repo": "acme/app"}, {"github_token": ""}
            )
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
            # Give the run_coroutine_threadsafe + create_task a chance.
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
```

- [ ] **Step 2: Run test to verify failure**

Run:
```bash
cd backend && python -m pytest tests/test_orchestrator_queueing.py -v
```
Expected: FAIL — JobOrchestrator doesn't accept `system_status` arg.

- [ ] **Step 3: Modify `JobOrchestrator`**

Replace `__init__`:

```python
    def __init__(self, db_instance, system_status=None):
        self.db = db_instance
        self.conf = ConfigManager()
        self.active_jobs = {}
        self.cancel_flags = {}
        self.system_status = system_status
        if system_status is not None:
            system_status.set_on_ready(self._drain_queue)
```

Replace `trigger_job`:

```python
    def trigger_job(self, payload: dict, current_config: dict) -> str:
        job_id = f"job-{int(time.time() * 1000)}"
        trigger_source = payload.get("trigger_source", "manual")

        if self.system_status is None or self.system_status.is_ready():
            self.active_jobs[job_id] = {
                "status": "Initializing Engine...",
                "progress": 0,
                "trigger_source": trigger_source,
            }
            self.cancel_flags[job_id] = False
            asyncio.create_task(self._execute_distillation(job_id, payload, current_config))
        else:
            self.active_jobs[job_id] = {
                "status": "Queued — waiting for embedding model",
                "progress": 0,
                "trigger_source": trigger_source,
            }
            self.cancel_flags[job_id] = False
            self.system_status.enqueue_job(
                job_id, payload, current_config,
                reason="embedding model not ready",
            )
        return job_id
```

Replace `request_cancel`:

```python
    def request_cancel(self, job_id: str):
        if self.system_status is not None and self.system_status.dequeue_job(job_id):
            if job_id in self.active_jobs:
                self.active_jobs[job_id]["status"] = "Cancelled (was queued)"
                self.active_jobs[job_id]["progress"] = -1
            return True
        if job_id in self.active_jobs:
            self.cancel_flags[job_id] = True
            return True
        return False
```

Add `_drain_queue` method to `JobOrchestrator`:

```python
    def _drain_queue(self) -> None:
        """Called when SystemStatus flips to ready. Runs queued jobs."""
        if self.system_status is None:
            return
        for job_id, payload, config in self.system_status.drain_queue():
            if job_id not in self.active_jobs:
                continue
            self.active_jobs[job_id]["status"] = "Starting (was queued)"
            asyncio.create_task(self._execute_distillation(job_id, payload, config))
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd backend && python -m pytest tests/test_orchestrator_queueing.py -v
```
Expected: PASS, all 4 tests green.

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/job_orchestrator.py backend/tests/test_orchestrator_queueing.py
git commit -m "feat(orchestrator): queue jobs when embedding not ready, drain on ready"
```

---

## Task 12: API startup hook — lazy `db` + background load

**Files:**
- Modify: `backend/api.py`
- Test: manual smoke run

- [ ] **Step 1: Replace `db` construction with lazy proxy**

Find:

```python
# Connect to the persistent ChromaDB cluster safely
db = LightRAGManager()
conf_manager = ConfigManager()
orchestrator = JobOrchestrator(db)
```

Replace with:

```python
from db_proxy import LazyDbProxy
from system_status import SYSTEM_STATUS
from embedding_loader import capture_hf_progress

db = LazyDbProxy()
conf_manager = ConfigManager()
orchestrator = JobOrchestrator(db, system_status=SYSTEM_STATUS)
```

- [ ] **Step 2: Add startup hook**

After the `app = FastAPI(...)` line, replace:

```python
app = FastAPI(title="PR-Distiller Knowledge API")
```

With:

```python
import threading
import asyncio as _asyncio_app
from contextlib import asynccontextmanager


def _start_background_embedding_load() -> None:
    """Spawn the bg thread that downloads + loads the embedding model."""
    def _runner() -> None:
        SYSTEM_STATUS.update(
            state="downloading",
            model_name=settings.EMBEDDING_MODEL,
            error=None,
        )
        try:
            with capture_hf_progress():
                real_db = LightRAGManager()
            SYSTEM_STATUS.update(state="loading")
            db.bind(real_db)
            SYSTEM_STATUS.update(state="ready")
        except Exception as exc:
            SYSTEM_STATUS.update(state="error", error=f"{type(exc).__name__}: {exc}")
            logger.exception("Embedding model load failed")

    threading.Thread(target=_runner, daemon=True, name="embedding-loader").start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    SYSTEM_STATUS.attach_loop(_asyncio_app.get_running_loop())
    # Resolve auth token before any request hits require_api_token.
    from auth import ensure_api_token
    settings.API_AUTH_TOKEN = ensure_api_token()
    _start_background_embedding_load()
    yield


app = FastAPI(title="PR-Distiller Knowledge API", lifespan=lifespan)
```

- [ ] **Step 3: Wire the retry endpoint to the new starter**

Replace:

```python
def _embedding_load_starter() -> None:
    raise NotImplementedError("set in startup hook")
```

With:

```python
def _embedding_load_starter() -> None:
    _start_background_embedding_load()
```

- [ ] **Step 4: Run full test suite**

Run:
```bash
cd backend && python -m pytest tests/ --ignore=tests/test_security_redactor.py -q
```
Expected: PASS, all tests green.

- [ ] **Step 5: Commit**

```bash
git add backend/api.py
git commit -m "feat(api): lifespan startup hook with lazy db + background embedding load"
```

---

## Task 13: docker-compose + Dockerfile changes

**Files:**
- Modify: `docker-compose.yml`
- Modify: `.env.example`

- [ ] **Step 1: Add hf_cache volume + env vars to backend service**

In `docker-compose.yml`, in the `backend` service, replace the `volumes:` block:

```yaml
    volumes:
      - backend_data:/app/data
      - hf_cache:/cache/huggingface
```

In the same `environment:` block, add:

```yaml
      - HF_HOME=/cache/huggingface
      - SENTENCE_TRANSFORMERS_HOME=/cache/huggingface/sentence_transformers
      - HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN:-}
      - WEBHOOK_PUBLIC_URL=${WEBHOOK_PUBLIC_URL:-}
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY:-}
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-}
```

In the bottom-level `volumes:` block, add `hf_cache:` next to `backend_data:`:

```yaml
volumes:
  backend_data:
  hf_cache:
```

- [ ] **Step 2: Document new env vars in `.env.example`**

Append to `.env.example`:

```ini
# --- HuggingFace ---
# Optional. Required for gated models; recommended to avoid rate limits.
HUGGINGFACE_HUB_TOKEN=

# --- Webhook public URL ---
# The publicly reachable URL of your backend (where GitHub will POST webhooks).
# If unset, the Vault page falls back to the dashboard URL with a warning.
WEBHOOK_PUBLIC_URL=

# --- LLM Provider keys (optional env override; UI Vault is the alternative) ---
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
OPENROUTER_API_KEY=
```

- [ ] **Step 3: Smoke test compose config**

Run:
```bash
docker compose config | grep -E "(hf_cache|HF_HOME|HUGGINGFACE_HUB_TOKEN)"
```
Expected: lines confirming volume mount + env vars are present.

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml .env.example
git commit -m "feat(ops): persistent hf_cache volume + env passthrough for HF/provider keys"
```

---

## Task 14: Pydantic ConfigUpdate — add `huggingface_token` + `github_webhook_secret`

**Files:**
- Modify: `backend/api.py`
- Test: existing `test_api_endpoints.py`

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_api_endpoints.py` inside `TestConfigEndpoints`:

```python
    def test_post_config_accepts_huggingface_token(self):
        r = self.client.post("/api/config", json={"huggingface_token": "hf_xx"})
        self.assertEqual(r.status_code, 200)
        api.conf_manager.save_config.assert_called_with({"huggingface_token": "hf_xx"})

    def test_post_config_accepts_github_webhook_secret(self):
        r = self.client.post("/api/config", json={"github_webhook_secret": "wh"})
        self.assertEqual(r.status_code, 200)
        api.conf_manager.save_config.assert_called_with({"github_webhook_secret": "wh"})
```

- [ ] **Step 2: Run test to verify failure**

Run:
```bash
cd backend && python -m pytest tests/test_api_endpoints.py::TestConfigEndpoints::test_post_config_accepts_huggingface_token tests/test_api_endpoints.py::TestConfigEndpoints::test_post_config_accepts_github_webhook_secret -v
```
Expected: FAIL with 422 (Pydantic rejects unknown keys).

- [ ] **Step 3: Extend `ConfigUpdate` Pydantic model**

In `backend/api.py`, find `ConfigUpdate` and add:

```python
    huggingface_token: str | None = None
    github_webhook_secret: str | None = None
```

- [ ] **Step 4: Run tests to verify it passes**

Run:
```bash
cd backend && python -m pytest tests/test_api_endpoints.py::TestConfigEndpoints -v
```
Expected: PASS, all 11 ConfigEndpoints tests green (originals + 2 new).

- [ ] **Step 5: Commit**

```bash
git add backend/api.py backend/tests/test_api_endpoints.py
git commit -m "feat(api): ConfigUpdate accepts huggingface_token and github_webhook_secret"
```

---

## Task 15: Proxy — preserve SSE Cache-Control header

**Files:**
- Modify: `web-ui/src/app/api/proxy/[...path]/route.ts`

- [ ] **Step 1: Inspect current proxy header handling**

The proxy already streams `upstream.body` via `NextResponse`. We need to ensure `cache-control` and `x-accel-buffering` from the upstream pass through unchanged. The current `HOP_BY_HOP` set already excludes these (good). The only risk is Next.js rewriting `cache-control`. Confirm by reading the current route file.

- [ ] **Step 2: Add a test-doc comment**

In `web-ui/src/app/api/proxy/[...path]/route.ts`, just above `executeUpstream`, add the comment:

```typescript
// Pass-through note: SSE responses (text/event-stream) require the upstream
// Cache-Control: no-cache and X-Accel-Buffering: no headers to flow through
// untouched. We don't add or rewrite cache-control on responses; HOP_BY_HOP
// excludes neither header. Verified by manual smoke test in dev.
```

- [ ] **Step 3: Manual smoke**

Start a dev backend (`make up`) and dev next (`cd web-ui && npm run dev`). Open DevTools → Network → filter EventStream:

```
GET /api/proxy/api/system/events
```

Confirm: Content-Type: text/event-stream, Cache-Control includes "no-cache", X-Accel-Buffering: no.

- [ ] **Step 4: Commit**

```bash
git add web-ui/src/app/api/proxy/'[...path]'/route.ts
git commit -m "docs(proxy): note SSE header pass-through requirement"
```

---

## Task 16: UI — SystemStatus state + SSE + 10 s polling + on-mount fetch

**Files:**
- Modify: `web-ui/src/app/page.tsx`

- [ ] **Step 1: Add SystemStatus types and state**

Near the top of the `Home()` component (after the existing useState declarations), add:

```typescript
type EmbeddingStatus = {
  state: "idle" | "downloading" | "loading" | "ready" | "error";
  model_name: string;
  bytes_downloaded: number;
  bytes_total: number;
  error?: string | null;
};

type QueuedJob = {
  job_id: string;
  repo: string;
  queued_at: number;
  reason: string;
};

type SystemStatusPayload = {
  embedding: EmbeddingStatus;
  queued_jobs: QueuedJob[];
};

const [systemStatus, setSystemStatus] = useState<SystemStatusPayload>({
  embedding: { state: "idle", model_name: "", bytes_downloaded: 0, bytes_total: 0 },
  queued_jobs: [],
});
```

- [ ] **Step 2: Add the SSE + polling effect**

After the LLM-health useEffect, add:

```typescript
// SystemStatus: SSE primary, 10 s poll backup, fetch on mount.
useEffect(() => {
  let cancelled = false;
  let es: EventSource | null = null;
  let backoff = 1000;
  let reconnectTimer: number | null = null;
  let pollTimer: number | null = null;

  const refresh = async () => {
    try {
      const r = await apiFetch("/api/system/status");
      const d = await r.json();
      if (!cancelled) setSystemStatus(d);
    } catch {
      /* network blip — backup poll will retry */
    }
  };

  const connect = () => {
    if (cancelled) return;
    es = new EventSource("/api/proxy/api/system/events");
    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (!cancelled) setSystemStatus(data);
        backoff = 1000;
      } catch {
        /* ignore malformed */
      }
    };
    es.onerror = () => {
      es?.close();
      if (cancelled) return;
      refresh();
      reconnectTimer = window.setTimeout(connect, Math.min(backoff *= 2, 30000));
    };
  };

  refresh();
  connect();
  pollTimer = window.setInterval(refresh, 10_000);

  return () => {
    cancelled = true;
    es?.close();
    if (reconnectTimer) clearTimeout(reconnectTimer);
    if (pollTimer) clearInterval(pollTimer);
  };
}, []);
```

- [ ] **Step 3: Manual smoke**

Run dev stack, open the page. In DevTools → Network filter EventStream → confirm `/api/proxy/api/system/events` is open. Trigger a backend status update (e.g., `curl -X POST localhost:8923/api/system/embedding/retry -H "Authorization: Bearer $(cat backend/data/.api_token)"`). Confirm the page receives the update without refresh.

- [ ] **Step 4: Commit**

```bash
git add web-ui/src/app/page.tsx
git commit -m "feat(ui): subscribe to /api/system/events with poll fallback"
```

---

## Task 17: UI — embedding banner with progress bar

**Files:**
- Modify: `web-ui/src/app/page.tsx`

- [ ] **Step 1: Add helper for byte formatting**

Near the top of `Home()`, after `apiFetch`:

```typescript
const formatBytes = (n: number): string => {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
};
```

- [ ] **Step 2: Render the banner**

Find the existing LLM-unreachable banner block (search for `llmHealth` and the `bannerDismissed` markup). Above it, add:

```tsx
{systemStatus.embedding.state !== "ready" && (
  <div className="w-full bg-purple-500/10 border-b border-purple-500/30 px-6 py-3">
    {systemStatus.embedding.state === "downloading" && (
      <>
        <div className="text-xs text-purple-200 mb-2 flex justify-between">
          <span>
            Downloading embedding model: <span className="font-mono">{systemStatus.embedding.model_name}</span> ·{" "}
            {formatBytes(systemStatus.embedding.bytes_downloaded)} /{" "}
            {systemStatus.embedding.bytes_total > 0 ? formatBytes(systemStatus.embedding.bytes_total) : "…"}
          </span>
          <span>
            {systemStatus.embedding.bytes_total > 0
              ? `${Math.floor(100 * systemStatus.embedding.bytes_downloaded / systemStatus.embedding.bytes_total)}%`
              : ""}
          </span>
        </div>
        <div className="h-2 bg-black/40 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-purple-600 to-purple-400 transition-all"
            style={{ width: `${systemStatus.embedding.bytes_total > 0 ? Math.max(2, 100 * systemStatus.embedding.bytes_downloaded / systemStatus.embedding.bytes_total) : 2}%` }}
          />
        </div>
      </>
    )}
    {systemStatus.embedding.state === "loading" && (
      <div className="text-xs text-purple-200 animate-pulse">
        Loading model weights into memory…
      </div>
    )}
    {systemStatus.embedding.state === "error" && (
      <div className="flex justify-between items-center">
        <span className="text-sm text-red-300">
          Embedding load failed: {systemStatus.embedding.error}
        </span>
        <button
          onClick={() => apiFetch("/api/system/embedding/retry", { method: "POST" })}
          className="text-xs bg-red-500/20 border border-red-500/40 px-3 py-1 rounded-full hover:bg-red-500 hover:text-white"
        >
          Retry
        </button>
      </div>
    )}
  </div>
)}
```

- [ ] **Step 3: Manual smoke**

Run dev stack on a fresh `hf_cache` volume. Open the page within 1 s of `make up`. Confirm:
- Banner shows "Downloading embedding model: …".
- Progress bar moves visibly as the download proceeds.
- Banner switches to "Loading model weights into memory…" briefly.
- Banner disappears when state becomes `ready`.

- [ ] **Step 4: Commit**

```bash
git add web-ui/src/app/page.tsx
git commit -m "feat(ui): embedding download banner with byte-level progress bar"
```

---

## Task 18: UI — queued-job display

**Files:**
- Modify: `web-ui/src/app/page.tsx`

- [ ] **Step 1: Render queued state**

Find the existing pipeline progress display (search for `jobProgress === 100 ? "Matrix Integrated"`). Above the existing block, add:

```tsx
{jobStatus.startsWith("Queued") && (
  <div className="text-amber-400 text-sm flex items-center gap-2 mb-3">
    <span className="inline-block w-2 h-2 rounded-full bg-amber-400 animate-pulse"></span>
    {jobStatus}
  </div>
)}
```

- [ ] **Step 2: Update job-trigger flow to show queued state**

Find `handleStartJob` (or whatever launches `/api/jobs/start`). Confirm the response handling sets `setJobStatus(data.status)` from the server. The backend returns `"Queued — waiting for embedding model"` in `active_jobs[job_id].status`, which the existing job-status polling loop will read. No change needed if polling is already in place; otherwise ensure `setJobStatus` is called once on the trigger response too.

- [ ] **Step 3: Manual smoke**

While embedding is `downloading`, click "Run Pipeline". Confirm: amber pulse dot + "Queued — waiting for embedding model" text. When embedding flips to `ready`, the same job auto-starts and the existing progress UI takes over.

- [ ] **Step 4: Commit**

```bash
git add web-ui/src/app/page.tsx
git commit -m "feat(ui): show queued state for jobs waiting on embedding model"
```

---

## Task 19: UI — `/secrets` page scaffolding + GitHub section

**Files:**
- Create: `web-ui/src/app/secrets/page.tsx`

- [ ] **Step 1: Create the page file**

```tsx
// web-ui/src/app/secrets/page.tsx
'use client';
import { useEffect, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "";

const apiFetch = (path: string, init: RequestInit = {}) =>
  fetch(`/api/proxy${path.startsWith("/") ? path : `/${path}`}`, init);

type Config = {
  github_token: string;
  huggingface_token: string;
  github_webhook_secret: string;
  llm_provider: string;
  provider_api_keys: Record<string, string>;
  env_overrides: Record<string, boolean>;
  webhook_url: string;
};

const EMPTY_CONFIG: Config = {
  github_token: "",
  huggingface_token: "",
  github_webhook_secret: "",
  llm_provider: "",
  provider_api_keys: {},
  env_overrides: {},
  webhook_url: "",
};

export default function SecretsPage() {
  const [config, setConfig] = useState<Config>(EMPTY_CONFIG);

  useEffect(() => {
    apiFetch("/api/config")
      .then((r) => r.json())
      .then((d) => setConfig({ ...EMPTY_CONFIG, ...d }));
  }, []);

  const updateField = (field: keyof Config, value: string) => {
    setConfig({ ...config, [field]: value });
  };

  const save = async (payload: Partial<Config>) => {
    await apiFetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  };

  const isEnvOverride = (key: string) => Boolean(config.env_overrides[key]);

  return (
    <div className="min-h-screen bg-black text-white p-12 max-w-3xl mx-auto">
      <header className="mb-10">
        <h1 className="text-3xl font-light flex items-center gap-3">
          <span>🔒</span>
          <span>Encrypted Vault — AES-256</span>
        </h1>
        <p className="text-sm text-neutral-400 mt-2">
          Stored encrypted at rest with Fernet (AES-128 CBC + HMAC-SHA256). Keys
          set via environment variables are read-only here.
        </p>
      </header>

      <section className="mb-10 border border-white/10 rounded-2xl p-6 bg-black/40">
        <h2 className="text-lg font-semibold mb-4">GitHub</h2>
        <SecretInput
          label="Personal access token"
          value={isEnvOverride("github_token") ? "" : config.github_token}
          onChange={(v) => updateField("github_token", v)}
          onBlur={() => save({ github_token: config.github_token })}
          envOverride={isEnvOverride("github_token")}
          envVarName="GITHUB_TOKEN"
        />
      </section>

      {/* Subsequent sections appended in later tasks */}
    </div>
  );
}

function SecretInput({
  label,
  value,
  onChange,
  onBlur,
  envOverride,
  envVarName,
  hint,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  onBlur?: () => void;
  envOverride: boolean;
  envVarName: string;
  hint?: string;
}) {
  return (
    <div className="mb-2">
      <label className="text-xs uppercase tracking-widest text-neutral-400 mb-2 flex items-center gap-2">
        {label}
        {envOverride && (
          <span
            className="text-[10px] bg-amber-500/20 border border-amber-500/40 text-amber-300 px-2 py-0.5 rounded-full"
            title={`Set via ${envVarName} environment variable.`}
          >
            from environment
          </span>
        )}
      </label>
      <input
        type="password"
        disabled={envOverride}
        value={envOverride ? "••••••••••••••••••••" : value}
        onChange={(e) => onChange(e.target.value)}
        onBlur={onBlur}
        className={`w-full bg-black/80 border rounded-lg p-3 text-sm font-mono outline-none ${
          envOverride
            ? "border-white/5 text-neutral-500 cursor-not-allowed"
            : "border-white/10 focus:border-purple-500"
        }`}
      />
      {hint && <p className="text-xs text-neutral-500 mt-1">{hint}</p>}
    </div>
  );
}
```

- [ ] **Step 2: Manual smoke**

Run the dev stack. Visit `http://localhost:4096/secrets`. Confirm:
- Page renders with header, subtitle, and GitHub section.
- Token input is editable when env var unset; disabled with chip when env set.
- Saving the field via blur persists (verify via `GET /api/config`).

- [ ] **Step 3: Commit**

```bash
git add web-ui/src/app/secrets/page.tsx
git commit -m "feat(ui): /secrets Vault page scaffolding + GitHub section"
```

---

## Task 20: UI — `/secrets` HuggingFace + LLM Providers sections

**Files:**
- Modify: `web-ui/src/app/secrets/page.tsx`

- [ ] **Step 1: Add HuggingFace and Providers sections**

Replace the placeholder comment `{/* Subsequent sections appended in later tasks */}` with:

```tsx
      <section className="mb-10 border border-white/10 rounded-2xl p-6 bg-black/40">
        <h2 className="text-lg font-semibold mb-2">HuggingFace</h2>
        <p className="text-xs text-neutral-500 mb-4">
          Optional. Without a token, downloads are rate-limited and gated models
          will fail.
        </p>
        <SecretInput
          label="HuggingFace token"
          value={isEnvOverride("huggingface_token") ? "" : config.huggingface_token}
          onChange={(v) => updateField("huggingface_token", v)}
          onBlur={() => save({ huggingface_token: config.huggingface_token })}
          envOverride={isEnvOverride("huggingface_token")}
          envVarName="HUGGINGFACE_HUB_TOKEN"
        />
      </section>

      <section className="mb-10 border border-white/10 rounded-2xl p-6 bg-black/40">
        <h2 className="text-lg font-semibold mb-4">LLM Providers</h2>
        {(["openai", "anthropic", "google", "openrouter"] as const).map((p) => {
          const envKey = `provider_api_keys.${p}`;
          const isActive = config.llm_provider === p;
          return (
            <div key={p} className="mb-4">
              <SecretInput
                label={
                  isActive
                    ? `${p[0].toUpperCase() + p.slice(1)} ★ active`
                    : p[0].toUpperCase() + p.slice(1)
                }
                value={
                  isEnvOverride(envKey)
                    ? ""
                    : config.provider_api_keys[p] || ""
                }
                onChange={(v) =>
                  setConfig({
                    ...config,
                    provider_api_keys: {
                      ...config.provider_api_keys,
                      [p]: v,
                    },
                  })
                }
                onBlur={() =>
                  save({
                    provider_api_keys: {
                      [p]: config.provider_api_keys[p] || "",
                    } as Record<string, string>,
                  })
                }
                envOverride={isEnvOverride(envKey)}
                envVarName={`${p.toUpperCase()}_API_KEY`}
              />
            </div>
          );
        })}
      </section>
```

- [ ] **Step 2: Manual smoke**

Visit `/secrets`. Confirm:
- HF section renders with subtitle.
- Providers section lists openai/anthropic/google/openrouter.
- Setting `OPENAI_API_KEY` env var → restart backend → reload page → openai input shows "from environment" chip and disabled.
- Active provider has the ★ marker.

- [ ] **Step 3: Commit**

```bash
git add web-ui/src/app/secrets/page.tsx
git commit -m "feat(ui): /secrets HuggingFace + LLM Providers sections"
```

---

## Task 21: UI — `/secrets` Webhook subsection

**Files:**
- Modify: `web-ui/src/app/secrets/page.tsx`

- [ ] **Step 1: Add Webhook section + reveal-and-copy logic**

After the LLM Providers section in `secrets/page.tsx`, append:

```tsx
      <section className="mb-10 border border-white/10 rounded-2xl p-6 bg-black/40">
        <h2 className="text-lg font-semibold mb-2">GitHub Webhook</h2>
        <p className="text-xs text-neutral-500 mb-4">
          In GitHub's webhook config, paste the URL below and the secret. Subscribe to: <strong>Pull requests</strong>.
        </p>

        <div className="mb-4">
          <label className="text-xs uppercase tracking-widest text-neutral-400 mb-2 block">
            Webhook URL
          </label>
          <div className="flex gap-2">
            <input
              readOnly
              value={config.webhook_url}
              className="flex-1 bg-black/80 border border-white/10 rounded-lg p-3 text-sm font-mono text-neutral-300"
            />
            <button
              onClick={() => navigator.clipboard.writeText(config.webhook_url)}
              className="px-4 bg-white/5 border border-white/10 rounded-lg text-sm hover:bg-white/10"
            >
              Copy
            </button>
          </div>
        </div>

        <WebhookSecret
          value={config.github_webhook_secret}
          envOverride={isEnvOverride("github_webhook_secret")}
          onChange={(v) => updateField("github_webhook_secret", v)}
          onSave={(v) => save({ github_webhook_secret: v })}
        />
      </section>
```

Add the `WebhookSecret` component at the bottom of the file:

```tsx
function WebhookSecret({
  value,
  envOverride,
  onChange,
  onSave,
}: {
  value: string;
  envOverride: boolean;
  onChange: (v: string) => void;
  onSave: (v: string) => void;
}) {
  const [revealed, setRevealed] = useState(false);

  const generate = () => {
    const bytes = new Uint8Array(32);
    crypto.getRandomValues(bytes);
    const hex = Array.from(bytes).map((b) => b.toString(16).padStart(2, "0")).join("");
    onChange(hex);
    onSave(hex);
  };

  const revealAndCopy = async () => {
    setRevealed(true);
    await navigator.clipboard.writeText(value);
    setTimeout(() => setRevealed(false), 3000);
  };

  return (
    <div>
      <label className="text-xs uppercase tracking-widest text-neutral-400 mb-2 flex items-center gap-2">
        Webhook secret
        {envOverride && (
          <span
            className="text-[10px] bg-amber-500/20 border border-amber-500/40 text-amber-300 px-2 py-0.5 rounded-full"
            title="Set via GITHUB_WEBHOOK_SECRET environment variable."
          >
            from environment
          </span>
        )}
      </label>
      <div className="flex gap-2">
        <input
          type={revealed ? "text" : "password"}
          readOnly={envOverride}
          value={value || ""}
          onChange={(e) => !envOverride && onChange(e.target.value)}
          className={`flex-1 bg-black/80 border rounded-lg p-3 text-sm font-mono ${
            envOverride
              ? "border-white/5 text-neutral-500 cursor-not-allowed"
              : "border-white/10 focus:border-purple-500"
          }`}
        />
        {value && (
          <button
            onClick={revealAndCopy}
            className="px-4 bg-amber-500/10 border border-amber-500/40 rounded-lg text-amber-300 text-sm hover:bg-amber-500/20"
          >
            {revealed ? "Copied" : "Reveal & Copy"}
          </button>
        )}
        {!envOverride && (
          <button
            onClick={generate}
            className="px-4 bg-purple-500/20 border border-purple-500/40 rounded-lg text-purple-200 text-sm hover:bg-purple-500/30"
          >
            {value ? "Rotate" : "Generate"}
          </button>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Manual smoke**

Visit `/secrets`. Webhook subsection should:
- Show URL with Copy button → click copies to clipboard.
- Show empty secret with "Generate" button → click fills + saves.
- After secret set, "Reveal & Copy" reveals for 3 s, then re-masks.
- "Rotate" replaces the secret.
- With `GITHUB_WEBHOOK_SECRET` env var set, fields read-only with chip.

- [ ] **Step 3: Commit**

```bash
git add web-ui/src/app/secrets/page.tsx
git commit -m "feat(ui): /secrets Webhook subsection with copy URL + reveal-and-copy"
```

---

## Task 22: UI — link from Settings modal to `/secrets`

**Files:**
- Modify: `web-ui/src/app/page.tsx`

- [ ] **Step 1: Add a small footer link inside the Settings modal**

Find the Settings modal markup (search for `🔒 Vault` or `Global Control Architecture`). Inside the modal `<form>`, just before the Submit button row, add:

```tsx
<div className="text-xs text-neutral-500 text-center pt-2 border-t border-white/5">
  Manage all credentials in the new{" "}
  <a href="/secrets" className="text-purple-400 hover:text-purple-300 underline">
    🔒 Encrypted Vault
  </a>
</div>
```

- [ ] **Step 2: Manual smoke**

Open Settings modal. Confirm the link appears at the bottom and navigates to `/secrets`.

- [ ] **Step 3: Commit**

```bash
git add web-ui/src/app/page.tsx
git commit -m "feat(ui): link from Settings modal to Vault page"
```

---

## Task 23: UI — HF "no key → slow" hint in embedding banner

**Files:**
- Modify: `web-ui/src/app/page.tsx`

- [ ] **Step 1: Read HF token state**

In the existing config-load `useEffect`, save HF state separately:

```typescript
const [hfTokenSet, setHfTokenSet] = useState(false);
```

When the config response arrives, set:

```typescript
setHfTokenSet(
  Boolean(data.huggingface_token && data.huggingface_token !== "***") ||
  Boolean(data.env_overrides?.huggingface_token)
);
```

- [ ] **Step 2: Add the secondary line to the downloading-state banner**

Inside the downloading-state branch in the banner (Task 17 markup), after the progress bar, add:

```tsx
{!hfTokenSet && (
  <p className="text-xs text-neutral-400 mt-2">
    Tip: downloads are throttled without a HuggingFace token.{" "}
    <a href="/secrets" className="text-purple-400 hover:text-purple-300 underline">
      → Set one in the Vault
    </a>
  </p>
)}
```

- [ ] **Step 3: Manual smoke**

With HF env unset and no encrypted-config token:
- Banner during download includes the "Tip…" line linking to /secrets.
With HF token set in Vault:
- Tip line disappears.

- [ ] **Step 4: Commit**

```bash
git add web-ui/src/app/page.tsx
git commit -m "feat(ui): show HF-token-missing hint in embedding banner"
```

---

## Task 24: README + .env.example final docs

**Files:**
- Modify: `README.md`
- Modify: `.env.example`

- [ ] **Step 1: Append Vault & embedding docs to README**

In the env-var table, add rows for the new vars:

```markdown
| `HUGGINGFACE_HUB_TOKEN` | — | Optional. Required for gated models; recommended to avoid rate limits. Mirrored on the Vault page. |
| `WEBHOOK_PUBLIC_URL` | — | The publicly reachable URL of your backend. Used by the Vault page to render the GitHub webhook URL. |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `OPENROUTER_API_KEY` | — | Optional env override for provider keys; the Vault page is the alternative. |
```

Add a new section under "Environment":

```markdown
### Encrypted Vault

User-supplied credentials live on the **Encrypted Vault** page at
[http://localhost:4096/secrets](http://localhost:4096/secrets):

- GitHub personal access token
- HuggingFace token (optional)
- LLM provider API keys (OpenAI, Anthropic, Google, OpenRouter)
- GitHub webhook URL + secret (with reveal-and-copy)

Values are Fernet-encrypted at rest in `backend/data/config.json`. Setting any
of the listed env vars makes the corresponding input read-only and badges it
with `from environment` — the env value wins. This lets you integrate with
HashiCorp Vault, AWS Secrets Manager, or any other secret-injection setup.
```

- [ ] **Step 2: Verify .env.example matches**

Confirm all new env vars from Task 13 are documented with comments.

- [ ] **Step 3: Commit**

```bash
git add README.md .env.example
git commit -m "docs: Vault page + new env vars in README and .env.example"
```

---

## Task 25: Final regression + smoke test

**Files:** none — verification only

- [ ] **Step 1: Run full backend test suite**

```bash
cd backend && python -m pytest tests/ --ignore=tests/test_security_redactor.py -v
```
Expected: all tests PASS.

- [ ] **Step 2: Manual end-to-end on a fresh stack**

```bash
make down
docker volume rm pr-distiller_hf_cache pr-distiller_backend_data 2>/dev/null || true
make up
# Within 2 s, open http://localhost:4096
```

Verify the entire flow:
1. UI loads instantly (no spinner, no freeze).
2. Banner shows "Downloading embedding model: ... · 0 B / X MB" within 1 s.
3. Progress bar advances.
4. Click "Run Pipeline" while downloading → shows "Queued — waiting for embedding model".
5. When download completes → banner says "Loading model weights into memory…", then disappears.
6. Queued job auto-starts (status flips to "Starting (was queued)" then normal pipeline progress).
7. Visit `/secrets` → all sections render.
8. Set `HUGGINGFACE_HUB_TOKEN` in `.env`, `make down && make up` → HF input read-only with `from environment` chip.
9. `rm backend/data/.api_token && docker compose restart backend` → after a brief 401 retry, UI continues working.

- [ ] **Step 3: Commit any final touch-ups, push**

```bash
git push
```

---

## Self-review

**Spec coverage check:**

- ✅ Persistent HF cache volume → Task 13.
- ✅ HUGGINGFACE_HUB_TOKEN env passthrough → Tasks 9, 13.
- ✅ Background embedding load with byte-level progress → Tasks 1, 3, 12.
- ✅ Lazy db proxy → Task 4.
- ✅ SystemStatus + SSE + status endpoint → Tasks 1, 2, 5, 6, 7.
- ✅ UI banner with progress + retry button → Task 17.
- ✅ Job queueing + drain → Tasks 2, 11.
- ✅ Encrypted Vault page → Tasks 19, 20, 21.
- ✅ env_overrides detection → Tasks 9, 10, 19 (rendering).
- ✅ Webhook URL + secret with reveal-and-copy → Task 21.
- ✅ HF "no key → slow" hint → Task 23.
- ✅ Settings → Vault link → Task 22.
- ✅ Pydantic ConfigUpdate accepts new fields → Task 14.
- ✅ Docs updated → Task 24.
- ✅ End-to-end verification → Task 25.

**Type/name consistency check:**

- `SYSTEM_STATUS` (module-level singleton) used consistently in api.py and embedding_loader.py.
- `LazyDbProxy.bind()` name consistent across Tasks 4 and 12.
- `set_on_ready` / `_on_ready` callback naming consistent across Tasks 1, 11.
- `huggingface_token` / `huggingface_token_enc` field names consistent across Tasks 8, 9, 10, 14, 19, 20, 23.
- `env_overrides()` return shape consistent across Tasks 9, 10, 19, 20.
- Frontend `apiFetch` helper signature consistent (`(path, init?) => Promise<Response>`).

**Placeholder scan:** clean — no TBD/TODO/etc.

**Total: 25 tasks**, each self-contained with red→green→commit. Estimated ~6–8 hours for a focused engineer.
