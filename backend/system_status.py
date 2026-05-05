"""Singleton holding embedding-load state, download progress, and a job queue.

Mutating methods take a threading lock because the embedding loader runs in a
background thread; subscribers are asyncio.Queue instances belonging to SSE
handlers on the FastAPI event loop.

Thread-safety contract for setup fields
----------------------------------------
``attach_loop`` and ``set_on_ready`` MUST be called during FastAPI startup,
before any background work (e.g. the embedding loader thread) begins.

``_loop`` and ``_on_ready`` are therefore written exactly once from the startup
thread and are only read thereafter — they are intentionally not lock-guarded.
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
        # SSE handlers must call unsubscribe() in their finally block; otherwise queues leak.
        self._subscribers: list[asyncio.Queue] = []
        # set once at startup; readers don't lock (see module docstring).
        self._loop: asyncio.AbstractEventLoop | None = None
        # set once at startup; readers don't lock (see module docstring).
        self._on_ready: Callable[[], None] | None = None
        self._last_progress_push = 0.0
        self._last_progress_pct = -1.0  # sentinel: ensures first call always passes the >=1% threshold
        # Full payload+config for pending jobs, kept off the snapshot (may contain secrets).
        self._pending: dict[str, tuple[dict, dict]] = {}

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
        with self._lock:
            snapshot = copy.deepcopy(self._state)
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
            # Loop already closed — nothing to enqueue.
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

    # --- job queueing ---

    def is_ready(self) -> bool:
        with self._lock:
            return self._state["embedding"]["state"] == "ready"

    def enqueue_job(self, job_id: str, payload: dict, config: dict, *, reason: str) -> None:
        """Append a pending job to the queue.

        The snapshot entry contains only non-secret fields (job_id, repo,
        queued_at, reason).  Full payload and config — which may carry tokens —
        are stored in ``_pending`` and excluded from every snapshot.
        """
        entry = {
            "job_id": job_id,
            "repo": payload.get("repo", ""),
            "queued_at": time.time(),
            "reason": reason,
        }
        with self._lock:
            self._state["queued_jobs"].append(entry)
            self._pending[job_id] = (payload, config)
        self._fanout()

    def drain_queue(self) -> list[tuple[str, dict, dict]]:
        """Remove and return all pending jobs as (job_id, payload, config) triples."""
        with self._lock:
            entries = self._state["queued_jobs"]
            self._state["queued_jobs"] = []
            drained = [
                (entry["job_id"], *self._pending.pop(entry["job_id"], ({}, {})))
                for entry in entries
            ]
        self._fanout()
        return drained

    def dequeue_job(self, job_id: str) -> bool:
        """Remove a single job by id.  Returns True if found and removed."""
        with self._lock:
            before = len(self._state["queued_jobs"])
            self._state["queued_jobs"] = [
                j for j in self._state["queued_jobs"] if j["job_id"] != job_id
            ]
            removed = before != len(self._state["queued_jobs"])
            if removed:
                self._pending.pop(job_id, None)
        if removed:
            self._fanout()
        return removed


# Module-level singleton used by api.py and tests.
SYSTEM_STATUS = SystemStatus()
