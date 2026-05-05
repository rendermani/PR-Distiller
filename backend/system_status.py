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


# Module-level singleton used by api.py and tests.
SYSTEM_STATUS = SystemStatus()
