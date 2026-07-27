"""Lazy proxy for LightRAGManager so api.py can be imported without blocking
on the embedding-model download. Attribute access blocks on a threading event
until bind() is called with the real manager, then raises TimeoutError."""
import threading
from typing import Any

# Long enough to cover a cold embedding-model download on a slow connection,
# short enough that a proxy nobody will ever bind fails with a usable error
# instead of parking the calling thread for the process lifetime.
DEFAULT_READY_TIMEOUT_SECONDS = 300.0


class LazyDbProxy:
    def __init__(self, ready_timeout: float | None = None) -> None:
        self._real: Any = None
        self._ready = threading.Event()
        # None means "use the module default, resolved on each wait". Deferring
        # the lookup lets the test suite shorten the default even for proxies
        # constructed at import time, such as api.db.
        self._explicit_timeout = ready_timeout

    @property
    def _ready_timeout(self) -> float:
        if self._explicit_timeout is not None:
            return self._explicit_timeout
        return DEFAULT_READY_TIMEOUT_SECONDS

    def bind(self, real: Any) -> None:
        if self._real is not None:
            raise RuntimeError("LazyDbProxy already bound")
        self._real = real
        self._ready.set()

    def __getattr__(self, name: str) -> Any:
        # __getattr__ is only called when normal attribute lookup fails, so
        # `_real` and `_ready` are reached via __dict__ without recursion.
        #
        # The wait is bounded. An unbounded wait made "bind() never happens"
        # indistinguishable from "the model is still loading": the calling
        # thread blocked forever with no diagnostic, which under TestClient
        # deadlocked the request and took the whole test process down with it.
        if not self._ready.wait(self._ready_timeout):
            raise TimeoutError(
                f"LazyDbProxy: attribute {name!r} requested but the proxy was "
                f"never bound within {self._ready_timeout}s. The embedding model "
                "failed to load or bind() was never called — check "
                "/api/system/status for the loader state."
            )
        return getattr(self._real, name)
