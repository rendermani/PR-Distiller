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
