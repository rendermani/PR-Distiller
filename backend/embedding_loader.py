"""Hooks `huggingface_hub` and `sentence_transformers` tqdm so we can publish
live byte-level download progress into SystemStatus.

Use as `with capture_hf_progress(): ...` around the model load. The hook is
installed by monkey-patching the `tqdm` attribute on each library's progress
module; the original is restored on exit (success or exception).

Implementation note — huggingface_hub import resolution
--------------------------------------------------------
``huggingface_hub/utils/__init__.py`` re-exports the ``tqdm`` class as a
package-level attribute, so ``import huggingface_hub.utils.tqdm as x`` binds
``x`` to the *class*. We deliberately do NOT overwrite that package attribute:
``huggingface_hub.file_download`` does ``from .utils import tqdm`` and uses the
result in a ``tqdm | None`` type annotation that blows up if ``tqdm`` is a
module. We grab the actual submodule via ``sys.modules`` and patch
``module.tqdm`` directly — that's the attribute hf's download code reads.
"""
import contextlib
import importlib
import sys
from typing import ClassVar

import tqdm.auto
from system_status import SYSTEM_STATUS

# Force-import the submodule so it's resolvable by sys.modules below.  We can't
# `import huggingface_hub.utils.tqdm` as a regular import, because the parent
# package's __init__ shadows the submodule attribute with a re-exported class.
importlib.import_module("huggingface_hub.utils.tqdm")
_hf_tqdm_module = sys.modules["huggingface_hub.utils.tqdm"]


class _ProgressCapturingTqdm(tqdm.auto.tqdm):
    """tqdm subclass that pushes aggregate byte progress into SYSTEM_STATUS.

    All live instances share the class-level ``_live`` set.  Each ``update()``
    and ``close()`` recomputes the aggregate across every live instance and
    writes it to ``SYSTEM_STATUS.update_download_progress``.
    """

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
        """Recompute aggregate totals and push to SYSTEM_STATUS."""
        bytes_downloaded = sum(int(t.n or 0) for t in cls._live)
        bytes_total = sum(int(t.total or 0) for t in cls._live)
        SYSTEM_STATUS.update_download_progress(bytes_downloaded, bytes_total)


@contextlib.contextmanager
def capture_hf_progress():
    """Patch tqdm in huggingface_hub (and sentence_transformers if present).

    Installs ``_ProgressCapturingTqdm`` as the tqdm implementation used by
    huggingface_hub's download utilities for the duration of the block.
    Restores the originals on exit, whether the block raises or not.
    """
    try:
        import sentence_transformers.util as st_util_mod
    except ImportError:  # pragma: no cover — sentence_transformers is optional
        st_util_mod = None

    original_hf = _hf_tqdm_module.tqdm
    original_st = getattr(st_util_mod, "tqdm", None) if st_util_mod else None
    try:
        _hf_tqdm_module.tqdm = _ProgressCapturingTqdm
        if st_util_mod is not None:
            st_util_mod.tqdm = _ProgressCapturingTqdm
        yield
    finally:
        _hf_tqdm_module.tqdm = original_hf
        if st_util_mod is not None and original_st is not None:
            st_util_mod.tqdm = original_st
        _ProgressCapturingTqdm._live.clear()
