"""Which implementation runs the neighbour phase of a step: the C++ kernel
``vicsek._kernel`` or the numpy cell list in ``vicsek.util.cell_list``.

Both give bit-for-bit identical results (see ``tests/test_kernel.py``); the
kernel is several times faster. It is used whenever it was built, unless the
environment variable ``VICSEK_NO_KERNEL`` is set to ``1`` (or ``true``,
``yes``, ``on``) when vicsek is first imported, or it is switched off with
``use_kernel(False)``. Installs without a C++ compiler have no kernel and
always use numpy.
"""
import os
from typing import Optional

try:
    from vicsek import _kernel
except ImportError:
    _kernel = None

_enabled = _kernel is not None and os.environ.get("VICSEK_NO_KERNEL", "").strip().lower() not in {"1", "true", "yes", "on"}
_threads = 1


def kernel_available() -> bool:
    """Whether the C++ kernel was built and imports."""
    return _kernel is not None


def kernel_enabled() -> bool:
    """Whether steps currently use the C++ kernel."""
    return _enabled


def kernel_threads() -> int:
    """Number of threads the kernel uses (default 1)."""
    return _threads


class use_kernel:
    """Switch the C++ kernel on or off, and optionally set its thread count.

    Takes effect immediately and stays in effect::

        vicsek.use_kernel(False)              # numpy from now on
        vicsek.use_kernel(True, threads=4)    # kernel, 4 threads

    or only inside a ``with`` block, restoring the previous setting after::

        with vicsek.use_kernel(False):
            model.step()

    ``use_kernel(True)`` raises ``RuntimeError`` when the kernel is not
    available. ``threads`` must be at least 1; the result does not depend on it.
    """

    def __init__(self, enabled: bool = True, threads: Optional[int] = None) -> None:
        global _enabled, _threads
        if enabled and _kernel is None:
            raise RuntimeError("the C++ kernel vicsek._kernel is not available in this install "
                               "(built without a C++ compiler?); the numpy path is in use")
        if threads is not None and int(threads) < 1:
            raise ValueError("threads must be at least 1")
        self._previous = (_enabled, _threads)
        _enabled = bool(enabled)
        if threads is not None:
            _threads = int(threads)

    def __enter__(self) -> "use_kernel":
        return self

    def __exit__(self, *exc) -> None:
        global _enabled, _threads
        _enabled, _threads = self._previous


def kernel():
    """The kernel module if it is enabled, else None."""
    return _kernel if _enabled else None
