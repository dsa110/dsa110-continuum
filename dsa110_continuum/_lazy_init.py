"""Lazy initialization for CASA and GPU environments.

Defers import-time side effects (os.environ mutations, casatools loading,
GPU safety checks) until the first function that actually needs them runs.
Each guard is thread-safe, idempotent, and essentially free after first call.

Usage — replace this pattern at module level::

    from dsa110_contimg.common.utils.casa_init import ensure_casa_path
    ensure_casa_path()

with a call inside the functions that need CASA::

    from dsa110_continuum._lazy_init import require_casa

    def my_function():
        require_casa()
        ...
"""

import threading

# ── CASA environment ──────────────────────────────────────────────────

_casa_lock = threading.Lock()
_casa_ready = False


def require_casa() -> None:
    """Ensure CASA environment is initialized (lazy, thread-safe, idempotent)."""
    global _casa_ready
    if _casa_ready:
        return
    with _casa_lock:
        if _casa_ready:
            return
        try:
            from dsa110_contimg.common.utils.casa_init import ensure_casa_path
            ensure_casa_path()
        except ImportError:
            pass  # dsa110_contimg not installed
        _casa_ready = True


# ── GPU safety ────────────────────────────────────────────────────────

_gpu_lock = threading.Lock()
_gpu_ready = False


def require_gpu_safety() -> None:
    """Ensure GPU safety limits are initialized (lazy, thread-safe, idempotent)."""
    global _gpu_ready
    if _gpu_ready:
        return
    with _gpu_lock:
        if _gpu_ready:
            return
        try:
            from dsa110_contimg.common.utils.gpu_safety import initialize_gpu_safety
            initialize_gpu_safety()
        except ImportError:
            pass  # dsa110_contimg not installed
        _gpu_ready = True


# ── Headless display ──────────────────────────────────────────────────

_headless_lock = threading.Lock()
_headless_ready = False


def require_headless() -> None:
    """Ensure headless display environment for CASA (lazy, idempotent).

    Suppresses casaplotserver X server errors by removing DISPLAY and
    setting QT_QPA_PLATFORM=offscreen.
    """
    global _headless_ready
    if _headless_ready:
        return
    with _headless_lock:
        if _headless_ready:
            return
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.environ.setdefault("CASA_NO_X", "1")
        if os.environ.get("DISPLAY"):
            os.environ.pop("DISPLAY", None)
        _headless_ready = True
