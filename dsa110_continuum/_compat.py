"""Compatibility stubs for dsa110_contimg symbols used across the package.

When dsa110_contimg is not installed (cloud/CI environments), these no-op
implementations allow modules to import and unit tests to run without CASA/H17.

On H17 with dsa110_contimg installed, these stubs are never loaded — the real
implementations take precedence via the try/except guards in each module.
"""
from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


# ── Performance tracking stub ─────────────────────────────────────────────────


def track_performance(name: str = "", log_result: bool = False) -> Callable[[F], F]:
    """No-op stub for dsa110_contimg.common.utils.performance.track_performance."""
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)
        return wrapper  # type: ignore[return-value]
    return decorator


# ── Memory safety stub ────────────────────────────────────────────────────────


def memory_safe(max_gb: float = 0, **kwargs: Any) -> Callable[[F], F]:
    """No-op stub for dsa110_contimg.common.utils.memory_safe.memory_safe."""
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs_inner: Any) -> Any:
            return fn(*args, **kwargs_inner)
        return wrapper  # type: ignore[return-value]
    return decorator


# ── timed / timed_debug stubs ─────────────────────────────────────────────────


def timed(
    fn_or_name: F | str | None = None,
    **kwargs: Any,
) -> Any:
    """No-op stub for dsa110_contimg.common.utils.timed.

    Supports both ``@timed`` and ``@timed("label")`` usage patterns.
    """
    def decorator(f: F) -> F:
        @functools.wraps(f)
        def wrapper(*args: Any, **kw: Any) -> Any:
            return f(*args, **kw)
        return wrapper  # type: ignore[return-value]

    if callable(fn_or_name):
        # Called as @timed — fn_or_name is the decorated function
        return decorator(fn_or_name)  # type: ignore[arg-type]
    # Called as @timed(...) or @timed("label") — return the decorator
    return decorator


def timed_debug(
    fn_or_name: F | str | None = None,
    **kwargs: Any,
) -> Any:
    """No-op stub for dsa110_contimg.common.utils.timed_debug.

    Supports both ``@timed_debug`` and ``@timed_debug("label")`` usage patterns.
    """
    def decorator(f: F) -> F:
        @functools.wraps(f)
        def wrapper(*args: Any, **kw: Any) -> Any:
            return f(*args, **kw)
        return wrapper  # type: ignore[return-value]

    if callable(fn_or_name):
        return decorator(fn_or_name)  # type: ignore[arg-type]
    return decorator


# ── GPU safety stubs ──────────────────────────────────────────────────────────


def gpu_safe(
    fn: F | None = None,
    *,
    require_gpu: bool = False,
    fallback: Any = None,
    **kwargs: Any,
) -> Any:
    """No-op stub for dsa110_contimg gpu_safe.

    Supports both ``@gpu_safe`` (plain decorator) and ``@gpu_safe()``
    (factory / keyword-argument call) usage patterns.
    """
    def decorator(f: F) -> F:
        @functools.wraps(f)
        def wrapper(*args: Any, **kw: Any) -> Any:
            return f(*args, **kw)
        return wrapper  # type: ignore[return-value]

    if fn is not None:
        # Called as @gpu_safe — fn is the decorated function
        return decorator(fn)
    # Called as @gpu_safe(...) — return the decorator
    return decorator


def safe_gpu_context(*args: Any, **kwargs: Any):
    """No-op stub for dsa110_contimg safe_gpu_context context manager."""
    from contextlib import contextmanager

    @contextmanager
    def _cm():
        yield

    return _cm()


def require_casa6_python(fn: F) -> F:
    """Identity decorator stub — passes through the function unchanged.

    On H17 the real implementation raises if called outside the CASA
    Python environment; here we allow tests to import and call the function
    with appropriate mocking.
    """
    return fn


def get_gpu_config() -> Any:
    """Return a stub GPU config indicating no GPU available."""
    return type("GPUConfig", (), {"has_gpu": False, "device_count": 0})()


def initialize_gpu_safety() -> None:
    """No-op stub."""


# ── Path/env utilities ────────────────────────────────────────────────────────


# ── DSA-110 site location ────────────────────────────────────────────────────


# DSA-110 is located at OVRO: 37.2339°N, 118.2825°W, 1222 m
# (matches docs/pipeline-specs.md and dsa110_continuum/calibration/runner.py)
def _make_dsa110_location():
    try:
        import astropy.units as _u
        from astropy.coordinates import EarthLocation as _EL
        return _EL(lat=37.2339 * _u.deg, lon=-118.2825 * _u.deg, height=1222 * _u.m)
    except ImportError:  # astropy not installed
        return None


DSA110_LOCATION = _make_dsa110_location()


def get_env_path(key: str, default: str = "") -> str:
    """Stub for dsa110_contimg path helpers."""
    import os
    return os.environ.get(key, default)


def get_repo_root():
    """Return workspace root as a Path."""
    from pathlib import Path
    return Path(__file__).resolve().parent.parent


# ── Casa environment stubs ────────────────────────────────────────────────────


@contextmanager
def casa_log_environment(*args: Any, **kwargs: Any):
    """No-op stub for CASA log context manager."""
    yield


def get_casa_task(name: str) -> Any:
    """Raise a clear error when CASA tasks are called without the runtime."""
    raise RuntimeError(
        f"CASA task '{name}' unavailable: dsa110_contimg not installed. "
        "Run on H17 with /opt/miniforge/envs/casa6/bin/python."
    )


# ── Validation error stubs ────────────────────────────────────────────────────


class ValidationError(Exception):
    """Stub for dsa110_contimg ValidationError."""


def validate_ms(ms_path: str, **kwargs: Any) -> bool:
    """Stub that raises if casacore is unavailable."""
    try:
        from dsa110_continuum.adapters import casa_tables  # noqa: F401
    except ImportError:
        raise RuntimeError("casatools not installed — cannot validate MS")
    return True


# ── Antenna classification stub ───────────────────────────────────────────────


def classify_antenna(ant_id: int) -> str:
    """Stub returning 'unknown' for all antennas."""
    return "unknown"


def get_outrigger_antenna_ids() -> list[int]:
    return list(range(103, 118))


def get_core_antenna_ids() -> list[int]:
    return list(range(0, 103))
