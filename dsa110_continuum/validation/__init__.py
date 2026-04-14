"""Validation utilities for dsa110_contimg package."""

try:
    from dsa110_contimg.core.validation.package_health import run_diagnostics
except ImportError:
    pass  # dsa110_contimg not installed (cloud/test env)

__all__ = ["run_diagnostics"]
