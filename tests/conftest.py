"""
pytest conftest for dsa110-continuum cloud/CI test environment.

Installs minimal mock stubs for packages that are only available on H17
(casacore, casa6, etc.) so that tests can import and mock them without the
real binary dependencies.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _install_casacore_mock() -> None:
    """Install a minimal casacore mock into sys.modules.

    Only installed when the real casacore is absent.  Tests that need to
    exercise casacore behaviour should patch the specific symbols they use
    (e.g. ``patch("casacore.tables.table", ...)``).
    """
    try:
        import casacore  # noqa: F401 — already installed, nothing to do
        return
    except ImportError:
        pass

    casacore_mod = types.ModuleType("casacore")
    casacore_tables = types.ModuleType("casacore.tables")
    casacore_quanta = types.ModuleType("casacore.quanta")
    casacore_measures = types.ModuleType("casacore.measures")

    # Minimal table stub: behaves like casacore.tables.table
    class _TableInstance(MagicMock):
        """Instance returned by table() — context-manager-compatible."""

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def colnames(self):
            return []  # default: no columns

        def nrows(self):
            return 0

    def _table_factory(path, *args, **kwargs):
        """Raise TypeError for non-string paths, OSError for non-existent paths.

        Mirrors real casacore.tables.table behaviour:
        - Non-string argument → TypeError
        - Non-existent path on disk → OSError (RuntimeError in some casacore builds)
        - Existing path → returns a context-manager-compatible _TableInstance

        Tests that need specific column data should patch
        ``casacore.tables.table`` with a MagicMock configured for their use case.
        """
        import os as _os
        if not isinstance(path, str):
            raise TypeError(
                f"casacore.tables.table: expected str path, got {type(path).__name__!r}"
            )
        # Strip casacore subtable suffix (e.g. "foo.ms::FIELD" → "foo.ms")
        base_path = path.split("::")[0]
        if not _os.path.exists(base_path):
            raise OSError(
                f"casacore.tables.table: path does not exist: {base_path!r}"
            )
        return _TableInstance()

    def _default_ms(path, *args, **kwargs):
        """Stub default_ms — this function requires real casacore to create a
        proper Measurement Set on disk.  Any test that calls default_ms is
        intentionally marked as requiring casacore and will be skipped."""
        import pytest as _pytest
        _pytest.skip("casacore stub: default_ms requires real casacore (H17/casa6 env)")

    casacore_tables.table = _table_factory
    casacore_tables.default_ms = _default_ms
    # Mark as stub so tests can detect it programmatically if needed
    casacore_tables._is_stub = True

    # Wire up the module hierarchy
    casacore_mod.tables = casacore_tables
    casacore_mod.quanta = casacore_quanta
    casacore_mod.measures = casacore_measures

    sys.modules["casacore"] = casacore_mod
    sys.modules["casacore.tables"] = casacore_tables
    sys.modules["casacore.quanta"] = casacore_quanta
    sys.modules["casacore.measures"] = casacore_measures


# Run at collection time so every test file sees the mock
_install_casacore_mock()
