"""Drop-in replacement for ``casacore.tables`` backed by ``casatools.table``.

This module provides the same API surface used by the DSA-110 pipeline from
``python-casacore`` (``casacore.tables``), but implemented on top of
``casatools.table`` from the modular CASA 6 distribution.  This avoids the
C++ shared-library conflict between ``python-casacore`` and ``casatools``
that causes segfaults when both are loaded in the same process.

Usage — replace::

    import casacore.tables as ct

with::

    from dsa110_continuum.adapters.casa_tables import table, taql, makecoldesc, maketabdesc

The ``table`` class supports context-manager (``with``) usage, ``readonly``
and ``ack`` constructor keyword translation, and all column/row methods used
in the pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import casatools as _casatools
except ModuleNotFoundError:  # pragma: no cover — only absent in non-CASA envs
    _casatools = None  # type: ignore[assignment]
import numpy as np

__all__ = [
    "table",
    "taql",
    "makecoldesc",
    "maketabdesc",
    "makescacoldesc",
    "makearrcoldesc",
    "default_ms",
]

_log = logging.getLogger(__name__)


class table:
    """Thin wrapper around :class:`casatools.table` matching the python-casacore API."""

    def __init__(
        self,
        tablename: str = "",
        readonly: bool = True,
        ack: bool = True,  # noqa: ARG002 — ignored, matches casacore signature
        _from_query: bool = False,
    ):
        """Create a table wrapper and open *tablename* when given."""
        if _casatools is None:
            raise RuntimeError(
                "casatools is not installed in this environment. "
                "Install the modular CASA 6 package to use casa_tables."
            )
        self._tb = _casatools.table()
        self._name = tablename
        self._from_query = _from_query
        if tablename and not _from_query:
            self._tb.open(tablename, nomodify=readonly)

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> "table":
        """Return this table for use in a ``with`` block."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the table when leaving a ``with`` block."""
        self.close()

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Close the table."""
        try:
            self._tb.close()
        except Exception:
            pass

    def done(self) -> None:
        """Close the table (casacore-compatible alias)."""
        self.close()

    def flush(self) -> None:
        """Flush pending table changes to disk."""
        self._tb.flush()

    # -- metadata ------------------------------------------------------------

    def nrows(self) -> int:
        """Return the number of rows."""
        return self._tb.nrows()

    def colnames(self) -> list[str]:
        """Return the column names."""
        return self._tb.colnames()

    def name(self) -> str:
        """Return the table name."""
        return self._tb.name() or self._name

    def iswritable(self) -> bool:
        """Return whether the table is writable."""
        return self._tb.iswritable()

    def fieldnames(self) -> list[str]:
        """Return the subtable (field) names."""
        return self._tb.fieldnames()

    # -- array axis convention ------------------------------------------------
    # python-casacore: row axis is FIRST  (nrow, …cell_shape…)
    # casatools:       row axis is LAST   (…cell_shape…, nrow)
    # The helpers below transparently translate between the two conventions.

    @staticmethod
    def _rows_first(arr: np.ndarray) -> np.ndarray:
        """Move the last axis (casatools row axis) to position 0."""
        if arr.ndim <= 1:
            return arr
        return np.moveaxis(arr, -1, 0)

    @staticmethod
    def _rows_last(arr: np.ndarray) -> np.ndarray:
        """Move axis 0 (python-casacore row axis) to the last position."""
        if arr.ndim <= 1:
            return arr
        return np.moveaxis(arr, 0, -1)

    # -- column I/O ----------------------------------------------------------

    def getcol(
        self,
        columnname: str,
        startrow: int = 0,
        nrow: int = -1,
        rowincr: int = 1,
    ) -> np.ndarray:
        """Read a column slice with python-casacore row axis ordering."""
        raw = self._tb.getcol(
            columnname, startrow=startrow, nrow=nrow, rowincr=rowincr,
        )
        if isinstance(raw, np.ndarray):
            return self._rows_first(raw)
        return raw

    def putcol(
        self,
        columnname: str,
        value: Any,
        startrow: int = 0,
        nrow: int = -1,
        rowincr: int = 1,
    ) -> None:
        """Write a column slice, translating row axis layout for casatools."""
        if isinstance(value, np.ndarray) and value.ndim > 1:
            value = self._rows_last(value)
        self._tb.putcol(
            columnname, value, startrow=startrow, nrow=nrow, rowincr=rowincr,
        )

    def getcell(self, columnname: str, rownr: int) -> Any:
        """Return the value of one cell."""
        return self._tb.getcell(columnname, rownr)

    def putcell(self, columnname: str, rownr: int, value: Any) -> None:
        """Write a value to one cell."""
        self._tb.putcell(columnname, rownr, value)

    def getvarcol(self, columnname: str) -> dict:
        """Return a variable-shaped column as a row-keyed dict."""
        return self._tb.getvarcol(columnname)

    def getcolslice(
        self,
        columnname: str,
        blc: list | tuple,
        trc: list | tuple,
        inc: list | tuple | None = None,
        startrow: int = 0,
        nrow: int = -1,
        rowincr: int = 1,
    ) -> np.ndarray:
        """Read a hyper-rectangular slice of an array column."""
        if inc is None:
            inc = [1] * len(blc)
        raw = self._tb.getcolslice(
            columnname, blc=list(blc), trc=list(trc), incr=list(inc),
            startrow=startrow, nrow=nrow, rowincr=rowincr,
        )
        if isinstance(raw, np.ndarray):
            return self._rows_first(raw)
        return raw

    def getcoldesc(self, columnname: str) -> dict:
        """Return the column description dict."""
        return self._tb.getcoldesc(columnname)

    def coldatatype(self, columnname: str) -> str:
        """Return the column value type string."""
        return self._tb.coldatatype(columnname)

    def isscalarcol(self, columnname: str) -> bool:
        """Return whether the column stores scalar cells."""
        return self._tb.isscalarcol(columnname)

    def isvarcol(self, columnname: str) -> bool:
        """Return whether the column has variable-shaped cells."""
        return self._tb.isvarcol(columnname)

    # -- row / column structural ops -----------------------------------------

    def addrows(self, nrows: int = 1) -> None:
        """Append *nrows* empty rows to the table."""
        self._tb.addrows(nrows)

    def removerows(self, rownrs: list[int] | np.ndarray) -> None:
        """Remove the rows with the given numbers."""
        self._tb.removerows(list(rownrs))

    def removecols(self, columnnames: list[str]) -> None:
        """Remove the named columns."""
        self._tb.removecols(columnnames)

    def addcols(self, desc: dict, dminfo: dict | None = None) -> None:
        """Add columns from a description dict and optional storage info."""
        if dminfo is None:
            dminfo = {}
        self._tb.addcols(desc, dminfo)

    # -- keyword ops ---------------------------------------------------------

    def getkeyword(self, keyword: str) -> Any:
        """Return a table-level keyword value."""
        return self._tb.getkeyword(keyword)

    def putkeyword(self, keyword: str, value: Any) -> None:
        """Set a table-level keyword."""
        self._tb.putkeyword(keyword, value)

    def getkeywords(self) -> dict:
        """Return all table-level keywords."""
        return self._tb.getkeywords()

    def getcolkeyword(self, columnname: str, keyword: str) -> Any:
        """Return a column keyword value."""
        return self._tb.getcolkeyword(columnname, keyword)

    def putcolkeyword(self, columnname: str, keyword: str, value: Any) -> None:
        """Set a column keyword."""
        self._tb.putcolkeyword(columnname, keyword, value)

    def getcolkeywords(self, columnname: str) -> dict:
        """Return all keywords attached to a column."""
        return self._tb.getcolkeywords(columnname)

    # -- query / selection ---------------------------------------------------

    def query(self, query: str, sortlist: str = "", columns: str = "") -> "table":
        """Run a table query and return the result as a new :class:`table`."""
        result_tb = self._tb.query(query, sortlist=sortlist, columns=columns)
        wrapper = table.__new__(table)
        wrapper._tb = result_tb
        wrapper._name = f"query({self._name})"
        wrapper._from_query = True
        return wrapper

    def taql(self, command: str) -> "table":
        """Execute TaQL and return the result as a new :class:`table`."""
        result_tb = self._tb.taql(command)
        wrapper = table.__new__(table)
        wrapper._tb = result_tb
        wrapper._name = f"taql({self._name})"
        wrapper._from_query = True
        return wrapper

    def selectrows(self, rownrs: list[int] | np.ndarray) -> "table":
        """Return a subtable containing only the selected row numbers."""
        result_tb = self._tb.selectrows(list(rownrs))
        wrapper = table.__new__(table)
        wrapper._tb = result_tb
        wrapper._name = f"selectrows({self._name})"
        wrapper._from_query = True
        return wrapper

    # -- schema creation (static helpers) ------------------------------------

    def getdesc(self) -> dict:
        """Return the full table description dict."""
        return self._tb.getdesc()

    def getdminfo(self) -> dict:
        """Return storage manager (dminfo) metadata for the table."""
        return self._tb.getdminfo()

    # -- repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a short debug representation of the table."""
        return f"<table({self._name!r})>"


# ---------------------------------------------------------------------------
# Module-level helpers matching ``casacore.tables`` utility functions
# ---------------------------------------------------------------------------


def taql(command: str) -> table:
    """Execute a TaQL command and return the result as a :class:`table`."""
    if _casatools is None:
        raise RuntimeError("casatools is not installed in this environment.")
    tb = _casatools.table()
    result = tb.taql(command)
    wrapper = table.__new__(table)
    wrapper._tb = result
    wrapper._name = f"taql({command[:60]})"
    wrapper._from_query = True
    return wrapper


def makecoldesc(name: str, desc: dict | None = None, **kwargs: Any) -> dict:
    """Build a column description dict compatible with casatools ``addcols``.

    Mirrors ``casacore.tables.makecoldesc`` — accepts either a template
    descriptor dict (from ``getcoldesc``) or explicit keyword arguments
    like ``valuetype``.
    """
    if desc is None:
        desc = {}
    result = dict(desc)
    result.update(kwargs)
    return {"name": name, "desc": result}


def maketabdesc(descs: dict | list) -> dict:
    """Build a table description from one or more column descriptions.

    Mirrors ``casacore.tables.maketabdesc``.
    """
    if isinstance(descs, dict) and "name" in descs:
        descs = [descs]
    elif isinstance(descs, dict):
        return descs

    out: dict[str, Any] = {}
    for d in descs:
        name = d["name"]
        out[name] = d.get("desc", d)
    return out


def makescacoldesc(name: str, value: Any, **kwargs: Any) -> dict:
    """Create a scalar column description."""
    return {"name": name, "desc": {"valueType": type(value).__name__, **kwargs}}


def makearrcoldesc(name: str, value: Any, **kwargs: Any) -> dict:
    """Create an array column description."""
    return {"name": name, "desc": {"valueType": type(value).__name__, **kwargs}}


def default_ms(name: str) -> "table":
    """Create a default (empty) Measurement Set with standard subtables.

    Mirrors ``casacore.tables.default_ms``.  Uses the CASA simulator tool
    to create a structurally valid MS that has all required subtables
    (FIELD, OBSERVATION, SPECTRAL_WINDOW, DATA_DESCRIPTION, POLARIZATION,
    ANTENNA, FEED, etc.) already present.
    """
    if _casatools is None:
        raise RuntimeError("casatools is not installed in this environment.")
    sim = _casatools.simulator()
    sim.open(name)
    sim.close()
    return table(name, readonly=False)
