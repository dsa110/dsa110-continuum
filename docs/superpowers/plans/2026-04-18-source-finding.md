# Source Finding (Step 8) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract blind source-finding into `dsa110_continuum/source_finding/core.py`, eliminate global state, add `--sim` mode, and add 11 CI-runnable tests.

**Architecture:** New `dsa110_continuum/source_finding/` package with pure functions. `scripts/source_finding.py` becomes a thin CLI wrapper. AegeanTools is always imported inside function bodies so tests can mock it without `sys.modules` gymnastics.

**Tech Stack:** Python 3.10+, astropy, AegeanTools (mocked in tests), unittest.mock, pytest, tempfile.NamedTemporaryFile

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `dsa110_continuum/source_finding/__init__.py` | Create | Package exports |
| `dsa110_continuum/source_finding/core.py` | Create | All pure functions + dataclass |
| `tests/test_source_finding.py` | Create | 11 CI-runnable tests |
| `scripts/source_finding.py` | Modify | Thin CLI wrapper (~60 lines) |

---

## IMPORTANT: Environment Constraints

- **`tempfile.NamedTemporaryFile`** — always use `delete=False` and clean up manually. Do NOT use `pytest` `tmp_path` fixture (not writable in sandbox).
- **`AegeanTools` not installed** in CI/sandbox — all tests that touch `run_bane` or `run_aegean` must mock the import.
- **Mocking pattern:** AegeanTools is imported inside functions. Patch at the module level:
  ```python
  with patch("dsa110_continuum.source_finding.core.AegeanTools_BANE") as mock_bane:
  ```
  But since we import `from AegeanTools import BANE as AegeanTools_BANE` inside the function, patching requires a different approach — patch `builtins.__import__` or restructure. The simplest correct approach: **import the module object at the top of the function and assign it to a name that `unittest.mock.patch` can target**.

  Concrete pattern used in `core.py`:
  ```python
  def run_bane(...):
      import AegeanTools.BANE as _bane_mod  # patchable
      _bane_mod.filter_image(...)
  ```
  Then in tests:
  ```python
  with patch("AegeanTools.BANE.filter_image") as mock_fi:
  ```
  But since AegeanTools is not installed, `import AegeanTools.BANE` itself fails.

  **Correct approach (used throughout):** use `importlib.import_module` inside the function and patch `importlib.import_module` — OR — patch the entire import using `sys.modules`:

  ```python
  # In core.py:
  def run_bane(...):
      try:
          from AegeanTools import BANE as _bane
      except ImportError as e:
          raise ImportError("AegeanTools not installed. pip install AegeanTools") from e
      _bane.filter_image(...)
  ```

  ```python
  # In tests:
  import sys
  from unittest.mock import MagicMock, patch
  
  mock_bane = MagicMock()
  mock_aegean_tools = MagicMock()
  mock_aegean_tools.BANE = mock_bane
  
  with patch.dict(sys.modules, {
      "AegeanTools": mock_aegean_tools,
      "AegeanTools.BANE": mock_bane,
      "AegeanTools.source_finder": MagicMock(),
  }):
      from dsa110_continuum.source_finding import core
      importlib.reload(core)  # force re-import with mocked modules
      ...
  ```

  **Simplest CI-safe pattern** (no reload needed): patch at the point of use with `sys.modules` injection BEFORE importing `core`. Since tests run in a clean module namespace, import `core` fresh after injecting mocks. See Task 2 for the exact setup.

---

## Task 1: `SourceCatalogEntry` dataclass + catalog writers

**Files:**
- Create: `dsa110_continuum/source_finding/__init__.py`
- Create: `dsa110_continuum/source_finding/core.py` (partial — dataclass + writers only)
- Create: `tests/test_source_finding.py` (partial — tests 1–3)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_source_finding.py
import tempfile
import os
import pytest
from astropy.table import Table

from dsa110_continuum.source_finding.core import (
    SourceCatalogEntry,
    write_catalog,
    write_empty_catalog,
)


def _make_entry(n=0) -> SourceCatalogEntry:
    return SourceCatalogEntry(
        source_name=f"AEG_J344.1234+16.5678_{n}",
        ra_deg=344.1234 + n,
        dec_deg=16.5678,
        peak_flux_jy=0.05 + n * 0.01,
        peak_flux_err_jy=0.002,
        int_flux_jy=0.06 + n * 0.01,
        a_arcsec=36.9,
        b_arcsec=25.5,
        pa_deg=130.75,
        local_rms_jy=0.003,
    )


def test_source_catalog_entry_fields():
    e = _make_entry()
    assert e.source_name.startswith("AEG_")
    assert isinstance(e.ra_deg, float)
    assert isinstance(e.dec_deg, float)
    assert isinstance(e.peak_flux_jy, float)
    assert isinstance(e.peak_flux_err_jy, float)
    assert isinstance(e.int_flux_jy, float)
    assert isinstance(e.a_arcsec, float)
    assert isinstance(e.b_arcsec, float)
    assert isinstance(e.pa_deg, float)
    assert isinstance(e.local_rms_jy, float)


def test_write_catalog_roundtrip():
    entries = [_make_entry(i) for i in range(3)]
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        path = f.name
    try:
        write_catalog(entries, path)
        t = Table.read(path)
        assert len(t) == 3
        expected_cols = [
            "source_name", "ra_deg", "dec_deg", "peak_flux_jy",
            "peak_flux_err_jy", "int_flux_jy", "a_arcsec", "b_arcsec",
            "pa_deg", "local_rms_jy",
        ]
        for col in expected_cols:
            assert col in t.colnames, f"Missing column: {col}"
        assert abs(t["ra_deg"][1] - 345.1234) < 1e-6
        assert abs(t["peak_flux_jy"][2] - 0.07) < 1e-6
    finally:
        os.unlink(path)


def test_write_empty_catalog_schema():
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        path = f.name
    try:
        write_empty_catalog(path)
        t = Table.read(path)
        assert len(t) == 0
        expected_cols = [
            "source_name", "ra_deg", "dec_deg", "peak_flux_jy",
            "peak_flux_err_jy", "int_flux_jy",
        ]
        for col in expected_cols:
            assert col in t.colnames, f"Missing column: {col}"
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/test_source_finding.py::test_source_catalog_entry_fields \
       tests/test_source_finding.py::test_write_catalog_roundtrip \
       tests/test_source_finding.py::test_write_empty_catalog_schema -v 2>&1 | tail -20
```

Expected: ImportError or ModuleNotFoundError — `dsa110_continuum.source_finding` doesn't exist yet.

- [ ] **Step 3: Create the package skeleton and implement dataclass + writers**

```python
# dsa110_continuum/source_finding/__init__.py
"""Blind source finding for DSA-110 continuum pipeline (BANE + Aegean)."""

from dsa110_continuum.source_finding.core import (
    SourceCatalogEntry,
    check_catalog,
    run_aegean,
    run_bane,
    run_source_finding,
    write_catalog,
    write_empty_catalog,
)

__all__ = [
    "SourceCatalogEntry",
    "run_bane",
    "run_aegean",
    "write_catalog",
    "write_empty_catalog",
    "check_catalog",
    "run_source_finding",
]
```

```python
# dsa110_continuum/source_finding/core.py
"""
Pure-function source-finding core for DSA-110 continuum pipeline.

Steps:
  1. run_bane()            — background/RMS estimation (AegeanTools.BANE)
  2. run_aegean()          — blind detection (AegeanTools.source_finder)
  3. write_catalog() /
     write_empty_catalog() — FITS table output
  4. check_catalog()       — QA / logging
  5. run_source_finding()  — orchestrator
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.table import Table

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SourceCatalogEntry:
    """One detected source component from Aegean."""
    source_name:      str
    ra_deg:           float
    dec_deg:          float
    peak_flux_jy:     float
    peak_flux_err_jy: float
    int_flux_jy:      float
    a_arcsec:         float
    b_arcsec:         float
    pa_deg:           float
    local_rms_jy:     float


# ---------------------------------------------------------------------------
# Catalog I/O
# ---------------------------------------------------------------------------

_CATALOG_COLS = [
    "source_name", "ra_deg", "dec_deg", "peak_flux_jy",
    "peak_flux_err_jy", "int_flux_jy", "a_arcsec", "b_arcsec",
    "pa_deg", "local_rms_jy",
]
_CATALOG_DTYPES = ["U64", float, float, float, float, float, float, float, float, float]


def write_catalog(entries: list[SourceCatalogEntry], out_path: str | Path) -> None:
    """Write source entries to a FITS binary table."""
    rows = [
        {
            "source_name": e.source_name,
            "ra_deg": e.ra_deg,
            "dec_deg": e.dec_deg,
            "peak_flux_jy": e.peak_flux_jy,
            "peak_flux_err_jy": e.peak_flux_err_jy,
            "int_flux_jy": e.int_flux_jy,
            "a_arcsec": e.a_arcsec,
            "b_arcsec": e.b_arcsec,
            "pa_deg": e.pa_deg,
            "local_rms_jy": e.local_rms_jy,
        }
        for e in entries
    ]
    t = Table(rows)
    t.write(str(out_path), overwrite=True)
    log.info("Catalog written: %s  (%d sources)", out_path, len(entries))


def write_empty_catalog(out_path: str | Path) -> None:
    """Write a zero-row FITS table with the correct column schema."""
    t = Table(names=_CATALOG_COLS, dtype=_CATALOG_DTYPES)
    t.write(str(out_path), overwrite=True)
    log.info("Empty catalog written: %s", out_path)
```

- [ ] **Step 4: Run tests — must pass**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/test_source_finding.py::test_source_catalog_entry_fields \
       tests/test_source_finding.py::test_write_catalog_roundtrip \
       tests/test_source_finding.py::test_write_empty_catalog_schema -v 2>&1 | tail -15
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
cd /home/user/workspace/dsa110-continuum
git add dsa110_continuum/source_finding/__init__.py \
        dsa110_continuum/source_finding/core.py \
        tests/test_source_finding.py
git commit -m "feat(source_finding): add SourceCatalogEntry dataclass and catalog writers"
```

---

## Task 2: `run_bane()` + tests 7–8

**Files:**
- Modify: `dsa110_continuum/source_finding/core.py` (add `run_bane`)
- Modify: `tests/test_source_finding.py` (add tests 7–8)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_source_finding.py`:

```python
import sys
from unittest.mock import MagicMock, call, patch


def _inject_bane_mock(mock_filter_image=None):
    """Inject mock AegeanTools.BANE into sys.modules. Returns mock module."""
    mock_bane = MagicMock()
    if mock_filter_image is not None:
        mock_bane.filter_image = mock_filter_image
    mock_at = MagicMock()
    mock_at.BANE = mock_bane
    sys.modules["AegeanTools"] = mock_at
    sys.modules["AegeanTools.BANE"] = mock_bane
    return mock_bane


def test_run_bane_skip_existing(tmp_path):
    """If both bkg and rms files exist, run_bane returns immediately."""
    import tempfile, os
    mosaic = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    mosaic.close()
    stem = mosaic.name.replace(".fits", "")
    bkg_path = stem + "_bkg.fits"
    rms_path = stem + "_rms.fits"
    # Create stub output files
    open(bkg_path, "w").close()
    open(rms_path, "w").close()
    try:
        from dsa110_continuum.source_finding.core import run_bane
        result_bkg, result_rms = run_bane(mosaic.name, skip_existing=True)
        assert result_bkg == bkg_path
        assert result_rms == rms_path
    finally:
        for p in [mosaic.name, bkg_path, rms_path]:
            if os.path.exists(p):
                os.unlink(p)


def test_run_bane_missing_output():
    """RuntimeError when BANE mock runs but does not produce output files."""
    import tempfile, os
    mosaic = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    mosaic.close()
    try:
        mock_bane = _inject_bane_mock()  # filter_image is a no-op MagicMock
        from dsa110_continuum.source_finding import core
        import importlib; importlib.reload(core)
        import pytest
        with pytest.raises(RuntimeError, match="BANE did not produce"):
            core.run_bane(mosaic.name, skip_existing=False)
    finally:
        os.unlink(mosaic.name)
        # Clean sys.modules
        for k in list(sys.modules.keys()):
            if "AegeanTools" in k:
                del sys.modules[k]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/test_source_finding.py::test_run_bane_skip_existing \
       tests/test_source_finding.py::test_run_bane_missing_output -v 2>&1 | tail -15
```

Expected: FAIL — `run_bane` not defined yet.

- [ ] **Step 3: Implement `run_bane` in `core.py`**

Add after `write_empty_catalog`:

```python
# ---------------------------------------------------------------------------
# BANE background estimator
# ---------------------------------------------------------------------------

def run_bane(
    mosaic_path: str | Path,
    *,
    box_size: int = 600,
    step_size: int = 300,
    cores: int = 1,
    skip_existing: bool = True,
) -> tuple[str, str]:
    """Run BANE background/RMS estimator. Returns (bkg_path, rms_path).

    Parameters
    ----------
    mosaic_path : str | Path
        Input mosaic FITS file.
    box_size : int
        BANE box size in pixels (default 600 ≈ 50 beams).
    step_size : int
        BANE step size in pixels (default 300).
    cores : int
        Number of worker cores (default 1 to avoid shared-memory deadlock
        on NaN-heavy mosaics).
    skip_existing : bool
        If True and both output files exist, return immediately without
        calling BANE.

    Returns
    -------
    tuple[str, str]
        Paths to (background_map.fits, rms_map.fits).

    Raises
    ------
    RuntimeError
        If BANE runs but does not produce expected output files.
    ImportError
        If AegeanTools is not installed.
    """
    mosaic_path = str(mosaic_path)
    stem = mosaic_path.replace(".fits", "")
    bkg_path = stem + "_bkg.fits"
    rms_path = stem + "_rms.fits"

    if skip_existing and Path(bkg_path).exists() and Path(rms_path).exists():
        log.info("BANE outputs already exist — skipping: %s, %s", bkg_path, rms_path)
        return bkg_path, rms_path

    log.info("Running BANE on %s (box=%d, step=%d, cores=%d) ...",
             mosaic_path, box_size, step_size, cores)
    try:
        from AegeanTools import BANE as _bane
    except ImportError as e:
        raise ImportError(
            "AegeanTools not installed. "
            "Install with: pip install git+https://github.com/PaulHancock/Aegean.git"
        ) from e

    _bane.filter_image(
        im_name=mosaic_path,
        out_base=stem,
        step_size=[step_size, step_size],
        box_size=[box_size, box_size],
        cores=cores,
        mask=True,
    )

    if not Path(bkg_path).exists() or not Path(rms_path).exists():
        raise RuntimeError(
            f"BANE did not produce expected outputs: {bkg_path}, {rms_path}"
        )

    log.info("BANE done: bkg=%s, rms=%s", bkg_path, rms_path)
    return bkg_path, rms_path
```

- [ ] **Step 4: Run tests — must pass**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/test_source_finding.py::test_run_bane_skip_existing \
       tests/test_source_finding.py::test_run_bane_missing_output -v 2>&1 | tail -15
```

Expected: 2 PASSED.

- [ ] **Step 5: Run full test suite to confirm no regressions**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/test_source_finding.py -v 2>&1 | tail -20
```

Expected: 5 PASSED (tests 1–3 from Task 1 + 2 new).

- [ ] **Step 6: Commit**

```bash
cd /home/user/workspace/dsa110-continuum
git add dsa110_continuum/source_finding/core.py tests/test_source_finding.py
git commit -m "feat(source_finding): implement run_bane with skip-existing and RuntimeError guard"
```

---

## Task 3: `run_aegean()` + `check_catalog()` + `run_source_finding()` + tests 9–11 + tests 4–6

**Files:**
- Modify: `dsa110_continuum/source_finding/core.py` (add remaining functions)
- Modify: `tests/test_source_finding.py` (add tests 4–6 and 9–11)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_source_finding.py`:

```python
# ── check_catalog tests (4–6) ───────────────────────────────────────────────

def test_check_catalog_empty():
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        path = f.name
    try:
        write_empty_catalog(path)
        from dsa110_continuum.source_finding.core import check_catalog
        assert check_catalog(path) is False
    finally:
        os.unlink(path)


def test_check_catalog_non_empty_no_bright():
    """Non-empty catalog with no bright (>1 Jy) sources → True with warning."""
    import tempfile, os
    entry = _make_entry(0)  # peak_flux_jy = 0.05 Jy
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        path = f.name
    try:
        write_catalog([entry], path)
        from dsa110_continuum.source_finding.core import check_catalog
        result = check_catalog(path)
        assert result is True  # non-empty → True; bright warning is logged only
    finally:
        os.unlink(path)


def test_check_catalog_with_bright_source():
    import tempfile, os
    entry = _make_entry(0)
    entry = SourceCatalogEntry(
        source_name="AEG_J344.0000+16.0000",
        ra_deg=344.0,
        dec_deg=16.0,
        peak_flux_jy=2.5,      # > 1 Jy
        peak_flux_err_jy=0.05,
        int_flux_jy=3.0,
        a_arcsec=36.9,
        b_arcsec=25.5,
        pa_deg=130.0,
        local_rms_jy=0.003,
    )
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        path = f.name
    try:
        write_catalog([entry], path)
        from dsa110_continuum.source_finding.core import check_catalog
        assert check_catalog(path) is True
    finally:
        os.unlink(path)


# ── run_aegean tests (9–11) ─────────────────────────────────────────────────

def _make_aegean_source(ra=344.0, dec=16.0, peak=0.05, rms=0.003):
    src = MagicMock()
    src.ra = ra
    src.dec = dec
    src.peak_flux = peak
    src.err_peak_flux = 0.002
    src.int_flux = peak * 1.1
    src.a = 36.9
    src.b = 25.5
    src.pa = 130.75
    src.local_rms = rms
    return src


def _inject_aegean_mock(found_sources):
    """Inject mock AegeanTools.source_finder into sys.modules."""
    mock_sf_instance = MagicMock()
    mock_sf_instance.find_sources_in_image.return_value = found_sources
    mock_sf_cls = MagicMock(return_value=mock_sf_instance)
    mock_sf_mod = MagicMock()
    mock_sf_mod.SourceFinder = mock_sf_cls
    mock_at = sys.modules.get("AegeanTools", MagicMock())
    mock_at.source_finder = mock_sf_mod
    sys.modules["AegeanTools"] = mock_at
    sys.modules["AegeanTools.source_finder"] = mock_sf_mod
    return mock_sf_instance


def test_run_aegean_import_error():
    """ImportError with install hint when AegeanTools is absent."""
    # Remove any existing mock
    for k in list(sys.modules.keys()):
        if "AegeanTools" in k:
            del sys.modules[k]
    import importlib
    from dsa110_continuum.source_finding import core
    importlib.reload(core)
    with pytest.raises(ImportError, match="AegeanTools not installed"):
        core.run_aegean("fake.fits", "fake_bkg.fits", "fake_rms.fits")


def test_run_aegean_returns_empty_list():
    """Aegean finds nothing → returns empty list."""
    _inject_aegean_mock(found_sources=[])
    import importlib
    from dsa110_continuum.source_finding import core
    importlib.reload(core)
    result = core.run_aegean("fake.fits", "fake_bkg.fits", "fake_rms.fits")
    assert result == []
    # Restore
    for k in list(sys.modules.keys()):
        if "AegeanTools" in k:
            del sys.modules[k]


def test_run_aegean_returns_entries():
    """Mock SourceFinder returns 2 sources → list of SourceCatalogEntry."""
    sources = [
        _make_aegean_source(ra=344.0, dec=16.0, peak=0.05),
        _make_aegean_source(ra=345.0, dec=16.5, peak=2.5),
    ]
    _inject_aegean_mock(found_sources=sources)
    import importlib
    from dsa110_continuum.source_finding import core
    importlib.reload(core)
    result = core.run_aegean("fake.fits", "fake_bkg.fits", "fake_rms.fits")
    assert len(result) == 2
    assert all(isinstance(e, core.SourceCatalogEntry) for e in result)
    assert abs(result[0].ra_deg - 344.0) < 1e-6
    assert abs(result[1].peak_flux_jy - 2.5) < 1e-6
    assert result[0].source_name.startswith("AEG_J")
    # Restore
    for k in list(sys.modules.keys()):
        if "AegeanTools" in k:
            del sys.modules[k]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/test_source_finding.py::test_check_catalog_empty \
       tests/test_source_finding.py::test_check_catalog_non_empty_no_bright \
       tests/test_source_finding.py::test_check_catalog_with_bright_source \
       tests/test_source_finding.py::test_run_aegean_import_error \
       tests/test_source_finding.py::test_run_aegean_returns_empty_list \
       tests/test_source_finding.py::test_run_aegean_returns_entries -v 2>&1 | tail -15
```

Expected: 6 FAIL — functions not defined yet.

- [ ] **Step 3: Implement `run_aegean`, `check_catalog`, `run_source_finding` in `core.py`**

Append to `dsa110_continuum/source_finding/core.py`:

```python
# ---------------------------------------------------------------------------
# Aegean blind source detection
# ---------------------------------------------------------------------------

def run_aegean(
    mosaic_path: str | Path,
    bkg_path: str | Path,
    rms_path: str | Path,
    *,
    sigma: float = 7.0,
) -> list[SourceCatalogEntry]:
    """Run Aegean blind source detection.

    Parameters
    ----------
    mosaic_path : str | Path
        Input mosaic FITS file.
    bkg_path : str | Path
        BANE background map (produced by run_bane).
    rms_path : str | Path
        BANE RMS map (produced by run_bane).
    sigma : float
        Detection threshold in units of local RMS (default 7.0).

    Returns
    -------
    list[SourceCatalogEntry]
        Detected source components (may be empty).

    Raises
    ------
    ImportError
        If AegeanTools is not installed.
    """
    try:
        from AegeanTools.source_finder import SourceFinder as _SourceFinder
    except ImportError as e:
        raise ImportError(
            "AegeanTools not installed. "
            "Install with: pip install git+https://github.com/PaulHancock/Aegean.git"
        ) from e

    log.info("Running Aegean (%.1fσ threshold) on %s ...", sigma, mosaic_path)
    sf = _SourceFinder(log=logging.getLogger("AegeanTools"))
    found = sf.find_sources_in_image(
        filename=str(mosaic_path),
        hdu_index=0,
        outfile=None,
        innerclip=sigma,
        outerclip=sigma - 1.0,
        rmsin=str(rms_path),
        bkgin=str(bkg_path),
        cores=1,
    )

    if not found:
        log.warning("Aegean found no sources at %.1fσ", sigma)
        return []

    log.info("Aegean found %d source components", len(found))
    entries = []
    for s in found:
        ra = float(s.ra)
        dec = float(s.dec)
        entries.append(SourceCatalogEntry(
            source_name=f"AEG_J{ra:.4f}{dec:+.4f}",
            ra_deg=ra,
            dec_deg=dec,
            peak_flux_jy=float(s.peak_flux),
            peak_flux_err_jy=float(getattr(s, "err_peak_flux", 0.0)),
            int_flux_jy=float(getattr(s, "int_flux", s.peak_flux)),
            a_arcsec=float(getattr(s, "a", 0.0)),
            b_arcsec=float(getattr(s, "b", 0.0)),
            pa_deg=float(getattr(s, "pa", 0.0)),
            local_rms_jy=float(getattr(s, "local_rms", 0.0)),
        ))
    return entries


# ---------------------------------------------------------------------------
# Catalog QA
# ---------------------------------------------------------------------------

def check_catalog(
    catalog_path: str | Path,
    *,
    sky_ra_range: tuple[float, float] = (300.0, 360.0),
    sky_dec_range: tuple[float, float] = (0.0, 40.0),
) -> bool:
    """Check catalog quality. Returns True if catalog is non-empty.

    Logs number of sources, bright sources (>1 Jy), and sources in the
    expected sky window. The bright-source criterion is a warning only,
    not a hard failure (sim dirty-image fluxes are suppressed).

    Parameters
    ----------
    catalog_path : str | Path
        Path to catalog FITS table.
    sky_ra_range : tuple[float, float]
        Expected RA window (deg) for positional sanity check.
    sky_dec_range : tuple[float, float]
        Expected Dec window (deg) for positional sanity check.

    Returns
    -------
    bool
        True if catalog has at least one source, False if empty.
    """
    t = Table.read(str(catalog_path))
    log.info("Catalog: %d sources", len(t))

    if len(t) == 0:
        log.error("QA FAIL: empty catalog")
        return False

    if "peak_flux_jy" in t.colnames:
        bright = t[t["peak_flux_jy"] > 1.0]
        log.info("Sources > 1 Jy: %d", len(bright))
        if len(bright) == 0:
            log.warning("QA WARNING: no bright sources (>1 Jy) detected")
        else:
            for row in bright:
                log.info("  %s  RA=%.3f  Dec=%.3f  peak=%.2f Jy",
                         row["source_name"], row["ra_deg"], row["dec_deg"],
                         row["peak_flux_jy"])

    if "ra_deg" in t.colnames and "dec_deg" in t.colnames:
        ra_lo, ra_hi = sky_ra_range
        dec_lo, dec_hi = sky_dec_range
        in_range = int(np.sum(
            (t["ra_deg"] > ra_lo) & (t["ra_deg"] < ra_hi) &
            (t["dec_deg"] > dec_lo) & (t["dec_deg"] < dec_hi)
        ))
        log.info("Sources in expected sky region (RA %.0f–%.0f, Dec %.0f–%.0f): %d/%d",
                 ra_lo, ra_hi, dec_lo, dec_hi, in_range, len(t))

    log.info("QA PASSED: catalog has %d source(s)", len(t))
    return True


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_source_finding(
    mosaic_path: str | Path,
    out_path: str | Path,
    *,
    bane_box: int = 600,
    bane_step: int = 300,
    aegean_sigma: float = 7.0,
) -> str:
    """Run the full source-finding pipeline: BANE → Aegean → catalog → QA.

    Parameters
    ----------
    mosaic_path : str | Path
        Input mosaic FITS file.
    out_path : str | Path
        Output catalog FITS path.
    bane_box : int
        BANE box size in pixels.
    bane_step : int
        BANE step size in pixels.
    aegean_sigma : float
        Aegean detection threshold (σ).

    Returns
    -------
    str
        Path to the written catalog.
    """
    mosaic_path = Path(mosaic_path)
    out_path = Path(out_path)

    bkg_path, rms_path = run_bane(
        mosaic_path, box_size=bane_box, step_size=bane_step
    )
    entries = run_aegean(mosaic_path, bkg_path, rms_path, sigma=aegean_sigma)

    if entries:
        write_catalog(entries, out_path)
    else:
        write_empty_catalog(out_path)

    check_catalog(out_path)
    return str(out_path)
```

- [ ] **Step 4: Run the 6 new tests — must pass**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/test_source_finding.py::test_check_catalog_empty \
       tests/test_source_finding.py::test_check_catalog_non_empty_no_bright \
       tests/test_source_finding.py::test_check_catalog_with_bright_source \
       tests/test_source_finding.py::test_run_aegean_import_error \
       tests/test_source_finding.py::test_run_aegean_returns_empty_list \
       tests/test_source_finding.py::test_run_aegean_returns_entries -v 2>&1 | tail -20
```

Expected: 6 PASSED.

- [ ] **Step 5: Run full test file — all 11 must pass**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/test_source_finding.py -v 2>&1 | tail -25
```

Expected: 11 PASSED.

- [ ] **Step 6: Commit**

```bash
cd /home/user/workspace/dsa110-continuum
git add dsa110_continuum/source_finding/core.py tests/test_source_finding.py
git commit -m "feat(source_finding): implement run_aegean, check_catalog, run_source_finding orchestrator"
```

---

## Task 4: Refactor `scripts/source_finding.py` with `--sim` and exit gate

**Files:**
- Modify: `scripts/source_finding.py` (full refactor to ~60 lines)

No new tests needed — the script is a thin CLI wrapper over already-tested
library functions. The `--sim` exit gate is validated by manual inspection.

- [ ] **Step 1: Replace `scripts/source_finding.py` with the refactored version**

```python
#!/opt/miniforge/envs/casa6/bin/python
"""
Source finding on a DSA-110 mosaic using BANE + Aegean.

Steps:
  1. Run BANE to estimate background (bkg) and local RMS (rms)
  2. Run Aegean at --sigma threshold on the mosaic
  3. Write source catalog as FITS table
  4. Report statistics and verify success criteria

Usage:
  source_finding.py [--mosaic PATH] [--out PATH] [--sigma FLOAT] [--sim]
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

from dsa110_continuum.source_finding import run_source_finding

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Production default — overridden by --mosaic or --sim
_DEFAULT_MOSAIC = "/stage/dsa110-contimg/images/mosaic_2026-01-25/full_mosaic.fits"

# Sim-mode mosaic (relative to repo root)
_SIM_MOSAIC = str(
    Path(__file__).resolve().parents[1]
    / "pipeline_outputs/step6/step6_mosaic.fits"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Source finding on a DSA-110 mosaic (BANE + Aegean).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mosaic", default=None, metavar="PATH",
        help="Path to mosaic FITS file.",
    )
    parser.add_argument(
        "--out", default=None, metavar="PATH",
        help="Output catalog path (default: {mosaic_stem}_sources.fits).",
    )
    parser.add_argument(
        "--sigma", type=float, default=7.0,
        help="Aegean detection threshold in σ.",
    )
    parser.add_argument(
        "--sim", action="store_true",
        help=(
            "Use the pipeline sim mosaic "
            "(pipeline_outputs/step6/step6_mosaic.fits). "
            "Runs full pipeline but exits 0 regardless of QA result."
        ),
    )
    args = parser.parse_args()

    # Resolve mosaic path
    if args.sim:
        mosaic_path = _SIM_MOSAIC
        log.info("[SIM MODE] Using simulated mosaic: %s", mosaic_path)
    else:
        mosaic_path = args.mosaic or _DEFAULT_MOSAIC

    if not Path(mosaic_path).exists():
        log.error("Mosaic not found: %s", mosaic_path)
        sys.exit(1)

    # Derive catalog output path
    out_path = args.out or str(Path(mosaic_path).with_suffix("")) + "_sources.fits"

    # Log mosaic quick-look
    with fits.open(mosaic_path) as hdul:
        data = hdul[0].data.squeeze()
        finite = data[np.isfinite(data)]
        peak = float(np.nanmax(data))
        med = float(np.median(finite))
        rms = float(1.4826 * np.median(np.abs(finite - med)))
        log.info(
            "Mosaic: peak=%.4f Jy/beam  MAD-RMS=%.4f Jy/beam  shape=%s",
            peak, rms, data.shape,
        )

    # Run pipeline
    catalog_path = run_source_finding(
        mosaic_path,
        out_path,
        aegean_sigma=args.sigma,
    )

    print(f"\nCatalog written: {catalog_path}")

    if args.sim:
        log.info("[SIM MODE] Exiting 0 (QA not enforced on dirty-image mosaic)")
        sys.exit(0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script is importable (syntax check)**

```bash
cd /home/user/workspace/dsa110-continuum
python -c "import ast; ast.parse(open('scripts/source_finding.py').read()); print('Syntax OK')"
```

Expected: `Syntax OK`

- [ ] **Step 3: Verify help output**

```bash
cd /home/user/workspace/dsa110-continuum
python scripts/source_finding.py --help
```

Expected: help text showing `--mosaic`, `--out`, `--sigma`, `--sim`.

- [ ] **Step 4: Run full test suite — all 11 still pass**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/test_source_finding.py -v 2>&1 | tail -15
```

Expected: 11 PASSED.

- [ ] **Step 5: Commit**

```bash
cd /home/user/workspace/dsa110-continuum
git add scripts/source_finding.py
git commit -m "feat(scripts): refactor source_finding.py — thin wrapper, add --sim, --out, --sigma"
```

---

## Task 5: Push and verify

- [ ] **Step 1: Run the full test suite one final time**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/test_source_finding.py tests/test_two_stage_photometry.py -v 2>&1 | tail -30
```

Expected: all tests pass (11 source_finding + 14 two_stage = 25 total).

- [ ] **Step 2: Show git log for the new commits**

```bash
cd /home/user/workspace/dsa110-continuum
git log --oneline -6
```

- [ ] **Step 3: Push to remote**

```bash
cd /home/user/workspace/dsa110-continuum
git push origin main
```

Expected: commits pushed cleanly.

- [ ] **Step 4: Report to user**

Show:
1. Final test counts (11 source_finding + 14 two_stage)
2. `git log --oneline -6` output
3. Confirm `dsa110_continuum/source_finding/` package is importable
