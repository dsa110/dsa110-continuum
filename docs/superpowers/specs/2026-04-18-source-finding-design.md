# Source Finding (Step 8) — Design Spec

**Date:** 2026-04-18
**Author:** Jakob Faber / AI pair

---

## Goal

Extract the blind source-finding logic from `scripts/source_finding.py` into a
clean library module (`dsa110_continuum/source_finding/`), eliminate all global
state, add a `--sim` mode that runs against the pipeline's simulated mosaic, and
add CI-runnable tests that mock AegeanTools.

---

## Context

`scripts/source_finding.py` (221 lines) currently:

- Uses two module-level globals (`MOSAIC`, `CATALOG_OUT`) that are mutated in
  `main()`, making the functions untestable without running `main()`.
- Imports `AegeanTools` inside functions (good pattern — keep it), but the
  `run_aegean` function reads `CATALOG_OUT` directly instead of taking it as a
  parameter.
- Has no `--sim` flag, so cannot be driven by the pipeline CI harness.
- Has no tests.

`dsa110_continuum/photometry/aegean_fitting.py` provides **forced-position
Aegean fitting** via CLI subprocess. That module is unrelated and untouched by
this work — the new module handles **blind detection** via the Python API.

---

## File Structure

```
dsa110_continuum/source_finding/
    __init__.py        # exports: SourceCatalogEntry, run_bane, run_aegean,
                       #          write_catalog, write_empty_catalog, check_catalog,
                       #          run_source_finding
    core.py            # all pure functions

tests/
    test_source_finding.py   # 11 CI-runnable tests, AegeanTools mocked

scripts/source_finding.py    # refactored: ~60 lines, thin CLI wrapper
```

---

## Data Model

### `SourceCatalogEntry`

```python
@dataclass
class SourceCatalogEntry:
    source_name:       str
    ra_deg:            float
    dec_deg:           float
    peak_flux_jy:      float
    peak_flux_err_jy:  float
    int_flux_jy:       float
    a_arcsec:          float
    b_arcsec:          float
    pa_deg:            float
    local_rms_jy:      float
```

Column names match the existing script's `_write_catalog` output exactly,
preserving any downstream readers.

---

## Function Signatures

### `run_bane`

```python
def run_bane(
    mosaic_path: str | Path,
    *,
    box_size: int = 600,
    step_size: int = 300,
    cores: int = 1,
    skip_existing: bool = True,
) -> tuple[str, str]:
```

Returns `(bkg_path, rms_path)`.  Both are `{stem}_bkg.fits` / `{stem}_rms.fits`
next to the input mosaic.  Raises `RuntimeError` if BANE runs but does not
produce both output files.  When `skip_existing=True` and both files exist,
returns immediately without importing AegeanTools.

### `run_aegean`

```python
def run_aegean(
    mosaic_path: str | Path,
    bkg_path: str | Path,
    rms_path: str | Path,
    *,
    sigma: float = 7.0,
) -> list[SourceCatalogEntry]:
```

Returns a list (possibly empty) of `SourceCatalogEntry` objects.  Does **not**
write any files — that is the caller's responsibility.  `AegeanTools` is
imported inside this function; raises `ImportError` with an install hint if
absent.  Aegean `SourceFinder` is constructed with
`log=logging.getLogger("AegeanTools")`.

### `write_catalog`

```python
def write_catalog(entries: list[SourceCatalogEntry], out_path: str | Path) -> None:
```

Writes all 10 columns to a FITS binary table via `astropy.table.Table`.
Overwrites existing file.

### `write_empty_catalog`

```python
def write_empty_catalog(out_path: str | Path) -> None:
```

Writes a zero-row FITS table with the correct column schema.

### `check_catalog`

```python
def check_catalog(
    catalog_path: str | Path,
    *,
    sky_ra_range: tuple[float, float] = (300.0, 360.0),
    sky_dec_range: tuple[float, float] = (0.0, 40.0),
) -> bool:
```

Reads catalog, logs N sources, N bright (>1 Jy), N in sky window.  Returns
`True` if the catalog is non-empty (bright-source requirement is logged as
warning only, not a hard failure, since sim-mode dirty-image fluxes are
suppressed).

### `run_source_finding` (orchestrator)

```python
def run_source_finding(
    mosaic_path: str | Path,
    out_path: str | Path,
    *,
    bane_box: int = 600,
    bane_step: int = 300,
    aegean_sigma: float = 7.0,
) -> str:
```

Chains: `run_bane` → `run_aegean` → `write_catalog` or `write_empty_catalog` →
`check_catalog`.  Returns the catalog path as a string.

---

## Script (`scripts/source_finding.py`)

Refactored to ~60 lines.  All logic moves to `core.py`.

```
usage: source_finding.py [-h] [--mosaic PATH] [--out PATH]
                         [--sigma FLOAT] [--sim]

--mosaic PATH   Path to mosaic FITS. Default: /stage/dsa110-contimg/.../full_mosaic.fits
--out PATH      Catalog output path (default: {mosaic_stem}_sources.fits)
--sigma FLOAT   Detection threshold in σ (default: 7.0)
--sim           Use pipeline_outputs/step6/step6_mosaic.fits; prints stats, no hard QA exit
```

**Sim-mode exit gate:**
When `--sim` is supplied, the script runs the full pipeline (BANE → Aegean →
write → QA), prints the catalog stats, then exits 0 regardless of QA result
(matching the forced photometry sim-mode pattern: the dirty-image mosaic is not
expected to pass production bright-source criteria).

---

## Tests (`tests/test_source_finding.py`)

All tests use `tempfile.NamedTemporaryFile` (not `tmp_path` — sandbox
constraint).  AegeanTools is mocked via `unittest.mock.patch` on the string
`"dsa110_continuum.source_finding.core.AegeanTools"` (imported inside
`run_aegean` as a module reference).

| # | Test name | What it verifies |
|---|-----------|-----------------|
| 1 | `test_source_catalog_entry_fields` | Dataclass has all 10 fields, correct types |
| 2 | `test_write_catalog_roundtrip` | Write N entries → read back, columns and values match |
| 3 | `test_write_empty_catalog_schema` | Zero-row table has correct column names |
| 4 | `test_check_catalog_empty` | Returns False on zero-row table |
| 5 | `test_check_catalog_non_empty_no_bright` | Returns True, bright count = 0 (warning only) |
| 6 | `test_check_catalog_with_bright_source` | Returns True, bright count = 1 |
| 7 | `test_run_bane_skip_existing` | Creates stub bkg/rms files; `run_bane` returns without calling BANE |
| 8 | `test_run_bane_missing_output` | RuntimeError when BANE mock doesn't write expected files |
| 9 | `test_run_aegean_import_error` | `ImportError` with install hint when AegeanTools unavailable |
| 10 | `test_run_aegean_returns_empty_list` | Empty `found` → returns `[]` |
| 11 | `test_run_aegean_returns_entries` | Mock SourceFinder returns 2 stub sources → `list[SourceCatalogEntry]` with correct field mapping |

---

## Design Decisions

1. **Deferred AegeanTools import** — `run_bane` imports `AegeanTools.BANE` and
   `run_aegean` imports `AegeanTools.source_finder` inside the function body.
   This allows tests to mock them via `unittest.mock.patch` without
   `sys.modules` gymnastics.

2. **No `run_source_finding` test** — the orchestrator is a straight chain of
   tested primitives; a full integration test would require a real FITS mosaic
   and AegeanTools.  It is not tested directly in CI.  Sim-mode validation is
   done manually via `--sim`.

3. **`check_catalog` always returns True for non-empty** — the bright-source
   criterion is logged as a warning, not enforced, because the simulated
   dirty-image mosaic has suppressed fluxes.  Production operators can raise
   this to a hard failure once calibration is complete.

4. **`aegean_fitting.py` untouched** — it handles forced-position fitting; this
   module handles blind detection.  They share no code and have separate
   responsibilities.

5. **Column names preserved** — `SourceCatalogEntry` field names match the
   existing `_write_catalog` output, so no downstream reader breakage.
