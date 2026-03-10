# Phase 0 — QA Infrastructure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the test infrastructure that makes all subsequent pipeline work fast and objectively verifiable: a canary tile script (single-tile, ~15-min regression test) and a composite three-gate epoch QA metric with diagnostic output.

**Architecture:** Two independent deliverables. First, a shell wrapper (`scripts/run_canary.sh`) that runs the existing `run_pipeline.py` on one hardcoded tile and immediately runs `verify_sources.py` to print all three QA metrics. Second, a new Python module (`dsa110_continuum/photometry/epoch_qa.py`) implementing `measure_epoch_qa()`, wired into `batch_pipeline.py`'s `run_photometry_phase` so every epoch automatically produces a three-gate verdict and a diagnostic PNG.

**Tech Stack:** Python 3, `astropy.io.fits`, `astropy.wcs`, `numpy`, `matplotlib`, `sqlite3`, existing `dsa110_continuum.photometry.simple_peak`, existing `scripts/verify_sources.py`, existing `scripts/run_pipeline.py`.

---

## Resolved design decisions

These were open ambiguities identified in the planning review. All are closed before implementation begins.

### "Recovered" definition
A catalog source is **recovered** if the peak pixel value in a 3×3-pixel box centered on the catalog sky position exceeds 5× the **local RMS**, where local RMS = 1.4826 × median(|px|) computed inside a 50×50-pixel box centred on the source with the central 10×10 pixels masked out.

### Thermal noise floor
The SEFD/Tsys for DSA-110 is not yet measured (placeholder 50 K in `dsa110_measured_parameters.yaml`). The RMS gate is therefore anchored to the **empirically validated** value:

- Empirical floor (Jan 25 22h, 2026-03-09): **8.54 mJy/beam**
- QA gate: median mosaic RMS ≤ 2 × 8.54 = **17.1 mJy/beam**
- Constant: `QA_RMS_LIMIT_MJY = 17.1`
- Add a `# TODO: recompute when Tsys/SEFD is measured` comment wherever this constant appears.

### Detection completeness
- Denominator: NVSS sources ≥ 50 mJy within the mosaic footprint
- Numerator: sources recovered (definition above)
- Gate: completeness ≥ **60%**
- Minimum denominator: **5 sources**. If fewer than 5 NVSS sources ≥ 50 mJy lie in the footprint, mark the completeness gate as `SKIP` (not FAIL). This avoids spurious failures on low-source fields.
- Use the same NVSS SQLite DB as `verify_sources.py`: `/data/dsa110-contimg/state/catalogs/nvss_full.sqlite3`

### Canary tile completeness criterion
The canary tile (`2026-01-25T22:26:05`) contains 3C454.3 (~12.5 Jy) plus several NVSS sources ≥ 50 mJy. Minimum acceptable canary output:
- `median_ratio` ∈ [0.85, 1.15]
- `n_recovered ≥ 3`
- `mosaic_rms_mjy ≤ 17.1`

### `qa_summary.csv` schema
```
date,epoch_utc,mosaic_path,n_catalog,n_recovered,completeness_frac,
median_ratio,ratio_gate,completeness_gate,rms_gate,mosaic_rms_mjy,
qa_result,gaincal_used
```
`ratio_gate`, `completeness_gate`, `rms_gate` each hold `PASS`, `FAIL`, or `SKIP`.
`qa_result` is `PASS` only when all non-SKIP gates are PASS; otherwise `FAIL`.

---

## Task 1: Canary tile script

**Files:**
- Create: `scripts/run_canary.sh`
- No unit tests (end-to-end shell script; tested by running it)

**Context:** `scripts/run_pipeline.py` is already hardcoded to the canary tile
(`2026-01-25T22:26:05`). The canary script wraps it and pipes output into
`verify_sources.py`.

**Step 1: Create the script**

```bash
#!/usr/bin/env bash
# Canary tile regression test.
# Runs the single hardcoded reference tile (2026-01-25T22:26:05, contains 3C454.3)
# through the full calibrate → image → verify pipeline.
# Expected: median_ratio 0.85–1.15, n_recovered ≥ 3, RMS ≤ 17.1 mJy/beam
# Runtime: ~15–20 minutes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CANARY_FITS="/stage/dsa110-contimg/images/3c454/2026-01-25T22:26:05-image.fits"
NVSS_DB="/data/dsa110-contimg/state/catalogs/nvss_full.sqlite3"
OUT_CSV="/data/dsa110-continuum/outputs/canary/canary_$(date +%Y%m%dT%H%M%S).csv"

mkdir -p "$(dirname "$OUT_CSV")"

echo "=== DSA-110 Canary Tile Test ==="
echo "Tile: 2026-01-25T22:26:05 (contains 3C454.3 ~12.5 Jy)"
echo ""

# Step 1: Run pipeline
conda run -n casa6 python "$REPO_ROOT/scripts/run_pipeline.py"

# Step 2: Verify sources
conda run -n casa6 python "$REPO_ROOT/scripts/verify_sources.py" \
    --fits "$CANARY_FITS" \
    --nvss-db "$NVSS_DB" \
    --out "$OUT_CSV" \
    --min-flux-jy 0.050

echo ""
echo "Canary output saved: $OUT_CSV"
```

**Step 2: Make executable**

```bash
chmod +x scripts/run_canary.sh
```

**Step 3: Smoke-test (dry run)**

```bash
bash -n scripts/run_canary.sh && echo "syntax OK"
```
Expected: `syntax OK` with no errors.

**Step 4: Commit**

```bash
git add scripts/run_canary.sh
git commit -m "feat: add run_canary.sh single-tile regression script"
```

---

## Task 2: `epoch_qa.py` — composite QA metric

**Files:**
- Create: `dsa110_continuum/photometry/epoch_qa.py`
- Create: `tests/test_epoch_qa.py`

**Context:** This module is the heart of Phase 0. It takes a mosaic FITS path and an NVSS
SQLite DB path and returns a dataclass holding all three gate results.

**Step 1: Write the failing tests**

File: `tests/test_epoch_qa.py`

```python
"""Tests for dsa110_continuum.photometry.epoch_qa."""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from dsa110_continuum.photometry.epoch_qa import (
    EpochQAResult,
    QA_RATIO_LOW,
    QA_RATIO_HIGH,
    QA_COMPLETENESS_MIN,
    QA_RMS_LIMIT_MJY,
    measure_epoch_qa,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_fits(tmp_path: Path, rms_jy: float = 0.0085, n_sources: int = 6) -> Path:
    """Write a synthetic 500×500 FITS mosaic with Gaussian noise + point sources."""
    ny, nx = 500, 500
    rng = np.random.default_rng(42)
    data = rng.normal(0, rms_jy, (ny, nx)).astype(np.float32)

    # Embed point sources in a grid so they land at predictable sky positions
    source_flux = 0.5  # 500 mJy — well above 50 mJy NVSS threshold
    ys = np.linspace(100, 400, n_sources, dtype=int)
    xs = np.full(n_sources, nx // 2, dtype=int)
    for y, x in zip(ys, xs):
        data[y, x] += source_flux

    w = WCS(naxis=2)
    w.wcs.crpix = [nx // 2, ny // 2]
    w.wcs.cdelt = [-6.0 / 3600, 6.0 / 3600]   # 6 arcsec pixels
    w.wcs.crval = [45.0, 16.1]                  # RA=45°, Dec=+16.1°
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]

    hdr = w.to_header()
    hdr["BUNIT"] = "Jy/beam"
    hdu = fits.PrimaryHDU(data=data[np.newaxis, np.newaxis], header=hdr)
    out = tmp_path / "mosaic.fits"
    hdu.writeto(str(out), overwrite=True)
    return out


def _make_nvss_db(tmp_path: Path, sources: list[tuple[float, float, float]]) -> Path:
    """Write a minimal NVSS SQLite DB with (ra_deg, dec_deg, flux_jy) rows."""
    db_path = tmp_path / "nvss.sqlite3"
    con = sqlite3.connect(str(db_path))
    con.execute(
        "CREATE TABLE nvss_full (ra REAL, dec REAL, flux_20_cm REAL, name TEXT)"
    )
    con.executemany(
        "INSERT INTO nvss_full VALUES (?, ?, ?, ?)",
        [(ra, dec, flux, f"SRC_{i}") for i, (ra, dec, flux) in enumerate(sources)],
    )
    con.commit()
    con.close()
    return db_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_epoch_qa_result_is_pass_when_all_gates_pass(tmp_path):
    fits_path = _make_test_fits(tmp_path, rms_jy=0.0085, n_sources=6)
    # Build NVSS sources at the exact embedded source sky positions
    # (WCS maps pixel (250, y) to RA≈45°, Dec varies)
    sources = [(45.0, 16.1 + i * 0.05, 0.5) for i in range(6)]
    nvss_db = _make_nvss_db(tmp_path, sources)

    result = measure_epoch_qa(str(fits_path), str(nvss_db))

    assert isinstance(result, EpochQAResult)
    assert result.ratio_gate == "PASS"
    assert result.rms_gate == "PASS"
    assert result.qa_result == "PASS"


def test_epoch_qa_fails_when_rms_too_high(tmp_path):
    fits_path = _make_test_fits(tmp_path, rms_jy=0.050, n_sources=6)  # 50 mJy — too noisy
    sources = [(45.0, 16.1 + i * 0.05, 0.5) for i in range(6)]
    nvss_db = _make_nvss_db(tmp_path, sources)

    result = measure_epoch_qa(str(fits_path), str(nvss_db))

    assert result.rms_gate == "FAIL"
    assert result.qa_result == "FAIL"


def test_epoch_qa_completeness_skipped_with_few_catalog_sources(tmp_path):
    fits_path = _make_test_fits(tmp_path, rms_jy=0.0085, n_sources=6)
    # Only 3 NVSS sources in footprint — below the minimum of 5
    sources = [(45.0, 16.1 + i * 0.05, 0.5) for i in range(3)]
    nvss_db = _make_nvss_db(tmp_path, sources)

    result = measure_epoch_qa(str(fits_path), str(nvss_db))

    assert result.completeness_gate == "SKIP"


def test_epoch_qa_completeness_fails_when_too_few_recovered(tmp_path):
    # Image is mostly noise — sources not embedded — so completeness will be low
    fits_path = _make_test_fits(tmp_path, rms_jy=0.0085, n_sources=0)
    sources = [(45.0, 16.1 + i * 0.05, 0.5) for i in range(8)]
    nvss_db = _make_nvss_db(tmp_path, sources)

    result = measure_epoch_qa(str(fits_path), str(nvss_db))

    assert result.completeness_gate == "FAIL"
    assert result.qa_result == "FAIL"


def test_epoch_qa_result_to_dict_has_all_csv_columns(tmp_path):
    fits_path = _make_test_fits(tmp_path)
    nvss_db = _make_nvss_db(tmp_path, [(45.0, 16.1, 0.5)])
    result = measure_epoch_qa(str(fits_path), str(nvss_db))
    d = result.to_dict()

    required = {
        "n_catalog", "n_recovered", "completeness_frac",
        "median_ratio", "ratio_gate", "completeness_gate",
        "rms_gate", "mosaic_rms_mjy", "qa_result",
    }
    assert required.issubset(d.keys())
```

**Step 2: Run tests to confirm they fail**

```bash
cd /data/dsa110-continuum
conda run -n casa6 python -m pytest tests/test_epoch_qa.py -v 2>&1 | tail -20
```
Expected: `ImportError: cannot import name 'EpochQAResult'`

**Step 3: Implement `epoch_qa.py`**

File: `dsa110_continuum/photometry/epoch_qa.py`

```python
"""Composite epoch QA metric for DSA-110 mosaics.

Three independent gates must all pass for an epoch to be QA-PASS:
  1. Flux scale:        median DSA/NVSS ratio in [0.8, 1.2]
  2. Detection compl.:  >= 60% of NVSS sources >= 50 mJy recovered above 5-sigma local RMS
  3. Noise floor:       median mosaic RMS <= 17.1 mJy/beam

Design decisions are documented in docs/plans/2026-03-09-phase0-qa-infrastructure.md.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QA_RATIO_LOW: float = 0.8
QA_RATIO_HIGH: float = 1.2
QA_COMPLETENESS_MIN: float = 0.60
QA_MIN_CATALOG_SOURCES: int = 5      # fewer → completeness gate is SKIP
QA_MIN_FLUX_JY: float = 0.050        # 50 mJy NVSS threshold
QA_RECOVERY_SIGMA: float = 5.0       # detection threshold in units of local RMS

# TODO: recompute when Tsys/SEFD is measured in
#       dsa110_continuum/simulation/config/dsa110_measured_parameters.yaml
# Empirical floor: 8.54 mJy/beam (Jan 25 22h validated run, 2026-03-09)
QA_RMS_LIMIT_MJY: float = 17.1       # 2 × empirical floor

_GateResult = Literal["PASS", "FAIL", "SKIP"]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EpochQAResult:
    n_catalog: int
    n_recovered: int
    completeness_frac: float
    median_ratio: float
    mosaic_rms_mjy: float
    ratio_gate: _GateResult
    completeness_gate: _GateResult
    rms_gate: _GateResult
    qa_result: _GateResult          # PASS only if all non-SKIP gates pass

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _image_rms_mad(data: np.ndarray) -> float:
    """Global MAD-based RMS estimate (Jy/beam). Ignores NaN."""
    flat = data[np.isfinite(data)].ravel()
    return float(1.4826 * np.median(np.abs(flat - np.median(flat))))


def _local_rms(data: np.ndarray, cy: int, cx: int, outer: int = 25, inner: int = 5) -> float:
    """Local RMS in an annular box around (cy, cx). Returns global RMS if box is out of bounds."""
    ny, nx = data.shape
    y0, y1 = max(0, cy - outer), min(ny, cy + outer)
    x0, x1 = max(0, cx - outer), min(nx, cx + outer)
    box = data[y0:y1, x0:x1].copy()
    # Mask central region
    iy0 = cy - y0 - inner
    iy1 = cy - y0 + inner
    ix0 = cx - x0 - inner
    ix1 = cx - x0 + inner
    if iy0 >= 0 and iy1 <= box.shape[0] and ix0 >= 0 and ix1 <= box.shape[1]:
        box[max(0, iy0):iy1, max(0, ix0):ix1] = np.nan
    flat = box[np.isfinite(box)].ravel()
    if len(flat) < 10:
        return _image_rms_mad(data)
    return float(1.4826 * np.median(np.abs(flat - np.median(flat))))


def _peak_in_box(data: np.ndarray, cy: int, cx: int, half: int = 1) -> float:
    """Peak value in a (2*half+1)^2 box around (cy, cx)."""
    ny, nx = data.shape
    y0, y1 = max(0, cy - half), min(ny, cy + half + 1)
    x0, x1 = max(0, cx - half), min(nx, cx + half + 1)
    sub = data[y0:y1, x0:x1]
    if sub.size == 0:
        return 0.0
    return float(np.nanmax(sub))


def _nvss_sources_in_footprint(
    nvss_db: str,
    ra_min: float, ra_max: float,
    dec_min: float, dec_max: float,
    min_flux_jy: float = QA_MIN_FLUX_JY,
) -> list[tuple[float, float, float]]:
    """Return (ra, dec, flux_jy) for NVSS sources within the sky footprint."""
    try:
        con = sqlite3.connect(nvss_db)
        rows = con.execute(
            "SELECT ra, dec, flux_20_cm FROM nvss_full "
            "WHERE ra BETWEEN ? AND ? AND dec BETWEEN ? AND ? "
            "AND flux_20_cm >= ?",
            (ra_min, ra_max, dec_min, dec_max, min_flux_jy),
        ).fetchall()
        con.close()
        return [(float(r[0]), float(r[1]), float(r[2])) for r in rows]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def measure_epoch_qa(
    mosaic_fits: str,
    nvss_db: str,
    min_flux_jy: float = QA_MIN_FLUX_JY,
) -> EpochQAResult:
    """Compute the three-gate composite QA metric for one epoch mosaic.

    Parameters
    ----------
    mosaic_fits:
        Path to the epoch mosaic FITS file (Jy/beam, primary beam corrected).
    nvss_db:
        Path to the NVSS SQLite database (table ``nvss_full``).
    min_flux_jy:
        Minimum NVSS catalog flux for inclusion (default 50 mJy).

    Returns
    -------
    EpochQAResult
    """
    # --- Load image ---
    with fits.open(mosaic_fits) as hdul:
        hdr = hdul[0].header
        raw = hdul[0].data
    # Squeeze to 2D (handles 4D Stokes / freq axes)
    while raw.ndim > 2:
        raw = raw[0]
    data = raw.astype(np.float64)

    wcs = WCS(hdr, naxis=2)

    # --- Global RMS ---
    rms_jy = _image_rms_mad(data)
    rms_mjy = rms_jy * 1000.0

    # --- Sky footprint ---
    ny, nx = data.shape
    corners = np.array(
        [[0, 0], [nx - 1, 0], [0, ny - 1], [nx - 1, ny - 1],
         [nx // 2, 0], [nx // 2, ny - 1], [0, ny // 2], [nx - 1, ny // 2]],
        dtype=float,
    )
    sky = wcs.pixel_to_world(corners[:, 0], corners[:, 1])
    ra_vals = np.array([c.ra.deg for c in sky])
    dec_vals = np.array([c.dec.deg for c in sky])
    ra_min, ra_max = ra_vals.min(), ra_vals.max()
    dec_min, dec_max = dec_vals.min(), dec_vals.max()

    # --- Catalog sources ---
    catalog = _nvss_sources_in_footprint(nvss_db, ra_min, ra_max, dec_min, dec_max, min_flux_jy)
    n_catalog = len(catalog)

    # --- Measure each source ---
    ratios: list[float] = []
    n_recovered = 0

    for ra, dec, cat_flux in catalog:
        try:
            pix = wcs.world_to_pixel_values(ra, dec)
            cx, cy = int(round(pix[0])), int(round(pix[1]))
        except Exception:
            continue
        if not (2 <= cy < ny - 2 and 2 <= cx < nx - 2):
            continue

        local = _local_rms(data, cy, cx)
        peak = _peak_in_box(data, cy, cx, half=1)
        recovered = peak > QA_RECOVERY_SIGMA * local

        if recovered:
            n_recovered += 1
            ratios.append(peak / cat_flux)

    # --- Gate evaluation ---
    median_ratio = float(np.median(ratios)) if ratios else float("nan")
    completeness_frac = n_recovered / n_catalog if n_catalog > 0 else 0.0

    # Gate 1: flux scale
    if np.isnan(median_ratio) or len(ratios) < 3:
        ratio_gate: _GateResult = "FAIL"
    elif QA_RATIO_LOW <= median_ratio <= QA_RATIO_HIGH:
        ratio_gate = "PASS"
    else:
        ratio_gate = "FAIL"

    # Gate 2: detection completeness
    if n_catalog < QA_MIN_CATALOG_SOURCES:
        completeness_gate: _GateResult = "SKIP"
    elif completeness_frac >= QA_COMPLETENESS_MIN:
        completeness_gate = "PASS"
    else:
        completeness_gate = "FAIL"

    # Gate 3: noise floor
    # TODO: recompute QA_RMS_LIMIT_MJY when Tsys/SEFD is measured
    if rms_mjy <= QA_RMS_LIMIT_MJY:
        rms_gate: _GateResult = "PASS"
    else:
        rms_gate = "FAIL"

    # Overall verdict
    active_gates = [g for g in (ratio_gate, completeness_gate, rms_gate) if g != "SKIP"]
    qa_result: _GateResult = "PASS" if all(g == "PASS" for g in active_gates) else "FAIL"

    return EpochQAResult(
        n_catalog=n_catalog,
        n_recovered=n_recovered,
        completeness_frac=round(completeness_frac, 4),
        median_ratio=round(median_ratio, 4) if not np.isnan(median_ratio) else float("nan"),
        mosaic_rms_mjy=round(rms_mjy, 3),
        ratio_gate=ratio_gate,
        completeness_gate=completeness_gate,
        rms_gate=rms_gate,
        qa_result=qa_result,
    )
```

**Step 4: Run tests to confirm they pass**

```bash
cd /data/dsa110-continuum
conda run -n casa6 python -m pytest tests/test_epoch_qa.py -v 2>&1 | tail -25
```
Expected: 5 tests PASS.

**Step 5: Commit**

```bash
git add dsa110_continuum/photometry/epoch_qa.py tests/test_epoch_qa.py
git commit -m "feat(qa): add EpochQAResult and measure_epoch_qa three-gate metric"
```

---

## Task 3: Diagnostic PNG generator

**Files:**
- Create: `dsa110_continuum/photometry/epoch_qa_plot.py`
- Create: `tests/test_epoch_qa_plot.py`

**Context:** Called after `measure_epoch_qa()` to produce a one-page diagnostic PNG
saved alongside the epoch mosaic FITS. This is the human-readable verdict.

**Step 1: Write the failing test**

File: `tests/test_epoch_qa_plot.py`

```python
"""Tests for epoch_qa_plot.py."""
import tempfile
from pathlib import Path
import numpy as np
import pytest
from dsa110_continuum.photometry.epoch_qa import EpochQAResult
from dsa110_continuum.photometry.epoch_qa_plot import plot_epoch_qa


def _dummy_result(qa_result="PASS") -> EpochQAResult:
    return EpochQAResult(
        n_catalog=20, n_recovered=14, completeness_frac=0.70,
        median_ratio=0.93, mosaic_rms_mjy=8.6,
        ratio_gate="PASS", completeness_gate="PASS", rms_gate="PASS",
        qa_result=qa_result,
    )


def test_plot_epoch_qa_creates_png(tmp_path):
    out = tmp_path / "qa_diag.png"
    ratios = np.random.uniform(0.7, 1.3, 20).tolist()
    tile_rms = np.random.uniform(7, 12, 11).tolist()
    plot_epoch_qa(_dummy_result(), ratios, tile_rms, str(out), epoch_label="2026-01-25T2200")
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_epoch_qa_fail_result_shows_red_title(tmp_path):
    out = tmp_path / "qa_fail.png"
    plot_epoch_qa(_dummy_result("FAIL"), [], [], str(out), epoch_label="2026-02-15T0000")
    assert out.exists()
```

**Step 2: Run to confirm failure**

```bash
conda run -n casa6 python -m pytest tests/test_epoch_qa_plot.py -v 2>&1 | tail -10
```
Expected: `ImportError: cannot import name 'plot_epoch_qa'`

**Step 3: Implement `epoch_qa_plot.py`**

File: `dsa110_continuum/photometry/epoch_qa_plot.py`

```python
"""Diagnostic PNG for epoch QA results.

Three-panel figure:
  Left:   DSA/NVSS ratio histogram with [0.8, 1.2] acceptance band
  Centre: Detection completeness bar with 60% threshold
  Right:  Per-tile RMS bar chart with 17.1 mJy/beam limit line

Title shows overall PASS (green) or FAIL (red) verdict.
"""
from __future__ import annotations

import numpy as np

from dsa110_continuum.photometry.epoch_qa import (
    EpochQAResult,
    QA_RATIO_LOW, QA_RATIO_HIGH,
    QA_COMPLETENESS_MIN,
    QA_RMS_LIMIT_MJY,
)


def plot_epoch_qa(
    result: EpochQAResult,
    ratios: list[float],
    tile_rms_mjy: list[float],
    out_path: str,
    epoch_label: str = "",
) -> None:
    """Write a three-panel QA diagnostic PNG.

    Parameters
    ----------
    result:
        Output of ``measure_epoch_qa()``.
    ratios:
        Per-source DSA/NVSS flux ratios (used for histogram).
    tile_rms_mjy:
        Per-tile RMS values in mJy/beam (one entry per mosaic tile).
    out_path:
        Destination PNG file path.
    epoch_label:
        Human-readable epoch string shown in the figure title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    verdict_color = "#2ecc71" if result.qa_result == "PASS" else "#e74c3c"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#16213e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#4ec9b0")
        ax.tick_params(colors="#e0e0e0")
        ax.xaxis.label.set_color("#e0e0e0")
        ax.yaxis.label.set_color("#e0e0e0")
        ax.title.set_color("#4ec9b0")

    # --- Panel 1: Ratio histogram ---
    ax = axes[0]
    if ratios:
        ax.hist(ratios, bins=20, range=(0, 2), color="#4ec9b0", edgecolor="#1a1a2e", alpha=0.85)
        ax.axvspan(QA_RATIO_LOW, QA_RATIO_HIGH, alpha=0.15, color="#2ecc71", label="Pass band")
        ax.axvline(np.median(ratios), color="#f39c12", linewidth=2,
                   label=f"median={np.median(ratios):.2f}")
        ax.legend(fontsize=8, labelcolor="#e0e0e0", facecolor="#1a1a2e")
    else:
        ax.text(0.5, 0.5, "No detections", ha="center", va="center",
                color="#e74c3c", transform=ax.transAxes)
    ax.set_xlabel("DSA / NVSS flux ratio")
    ax.set_ylabel("N sources")
    gate_str = f"[{result.ratio_gate}]"
    ax.set_title(f"Flux scale  {gate_str}")

    # --- Panel 2: Detection completeness ---
    ax = axes[1]
    pct = result.completeness_frac * 100
    bar_color = "#2ecc71" if result.completeness_gate == "PASS" else (
        "#95a5a6" if result.completeness_gate == "SKIP" else "#e74c3c"
    )
    ax.bar(["Completeness"], [pct], color=bar_color, edgecolor="#1a1a2e")
    ax.axhline(QA_COMPLETENESS_MIN * 100, color="#f39c12", linewidth=2,
               linestyle="--", label=f"{QA_COMPLETENESS_MIN*100:.0f}% threshold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("% recovered")
    ax.legend(fontsize=8, labelcolor="#e0e0e0", facecolor="#1a1a2e")
    gate_str = f"[{result.completeness_gate}]"
    n_str = f"{result.n_recovered}/{result.n_catalog}"
    ax.set_title(f"Completeness {n_str}  {gate_str}")

    # --- Panel 3: Per-tile RMS ---
    ax = axes[2]
    if tile_rms_mjy:
        colors = ["#2ecc71" if r <= QA_RMS_LIMIT_MJY else "#e74c3c" for r in tile_rms_mjy]
        ax.bar(range(len(tile_rms_mjy)), tile_rms_mjy, color=colors, edgecolor="#1a1a2e")
        ax.axhline(QA_RMS_LIMIT_MJY, color="#f39c12", linewidth=2,
                   linestyle="--", label=f"Limit {QA_RMS_LIMIT_MJY} mJy")
        ax.legend(fontsize=8, labelcolor="#e0e0e0", facecolor="#1a1a2e")
    else:
        ax.bar([0], [result.mosaic_rms_mjy], color="#4ec9b0", edgecolor="#1a1a2e")
        ax.axhline(QA_RMS_LIMIT_MJY, color="#f39c12", linewidth=2, linestyle="--")
    ax.set_xlabel("Tile index")
    ax.set_ylabel("RMS (mJy/beam)")
    gate_str = f"[{result.rms_gate}]"
    ax.set_title(f"Noise floor  {gate_str}")

    # --- Overall title ---
    verdict = result.qa_result
    title = f"Epoch QA — {epoch_label}   Overall: {verdict}"
    fig.suptitle(title, fontsize=13, color=verdict_color, fontweight="bold", y=1.02)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
```

**Step 4: Run tests to confirm they pass**

```bash
conda run -n casa6 python -m pytest tests/test_epoch_qa_plot.py -v 2>&1 | tail -10
```
Expected: 2 tests PASS.

**Step 5: Commit**

```bash
git add dsa110_continuum/photometry/epoch_qa_plot.py tests/test_epoch_qa_plot.py
git commit -m "feat(qa): add epoch_qa_plot three-panel diagnostic PNG generator"
```

---

## Task 4: Wire into `batch_pipeline.py`

**Files:**
- Modify: `scripts/batch_pipeline.py` — `run_photometry_phase()` and `emit_run_summary()`
- Modify: `scripts/batch_pipeline.py` — `qa_summary.csv` write logic

**Context:** `run_photometry_phase()` (line ~192) currently calls `verify_sources.py` via
subprocess and parses its stdout. Replace the inner call with a direct call to
`measure_epoch_qa()` and save the expanded result to `qa_summary.csv`. The per-tile RMS
values come from `mosaic_stats()` (line ~267) — pass them to `plot_epoch_qa()`.

**There is no unit test for this task** — integration-level behaviour is validated by running
the canary script after this change (Task 5 below).

**Step 1: Add imports at top of `batch_pipeline.py`**

Find the existing imports block and add:
```python
from dsa110_continuum.photometry.epoch_qa import measure_epoch_qa, EpochQAResult
from dsa110_continuum.photometry.epoch_qa_plot import plot_epoch_qa
```

**Step 2: Replace QA logic in `run_photometry_phase()`**

Find `def run_photometry_phase(mosaic_path: str)` (~line 192). The function currently
runs `verify_sources.py` via subprocess. Replace the body to call `measure_epoch_qa()`
directly and return the `EpochQAResult` alongside the existing photometry rows:

```python
def run_photometry_phase(
    mosaic_path: str,
    nvss_db: str = "/data/dsa110-contimg/state/catalogs/nvss_full.sqlite3",
) -> tuple[list[dict] | None, EpochQAResult | None]:
    """Run forced photometry + composite QA on a mosaic. Returns (rows, qa_result)."""
    import subprocess, json
    from pathlib import Path

    rows = None
    qa = None

    # Existing forced photometry (unchanged)
    try:
        phot_csv = Path(mosaic_path).with_suffix(".phot.csv")
        result = subprocess.run(
            ["conda", "run", "-n", "casa6", "python",
             str(Path(__file__).parent / "verify_sources.py"),
             "--fits", mosaic_path, "--out", str(phot_csv)],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0 and phot_csv.exists():
            import csv as _csv
            with open(phot_csv) as f:
                rows = list(_csv.DictReader(f))
    except Exception as e:
        log.warning("run_photometry_phase: forced photometry failed: %s", e)

    # Composite QA metric
    try:
        qa = measure_epoch_qa(mosaic_path, nvss_db)
        log.info(
            "Epoch QA: ratio=%s gate=%s | completeness=%s gate=%s | rms=%.1f mJy gate=%s | OVERALL=%s",
            f"{qa.median_ratio:.3f}", qa.ratio_gate,
            f"{qa.completeness_frac:.2f}", qa.completeness_gate,
            qa.mosaic_rms_mjy, qa.rms_gate,
            qa.qa_result,
        )
    except Exception as e:
        log.warning("run_photometry_phase: measure_epoch_qa failed: %s", e)

    return rows, qa
```

**Step 3: Update callers of `run_photometry_phase()`**

Search for all calls to `run_photometry_phase(` in `batch_pipeline.py` and update them to
unpack the tuple: `rows, qa = run_photometry_phase(mosaic_path)`.

**Step 4: Write diagnostic PNG after QA**

Immediately after calling `run_photometry_phase()`, add:

```python
if qa is not None:
    diag_png = str(Path(mosaic_path).with_suffix(".qa_diag.png"))
    try:
        tile_rms = [mosaic_stats(t)[0] * 1000 for t in epoch_tiles if os.path.exists(t)]
        ratios_list = [r["ratio"] for r in (rows or []) if r.get("ratio") not in ("", None)]
        ratios_floats = [float(x) for x in ratios_list if x]
        plot_epoch_qa(qa, ratios_floats, tile_rms, diag_png, epoch_label=label)
        log.info("QA diagnostic PNG: %s", diag_png)
    except Exception as e:
        log.warning("Could not generate QA diagnostic PNG: %s", e)
```

**Step 5: Update `qa_summary.csv` write to include all three gate columns**

Find where `qa_summary.csv` is written (search for `qa_summary`). Update the fieldnames
and row dict to include the full schema:

```python
QA_CSV_FIELDS = [
    "date", "epoch_utc", "mosaic_path",
    "n_catalog", "n_recovered", "completeness_frac",
    "median_ratio", "ratio_gate", "completeness_gate",
    "rms_gate", "mosaic_rms_mjy",
    "qa_result", "gaincal_used",
]
```

**Step 6: Verify syntax**

```bash
cd /data/dsa110-continuum
python -c "import ast; ast.parse(open('scripts/batch_pipeline.py').read()); print('syntax OK')"
```
Expected: `syntax OK`

**Step 7: Commit**

```bash
git add scripts/batch_pipeline.py
git commit -m "feat(pipeline): wire measure_epoch_qa into run_photometry_phase, expand qa_summary.csv"
```

---

## Task 5: End-to-end validation via canary

**This is the acceptance test for the entire Phase 0.**

**Step 1: Run the canary script**

```bash
cd /data/dsa110-continuum
bash scripts/run_canary.sh 2>&1 | tee outputs/canary/latest_canary.log
```

Expected output (last lines):
```
VERIFY PASS: median_ratio=0.93 n_sources=...
QA diagnostic PNG: /stage/dsa110-contimg/images/3c454/2026-01-25T22:26:05-image.qa_diag.png
```

**Step 2: Inspect the diagnostic PNG**

```bash
ls -lh /stage/dsa110-contimg/images/3c454/*.qa_diag.png
cp /stage/dsa110-contimg/images/3c454/*.qa_diag.png \
   /data/dsa110-continuum/outputs/diagnostics/$(date +%Y-%m-%d)/canary_qa_diag.png
```

**Step 3: Confirm `outputs/canary/` CSV has all columns**

```bash
head -2 /data/dsa110-continuum/outputs/canary/canary_*.csv
```
Expected: header row contains `ratio_gate,completeness_gate,rms_gate,qa_result`.

**Step 4: Final commit**

```bash
git add outputs/canary/.gitkeep
git commit -m "test(canary): add canary output directory placeholder"
```

---

## Done

Phase 0 is complete when:
- [ ] `scripts/run_canary.sh` runs in ~15–20 min and prints a three-metric QA verdict
- [ ] `measure_epoch_qa()` passes all 5 unit tests
- [ ] `plot_epoch_qa()` passes all 2 unit tests
- [ ] Every `batch_pipeline.py` epoch run writes a `.qa_diag.png` and expanded `qa_summary.csv`
- [ ] Canary run produces `qa_result=PASS` with `median_ratio` ∈ [0.85, 1.15]
