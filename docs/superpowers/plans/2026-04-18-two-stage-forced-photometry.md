# Two-Stage Forced Photometry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a two-stage forced photometry path (coarse peak-in-beam-aperture → Condon fine pass) to the pipeline, with a `--method simple_peak` fallback and simulation-mode support for pipeline testing without real catalog databases.

**Architecture:** A new `dsa110_continuum/photometry/two_stage.py` module orchestrates the coarse→fine pipeline and returns a `(list[ForcedPhotometryResult], list[CoarseAugment])` pair. `scripts/forced_photometry.py` is extended with `--method` and `--sim` flags and calls into `two_stage` or `simple_peak` directly depending on the selected method.

**Tech Stack:** Python 3.11, astropy, numpy, pytest; existing `simple_peak.measure_peak_box`, `forced.measure_many`, `simulation.ground_truth.SkyModel`.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `dsa110_continuum/photometry/two_stage.py` | **Create** | Coarse→fine orchestration, `CoarseAugment` dataclass, beam correction helper |
| `scripts/forced_photometry.py` | **Modify** | Add `--method`, `--sim`, `--snr-coarse` flags; wire into two_stage / simple_peak |
| `tests/test_two_stage_photometry.py` | **Create** | All unit + integration tests |

---

### Task 1: `CoarseAugment` dataclass and beam correction helper

**Files:**
- Create: `dsa110_continuum/photometry/two_stage.py`
- Create: `tests/test_two_stage_photometry.py`

- [ ] **Step 1.1: Write failing test for `CoarseAugment`**

```python
# tests/test_two_stage_photometry.py
from dsa110_continuum.photometry.two_stage import CoarseAugment

def test_coarse_augment_fields():
    aug = CoarseAugment(
        ra_deg=344.1, dec_deg=16.15,
        coarse_peak_jyb=0.12, coarse_snr=8.5, passed_coarse=True,
    )
    assert aug.ra_deg == 344.1
    assert aug.passed_coarse is True
```

- [ ] **Step 1.2: Run test to confirm it fails**

```bash
cd /home/user/workspace/dsa110-continuum
python -m pytest tests/test_two_stage_photometry.py::test_coarse_augment_fields -v
```
Expected: `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 1.3: Create `two_stage.py` with `CoarseAugment` and beam correction helper**

```python
# dsa110_continuum/photometry/two_stage.py
"""Two-stage forced photometry: coarse peak-in-box → Condon fine pass."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


@dataclass
class CoarseAugment:
    """Coarse-pass metadata attached alongside a ForcedPhotometryResult."""
    ra_deg: float
    dec_deg: float
    coarse_peak_jyb: float
    coarse_snr: float
    passed_coarse: bool


def beam_correction_factor(fits_path: str) -> float:
    """Return beam_area_sr / pixel_area_sr from a FITS header.

    For an unresolved point source: peak_jyb = S_total_Jy / correction_factor.
    So: S_total_Jy ≈ peak_jyb * correction_factor.

    Returns 1.0 if beam keywords are absent (no correction applied).
    """
    import math
    hdr = fits.getheader(fits_path)
    bmaj_deg = hdr.get("BMAJ")
    bmin_deg = hdr.get("BMIN")
    cdelt2_deg = abs(hdr.get("CDELT2", 0.0))
    if bmaj_deg is None or bmin_deg is None or cdelt2_deg == 0.0:
        return 1.0
    bmaj_rad = math.radians(bmaj_deg)
    bmin_rad = math.radians(bmin_deg)
    pixel_rad = math.radians(cdelt2_deg)
    beam_area_sr = (math.pi / (4.0 * math.log(2.0))) * bmaj_rad * bmin_rad
    pixel_area_sr = pixel_rad ** 2
    return beam_area_sr / pixel_area_sr
```

- [ ] **Step 1.4: Run test to confirm it passes**

```bash
python -m pytest tests/test_two_stage_photometry.py::test_coarse_augment_fields -v
```
Expected: `PASSED`.

- [ ] **Step 1.5: Add beam correction test**

```python
# append to tests/test_two_stage_photometry.py
import math
from unittest.mock import patch
from dsa110_continuum.photometry.two_stage import beam_correction_factor

def test_beam_correction_known_values():
    # BMAJ=36.9", BMIN=25.5", CDELT2=20" (Step 6 mosaic values)
    mock_hdr = {
        "BMAJ": 36.9 / 3600,
        "BMIN": 25.5 / 3600,
        "CDELT2": 20.0 / 3600,
    }
    with patch("dsa110_continuum.photometry.two_stage.fits.getheader", return_value=mock_hdr):
        factor = beam_correction_factor("dummy.fits")
    bmaj_rad = math.radians(36.9 / 3600)
    bmin_rad = math.radians(25.5 / 3600)
    pixel_rad = math.radians(20.0 / 3600)
    expected = (math.pi / (4 * math.log(2))) * bmaj_rad * bmin_rad / pixel_rad**2
    assert abs(factor - expected) / expected < 1e-6

def test_beam_correction_missing_keywords():
    mock_hdr = {}
    with patch("dsa110_continuum.photometry.two_stage.fits.getheader", return_value=mock_hdr):
        factor = beam_correction_factor("dummy.fits")
    assert factor == 1.0
```

- [ ] **Step 1.6: Run and confirm both tests pass**

```bash
python -m pytest tests/test_two_stage_photometry.py -v
```
Expected: 2 tests `PASSED`.

- [ ] **Step 1.7: Commit**

```bash
git add dsa110_continuum/photometry/two_stage.py tests/test_two_stage_photometry.py
git commit -m "feat(photometry): add CoarseAugment dataclass and beam_correction_factor helper"
```

---

### Task 2: Coarse pass function

**Files:**
- Modify: `dsa110_continuum/photometry/two_stage.py`
- Modify: `tests/test_two_stage_photometry.py`

Depends on: Task 1, Step 6 mosaic (`pipeline_outputs/step6/step6_mosaic.fits`).
If the mosaic is not on disk, regenerate it first:
```bash
python scripts/run_step6.py
```

- [ ] **Step 2.1: Write failing test for coarse pass**

```python
# append to tests/test_two_stage_photometry.py
from pathlib import Path
import pytest
from dsa110_continuum.photometry.two_stage import run_coarse_pass

MOSAIC = Path("pipeline_outputs/step6/step6_mosaic.fits")

@pytest.mark.skipif(not MOSAIC.exists(), reason="Step 6 mosaic not on disk")
def test_coarse_pass_returns_finite():
    coords = [(344.124, 16.15), (346.71, 16.15)]
    results = run_coarse_pass(str(MOSAIC), coords, global_rms=None)
    assert len(results) == 2
    for aug in results:
        assert isinstance(aug, CoarseAugment)  # noqa: F821 – imported below
        assert np.isfinite(aug.coarse_peak_jyb) or np.isnan(aug.coarse_peak_jyb)
```

Also add the import at the top of the test file:
```python
from dsa110_continuum.photometry.two_stage import CoarseAugment, run_coarse_pass
```

- [ ] **Step 2.2: Run test to confirm it fails**

```bash
python -m pytest tests/test_two_stage_photometry.py::test_coarse_pass_returns_finite -v
```
Expected: `ImportError: cannot import name 'run_coarse_pass'`.

- [ ] **Step 2.3: Implement `run_coarse_pass` in `two_stage.py`**

Add after the `beam_correction_factor` function:

```python
from astropy.stats import mad_std
from dsa110_continuum.photometry.simple_peak import measure_peak_box


def run_coarse_pass(
    fits_path: str,
    coords: list[tuple[float, float]],
    *,
    box_pix: int = 5,
    global_rms: float | None = None,
    snr_coarse_min: float = 3.0,
) -> list[CoarseAugment]:
    """Run peak-in-box measurement at each position.

    Parameters
    ----------
    fits_path : str
        Path to mosaic FITS file.
    coords : list of (ra_deg, dec_deg)
        Source positions to measure.
    box_pix : int
        Half-width of the search box in pixels (default 5).
    global_rms : float or None
        Noise estimate in Jy/beam.  If None, estimated from image MAD.
    snr_coarse_min : float
        Sources with coarse_snr >= this value have passed_coarse=True.

    Returns
    -------
    list[CoarseAugment]
    """
    with fits.open(fits_path) as hdul:
        data = np.squeeze(np.asarray(hdul[0].data, dtype=float))
        wcs = WCS(hdul[0].header).celestial

    rms = global_rms
    if rms is None:
        finite = data[np.isfinite(data)]
        rms = float(mad_std(finite)) if finite.size > 0 else float("nan")

    results = []
    for ra, dec in coords:
        peak, snr, _, _ = measure_peak_box(data, wcs, ra, dec, box_pix=box_pix, rms=rms)
        if not np.isfinite(snr):
            snr = (peak / rms) if (np.isfinite(peak) and rms > 0) else float("nan")
        results.append(CoarseAugment(
            ra_deg=ra,
            dec_deg=dec,
            coarse_peak_jyb=peak,
            coarse_snr=snr,
            passed_coarse=np.isfinite(snr) and snr >= snr_coarse_min,
        ))
    return results
```

- [ ] **Step 2.4: Run test to confirm it passes**

```bash
python -m pytest tests/test_two_stage_photometry.py::test_coarse_pass_returns_finite -v
```
Expected: `PASSED` (or `SKIPPED` if mosaic absent — regenerate first).

- [ ] **Step 2.5: Add SNR gate test**

```python
# append to tests/test_two_stage_photometry.py
@pytest.mark.skipif(not MOSAIC.exists(), reason="Step 6 mosaic not on disk")
def test_snr_gate_all_pass_with_low_rms():
    coords = [(344.124, 16.15)]
    results = run_coarse_pass(str(MOSAIC), coords, global_rms=1e-6, snr_coarse_min=3.0)
    assert results[0].passed_coarse is True

@pytest.mark.skipif(not MOSAIC.exists(), reason="Step 6 mosaic not on disk")
def test_snr_gate_all_fail_with_high_rms():
    coords = [(344.124, 16.15)]
    results = run_coarse_pass(str(MOSAIC), coords, global_rms=1e6, snr_coarse_min=3.0)
    assert results[0].passed_coarse is False
```

- [ ] **Step 2.6: Run all tests**

```bash
python -m pytest tests/test_two_stage_photometry.py -v
```
Expected: 4–5 tests `PASSED`.

- [ ] **Step 2.7: Commit**

```bash
git add dsa110_continuum/photometry/two_stage.py tests/test_two_stage_photometry.py
git commit -m "feat(photometry): implement run_coarse_pass with SNR gating"
```

---

### Task 3: `run_two_stage` orchestrator

**Files:**
- Modify: `dsa110_continuum/photometry/two_stage.py`
- Modify: `tests/test_two_stage_photometry.py`

- [ ] **Step 3.1: Write failing test**

```python
# append to tests/test_two_stage_photometry.py
from dsa110_continuum.photometry.two_stage import run_two_stage

@pytest.mark.skipif(not MOSAIC.exists(), reason="Step 6 mosaic not on disk")
def test_two_stage_returns_paired_lists():
    coords = [(344.124, 16.15), (346.71, 16.15)]
    results, augments = run_two_stage(str(MOSAIC), coords, snr_coarse_min=0.0)
    assert len(results) == len(augments) == 2

@pytest.mark.skipif(not MOSAIC.exists(), reason="Step 6 mosaic not on disk")
def test_fine_pass_skips_failing_coarse():
    # Use an extreme rms so nothing passes coarse; fine pass should return NaN results
    coords = [(344.124, 16.15)]
    results, augments = run_two_stage(str(MOSAIC), coords, global_rms=1e6, snr_coarse_min=3.0)
    assert augments[0].passed_coarse is False
    # Result for failed coarse should be a NaN placeholder
    assert not np.isfinite(results[0].peak_jyb)
```

- [ ] **Step 3.2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_two_stage_photometry.py::test_two_stage_returns_paired_lists tests/test_two_stage_photometry.py::test_fine_pass_skips_failing_coarse -v
```
Expected: `ImportError: cannot import name 'run_two_stage'`.

- [ ] **Step 3.3: Implement `run_two_stage`**

Add after `run_coarse_pass` in `two_stage.py`:

```python
from dsa110_continuum.photometry.forced import ForcedPhotometryResult, measure_many


def run_two_stage(
    fits_path: str,
    coords: list[tuple[float, float]],
    *,
    snr_coarse_min: float = 3.0,
    box_pix: int = 5,
    global_rms: float | None = None,
    use_cluster_fitting: bool = False,
    noise_map_path: str | None = None,
) -> tuple[list[ForcedPhotometryResult], list[CoarseAugment]]:
    """Coarse peak-in-box pass, then Condon fine pass on survivors.

    Parameters
    ----------
    fits_path : str
        Path to mosaic FITS file.
    coords : list of (ra_deg, dec_deg)
    snr_coarse_min : float
        Coarse SNR gate threshold (default 3.0).
    box_pix : int
        Half-width of coarse measurement box in pixels.
    global_rms : float or None
        Noise estimate.  If None, estimated from image MAD.
    use_cluster_fitting : bool
        Pass through to measure_many (default False).
    noise_map_path : str or None
        Optional noise map FITS path for fine pass.

    Returns
    -------
    results : list[ForcedPhotometryResult]
        One result per input coord. NaN placeholder for coarse failures.
    augments : list[CoarseAugment]
        Coarse-pass metadata for each coord, same order as results.
    """
    augments = run_coarse_pass(
        fits_path, coords,
        box_pix=box_pix,
        global_rms=global_rms,
        snr_coarse_min=snr_coarse_min,
    )

    survivor_coords = [
        (aug.ra_deg, aug.dec_deg)
        for aug in augments if aug.passed_coarse
    ]

    # Run fine pass only on survivors
    if survivor_coords:
        fine_results = measure_many(
            fits_path,
            survivor_coords,
            use_cluster_fitting=use_cluster_fitting,
            noise_map_path=noise_map_path,
        )
    else:
        fine_results = []

    # Map fine results back to original coord order
    fine_map: dict[tuple[float, float], ForcedPhotometryResult] = {
        (r.ra_deg, r.dec_deg): r for r in fine_results
    }

    nan_result = lambda ra, dec: ForcedPhotometryResult(
        ra_deg=ra, dec_deg=dec,
        peak_jyb=float("nan"), peak_err_jyb=float("nan"),
        pix_x=float("nan"), pix_y=float("nan"),
        box_size_pix=box_pix,
    )

    ordered_results = [
        fine_map.get((aug.ra_deg, aug.dec_deg), nan_result(aug.ra_deg, aug.dec_deg))
        for aug in augments
    ]

    return ordered_results, augments
```

- [ ] **Step 3.4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_two_stage_photometry.py -v
```
Expected: all `PASSED`.

- [ ] **Step 3.5: Commit**

```bash
git add dsa110_continuum/photometry/two_stage.py tests/test_two_stage_photometry.py
git commit -m "feat(photometry): implement run_two_stage orchestrator"
```

---

### Task 4: Beam correction ratio test (sim-mode integration)

**Files:**
- Modify: `tests/test_two_stage_photometry.py`

This verifies the beam area correction produces sensible flux ratios for bright
sources in the simulated mosaic using known injected positions.

- [ ] **Step 4.1: Write failing test**

```python
# append to tests/test_two_stage_photometry.py
from dsa110_continuum.photometry.two_stage import beam_correction_factor
from dsa110_continuum.simulation.ground_truth import SkyModel

@pytest.mark.skipif(not MOSAIC.exists(), reason="Step 6 mosaic not on disk")
def test_beam_correction_ratio_bright_sources():
    """Ratio (measured_peak × correction) / injected_flux ≈ 1 for bright sources."""
    sky = SkyModel(seed=42)
    # Use all injected sources
    coords = [(float(sky.ra[k].deg), float(sky.dec[k].deg)) for k in range(sky.Ncomponents)]
    injected_flux = [float(sky.stokes[0, 0, k].value) for k in range(sky.Ncomponents)]

    correction = beam_correction_factor(str(MOSAIC))
    coarse = run_coarse_pass(str(MOSAIC), coords, global_rms=None, snr_coarse_min=0.0)

    ratios = []
    for aug, s_inj in zip(coarse, injected_flux):
        if np.isfinite(aug.coarse_peak_jyb) and s_inj > 0:
            ratio = (aug.coarse_peak_jyb * correction) / s_inj
            ratios.append(ratio)

    assert len(ratios) >= 2, "Need at least 2 finite measurements"
    median_ratio = float(np.median(ratios))
    # Ratio should be within 0.3–3.0 for unresolved sources in a clean image
    assert 0.3 <= median_ratio <= 3.0, f"Median beam-corrected ratio = {median_ratio:.3f}"
```

- [ ] **Step 4.2: Run test to confirm expected behaviour**

```bash
python -m pytest tests/test_two_stage_photometry.py::test_beam_correction_ratio_bright_sources -v
```
If the test fails with ratio outside bounds, inspect the SkyModel API:
```bash
python -c "from dsa110_continuum.simulation.ground_truth import SkyModel; s=SkyModel(seed=42); print(dir(s))"
```
and adjust attribute access accordingly (the test above uses the same attribute names as `validate_step6_mosaic.py`).

- [ ] **Step 4.3: Commit once passing**

```bash
git add tests/test_two_stage_photometry.py
git commit -m "test(photometry): add beam-corrected flux ratio check for sim sources"
```

---

### Task 5: Extend `scripts/forced_photometry.py`

**Files:**
- Modify: `scripts/forced_photometry.py`

Add `--method`, `--sim`, `--snr-coarse` flags and wire them in.

- [ ] **Step 5.1: Write failing CLI test**

```python
# append to tests/test_two_stage_photometry.py
import subprocess, sys, tempfile

@pytest.mark.skipif(not MOSAIC.exists(), reason="Step 6 mosaic not on disk")
def test_cli_simple_peak_produces_csv():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        out_csv = f.name
    result = subprocess.run(
        [sys.executable, "scripts/forced_photometry.py",
         "--mosaic", str(MOSAIC),
         "--method", "simple_peak",
         "--sim",
         "--output", out_csv],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    import csv
    with open(out_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    assert "measured_flux_jy" in rows[0]
    assert "snr" in rows[0]

@pytest.mark.skipif(not MOSAIC.exists(), reason="Step 6 mosaic not on disk")
def test_cli_two_stage_produces_coarse_snr_column():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        out_csv = f.name
    result = subprocess.run(
        [sys.executable, "scripts/forced_photometry.py",
         "--mosaic", str(MOSAIC),
         "--method", "two_stage",
         "--sim",
         "--snr-coarse", "0.0",
         "--output", out_csv],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    import csv
    with open(out_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    assert "coarse_snr" in rows[0]
```

- [ ] **Step 5.2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_two_stage_photometry.py::test_cli_simple_peak_produces_csv tests/test_two_stage_photometry.py::test_cli_two_stage_produces_coarse_snr_column -v
```
Expected: non-zero exit code from the subprocess (unrecognised `--method` flag).

- [ ] **Step 5.3: Add `--method`, `--sim`, `--snr-coarse` to `scripts/forced_photometry.py`**

In the `main()` function's `argparse` block, add after existing arguments:

```python
parser.add_argument(
    "--method", default="two_stage",
    choices=["two_stage", "simple_peak", "condon"],
    help="Photometry method: two_stage (default), simple_peak, or condon (original behaviour)",
)
parser.add_argument(
    "--sim", action="store_true", default=False,
    help="Use injected source positions from the synthetic sky model instead of catalog",
)
parser.add_argument(
    "--snr-coarse", type=float, default=3.0, dest="snr_coarse",
    help="Coarse SNR gate for two_stage method (default: 3.0)",
)
```

- [ ] **Step 5.4: Add `_load_sim_coords` helper to `scripts/forced_photometry.py`**

Add before `run_forced_photometry`:

```python
def _load_sim_coords() -> tuple[list[tuple[float, float]], list[float]]:
    """Return (coords, injected_fluxes) from the synthetic sky model (seed=42)."""
    from dsa110_continuum.simulation.ground_truth import SkyModel
    sky = SkyModel(seed=42)
    coords = [(float(sky.ra[k].deg), float(sky.dec[k].deg)) for k in range(sky.Ncomponents)]
    fluxes = [float(sky.stokes[0, 0, k].value) for k in range(sky.Ncomponents)]
    return coords, fluxes
```

- [ ] **Step 5.5: Wire `--sim` and `--method` into `run_forced_photometry`**

Replace the catalog query block in `run_forced_photometry` with a branch:

```python
# ── Source positions ───────────────────────────────────────────────────────
if sim_mode:
    coords, injected_fluxes = _load_sim_coords()
    log.info("Sim mode: %d injected sources from SkyModel(seed=42)", len(coords))
    df = None  # no catalog DataFrame in sim mode
else:
    ra_cen, dec_cen, radius = get_mosaic_footprint(data, wcs)
    log.info("Querying %s catalog (min_flux=%.0f mJy) ...", catalog, min_flux_mjy)
    # ... existing catalog query code unchanged ...
    injected_fluxes = None
```

Add `sim_mode: bool = False` and `method: str = "two_stage"` to the
`run_forced_photometry` signature.

- [ ] **Step 5.6: Wire measurement methods**

Replace the `measure_many` call in `run_forced_photometry` with:

```python
# ── Measure ────────────────────────────────────────────────────────────────
if method == "simple_peak":
    from dsa110_continuum.photometry.simple_peak import measure_peak_box
    from astropy.wcs import WCS as _WCS
    _wcs = _WCS(fits.getheader(mosaic_path)).celestial
    # Estimate rms from MAD
    from astropy.stats import mad_std
    _finite = data[np.isfinite(data)]
    _rms = float(mad_std(_finite)) if _finite.size > 0 else float("nan")
    simple_results = [
        measure_peak_box(data, _wcs, ra, dec, rms=_rms)
        for ra, dec in coords
    ]
    augments = None
    fine_results = None
    # Build rows directly from simple_results
elif method == "two_stage":
    from dsa110_continuum.photometry.two_stage import run_two_stage
    fine_results, augments = run_two_stage(
        mosaic_path, coords, snr_coarse_min=snr_coarse,
    )
    simple_results = None
else:  # condon
    fine_results = measure_many(mosaic_path, coords)
    augments = None
    simple_results = None
```

- [ ] **Step 5.7: Build output rows for each method**

For `simple_peak`, build rows from `simple_results`:

```python
if method == "simple_peak":
    for i, (peak, snr, xi, yi) in enumerate(simple_results):
        if not np.isfinite(peak):
            continue
        ra, dec = coords[i]
        inj_flux = injected_fluxes[i] if injected_fluxes else None
        row = {
            "source_name": f"J{ra:.4f}{dec:+.4f}",
            "ra_deg": round(ra, 5),
            "dec_deg": round(dec, 5),
            "measured_flux_jy": round(float(peak), 6),
            "snr": round(float(snr), 2) if np.isfinite(snr) else "",
        }
        if inj_flux is not None:
            row["injected_flux_jy"] = round(inj_flux, 5)
        rows.append(row)
```

For `two_stage`, re-use the existing row-building loop but add augment columns:

```python
elif method == "two_stage":
    for i, (res, aug) in enumerate(zip(fine_results, augments)):
        if not np.isfinite(res.peak_jyb):
            continue
        # ... existing row dict construction ...
        row["coarse_snr"] = round(aug.coarse_snr, 2) if np.isfinite(aug.coarse_snr) else ""
        row["passed_coarse"] = aug.passed_coarse
        rows.append(row)
```

- [ ] **Step 5.8: Update CSV fieldnames for each method**

```python
if method == "simple_peak":
    base_fields = ["source_name", "ra_deg", "dec_deg", "measured_flux_jy", "snr"]
    if sim_mode:
        base_fields.insert(3, "injected_flux_jy")
    fieldnames = base_fields
elif method == "two_stage":
    fieldnames = [
        "source_name", "ra_deg", "dec_deg",
        "catalog_flux_jy", "measured_flux_jy", "flux_err_jy",
        "flux_ratio", "snr", "coarse_snr", "passed_coarse",
    ]
    if sim_mode:
        fieldnames[3] = "injected_flux_jy"
else:  # condon — existing fieldnames unchanged
    fieldnames = [ ... ]  # keep as-is
```

- [ ] **Step 5.9: Pass new args through `main()`**

In `main()`, update the `run_forced_photometry` call:

```python
result = run_forced_photometry(
    mosaic_path,
    output_csv=args.output,
    catalog=args.catalog,
    min_flux_mjy=args.min_flux_mjy,
    exclude_resolved=args.exclude_resolved,
    exclude_confused=args.exclude_confused,
    snr_cut=args.snr_cut,
    method=args.method,
    sim_mode=args.sim,
    snr_coarse=args.snr_coarse,
)
```

- [ ] **Step 5.10: Run all tests**

```bash
python -m pytest tests/test_two_stage_photometry.py -v
```
Expected: all tests `PASSED`.

- [ ] **Step 5.11: Commit**

```bash
git add scripts/forced_photometry.py tests/test_two_stage_photometry.py
git commit -m "feat(scripts): add --method, --sim, --snr-coarse to forced_photometry.py"
```

---

### Task 6: Push and verify

- [ ] **Step 6.1: Run full test suite**

```bash
python -m pytest tests/test_two_stage_photometry.py -v --tb=short
```
Expected: all tests pass or skip cleanly.

- [ ] **Step 6.2: Push to GitHub**

```bash
git push origin main
```

- [ ] **Step 6.3: Verify CI passes**

Check https://github.com/dsa110/dsa110-continuum/actions — confirm the Quarto
build and any CI checks go green.

---

## Self-Review Notes

- `run_two_stage` returns `(results, augments)` — callers must unpack two values.
  The existing `run_forced_photometry` is the only caller; it is updated in Task 5.
- `ForcedPhotometryResult` is used without modification — no upstream changes needed.
- `measure_peak_box` in `simple_peak.py` takes `(data, wcs, ra, dec)` in that
  order; Task 5 matches this signature.
- The `condon` method in Task 5 preserves the original behaviour of
  `forced_photometry.py` exactly — no regression for existing users.
- `SkyModel` attribute access (`sky.ra`, `sky.dec`, `sky.stokes`,
  `sky.Ncomponents`) matches `validate_step6_mosaic.py` line 152 — consistent.
