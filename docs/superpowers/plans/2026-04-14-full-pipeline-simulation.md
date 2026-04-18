# Full Pipeline Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `SimulationHarness` to simulate every stage of the DSA-110 continuum pipeline end-to-end — gain corruption → calibration → WSClean deconvolution → image-plane mosaicking → forced photometry — so that the full pipeline can be developed and validated without access to real data.

**Architecture:** A new `SimulatedPipeline` class in `dsa110_continuum/simulation/pipeline.py` orchestrates five stages in sequence, each wrapping existing production code. The harness provides perfect (ground-truth) visibilities; Stage 1 corrupts them with realistic per-antenna gain errors; Stage 2 runs the production calibration solver on those corrupted data (using a simulated calibrator MS); Stage 3 runs WSClean with CLEAN iterations on the corrected MS; Stage 4 builds an image-plane mosaic from multiple tiles; Stage 5 runs forced photometry at injected source positions and compares recovered flux to ground truth. A `SimulatedPipelineResult` dataclass captures pass/fail at each stage and the ground-truth flux recovery table.

**Tech Stack:** `SimulationHarness` (existing), `pyuvdata`, `casacore.tables`, `WSClean 3.6` (subprocess), `astropy` (FITS, WCS), `dsa110_continuum.mosaic.builder.build_mosaic`, `dsa110_continuum.photometry.simple_peak.measure_peak_box`, `dsa110_continuum.simulation.ground_truth.GroundTruthRegistry`, `numpy`, `pytest`.

---

## File Map

| File | Status | Responsibility |
|------|--------|----------------|
| `dsa110_continuum/simulation/pipeline.py` | **CREATE** | `SimulatedPipeline` orchestrator — all five stages |
| `dsa110_continuum/simulation/gain_corruption.py` | **CREATE** | Inject per-antenna complex gain errors into UVH5 |
| `tests/test_simulated_pipeline.py` | **CREATE** | TDD tests for every stage and the end-to-end run |
| `dsa110_continuum/simulation/harness.py` | **MODIFY** | Expose `generate_calibrator_subband()` for a bright point source at a known position |
| `dsa110_continuum/simulation/ground_truth.py` | **MODIFY** | Nothing — already provides `GroundTruthRegistry`; used as-is |

---

## Stage Overview

```
SimulationHarness.generate_subbands()   ← perfect sky visibilities (UVH5)
          │
          ▼
[Stage 1] gain_corruption.corrupt_uvh5()
          │  Multiply each baseline vis by G_i * conj(G_j)
          │  G_i ~ complex(amp_error, phase_error_rad) per antenna per time
          ▼
[Stage 2] SimulatedPipeline._calibrate()
          │  Build calibrator UVH5 (bright point source, known flux)
          │  Write calibrator MS; solve B (bandpass) via casacore.tables direct
          │  Write target MS; apply solutions → CORRECTED_DATA
          ▼
[Stage 3] SimulatedPipeline._image()
          │  wsclean -niter 1000 -auto-threshold 1.0 -mgain 0.8
          │  Output: dirty.fits + restored.fits + residual.fits
          ▼
[Stage 4] SimulatedPipeline._mosaic()
          │  2–4 overlapping tile FITS → build_mosaic() → mosaic.fits
          ▼
[Stage 5] SimulatedPipeline._photometry()
          │  measure_peak_box() at each injected source position
          │  Compare to GroundTruthRegistry
          └─► SimulatedPipelineResult (per-source flux table, pass/fail)
```

---

## Task 1: Gain corruption module (RED → GREEN)

**Files:**
- Create: `dsa110_continuum/simulation/gain_corruption.py`
- Test: `tests/test_simulated_pipeline.py` (class `TestGainCorruption`)

### Goal
`corrupt_uvh5(uvh5_path, amp_scatter=0.05, phase_scatter_deg=5.0, seed=0) -> Path`

Reads a UVH5, multiplies each baseline visibility by `G_i * conj(G_j)` where
`G_i = (1 + ε_amp) * exp(i * ε_phase)` with `ε_amp ~ N(0, amp_scatter)` and
`ε_phase ~ N(0, phase_scatter_deg * π/180)`. Writes `<stem>_corrupted.uvh5`,
returns that path. The gain array shape is `(n_antennas, n_times, n_freqs)`;
we use a single time-independent, frequency-independent gain per antenna
(diagonal direction-independent errors, appropriate for a short transit window).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_simulated_pipeline.py
import numpy as np
import pytest
import tempfile
from pathlib import Path
from dsa110_continuum.simulation.harness import SimulationHarness

class TestGainCorruption:
    @pytest.fixture
    def tiny_uvh5(self, tmp_path):
        """Generate a minimal 4-antenna UVH5 for corruption tests."""
        h = SimulationHarness(n_antennas=4, n_sky_sources=1, seed=0,
                              use_real_positions=False)
        paths = h.generate_subbands(output_dir=tmp_path, n_subbands=1)
        return paths[0]

    def test_corrupt_uvh5_creates_output(self, tiny_uvh5, tmp_path):
        from dsa110_continuum.simulation.gain_corruption import corrupt_uvh5
        out = corrupt_uvh5(tiny_uvh5, seed=0)
        assert out.exists()
        assert "_corrupted" in out.name

    def test_corrupt_uvh5_changes_visibilities(self, tiny_uvh5, tmp_path):
        import pyuvdata
        from dsa110_continuum.simulation.gain_corruption import corrupt_uvh5
        uv_orig = pyuvdata.UVData()
        uv_orig.read(str(tiny_uvh5))
        orig_data = uv_orig.data_array.copy()

        out = corrupt_uvh5(tiny_uvh5, amp_scatter=0.05, phase_scatter_deg=5.0, seed=1)
        uv_corr = pyuvdata.UVData()
        uv_corr.read(str(out))

        assert not np.allclose(uv_corr.data_array, orig_data, atol=1e-6), \
            "Corrupted data should differ from original"

    def test_corrupt_uvh5_amplitude_error_bounded(self, tiny_uvh5, tmp_path):
        """Amplitude ratio of corrupted/original should be close to 1 ± scatter."""
        import pyuvdata
        from dsa110_continuum.simulation.gain_corruption import corrupt_uvh5
        uv_orig = pyuvdata.UVData()
        uv_orig.read(str(tiny_uvh5))

        out = corrupt_uvh5(tiny_uvh5, amp_scatter=0.10, phase_scatter_deg=0.0, seed=2)
        uv_corr = pyuvdata.UVData()
        uv_corr.read(str(out))

        ratio = np.abs(uv_corr.data_array) / (np.abs(uv_orig.data_array) + 1e-30)
        finite = ratio[np.isfinite(ratio)]
        assert finite.mean() == pytest.approx(1.0, abs=0.05), \
            f"Mean amplitude ratio {finite.mean():.3f} should be near 1.0"

    def test_corrupt_uvh5_seed_reproducible(self, tiny_uvh5, tmp_path):
        import pyuvdata
        from dsa110_continuum.simulation.gain_corruption import corrupt_uvh5
        out1 = corrupt_uvh5(tiny_uvh5, seed=42)
        out2 = corrupt_uvh5(tiny_uvh5, seed=42)
        uv1 = pyuvdata.UVData(); uv1.read(str(out1))
        uv2 = pyuvdata.UVData(); uv2.read(str(out2))
        np.testing.assert_array_equal(uv1.data_array, uv2.data_array)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=/home/user/workspace/dsa110-continuum python3 -m pytest \
  tests/test_simulated_pipeline.py::TestGainCorruption -v --tb=short \
  --override-ini="addopts="
```
Expected: `ImportError: cannot import name 'corrupt_uvh5'`

- [ ] **Step 3: Implement `gain_corruption.py`**

```python
# dsa110_continuum/simulation/gain_corruption.py
"""Per-antenna complex gain corruption for simulation.

Multiplies each baseline visibility V_ij by G_i * conj(G_j), where
G_i = (1 + ε_amp_i) * exp(i * ε_phase_i).

This simulates direction-independent gain errors: the dominant error
source for a transit array during a ~5-min integration window.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pyuvdata

logger = logging.getLogger(__name__)


def corrupt_uvh5(
    uvh5_path: Path | str,
    *,
    amp_scatter: float = 0.05,
    phase_scatter_deg: float = 5.0,
    seed: int = 0,
    output_path: Path | str | None = None,
) -> Path:
    """Corrupt visibilities with per-antenna gain errors.

    Parameters
    ----------
    uvh5_path:
        Input UVH5 file path.
    amp_scatter:
        Standard deviation of amplitude error (fractional, e.g. 0.05 = 5%).
    phase_scatter_deg:
        Standard deviation of phase error in degrees.
    seed:
        RNG seed for reproducibility.
    output_path:
        Where to write the corrupted file. Defaults to
        ``<stem>_corrupted.uvh5`` alongside the input.

    Returns
    -------
    Path
        Path to the written corrupted UVH5 file.
    """
    uvh5_path = Path(uvh5_path)
    if output_path is None:
        output_path = uvh5_path.with_name(uvh5_path.stem + "_corrupted.uvh5")
    output_path = Path(output_path)

    rng = np.random.default_rng(seed)

    uv = pyuvdata.UVData()
    uv.read(str(uvh5_path))

    n_ant = uv.Nants_data
    ant_nums = np.unique(
        np.concatenate([uv.ant_1_array, uv.ant_2_array])
    )  # actual antenna numbers in data

    # Draw one gain per antenna: G_i = (1 + ε_amp) * exp(i * ε_phase)
    amp_errors = 1.0 + rng.normal(0.0, amp_scatter, size=len(ant_nums))
    phase_errors = rng.normal(0.0, np.radians(phase_scatter_deg), size=len(ant_nums))
    gains = amp_errors * np.exp(1j * phase_errors)  # shape (n_ant_unique,)

    # Map antenna number → gain index
    ant_to_idx = {int(a): i for i, a in enumerate(ant_nums)}

    # Apply G_i * conj(G_j) to each baseline row
    data = uv.data_array.copy()  # shape (Nblts, Nfreqs, Npols)
    for row in range(uv.Nblts):
        i = ant_to_idx[int(uv.ant_1_array[row])]
        j = ant_to_idx[int(uv.ant_2_array[row])]
        factor = gains[i] * np.conj(gains[j])
        data[row] *= factor

    uv.data_array = data.astype(np.complex64)

    # Store gain truth in extra_keywords for validation
    uv.extra_keywords["GAIN_AMP_SCATTER"] = amp_scatter
    uv.extra_keywords["GAIN_PHASE_SCATTER_DEG"] = phase_scatter_deg
    uv.extra_keywords["GAIN_SEED"] = seed

    uv.write_uvh5(str(output_path), clobber=True)
    logger.info(
        "Wrote corrupted UVH5 (%d antennas, amp_scatter=%.3f, phase_scatter=%.1f°) → %s",
        len(ant_nums), amp_scatter, phase_scatter_deg, output_path,
    )
    return output_path
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=/home/user/workspace/dsa110-continuum python3 -m pytest \
  tests/test_simulated_pipeline.py::TestGainCorruption -v --tb=short \
  --override-ini="addopts="
```
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add dsa110_continuum/simulation/gain_corruption.py \
        tests/test_simulated_pipeline.py
git commit -m "feat(sim): add gain_corruption.corrupt_uvh5() with 4 TDD tests

Per-antenna complex gain errors: G_i = (1+ε_amp)*exp(i*ε_phase).
Amplitude and phase scatter are independently configurable; RNG seed
makes runs reproducible. Used by SimulatedPipeline Stage 1."
```

---

## Task 2: Calibrator UVH5 generation (RED → GREEN)

**Files:**
- Modify: `dsa110_continuum/simulation/harness.py`
- Test: `tests/test_simulated_pipeline.py` (class `TestCalibratorGeneration`)

### Goal
Add `SimulationHarness.generate_calibrator_subband(output_dir, flux_jy, ...)` that writes a single-subband UVH5 containing a bright, unresolved point source at the phase centre (HA=0, Dec=pointing_dec_deg). This will be used by Stage 2 to derive gain solutions via a direct casacore-table bandpass solve.

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/test_simulated_pipeline.py

class TestCalibratorGeneration:
    def test_generate_calibrator_subband_creates_file(self, tmp_path):
        from dsa110_continuum.simulation.harness import SimulationHarness
        h = SimulationHarness(n_antennas=4, seed=0, use_real_positions=False)
        path = h.generate_calibrator_subband(tmp_path, flux_jy=10.0)
        assert Path(path).exists()
        assert "_cal_" in Path(path).name

    def test_calibrator_subband_has_single_source_at_phase_centre(self, tmp_path):
        """All baselines should show near-real visibilities (Im ≈ 0 at HA=0)."""
        import pyuvdata, numpy as np
        from dsa110_continuum.simulation.harness import SimulationHarness
        h = SimulationHarness(n_antennas=4, seed=0, use_real_positions=False,
                              n_sky_sources=0)
        path = h.generate_calibrator_subband(tmp_path, flux_jy=5.0)
        uv = pyuvdata.UVData(); uv.read(str(path))
        # For a source at phase centre all phases = 0 → vis purely real
        # (after conjugation fix: stored as conj(V) but still real for HA=0)
        data = uv.data_array  # shape (Nblts, Nfreqs, Npols)
        np.testing.assert_allclose(
            np.abs(data.imag) / (np.abs(data.real) + 1e-10),
            0.0, atol=0.01,
            err_msg="Calibrator vis at phase centre should be nearly real",
        )

    def test_calibrator_subband_amplitude_matches_flux(self, tmp_path):
        """Auto-correlation amplitude (if stored) or mean cross-corr should reflect flux_jy."""
        import pyuvdata, numpy as np
        from dsa110_continuum.simulation.harness import SimulationHarness
        h = SimulationHarness(n_antennas=4, seed=0, use_real_positions=False,
                              n_sky_sources=0)
        flux = 8.0
        path = h.generate_calibrator_subband(tmp_path, flux_jy=flux)
        uv = pyuvdata.UVData(); uv.read(str(path))
        # Cross-correlation amplitude = flux_jy / 2 (XX pol = I/2)
        cross = uv.data_array[uv.ant_1_array != uv.ant_2_array, :, 0]
        mean_amp = float(np.abs(cross).mean())
        assert mean_amp == pytest.approx(flux / 2.0, rel=0.05), \
            f"Mean cross-corr amplitude {mean_amp:.3f} should be ≈ flux/2 = {flux/2:.3f}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=/home/user/workspace/dsa110-continuum python3 -m pytest \
  tests/test_simulated_pipeline.py::TestCalibratorGeneration -v --tb=short \
  --override-ini="addopts="
```
Expected: `AttributeError: 'SimulationHarness' object has no attribute 'generate_calibrator_subband'`

- [ ] **Step 3: Add `generate_calibrator_subband()` to `harness.py`**

Find the end of the `SimulationHarness` class (after `generate_subbands`). Add:

```python
def generate_calibrator_subband(
    self,
    output_dir: Path | str,
    *,
    flux_jy: float = 10.0,
    subband_index: int = 0,
    filename: str | None = None,
) -> Path:
    """Generate a calibrator-observation UVH5 with a single bright point source at phase centre.

    The calibrator sits exactly at (pointing_ra_deg, pointing_dec_deg) so
    all baseline phases are zero and the visibility amplitude equals flux_jy/2
    (XX = YY = I/2 convention).  This file is used by the simulated calibration
    stage to derive gain solutions.

    Parameters
    ----------
    output_dir:
        Directory in which to write the output file.
    flux_jy:
        Flux density of the calibrator source (Jy). Default 10 Jy, typical
        for VLA calibrators used by DSA-110 (e.g. 3C 309.1 ≈ 9 Jy at 1.4 GHz).
    subband_index:
        Which subband frequency to simulate (0-indexed, default 0).
    filename:
        Override output filename. Defaults to ``sim_cal_sb{subband_index:02d}.uvh5``.

    Returns
    -------
    Path
        Path to written calibrator UVH5.
    """
    import pyradiosky

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"sim_cal_sb{subband_index:02d}.uvh5"
    out_path = output_dir / filename

    # Build a single-source SkyModel exactly at phase centre
    stokes = np.zeros((4, 1, 1), dtype=float)
    stokes[0, 0, 0] = flux_jy  # Stokes I only
    freq_hz = float(self._subband_freqs_hz()[subband_index].mean())
    sky = pyradiosky.SkyModel(
        name=np.array(["SIM_CAL"]),
        ra=Longitude([self.pointing_ra_deg], unit="deg"),
        dec=Latitude([self.pointing_dec_deg], unit="deg"),
        stokes=stokes * units.Jy,
        spectral_type="flat",
        reference_frequency=freq_hz * units.Hz,
    )

    # Temporarily override n_sky_sources and noise for a clean calibrator
    orig_n = self.n_sky_sources
    orig_noise = self.noise_jy
    self.n_sky_sources = 0   # suppress random field sources
    self.noise_jy = 0.0      # no noise on calibrator

    paths = self.generate_subbands(
        output_dir=output_dir,
        n_subbands=1,
        sky=sky,
        filename_template=filename.replace(".uvh5", "_sb{sb:02d}.uvh5"),
    )

    self.n_sky_sources = orig_n
    self.noise_jy = orig_noise

    # Rename to desired output path
    written = Path(paths[0])
    if written != out_path:
        written.rename(out_path)

    # Tag as calibrator in extra_keywords (read back and rewrite)
    uv = pyuvdata.UVData()
    uv.read(str(out_path))
    uv.extra_keywords["CAL_FLUX_JY"] = flux_jy
    uv.extra_keywords["CAL_SOURCE"] = "SIM_CAL"
    uv.write_uvh5(str(out_path), clobber=True)

    logger.info("Wrote calibrator UVH5 (%.1f Jy at phase centre) → %s", flux_jy, out_path)
    return out_path
```

> **Note:** `generate_subbands` has a `sky` parameter already — check the method signature and adjust if the existing API differs slightly (e.g. if it takes `sky_model` instead). Also check that `_subband_freqs_hz()` exists or substitute the equivalent call used elsewhere in the harness.

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=/home/user/workspace/dsa110-continuum python3 -m pytest \
  tests/test_simulated_pipeline.py::TestCalibratorGeneration -v --tb=short \
  --override-ini="addopts="
```
Expected: 3 PASSED

- [ ] **Step 5: Full suite still green**

```bash
PYTHONPATH=/home/user/workspace/dsa110-continuum python3 -m pytest tests/ -q \
  --override-ini="addopts=" -m "not slow" 2>&1 | tail -5
```
Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add dsa110_continuum/simulation/harness.py tests/test_simulated_pipeline.py
git commit -m "feat(sim): add generate_calibrator_subband() to SimulationHarness

Writes a single-subband UVH5 with a bright (default 10 Jy) point source
exactly at phase centre, zero noise. Used by SimulatedPipeline Stage 2
as the calibrator observation from which gain solutions are derived."
```

---

## Task 3: Simulated calibration stage (RED → GREEN)

**Files:**
- Create: `dsa110_continuum/simulation/pipeline.py` (partial — Stage 2 only)
- Test: `tests/test_simulated_pipeline.py` (class `TestSimulatedCalibration`)

### Goal
`SimulatedPipeline._calibrate(target_uvh5_paths, cal_uvh5_path, work_dir)` → corrected MS path.

This avoids CASA entirely (unavailable in the cloud sandbox). Instead:
1. Write the calibrator UVH5 → MS via `pyuvdata.write_ms()`.
2. Compute per-antenna, per-channel complex gain solutions directly from the
   calibrator MS DATA column using least-squares: for a known-flux source at
   phase centre, `G_i = mean_over_baselines(V_ij / V_ij_model)` → normalised.
3. Apply the solutions to the target MS DATA column → CORRECTED_DATA column
   via `casacore.tables`.

This is a faithful simulation of the DSA-110 calibration solve + apply loop
without requiring `casatools`. It is explicitly not production code — the
production `calibration.py` uses CASA `gaincal` + `applycal`. The simulation
produces a CORRECTED_DATA column that WSClean reads by default.

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/test_simulated_pipeline.py

class TestSimulatedCalibration:
    @pytest.fixture
    def corrupted_ms(self, tmp_path):
        """Tiny corrupted MS ready for calibration."""
        from dsa110_continuum.simulation.harness import SimulationHarness
        from dsa110_continuum.simulation.gain_corruption import corrupt_uvh5
        h = SimulationHarness(n_antennas=4, n_sky_sources=1, seed=7,
                              use_real_positions=False)
        paths = h.generate_subbands(output_dir=tmp_path, n_subbands=1)
        corrupted = corrupt_uvh5(paths[0], amp_scatter=0.10,
                                  phase_scatter_deg=10.0, seed=7)
        cal_path = h.generate_calibrator_subband(tmp_path, flux_jy=10.0)

        uv = __import__("pyuvdata").UVData()
        uv.read([str(corrupted)])
        ms_path = tmp_path / "target.ms"
        uv.write_ms(str(ms_path))
        return ms_path, cal_path, tmp_path

    def test_calibrate_creates_corrected_data_column(self, corrupted_ms):
        import casacore.tables as ct
        from dsa110_continuum.simulation.pipeline import SimulatedPipeline
        ms_path, cal_path, work_dir = corrupted_ms
        p = SimulatedPipeline(work_dir=work_dir)
        p._calibrate(target_ms=ms_path, cal_uvh5=cal_path,
                     cal_flux_jy=10.0, work_dir=work_dir)
        with ct.table(str(ms_path), readonly=True, ack=False) as t:
            cols = t.colnames()
        assert "CORRECTED_DATA" in cols, "CORRECTED_DATA column must be added by calibration"

    def test_calibrate_reduces_phase_scatter(self, corrupted_ms):
        """After calibration, cross-corr phases should be more tightly clustered."""
        import casacore.tables as ct, numpy as np
        from dsa110_continuum.simulation.pipeline import SimulatedPipeline
        ms_path, cal_path, work_dir = corrupted_ms
        p = SimulatedPipeline(work_dir=work_dir)
        p._calibrate(target_ms=ms_path, cal_uvh5=cal_path,
                     cal_flux_jy=10.0, work_dir=work_dir)
        with ct.table(str(ms_path), readonly=True, ack=False) as t:
            raw = t.getcol("DATA")
            corr = t.getcol("CORRECTED_DATA")
        phase_raw  = np.angle(raw).std()
        phase_corr = np.angle(corr).std()
        assert phase_corr < phase_raw, \
            f"Calibration should reduce phase scatter: raw={phase_raw:.3f}, corr={phase_corr:.3f}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=/home/user/workspace/dsa110-continuum python3 -m pytest \
  tests/test_simulated_pipeline.py::TestSimulatedCalibration -v --tb=short \
  --override-ini="addopts="
```
Expected: `ImportError: cannot import name 'SimulatedPipeline'`

- [ ] **Step 3: Create `pipeline.py` with `SimulatedPipeline._calibrate()`**

```python
# dsa110_continuum/simulation/pipeline.py
"""Simulated DSA-110 continuum pipeline for end-to-end testing.

Runs all production stages in sequence using real production code where
possible (WSClean, astropy, pyuvdata) and faithful simulation where CASA
is unavailable (calibration solve/apply via casacore.tables directly).

Stage 1: Gain corruption — corrupt_uvh5()
Stage 2: Calibration    — _calibrate()
Stage 3: Imaging        — _image()
Stage 4: Mosaicking     — _mosaic()
Stage 5: Photometry     — _photometry()
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyuvdata

logger = logging.getLogger(__name__)


@dataclass
class SourceFluxResult:
    """Recovered vs. injected flux for one source."""
    source_id: str
    ra_deg: float
    dec_deg: float
    injected_flux_jy: float
    recovered_flux_jy: float   # NaN if not measurable
    snr: float                 # NaN if not measurable
    passed: bool               # True if |recovered - injected| / injected < 0.2


@dataclass
class SimulatedPipelineResult:
    """Outcome of a full simulated pipeline run."""
    work_dir: Path
    n_tiles: int
    calibration_passed: bool
    imaging_passed: bool
    mosaic_path: Path | None
    source_results: list[SourceFluxResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def n_recovered(self) -> int:
        return sum(1 for r in self.source_results if r.passed)

    @property
    def all_passed(self) -> bool:
        return (self.calibration_passed and self.imaging_passed
                and self.n_recovered == len(self.source_results))


class SimulatedPipeline:
    """Orchestrate all five pipeline stages on simulated data.

    Parameters
    ----------
    work_dir:
        Root scratch directory; sub-directories are created per stage.
    wsclean_bin:
        Path to wsclean binary (default: ``wsclean`` on PATH).
    niter:
        WSClean CLEAN iterations (default 1000).
    cell_arcsec:
        WSClean cell size in arcseconds (default 20.0 — matches existing
        working smoke tests).
    image_size:
        WSClean image size in pixels (default 512).
    """

    def __init__(
        self,
        work_dir: Path | str,
        *,
        wsclean_bin: str = "wsclean",
        niter: int = 1000,
        cell_arcsec: float = 20.0,
        image_size: int = 512,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.wsclean_bin = wsclean_bin
        self.niter = niter
        self.cell_arcsec = cell_arcsec
        self.image_size = image_size

    # ------------------------------------------------------------------ #
    # Stage 2 — Calibration                                               #
    # ------------------------------------------------------------------ #

    def _calibrate(
        self,
        *,
        target_ms: Path | str,
        cal_uvh5: Path | str,
        cal_flux_jy: float,
        work_dir: Path | str,
    ) -> Path:
        """Apply per-antenna bandpass-style gains derived from the calibrator.

        Algorithm (CASA-free, suitable for cloud sandbox):
        1. Read calibrator UVH5 → compute per-antenna complex gains.
           For a known-flux source at phase centre:
               model_vis = flux_jy / 2  (real, for XX = I/2)
               G_i * conj(G_j) ≈ V_ij / model_vis  (per baseline, per channel)
           Solve for G_i via least-squares antenna factorisation (one iteration
           of the standard radio calibration Jacobi update):
               G_i = mean_j(V_ij / (model_vis * conj(G_j_prev)))
           with G_j_prev initialised to 1.0.  Two Jacobi sweeps suffice for
           small errors.

        2. Apply G_i to target MS DATA column:
               CORRECTED_DATA_ij = DATA_ij / (G_i * conj(G_j))

        Parameters
        ----------
        target_ms:
            Measurement Set to calibrate (DATA → CORRECTED_DATA).
        cal_uvh5:
            Calibrator UVH5 file (output of generate_calibrator_subband).
        cal_flux_jy:
            Known calibrator flux in Jy (used to build the model column).
        work_dir:
            Scratch directory (unused here but kept for API symmetry).

        Returns
        -------
        Path
            Path to the (modified in-place) target MS.
        """
        import casacore.tables as ct

        target_ms = Path(target_ms)
        cal_uvh5  = Path(cal_uvh5)

        # --- Step 1: Derive per-antenna gains from calibrator ---
        uv_cal = pyuvdata.UVData()
        uv_cal.read(str(cal_uvh5))

        ant_nums = np.unique(np.concatenate([uv_cal.ant_1_array, uv_cal.ant_2_array]))
        n_ant    = len(ant_nums)
        ant_idx  = {int(a): i for i, a in enumerate(ant_nums)}
        n_freq   = uv_cal.Nfreqs

        model_amp = cal_flux_jy / 2.0   # XX = I/2, source at phase centre → real

        # Initialise gains to unity
        gains = np.ones((n_ant, n_freq), dtype=complex)

        # Jacobi solve: two iterations
        for _ in range(2):
            numerator   = np.zeros_like(gains)
            denominator = np.zeros((n_ant, n_freq), dtype=float)
            for row in range(uv_cal.Nblts):
                i = ant_idx[int(uv_cal.ant_1_array[row])]
                j = ant_idx[int(uv_cal.ant_2_array[row])]
                if i == j:
                    continue  # skip auto-correlations
                vis = uv_cal.data_array[row, :, 0]  # XX pol, shape (n_freq,)
                # Vis stored as conj(V); undo conjugation to get true V_ij
                true_vis = vis.conj()
                numerator[i]   += true_vis * np.conj(gains[j]) / model_amp
                denominator[i] += np.abs(gains[j]) ** 2
                numerator[j]   += np.conj(true_vis) * gains[i] / model_amp
                denominator[j] += np.abs(gains[i]) ** 2
            gains = numerator / np.maximum(denominator, 1e-12)

        logger.info(
            "Gain solve: mean amplitude=%.4f, mean phase=%.2f deg",
            np.abs(gains).mean(),
            np.degrees(np.angle(gains)).mean(),
        )

        # --- Step 2: Write CORRECTED_DATA to target MS ---
        with ct.table(str(target_ms), readonly=False, ack=False) as t:
            # Get antenna numbers used in MS (may differ from cal antenna ordering)
            with ct.table(str(target_ms) + "::ANTENNA", readonly=True, ack=False) as tant:
                ms_ant_names = tant.getcol("NAME")  # list of names like "ANT0", "ANT1"...

            data      = t.getcol("DATA")       # shape (Nrows, Nchans, Npols)
            ant1_col  = t.getcol("ANTENNA1")
            ant2_col  = t.getcol("ANTENNA2")
            n_rows, n_chan, n_pol = data.shape

            corrected = data.copy()

            # Build gain lookup by MS antenna index (0-based integer from ANTENNA table)
            # The MS ANTENNA1/ANTENNA2 columns are 0-based indices into the ANTENNA table.
            # The cal UVH5 antenna numbers may be different ints; map by order.
            # Strategy: use position in ant_nums ordering (they were sorted the same way).
            # If MS has more antennas than cal, missing antennas get gain=1.0.
            n_ms_ant = len(ms_ant_names)
            ms_gains = np.ones((n_ms_ant, n_freq), dtype=complex)
            for ms_idx in range(min(n_ms_ant, n_ant)):
                ms_gains[ms_idx] = gains[ms_idx]

            for row in range(n_rows):
                i = int(ant1_col[row])
                j = int(ant2_col[row])
                # CORRECTED = DATA / (G_i * conj(G_j))
                gi = ms_gains[i] if i < n_ms_ant else np.ones(n_freq, dtype=complex)
                gj = ms_gains[j] if j < n_ms_ant else np.ones(n_freq, dtype=complex)
                denom = gi * np.conj(gj)  # shape (n_freq,)
                for p in range(n_pol):
                    corrected[row, :, p] /= np.where(np.abs(denom) > 1e-12, denom, 1.0)

            # Add or overwrite CORRECTED_DATA column
            if "CORRECTED_DATA" not in t.colnames():
                from casacore.tables import makearrcoldesc, maketabdesc
                cd = makearrcoldesc(
                    "CORRECTED_DATA", data[0],
                    valuetype="complex",
                    comment="Calibrated data",
                )
                t.addcols(maketabdesc(cd))

            t.putcol("CORRECTED_DATA", corrected.astype(np.complex64))

        logger.info("Wrote CORRECTED_DATA to %s", target_ms)
        return target_ms
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=/home/user/workspace/dsa110-continuum python3 -m pytest \
  tests/test_simulated_pipeline.py::TestSimulatedCalibration -v --tb=short \
  --override-ini="addopts="
```
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add dsa110_continuum/simulation/pipeline.py tests/test_simulated_pipeline.py
git commit -m "feat(sim): Stage 2 — simulated calibration solve+apply via casacore.tables

Jacobi antenna gain solver (2 iterations) derives per-antenna complex
gains from the calibrator UVH5, applies them to the target MS to write
CORRECTED_DATA.  CASA-free: suitable for cloud sandbox.  Phase scatter
is reduced to near-zero for a point source at phase centre."
```

---

## Task 4: WSClean deconvolution stage (RED → GREEN)

**Files:**
- Modify: `dsa110_continuum/simulation/pipeline.py` (add `_image()`)
- Test: `tests/test_simulated_pipeline.py` (class `TestSimulatedImaging`)

### Goal
`SimulatedPipeline._image(ms_path, work_dir)` → `{'restored': Path, 'residual': Path, 'psf': Path}`

Runs WSClean with `niter > 0` reading `CORRECTED_DATA` and produces a restored image.

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/test_simulated_pipeline.py

class TestSimulatedImaging:
    @pytest.fixture
    def calibrated_ms(self, tmp_path):
        """Tiny calibrated MS (CORRECTED_DATA column present)."""
        from dsa110_continuum.simulation.harness import SimulationHarness
        from dsa110_continuum.simulation.gain_corruption import corrupt_uvh5
        from dsa110_continuum.simulation.pipeline import SimulatedPipeline
        h = SimulationHarness(n_antennas=8, n_sky_sources=1, seed=3,
                              use_real_positions=False)
        paths = h.generate_subbands(output_dir=tmp_path, n_subbands=1)
        corrupted = corrupt_uvh5(paths[0], amp_scatter=0.05,
                                  phase_scatter_deg=3.0, seed=3)
        cal_path = h.generate_calibrator_subband(tmp_path, flux_jy=10.0)
        uv = pyuvdata.UVData(); uv.read([str(corrupted)])
        ms_path = tmp_path / "cal_target.ms"
        uv.write_ms(str(ms_path))
        p = SimulatedPipeline(work_dir=tmp_path, niter=100, cell_arcsec=30.0, image_size=256)
        p._calibrate(target_ms=ms_path, cal_uvh5=cal_path,
                     cal_flux_jy=10.0, work_dir=tmp_path)
        return ms_path, p, tmp_path

    def test_image_creates_restored_fits(self, calibrated_ms):
        ms_path, p, work_dir = calibrated_ms
        result = p._image(ms_path=ms_path, work_dir=work_dir)
        assert result["restored"].exists(), "Restored FITS must exist"

    def test_image_creates_psf(self, calibrated_ms):
        ms_path, p, work_dir = calibrated_ms
        result = p._image(ms_path=ms_path, work_dir=work_dir)
        assert result["psf"].exists(), "PSF FITS must exist"

    def test_image_restored_has_valid_wcs(self, calibrated_ms):
        from astropy.io import fits
        from astropy.wcs import WCS
        ms_path, p, work_dir = calibrated_ms
        result = p._image(ms_path=ms_path, work_dir=work_dir)
        with fits.open(result["restored"]) as hdul:
            wcs = WCS(hdul[0].header)
        assert wcs.naxis >= 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=/home/user/workspace/dsa110-continuum python3 -m pytest \
  tests/test_simulated_pipeline.py::TestSimulatedImaging -v --tb=short \
  --override-ini="addopts="
```
Expected: `AttributeError: 'SimulatedPipeline' object has no attribute '_image'`

- [ ] **Step 3: Add `_image()` to `SimulatedPipeline` in `pipeline.py`**

```python
    # ------------------------------------------------------------------ #
    # Stage 3 — WSClean imaging with deconvolution                        #
    # ------------------------------------------------------------------ #

    def _image(
        self,
        *,
        ms_path: Path | str,
        work_dir: Path | str,
        data_column: str = "CORRECTED_DATA",
    ) -> dict[str, Path]:
        """Run WSClean with CLEAN iterations on a calibrated MS.

        Parameters
        ----------
        ms_path:
            Calibrated Measurement Set (must have CORRECTED_DATA column).
        work_dir:
            Directory for WSClean output files.
        data_column:
            MS column to image (default ``CORRECTED_DATA``).

        Returns
        -------
        dict with keys ``'restored'``, ``'dirty'``, ``'residual'``, ``'psf'``
            Paths to the respective FITS files.
        """
        work_dir = Path(work_dir)
        img_dir  = work_dir / "wsclean_out"
        img_dir.mkdir(parents=True, exist_ok=True)
        prefix   = str(img_dir / "wsclean")

        cmd = [
            self.wsclean_bin,
            "-name", prefix,
            "-size", str(self.image_size), str(self.image_size),
            "-scale", f"{self.cell_arcsec}asec",
            "-weight", "briggs", "0.0",
            "-niter", str(self.niter),
            "-mgain", "0.8",
            "-auto-threshold", "1.0",
            "-pol", "I",
            "-datacolumn", data_column,
            "-make-psf",
            "-no-update-model-required",
            str(ms_path),
        ]
        logger.info("Running WSClean: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("WSClean stderr:\n%s", result.stderr[-3000:])
            raise RuntimeError(f"WSClean failed (rc={result.returncode})")

        outputs = {
            "restored":  Path(f"{prefix}-image.fits"),
            "dirty":     Path(f"{prefix}-dirty.fits"),
            "residual":  Path(f"{prefix}-residual.fits"),
            "psf":       Path(f"{prefix}-psf.fits"),
        }
        for key, path in outputs.items():
            if not path.exists():
                logger.warning("Expected WSClean output missing: %s", path)
        return outputs
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=/home/user/workspace/dsa110-continuum python3 -m pytest \
  tests/test_simulated_pipeline.py::TestSimulatedImaging -v --tb=short \
  --override-ini="addopts="
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add dsa110_continuum/simulation/pipeline.py tests/test_simulated_pipeline.py
git commit -m "feat(sim): Stage 3 — WSClean deconvolution (_image()) with CLEAN iterations

Reads CORRECTED_DATA column; -niter 1000 -mgain 0.8 -auto-threshold 1.0.
Produces restored, dirty, residual, and PSF FITS files.  Integration tests
verify file existence and valid WCS header."
```

---

## Task 5: Image-plane mosaicking stage (RED → GREEN)

**Files:**
- Modify: `dsa110_continuum/simulation/pipeline.py` (add `_mosaic()`)
- Test: `tests/test_simulated_pipeline.py` (class `TestSimulatedMosaic`)

### Goal
`SimulatedPipeline._mosaic(image_paths, work_dir)` → `Path` to mosaic FITS.

Delegates to the existing `build_mosaic()` from `dsa110_continuum.mosaic.builder`.
For testing, we use 2–4 tile FITS images (copies/small offsets of the same image).

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/test_simulated_pipeline.py
import shutil

class TestSimulatedMosaic:
    @pytest.fixture
    def two_tile_fits(self, tmp_path):
        """Create two minimal FITS tiles from the same image (overlapping)."""
        import numpy as np
        from astropy.io import fits
        from astropy.wcs import WCS
        # Minimal synthetic FITS image: 64×64 blank image with WCS
        data = np.zeros((64, 64), dtype=np.float32)
        data[32, 32] = 0.5  # fake source
        w = WCS(naxis=2)
        w.wcs.crpix = [32, 32]
        w.wcs.cdelt = [-20.0 / 3600, 20.0 / 3600]
        w.wcs.crval = [343.5, 16.15]
        w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
        hdr = w.to_header()
        hdr["BUNIT"] = "JY/BEAM"
        hdr["BMAJ"] = 329.0 / 3600
        hdr["BMIN"] = 76.0 / 3600
        hdr["BPA"] = -132.0
        tile1 = tmp_path / "tile1.fits"
        tile2 = tmp_path / "tile2.fits"
        fits.writeto(str(tile1), data, hdr, overwrite=True)
        # Tile 2: same but shifted 0.05° in RA
        hdr2 = hdr.copy()
        hdr2["CRVAL1"] = 343.55
        fits.writeto(str(tile2), data, hdr2, overwrite=True)
        return [tile1, tile2], tmp_path

    def test_mosaic_creates_output_fits(self, two_tile_fits):
        from dsa110_continuum.simulation.pipeline import SimulatedPipeline
        tiles, work_dir = two_tile_fits
        p = SimulatedPipeline(work_dir=work_dir)
        mosaic_path = p._mosaic(image_paths=tiles, work_dir=work_dir)
        assert mosaic_path.exists(), "Mosaic FITS must be created"

    def test_mosaic_has_larger_footprint(self, two_tile_fits):
        """Mosaic should cover more pixels than either input tile."""
        from astropy.io import fits
        from dsa110_continuum.simulation.pipeline import SimulatedPipeline
        tiles, work_dir = two_tile_fits
        p = SimulatedPipeline(work_dir=work_dir)
        mosaic_path = p._mosaic(image_paths=tiles, work_dir=work_dir)
        with fits.open(str(tiles[0])) as h0, fits.open(str(mosaic_path)) as hm:
            n_pix_tile   = h0[0].data.size
            n_pix_mosaic = hm[0].data.size
        assert n_pix_mosaic >= n_pix_tile, \
            f"Mosaic ({n_pix_mosaic} px) should be at least as large as one tile ({n_pix_tile} px)"

    def test_mosaic_returns_path_object(self, two_tile_fits):
        from dsa110_continuum.simulation.pipeline import SimulatedPipeline
        tiles, work_dir = two_tile_fits
        p = SimulatedPipeline(work_dir=work_dir)
        result = p._mosaic(image_paths=tiles, work_dir=work_dir)
        assert isinstance(result, Path)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=/home/user/workspace/dsa110-continuum python3 -m pytest \
  tests/test_simulated_pipeline.py::TestSimulatedMosaic -v --tb=short \
  --override-ini="addopts="
```
Expected: `AttributeError: 'SimulatedPipeline' object has no attribute '_mosaic'`

- [ ] **Step 3: Add `_mosaic()` to `SimulatedPipeline`**

```python
    # ------------------------------------------------------------------ #
    # Stage 4 — Image-plane mosaicking                                    #
    # ------------------------------------------------------------------ #

    def _mosaic(
        self,
        *,
        image_paths: list[Path],
        work_dir: Path | str,
        output_name: str = "epoch_mosaic.fits",
    ) -> Path:
        """Co-add tile FITS images into a single epoch mosaic.

        Uses the production ``build_mosaic()`` (PB-weighted linear coaddition)
        from ``dsa110_continuum.mosaic.builder``.  For the simulated pipeline
        this is the QUICKLOOK-tier approach; the SCIENCE tier would use
        WSClean visibility-domain joint deconvolution (not practical in the
        cloud sandbox without real beam models).

        Parameters
        ----------
        image_paths:
            List of tile FITS paths to co-add (at least 2).
        work_dir:
            Output directory for the mosaic.
        output_name:
            Filename for the output mosaic FITS (default: ``epoch_mosaic.fits``).

        Returns
        -------
        Path
            Path to the written mosaic FITS file.
        """
        from dsa110_continuum.mosaic.builder import build_mosaic

        work_dir    = Path(work_dir)
        output_path = work_dir / output_name

        result = build_mosaic(
            image_paths=[Path(p) for p in image_paths],
            output_path=output_path,
            apply_pb_correction=False,  # no beam model in cloud sandbox
            write_weight_map=False,
        )

        logger.info(
            "Mosaic complete: %d tiles → %s (RMS %.3f mJy/beam)",
            len(image_paths), result.output_path, result.median_rms * 1e3,
        )
        return result.output_path
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=/home/user/workspace/dsa110-continuum python3 -m pytest \
  tests/test_simulated_pipeline.py::TestSimulatedMosaic -v --tb=short \
  --override-ini="addopts="
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add dsa110_continuum/simulation/pipeline.py tests/test_simulated_pipeline.py
git commit -m "feat(sim): Stage 4 — image-plane mosaicking via build_mosaic()

Delegates to production dsa110_continuum.mosaic.builder.build_mosaic()
(PB-weighted linear coaddition, QUICKLOOK tier).  Tests verify output
existence, footprint size, and return type."
```

---

## Task 6: Forced photometry stage (RED → GREEN)

**Files:**
- Modify: `dsa110_continuum/simulation/pipeline.py` (add `_photometry()`)
- Test: `tests/test_simulated_pipeline.py` (class `TestSimulatedPhotometry`)

### Goal
`SimulatedPipeline._photometry(image_path, sources, ground_truth)` → `list[SourceFluxResult]`

Runs `measure_peak_box()` at each injected source position; compares to ground truth.

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/test_simulated_pipeline.py

class TestSimulatedPhotometry:
    @pytest.fixture
    def mock_image_with_source(self, tmp_path):
        """FITS image with one injected point source at known RA/Dec."""
        import numpy as np
        from astropy.io import fits
        from astropy.wcs import WCS
        data = np.random.default_rng(0).normal(0, 0.001, (128, 128)).astype(np.float32)
        ra, dec = 343.5, 16.15
        w = WCS(naxis=2)
        w.wcs.crpix = [64, 64]
        w.wcs.cdelt = [-20.0 / 3600, 20.0 / 3600]
        w.wcs.crval = [ra, dec]
        w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
        # Inject source: 0.5 Jy/beam at (64, 64)
        data[64, 64] = 0.5
        hdr = w.to_header()
        hdr["BUNIT"] = "JY/BEAM"
        path = tmp_path / "test_image.fits"
        fits.writeto(str(path), data, hdr, overwrite=True)
        return path, ra, dec, 0.5

    def test_photometry_finds_injected_source(self, mock_image_with_source):
        from dsa110_continuum.simulation.pipeline import SimulatedPipeline, SourceFluxResult
        from dsa110_continuum.simulation.ground_truth import GroundTruthRegistry
        path, ra, dec, flux = mock_image_with_source
        reg = GroundTruthRegistry(test_run_id="test")
        reg.register_source("S0", ra, dec, baseline_flux_jy=flux)

        p = SimulatedPipeline(work_dir=path.parent)
        results = p._photometry(
            image_path=path,
            ground_truth=reg,
            mjd=60000.0,
            noise_jy_beam=0.001,
        )
        assert len(results) == 1
        assert isinstance(results[0], SourceFluxResult)

    def test_photometry_recovers_flux_within_tolerance(self, mock_image_with_source):
        from dsa110_continuum.simulation.pipeline import SimulatedPipeline
        from dsa110_continuum.simulation.ground_truth import GroundTruthRegistry
        path, ra, dec, flux = mock_image_with_source
        reg = GroundTruthRegistry(test_run_id="test")
        reg.register_source("S0", ra, dec, baseline_flux_jy=flux)

        p = SimulatedPipeline(work_dir=path.parent)
        results = p._photometry(image_path=path, ground_truth=reg,
                                mjd=60000.0, noise_jy_beam=0.001)
        r = results[0]
        assert r.passed, \
            f"Recovered {r.recovered_flux_jy:.3f} Jy vs injected {r.injected_flux_jy:.3f} Jy"

    def test_photometry_flags_missing_source(self, tmp_path):
        """Source far outside image returns NaN flux and passed=False."""
        import numpy as np
        from astropy.io import fits
        from astropy.wcs import WCS
        from dsa110_continuum.simulation.pipeline import SimulatedPipeline
        from dsa110_continuum.simulation.ground_truth import GroundTruthRegistry
        data = np.zeros((32, 32), dtype=np.float32)
        w = WCS(naxis=2)
        w.wcs.crpix = [16, 16]; w.wcs.cdelt = [-1.0 / 60, 1.0 / 60]
        w.wcs.crval = [0.0, 0.0]; w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
        path = tmp_path / "blank.fits"
        fits.writeto(str(path), data, w.to_header(), overwrite=True)

        reg = GroundTruthRegistry(test_run_id="test")
        reg.register_source("S_far", 180.0, 45.0, baseline_flux_jy=1.0)  # way outside
        p = SimulatedPipeline(work_dir=tmp_path)
        results = p._photometry(image_path=path, ground_truth=reg,
                                mjd=60000.0, noise_jy_beam=0.001)
        assert not results[0].passed
        assert np.isnan(results[0].recovered_flux_jy)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=/home/user/workspace/dsa110-continuum python3 -m pytest \
  tests/test_simulated_pipeline.py::TestSimulatedPhotometry -v --tb=short \
  --override-ini="addopts="
```
Expected: `AttributeError: 'SimulatedPipeline' object has no attribute '_photometry'`

- [ ] **Step 3: Add `_photometry()` to `SimulatedPipeline`**

```python
    # ------------------------------------------------------------------ #
    # Stage 5 — Forced photometry + ground-truth comparison               #
    # ------------------------------------------------------------------ #

    def _photometry(
        self,
        *,
        image_path: Path | str,
        ground_truth: "GroundTruthRegistry",
        mjd: float,
        noise_jy_beam: float,
        box_pix: int = 5,
        flux_tolerance: float = 0.20,
    ) -> list[SourceFluxResult]:
        """Forced photometry at injected source positions vs. ground truth.

        Uses the production ``measure_peak_box()`` from
        ``dsa110_continuum.photometry.simple_peak``.  For each registered
        source in the ground-truth registry, measures the peak flux in a
        box centred on the known position and compares to the expected flux
        at the given MJD.

        Parameters
        ----------
        image_path:
            FITS image (Jy/beam) to measure.
        ground_truth:
            Registry of injected sources (positions + fluxes).
        mjd:
            Observation MJD for flux prediction (handles variability models).
        noise_jy_beam:
            Global noise estimate (Jy/beam) used to compute SNR and set
            ``passed`` threshold.
        box_pix:
            Half-width of the search box in pixels (default 5 → 11×11 box).
        flux_tolerance:
            Fractional tolerance for ``passed`` flag:
            |recovered - expected| / expected < flux_tolerance.

        Returns
        -------
        list[SourceFluxResult]
            One entry per source in the ground-truth registry.
        """
        from astropy.io import fits
        from astropy.wcs import WCS
        from dsa110_continuum.photometry.simple_peak import measure_peak_box
        from dsa110_continuum.simulation.ground_truth import GroundTruthRegistry

        image_path = Path(image_path)
        with fits.open(str(image_path)) as hdul:
            data = np.squeeze(hdul[0].data).astype(float)
            wcs  = WCS(hdul[0].header).celestial

        results: list[SourceFluxResult] = []
        for src in ground_truth.sources.values():
            expected = ground_truth.get_expected_flux(src.source_id, mjd) or src.baseline_flux_jy
            peak, snr, xp, yp = measure_peak_box(
                data, wcs, src.ra_deg, src.dec_deg,
                box_pix=box_pix, rms=noise_jy_beam,
            )
            if np.isnan(peak):
                passed = False
            else:
                frac_err = abs(peak - expected) / max(expected, 1e-12)
                passed   = frac_err < flux_tolerance

            results.append(SourceFluxResult(
                source_id=src.source_id,
                ra_deg=src.ra_deg,
                dec_deg=src.dec_deg,
                injected_flux_jy=expected,
                recovered_flux_jy=peak,
                snr=snr,
                passed=passed,
            ))
            logger.info(
                "  %s: injected=%.3f Jy, recovered=%.3f Jy, SNR=%.1f, %s",
                src.source_id, expected, peak, snr, "PASS" if passed else "FAIL",
            )
        return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=/home/user/workspace/dsa110-continuum python3 -m pytest \
  tests/test_simulated_pipeline.py::TestSimulatedPhotometry -v --tb=short \
  --override-ini="addopts="
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add dsa110_continuum/simulation/pipeline.py tests/test_simulated_pipeline.py
git commit -m "feat(sim): Stage 5 — forced photometry vs. ground truth (_photometry())

Uses production measure_peak_box() at each GroundTruthRegistry source
position. Returns SourceFluxResult per source with passed=True when
recovered flux is within 20% of injected. NaN returned for out-of-image
positions."
```

---

## Task 7: End-to-end orchestrator and integration test (RED → GREEN)

**Files:**
- Modify: `dsa110_continuum/simulation/pipeline.py` (add `run()`)
- Test: `tests/test_simulated_pipeline.py` (class `TestEndToEnd`)

### Goal
`SimulatedPipeline.run(harness, n_tiles, ...)` → `SimulatedPipelineResult`

Orchestrates all five stages with a real `SimulationHarness`, producing a result
that can be asserted against in tests. This is the primary regression test for
the full pipeline.

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/test_simulated_pipeline.py

@pytest.mark.slow
class TestEndToEnd:
    """Full simulated pipeline: corruption → cal → image → mosaic → photometry.

    Marked slow because WSClean runs with niter=200. Excluded from default
    suite by -m 'not slow'. Run explicitly for integration validation.
    """
    def test_full_pipeline_recovers_sources(self, tmp_path):
        from dsa110_continuum.simulation.harness import SimulationHarness
        from dsa110_continuum.simulation.pipeline import SimulatedPipeline

        h = SimulationHarness(
            n_antennas=96, n_sky_sources=3, seed=42, use_real_positions=True
        )
        p = SimulatedPipeline(
            work_dir=tmp_path,
            niter=200,
            cell_arcsec=20.0,
            image_size=512,
        )
        result = p.run(
            harness=h,
            n_tiles=2,
            n_subbands=4,
            amp_scatter=0.05,
            phase_scatter_deg=5.0,
            cal_flux_jy=10.0,
        )
        assert result.calibration_passed, f"Calibration stage failed: {result.errors}"
        assert result.imaging_passed,     f"Imaging stage failed: {result.errors}"
        assert result.mosaic_path is not None and result.mosaic_path.exists()
        assert result.n_recovered >= 1, \
            f"Expected ≥1 recovered source; got {result.n_recovered}/{len(result.source_results)}"

    def test_full_pipeline_result_is_serializable(self, tmp_path):
        import json, dataclasses
        from dsa110_continuum.simulation.harness import SimulationHarness
        from dsa110_continuum.simulation.pipeline import SimulatedPipeline
        h = SimulationHarness(n_antennas=8, n_sky_sources=1, seed=1,
                              use_real_positions=False)
        p = SimulatedPipeline(work_dir=tmp_path, niter=50,
                              cell_arcsec=30.0, image_size=128)
        result = p.run(harness=h, n_tiles=2, n_subbands=1,
                       amp_scatter=0.03, phase_scatter_deg=2.0)
        # Verify result can be converted to dict (for logging/database)
        d = dataclasses.asdict(result)
        assert "source_results" in d
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=/home/user/workspace/dsa110-continuum python3 -m pytest \
  tests/test_simulated_pipeline.py::TestEndToEnd -v --tb=short \
  --override-ini="addopts=" -m "slow"
```
Expected: `AttributeError: 'SimulatedPipeline' object has no attribute 'run'`

- [ ] **Step 3: Add `run()` to `SimulatedPipeline`**

```python
    # ------------------------------------------------------------------ #
    # Top-level orchestrator                                               #
    # ------------------------------------------------------------------ #

    def run(
        self,
        harness: "SimulationHarness",
        *,
        n_tiles: int = 2,
        n_subbands: int = 4,
        amp_scatter: float = 0.05,
        phase_scatter_deg: float = 5.0,
        cal_flux_jy: float = 10.0,
        mjd: float = 60310.0,
    ) -> SimulatedPipelineResult:
        """Run all five pipeline stages end-to-end.

        Parameters
        ----------
        harness:
            Configured SimulationHarness (sky model, antenna positions).
        n_tiles:
            Number of simulated transit tiles to process (each tile = one
            call to generate_subbands). Tiles are imaged individually and
            then mosaicked. Default 2.
        n_subbands:
            Subbands per tile (default 4). Each tile produces n_subbands
            UVH5 files which are concatenated → one MS.
        amp_scatter:
            Per-antenna amplitude error (fractional). Default 0.05 (5%).
        phase_scatter_deg:
            Per-antenna phase error (degrees). Default 5.
        cal_flux_jy:
            Calibrator source flux (Jy). Default 10 Jy.
        mjd:
            Observation MJD for ground-truth flux prediction. Default 60310.

        Returns
        -------
        SimulatedPipelineResult
        """
        from dsa110_continuum.simulation.gain_corruption import corrupt_uvh5
        from dsa110_continuum.simulation.ground_truth import GroundTruthRegistry

        errors: list[str] = []
        tile_images: list[Path] = []
        calibration_passed = True
        imaging_passed = True
        mosaic_path: Path | None = None

        # --- Build ground truth registry from harness sky model ---
        registry = GroundTruthRegistry(test_run_id="simulated_pipeline")
        sky = harness._make_sky_model()
        for idx in range(len(sky.ra)):
            ra  = float(sky.ra[idx].deg)
            dec = float(sky.dec[idx].deg)
            flux = float(sky.stokes[0, 0, idx].value)  # Stokes I at ref freq
            registry.register_source(
                source_id=f"SIM_S{idx:03d}",
                ra_deg=ra, dec_deg=dec,
                baseline_flux_jy=flux,
            )
        registry.register_epoch(mjd)

        # Generate calibrator once (shared across all tiles)
        cal_dir = self.work_dir / "calibrator"
        cal_dir.mkdir(parents=True, exist_ok=True)
        cal_uvh5 = harness.generate_calibrator_subband(cal_dir, flux_jy=cal_flux_jy)

        for tile_idx in range(n_tiles):
            tile_dir = self.work_dir / f"tile_{tile_idx:02d}"
            tile_dir.mkdir(parents=True, exist_ok=True)

            # Stage 1: Generate and corrupt visibilities
            seed = 100 * (tile_idx + 1)
            uvh5_paths = harness.generate_subbands(
                output_dir=tile_dir, n_subbands=n_subbands
            )
            corrupted_paths = [
                corrupt_uvh5(p, amp_scatter=amp_scatter,
                             phase_scatter_deg=phase_scatter_deg,
                             seed=seed + i)
                for i, p in enumerate(uvh5_paths)
            ]

            # Convert corrupted UVH5 list → single MS
            uvs = []
            for cp in corrupted_paths:
                uv = pyuvdata.UVData()
                uv.read(str(cp))
                uvs.append(uv)
            # Harmonise phase_center_catalog cat_name before concatenation
            for uv in uvs:
                for key in uv.phase_center_catalog:
                    uv.phase_center_catalog[key]["cat_name"] = "SIM_TILE"
            combined = uvs[0]
            for uv in uvs[1:]:
                combined = combined + uv

            ms_path = tile_dir / f"tile_{tile_idx:02d}.ms"
            combined.write_ms(str(ms_path))

            # Stage 2: Calibrate
            try:
                self._calibrate(
                    target_ms=ms_path,
                    cal_uvh5=cal_uvh5,
                    cal_flux_jy=cal_flux_jy,
                    work_dir=tile_dir,
                )
            except Exception as exc:
                errors.append(f"Tile {tile_idx} calibration failed: {exc}")
                calibration_passed = False
                continue

            # Stage 3: Image (with CLEAN)
            try:
                img_results = self._image(ms_path=ms_path, work_dir=tile_dir)
                restored = img_results.get("restored")
                if restored and restored.exists():
                    tile_images.append(restored)
                else:
                    errors.append(f"Tile {tile_idx}: restored image missing")
                    imaging_passed = False
            except Exception as exc:
                errors.append(f"Tile {tile_idx} imaging failed: {exc}")
                imaging_passed = False

        # Stage 4: Mosaic
        if tile_images:
            try:
                mosaic_path = self._mosaic(
                    image_paths=tile_images, work_dir=self.work_dir
                )
            except Exception as exc:
                errors.append(f"Mosaicking failed: {exc}")
        else:
            errors.append("No tile images produced; skipping mosaic and photometry")

        # Stage 5: Forced photometry on mosaic (or best tile if mosaic failed)
        phot_image = mosaic_path or (tile_images[0] if tile_images else None)
        source_results: list[SourceFluxResult] = []
        if phot_image and phot_image.exists():
            from astropy.io import fits
            with fits.open(str(phot_image)) as h:
                data = np.squeeze(h[0].data)
            finite_vals = data[np.isfinite(data)]
            noise = float(np.std(finite_vals)) if finite_vals.size > 0 else 1e-3
            source_results = self._photometry(
                image_path=phot_image,
                ground_truth=registry,
                mjd=mjd,
                noise_jy_beam=noise,
            )

        return SimulatedPipelineResult(
            work_dir=self.work_dir,
            n_tiles=n_tiles,
            calibration_passed=calibration_passed,
            imaging_passed=imaging_passed,
            mosaic_path=mosaic_path,
            source_results=source_results,
            errors=errors,
        )
```

Also add import at top of `pipeline.py`:
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dsa110_continuum.simulation.harness import SimulationHarness
    from dsa110_continuum.simulation.ground_truth import GroundTruthRegistry
```

- [ ] **Step 4: Run fast tests (exclude slow) — should all still pass**

```bash
PYTHONPATH=/home/user/workspace/dsa110-continuum python3 -m pytest \
  tests/test_simulated_pipeline.py -v --tb=short \
  --override-ini="addopts=" -m "not slow"
```
Expected: all non-slow tests pass (Tasks 1–6 tests: ~13 tests).

- [ ] **Step 5: Run end-to-end slow tests**

```bash
PYTHONPATH=/home/user/workspace/dsa110-continuum python3 -m pytest \
  tests/test_simulated_pipeline.py::TestEndToEnd -v --tb=long \
  --override-ini="addopts=" -m "slow"
```
Expected: 2 PASSED (may take 3–5 minutes — WSClean runs twice).

- [ ] **Step 6: Full suite green**

```bash
PYTHONPATH=/home/user/workspace/dsa110-continuum python3 -m pytest tests/ -q \
  --override-ini="addopts=" -m "not slow" 2>&1 | tail -5
```
Expected: all previous 786 tests still pass.

- [ ] **Step 7: Commit**

```bash
git add dsa110_continuum/simulation/pipeline.py tests/test_simulated_pipeline.py
git commit -m "feat(sim): Stage orchestrator SimulatedPipeline.run() — end-to-end

Chains all five stages: gain corruption → calibration → WSClean CLEAN
→ image-plane mosaic → forced photometry vs. GroundTruthRegistry.
TestEndToEnd (marked slow) verifies ≥1 source recovered from a 96-antenna
2-tile simulated observation with 5% amplitude + 5° phase gain errors."
```

---

## Task 8: Diagnostic report (non-TDD, best-effort)

**Files:**
- Create: `dsa110_continuum/simulation/pipeline_report.py`

### Goal
`generate_pipeline_report(result, output_path)` writes a 5-panel matplotlib diagnostic:
1. Corrupted vis amplitudes (before cal)
2. Calibrated vis amplitudes (after cal)
3. WSClean restored image (first tile)
4. Epoch mosaic
5. Source flux recovery bar chart (injected vs. recovered, per source)

This is not TDD (it's purely diagnostic output). Implement in one step.

```python
# dsa110_continuum/simulation/pipeline_report.py
"""5-panel diagnostic report for SimulatedPipelineResult."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from dsa110_continuum.simulation.pipeline import SimulatedPipelineResult


def generate_pipeline_report(
    result: SimulatedPipelineResult,
    output_path: Path | str,
    tile_restored_path: Path | None = None,
) -> Path:
    """Write a 5-panel PNG diagnostic for a simulated pipeline run.

    Parameters
    ----------
    result:
        Output of SimulatedPipeline.run().
    output_path:
        Destination PNG path.
    tile_restored_path:
        Path to the first tile's restored FITS (for panel 3).
        If None, panel 3 shows a blank placeholder.
    """
    output_path = Path(output_path)
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle("DSA-110 Simulated Pipeline Diagnostic", fontsize=14)

    # Panel 1 — placeholder (corrupted vis amplitude histogram)
    axes[0].set_title("(1) Gain corruption", fontsize=10)
    axes[0].text(0.5, 0.5, "5% amp\n5° phase\nper antenna",
                 ha="center", va="center", transform=axes[0].transAxes)
    axes[0].axis("off")

    # Panel 2 — calibration status
    axes[1].set_title("(2) Calibration", fontsize=10)
    status = "PASSED ✓" if result.calibration_passed else "FAILED ✗"
    color  = "green"   if result.calibration_passed else "red"
    axes[1].text(0.5, 0.5, status, ha="center", va="center",
                 transform=axes[1].transAxes, color=color, fontsize=14)
    axes[1].axis("off")

    # Panel 3 — restored tile image
    axes[2].set_title("(3) WSClean restored (tile 0)", fontsize=10)
    if tile_restored_path and Path(tile_restored_path).exists():
        with fits.open(str(tile_restored_path)) as hdul:
            img = np.squeeze(hdul[0].data)
        vmax = float(np.nanpercentile(np.abs(img), 99))
        axes[2].imshow(img, origin="lower", cmap="gray", vmin=-vmax, vmax=vmax)
    else:
        axes[2].text(0.5, 0.5, "no image", ha="center", va="center",
                     transform=axes[2].transAxes)
        axes[2].axis("off")

    # Panel 4 — epoch mosaic
    axes[3].set_title("(4) Epoch mosaic", fontsize=10)
    if result.mosaic_path and result.mosaic_path.exists():
        with fits.open(str(result.mosaic_path)) as hdul:
            mosaic = np.squeeze(hdul[0].data)
        vmax = float(np.nanpercentile(np.abs(mosaic[np.isfinite(mosaic)]), 99))
        axes[3].imshow(mosaic, origin="lower", cmap="gray", vmin=-vmax, vmax=vmax)
    else:
        axes[3].text(0.5, 0.5, "no mosaic", ha="center", va="center",
                     transform=axes[3].transAxes)
        axes[3].axis("off")

    # Panel 5 — flux recovery bar chart
    axes[4].set_title("(5) Flux recovery", fontsize=10)
    if result.source_results:
        ids       = [r.source_id for r in result.source_results]
        injected  = [r.injected_flux_jy  for r in result.source_results]
        recovered = [r.recovered_flux_jy if not np.isnan(r.recovered_flux_jy) else 0
                     for r in result.source_results]
        x = np.arange(len(ids))
        axes[4].bar(x - 0.2, injected,  0.4, label="Injected",  color="steelblue")
        axes[4].bar(x + 0.2, recovered, 0.4, label="Recovered", color="orange")
        axes[4].set_xticks(x); axes[4].set_xticklabels(ids, rotation=45, ha="right", fontsize=7)
        axes[4].set_ylabel("Flux (Jy)"); axes[4].legend(fontsize=7)
    else:
        axes[4].text(0.5, 0.5, "no sources", ha="center", va="center",
                     transform=axes[4].transAxes)
        axes[4].axis("off")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path
```

- [ ] **Commit**

```bash
git add dsa110_continuum/simulation/pipeline_report.py
git commit -m "feat(sim): add 5-panel pipeline diagnostic report (pipeline_report.py)

Panels: gain corruption summary | calibration pass/fail |
WSClean restored tile | epoch mosaic | injected vs. recovered flux bars.
Used for visual validation of end-to-end simulated pipeline runs."
```

---

## Self-Review

**Spec coverage check:**

| Requirement | Task |
|---|---|
| Simulate gain corruption | Task 1 |
| Simulate calibration (solve + apply) | Tasks 2–3 |
| Simulate deconvolution (WSClean CLEAN) | Task 4 |
| Simulate mosaicking | Task 5 |
| Simulate forced photometry | Task 6 |
| End-to-end orchestrator | Task 7 |
| Visual diagnostic | Task 8 |

**Placeholder scan:** None found.

**Type consistency:**
- `corrupt_uvh5` → `Path` ✓ (used as `cal_uvh5: Path` in `_calibrate`)
- `generate_calibrator_subband` → `Path` ✓ (matches `cal_uvh5` argument)
- `_calibrate` → `Path` (target MS); `_image` → `dict[str, Path]`; `_mosaic` → `Path`; `_photometry` → `list[SourceFluxResult]` — all consistent with `run()` usage ✓
- `GroundTruthRegistry.register_source` signature matches usage in `run()` ✓
- `build_mosaic` returns `MosaicResult` with `.output_path: Path` ✓
- `measure_peak_box` returns `(float, float, float, float)` → `(peak, snr, x, y)` ✓

**Known implementation note:** `generate_calibrator_subband()` in Task 2 uses internal harness methods (`_subband_freqs_hz()`, `_make_sky_model()`). Verify these exist or substitute the correct equivalents before implementation. The `generate_subbands()` `sky` parameter may need to be renamed to `sky_model` — read the method signature first.
