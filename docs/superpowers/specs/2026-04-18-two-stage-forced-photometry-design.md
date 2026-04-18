# Two-Stage Forced Photometry Design

**Date:** 2026-04-18  
**Author:** Jakob Faber  
**Status:** Approved

---

## Goal

Wire a two-stage forced photometry path into the existing pipeline: a fast
peak-in-beam-aperture coarse pass using `simple_peak.measure_peak_box`, followed
by a Condon-weighted convolution fine pass using `forced.measure_many` for
sources that survive an SNR gate.  A `--method simple_peak` flag on
`scripts/forced_photometry.py` enables the coarse path in isolation (for
commissioning, quick-look QA, and testing without real catalog databases).

In simulation mode (no real catalog), both paths operate on injected source
positions from the synthetic sky model.

---

## Architecture

### Layers

```
scripts/forced_photometry.py          ← CLI entry point (modified)
    └─ dsa110_continuum/photometry/
           simple_peak.py             ← coarse pass (existing, unchanged)
           forced.py                  ← fine pass  (existing, unchanged)
           two_stage.py               ← NEW: orchestrates coarse→fine pipeline
```

The new `two_stage.py` module is the only net-new library code.
`scripts/forced_photometry.py` grows one flag and calls `two_stage` instead of
`forced.measure_many` directly.

### Data flow (production, `--method two_stage`)

```
catalog cone_search
    → all_coords  (N positions)
    → coarse pass: simple_peak.measure_peak_box per position
    → filter: peak_jyb / local_rms >= snr_coarse_min  →  survivor_coords  (M ≤ N)
    → fine pass: forced.measure_many(survivor_coords)
    → merge: attach coarse SNR + upper-limit flag to ForcedPhotometryResult
    → CSV output + QA log
```

### Data flow (simulation, `--sim`)

```
synthetic sky model (SkyModel from dsa110_continuum.simulation.ground_truth)
    → injected source (RA, Dec, flux_Jy) list
    → same coarse→fine pipeline
    → ratio = measured_peak / (injected_flux × beam_area_sr / pixel_area_sr)
           (beam area correction applied so ratio ≈ 1.0 for unresolved sources)
    → QA figure: ratio vs. injected flux, coloured by N_coverage
```

---

## Components

### `dsa110_continuum/photometry/two_stage.py`

Single public function:

```python
def run_two_stage(
    mosaic_path: str,
    coords: list[tuple[float, float]],         # (ra_deg, dec_deg)
    *,
    snr_coarse_min: float = 3.0,               # coarse SNR gate
    box_pix: int = 5,                          # simple_peak box half-width
    global_rms: float | None = None,           # if None, estimated from image MAD
    # fine-pass kwargs forwarded to measure_many:
    use_cluster_fitting: bool = False,
    noise_map_path: str | None = None,
) -> list[ForcedPhotometryResult]:
    """Run coarse peak-in-box pass, then Condon fine pass on survivors."""
```

Returns a `list[ForcedPhotometryResult]` — same type as `measure_many` — so the
rest of `forced_photometry.py` (CSV writer, QA summary) is unchanged.

Extra fields populated on each result:
- `coarse_snr`: SNR from the simple-peak pass (stored in a lightweight wrapper)
- `passed_coarse`: bool flag

Because `ForcedPhotometryResult` is a frozen dataclass we cannot add fields to
it without modifying `forced.py`.  Instead `two_stage.py` returns a parallel
`list[CoarseAugment]` dataclass alongside the result list, and the CSV writer in
`forced_photometry.py` zips them.

```python
@dataclass
class CoarseAugment:
    ra_deg: float
    dec_deg: float
    coarse_peak_jyb: float
    coarse_snr: float
    passed_coarse: bool
```

### `scripts/forced_photometry.py` changes

- Add `--method {two_stage,simple_peak,condon}` (default: `two_stage`)
- Add `--sim` flag: use injected source positions from synthetic sky model instead of catalog
- Add `--snr-coarse` float (default 3.0): coarse SNR gate
- Beam area correction applied in sim mode before computing flux ratios
- Extra CSV columns when `--method two_stage`: `coarse_peak_jy`, `coarse_snr`, `passed_coarse`

### `scripts/forced_photometry.py` — `--method simple_peak`

Calls `simple_peak.measure_peak_box` directly for all positions, skips fine
pass entirely.  Output CSV has columns: `source_name`, `ra_deg`, `dec_deg`,
`catalog_flux_jy` (or `injected_flux_jy` in sim mode), `measured_flux_jy`,
`snr`.

---

## Beam Area Correction (sim mode)

Injected flux is total flux in Jy.  The image pixel values are in Jy/beam.
For an unresolved source the peak pixel ≈ S_total × (pixel_area / beam_area).

```
beam_area_sr  = (π / (4 ln 2)) × BMAJ_rad × BMIN_rad
pixel_area_sr = pixel_scale_rad²
correction    = beam_area_sr / pixel_area_sr
ratio         = measured_peak_jyb × correction / injected_flux_jy
```

This correction is read from the FITS header (`BMAJ`, `BMIN`, `CDELT2`).

---

## Testing strategy

All tests use the simulated mosaic already on disk:
`pipeline_outputs/step6/step6_mosaic.fits` (regenerable via `run_step6.py`).

Injected positions come from:
`dsa110_continuum.simulation.ground_truth.SkyModel` with `seed=42`.

Tests live in `tests/test_two_stage_photometry.py`.

Key test cases:
1. `test_coarse_pass_returns_finite` — all injected positions return finite peak values
2. `test_snr_gate_filters_low_snr` — artificially low rms causes all sources to pass; artificially high rms causes all to fail
3. `test_fine_pass_survivors_only` — verify `measure_many` is called only with survivor coords
4. `test_beam_correction_ratio` — ratio ≈ 1.0 ± 0.5 for bright isolated sources (T3·S1, T3·S4)
5. `test_simple_peak_method` — `--method simple_peak` script flag produces valid CSV with correct column set
6. `test_two_stage_method` — `--method two_stage` script flag produces valid CSV with `coarse_snr` column

---

## Out of scope

- Real catalog querying (SQLite databases not present in sandbox; tested via mock)
- Condon error computation (already in `forced.py`; not changed)
- Source finding / blind extraction (separate future step)
- Self-calibration loop

---

## Files touched

| File | Action |
|------|--------|
| `dsa110_continuum/photometry/two_stage.py` | Create |
| `scripts/forced_photometry.py` | Modify |
| `tests/test_two_stage_photometry.py` | Create |
