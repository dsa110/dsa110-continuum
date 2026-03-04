# Per-Epoch Gain Calibration — Design

**Date:** 2026-03-04
**Status:** Approved

---

## Science motivation

The current pipeline applies a single static gain table derived from a 2026-01-25
calibrator observation to every tile, regardless of date or epoch. Atmospheric
amplitude and phase variations on timescales of minutes to hours are not corrected.
This produces epoch-to-epoch flux scale offsets that appear as spurious variability
in the light curves, contaminating the DSA/NVSS ratio and variability metrics.

Per-epoch gain calibration derives fresh gain solutions once per 1-hour mosaic epoch,
anchored to the catalog flux scale (FIRST/RACS/NVSS/VLASS) and refined by one round
of self-calibration against a WSClean clean image.

---

## Chosen approach: Catalog bootstrap + one self-cal round

From three alternatives, this was selected as the best balance of calibration quality
and compute time (~7 min per epoch vs ~2 min for catalog-only or ~30 min for full
multi-round self-cal).

**Rejected: Catalog-only** — Doesn't correct amplitude scale. NVSS/FIRST flux scatter
(5–10%) would remain as per-epoch offsets in the DSA/NVSS ratio.

**Rejected: Multi-round self-cal** — ~30 min per epoch × 12 epochs per day. The 5-min
tile duration limits SNR gains from shorter solution intervals; rounds beyond the first
buy little. Overkill.

---

## Architecture

A new module `dsa110_continuum/calibration/epoch_gaincal.py` exposes one public
function:

```python
def calibrate_epoch(
    epoch_ms_paths: list[str],   # MS paths for all 12 tiles in the epoch (raw, unphaseshifted)
    bp_table: str,               # Daily bandpass table (validated to exist)
    work_dir: str,               # Scratch space for G tables and WSClean output
    *,
    refant: str = "103",
    min_flux_mjy: float = 5.0,
    source_radius_deg: float = 0.3,
) -> str | None:                 # Path to solved ap.G table, or None on failure
```

`batch_pipeline.py` calls this **before** the tile imaging loop for each epoch:

```
For each epoch:
  1. calibrate_epoch(epoch_ms_paths, bp_table, work_dir) → epoch_G | None
  2. _md.G_TABLE = epoch_G  (or daily static if None)
  3. For each tile: phaseshift → applycal(BP + epoch_G) → wsclean image
  4. Build mosaic, forced photometry (unchanged)
```

---

## Central tile selection

`epoch_gaincal.select_calibration_tile_from_ms(epoch_ms_paths)` bridges the pipeline's
MS-path-based world to `mosaic_scheduler.MosaicCalibrationManager.select_gain_calibration_tile()`:

1. Read the median field phase center (FIELD::PHASE_DIR) from each of the two central
   MSes (indices 5 and 6 of the sorted 12-tile list)
2. Call `count_bright_sources_in_tile(ra, dec, min_flux_mjy, radius_deg)` for each
3. Return the MS path whose pointing contains more bright catalog sources

Catalog priority in source count: FIRST > RACS > NVSS > VLASS (via unified catalog).

---

## Gain solve workflow (5 steps on the central tile)

The central tile's meridian MS is created by phaseshifting if it doesn't already exist,
then used in-place for the calibration solve.

| Step | Action | Output |
|------|--------|--------|
| 1 | `apply_to_target(central_ms, gaintables=[bp_table])` | CORRECTED\_DATA with BP correction |
| 2 | `make_unified_skymodel()` + `predict_from_skymodel_wsclean()` | MODEL\_DATA from FIRST+RACS+NVSS+VLASS |
| 3 | `gaincal(calmode='p', solint='inf', gaintable=[bp_table])` | `epoch_{label}.p.G` |
| 4 | WSClean `-niter 1000 -auto-threshold 3 -save-model` on central tile | MODEL\_DATA from clean components |
| 5 | `gaincal(calmode='ap', solint='inf', gaintable=[bp_table, p.G])` | `epoch_{label}.ap.G` ← returned |

All intermediate files (p.G, WSClean FITS) live in `work_dir` and are not exposed to
the caller.

---

## Solution transfer

The returned `epoch_ap.G` path is assigned to `_md.G_TABLE` before the tile imaging
loop. All 12 tiles call `apply_to_target(gaintables=[bp_table, epoch_ap.G])`.

The central tile's meridian MS already has CORRECTED\_DATA from step 1 above. To
prevent `process_ms()` from skipping applycal on it (due to the `needs_calibration()`
heuristic), a `force_recal: bool = False` parameter is added to `process_ms()`. When
`True`, applycal is always called regardless of CORRECTED\_DATA state.

---

## Error handling and fallback

Any exception in `calibrate_epoch()` is caught internally. The function logs `ERROR`
and returns `None`. `batch_pipeline.py` then:
- Keeps `_md.G_TABLE` pointing to the static daily table
- Records `gaincal_status: "fallback"` in the epoch result dict
- Prints `WARNING` in the per-epoch summary table

No epoch is skipped due to gain calibration failure. The daily static table is always
the fallback.

---

## VLASS gap (also addressed)

`make_unified_skymodel()` currently fetches only FIRST, RACS, NVSS. VLASS is supported
by `query_sources()` but omitted. This design adds VLASS at the lowest priority
(FIRST > RACS > NVSS > VLASS), consistent with its lower spatial resolution relative
to FIRST at L-band but broader sky coverage than FIRST.

---

## Files changed

| File | Change |
|------|--------|
| `dsa110_continuum/calibration/epoch_gaincal.py` | **New** — `select_calibration_tile_from_ms()`, `calibrate_epoch()` |
| `dsa110_continuum/calibration/skymodels.py` | Add VLASS to `make_unified_skymodel()` |
| `scripts/mosaic_day.py` | Add `force_recal: bool = False` to `process_ms()` |
| `scripts/batch_pipeline.py` | Call `calibrate_epoch()` per epoch; `--skip-epoch-gaincal` flag |
| `tests/test_epoch_gaincal.py` | **New** — 3 tests |

---

## Success criteria

- `calibrate_epoch()` returns a valid G table path for a real tile MS
- Median DSA/NVSS ratio across all epochs within a day is within 0.8–1.2
- Epoch-to-epoch ratio scatter (σ) is < 0.05 (down from current ~0.3–0.5)
- No epoch is blocked or skipped due to gain calibration failure
