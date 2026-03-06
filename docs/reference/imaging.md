# Reference: Imaging

Source: `/data/dsa110-contimg/backend/src/dsa110_contimg/core/imaging/`
Files: `params.py`, `cli_imaging.py`, `worker.py`, `masks.py`

---

## Validated default WSClean parameters

From `ImagingParams` dataclass (`params.py`) and `run_wsclean()` hardcodes
(`cli_imaging.py` lines 341–343). Parameters marked (hardcoded) are set inside
`run_wsclean()` regardless of `ImagingParams` — they cannot be overridden via the
dataclass.

| Parameter | Value | Notes |
|---|---|---|
| `imsize` | **2400** pixels | Covers primary beam |
| `cell_arcsec` | **6.0** arcsec | ~2 px per ~12" synthesized beam |
| `weighting` | `briggs`, robust=0.5 | Standard continuum |
| `specmode` | `mfs`, `nterms=1` | Single-term MFS |
| `deconvolver` | `hogbom` | Point sources; use multiscale for survey preset |
| `niter` | 1000 (standard) | 10000 (survey), 5000 (high_precision) |
| `threshold` | `"0.005Jy"` | 5 mJy standard |
| `-auto-mask` | **5σ** (hardcoded) | `cli_imaging.py` line 341 |
| `-auto-threshold` | **1.0σ** (hardcoded) | line 342 |
| `-mgain` | **0.8** (hardcoded) | line 343 |
| `gridder` | `idg` (default) | `wgridder` in `run_pipeline.py` |
| `-idg-mode` | `gpu` | Falls back to `cpu` if no GPU |
| `uvrange` | `">1klambda"` | Exclude short baselines |
| `pblimit` | 0.2 | |
| `-apply-primary-beam` | Always set | Via EveryBeam |
| `-pol I` | Always set | Stokes I only |
| `-reorder` | Always set | Required for multi-SPW MS |
| Memory | 32 GB (imsize ≥ 2400), 16 GB otherwise | |
| Timeout | 1800 s | Override via `WSCLEAN_DOCKER_TIMEOUT` |

**Survey preset** (`ImagingParams.for_survey`): `deconvolver=multiscale`,
`nterms=2`, `niter=10000`, `auto_mask=4.0`, `auto_threshold=0.5`, `robust=0.0`,
multiscale scales `[0, 5, 15, 45]` pixels.

**Development tier**: cell ×4 coarser, `niter ≤ 300`, unicat threshold 10 mJy.
Explicitly flagged non-science quality in the source.

---

## IDG SPW-merge workflow

IDG requires a single-SPW MS. `DirectSubbandWriter` by default does NOT merge SPWs
(it writes a multi-SPW MS). When `gridder == "idg"` and the input MS has > 1 SPW:

1. `merge_spws(ms_path)` → temporary `{stem}_idg_merged.ms`
2. Run WSClean on the merged MS
3. Delete temporary merged MS in `finally` block

This is the standard production path. It happens automatically inside `image_ms()`.

---

## Sky model seeding — validated two-step WSClean predict workflow

When `use_unicat_mask=True` and `unicat_min_mjy` is set (`cli_imaging.py` lines 928–1228):

```
Step 1: wsclean -draw-model <sources.txt>
              -draw-frequencies <freq> <bw>
              -draw-spectral-terms 2
              -size <N> <N> -scale <cell>arcsec
              -draw-centre <RA> <Dec>
              -name <prefix>
        → output: {prefix}-term-0.fits

Step 2: rename {prefix}-term-0.fits → {prefix}-model.fits

Step 3: wsclean -predict -reorder -name <prefix> <ms_path>
        → populates MODEL_DATA
```

The `-reorder` flag in Step 3 is required for multi-SPW MS.
`savemodel="none"` in the main imaging call prevents overwriting the seeded model.

Seeding radius: `min(image_half_diagonal, FWHM × sqrt(-ln(pblimit)) / sqrt(-ln(0.5)))`
at `pblimit=0.2`, where `FWHM = 1.22λ/D`, `D = 4.7 m`.

Catalog priority for unified source list: **FIRST > RACS > NVSS**.
Default `min_mjy=2.0 mJy` in `make_unified_wsclean_list()`.

---

## Catalog mask generation

`create_catalog_fits_mask()` (`cli_imaging.py` lines 1309–1342):
- When `use_unicat_mask=True`, generates a FITS binary mask with circular apertures
  of `mask_radius_arcsec=60.0` around all catalog sources above `unicat_min_mjy`.
- Claimed 2–4× faster imaging vs no mask.
- Passed to WSClean via `-fits-mask <path>`.

---

## Galvin adaptive clipping (`masks.py`)

`prepare_cleaning_mask()` → `minimum_absolute_clip()` (lines 135–171):

```
mask = image > (|rolling_box_minimum| × increase_factor)
```

Default parameters:
- `increase_factor = 2.0`
- `box_size = 100` pixels
- `adaptive_max_depth = 3` (with `SelfCalConfig.galvin_adaptive_depth=3`)

Adaptive path (up to 3 rounds): doubles box size if skewed islands are detected.
Skew test: `positive_fraction > 0.5 + 0.2` (i.e., > 70% positive pixels in box).
If skewed, replaces minimum values only within skewed islands, recalculates mask.

This is a sliding-window artifact suppression technique, not a standard WSClean feature.
Required for self-calibration iterations to suppress sidelobes from bright sources.

---

## Primary beam correction

`-apply-primary-beam` passes beam correction to EveryBeam.
- Requires `OBSERVATION::TELESCOPE_NAME = DSA_110` (see `conversion-and-qa.md`).
- EveryBeam uses Airy disk model for DSA-110 (added in EveryBeam 0.7.2).
- `-grid-with-beam` (direction-dependent, A-projection): used in SCIENCE/DEEP
  mosaics only, not single-field imaging in `run_pipeline.py`.
- `-reuse-primary-beam`: available for self-cal iterations (60–80% faster after first run).
- `EVERYBEAM_PATH` env var, default `/opt/everybeam`.
