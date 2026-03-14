# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A radio astronomy continuum imaging pipeline for DSA-110 (Deep Synoptic Array, 110 antennas)
at OVRO, ported from the older dsa110-contimg codebase. Science goal: detect variable/transient
compact radio sources (ESEs, AGN flares) via daily sky mosaics and per-source forced photometry.
No web infrastructure — only verified science code.

## Python environment

ALWAYS use the casa6 conda env — it has numpy, casacore, astropy, CASA, and all scientific deps:

    /opt/miniforge/envs/casa6/bin/python

Do NOT use system python3 (3.13, missing scientific deps). The `pip` command maps to
system 3.8; if you need pip, use `/opt/miniforge/envs/casa6/bin/python -m pip`.

## Build / Test / Lint

    # Run all tests (118 tests, ~1 s)
    /opt/miniforge/envs/casa6/bin/python -m pytest tests/ -q

    # Run a single test file or test
    /opt/miniforge/envs/casa6/bin/python -m pytest tests/test_verify_sources.py -q
    /opt/miniforge/envs/casa6/bin/python -m pytest tests/test_verify_sources.py::test_measure_peak_box_returns_correct_flux -q

    # Lint (ruff, configured in pyproject.toml — NumPy docstring convention, 100-char lines)
    ruff check dsa110_continuum/ scripts/ tests/
    ruff format --check dsa110_continuum/ scripts/ tests/

    # Run a pipeline script
    /opt/miniforge/envs/casa6/bin/python scripts/batch_pipeline.py --date 2026-01-25

## Import architecture

`dsa110_continuum/` is the canonical package. The import rename from the old
`dsa110_contimg.core.*` paths is complete (~370 imports replaced across 136 files).
The `__init__.py` re-export layers intentionally still reference old paths — do NOT
change them (the old package's bootstrap chain loads core.calibration.jobs →
register_job; using the new path causes ValueError from double job-registration).

The old package remains installed from `/data/dsa110-contimg/backend/src` and is
still loaded at runtime via those `__init__.py` re-exports. When adding new code,
always use `dsa110_continuum.*` imports; do not add new `dsa110_contimg` references.

## Verified working state

run_pipeline.py produces a calibrated image of 3C454.3 at 12.5 Jy/beam.
Test data: 2026-01-25 HDF5 files at /data/incoming/ on H17.

## Data flow

```
HDF5 (16 subbands × N timestamps) → [conversion/] MS
  → [calibration/] flagging + bandpass/gain solve + applycal
  → [imaging/] phaseshift → WSClean (wgridder/IDG) → FITS tile
  → [mosaic/] tiles → hourly-epoch mosaic (QUICKLOOK or SCIENCE/DEEP)
  → [photometry/] forced photometry → variability metrics → light curves
```

## Package structure (17 submodules)

dsa110_continuum/
  conversion/      - HDF5 to MS (UVH5 subband grouping, phase centre, UVW reconstruction)
  calibration/     - bandpass, gain cal, applycal, phaseshift, self-cal, presets
  imaging/         - WSClean/CASA tclean interface, ImagingParams, sky model seeding
  mosaic/          - QUICKLOOK (image-domain) and SCIENCE/DEEP (visibility-domain) mosaicking
  photometry/      - forced photometry, ESE detection, variability metrics (Mooley eta/Vs/m)
  catalog/         - source catalog management (NVSS, RACS, FIRST, VLA cal list); SQLite backend
  qa/              - delay validation, image quality, pipeline QA hooks
  simulation/      - synthetic UVH5 generation for testing
  visualization/   - diagnostic plots (bandpass, UV coverage, calibration, mosaics, light curves)
  validation/      - MS/image validators, storage checks
  evaluation/      - pipeline stage evaluation harness
  selfcal/         - self-calibration logic
  rfi/             - RFI flagging strategies
  search/          - source searching
  spectral/        - spectral analysis
  pointing/        - pointing corrections
  adapters/        - external tool adapters

## Key scripts

scripts/run_pipeline.py        Single-tile reference run: phaseshift → applycal → WSClean → check flux
scripts/mosaic_day.py          Process all tiles for one date → full-day mosaic (contiguous RA strips)
scripts/batch_pipeline.py      Full orchestration: tiles → hourly-epoch mosaics → forced photometry
scripts/source_finding.py      BANE + Aegean on mosaics → blind source catalog
scripts/forced_photometry.py   Standalone forced photometry against reference catalog
scripts/inventory.py           HDF5 data inventory with conversion status

## Key paths (H17)

/data/incoming/                              raw HDF5 files
/stage/dsa110-contimg/ms/                   Measurement Sets
/opt/miniforge/envs/casa6                   CASA conda env

## Pipeline DB

dsa110 convert queries the pipeline SQLite DB, not the filesystem.
New dates must be indexed first: dsa110 index add --start YYYY-MM-DD --end YYYY-MM-DD --directory /data/incoming

## Calibration tables

Cal tables live at /stage/dsa110-contimg/ms/{date}T22:26:05_0~23.{b,g}.
Until per-date calibration runs are available, symlink new dates from 2026-01-25:
  ln -s /stage/dsa110-contimg/ms/2026-01-25T22:26:05_0~23.b /stage/dsa110-contimg/ms/YYYY-MM-DDT22:26:05_0~23.b
  ln -s /stage/dsa110-contimg/ms/2026-01-25T22:26:05_0~23.g /stage/dsa110-contimg/ms/YYYY-MM-DDT22:26:05_0~23.g
Then run batch_pipeline.py with --cal-date 2026-01-25 to use those tables for a different --date.

## Reference docs

Two layers of documentation exist. Read both before implementing any pipeline subsystem.

### docs/skills/ — implementation notes for the current pipeline

Verified notes on how the code in THIS repository works.
Read before writing new code for any step in this pipeline.

### docs/reference/ — validated knowledge from reference codebases

Distilled source analysis of the OLD dsa110-contimg pipeline and the ASKAP VAST
pipeline. These contain validated numerical parameters, known failure modes, and
instrument-specific constraints that are not obvious from the current codebase alone.

READ THESE BEFORE IMPLEMENTING any of the following:

  flagging.md           AOFlagger Lua strategy, OVRO RFI, validated fractions,
                        two-stage flagging contract (do NOT omit Stage 2)
  calibration.md        K/B/G parameters, DEFAULT_PRESET, self-cal SelfCalConfig,
                        bp_minsnr=5.0 (not the function default of 3.0)
  conversion-and-qa.md  UVH5 ingest workarounds, PyUVData float64/run_check,
                        TELESCOPE_NAME dual-identity, post-conversion QA gates
  imaging.md            WSClean hardcoded flags, sky model seeding two-step workflow,
                        IDG SPW-merge, Galvin adaptive clip
  mosaicking.md         QUICKLOOK vs SCIENCE/DEEP configs, mean-RA wrap bug,
                        -grid-with-beam vs -apply-primary-beam distinction
  photometry-and-ese.md Condon matched-filter, differential photometry reference
                        selection, ESE scoring, variability thresholds
  vast-crossref.md      Variability metric formulas (Vs, m, eta), ForcedPhot library
                        interface, Condon errors, Huber flux-scale correction

## Critical silent failures

The following bugs produce no runtime exception but yield wrong science output.
They must be preserved in any refactoring of conversion, phaseshift, or imaging code.

### 1. FIELD::PHASE_DIR not updated by chgcentre

After running WSClean's chgcentre, the FIELD::PHASE_DIR column in the Measurement Set
may not reflect the new phase centre. It must be explicitly patched:

    update_phase_dir_to_target(ms_path, target_ra_deg, target_dec_deg)

Symptom if missing: CASA tasks and some imaging tools compute phase gradients relative
to the old field centre, producing smeared or offset sources.

### 2. FIELD::REFERENCE_DIR must be synchronised with PHASE_DIR

CASA's ft() task reads FIELD::REFERENCE_DIR (not PHASE_DIR) when computing model
visibilities for self-calibration and sky model prediction. After any phaseshift
operation, both columns must be updated:

    sync_reference_dir_with_phase_dir(ms_path)

Symptom if missing: MODEL_DATA is predicted at the wrong sky position; self-calibration
diverges; sky model seeding is applied at an incorrect offset.

### 3. TELESCOPE_NAME must be DSA_110 before each WSClean run

The SPW-merge step (merge_spws(), required before IDG imaging) resets
OBSERVATION::TELESCOPE_NAME to OVRO_MMA for CASA compatibility. EveryBeam requires
DSA_110 to load the correct beam model. The name must be patched back:

    set_ms_telescope_name(ms_path, name="DSA_110")

This is called automatically inside run_wsclean() but must be retained if the
imaging workflow is modified.

Symptom if missing: EveryBeam silently selects the wrong beam model, producing
incorrect primary beam correction and photometric errors up to ~20% near the field edge.

## Output artifacts

Save figures, images, preview PNGs, analysis CSVs, and other derived artifacts under
`/data/dsa110-continuum/outputs/` (organized by topic or date). Do not leave user-facing
artifacts in `/tmp`.

## Instrument quick-reference

DSA-110 is a meridian drift-scan transit array (doesn't track — sky drifts through fixed beam):
- 117 antennas, 4.65 m dishes, L-band (1.31–1.50 GHz, 187.5 MHz bandwidth)
- 16 subbands × 48 channels, 12.885 s integrations
- Each "tile" ≈ 5-minute transit; imaging produces 4800×4800 px at 3 arcsec/px

## Next tasks

- Multi-epoch production runs (batch_pipeline.py on 2026-02-12 through 2026-03-05)
- Per-date gain calibration (eliminate cross-date phase cal transfer)
- Run source finding (BANE + Aegean) on mosaics
- Generate multi-epoch light curves and variability analysis
