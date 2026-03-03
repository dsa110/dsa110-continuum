# Claude Code Instructions — dsa110-continuum

## What this is

A clean continuum imaging pipeline for DSA-110 (Caltech), ported from dsa110-contimg.
Only the verified science code lives here. No web infrastructure.

## Verified working state

run_pipeline.py produces a calibrated image of 3C454.3 at 12.5 Jy/beam.
Test data: 2026-01-25 HDF5 files at /data/incoming/ on H17.

## Package structure

dsa110_continuum/
  conversion/   - HDF5 to MS (UVH5 subband grouping, phase centre assignment, UVW reconstruction)
  calibration/  - bandpass, gain cal, applycal, phaseshift, self-cal
  imaging/      - WSClean/CASA tclean interface, ImagingParams, sky model seeding
  mosaic/       - mosaicking (QUICKLOOK image-domain and SCIENCE/DEEP visibility-domain tiers)
  photometry/   - forced photometry, ESE detection, variability metrics (Mooley eta/Vs/m)
  qa/           - delay validation, quality checks

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

docs/skills/ contains verified implementation notes for each pipeline step.
Read these before writing any new code for that step.

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

## Next tasks

- Extend scripts/run_pipeline.py to mosaic a full day of drift observations
- Run source finding (BANE + Aegean) on the mosaic
- Run forced photometry against NVSS/RACS catalog positions
