# Claude Code Instructions — dsa110-continuum
What this is
A clean continuum imaging pipeline for DSA-110 (Caltech), ported from dsa110-contimg.
Only the verified science code lives here. No web infrastructure.
Verified working state
run_pipeline.py produces a calibrated image of 3C454.3 at 12.5 Jy/beam.
Test data: 2026-01-25 HDF5 files at /data/incoming/ on H17.
Package structure
dsa110_continuum/
  conversion/   - HDF5 to MS
  calibration/  - bandpass, gain cal, applycal, phaseshift
  imaging/      - WSClean/CASA tclean interface
  mosaic/       - mosaicking (to be built)
  photometry/   - source finding + forced photometry (to be built)
  qa/           - delay validation, quality checks
Key paths (H17)
/data/incoming/                              raw HDF5 files
/stage/dsa110-contimg/ms/                   Measurement Sets
/opt/miniforge/envs/casa6                   CASA conda env
Pipeline DB
dsa110 convert queries the pipeline SQLite DB, not the filesystem.
New dates must be indexed first: dsa110 index add --start YYYY-MM-DD --end YYYY-MM-DD --directory /data/incoming
Reference docs
docs/skills/ contains verified implementation notes for each pipeline step.
Read these before writing any new code for that step.
Next tasks
- Extend scripts/run_pipeline.py to mosaic a full day of drift observations
- Run source finding (BANE + Aegean) on the mosaic
- Run forced photometry against NVSS/RACS catalog positions
