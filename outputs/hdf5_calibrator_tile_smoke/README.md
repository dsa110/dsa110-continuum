# HDF5 Calibrator Tile Smoke Evidence

This directory is the evidence root for issue #38: generating a single
bandpass-calibrator tile image from raw HDF5 inputs with current pipeline code.

The workflow is intentionally split into two phases:

1. Discovery/preflight, cloud-safe:
   - validates the configured VLA calibrator database;
   - uses the pipeline SQLite HDF5 index as the authoritative discovery source;
   - searches indexed dates only;
   - ranks primary calibrators before bright VLA fallback calibrators;
   - rejects candidates before conversion when DB, index, HDF5 path, subband,
     integration, beam, or transit-window requirements are not met.
2. Pinned H17/CASA smoke execution, H17-only:
   - generates a fresh MS from the pinned HDF5 group;
   - duplicates the MS into solve and image copies;
   - phase-shifts the solve copy to the VLA catalog RA/Dec;
   - phase-shifts the image copy to the production meridian tile center;
   - generates fresh evidence-scoped BP/G tables;
   - applies calibration and runs WSClean.

Existing MS, calibration tables, FITS products, and prior run logs are not valid
inputs to this evidence workflow.

## Discovery Command

```bash
PYTHONPATH=/workspace python3 scripts/hdf5_calibrator_tile_smoke.py \
  --date 2026-01-25 \
  --output-dir /data/dsa110-continuum/outputs/hdf5_calibrator_tile_smoke/<run_id>/discovery
```

Optional path overrides:

```bash
PYTHONPATH=/workspace python3 scripts/hdf5_calibrator_tile_smoke.py \
  --pipeline-db /data/dsa110-contimg/state/db/pipeline.sqlite3 \
  --vla-calibrator-db /data/dsa110-contimg/state/catalogs/vla_calibrators.sqlite3 \
  --date 2026-01-25 \
  --output-dir /data/dsa110-continuum/outputs/hdf5_calibrator_tile_smoke/<run_id>/discovery
```

Outputs:

- `discovery.json`
- `candidate_matrix.csv`

Exit codes:

- `0`: at least one candidate passed and was selected.
- `2`: discovery ran, but no candidate passed the strict preflight.

## Current Scope

The current implementation covers the discovery/preflight phase only. The H17
execution phase still needs the production conversion/calibration/imaging wiring
described in issue #38.
