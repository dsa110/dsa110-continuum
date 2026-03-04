# DSA-110 Light Curve Pipeline — Design Document

**Date:** 2026-03-04  
**Goal:** Produce per-source flux light curves with variability metrics from existing forced photometry data, and fix the pipeline to propagate source IDs for future epochs.

---

## Science Goal

Detect and monitor variable and possibly transient compact radio sources at daily cadence over weeks/months, in search of Extreme Scattering Events (ESEs) and other flux-variable phenomena. The instrument is the DSA-110 meridian drift-scan array at OVRO; observations are L-band (1.31–1.50 GHz).

---

## Current State

Six dates have been processed through the full pipeline (HDF5 → MS → calibration → imaging → mosaicking → forced photometry):

| Date       | Tile FITS | Epoch mosaics | Forced phot CSVs |
|------------|-----------|---------------|------------------|
| 2026-01-25 | 180       | 2 epochs      | 2 CSVs           |
| 2026-02-12 | 11        | 2 epochs      | 3 CSVs           |
| 2026-02-15 | 6         | 2 epochs      | 2 CSVs           |
| 2026-02-23 | 8         | 1 epoch       | 1 CSV            |
| 2026-02-25 | 11        | 1 epoch       | 1 CSV            |
| 2026-02-26 | 10        | 1 epoch       | 1 CSV            |

**Total: 10 epoch-level forced photometry CSVs** across 6 dates.

**Critical gap:** The forced phot CSV schema (`ra_deg, dec_deg, nvss_flux_jy, dsa_peak_jyb, dsa_peak_err_jyb, dsa_nvss_ratio`) has no `source_id`. Cross-epoch stacking requires matching by RA/Dec to NVSS positions.

---

## Design: Option C — Parallel Build + Fix

Four independent tasks dispatched in parallel:

### Task 1 — Light Curve Stack (`scripts/stack_lightcurves.py`)

Read all 10 forced phot CSVs. Match each row to a unique NVSS source by RA/Dec proximity (5 arcsec tolerance). Assign a stable `source_id` (NVSS catalog row index). Produce a stacked Parquet at `products/lightcurves/lightcurves.parquet`.

**Output schema:**
```
source_id (int), ra_deg (float64), dec_deg (float64), nvss_flux_jy (float64),
epoch_utc (str ISO8601), dsa_peak_jyb (float64), dsa_peak_err_jyb (float64),
dsa_nvss_ratio (float64), mosaic_path (str), date (str)
```

### Task 2 — Variability Metrics (`scripts/variability_metrics.py`)

Read `products/lightcurves/lightcurves.parquet`. For each source with ≥ 2 epochs, compute:
- **m** = σ_S / ⟨S⟩  (Mooley modulation index)
- **Vs** = (S_max − S_min) / sqrt(σ_max² + σ_min²)  (variability significance)
- **η** = reduced χ² against constant-flux null hypothesis

Flag candidates where Vs > 4.0 OR η > 2.5. Output `products/lightcurves/variability_metrics.parquet`.

**Output schema:**
```
source_id (int), ra_deg, dec_deg, nvss_flux_jy,
n_epochs (int), mean_flux (float64), std_flux (float64),
m (float64), Vs (float64), eta (float64),
is_variable_candidate (bool)
```

### Task 3 — Visualization (`scripts/plot_lightcurves.py`)

Read metrics Parquet. Produce:
- Per-source flux-vs-time PNG plots saved to `products/lightcurves/plots/{source_id}.png`
- `products/lightcurves/variable_candidates_summary.html` ranking top-N most variable sources by η

### Task 4 — Pipeline Fix (`scripts/batch_pipeline.py`, `dsa110_continuum/photometry/forced.py`)

1. Add `source_id` column to forced phot output (NVSS row index via RA/Dec match at measurement time)
2. Fix mosaic FITS archiving: copy epoch mosaic FITS from stage to `products/mosaics/{date}/` alongside CSVs
3. Extract hardcoded stage paths into environment variables with documented defaults

---

## File Interface Contract (for parallel agents)

All agents share this interface — Agents 2 and 3 write code that reads from paths Agents 1/2 produce, even if those files do not exist at write time.

| File | Producer | Consumer |
|------|----------|----------|
| `products/lightcurves/lightcurves.parquet` | Agent 1 | Agent 2, Agent 3 |
| `products/lightcurves/variability_metrics.parquet` | Agent 2 | Agent 3 |
| `products/lightcurves/plots/{source_id}.png` | Agent 3 | — |
| `products/lightcurves/variable_candidates_summary.html` | Agent 3 | — |

---

## What Is NOT in Scope

- Re-running the imaging pipeline on new dates (multi-hour compute job)
- Robust Huber flux-scale correction (design documented in pipeline-specs.md §9.3; deferred)
- Source finding / blind catalog (no new mosaic needed for existing data; deferred)
- GPU calibration, spectral index maps

---

## Success Criteria

1. `products/lightcurves/lightcurves.parquet` exists with ≥ 500 source-epoch rows
2. `products/lightcurves/variability_metrics.parquet` exists with m/Vs/η per source
3. `products/lightcurves/variable_candidates_summary.html` renders and shows ranked candidates
4. `batch_pipeline.py` forced phot output includes `source_id` for future runs
