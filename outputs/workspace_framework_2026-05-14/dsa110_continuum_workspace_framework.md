# DSA-110 Continuum Imaging Pipeline: Workspace Framework

Generated: 2026-05-14

## Executive Summary

The DSA-110 continuum imaging pipeline is a production-grade radio astronomy data processing system for the Deep Synoptic Array at Owens Valley Radio Observatory. It processes raw HDF5 visibility data through six sequential stages to produce science-ready light curves for detecting variable and transient compact radio sources, especially Extreme Scattering Events (ESEs).

The codebase shows strong domain specialization, robust orchestration, clean acyclic architecture, excellent documentation, and science-safe engineering practices.

Overall code health: 7.2/10.

- Readability: 8/10
- Maintainability: 6/10
- Testability: 7/10
- Complexity: 6/10
- Documentation: 8/10

## 1. Code Structure

### Core Package Architecture

Primary package: `dsa110_continuum/`

Pipeline stage modules:

1. `conversion/` - HDF5 to Measurement Set conversion via PyUVData, UVW reconstruction, SPW merging.
2. `calibration/` - flagging, bandpass/gain solve, applycal, self-cal, phaseshift.
3. `imaging/` - WSClean/CASA tclean interface, `ImagingParams`, sky model seeding.
4. `mosaic/` - QUICKLOOK and SCIENCE/DEEP mosaicking tiers.
5. `photometry/` - forced photometry, ESE detection, variability metrics.
6. `catalog/` - NVSS/RACS/FIRST/VLA calibrator catalog management, SQLite backend.

Support modules:

- `qa/` - delay validation, image quality, pipeline QA hooks.
- `simulation/` - synthetic UVH5 generation, variability models.
- `visualization/` - FITS plots and calibration diagnostics.
- `validation/` - package health diagnostics.
- `evaluation/` - pipeline stage evaluation harness.
- `rfi/` - GPU-accelerated RFI detection.
- `pipeline/` - per-epoch orchestration layer.
- `lightcurves/` - light curve stacking and variability metrics.
- `search/` - fast folding search.
- `pointing/` - calibrator transit prediction.
- `spectral/` - spectral analysis utilities.
- `selfcal/` - self-calibration workflows.

Key entry points:

- `scripts/batch_pipeline.py` - production full orchestration.
- `scripts/run_pipeline.py` - single-tile reference run.
- `scripts/mosaic_day.py` - legacy day-batch path.
- `scripts/forced_photometry.py` - standalone forced photometry.
- `scripts/source_finding.py` - BANE + Aegean blind source catalog.
- Diagnostic and validation scripts include `validate_date.py`, `verify_sources.py`, `inspect_fits.py`, `plot_lightcurves.py`, `stack_lightcurves.py`, and `variability_metrics.py`.

Architecture pattern: layered pipeline stage modules, support modules, adapter layer, orchestration layer, and compatibility layer.

## 2. Data Flow

The end-to-end pipeline:

```text
HDF5 (16 subbands x N timestamps)
  -> conversion/
Measurement Set (MS)
  -> calibration/
Calibrated MS (CORRECTED_DATA)
  -> imaging/
FITS tile images (4800 x 4800 px, 3 arcsec/px)
  -> mosaic/
Hourly-epoch mosaic FITS (~12 tiles, ~1 hour)
  -> qa/
QA-validated mosaic
  -> photometry/
Forced photometry CSV and variability metrics
  -> lightcurves/
Light curve products and Mooley eta/Vs/m metrics
```

### Stage 1: Conversion

- Groups 16 subband HDF5 files by observation timestamp.
- Loads subbands via PyUVData.
- Reconstructs UVW from antenna positions.
- Writes Measurement Sets.
- Merges SPWs when needed for IDG imaging.
- Handles TELESCOPE_NAME dual identity: CASA-compatible `OVRO_MMA` during some operations, then `DSA_110` before WSClean/EveryBeam.

### Stage 2: Calibration

- Runs two-stage RFI flagging.
- Solves bandpass and gain tables from calibrator transits.
- Supports same-date tables, generated tables, bright-source fallback, and validated borrowed tables.
- Performs phaseshift to median meridian.
- Applies BP/G tables and populates `CORRECTED_DATA`.
- Enforces phase-centre and telescope-name invariants.

### Stage 3: Imaging

- Runs WSClean with wgridder or IDG.
- Supports sky model seeding with predict-then-deconvolve workflow.
- Applies primary-beam correction through EveryBeam.
- Produces tile FITS images.

### Stage 4: Mosaicking

- Batch production mode bins tiles by UTC hour with two-tile overlap on each side.
- Sliding target mode uses 12-tile windows with stride 6.
- QUICKLOOK uses image-domain reprojection and coadd.
- SCIENCE/DEEP use visibility-domain joint deconvolution.

### Stage 5: QA

Three-gate epoch QA:

1. Ratio gate: median DSA/catalog flux ratio near expected range.
2. Completeness gate: catalog source recovery fraction.
3. RMS gate: mosaic noise threshold.

Strict QA skips forced photometry for QA-fail epochs by default.

### Stage 6: Photometry

- Runs Condon matched-filter forced photometry.
- Applies differential photometry normalization.
- Computes ESE candidates and variability metrics.
- Produces light curves and per-source metrics.

## 3. Backend Architecture

### Main Patterns

- Sequential modular pipeline with file-based communication: MS, FITS, CSV, SQLite, JSON.
- Job/pipeline abstractions compatible with Dagster.
- Dataclass-based configuration, including calibration presets and imaging parameters.
- Adapter layer around CASA and Measurement Set table access.
- Process isolation for CASA tasks through `CASAService`.

### External Integrations

1. CASA 6 - calibration and CASA task execution.
2. WSClean - imaging and visibility-domain mosaicking.
3. AOFlagger - RFI flagging.
4. EveryBeam - primary beam modeling.
5. PyUVData - UVH5 and MS conversion.
6. Aegean - source finding and forced-fit integration.
7. Dagster - workflow orchestration components.

### Dependency Categories

- Scientific: PyUVData, NumPy, Astropy, pyradiosky, pyuvsim, matvis, SciPy.
- CASA: casatools, casatasks.
- Web/API: FastAPI, uvicorn, Strawberry GraphQL, Pydantic, HTTPX, slowapi.
- Database: SQLAlchemy, Alembic, aiosqlite, asyncpg, PyArrow.
- Orchestration: Dagster packages.
- Distributed: dask and distributed, pinned below 2024.11.0 for dask-ms compatibility.
- Async/cache: Redis, aiohttp, APScheduler.
- Visualization: matplotlib, scienceplots, bokeh.
- Utilities and security: tenacity, packaging, prometheus-client, bcrypt, google-auth, MCP.

## 4. Scientific Domain Framework

### Instrument Constraints

- DSA-110 is a meridian drift-scan transit array.
- The array does not track; sky drifts through fixed beams.
- 96 active antennas are used from a larger set of allocated array elements.
- Data are split into 16 subbands with 48 channels each.
- Total science bandwidth is 187.5 MHz across L-band.
- Integration time is 12.885 seconds.
- A tile is a roughly five-minute transit image.

### Calibration Science

- Bandpass cycle is operationally one bandpass-calibrator transit per sidereal day.
- Calibration acquisition order:
  1. Valid same-date tables.
  2. Generate from primary flux calibrator transit.
  3. Generate from bright-source VLA catalog fallback.
  4. Borrow nearest validated strip-compatible tables.
  5. Fail loudly.
- Absolute flux anchor is either Perley-Butler primary model or VLA catalog fallback.
- Provenance sidecars record calibrator selection, flux anchor, source date, and related metadata.
- Default preset includes `field="0~23"`, `refant="103"`, `prebp_phase=True`, and `bp_minsnr=5.0`.

### Imaging Science

- Tile geometry: 4800 x 4800 pixels, 3 arcsec/pixel.
- WSClean defaults include auto-mask, auto-threshold, and major-cycle gain choices validated for the science case.
- Sky model seeding reduces major cycles by predicting catalog sources into `MODEL_DATA`.
- Primary beam correction depends on `TELESCOPE_NAME = DSA_110` for EveryBeam.
- IDG imaging requires SPW merge.

### Mosaicking Science

- Hourly-epoch mosaic: coadd of about 12 sequential tiles along a Dec strip.
- Batch mode uses UTC hour bins with overlap.
- Sliding mode target uses a fixed 12-tile window and stride 6.
- QUICKLOOK is image-domain and fast.
- SCIENCE/DEEP are visibility-domain and scientifically stronger for wide fields.
- RA wrap must use circular mean with `arctan2(mean(sin(RA)), mean(cos(RA)))`.

### Photometry Science

- Forced photometry uses a Condon matched-filter approach with a PSF kernel.
- Differential normalization selects stable reference sources.
- ESE detection primarily uses `sigma_deviation >= 5.0`.
- Variability metrics follow Mooley/VAST conventions:
  - eta: weighted variability statistic.
  - Vs: two-epoch significance statistic.
  - m: modulation index.

## 5. Product Perspective

Primary science goal: detect and monitor variable/transient compact radio sources through daily-cadenced forced photometry on hourly-epoch mosaics.

Target users:

- DSA-110/OVRO radio astronomers.
- Researchers studying ESEs, AGN variability, radio transients, and scattering diagnostics.

Value proposition:

- End-to-end automation from raw HDF5 to science-ready light curves.
- Daily cadence with hourly-epoch products.
- Built-in three-gate QA.
- Reproducible checkpoint/resume and provenance tracking.
- Parallel tile processing and photometry.

Operational constraints:

- Production work expects the CASA environment at `/opt/miniforge/envs/casa6/bin/python`.
- H17 has real telescope data paths and CASA/WSClean/EveryBeam.
- Cloud environments may lack CASA and stage data, so tests use mocks/shims.
- Full-day processing is compute- and storage-heavy.

## 6. Dependency Hotspots

Overall coupling score: 6/10.

Architecture has zero circular dependencies; dependency flow is acyclic. Main risk is critical hubs with high fan-in.

Critical hubs:

| Module | Fan-in | Fan-out | Risk |
|---|---:|---:|---|
| `calibration/casa_service.py` | 18 | 0 | Single adapter point for CASA tasks |
| `calibration/runner.py` | 21 | 6 | Phase-centre logic used broadly |
| `calibration/model.py` | 14 | 14 | Sky model logic couples calibration and imaging |
| `imaging/cli_imaging.py` | 6 | 12 | Main WSClean interface with broad dependencies |
| `calibration/applycal.py` | 3 | 3 | Calibration application hub |

Change impact:

- Changing `casa_service.py` signatures can affect 18 files.
- Changing `runner.py::phaseshift_ms()` can affect 8+ workflows and 21 direct importers.
- Changing `model.py` sky model logic can affect calibration, imaging, and self-cal flows.
- Changing `cli_imaging.py::image_ms()` affects core imaging, self-cal, SPW imaging, and smoke tests.

Recommended dependency work:

1. Add strong integration tests around `casa_service.py`.
2. Extract field-direction utilities from `runner.py`.
3. Decouple sky model seeding from `calibration/model.py`.
4. Add dependency injection for CASA service access.
5. Create a calibration facade only if it reduces fan-in without hiding science-critical invariants.

## 7. Code Health

Overall: 7.2/10.

Dimension scores:

- Readability: 8/10.
- Maintainability: 6/10.
- Testability: 7/10.
- Complexity: 6/10.
- Documentation: 8/10.

### Strengths

- Clear stage-level organization.
- Strong domain naming.
- Excellent reference documentation in `CONTEXT.md`, `CLAUDE.md`, and `docs/reference/`.
- Broad tests and simulation support.
- Explicit science-safety invariants documented and implemented.
- Adapter abstractions make CASA and MS table behavior more testable than direct imports.

### Main Health Hotspots

1. `dsa110_continuum/calibration/calibration.py` is a monolithic module around 2600 lines.
2. `dsa110_continuum/calibration/flagging.py` is large and mixes preflight, RFI flagging, bad-pol detection, and recovery logic.
3. `dsa110_continuum/photometry/ese_detection.py` has duplicate imports.
4. `dsa110_continuum/conversion/conversion_orchestrator.py` has defensive import stubs that obscure import failure behavior.
5. `dsa110_continuum/imaging/cli_imaging.py` has deferred imports and fallback complexity.
6. `dsa110_continuum/calibration/runner.py` contains fragile FIELD direction shape normalization.
7. Optional GPU/Numba acceleration paths need clearer contracts.
8. `dsa110_continuum/mosaic/builder.py` contains instrument-specific primary beam assumptions.
9. `dsa110_continuum/calibration/ensure.py` would benefit from an explicit validation contract for borrowed tables.

Recommended health work:

1. Split `calibration/calibration.py` into focused solve modules.
2. Extract flagging strategies from `flagging.py`.
3. Remove duplicate imports in `photometry/ese_detection.py`.
4. Centralize CASA log redirection.
5. Consolidate FIELD direction shape normalization in a tested utility.
6. Formalize GPU/Numba fallback contracts.
7. Add tests for all FIELD direction column shapes.
8. Add mock-GPU integration tests for fallback behavior.
9. Address broad exception handlers incrementally, not as a bulk style-only change.

## 8. Critical Invariants and Pitfalls

The following invariants can fail silently and produce wrong science without exceptions.

### 1. `FIELD::PHASE_DIR` after `chgcentre`

WSClean's `chgcentre` may not update `FIELD::PHASE_DIR`. After phaseshift, code must call the helper that updates `PHASE_DIR` to the target position.

Symptom if missing: CASA computes phase gradients relative to the old field centre; sources become smeared or offset.

### 2. `FIELD::REFERENCE_DIR` sync

CASA `ft()` reads `REFERENCE_DIR` for model visibilities. After phaseshift, `REFERENCE_DIR` must match `PHASE_DIR`.

Symptom if missing: `MODEL_DATA` is predicted at the wrong sky position; self-cal can diverge.

### 3. `TELESCOPE_NAME = DSA_110` before WSClean

`merge_spws()` can reset `OBSERVATION::TELESCOPE_NAME` to `OVRO_MMA` for CASA compatibility. EveryBeam requires `DSA_110`.

Symptom if missing: EveryBeam selects the wrong beam model; primary beam errors can reach about 20 percent near field edge.

Known issues:

- `mosaic/wsclean_mosaic.py` was reported to use arithmetic mean for RA in at least one path; RA wrap requires circular mean.
- Legacy compatibility with `dsa110_contimg` remains present by design.
- Cloud environments do not fully match H17 production data/tooling.

## 9. Testing and Validation

Test infrastructure:

- Dozens of test files and hundreds of tests.
- Test categories include unit, integration, contract, benchmark, and slow.
- Synthetic data support through simulation utilities.
- Mocking patterns exist for CASA tables, WSClean subprocesses, and catalogs.

Preferred test command on H17:

```bash
/opt/miniforge/envs/casa6/bin/python -m pytest tests/ -q
```

Important validation gates:

- Preflight: Dec-strip check, calibration table existence, strip compatibility.
- Per tile: MS validity, `CORRECTED_DATA` sanity, FITS validity.
- Per epoch: flux ratio, completeness, RMS.
- Photometry: skip QA-fail epochs unless using diagnostic lenient mode.

## 10. Documentation and Knowledge Artifacts

Core docs:

- `CONTEXT.md` - domain glossary with code citations.
- `CLAUDE.md` - project guidelines, environment, operational constraints, silent-failure guards.
- `AGENTS.md` - agent workspace defaults and learned facts.
- `README.md` - high-level overview.

Reference docs:

- `docs/reference/calibration.md`
- `docs/reference/conversion-and-qa.md`
- `docs/reference/flagging.md`
- `docs/reference/imaging.md`
- `docs/reference/mosaicking.md`
- `docs/reference/photometry-and-ese.md`
- `docs/reference/vast-crossref.md`

Skills docs:

- `docs/skills/` contains subsystem-specific operational notes.

## 11. Working Guidelines

When working in this repository:

1. Use `/opt/miniforge/envs/casa6/bin/python` for pipeline work on H17.
2. Read `docs/reference/` before touching pipeline subsystems.
3. Preserve PHASE_DIR, REFERENCE_DIR, and TELESCOPE_NAME invariants.
4. Use `dsa110_continuum.*` imports for new code.
5. Do not casually rewrite legacy compatibility `__init__.py` layers.
6. Do not bulk-fix pre-existing ruff violations during unrelated edits.
7. Use domain vocabulary from `CONTEXT.md`: tile, hourly-epoch mosaic, Dec strip.
8. Treat `scripts/qa_server.py` and `scripts/monitor_server.py` as live services.
9. Save derived artifacts under `/data/dsa110-continuum/outputs/`.
10. Add regression tests for behavior changes.

## 12. Technical Debt Priorities

High priority:

- Split `calibration/calibration.py`.
- Extract flagging strategy modules.
- Strengthen tests around `casa_service.py` and `runner.py`.
- Add tests for FIELD direction column shapes.

Medium priority:

- Centralize CASA log redirection.
- Formalize optional acceleration contracts.
- Add validation contracts for calibration table borrowing.
- Decouple sky model seeding from high-coupling modules.

Low priority:

- Consider plugin architecture for CASA tasks.
- Extract photometry into a more service-like boundary if coupling grows.
- Add a dedicated `ARCHITECTURE.md` for dependency and module boundaries.

## Final Summary

This workspace contains a production-grade, domain-specialized radio astronomy imaging pipeline with:

- 18 main modules.
- 35 entry point scripts.
- Six-stage data flow from HDF5 to science light curves.
- Seven major external integrations.
- Thirty-plus key dependencies.
- Twenty-plus domain-specific scientific patterns.
- Three critical silent-failure guards.
- Broad test coverage and strong documentation.
- Clean acyclic dependency flow with moderate coupling.
- Code health around 7.2/10, with maintainability and complexity as the main improvement areas.

The most important mental model is:

```text
conversion creates correct MS structure
calibration creates trustworthy CORRECTED_DATA
imaging turns calibrated MS into PB-corrected tiles
mosaic builds hourly-epoch products
QA gates science usability
photometry extracts per-source light curves and variability metrics
```
