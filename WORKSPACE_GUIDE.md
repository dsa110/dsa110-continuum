I'll explore this workspace to build a comprehensive understanding of its structure and architecture. Let me check if there's an existing graphify output first, then use the explore skill.

None

None

No memory MCP server available. Proceeding to exploration agents.

Launched 4 parallel exploration agents. Checking their progress...

None

None

None

None

None

None

None

None

Based on the exploration completed so far, let me synthesize the product perspective:

## Product Perspective: DSA-110 Continuum Imaging Pipeline

### Scientific Mission
**Primary Goal**: Detect and monitor variable/transient compact radio sources (Extreme Scattering Events - ESEs) via daily-cadenced per-source forced photometry on hourly-epoch mosaics.

**Target Users**: Radio astronomers at Caltech/OVRO studying:
- ESEs (plasma lensing events)
- AGN variability
- Transient radio sources
- Scattering diagnostics

### Value Proposition
1. **Automated Pipeline**: End-to-end processing from raw HDF5 to science-ready light curves
2. **Daily Cadence**: Hourly-epoch mosaics enable high-time-resolution variability studies
3. **Quality Assurance**: Three-gate QA (ratio, completeness, RMS) ensures reliable flux measurements
4. **Reproducibility**: Checkpoint/resume, provenance tracking, validated calibration parameters
5. **Scalability**: Parallel tile processing, batch orchestration for multi-day campaigns

### User Journey
```
1. Data Acquisition
   - HDF5 files arrive at /data/incoming/
   - 16 subbands × N timestamps per observation

2. Pipeline Execution
   - User runs: python scripts/batch_pipeline.py --date 2026-01-25
   - Pipeline auto-discovers MS, generates/borrows calibration tables
   - Processes tiles → mosaics → forced photometry

3. Quality Review
   - User reviews run_report.md
   - Checks QA verdicts per epoch
   - Investigates DEGRADED epochs if any

4. Science Analysis
   - User accesses light curves at /data/dsa110-proc/products/lightcurves/
   - Applies variability metrics (η, Vs, m) for ESE detection
   - Plots light curves via scripts/plot_lightcurves.py
```

### Key Differentiators
1. **Drift-Scan Optimization**: Specialized for DSA-110's meridian transit geometry (not generic interferometry)
2. **Silent-Failure Protection**: Three critical invariants explicitly enforced (PHASE_DIR, REFERENCE_DIR, TELESCOPE_NAME)
3. **Calibration Robustness**: Auto-generation, borrowing, provenance tracking, strip-compatibility validation
4. **Domain-Specific Science**: Condon matched-filter photometry, Mooley variability metrics, ESE detection algorithms

### Operational Constraints
- **H17 Environment**: Requires CASA 6, WSClean, EveryBeam, telescope data paths
- **Cloud Limitations**: No CASA environment; uses compatibility shim for testing
- **Compute Requirements**: 8-12 hours for 24 epochs (GPU acceleration preferred for IDG)
- **Storage**: ~GB/day for MS + FITS + products

# DSA-110 Continuum Imaging Pipeline: Comprehensive Workspace Framework

## Executive Summary

The DSA-110 continuum imaging pipeline is a **production-grade radio astronomy data processing system** for the Deep Synoptic Array (110 antennas) at Owens Valley Radio Observatory. It processes raw HDF5 visibility data through 6 sequential stages to produce science-ready light curves for detecting variable and transient compact radio sources (ESEs). The codebase demonstrates **strong domain specialization**, **robust orchestration**, and **science-safe engineering practices**.

---

## 1. Code Structure Analysis

### Core Package Architecture (18 modules)

**Pipeline Stage Modules (6 primary)**:
1. **conversion/** (94 exports) - HDF5 → MS via PyUVData, UVW reconstruction, SPW merging
2. **calibration/** (352 exports) - Flagging, bandpass/gain solve, applycal, self-cal, phaseshift
3. **imaging/** (80 exports) - WSClean/CASA tclean interface, ImagingParams, sky model seeding
4. **mosaic/** (178 exports) - QUICKLOOK + SCIENCE/DEEP mosaicking tiers
5. **photometry/** (117 exports) - Forced photometry, ESE detection, variability metrics
6. **catalog/** (70 exports) - NVSS/RACS/FIRST/VLA calibrators, SQLite backend

**Support Modules (12 secondary)**:
- qa/ (104 exports) - Delay validation, image quality, pipeline QA hooks
- simulation/ (178 exports) - Synthetic UVH5 generation, variability models
- visualization/ (483 exports) - FITS plots, calibration diagnostics
- validation/ (8 exports) - Package health diagnostics
- evaluation/ (131 exports) - Pipeline stage evaluation harness
- rfi/ (36 exports) - GPU-accelerated RFI detection
- pipeline/ (28 exports) - Per-epoch orchestration layer
- lightcurves/ (45 exports) - Light curve stacking, variability metrics
- search/ (5 exports) - Fast folding search
- pointing/ (46 exports) - Calibrator transit prediction
- spectral/ (1 export) - Spectral analysis utilities
- selfcal/ (1 export) - Self-calibration workflows

**Entry Points (35 scripts)**:
- `batch_pipeline.py` - Full orchestration (production)
- `run_pipeline.py` - Single-tile reference
- `mosaic_day.py` - Legacy day-batch
- `forced_photometry.py` - Standalone photometry
- `source_finding.py` - BANE + Aegean blind source catalog
- Plus 30 diagnostic, validation, and utility scripts

**Architecture Pattern**: Pipeline stage modules with layered architecture (stage → support → adapter → orchestration → compatibility layers)

---

## 2. Data Flow Analysis

### 6-Stage Pipeline

```
HDF5 (16 subbands × N timestamps)
  ↓ [conversion/]
Measurement Set (MS)
  ↓ [calibration/]
Calibrated MS (CORRECTED_DATA)
  ↓ [imaging/]
FITS tile images (4800×4800 px, 3 arcsec/px)
  ↓ [mosaic/]
Hourly-epoch mosaic FITS (~12 tiles, ~1 hour)
  ↓ [qa/]
QA-validated mosaic (three-gate QA)
  ↓ [photometry/]
Forced photometry CSV → variability metrics
  ↓ [lightcurves/]
Light curve Parquet + Mooley η/Vs/m metrics
```

### Critical Data Transformations

**Stage 1 (Conversion)**:
- 16 subbands loaded in parallel via PyUVData
- UVW reconstruction from antenna positions
- SPW merge for IDG compatibility
- TELESCOPE_NAME dual-identity handling

**Stage 2 (Calibration)**:
- Two-stage AOFlagger RFI flagging
- Bandpass/gain solve on calibrator transit
- Calibration table borrowing (±30 days, bidirectional)
- Phaseshift to median meridian
- Three silent-failure invariants (PHASE_DIR, REFERENCE_DIR, TELESCOPE_NAME)

**Stage 3 (Imaging)**:
- WSClean with wgridder (default) or IDG (GPU)
- Sky model seeding (two-step predict → deconvolve)
- Primary beam correction via EveryBeam
- Hardcoded parameters (auto-mask 5σ, mgain 0.8)

**Stage 4 (Mosaicking)**:
- Batch mode: UTC hour bins with ±2 tile overlap
- Sliding mode: 12-tile window, stride 6 (50% overlap)
- QUICKLOOK: image-domain, <1 h, nearest-neighbor regridding
- SCIENCE/DEEP: visibility-domain joint deconvolution

**Stage 5 (QA)**:
- Ratio gate: Median DSA/catalog flux ratio 0.8–1.2
- Completeness gate: Catalog source recovery > 80%
- RMS gate: Mosaic noise < 2 mJy/beam

**Stage 6 (Photometry)**:
- Condon matched-filter forced photometry (Numba-accelerated)
- Differential photometry normalization
- ESE detection (sigma_deviation ≥ 5.0)
- Mooley variability metrics (η, Vs, m)

---

## 3. Backend Architecture Analysis

### Architecture Patterns

**1. Sequential Modular Pipeline**:
- 5 core stages with file-based communication (MS, FITS, SQLite)
- Minimal cross-stage coupling
- Each stage independently testable

**2. Job/Pipeline Abstraction (Dagster-compatible)**:
- `CalibrationSolveJob`, `CalibrationApplyJob`, `CalibrationValidateJob`
- `MosaicPlanningJob`, `MosaicBuildJob`, `MosaicQAJob`
- Decouples job logic from orchestration framework

**3. Configuration Presets**:
- Single `DEFAULT_PRESET` optimized for DSA-110
- Encapsulates K/BP/G parameters, refant selection, model source
- Supports runtime customization via `.with_overrides()`

**4. Adapter Layer (Anti-Corruption)**:
- `adapters/casa.py`: CASA 6 tool isolation
- `adapters/casa_tables.py`: Drop-in replacement for casacore.tables
- `adapters/ms_write.py`: PyUVData → MS writer abstraction

**5. Service Isolation**:
- `calibration/casa_service.py`: Process-isolated CASA execution
- Prevents CASA state pollution across pipeline stages

### External Integrations (7)

1. **CASA 6** (casatools, casatasks ≥6.7.3) - Calibration & imaging
2. **WSClean** (subprocess) - Imaging & mosaicking
3. **AOFlagger** (Docker/CLI) - RFI flagging (2-stage)
4. **EveryBeam** (Docker) - Primary beam modeling
5. **PyUVData** (3.25) - UVH5 ↔ MS conversion
6. **Aegean** (subprocess) - Source finding & forced photometry
7. **Dagster** (≥1.12.10) - Workflow orchestration

### Key Dependencies (32)

**Scientific** (7): pyuvdata, numpy, astropy, pyradiosky, pyuvsim, matvis, scipy
**CASA** (2): casatools, casatasks
**Web/API** (7): fastapi, uvicorn, strawberry-graphql, pydantic, pydantic-settings, httpx, slowapi
**Database** (5): sqlalchemy, alembic, aiosqlite, asyncpg, pyarrow
**Orchestration** (3): dagster, dagster-webserver, dagster-components
**Distributed** (2): dask, distributed (pinned <2024.11.0 for dask-ms compatibility)
**Async/Cache** (3): redis, aiohttp, apscheduler
**Visualization** (3): matplotlib, scienceplots, bokeh
**Security** (3): bcrypt, google-auth, google-auth-oauthlib
**Utilities** (5): tenacity, packaging, antlr4-python3-runtime, prometheus-client, mcp

---

## 4. Scientific Domain Analysis

### Instrument-Specific Constraints (26 patterns identified)

**Drift-Scan Transit Array**:
- 96 active antennas (T-shaped array: 47 E-W + 35 N-S + 14 outriggers)
- 16 subbands × 48 channels = 768 total channels (1.31–1.50 GHz, 187.5 MHz)
- 12.885 s integration time, ~24 samples per 5-minute tile
- Meridian drift-scan: sky drifts through fixed beams, no tracking

**Calibration Science**:
- Bandpass cycle: One per sidereal day (operational, not code-enforced)
- Calibration table borrowing: ±30 days bidirectional search
- Flux anchor: Perley-Butler 2017 model or VLA catalog fallback
- Selection pool: Primary vs bright_fallback calibrators
- Provenance sidecar: JSON metadata for BP tables
- DEFAULT_PRESET: field="0~23", refant="103", bp_minsnr=5.0

**Imaging Science**:
- Tile geometry: 4800×4800 px, 3 arcsec/px, ~5-minute transit
- WSClean hardcoded: auto-mask 5σ, auto-threshold 1.0σ, mgain 0.8
- Sky model seeding: Two-step predict → deconvolve workflow
- Three phase-centre invariants (silent-failure guards)
- Primary beam correction: Airy disk model, 4.65 m dish diameter
- IDG SPW-merge workflow: Required for image-domain gridder

**Mosaicking Strategies**:
- Three-tier architecture: QUICKLOOK (fast), SCIENCE (quality), DEEP (best)
- Hourly-epoch mosaic: ~12 tiles with overlap
- RA-wrap circular mean: arctan2(sin, cos) for 0°/360° boundary
- QUICKLOOK: Image-domain, nearest-neighbor regridding, PB weighting
- SCIENCE/DEEP: Visibility-domain joint deconvolution with direction-dependent beam correction

**Photometry Science**:
- Forced photometry: Condon matched-filter with PSF kernel
- Differential photometry normalization: Reference source selection
- ESE detection: Sigma_deviation ≥ 5.0 (CONSERVATIVE, not empirically calibrated)
- Variability metrics: Mooley η, Vs, m from VAST Tools

---

## 5. Product Perspective

### Scientific Mission
**Primary Goal**: Detect and monitor variable/transient compact radio sources (ESEs) via daily-cadenced per-source forced photometry on hourly-epoch mosaics.

**Target Users**: Radio astronomers at Caltech/OVRO studying ESEs, AGN variability, transient radio sources, and scattering diagnostics.

### Value Proposition
1. **Automated Pipeline**: End-to-end processing from raw HDF5 to science-ready light curves
2. **Daily Cadence**: Hourly-epoch mosaics enable high-time-resolution variability studies
3. **Quality Assurance**: Three-gate QA ensures reliable flux measurements
4. **Reproducibility**: Checkpoint/resume, provenance tracking, validated parameters
5. **Scalability**: Parallel tile processing, batch orchestration for multi-day campaigns

### Key Differentiators
1. **Drift-Scan Optimization**: Specialized for DSA-110's meridian transit geometry
2. **Silent-Failure Protection**: Three critical invariants explicitly enforced
3. **Calibration Robustness**: Auto-generation, borrowing, provenance tracking
4. **Domain-Specific Science**: Condon photometry, Mooley metrics, ESE detection

### Operational Constraints
- **H17 Environment**: Requires CASA 6, WSClean, EveryBeam, telescope data paths
- **Cloud Limitations**: No CASA environment; uses compatibility shim for testing
- **Compute Requirements**: 8-12 hours for 24 epochs (GPU acceleration preferred)
- **Storage**: ~GB/day for MS + FITS + products

---

## 6. Critical Invariants & Pitfalls

### Three Silent-Failure Guards (CLAUDE.md)

1. **`FIELD::PHASE_DIR` after `chgcentre`**
   - WSClean's `chgcentre` may not update `FIELD::PHASE_DIR`
   - Missing fix: CASA computes phase gradients vs old field centre → smeared/offset sources

2. **`FIELD::REFERENCE_DIR` sync with `PHASE_DIR`**
   - CASA's `ft()` reads `REFERENCE_DIR` for model visibilities
   - Missing fix: MODEL_DATA predicted at wrong sky position; self-cal diverges

3. **`TELESCOPE_NAME = DSA_110` before WSClean**
   - `merge_spws()` resets to `OVRO_MMA` for CASA compatibility
   - Missing fix: EveryBeam selects wrong beam model → primary beam errors up to ~20%

### Known Issues
- **RA-wrap bug in wsclean_mosaic.py**: Uses arithmetic mean instead of circular mean for tiles crossing 0°/360°
- **Legacy import migration**: 370 imports migrated from `dsa110_contimg` to `dsa110_continuum`, but compatibility shim still needed
- **Cloud environment**: CASA unavailable; uses shim at `~/.local/lib/python3.12/site-packages/dsa110_contimg_shim.py`

---

## 7. Testing & Validation

### Test Infrastructure
- **62 test files** with 220+ tests
- **Markers**: slow, benchmark, contract, integration, unit
- **Categories**: Unit (fast), Integration (medium), Contract (slow, real data)
- **Fixtures**: SimulationHarness (synthetic UVH5 generation), mock CASA tables, temporary SQLite

### Test Execution
```bash
# Full suite (220 tests, ~20s)
/opt/miniforge/envs/casa6/bin/python -m pytest tests/ -q

# Exclude slow tests
pytest tests/ -m "not slow" -q

# Parallel execution (4 workers)
pytest tests/ -n 4 -q
```

### Validation Gates
- **Preflight**: Dec-strip check, cal table existence, strip compatibility
- **Per-tile**: MS validity, CORRECTED_DATA ratio > 5.0, FITS validity
- **Per-epoch**: Three-gate QA (ratio, completeness, RMS)
- **Photometry**: QA-FAIL skip (default-strict) or lenient override

---

## 8. Documentation & Knowledge Artifacts

### Core Documentation
- **CONTEXT.md**: Domain glossary with path::Symbol citations
- **CLAUDE.md**: Project guidelines, CASA environment, calibration order, silent-failure guards
- **AGENTS.md**: Agent workspace defaults
- **README.md**: Overview of 11 core submodules + data flow

### Reference Documentation
- **docs/reference/calibration.md**: K/B/G parameters, DEFAULT_PRESET, SelfCalConfig
- **docs/reference/imaging.md**: WSClean flags, sky model seeding, IDG SPW-merge
- **docs/reference/mosaicking.md**: QUICKLOOK vs SCIENCE/DEEP, RA-wrap handling
- **docs/reference/photometry-and-ese.md**: Condon matched-filter, ESE detection, variability metrics

### Skills Documentation
- **docs/skills/**: Verified implementation notes for each subsystem
- **docs/plans/**: Future work (lightcurve pipeline, variability roadmap, live-monitor integration)

---

## 9. Code Health & Dependency Analysis (In Progress)

**Status**: Code health assessment and dependency hotspot analysis agents are still running. Initial assessment based on exploration:

**Strengths**:
- Strong domain documentation (CONTEXT.md, reference docs)
- Modular architecture with clear stage boundaries
- Comprehensive test coverage (220+ tests)
- Science-safe engineering (silent-failure guards, validated parameters)

**Areas for Attention**:
- Large module sizes (calibration/ 352 exports, visualization/ 483 exports)
- Legacy compatibility layer (dsa110_contimg shim)
- Known bug in wsclean_mosaic.py (RA-wrap arithmetic mean)
- Pre-existing ruff violations (~900, tracked separately)

---

## 10. Key Takeaways

### What Makes This Codebase Special
1. **Domain Specialization**: Deeply optimized for DSA-110 drift-scan transit array geometry
2. **Science Safety**: Explicit silent-failure guards protect against invisible science errors
3. **Calibration Robustness**: Multi-strategy calibration (auto-generate, borrow, validate)
4. **Production-Grade**: Checkpoint/resume, quarantine policies, provenance tracking

### How to Work With This Codebase
1. **Always use CASA environment**: `/opt/miniforge/envs/casa6/bin/python`
2. **Read reference docs before touching pipeline logic**: `docs/reference/*.md`
3. **Preserve silent-failure guards**: Never bypass PHASE_DIR, REFERENCE_DIR, TELESCOPE_NAME checks
4. **Use existing test patterns**: SimulationHarness for synthetic data, contract tests for real data
5. **Follow domain vocabulary**: Use terms verbatim from CONTEXT.md (tile, hourly-epoch mosaic, Dec strip)

### Critical Paths to Understand
1. **Calibration table acquisition**: ensure.py → borrowing → provenance validation
2. **Phaseshift workflow**: runner.py → chgcentre → PHASE_DIR sync → REFERENCE_DIR sync
3. **WSClean invocation**: cli_imaging.py → TELESCOPE_NAME reset → sky model seeding
4. **Mosaicking QA**: epoch_qa.py → three-gate validation → photometry gating
5. **Batch orchestration**: batch_pipeline.py → checkpoint → quarantine → resume

---

## Summary

**RESULT: Comprehensive framework established for DSA-110 continuum imaging pipeline**

The workspace contains a **production-grade radio astronomy imaging pipeline** with:
- **18 main modules** organized as sequential pipeline stages (conversion → calibration → imaging → mosaic → photometry → lightcurves)
- **35 entry point scripts** for orchestration, diagnostics, and validation
- **6-stage data flow** from HDF5 to science-ready light curves
- **7 external integrations** (CASA, WSClean, AOFlagger, EveryBeam, PyUVData, Aegean, Dagster)
- **32 key dependencies** across scientific, web, database, orchestration, and utility categories
- **26 domain-specific patterns** for drift-scan transit array, calibration science, imaging, mosaicking, and photometry
- **Three silent-failure guards** protecting against invisible science errors
- **220+ tests** with unit, integration, and contract categories
- **Comprehensive documentation** (CONTEXT.md, CLAUDE.md, reference docs, skills)

**Code health and dependency hotspot analysis are still in progress.** This framework provides a solid foundation for understanding the codebase architecture, data flow, scientific constraints, and operational patterns.

**Exploration complete.** I've built a comprehensive local framework for understanding the DSA-110 continuum imaging workspace. The analysis covers:

- **Code structure**: 18 modules (6 pipeline stages + 12 support), 35 entry scripts
- **Data flow**: 6-stage pipeline from HDF5 to light curves with critical transformations
- **Backend architecture**: 7 external integrations (CASA, WSClean, AOFlagger, etc.), 32 key dependencies
- **Scientific domain**: 26 domain-specific patterns for drift-scan array, calibration, imaging, mosaicking, photometry
- **Product perspective**: Scientific mission, value proposition, operational constraints
- **Critical invariants**: Three silent-failure guards protecting against invisible science errors
- **Testing**: 220+ tests with unit/integration/contract categories
- **Documentation**: CONTEXT.md glossary, reference docs, skills documentation

Key insight: This is a **domain-specialized, production-grade radio astronomy pipeline** with strong science-safe engineering practices. The codebase is optimized for DSA-110's drift-scan transit geometry and includes explicit protections against common invisible failure modes in interferometric imaging.

The framework is saved in this conversation for future reference. Code health and dependency hotspot analyses are still running in the background.

Excellent! The dependency hotspot analysis is complete. Let me update the comprehensive framework with these findings.

# DSA-110 Continuum Imaging Pipeline: Comprehensive Workspace Framework (Updated)

## Executive Summary

The DSA-110 continuum imaging pipeline is a **production-grade radio astronomy data processing system** for the Deep Synoptic Array (110 antennas) at Owens Valley Radio Observatory. It processes raw HDF5 visibility data through 6 sequential stages to produce science-ready light curves for detecting variable and transient compact radio sources (ESEs). The codebase demonstrates **strong domain specialization**, **robust orchestration**, **clean acyclic architecture** (zero circular dependencies), and **science-safe engineering practices**.

---

## 1. Code Structure Analysis

### Core Package Architecture (18 modules)

**Pipeline Stage Modules (6 primary)**:
1. **conversion/** (94 exports) - HDF5 → MS via PyUVData, UVW reconstruction, SPW merging
2. **calibration/** (352 exports) - Flagging, bandpass/gain solve, applycal, self-cal, phaseshift
3. **imaging/** (80 exports) - WSClean/CASA tclean interface, ImagingParams, sky model seeding
4. **mosaic/** (178 exports) - QUICKLOOK + SCIENCE/DEEP mosaicking tiers
5. **photometry/** (117 exports) - Forced photometry, ESE detection, variability metrics
6. **catalog/** (70 exports) - NVSS/RACS/FIRST/VLA calibrators, SQLite backend

**Support Modules (12 secondary)**:
- qa/ (104 exports) - Delay validation, image quality, pipeline QA hooks
- simulation/ (178 exports) - Synthetic UVH5 generation, variability models
- visualization/ (483 exports) - FITS plots, calibration diagnostics
- validation/ (8 exports) - Package health diagnostics
- evaluation/ (131 exports) - Pipeline stage evaluation harness
- rfi/ (36 exports) - GPU-accelerated RFI detection
- pipeline/ (28 exports) - Per-epoch orchestration layer
- lightcurves/ (45 exports) - Light curve stacking, variability metrics
- search/ (5 exports) - Fast folding search
- pointing/ (46 exports) - Calibrator transit prediction
- spectral/ (1 export) - Spectral analysis utilities
- selfcal/ (1 export) - Self-calibration workflows

**Entry Points (35 scripts)**:
- `batch_pipeline.py` - Full orchestration (production)
- `run_pipeline.py` - Single-tile reference
- `mosaic_day.py` - Legacy day-batch
- `forced_photometry.py` - Standalone photometry
- `source_finding.py` - BANE + Aegean blind source catalog
- Plus 30 diagnostic, validation, and utility scripts

**Architecture Pattern**: Pipeline stage modules with layered architecture (stage → support → adapter → orchestration → compatibility layers)

---

## 2. Data Flow Analysis

### 6-Stage Pipeline

```
HDF5 (16 subbands × N timestamps)
  ↓ [conversion/]
Measurement Set (MS)
  ↓ [calibration/]
Calibrated MS (CORRECTED_DATA)
  ↓ [imaging/]
FITS tile images (4800×4800 px, 3 arcsec/px)
  ↓ [mosaic/]
Hourly-epoch mosaic FITS (~12 tiles, ~1 hour)
  ↓ [qa/]
QA-validated mosaic (three-gate QA)
  ↓ [photometry/]
Forced photometry CSV → variability metrics
  ↓ [lightcurves/]
Light curve Parquet + Mooley η/Vs/m metrics
```

### Critical Data Transformations

**Stage 1 (Conversion)**:
- 16 subbands loaded in parallel via PyUVData
- UVW reconstruction from antenna positions
- SPW merge for IDG compatibility
- TELESCOPE_NAME dual-identity handling

**Stage 2 (Calibration)**:
- Two-stage AOFlagger RFI flagging
- Bandpass/gain solve on calibrator transit
- Calibration table borrowing (±30 days, bidirectional)
- Phaseshift to median meridian
- Three silent-failure invariants (PHASE_DIR, REFERENCE_DIR, TELESCOPE_NAME)

**Stage 3 (Imaging)**:
- WSClean with wgridder (default) or IDG (GPU)
- Sky model seeding (two-step predict → deconvolve)
- Primary beam correction via EveryBeam
- Hardcoded parameters (auto-mask 5σ, mgain 0.8)

**Stage 4 (Mosaicking)**:
- Batch mode: UTC hour bins with ±2 tile overlap
- Sliding mode: 12-tile window, stride 6 (50% overlap)
- QUICKLOOK: image-domain, <1 h, nearest-neighbor regridding
- SCIENCE/DEEP: visibility-domain joint deconvolution

**Stage 5 (QA)**:
- Ratio gate: Median DSA/catalog flux ratio 0.8–1.2
- Completeness gate: Catalog source recovery > 80%
- RMS gate: Mosaic noise < 2 mJy/beam

**Stage 6 (Photometry)**:
- Condon matched-filter forced photometry (Numba-accelerated)
- Differential photometry normalization
- ESE detection (sigma_deviation ≥ 5.0)
- Mooley variability metrics (η, Vs, m)

---

## 3. Backend Architecture Analysis

### Architecture Patterns

**1. Sequential Modular Pipeline**:
- 5 core stages with file-based communication (MS, FITS, SQLite)
- Minimal cross-stage coupling
- Each stage independently testable

**2. Job/Pipeline Abstraction (Dagster-compatible)**:
- `CalibrationSolveJob`, `CalibrationApplyJob`, `CalibrationValidateJob`
- `MosaicPlanningJob`, `MosaicBuildJob`, `MosaicQAJob`
- Decouples job logic from orchestration framework

**3. Configuration Presets**:
- Single `DEFAULT_PRESET` optimized for DSA-110
- Encapsulates K/BP/G parameters, refant selection, model source
- Supports runtime customization via `.with_overrides()`

**4. Adapter Layer (Anti-Corruption)**:
- `adapters/casa.py`: CASA 6 tool isolation
- `adapters/casa_tables.py`: Drop-in replacement for casacore.tables
- `adapters/ms_write.py`: PyUVData → MS writer abstraction

**5. Service Isolation**:
- `calibration/casa_service.py`: Process-isolated CASA execution
- Prevents CASA state pollution across pipeline stages

### External Integrations (7)

1. **CASA 6** (casatools, casatasks ≥6.7.3) - Calibration & imaging
2. **WSClean** (subprocess) - Imaging & mosaicking
3. **AOFlagger** (Docker/CLI) - RFI flagging (2-stage)
4. **EveryBeam** (Docker) - Primary beam modeling
5. **PyUVData** (3.25) - UVH5 ↔ MS conversion
6. **Aegean** (subprocess) - Source finding & forced photometry
7. **Dagster** (≥1.12.10) - Workflow orchestration

### Key Dependencies (32)

**Scientific** (7): pyuvdata, numpy, astropy, pyradiosky, pyuvsim, matvis, scipy
**CASA** (2): casatools, casatasks
**Web/API** (7): fastapi, uvicorn, strawberry-graphql, pydantic, pydantic-settings, httpx, slowapi
**Database** (5): sqlalchemy, alembic, aiosqlite, asyncpg, pyarrow
**Orchestration** (3): dagster, dagster-webserver, dagster-components
**Distributed** (2): dask, distributed (pinned <2024.11.0 for dask-ms compatibility)
**Async/Cache** (3): redis, aiohttp, apscheduler
**Visualization** (3): matplotlib, scienceplots, bokeh
**Security** (3): bcrypt, google-auth, google-auth-oauthlib
**Utilities** (5): tenacity, packaging, antlr4-python3-runtime, prometheus-client, mcp

---

## 4. Scientific Domain Analysis

### Instrument-Specific Constraints (26 patterns identified)

**Drift-Scan Transit Array**:
- 96 active antennas (T-shaped array: 47 E-W + 35 N-S + 14 outriggers)
- 16 subbands × 48 channels = 768 total channels (1.31–1.50 GHz, 187.5 MHz)
- 12.885 s integration time, ~24 samples per 5-minute tile
- Meridian drift-scan: sky drifts through fixed beams, no tracking

**Calibration Science**:
- Bandpass cycle: One per sidereal day (operational, not code-enforced)
- Calibration table borrowing: ±30 days bidirectional search
- Flux anchor: Perley-Butler 2017 model or VLA catalog fallback
- Selection pool: Primary vs bright_fallback calibrators
- Provenance sidecar: JSON metadata for BP tables
- DEFAULT_PRESET: field="0~23", refant="103", bp_minsnr=5.0

**Imaging Science**:
- Tile geometry: 4800×4800 px, 3 arcsec/px, ~5-minute transit
- WSClean hardcoded: auto-mask 5σ, auto-threshold 1.0σ, mgain 0.8
- Sky model seeding: Two-step predict → deconvolve workflow
- Three phase-centre invariants (silent-failure guards)
- Primary beam correction: Airy disk model, 4.65 m dish diameter
- IDG SPW-merge workflow: Required for image-domain gridder

**Mosaicking Strategies**:
- Three-tier architecture: QUICKLOOK (fast), SCIENCE (quality), DEEP (best)
- Hourly-epoch mosaic: ~12 tiles with overlap
- RA-wrap circular mean: arctan2(sin, cos) for 0°/360° boundary
- QUICKLOOK: Image-domain, nearest-neighbor regridding, PB weighting
- SCIENCE/DEEP: Visibility-domain joint deconvolution with direction-dependent beam correction

**Photometry Science**:
- Forced photometry: Condon matched-filter with PSF kernel
- Differential photometry normalization: Reference source selection
- ESE detection: Sigma_deviation ≥ 5.0 (CONSERVATIVE, not empirically calibrated)
- Variability metrics: Mooley η, Vs, m from VAST Tools

---

## 5. Product Perspective

### Scientific Mission
**Primary Goal**: Detect and monitor variable/transient compact radio sources (ESEs) via daily-cadenced per-source forced photometry on hourly-epoch mosaics.

**Target Users**: Radio astronomers at Caltech/OVRO studying ESEs, AGN variability, transient radio sources, and scattering diagnostics.

### Value Proposition
1. **Automated Pipeline**: End-to-end processing from raw HDF5 to science-ready light curves
2. **Daily Cadence**: Hourly-epoch mosaics enable high-time-resolution variability studies
3. **Quality Assurance**: Three-gate QA ensures reliable flux measurements
4. **Reproducibility**: Checkpoint/resume, provenance tracking, validated parameters
5. **Scalability**: Parallel tile processing, batch orchestration for multi-day campaigns

### Key Differentiators
1. **Drift-Scan Optimization**: Specialized for DSA-110's meridian transit geometry
2. **Silent-Failure Protection**: Three critical invariants explicitly enforced
3. **Calibration Robustness**: Auto-generation, borrowing, provenance tracking
4. **Domain-Specific Science**: Condon photometry, Mooley metrics, ESE detection

### Operational Constraints
- **H17 Environment**: Requires CASA 6, WSClean, EveryBeam, telescope data paths
- **Cloud Limitations**: No CASA environment; uses compatibility shim for testing
- **Compute Requirements**: 8-12 hours for 24 epochs (GPU acceleration preferred)
- **Storage**: ~GB/day for MS + FITS + products

---

## 6. Dependency Hotspot Analysis

### Coupling Score: 6/10 (Moderate)

**Key Metrics**:
- **Total incoming dependencies**: 21 (calibration/runner.py)
- **Total outgoing dependencies**: 12 (imaging/cli_imaging.py)
- **Circular dependencies**: 0 (clean acyclic architecture)
- **High fan-in nodes (>10)**: 3 critical hubs
- **High fan-out nodes (>8)**: 3 critical hubs

### Critical Dependency Hotspots

| Module | Fan-In | Fan-Out | Criticality | Risk |
|--------|--------|---------|------------|------|
| **calibration/casa_service.py** | 18 | 0 | CRITICAL | Single point of failure for all CASA operations |
| **calibration/runner.py** | 21 | 6 | CRITICAL | Phase center logic used by 21 files |
| **calibration/model.py** | 14 | 14 | CRITICAL | Sky model population with bidirectional coupling |
| **imaging/cli_imaging.py** | 6 | 12 | HIGH | WSClean interface with high fan-out |
| **calibration/applycal.py** | 3 | 3 | HIGH | Calibration application hub |

### Change Impact Analysis

**Blast Radius for Critical Modules**:
- **casa_service.py**: 18 files would break on method signature change
- **runner.py**: 21 files would break on phaseshift_ms() change
- **model.py**: 14 files would break on sky model logic change
- **cli_imaging.py**: 6 files would break on image_ms() change

### Strengths
1. **Zero circular dependencies** - Clean acyclic architecture
2. **Deferred imports** - imaging/cli_imaging.py uses function-level imports to reduce startup coupling
3. **Data layer isolation** - catalog/ module is read-only with minimal outgoing dependencies
4. **Clear separation of concerns** - calibration, imaging, mosaic, photometry are distinct layers

### Actionable Recommendations

**Immediate (High Priority)**:
1. Add integration tests for casa_service.py changes (18 dependent files)
2. Extract field_utils.py from runner.py to reduce fan-in from 21 to ~15
3. Decouple model.py from imaging by moving sky model seeding to separate module

**Medium Priority**:
4. Add dependency injection for casa_service.py (replace singleton pattern)
5. Create calibration_facade.py to consolidate runner/applycal/flagging (reduce runner.py fan-in to ~5)
6. Add validation gates between modules (calibration → imaging → photometry)

**Long-term (Low Priority)**:
7. Migrate to plugin architecture for CASA tasks (reduce coupling from 18 to ~3 files)
8. Extract photometry as a separate service (reduce imaging-photometry coupling)

---

## 7. Critical Invariants & Pitfalls

### Three Silent-Failure Guards (CLAUDE.md)

1. **`FIELD::PHASE_DIR` after `chgcentre`**
   - WSClean's `chgcentre` may not update `FIELD::PHASE_DIR`
   - Missing fix: CASA computes phase gradients vs old field centre → smeared/offset sources

2. **`FIELD::REFERENCE_DIR` sync with `PHASE_DIR`**
   - CASA's `ft()` reads `REFERENCE_DIR` for model visibilities
   - Missing fix: MODEL_DATA predicted at wrong sky position; self-cal diverges

3. **`TELESCOPE_NAME = DSA_110` before WSClean**
   - `merge_spws()` resets to `OVRO_MMA` for CASA compatibility
   - Missing fix: EveryBeam selects wrong beam model → primary beam errors up to ~20%

### Known Issues
- **RA-wrap bug in wsclean_mosaic.py**: Uses arithmetic mean instead of circular mean for tiles crossing 0°/360°
- **Legacy import migration**: 370 imports migrated from `dsa110_contimg` to `dsa110_continuum`, but compatibility shim still needed
- **Cloud environment**: CASA unavailable; uses shim at `~/.local/lib/python3.12/site-packages/dsa110_contimg_shim.py`

---

## 8. Testing & Validation

### Test Infrastructure
- **62 test files** with 220+ tests
- **Markers**: slow, benchmark, contract, integration, unit
- **Categories**: Unit (fast), Integration (medium), Contract (slow, real data)
- **Fixtures**: SimulationHarness (synthetic UVH5 generation), mock CASA tables, temporary SQLite

### Test Execution
```bash
# Full suite (220 tests, ~20s)
/opt/miniforge/envs/casa6/bin/python -m pytest tests/ -q

# Exclude slow tests
pytest tests/ -m "not slow" -q

# Parallel execution (4 workers)
pytest tests/ -n 4 -q
```

### Validation Gates
- **Preflight**: Dec-strip check, cal table existence, strip compatibility
- **Per-tile**: MS validity, CORRECTED_DATA ratio > 5.0, FITS validity
- **Per-epoch**: Three-gate QA (ratio, completeness, RMS)
- **Photometry**: QA-FAIL skip (default-strict) or lenient override

---

## 9. Documentation & Knowledge Artifacts

### Core Documentation
- **CONTEXT.md**: Domain glossary with path::Symbol citations
- **CLAUDE.md**: Project guidelines, CASA environment, calibration order, silent-failure guards
- **AGENTS.md**: Agent workspace defaults
- **README.md**: Overview of 11 core submodules + data flow

### Reference Documentation
- **docs/reference/calibration.md**: K/B/G parameters, DEFAULT_PRESET, SelfCalConfig
- **docs/reference/imaging.md**: WSClean flags, sky model seeding, IDG SPW-merge
- **docs/reference/mosaicking.md**: QUICKLOOK vs SCIENCE/DEEP, RA-wrap handling
- **docs/reference/photometry-and-ese.md**: Condon matched-filter, ESE detection, variability metrics

### Skills Documentation
- **docs/skills/**: Verified implementation notes for each subsystem
- **docs/plans/**: Future work (lightcurve pipeline, variability roadmap, live-monitor integration)

---

## 10. Key Takeaways

### What Makes This Codebase Special
1. **Domain Specialization**: Deeply optimized for DSA-110 drift-scan transit array geometry
2. **Science Safety**: Explicit silent-failure guards protect against invisible science errors
3. **Calibration Robustness**: Multi-strategy calibration (auto-generate, borrow, validate)
4. **Production-Grade**: Checkpoint/resume, quarantine policies, provenance tracking
5. **Clean Architecture**: Zero circular dependencies, acyclic module flow

### How to Work With This Codebase
1. **Always use CASA environment**: `/opt/miniforge/envs/casa6/bin/python`
2. **Read reference docs before touching pipeline logic**: `docs/reference/*.md`
3. **Preserve silent-failure guards**: Never bypass PHASE_DIR, REFERENCE_DIR, TELESCOPE_NAME checks
4. **Use existing test patterns**: SimulationHarness for synthetic data, contract tests for real data
5. **Follow domain vocabulary**: Use terms verbatim from CONTEXT.md (tile, hourly-epoch mosaic, Dec strip)
6. **Be cautious with critical hubs**: casa_service.py (18 deps), runner.py (21 deps) require careful testing

### Critical Paths to Understand
1. **Calibration table acquisition**: ensure.py → borrowing → provenance validation
2. **Phaseshift workflow**: runner.py → chgcentre → PHASE_DIR sync → REFERENCE_DIR sync
3. **WSClean invocation**: cli_imaging.py → TELESCOPE_NAME reset → sky model seeding
4. **Mosaicking QA**: epoch_qa.py → three-gate validation → photometry gating
5. **Batch orchestration**: batch_pipeline.py → checkpoint → quarantine → resume

### Dependency Management Strategy
1. **Test casa_service.py changes comprehensively**: 18 dependent files require integration testing
2. **Extract utilities from runner.py**: Reduce fan-in by extracting field_utils.py and meridian_utils.py
3. **Add validation gates between stages**: Prevent cascading failures from critical hub changes
4. **Consider dependency injection**: Replace casa_service.py singleton with DI pattern
5. **Plan for plugin architecture**: Long-term goal to reduce CASA coupling from 18 to ~3 files

---

## Summary

**RESULT: Comprehensive framework established for DSA-110 continuum imaging pipeline**

The workspace contains a **production-grade radio astronomy imaging pipeline** with:
- **18 main modules** organized as sequential pipeline stages (conversion → calibration → imaging → mosaic → photometry → lightcurves)
- **35 entry point scripts** for orchestration, diagnostics, and validation
- **6-stage data flow** from HDF5 to science-ready light curves
- **7 external integrations** (CASA, WSClean, AOFlagger, EveryBeam, PyUVData, Aegean, Dagster)
- **32 key dependencies** across scientific, web, database, orchestration, and utility categories
- **26 domain-specific patterns** for drift-scan transit array, calibration science, imaging, mosaicking, and photometry
- **Three silent-failure guards** protecting against invisible science errors
- **220+ tests** with unit, integration, and contract categories
- **Comprehensive documentation** (CONTEXT.md, CLAUDE.md, reference docs, skills)
- **Clean acyclic architecture** with zero circular dependencies (coupling score 6/10)
- **Critical dependency hubs**: casa_service.py (18 fan-in), runner.py (21 fan-in), model.py (14 bidirectional)

**Code health assessment is still in progress.** This framework provides a complete foundation for understanding the codebase architecture, data flow, scientific constraints, dependency structure, operational patterns, and change impact risks.

Perfect! The code health assessment is complete. Let me update the comprehensive framework with the final findings.

# DSA-110 Continuum Imaging Pipeline: Comprehensive Workspace Framework (Final)

## Executive Summary

The DSA-110 continuum imaging pipeline is a **production-grade radio astronomy data processing system** for the Deep Synoptic Array (110 antennas) at Owens Valley Radio Observatory. It processes raw HDF5 visibility data through 6 sequential stages to produce science-ready light curves for detecting variable and transient compact radio sources (ESEs). The codebase demonstrates **strong domain specialization**, **robust orchestration**, **clean acyclic architecture** (zero circular dependencies), **excellent documentation**, and **science-safe engineering practices**.

**Overall Code Health: 7.2/10** — Documentation & Readability strong (8/10), Maintainability & Complexity need work (6/10 each).

---

## 1. Code Structure Analysis

### Core Package Architecture (18 modules)

**Pipeline Stage Modules (6 primary)**:
1. **conversion/** (94 exports) - HDF5 → MS via PyUVData, UVW reconstruction, SPW merging
2. **calibration/** (352 exports) - Flagging, bandpass/gain solve, applycal, self-cal, phaseshift
3. **imaging/** (80 exports) - WSClean/CASA tclean interface, ImagingParams, sky model seeding
4. **mosaic/** (178 exports) - QUICKLOOK + SCIENCE/DEEP mosaicking tiers
5. **photometry/** (117 exports) - Forced photometry, ESE detection, variability metrics
6. **catalog/** (70 exports) - NVSS/RACS/FIRST/VLA calibrators, SQLite backend

**Support Modules (12 secondary)**:
- qa/ (104 exports), simulation/ (178 exports), visualization/ (483 exports), validation/ (8 exports), evaluation/ (131 exports), rfi/ (36 exports), pipeline/ (28 exports), lightcurves/ (45 exports), search/ (5 exports), pointing/ (46 exports), spectral/ (1 export), selfcal/ (1 export)

**Entry Points (35 scripts)**:
- `batch_pipeline.py` (full orchestration), `run_pipeline.py` (single-tile reference), `mosaic_day.py` (legacy day-batch), `forced_photometry.py`, `source_finding.py`, plus 30 diagnostic/validation/utility scripts

**Architecture Pattern**: Pipeline stage modules with layered architecture (stage → support → adapter → orchestration → compatibility layers)

---

## 2. Data Flow Analysis

### 6-Stage Pipeline

```
HDF5 (16 subbands × N timestamps)
  ↓ [conversion/]
Measurement Set (MS)
  ↓ [calibration/]
Calibrated MS (CORRECTED_DATA)
  ↓ [imaging/]
FITS tile images (4800×4800 px, 3 arcsec/px)
  ↓ [mosaic/]
Hourly-epoch mosaic FITS (~12 tiles, ~1 hour)
  ↓ [qa/]
QA-validated mosaic (three-gate QA)
  ↓ [photometry/]
Forced photometry CSV → variability metrics
  ↓ [lightcurves/]
Light curve Parquet + Mooley η/Vs/m metrics
```

---

## 3. Backend Architecture Analysis

### Architecture Patterns
- Sequential modular pipeline with file-based communication
- Job/Pipeline abstraction (Dagster-compatible)
- Configuration presets (DEFAULT_PRESET)
- Adapter layer (anti-corruption for CASA)
- Service isolation (CASAService subprocess wrapper)

### External Integrations (7)
CASA 6, WSClean, AOFlagger, EveryBeam, PyUVData, Aegean, Dagster

### Key Dependencies (32)
Scientific (7), CASA (2), Web/API (7), Database (5), Orchestration (3), Distributed (2), Async/Cache (3), Visualization (3), Security (3), Utilities (5)

---

## 4. Scientific Domain Analysis

### 26 Domain-Specific Patterns Identified

**Drift-Scan Transit Array**: 96 active antennas, 16 subbands (187.5 MHz), 12.885 s integration, meridian drift-scan

**Calibration Science**: Bandpass cycle per sidereal day, table borrowing (±30 days), flux anchoring, provenance tracking

**Imaging Science**: Tile geometry (4800×4800 px, 3 arcsec/px), WSClean parameters, sky model seeding, phase-centre invariants

**Mosaicking Strategies**: Three-tier architecture, hourly-epoch mosaics, RA-wrap circular mean, QUICKLOOK vs SCIENCE/DEEP

**Photometry Science**: Condon matched-filter, differential normalization, ESE detection (σ≥5.0), Mooley metrics (η, Vs, m)

---

## 5. Product Perspective

### Scientific Mission
Detect and monitor variable/transient compact radio sources (ESEs) via daily-cadenced per-source forced photometry on hourly-epoch mosaics.

### Value Proposition
Automated pipeline, daily cadence, three-gate QA, reproducibility (checkpoint/resume), scalability (parallel processing)

### Key Differentiators
Drift-scan optimization, silent-failure protection, calibration robustness, domain-specific science

---

## 6. Dependency Hotspot Analysis

### Coupling Score: 6/10 (Moderate)

**Critical Hubs**:
- **calibration/casa_service.py** (18 fan-in, 0 fan-out) - CRITICAL: Single CASA adapter failure point
- **calibration/runner.py** (21 fan-in, 6 fan-out) - CRITICAL: Phase center logic used by 21 files
- **calibration/model.py** (14 fan-in, 14 fan-out) - CRITICAL: Sky model with bidirectional coupling
- **imaging/cli_imaging.py** (6 fan-in, 12 fan-out) - HIGH: WSClean interface with high fan-out

**Change Impact**:
- casa_service.py: 18 files break on method signature change
- runner.py: 21 files break on phaseshift_ms() change
- model.py: 14 files break on sky model logic change

**Strengths**: Zero circular dependencies, deferred imports reduce startup coupling, data layer isolation

**Recommendations**: Extract utilities from runner.py, add DI for casa_service.py, create calibration_facade.py, add validation gates

---

## 7. Code Health Assessment

### Overall Score: 7.2/10

| Dimension | Score | Status |
|-----------|-------|--------|
| **Readability** | 8/10 | ✓ Strong |
| **Maintainability** | 6/10 | ⚠ Needs Work |
| **Testability** | 7/10 | ✓ Good |
| **Complexity** | 6/10 | ⚠ Moderate-High |
| **Documentation** | 8/10 | ✓ Excellent |

### Key Findings

**Strengths**:
- Excellent naming conventions and comprehensive docstrings
- Clear module organization at stage level
- Extensive test coverage (1060 test functions, 70+ test files)
- Comprehensive reference docs with validated parameters
- Domain glossary (CONTEXT.md) and silent-failure invariants (CLAUDE.md)

**Weaknesses**:
- **Monolithic calibration/calibration.py (2600 lines, 24 functions)** - Violates SRP
- **High coupling in imaging/** - 14 cross-module imports
- **Fragile array shape handling** - 3 FIELD column formats with duplicated logic
- **Optional dependencies create silent paths** - GPU/Numba with no explicit contracts
- **Overly-broad exception handling** - 426 bare-except handlers in calibration/

### Hotspots (10 identified)
1. calibration/calibration.py:1-2587 - 2600-line monolithic module
2. calibration/flagging.py:1-1500+ - 1500+ lines mixing concerns
3. photometry/ese_detection.py:16-23 - Duplicate imports
4. conversion/conchestrator.py:36-105 - Defensive import stubs
5. imaging/cli_imaging.py:1-100+ - Deferred imports complexity
6. calibration/runner.py:35-68 - Fragile array shape normalization
7. calibration/applycal.py:70-80 - GPU availability without clear contract
8. photometry/forced.py:31-93 - Numba/GPU with 4 fallback paths
9. mosaic/builder.py:71-100+ - Hardcoded primary beam model
10. calibration/ensure.py - Calibration borrowing lacks validation contract

### Recommendations (14 action items)
- REFACTOR: Split calibration.py into solve_*.py modules
- REFACTOR: Extract flagging strategies into separate modules
- FIX: Remove duplicate imports
- IMPROVE: Replace defensive stubs with explicit ImportError handling
- IMPROVE: Centralize CASA log redirection
- IMPROVE: Consolidate array shape normalization
- IMPROVE: Formalize GPU availability contract
- IMPROVE: Extract Numba/GPU initialization logic
- DOCUMENT: Add instrument-specific configuration
- IMPROVE: Add validation contract for calibration borrowing
- TEST: Add parametrized tests for FIELD column shapes
- TEST: Add integration tests for GPU code paths
- LINT: Address 55 bare-except handlers
- DOCUMENT: Create ARCHITECTURE.md

---

## 8. Critical Invariants & Pitfalls

### Three Silent-Failure Guards (CLAUDE.md)
1. **FIELD::PHASE_DIR after chgcentre** - Missing → smeared/offset sources
2. **FIELD::REFERENCE_DIR sync with PHASE_DIR** - Missing → MODEL_DATA at wrong position
3. **TELESCOPE_NAME = DSA_110 before WSClean** - Missing → wrong beam model → 20% PB errors

### Known Issues
- RA-wrap bug in wsclean_mosaic.py (arithmetic vs circular mean)
- Legacy import migration (370 imports migrated, shim still needed)
- Cloud environment (CASA unavailable, uses compatibility shim)

---

## 9. Testing & Validation

### Test Infrastructure
- 62 test files, 220+ tests, 1060 test functions
- Markers: slow, benchmark, contract, integration, unit
- Fixtures: SimulationHarness, mock CASA tables, temporary SQLite

### Validation Gates
- Preflight: Dec-strip check, cal table existence, strip compatibility
- Per-tile: MS validity, CORRECTED_DATA ratio > 5.0, FITS validity
- Per-epoch: Three-gate QA (ratio, completeness, RMS)
- Photometry: QA-FAIL skip (default-strict) or lenient override

---

## 10. Documentation & Knowledge Artifacts

### Core Documentation
- CONTEXT.md: Domain glossary with path::Symbol citations
- CLAUDE.md: Project guidelines, CASA environment, silent-failure guards
- AGENTS.md: Agent workspace defaults
- README.md: Overview of 11 core submodules + data flow

### Reference Documentation
- docs/reference/calibration.md, imaging.md, mosaicking.md, photometry-and-ese.md

### Skills Documentation
- docs/skills/: 20+ guides covering calibration, imaging, photometry, mosaic workflows

---

## 11. Key Takeaways

### What Makes This Codebase Special
1. **Domain Specialization**: Deeply optimized for DSA-110 drift-scan transit array geometry
2. **Science Safety**: Explicit silent-failure guards protect against invisible science errors
3. **Calibration Robustness**: Multi-strategy calibration (auto-generate, borrow, validate)
4. **Production-Grade**: Checkpoint/resume, quarantine policies, provenance tracking
5. **Clean Architecture**: Zero circular dependencies, acyclic module flow
6. **Excellent Documentation**: Comprehensive reference docs, domain glossary, validated parameters

### How to Work With This Codebase
1. **Always use CASA environment**: `/opt/miniforge/envs/casa6/bin/python`
2. **Read reference docs before touching pipeline logic**: `docs/reference/*.md`
3. **Preserve silent-failure guards**: Never bypass PHASE_DIR, REFERENCE_DIR, TELESCOPE_NAME checks
4. **Use existing test patterns**: SimulationHarness for synthetic data, contract tests for real data
5. **Follow domain vocabulary**: Use terms verbatim from CONTEXT.md
6. **Be cautious with critical hubs**: casa_service.py (18 deps), runner.py (21 deps) require careful testing
7. **Address technical debt**: Split monolithic modules, reduce coupling, improve exception handling

### Critical Paths to Understand
1. Calibration table acquisition: ensure.py → borrowing → provenance validation
2. Phaseshift workflow: runner.py → chgcentre → PHASE_DIR sync → REFERENCE_DIR sync
3. WSClean invocation: cli_imaging.py → TELESCOPE_NAME reset → sky model seeding
4. Mosaicking QA: epoch_qa.py → three-gate validation → photometry gating
5. Batch orchestration: batch_pipeline.py → checkpoint → quarantine → resume

### Dependency Management Strategy
1. Test casa_service.py changes comprehensively (18 dependent files)
2. Extract utilities from runner.py (reduce fan-in from 21 to ~15)
3. Add validation gates between stages (prevent cascading failures)
4. Consider dependency injection for casa_service.py (replace singleton)
5. Plan for plugin architecture (long-term: reduce CASA coupling from 18 to ~3)

### Technical Debt Priorities
1. **High Priority**: Split calibration/calibration.py (2600 lines), extract flagging strategies
2. **Medium Priority**: Consolidate array shape normalization, formalize GPU contracts
3. **Low Priority**: Plugin architecture for CASA tasks, extract photometry as separate service

---

## Final Summary

**RESULT: Comprehensive framework established for DSA-110 continuum imaging pipeline**

The workspace contains a **production-grade radio astronomy imaging pipeline** with:

- **18 main modules** organized as sequential pipeline stages
- **35 entry point scripts** for orchestration, diagnostics, and validation
- **6-stage data flow** from HDF5 to science-ready light curves
- **7 external integrations** (CASA, WSClean, AOFlagger, EveryBeam, PyUVData, Aegean, Dagster)
- **32 key dependencies** across scientific, web, database, orchestration, and utility categories
- **26 domain-specific patterns** for drift-scan transit array, calibration science, imaging, mosaicking, and photometry
- **Three silent-failure guards** protecting against invisible science errors
- **220+ tests** with unit, integration, and contract categories
- **Comprehensive documentation** (CONTEXT.md, CLAUDE.md, reference docs, skills)
- **Clean acyclic architecture** with zero circular dependencies (coupling score 6/10)
- **Critical dependency hubs**: casa_service.py (18 fan-in), runner.py (21 fan-in), model.py (14 bidirectional)
- **Overall code health**: 7.2/10 (readability 8, maintainability 6, testability 7, complexity 6, documentation 8)

This framework provides a complete foundation for understanding the codebase architecture, data flow, scientific constraints, dependency structure, code quality, operational patterns, and change impact risks. The pipeline demonstrates strong domain specialization and science-safe engineering practices, with moderate technical debt in monolithic modules and dependency coupling that can be addressed through targeted refactoring.
