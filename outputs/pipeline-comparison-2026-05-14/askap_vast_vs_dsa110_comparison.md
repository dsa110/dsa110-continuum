# Pipeline Comparison: ASKAP VAST vs DSA-110 Continuum

**Prepared:** 2026-05-14

**ASKAP VAST workspace:** `/data/radio-pipelines/askap-vast/`

**DSA-110 workspace:** `/data/dsa110-continuum/`

**Framework docs used:**

- ASKAP VAST: `/data/radio-pipelines/askap-vast/WORKSPACE_GUIDE.md` (verified 2026-05-14)
- DSA-110: `/data/dsa110-continuum/outputs/workspace_framework_2026-05-14/dsa110_continuum_workspace_framework.md`

All counts verified by direct filesystem inspection and grep analysis.

---

## 1. Science Mission and Context

| Dimension | ASKAP VAST | DSA-110 |
| --- | --- | --- |
| **Telescope** | ASKAP (36 antennas, 12 m dishes, PAF feeds), Murchison WA | DSA-110 (96 antennas, 4.65 m dishes), OVRO CA |
| **Frequency** | ~887 MHz (L/UHF-band), ~288 MHz BW | ~1.4 GHz (L-band), 187.5 MHz BW, 16 subbands |
| **Science goal** | Detect and monitor variable/transient radio sources across wide sky (VAST survey: variables, slow transients) | Daily-cadenced forced photometry on known sources for ESE (Extreme Scattering Event) detection; variable/transient compact sources |
| **Operational cadence** | Epoch-based survey (13 Phase 1 epochs, 21+ Phase 2 epochs); not a daily pipeline | Hourly-epoch mosaics of ~12 sequential tiles per Dec strip; intended daily production |
| **Field of view** | Wide-field PAF (up to ~30 sq deg per pointing) | Per-tile transit strips, ~4800×4800 px @ 3 arcsec/px |
| **Source finding** | Selavy (ASKAP's own sourcefinder) → Selavy catalogues | BANE + Aegean for blind finding; forced photometry on NVSS/RACS/FIRST reference catalog |
| **Transient strategy** | Multi-epoch source association; variability metrics η, V across pipeline database | Per-source forced photometry with differential normalization; Mooley η, Vs, m; ESE sigma scoring |

The two pipelines share the same top-level scientific problem (find variable/transient radio sources) but approach it differently: VAST uses blind survey association across pre-defined epochs; DSA-110 uses continuous daily production with per-source monitoring against a reference catalog.

---

## 2. Repository Architecture

| Dimension | ASKAP VAST | DSA-110 |
| --- | --- | --- |
| **Structure** | 6 independent Git repositories (sub-repos) in one workspace directory | Single monorepo (`dsa110_continuum/` package) |
| **Primary package** | `vast-pipeline` (Django app + CLI), `vast-tools` (library), `vast-post-processing` (Typer CLI), `vaster-webapp` (Django app), `askap_surveys`, `forced_phot` | `dsa110_continuum/` — 23 subdirectories, 309 Python modules |
| **Package management** | Poetry for 3 main projects; pip/setuptools for 2 libraries; Docker for webapp | Single `pyproject.toml`, pip-based; conda env (`casa6`) on H17 for CASA runtime |
| **Python version** | `>=3.8,<3.11` (vast-pipeline, vast-tools); `>3.9,<3.11` (vast-post-processing); 3.10 (vaster-webapp Docker) | `>=3.11` |
| **Total Python source files** | ~160 across all 6 projects | 309 in `dsa110_continuum/` alone |
| **Total source lines (non-test)** | ~41,600 across all 6 projects | ~155,000 in `dsa110_continuum/` |
| **Linting** | Flake8 (vast-pipeline, vast-tools, vast-post-processing); Black + Flake8-bugbear + Prettier (vaster-webapp) | Ruff (E, F, W, I, TID, D); 100-char lines; NumPy docstring convention |

The DSA-110 codebase is roughly 3.7× larger by source line count despite covering a single pipeline, compared to VAST's six separate projects. This reflects both DSA-110's deeper internal specialization (23 subdirectories vs VAST's project-level boundaries) and its more thorough coverage of calibration, simulation, and visualization within a single package.

---

## 3. Data Flow Architecture

Both pipelines follow a broadly similar multi-stage flow, but the ingestion format, calibration backend, and mosaicking approach differ substantially.

```
ASKAP VAST
  Raw FITS (ASKAP telescope)
    → vast-post-processing:  Cutout2D crop → astrometric/flux correction (Huber regression) → SWarp mosaic → MOC/STMOC
    → Selavy source finding (external)
    → vast-pipeline:         image ingest → Selavy translation → source association → forced extraction → η/V/Vs/m statistics
    → vast-tools / vaster-webapp: interactive analysis / candidate classification

DSA-110
  Raw HDF5 (16 subbands × N timestamps)
    → conversion/:   PyUVData ingest → MS → UVW reconstruction → SPW merging
    → calibration/:  2-stage RFI flagging → bandpass/gain solve → applycal → phaseshift
    → imaging/:      WSClean (wgridder/IDG) → 4800×4800 px FITS tile (~5-min transit)
    → mosaic/:       tiles → hourly-epoch mosaic (QUICKLOOK or SCIENCE/DEEP)
    → qa/:           3-gate epoch QA (flux ratio, completeness, RMS)
    → photometry/:   Condon matched-filter forced phot → differential normalization → ESE detection → η/Vs/m
```

Key structural differences:

- **Input format:** ASKAP starts from pre-formed FITS images (Selavy already run by the observatory); DSA-110 starts from raw HDF5 visibilities and performs its own calibration and imaging from scratch.
- **Calibration:** VAST has no in-house calibration stage (ASKAP calibration is observatory-side); DSA-110 implements a full bandpass/gain solver with a documented acquisition order (same-date → generated → fallback → borrowed → fail) and three silent-failure invariants (`FIELD::PHASE_DIR`, `FIELD::REFERENCE_DIR`, `TELESCOPE_NAME`).
- **Mosaicking:** VAST uses SWarp (external); DSA-110 implements two internal mosaicking tiers (QUICKLOOK image-domain and SCIENCE/DEEP visibility-domain) with two operational modes (batch UTC-hour bins and sliding 12-tile window).
- **Source finding:** VAST delegates to Selavy (observatory-produced); DSA-110 uses BANE + Aegean for blind finding and a pre-built reference catalog for forced photometry.

---

## 4. Scientific Methodology and Workflow

This section documents the methodological differences in detail. The engineering implications are covered in the sections that follow; the focus here is on what each pipeline does scientifically and why.

### 4.1 Entry point into the processing chain

The most fundamental difference is where each pipeline enters the reduction chain.

**VAST** receives pre-calibrated, pre-imaged FITS files and Selavy source catalogues produced by the ASKAP observatory's own processing system (ASKAPSoft). The pipeline never touches raw visibilities. Its scientific contribution starts at the catalogue level: ingesting source component tables, translating them into a relational database, and reasoning about variability across time. The accuracy of the images and catalogues is the observatory's responsibility.

**DSA-110** starts from raw HDF5 visibility data (16 subbands × N timestamps per transit observation) and performs every reduction step internally: format conversion, UVW reconstruction, RFI flagging, calibration, imaging, and mosaicking. The pipeline is directly responsible for the scientific fidelity of all derived products.

This is not a minor implementation detail. It means DSA-110 must own calibration errors, primary beam errors, and mosaicking artefacts — failure modes that do not exist for VAST because the observatory absorbs them.

### 4.2 Calibration strategy

**VAST** has no calibration code. All bandpass, gain, and leakage calibration is performed by ASKAPSoft at the ASKAP site before data are transferred.

**DSA-110** implements a full in-house calibration hierarchy. The acquisition order is formally documented and enforced:

1. Same-date tables if valid
2. Generate from the primary flux calibrator transit (preferred)
3. Generate from a bright-source VLA calibrator catalog fallback (Dec-local search; falls back to full-sky if Dec is unknown)
4. Borrow the nearest validated tables from a compatible Dec strip
5. Fail loudly — never proceed with unknown calibration provenance

The pipeline also maintains three formally documented **silent-failure invariants** — conditions that produce scientifically wrong output without raising an exception:

- `FIELD::PHASE_DIR` must be updated after `chgcentre`. If omitted, CASA computes phase gradients relative to the old field centre, producing smeared or offset sources.
- `FIELD::REFERENCE_DIR` must be synced to `PHASE_DIR` after a phase shift. If omitted, CASA's `ft()` predicts model visibilities at the wrong sky position, causing self-calibration to diverge.
- `TELESCOPE_NAME` must be set to `DSA_110` before each WSClean run. `merge_spws()` silently resets it to `OVRO_MMA` for CASA compatibility; EveryBeam then selects the wrong primary beam model, producing beam errors up to ~20% near the field edge.

None of these failure modes are relevant to VAST because there is no calibration or imaging stage in the VAST pipeline.

### 4.3 Source detection: blind survey vs. targeted monitoring

The two pipelines represent opposite ends of the detection philosophy spectrum.

**VAST** is a **blind survey pipeline**. Selavy runs on every image and produces a full component catalogue regardless of prior knowledge. The pipeline ingests all detected components as `Measurement` objects and associates them across epochs using one of three association algorithms:

- **Basic**: simple nearest-neighbour spatial matching
- **Advanced**: likelihood ratio association using position uncertainties
- **De Ruiter**: angular separation metric that accounts for flux-weighted position errors

Sources that appear in a later epoch without a counterpart in an earlier one are flagged as new-source candidates. The scientific question is: *what changed between epoch N and epoch M?*

**DSA-110** uses **targeted forced photometry against a pre-built reference catalog** (NVSS, RACS, FIRST, VLA calibrator list; managed in SQLite). Rather than searching for sources, it measures the flux of sources it already knows about at their known positions, using a Condon matched-filter (noise-weighted Gaussian convolution in the image plane). Aegean and BANE are available for blind source-finding to update the reference catalog, but they are not the primary science path. The scientific question is: *did this particular known source change today?*

This distinction has deep implications:

- VAST can discover entirely new transient phenomena that were not in any prior catalogue. DSA-110 cannot; a genuinely new source would only be found on the next catalog-refresh cycle.
- DSA-110 is more sensitive to faint fractional flux changes in known sources, because the noise floor is set by the matched filter rather than the source-detection threshold.
- VAST's association step can produce false associations in crowded fields; DSA-110 has no association step, so crowding affects photometry noise but not source identity.

### 4.4 Mosaicking approach

**VAST** uses SWarp, a widely-used external co-addition tool, operating purely in the image domain. Before mosaicking, `vast-post-processing` applies per-tile astrometric corrections and flux-scale corrections (Huber robust linear regression against a reference catalog) to normalise residual offsets between tiles. The outputs include MOC and STMOC sky coverage maps for spatial and spatial-temporal querying.

**DSA-110** implements two internal mosaicking tiers with different scientific trade-offs:

- **QUICKLOOK** (image-domain): fast, suitable for real-time monitoring and alerts. Equivalent in concept to VAST's SWarp approach.
- **SCIENCE/DEEP** (visibility-domain): re-images the combined visibilities from multiple tiles using WSClean. This is scientifically superior — it allows full uv-plane coverage, consistent weighting across the combined dataset, and better PSF control — but requires that all per-tile Measurement Sets be available simultaneously and is considerably more compute-intensive.

Two operational modes control which tiles enter each mosaic:

- **Batch** (UTC-hour bins, ±2 tiles overlap): current production mode. Groups tiles by the UTC hour of their observation.
- **Sliding** (12-tile window, stride 6): streaming target. Produces more temporal overlap between consecutive mosaics, improving cadence for fast transients.

A known open bug affects the visibility-domain mosaic path: two code locations (`mosaic/wsclean_mosaic.py::compute_mean_meridian()` and `mosaic/jobs_wsclean.py`) compute the mosaic phase centre using arithmetic mean RA, which wraps incorrectly for tile groups that straddle RA=0°. The correct formula is the circular mean: `arctan2(mean(sin(RA)), mean(cos(RA)))`. The image-domain path already applies this correctly; the fix has not yet been ported to the visibility-domain path.

### 4.5 Variability metrics

Both pipelines compute the same family of variability metrics, derived from Mooley et al. (2016), but apply them differently.

| Metric | Formula | VAST usage | DSA-110 usage |
| --- | --- | --- | --- |
| **η (eta)** | Weighted reduced χ² of flux measurements | Per-source across all epochs; survey-wide ranking | Per-source per epoch; continuous monitoring |
| **V** | σ / mean flux (fractional variability) | Per-source population statistic | Not a primary output |
| **Vs** | t-statistic between a flux pair (A, B) | Computed in `vast-pipeline/pipeline/pairs.py` | Per-source per epoch pair; used for ESE scoring |
| **m** | Modulation index for a flux pair | Computed alongside Vs | Per-source per epoch pair |

The key difference is in how the metrics are used to make decisions:

**VAST** computes η and V as database-wide population statistics. Sources above configurable thresholds are surfaced to scientists for human review in `vaster-webapp`. The pipeline does not make autonomous pass/fail decisions; it generates a ranked candidate list for manual classification.

**DSA-110** applies differential photometry before computing metrics: each source's flux in a given epoch is normalised against the contemporaneous flux of nearby reference sources in the same mosaic. This removes correlated systematic offsets (residual gain errors, ionospheric phase gradients) that would otherwise inflate η and Vs. An ESE candidate is triggered when the differential deviation exceeds 5σ. The pipeline is designed to reach a per-source verdict autonomously, without human review of every epoch.

### 4.6 Quality assurance philosophy

**VAST** has no automated in-pipeline QA gating. Image quality and transient significance are assessed post-hoc by scientists using `vast-tools` (Jupyter notebooks) and `vaster-webapp` (web classification UI). The implicit assumption is that an observatory-calibrated FITS image either passes or fails at the human review stage.

**DSA-110** applies a three-gate automated QA check at the epoch level before any forced photometry runs:

1. **Flux ratio gate**: the median flux of reference sources in the epoch mosaic must be within a configured tolerance of their catalogue values. This catches calibration failures.
2. **Catalog completeness gate**: the fraction of reference sources detected above a SNR threshold must exceed a minimum. This catches imaging failures and severe RFI contamination.
3. **RMS gate**: the image noise level must be below a ceiling. This catches data-quality failures (e.g., data loss, severe flagging).

If any gate fails, the epoch is marked QA-FAIL, photometry is skipped, and the mosaic is not archived. The `--lenient-qa` flag is available only for diagnostics. This design is driven by the daily-production requirement: a scientist cannot review every hourly epoch, so the pipeline must gate its own outputs.

### 4.7 Temporal data model

**VAST** is **epoch-based with a sparse time axis**. Survey epochs are labelled (RACS 0, 14, 28, 29; Phase 1 epochs 1–13; Phase 2 epochs 17–21+), each representing a deliberate re-observation of the sky. The pipeline processes one epoch at a time and stores source–epoch flux pairs in PostgreSQL. The temporal spacing between epochs is typically weeks to months. This makes VAST well-suited for detecting sources that appear, disappear, or brighten over long baselines, but poorly suited for monitoring sources on daily or sub-daily timescales.

**DSA-110** has a **dense, continuous time axis**. Each day produces ~24 UTC-hour bins of transit mosaics. The science targets (ESEs) can evolve on timescales of hours to weeks, so daily cadence is scientifically required. The pipeline is designed to run unattended and to accumulate a multi-year light curve for every source in the reference catalog. This motivates the checkpoint recovery, quarantine logic, and automated QA gating — the pipeline must be robust enough to produce a trustworthy output every day without human oversight.

### 4.8 Human-in-the-loop vs. automated verdict

The two pipelines embody opposite assumptions about the role of human judgment:

**VAST** is explicitly human-in-the-loop. The `vaster-webapp` classification interface is a core pipeline component, not an afterthought. Candidates are tagged `true`, `false`, or `unsure` by researchers; the classification tags are stored in the database and inform subsequent analysis. This is appropriate for a survey discovering rare or unexpected phenomena where expert judgment is needed to distinguish genuine astrophysical signals from artefacts.

**DSA-110** targets autonomous operation. The ESE sigma threshold, differential photometry normalisation, and three-gate QA system are all designed to produce a per-source, per-epoch verdict without requiring a human to look at every result. The monitoring server (`scripts/monitor_server.py`) exposes Prometheus metrics for operational health, not scientific review. A scientist's attention is triggered only when an ESE candidate crosses the detection threshold — the pipeline handles the rest. This is appropriate for a monitoring program where the event class is well-defined (ESEs have a known signature) and daily throughput makes manual review impractical.

---

## 5. Technology Stack

### Calibration and Imaging

| Component | ASKAP VAST | DSA-110 |
| --- | --- | --- |
| **Calibration** | Observatory-side (not in-repo) | CASA 6 (casatools/casatasks ≥6.7.3), full in-repo bandpass/gain solver |
| **Source finding** | Selavy (external) | BANE + Aegean (via `source_finding/`) |
| **Imaging** | Not in-repo (observatory-produced FITS) | WSClean (wgridder/IDG) via `imaging/cli_imaging.py` + optional CASA tclean |
| **Primary beam** | Not in-repo | EveryBeam (requires `TELESCOPE_NAME=DSA_110`) |
| **Visibility format** | FITS (images only) | Measurement Set via casacore; PyUVData for HDF5 ingest |
| **Mosaicking** | SWarp (external) | Internal (`mosaic/`): image-domain and visibility-domain paths |
| **RFI flagging** | Not in-repo | AOFlagger (Lua strategy) + GPU-accelerated fallback (`rfi/`) |

### Parallel Processing

| Component | ASKAP VAST | DSA-110 |
| --- | --- | --- |
| **Parallelism** | Dask (vast-pipeline); Schwimmbad/MPI (vast-post-processing) | Dask + distributed (pinned <2024.11.0 for dask-ms compatibility); process-based workers for photometry |
| **Task orchestration** | Django-Q async task queue (vast-pipeline) | Dagster (>=1.12.10) with dagster-webserver; `batch_pipeline.py` CLI |
| **GPU acceleration** | None | Optional GPU-accelerated RFI (`rfi/`); optional cupy for GPU compute |

### Web / API / Database

| Component | ASKAP VAST | DSA-110 |
| --- | --- | --- |
| **Web frameworks** | Django 3.2 (vast-pipeline), Django 5.1 (vaster-webapp) | Django 4.2 (internal tooling); FastAPI + Strawberry GraphQL + Pydantic for monitor/QA APIs |
| **API** | DRF 3.11.0 (vast-pipeline), DRF 3.15.2 (vaster-webapp) | FastAPI ≥0.115.0; uvicorn; REST + GraphQL |
| **Database** | PostgreSQL + Q3C extension (spatial cone searches) | SQLite (catalog + pipeline state, via SQLAlchemy 2.0 + Alembic); async options (aiosqlite, asyncpg) |
| **Candidate classification UI** | vaster-webapp (Django 5.1, Docker, token auth) | None (no equivalent web classification UI) |
| **Monitoring** | None | `scripts/monitor_server.py` (FastAPI + Prometheus metrics); `scripts/qa_server.py` |
| **Cache/queue** | None | Redis ≥5.0.0; APScheduler |

### Dependency Footprint

| | ASKAP VAST (vast-pipeline) | ASKAP VAST (vast-tools) | DSA-110 |
| --- | --- | --- | --- |
| **Direct deps** | ~68 | ~58 | 101 |
| **Python floor** | 3.8 | 3.8 | 3.11 |
| **Notable** | Django 3.2, Dask 2022, Bokeh 2.4.2, vaex 4.17 | Bokeh ^3.1, vaex 4.17, mocpy | CASA 6, PyUVData 3.2.5, Dagster ≥1.12.10, FastAPI, Strawberry, PyArrow ≥14, Redis |

DSA-110 has significantly more modern dependencies (Python ≥3.11, Dagster, FastAPI, async DB drivers) while VAST's core projects are constrained to Python <3.11 with older library pins.

---

## 6. Testing Infrastructure

This is the sharpest difference between the two codebases and the most important one operationally.

| Metric | ASKAP VAST | DSA-110 |
| --- | --- | --- |
| **Total `def test` functions** | 324 (verified) | 1,060 (verified) |
| **Test files** | ~15 across 3 active projects | 65 |
| **Test framework** | Django `unittest.TestCase` (vast-pipeline); pytest (vast-tools, vast-post-processing) | pytest with pytest-cov, pytest-asyncio, pytest-timeout, pytest-xdist, pytest-doctestplus, hypothesis |
| **Test markers** | None documented | unit, integration, contract, benchmark, slow, requires_router, requires_dagster, requires_dsa110 |
| **Synthetic test data** | No test data generation; regression tests require external download | `simulation/` module (26 files, ~11,000 lines) generates synthetic UVH5 for full pipeline testing |
| **Integration/E2E tests** | None (vaster-webapp 0 tests; forced_phot tests are injection-validation scripts, not pytest) | `test_integration_e2e.py` (41 tests), `test_simulated_pipeline.py` (20 tests), `test_epoch_orchestrator.py` (45 tests) |
| **Mocking infrastructure** | Basic; no documented patterns | Explicit mock patterns for CASA tables, WSClean subprocesses, and catalogs; adapter layer makes mocking tractable |
| **CI** | GitHub Actions: test suite + lint (vast-pipeline, vast-tools only); no CI for vast-post-processing, vaster-webapp, forced_phot | GitHub Actions: `python-tests.yml` + `docs.yml` for the single repo |
| **Placeholder tests** | 11 in `vast-post-processing/tests/vast_post_processing/test_core.py` (all `pass`) | 0 |

**Ratio:** DSA-110 has 3.3× more test functions despite the user's note that it is "effectively untested" — this reflects that DSA-110 has invested heavily in test scaffolding, while VAST's testing is spread thinner across 6 projects. The more meaningful gap is in *integration* coverage: VAST has no end-to-end integration tests and several projects with zero runnable tests; DSA-110 has synthetic pipeline runs and epoch orchestrator tests.

One important nuance: DSA-110's AGENTS.md states the full suite takes ~14 minutes (dominated by `test_integration_e2e.py` and `test_simulated_pipeline.py`). This is the signature of genuine end-to-end coverage, not just unit scaffolding.

---

## 7. Code Quality Metrics

| Metric | ASKAP VAST | DSA-110 |
| --- | --- | --- |
| **Overall health score (assessed)** | 6.8/10 | 7.2/10 |
| **Bare except clauses** | 10 (7 in vaster-webapp, 2 vast-pipeline, 1 vast-tools) | 1 |
| **TODO/FIXME comments** | 62 (yandasoft 26, vast-pipeline 20, vast-tools 8, vast-post-processing 7, vaster-webapp 1) | 20 |
| **Coupling score (assessed)** | 3/10 (low — single shared library `forced_phot`) | 6/10 (higher — 9 module-level SCCs; one top-level SCC spanning 9 modules) |
| **Largest hotspot** | `vast-tools/vasttools/source.py`: 45 imports, 33 methods in one class | `calibration/calibration.py`: 2,996 lines; `calibration/flagging.py`: 2,662 lines |
| **Pre-existing lint violations** | Not formally counted; flake8 used selectively | ~1,300 ruff violations (W291/W293 whitespace, I001 import sort, D103 missing docstrings) — tracked, not bulk-fixed |
| **Import cycles** | None reported | 9 module-level SCCs; 530 internal import edges across 309 modules |
| **Silent-failure invariants documented** | None | 3 formally documented (PHASE_DIR, REFERENCE_DIR, TELESCOPE_NAME) + 1 known bug (RA circular mean in visibility-domain mosaic) |
| **Legacy namespace debt** | Minimal (clean project separation) | Ongoing: `dsa110_contimg` → `dsa110_continuum` rename ~370 imports across 136 files (completed); compatibility `__init__.py` layers must not be changed |

VAST's low coupling score (3/10) is a genuine architectural virtue: the six projects communicate through files (FITS, Arrow, Parquet) rather than code imports, making each independently deployable. DSA-110's higher coupling is the natural cost of packing a full calibration-through-photometry pipeline into a single package; the import SCC is the risk surface to watch.

---

## 8. Documentation

| Dimension | ASKAP VAST | DSA-110 |
| --- | --- | --- |
| **Root README** | None at workspace root; individual project READMEs only | `README.md` at repo root (81 lines): instrument overview, data flow, package structure, key scripts |
| **Architecture guide** | `AGENTS.md` / `CLAUDE.md` (in workspace root); `WORKSPACE_GUIDE.md` (this file) | `CLAUDE.md` (environment, operational constraints, invariants); `CONTEXT.md` (domain glossary with code citations); `AGENTS.md`; `WORKSPACE_GUIDE.md` |
| **Reference docs** | None per-subsystem | `docs/reference/` — 9 focused markdown files (calibration, flagging, imaging, mosaicking, photometry, ESE, conversion/QA, vast-crossref) |
| **Operational doc tiers** | Single-tier: AGENTS.md + per-project README | Four tiers: README (overview) → CLAUDE.md (operational) → docs/reference/ (subsystem) → docs/skills/ (agent-facing notes) |
| **Cross-telescope reference** | Not present | `docs/reference/vast-crossref.md` — explicitly documents ASKAP VAST patterns used as design reference |
| **Domain glossary** | In AGENTS.md (14 terms, code-cited) | `CONTEXT.md` (dedicated domain glossary with code citations, calibration vocabulary, variability metric formulas) |
| **MkDocs sites** | MkDocs Material: vast-pipeline (44 pages), vast-tools; Sphinx: vast-post-processing, vaster-webapp | MkDocs Material (`docs/quarto/` Quarto site: 10 .qmd files); docs.yml CI workflow; single unified site |
| **Architecture Decision Records** | None | `docs/adr/` directory present (currently empty placeholder) |
| **Docs total** | ~75 Markdown/RST files across 6 project doc trees | ~80 Markdown/Quarto/LaTeX files in one unified `docs/` tree |

Both codebases are well-documented for their complexity. VAST's documentation is distributed across 6 separate sites (a discoverability problem noted in its own `WORKSPACE_GUIDE.md`); DSA-110 consolidates everything in one tree with an explicit four-tier hierarchy and a dedicated cross-reference to VAST as an architectural touchstone.

---

## 9. Operational Infrastructure

| Dimension | ASKAP VAST | DSA-110 |
| --- | --- | --- |
| **Production entry point** | `./manage.py runpipeline` (Django management command) | `scripts/batch_pipeline.py --date YYYY-MM-DD` |
| **Orchestration** | Django-Q task queue (in-process scheduling) | Dagster ≥1.12.10 (workflow DAG, webserver, checkpoint recovery); `--dry-run` support |
| **Checkpoint / recovery** | Not documented | Full checkpoint recovery: quarantine after N failures (`--quarantine-after-failures`), `--retry-failed`, `--clear-quarantine` |
| **QA gating** | Post-run via vast-tools / vaster-webapp review | Strict in-pipeline: 3-gate epoch QA (flux ratio, completeness, RMS); QA-fail epochs skip photometry and archiving; `--lenient-qa` diagnostic override |
| **Monitoring** | None | FastAPI monitor server (`scripts/monitor_server.py`); Prometheus metrics; `scripts/qa_server.py`; `scripts/run_canary.sh` smoke test |
| **Containerization** | Docker Compose for vaster-webapp; optional for vast-post-processing | None documented (host-specific H17 deployment; casa6 conda env) |
| **Database backend** | PostgreSQL + Q3C (vast-pipeline, vaster-webapp) | SQLite via SQLAlchemy + Alembic (pipeline state and catalog); Redis cache |
| **Scripts** | ~25 management commands across 6 projects | 33 scripts under `scripts/`; distinct operational vs. diagnostic categories |
| **Canary / smoke tests** | `./manage.py test vast_pipeline.tests.test_pipeline` | `scripts/run_canary.sh` (QA smoke test against reference FITS tile); verified H17 result documented in AGENTS.md |

VAST's operational model is more human-in-the-loop (run pipeline → review in web UI → classify); DSA-110 aims for automated daily production with programmatic QA gating and recovery, reflecting its continuous-monitoring science goal.

---

## 10. Dependency and Version Risk

| Risk | ASKAP VAST | DSA-110 |
| --- | --- | --- |
| **numpy** | HIGH: spans `>=1.18.1,<1.23` → `~1.22.1` → `^1.21.2` → `2.1.0` across projects | Managed within one pyproject.toml; numpy constrained via pyuvdata/casatools floor |
| **Shared library version mismatch** | HIGH: vast-pipeline pins `forced-phot@v0.2.0`, vast-tools uses `forced-phot@latest` (git, no tag) | N/A: forced_phot not used; photometry is internal |
| **Bokeh** | MEDIUM: vast-pipeline pins `2.4.2`; vast-tools requires `^3.1` (major version gap) | Unified: bokeh `>=3.8.2` |
| **pandas** | MEDIUM: spans `^1.2.0` → `<2.0` | Not a primary dependency |
| **CASA version** | N/A | MEDIUM: casatools/casatasks `>=6.7.3` but pinned runtime is casa6 conda env; upgrading CASA requires re-validation of all invariants |
| **Dask** | Pinned `^2022.1.0,<2022.4.2` (vast-pipeline) | Pinned `<2024.11.0` (dask-ms compatibility) |
| **Legacy namespace** | None | MEDIUM: `dsa110_contimg` compatibility `__init__.py` layers must not be changed; double job-registration risk if re-export order changes |
| **Python version floor** | LOW but constrained: `<3.11` across VAST locks out modern typing, match statements, etc. | Modern: `>=3.11`; allows full modern stdlib |

---

## 11. Summary Assessment

### Where ASKAP VAST leads

1. **Project separation and low coupling.** Six independent projects communicating through files is architecturally clean and operationally flexible. Each project can be versioned, deployed, and tested independently. DSA-110's monorepo has an import SCC spanning most pipeline stages, which is a latent refactoring risk.

2. **Candidate classification infrastructure.** The vaster-webapp provides a complete web UI for human-in-the-loop candidate review with token authentication and Q3C spatial indexing. DSA-110 has no equivalent.

3. **Survey data model.** The vast-pipeline source association engine (basic/advanced/De Ruiter), PostgreSQL/Q3C backend, and Arrow output format are well-engineered for a multi-epoch survey with hundreds of thousands of sources. DSA-110 has no equivalent multi-epoch relational model.

4. **External ecosystem integration.** vast-tools has mature integration with SIMBAD, NED, Vizier, Gaia, and CASDA via astroquery. DSA-110 does not.

### Where DSA-110 leads

1. **Test coverage depth.** 1,060 verified test functions across 65 files vs. 324 across ~15 active test files. More critically, DSA-110 has end-to-end integration tests (`test_integration_e2e.py`, `test_simulated_pipeline.py`), epoch orchestrator tests, a full simulation module for synthetic data generation, and formal test markers. VAST has no integration tests and three projects with zero runnable tests.

2. **Calibration and imaging completeness.** DSA-110 implements the full visibility-domain pipeline (HDF5 → MS → calibration → imaging → mosaicking), including bandpass/gain solving, self-calibration, phaseshift, and primary beam correction. VAST receives pre-calibrated, pre-imaged FITS and begins from source catalogues.

3. **Code quality hygiene.** 1 bare except clause vs. 10; 20 TODO/FIXME comments vs. 62; zero placeholder tests vs. 11.

4. **Operational infrastructure.** Dagster orchestration, checkpoint/quarantine recovery, strict QA gating, Prometheus monitoring, and a canary smoke test are production-grade features absent from VAST.

5. **Documentation structure.** Four-tier documentation hierarchy with subsystem reference docs, domain glossary in `CONTEXT.md`, and an explicit cross-reference to ASKAP VAST patterns. VAST's documentation is stronger per-project but fragmented across 6 separate sites.

6. **Modern stack.** Python ≥3.11, Pydantic v2, FastAPI, Strawberry GraphQL, async DB drivers, ruff linting. VAST is constrained to Python <3.11 with older library pins.

### Net assessment

The user's framing — ASKAP VAST is more advanced and better tested than DSA-110 — requires some qualification:

- VAST is more **mature as a survey instrument** (multi-epoch association, web classification UI, external catalogue integration, Arrow/Parquet outputs, established PostgreSQL schema).
- DSA-110 is more **advanced as an engineering system** (deeper test coverage, end-to-end integration tests, full calibration/imaging implementation, production orchestration, monitoring).
- The "effectively untested" characterization of DSA-110 is not supported by the raw numbers (1,060 test functions, synthetic pipeline tests, integration E2E tests). It may instead reflect that the test suite has not been run recently in a validated production state, or that coverage of specific subsystems is uneven — but the scaffolding and volume are substantially ahead of VAST's.

The most important gaps to address in DSA-110, seen through the lens of VAST's maturity:

1. The import SCC should be documented and progressively broken (VAST's file-based decoupling is worth emulating).
2. The two monolithic files (`calibration.py` 2,996 lines, `flagging.py` 2,662 lines) should be split — VAST avoids this pattern across its projects.
3. The RA circular mean bug in visibility-domain mosaicking (known, documented) should be fixed with a regression test.
4. The `forced-phot` version mismatch risk in VAST should be resolved by either tagging the dependency in vast-tools or extracting it as a versioned internal library.

---

*All counts in this document are verified. Scores marked "(assessed)" are qualitative judgments, not automated metrics.*
