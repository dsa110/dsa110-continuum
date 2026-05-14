# CLAUDE.md

A radio astronomy continuum imaging pipeline for DSA-110 (Deep Synoptic Array, 110 antennas) at OVRO, ported from the older `dsa110-contimg` codebase. Science goal: detect variable/transient compact radio sources (ESEs) via daily-cadenced per-source forced photometry on hourly-epoch mosaics (~1 hour each).

Verified working state: `scripts/run_pipeline.py` produces a calibrated image of 3C454.3 at 12.5 Jy/beam against test HDF5 in `/data/incoming/` on H17.

## Project map

```
dsa110_continuum/
  conversion/      HDF5 → MS (UVH5 subband grouping, phase centre, UVW reconstruction)
  calibration/     bandpass, gain cal, applycal, phaseshift, self-cal, presets
  imaging/         WSClean / CASA tclean interface, ImagingParams, sky model seeding
  mosaic/          QUICKLOOK (image-domain) and SCIENCE/DEEP (visibility-domain) mosaicking
  photometry/      forced photometry, ESE detection, variability metrics (Mooley eta/Vs/m)
  catalog/         source catalog management (NVSS, RACS, FIRST, VLA cal list); SQLite backend
  qa/              delay validation, image quality, pipeline QA hooks
  simulation/      synthetic UVH5 generation for testing
  visualization/   diagnostic plots
  validation/      MS / image validators, storage checks
  evaluation/      pipeline stage evaluation harness
  selfcal/         self-calibration logic
  rfi/             RFI flagging strategies
  search/          source searching
  spectral/        spectral analysis
  pointing/        pointing corrections
  adapters/        external tool adapters
```

## Data flow

```
HDF5 (16 subbands × N timestamps)
  → [conversion/]   MS
  → [calibration/]  flagging + bandpass/gain solve + applycal
  → [imaging/]      phaseshift → WSClean (wgridder/IDG) → 4800×4800 px FITS tile (~5-min transit)
  → [mosaic/]       tiles → hourly-epoch mosaic (Quicklook or Science/Deep); ~12 tiles/epoch with overlap
  → [photometry/]   forced photometry → variability metrics → light curves
```

Science cadence: hourly-epoch mosaics of ~12 sequential tiles (~1 hour) along the current *Dec strip*, with overlap into adjacent epochs. **Not** a single 24-hour mosaic. Two operational modes: *batch* (UTC-hour bins, ±2 tiles overlap; current production) and *sliding* (12-tile window, stride 6; streaming target). See `CONTEXT.md` for citations.

## Key paths

Host-specific — workflows should use per-host roots rather than assuming a single one.

```
H17 / dsacamera:
  /data/incoming/                  raw HDF5 files
  /stage/dsa110-contimg/ms/        Measurement Sets
  /opt/miniforge/envs/casa6        CASA conda env (use this for ALL pipeline work)

H23 (correlator):
  /dataz/dsa110/operations/correlator/   raw correlator data (ZFS)

Optional sibling checkouts (useful when present):
  /data/radio-pipelines/askap-vast       reference for orchestration / QA parity
  /data/dsa110-antpos                    `ant_ids*.csv` for antenna-selection cross-checks
```

Do NOT track Measurement Sets or other large stage/correlation data in Git.

## Collaboration preferences

- When repo/runtime state is uncertain, prefer tests, diagnostics, or filesystem inspection over asking — ground recommendations in concrete evidence.
- Default to short, plain one-sentence answers unless asked for more depth.

<important if="you need to run anything in this repo (tests, scripts, lint, pipeline)">

ALWAYS use the casa6 conda env: `/opt/miniforge/envs/casa6/bin/python`. It has numpy, casacore, astropy, CASA. System `python3` is 3.13 (no scientific deps). The `pip` shim points to system 3.8 — if you need pip, use `/opt/miniforge/envs/casa6/bin/python -m pip`.

The editable install is unreliable (console-script entry points still point at the legacy `dsa110_contimg.*` namespace; see `pyproject.toml:150`). Set `PYTHONPATH=/workspace` when running scripts/tests from this workspace context.

| Command | What it does |
|---|---|
| `/opt/miniforge/envs/casa6/bin/python -m pytest tests/ -q` | Full suite (~1000 tests as of 2026-05; collection ~47 s, full run ~14 min — dominated by `test_integration_e2e.py` and `test_simulated_pipeline.py`) |
| `/opt/miniforge/envs/casa6/bin/python -m pytest tests/test_X.py::test_Y -q` | Single test |
| `ruff check dsa110_continuum/ scripts/ tests/` | Lint |
| `ruff check --fix dsa110_continuum/ scripts/ tests/` | Auto-fix safe issues |
| `ruff format --check dsa110_continuum/ scripts/ tests/` | Format check (NumPy docstrings, 100-char lines) |
| `scripts/run_pipeline.py` | Single-tile reference run: phaseshift → applycal → WSClean → check flux |
| `scripts/mosaic_day.py` | One date's tiles → hourly-epoch mosaics (legacy day-batch path; partitions by RA gap via `group_tiles_by_ra`). Production uses `batch_pipeline.py`. |
| `scripts/batch_pipeline.py --date YYYY-MM-DD` | Full orchestration: tiles → hourly-epoch mosaics → forced photometry |
| `scripts/source_finding.py` | BANE + Aegean on mosaics → blind source catalog |
| `scripts/forced_photometry.py` | Forced photometry vs reference catalog |
| `scripts/inventory.py` | HDF5 inventory + conversion status |
| `scripts/plot_lightcurves.py` | Plot multi-epoch light curves from CSVs |
| `scripts/stack_lightcurves.py` | Stack per-epoch CSVs into combined light curves |
| `scripts/variability_metrics.py` | Compute Mooley eta / Vs / m |
| `scripts/verify_sources.py` | Verify fluxes against expected values |
| `scripts/validate_date.py` | Run validation on one date's outputs |
| `scripts/run_canary.sh` | QA smoke test against a reference FITS tile |
| `PYTHONPATH=/workspace uvicorn scripts.monitor_server:app --host 0.0.0.0 --port 8765` | Start the monitor API (host-ops service) |
| `scripts/check_import_migration.py` | List stale `dsa110_contimg` imports remaining under `dsa110_continuum/` |

Pipeline DB: `dsa110 convert` queries the SQLite DB, NOT the filesystem. New dates must be indexed first:

    dsa110 index add --start YYYY-MM-DD --end YYYY-MM-DD --directory /data/incoming

</important>

<important if="the casa6 conda env is unavailable (cloud VM, Cursor Cloud, fresh container)">

Fallback: system Python 3.12 with `PYTHONPATH=/workspace` mandatory.

| Command | What it does |
|---|---|
| `PYTHONPATH=/workspace python3 -m pytest tests/ -q` | Full suite (cloud-VM variant) |
| `ruff check dsa110_continuum/ scripts/ tests/` | Lint |
| `ruff format --check dsa110_continuum/ scripts/ tests/` | Format check |
| `PYTHONPATH=/workspace uvicorn scripts.monitor_server:app --host 0.0.0.0 --port 8765` | Start monitor API |

A compatibility shim at `~/.local/lib/python3.12/site-packages/dsa110_contimg_shim.py` (auto-loaded via `.pth`) redirects `dsa110_contimg.core.*` imports to `dsa110_continuum.*` and stubs out `dsa110_contimg.common.*`, `dsa110_contimg.infrastructure.*`, and `dsa110_contimg.workflow.*`. Required for the bulk of the test suite to import successfully on cloud VMs.

All `casacore.tables` imports use `dsa110_continuum.adapters.casa_tables` — a drop-in wrapper over `casatools.table` — to avoid the C++ shared-library conflict that segfaults when both `python-casacore` and `casatools` load. The wrapper handles row-axis layout differences (`_rows_first`/`_rows_last`).

After installing `casatools` in a cloud VM, fix the bundled SQLite conflict:

    rm ~/.local/lib/python3.12/site-packages/casatools/__casac__/lib/libsqlite3.so.0
    ln -s /usr/lib/x86_64-linux-gnu/libsqlite3.so.0 ~/.local/lib/python3.12/site-packages/casatools/__casac__/lib/libsqlite3.so.0

Cloud-only test failures (pre-existing, not bugs):
- `test_ensure_calibration::test_fallback_full_sky_when_no_obs_dec` — needs VLA calibrator DB at `/data/dsa110-contimg/state/catalogs/vla_calibrators.sqlite3`.
- `test_epoch_gaincal::test_wsclean_runs_when_flag_fraction_below_limit` — mock expects single `subprocess.run` but epoch gaincal makes two WSClean invocations.

Telescope data paths (`/data/incoming/`, `/stage/dsa110-contimg/ms/`) do not exist on the cloud VM. Tests/scripts use mocks or skip gracefully.

</important>

<important if="you are importing, testing, or running anything under `dsa110_continuum.mosaic.*`">

The mosaic subsystem requires `/dev/shm/dsa110-contimg/` to be writable by the current user **at import time**. `dsa110_continuum/mosaic/__init__.py` → `mosaic/pipeline.py` → `mosaic/science_jobs.py` reaches into the legacy `dsa110_contimg.workflow.dagster.definitions`, which calls `_validate_pipeline_prerequisites()` on module load — a hard requirement, not lazy. Symptom: `RuntimeError: Pipeline validation failed: [DIR] /dev/shm/dsa110-contimg: Not writable`, raised the moment the import chain is touched (including by pure-Python WCS tests).

Fix when blocked: either (a) make the directory writable for your user, or (b) bypass the import chain by loading `dsa110_continuum.mosaic.builder` directly with `importlib.util.spec_from_file_location` — but if you do (b), resolve the path relative to `__file__`, NEVER hard-code an absolute path.

</important>

<important if="you are debugging test failures or interpreting suite results">

Known-fragile tests (as of 2026-05; verified by full-suite run):

- `tests/test_mosaic_ra_wrap.py` — collection error. Loads `dsa110_continuum/mosaic/builder.py` via `importlib.util.spec_from_file_location` with a hard-coded `/home/user/workspace/...` path (only valid on the Cursor Cloud VM). Move the path to `pathlib.Path(__file__).resolve().parents[1] / "dsa110_continuum/mosaic/builder.py"` to restore portability.
- `tests/test_simulated_pipeline.py::TestSimulatedMosaic` — three tests fail when `/dev/shm/dsa110-contimg/` is unwritable; root cause is the legacy validator described in the mosaic block above. Fail count is also order-sensitive (1/3 standalone, 3/3 in the full suite) because Python's failed-import caching differs across paths into the same module.

</important>

<important if="you are linting or considering bulk style fixes">

~1300 pre-existing ruff violations exist (as of 2026-05; whitespace W293/W291, unsorted imports I001, missing docstrings D103). Keep new code clean but DO NOT bulk-fix existing violations — they are tracked separately.

</important>

<important if="you are adding or modifying imports, package __init__.py, or anything that runs at import time">

`dsa110_continuum/` is the canonical package. The import rename from `dsa110_contimg.core.*` is complete (~370 imports across 136 files). New code must use `dsa110_continuum.*` imports — do NOT add new `dsa110_contimg` references.

The `__init__.py` re-export layers intentionally still reference old paths. Do NOT change them — the old package's bootstrap chain loads `core.calibration.jobs → register_job`, and switching to the new path triggers `ValueError` from double job-registration. The old package is still installed from `/data/dsa110-contimg/backend/src` and is loaded at runtime via those re-exports.

</important>

<important if="you are running batch_pipeline.py for production or smoke testing">

Use `--dry-run` before real compute to verify MS discovery, calibration-table resolution, checkpoint state, quarantine state, and rebuild/skip decisions:

    /opt/miniforge/envs/casa6/bin/python scripts/batch_pipeline.py \
      --date 2026-01-25 --start-hour 22 --end-hour 23 \
      --dry-run --quarantine-after-failures 3

Production smoke-test pattern (default-strict QA, bounded runtime, retry, parallel photometry):

    --quarantine-after-failures 3 --tile-timeout 1800 --retry-failed \
    --photometry-workers 4 --photometry-chunk-size 0

Operational flags:
- `--dry-run` prints execution plan, exits before writing run products.
- `--quarantine-after-failures N` skips MS entries whose checkpoint failure count reaches N. `--clear-quarantine` resets counts.
- QA gating is strict by default: QA-FAIL epochs do NOT run forced photometry or archive mosaics. `--lenient-qa` is a diagnostic override only.
- `--photometry-workers` enables process-based parallelism. `--photometry-chunk-size 0` uses automatic deterministic chunking.
- `--cal-date` takes a bare date (`2026-01-25`), NOT a timestamp. The code appends `T22:26:05` internally; passing a full timestamp produces a wrong path.

Each real run writes `run_<utc>.log`, `{date}_manifest.json`, `{date}_run_summary.json`, `run_report.md` under the date products directory.

Validated H17 result: `2026-01-25` hour 22 rebuilt 11 tiles → `/stage/dsa110-contimg/images/mosaic_2026-01-25/2026-01-25T2200_mosaic.fits`. QA failed on catalog completeness → manifest verdict `DEGRADED`, photometry skipped, `run_report.md` captured the failure.

</important>

<important if="you are working on calibration: bandpass, gain, applycal, or table discovery">

BP/G table acquisition order (must match implementation; update both this section and code in the same PR):
1. Same-date tables if valid.
2. Generate from primary flux calibrator transit (preferred).
3. Generate from bright-source VLA catalog fallback.
4. Borrow nearest validated tables (strip-compatibility required).
5. Fail loudly. NEVER proceed silently with unknown calibration provenance.

Bright-source fallback search: if `obs_dec_deg` is known, search is Dec-local (configured tolerance). If unknown, search uses full-sky Dec range to avoid false "no usable calibrator" failures.

Legacy table names like `/stage/dsa110-contimg/ms/{date}T22:26:05_0~23.{b,g}` are still supported. Manual symlinking from a validated reference date is acceptable for operational recovery, but must satisfy strip-compatibility validation.

See `CONTEXT.md` `## Calibration` for vocabulary (table conventions, borrowing semantics, `flux_anchor`, `selection_pool`, refant default, provenance sidecar) and `docs/reference/calibration.md` for K/B/G parameters, DEFAULT_PRESET, SelfCalConfig, and `bp_minsnr=5.0` (NOT the function default of 3.0).

</important>

<important if="you are implementing or modifying any pipeline subsystem (flagging, calibration, conversion/QA, imaging, mosaicking, photometry, ESE)">

Read both layers BEFORE touching code:
- `docs/skills/` — verified notes on how this repo's code actually works.
- `docs/reference/` — distilled analysis of the OLD dsa110-contimg + ASKAP VAST pipelines: validated numerical parameters, known failure modes, instrument-specific constraints not visible in the current code.

Reference files:
- `flagging.md` — AOFlagger Lua strategy, OVRO RFI, validated fractions, two-stage flagging contract (do NOT omit Stage 2).
- `calibration.md` — K/B/G params, DEFAULT_PRESET, SelfCalConfig, `bp_minsnr=5.0`.
- `conversion-and-qa.md` — UVH5 ingest workarounds, PyUVData float64/`run_check`, TELESCOPE_NAME dual-identity, post-conversion QA gates.
- `imaging.md` — WSClean hardcoded flags, sky model seeding two-step workflow, IDG SPW-merge, Galvin adaptive clip.
- `mosaicking.md` — QUICKLOOK vs SCIENCE/DEEP, mean-RA wrap bug, `-grid-with-beam` vs `-apply-primary-beam`.
- `photometry-and-ese.md` — Condon matched-filter, differential photometry reference selection, ESE scoring, variability thresholds.
- `vast-crossref.md` — Variability metric formulas (Vs, m, eta), ForcedPhot library, Condon errors, Huber flux-scale correction.

</important>

<important if="you are refactoring conversion, phaseshift, or imaging code">

Three silent-failure invariants that produce no exception but yield wrong science. They MUST be preserved.

1. **`FIELD::PHASE_DIR` after `chgcentre`.** WSClean's `chgcentre` may not update `FIELD::PHASE_DIR`. Patch with `update_phase_dir_to_target(ms_path, ra_deg, dec_deg)`. Symptom if missing: CASA computes phase gradients vs the old field centre → smeared/offset sources.

2. **`FIELD::REFERENCE_DIR` sync.** CASA's `ft()` reads `REFERENCE_DIR` (not `PHASE_DIR`) when computing model visibilities for self-cal and sky model prediction. After phaseshift, both must be updated: `sync_reference_dir_with_phase_dir(ms_path)`. Symptom: `MODEL_DATA` predicted at wrong sky position; self-cal diverges; sky model seeded at offset.

3. **`TELESCOPE_NAME = DSA_110` before each WSClean run.** `merge_spws()` (required before IDG) resets `OBSERVATION::TELESCOPE_NAME` to `OVRO_MMA` for CASA compatibility. EveryBeam needs `DSA_110`. Patch with `set_ms_telescope_name(ms_path, name="DSA_110")`. This is automatic inside `run_wsclean()` — preserve it if you change the imaging workflow. Symptom if missing: EveryBeam silently selects the wrong beam model → primary beam errors up to ~20% near field edge.

4. **FIELD direction column shape normalization.** `FIELD::PHASE_DIR` and `FIELD::REFERENCE_DIR` may appear as rows-first `(nfields, 1, 2)` or CASA column-major `(nfields, 2, 1)`. Fresh converted MS via the CASA table adapter typically return `(nfields, 2, 1)`. Code touching these columns must normalize both shapes via the calibration runner helpers — do NOT hard-code `(nfields, 1, 2)` indexing.

</important>

<important if="you are using ThreadPoolExecutor or signal-handling code">

The `@memory_safe` decorator uses `SIGALRM`, which only works in the main thread. Combining with `ThreadPoolExecutor` raises `ValueError: signal only works in main thread`. Use `ProcessPoolExecutor` instead.

</important>

<important if="you are touching imaging/cli_utils.py or interpreting CORRECTED_DATA fallbacks">

`detect_datacolumn()` raises `RuntimeError` if `CORRECTED_DATA` exists but is all zeros. It only falls back to `DATA` when `CORRECTED_DATA` is genuinely absent. This is intentional, not a silent failure.

</important>

<important if="you are saving derived artifacts (figures, PNGs, CSVs, FITS previews)">

Save under `/data/dsa110-continuum/outputs/` (organize by topic or date). Do NOT leave user-facing artifacts in `/tmp`.

</important>

<important if="you are touching FastAPI services: dsa110_continuum/mosaic/api.py, scripts/qa_server.py, or scripts/monitor_server.py">

Three FastAPI services exist; their statuses differ:
- `dsa110_continuum/mosaic/api.py` — **dormant**. Defines a router but no caller currently mounts it. Do NOT assume users hit this path; verify the mount before changing behavior.
- `scripts/qa_server.py` — **live**. The QA dashboard users currently rely on. Treat as production: changes need the same care as pipeline code.
- `scripts/monitor_server.py` — **live, host-ops**. Exposes a `POST /exec` shell hook; any change to that endpoint is a security-relevant edit and must be flagged.

The live-observability-stack work lands across these services; tracking issues #48–#62 (`gh issue list --label needs-triage --state open`).

</important>

<important if="you are reasoning about instrument geometry, data volumes, or observation cadence">

See `CONTEXT.md` `## Instrument` and `## Pipeline stages and products` for antenna count, dish size, band/subband structure, integration time, tile geometry, and hourly-epoch mosaic definition (batch and sliding modes), all with `path::Symbol` citations. Use the glossary's vocabulary verbatim — *tile*, *hourly-epoch mosaic*, *Dec strip* (not "RA strip", not "daily mosaic", not "snapshot/frame"); see `docs/agents/domain.md`.

Two non-obvious science facts:

- **Sliding-window mosaic parameters describe product cadence, not beam overlap.** "Tiles per mosaic" and "stride" set how successive mosaic products are built from the tile stream — they do NOT describe how many tile beams overlap any given sky location. The latter follows beam geometry, drift spacing, and coadd weights.
- **Per-position mosaic depth saturates after ~3 overlapping drift tiles** for compact-source variability science. Hour-scale windowed mosaics are the default science product; >1-hour / full-day coadds are diagnostic, not default science.

`scripts/batch_pipeline.py` runs an early Dec-strip guard (`check_dec_strip` vs `--expected-dec`, default 16.1°) when a same-date MS is present. Pointing declination flows from HDF5/UVH5 into the MS during conversion; the batch driver reads Dec from the MS, not by reopening HDF5.

</important>

## Agent skills

### Issue tracker

GitHub Issues on `dsa110/dsa110-continuum`. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical workflow labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context. Glossary at repo-root `CONTEXT.md` (with `path::Symbol` citations verifiable via `scripts/verify_glossary.py`); ADRs in `docs/adr/`. See `docs/agents/domain.md`.

## Current focus

Live observability stack — issues #48–#62 (`gh issue list --label needs-triage --state open` for the full set). See `docs/agents/issue-tracker.md` for gh-CLI conventions.
