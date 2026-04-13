# Project Guidelines

## Scope

`dsa110-continuum` is a Python radio astronomy continuum imaging pipeline for DSA-110.
Science target: variable (and possibly transient) compact radio sources (e.g. ESEs) using
daily sky products and flux time series over week-to-month baselines.
Primary package: `dsa110_continuum/` (distribution name remains `dsa110_contimg`).

Use this file for workspace-wide defaults. For subsystem details and validated science constraints,
see `CLAUDE.md` and `docs/reference/`.

## Environment and setup

- Prefer CASA environment for scientific/runtime work:
  - `/opt/miniforge/envs/casa6/bin/python`
- Known packaging mismatch: `pyproject.toml` points setuptools to `src/`, but code lives in
  `dsa110_continuum/`. Treat editable install as unreliable.
- Use `PYTHONPATH=/workspace` when running scripts/tests from this workspace context.

## Build and test commands

- Lint: `ruff check dsa110_continuum/ scripts/ tests/`
- Format check: `ruff format --check dsa110_continuum/ scripts/ tests/`
- Tests (full): `/opt/miniforge/envs/casa6/bin/python -m pytest tests/ -q`
- Tests (single): `/opt/miniforge/envs/casa6/bin/python -m pytest tests/test_verify_sources.py -q`
- Monitor API (cloud-safe service):
  - `PYTHONPATH=/workspace uvicorn scripts.monitor_server:app --host 0.0.0.0 --port 8765`

## Architecture map

- Core pipeline stages are split across:
  - `conversion/` → `calibration/` → `imaging/` → `mosaic/` → `photometry/`
- Primary orchestrators:
  - `scripts/batch_pipeline.py` (full-day orchestration)
  - `scripts/run_pipeline.py` (single-tile reference)
  - `scripts/mosaic_day.py` (day mosaicking)

For implementation details and validated parameters, read `docs/reference/*.md` before touching
pipeline logic.

## Project-specific conventions

- New imports should use `dsa110_continuum.*` paths.
- Do **not** casually rewrite legacy compatibility wiring in package `__init__.py` layers.
- Keep changes minimal and science-safe; prefer adding tests for behavioral changes.
- Use existing test patterns in `tests/` (e.g., epoch QA, photometry, and pipeline regression tests).

## Critical pitfalls

- Do not remove or bypass silent-failure guards documented in `CLAUDE.md`:
  - `FIELD::PHASE_DIR` synchronization after `chgcentre`
  - `FIELD::REFERENCE_DIR` sync with `PHASE_DIR`
  - `OBSERVATION::TELESCOPE_NAME` reset to `DSA_110` before WSClean
- Cloud environments may not have CASA, WSClean, or telescope data paths (`/data/incoming/`,
  `/stage/dsa110-contimg/ms/`). Design tests/mocks accordingly.

## Output artifacts

- Save generated figures/csvs/previews under `/data/dsa110-continuum/outputs/`.
- Do not leave user-facing artifacts in `/tmp`.

## Learned User Preferences

- When pipeline or repo state is uncertain, prefer tests, diagnostics, or direct
  filesystem inspection over asking the user to supply answers the tools can
  determine.

## Learned Workspace Facts

- Do not track Measurement Sets or other large stage/correlation data in Git.
- Sliding-window mosaic settings (tiles per mosaic product and stride between
  products) describe how successive mosaic outputs are built from the tile
  stream, not the number of tiles whose beams significantly overlap one sky
  location; the latter follows beam geometry, drift spacing, and coadd weights.
- For compact-source variability science, per-position mosaic depth saturates
  after only a few overlapping drift tiles (about 3), so hour-scale windowed
  mosaics are the default science product and >1 hour/full-day coadds are
  diagnostic rather than default science products.

## Cursor Cloud specific instructions

### Python environment

The cloud VM does **not** have the CASA conda environment (`/opt/miniforge/envs/casa6/`).
Use the system Python 3.12 instead:

- Run tests: `PYTHONPATH=/workspace python3 -m pytest tests/ -q`
- Run lint: `ruff check dsa110_continuum/ scripts/ tests/`
- Run format check: `ruff format --check dsa110_continuum/ scripts/ tests/`
- Start monitor server: `PYTHONPATH=/workspace uvicorn scripts.monitor_server:app --host 0.0.0.0 --port 8765`

Always set `PYTHONPATH=/workspace` — the editable install is broken due to the
`pyproject.toml` `src/` vs `dsa110_continuum/` mismatch.

### `dsa110_contimg` compatibility shim

A compatibility shim at `~/.local/lib/python3.12/site-packages/dsa110_contimg_shim.py`
(auto-loaded via `.pth` file) redirects `dsa110_contimg.core.*` imports to their
`dsa110_continuum.*` equivalents and provides no-op stubs for `dsa110_contimg.common.*`,
`dsa110_contimg.infrastructure.*`, and `dsa110_contimg.workflow.*` paths (old package
internals not ported to the new codebase). This enables 307/312 tests to pass.

### Known test failures (pre-existing, cloud-only)

Two tests fail due to missing H17-specific resources, not code bugs:
- `test_ensure_calibration::test_fallback_full_sky_when_no_obs_dec` — needs VLA calibrator
  DB at `/data/dsa110-contimg/state/catalogs/vla_calibrators.sqlite3`
- `test_epoch_gaincal::test_wsclean_runs_when_flag_fraction_below_limit` — CASA `flagdata`
  task unavailable; extra subprocess call changes assertion count

### Data directories

`/data/dsa110-continuum/.pytest_tmp` is created by the update script for pytest basetemp.
Telescope data paths (`/data/incoming/`, `/stage/dsa110-contimg/ms/`) do not exist in the
cloud VM — tests and scripts that need them use mocks or skip gracefully.
