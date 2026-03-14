# Project Guidelines

## Scope

`dsa110-continuum` is a Python radio astronomy continuum imaging pipeline for DSA-110.
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
