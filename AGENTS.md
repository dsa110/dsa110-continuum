# AGENTS.md

## Cursor Cloud specific instructions

### Overview

dsa110-continuum is a Python radio astronomy continuum imaging pipeline for the DSA-110 telescope.
The package directory is `dsa110_continuum/` with the pip package name `dsa110_contimg`.
See `CLAUDE.md` for pipeline-specific paths, calibration table notes, and next tasks.

### Known package structure issue

`pyproject.toml` declares `[tool.setuptools.packages.find] where = ["src"]` but the code lives in `dsa110_continuum/`, not `src/`. This means `pip install -e .` fails. Instead, install dependencies directly from the dependency list and use `PYTHONPATH=/workspace` when running scripts or tests. The scripts in `scripts/` already add the project root to `sys.path`.

Internal imports within `dsa110_continuum/` reference `dsa110_contimg.core.*` and `dsa110_contimg.infrastructure.*` (828+ references per TODOS.md) — an incomplete rename from the predecessor repo. Many submodule imports will fail at runtime until this rename is completed.

### Lint / Test / Run

- **Lint:** `ruff check dsa110_continuum/ scripts/` (819 pre-existing issues, all from before any agent work)
- **Test:** `PYTHONPATH=/workspace pytest` (no `tests/` directory exists yet; pytest collects 0 tests)
- **Run (monitor server):** `PYTHONPATH=/workspace uvicorn scripts.monitor_server:app --host 0.0.0.0 --port 8765`
  - Swagger UI at `http://localhost:8765/docs`
  - This is the only service runnable without DSA-110 hardware data paths (`/data/incoming/`, `/stage/dsa110-contimg/ms/`)
- **Pipeline scripts** (require data on H17): `scripts/run_pipeline.py`, `scripts/batch_pipeline.py`, `scripts/mosaic_day.py`

### Virtual environment

The venv lives at `/workspace/.venv`. Activate with `source /workspace/.venv/bin/activate`.
`python-casacore` is installed from the manylinux wheel and does not need system casacore libraries for basic table operations.

### External dependencies not available in cloud

- **CASA 6** (casatools/casatasks): needed for calibration steps; requires the conda env at `/opt/miniforge/envs/casa6` which is only on H17.
- **WSClean**: needed for imaging; requires native binary or Docker GPU image.
- **Telescope data**: raw HDF5 at `/data/incoming/`, MS at `/stage/dsa110-contimg/ms/` — only on H17.
