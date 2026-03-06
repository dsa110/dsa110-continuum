# AGENTS.md

## Cursor Cloud specific instructions

### Overview

dsa110-continuum is a radio astronomy continuum imaging pipeline for DSA-110. See `CLAUDE.md` for project structure, critical silent failure notes, and key paths.

### Dependencies

- **System**: `casacore-dev`, `libcfitsio-dev`, `wcslib-dev`, `libhdf5-dev`, `libboost-python-dev` (needed by `python-casacore`)
- **Python**: All deps declared in `pyproject.toml`. Install with `pip install` of listed deps + `.[dev]` extras.
- The package itself cannot be installed via `pip install -e .` because `pyproject.toml` references `[tool.setuptools.packages.find] where = ["src"]` but the actual source lives at the repo root under `dsa110_continuum/`. Install dependencies directly instead.

### Running tests

```bash
PYTHONPATH=/workspace:/workspace/scripts pytest tests/ -v
```

- `PYTHONPATH` must include both `/workspace` (for `dsa110_continuum` package) and `/workspace/scripts` (for script modules like `variability_metrics`, `stack_lightcurves`, etc.).
- Tests that import from `dsa110_continuum.calibration` will fail with `ModuleNotFoundError: No module named 'dsa110_contimg'` — this is a pre-existing issue where `dsa110_continuum/calibration/__init__.py` references the `dsa110_contimg` namespace which does not exist in this repo layout.
- Passing tests: `test_pipeline_fix`, `test_plot_lightcurves`, `test_stack_lightcurves`, `test_variability_metrics`.

### Linting

```bash
ruff check .        # Lint (configured in pyproject.toml)
ruff format --check .  # Format check
```

Pre-existing lint/format issues exist; this is expected.

### Running scripts

Scripts are in `/workspace/scripts/` and use `#!/opt/miniforge/envs/casa6/bin/python` shebangs (not available in cloud). Run them directly with `python3 scripts/<name>.py`. Most accept `--help`.

### Gotchas

- The `casacore-data-tai-utc` apt package postinst script fails due to numpy ABI incompatibility (system casacore Python bindings compiled against numpy 1.x vs numpy 2.x installed). The C libraries themselves work fine; only the data-update script is affected.
- `~/.local/bin` must be on `PATH` for pytest, ruff, and other pip-installed CLI tools.
