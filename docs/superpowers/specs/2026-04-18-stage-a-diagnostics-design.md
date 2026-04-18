# Stage A Diagnostics — Design Spec

**Date:** 2026-04-18
**Author:** Jakob Faber / AI pair

---

## Goal

Wire the existing but unconnected diagnostic plot infrastructure into Stage A
(forced photometry), establish `["science", "notebook"]` from SciencePlots as
the universal plot standard across all pipeline output figures, and fix the same
gap in `plot_lightcurves.py`.

---

## Background

The following infrastructure exists but is never called from `forced_photometry.py`:

| Symbol | Location | Purpose |
|--------|----------|---------|
| `plot_catalog_comparison` | `visualization/photometry_plots.py` | Log-log flux scatter + source overlay |
| `plot_field_sources` | `visualization/photometry_plots.py` | Field image with detection markers |
| `FigureConfig(style=PlotStyle.PUBLICATION)` | `visualization/config.py` | Publication-grade config object |
| `plt.style.context(["science", "notebook"])` | Only in `validate_step6_mosaic.py` | SciencePlots style |

`scienceplots` is installed; `"science"` and `"notebook"` styles are confirmed
available via `matplotlib.pyplot.style.available`.

---

## File Structure

```
dsa110_continuum/visualization/
    config.py                  ← Modify: apply scienceplots in PUBLICATION style
    stage_a_diagnostics.py     ← Create: plot_flux_scale, plot_source_field

scripts/
    forced_photometry.py       ← Modify: add --plots flag, call diagnostics
    plot_lightcurves.py        ← Modify: wrap with scienceplots context

tests/
    test_stage_a_diagnostics.py  ← Create: 6 CI-runnable tests
```

---

## Components

### 1. `FigureConfig` scienceplots integration (`visualization/config.py`)

`FigureConfig.__post_init__` already sets style-specific parameters (dpi, cmap,
linewidth) for `PlotStyle.PUBLICATION`, but never applies the matplotlib style.
Add a `apply_style()` method and a `style_context()` context manager:

```python
@contextmanager
def style_context(self):
    """Context manager that applies this config's matplotlib style."""
    import matplotlib.pyplot as plt
    styles = self._get_mpl_styles()
    if styles:
        with plt.style.context(styles):
            yield
    else:
        yield

def _get_mpl_styles(self) -> list[str]:
    if self.style == PlotStyle.PUBLICATION:
        return ["science", "notebook"]
    return []
```

All existing plot functions in `photometry_plots.py` and `source_plots.py`
already accept a `config: FigureConfig` parameter — they just need to wrap their
`matplotlib.pyplot` calls with `with config.style_context():`.

### 2. `stage_a_diagnostics.py`

Two pure functions, both CI-testable with synthetic data:

**`plot_flux_scale(csv_path, mosaic_path, out_dir, *, config=None) → Path`**

Reads the forced photometry CSV (columns: `ra_deg`, `dec_deg`,
`measured_flux_jy`, `catalog_flux_jy` or `injected_flux_jy`, `snr`).
Opens the mosaic FITS for the image panel. Converts Jy → mJy for the scatter
plot. Calls `plot_catalog_comparison` under `config.style_context()`.
Returns the output path.

If the mosaic is not readable (e.g., sim mode), falls back to a text-only
scatter panel (no image overlay).

**`plot_source_field(csv_path, mosaic_path, out_dir, *, config=None) → Path`**

Reads the same CSV, builds the `sources` list expected by `plot_field_sources`,
calls it under `config.style_context()`, returns the output path.

Both functions:
- Use `PlotStyle.PUBLICATION` (scienceplots) by default
- Write to `{out_dir}/{mosaic_stem}_flux_scale.png` /
  `{mosaic_stem}_field_sources.png`
- Raise `ValueError` if the CSV has no rows
- Return `None` (no crash) if the mosaic is absent (graceful degradation for
  sim mode where mosaic exists but may be minimal)

### 3. `forced_photometry.py` — `--plots` flag

Add `--plots` / `--no-plots` (default: `--plots`). After writing the CSV call:

```python
if args.plots and result["n_sources"] > 0:
    from dsa110_continuum.visualization.stage_a_diagnostics import (
        plot_flux_scale, plot_source_field,
    )
    out_dir = Path(result["csv_path"]).parent
    plot_flux_scale(result["csv_path"], mosaic_path, out_dir)
    plot_source_field(result["csv_path"], mosaic_path, out_dir)
```

In sim mode this still runs — producing plots of the sim flux ratios using
`injected_flux_jy` as the reference column.

### 4. `plot_lightcurves.py` — scienceplots

The `plot_source_lightcurve` function uses bare matplotlib with no style
context. Wrap the figure creation with:

```python
import scienceplots  # noqa
with plt.style.context(["science", "notebook"]):
    fig, ax = plt.subplots(...)
    ...
```

Same for `build_summary_html` if it creates any figures.

---

## Tests (`tests/test_stage_a_diagnostics.py`)

All CI-runnable. Use `tempfile.NamedTemporaryFile(delete=False)` + manual
cleanup. Matplotlib backend set to `"Agg"` at module level.

| # | Test | What it checks |
|---|------|---------------|
| 1 | `test_plot_flux_scale_sim_mode` | Writes synthetic CSV with `injected_flux_jy`, calls `plot_flux_scale`, asserts PNG file written |
| 2 | `test_plot_flux_scale_catalog_mode` | Same with `catalog_flux_jy` column |
| 3 | `test_plot_flux_scale_empty_csv` | `ValueError` on zero-row CSV |
| 4 | `test_plot_source_field_produces_file` | Synthetic CSV + stub FITS, asserts PNG written |
| 5 | `test_plot_source_field_no_mosaic` | Returns gracefully when mosaic path doesn't exist |
| 6 | `test_style_context_publication` | `FigureConfig(PlotStyle.PUBLICATION).style_context()` sets rcParams matching scienceplots |

---

## Design Decisions

1. **`style_context()` on `FigureConfig`** — rather than a global `import
   scienceplots` at module level (which would make all plot functions depend on
   scienceplots being installed), the context manager is opt-in per call. The
   default config for both new functions is `PUBLICATION`, so the standard is
   automatically applied without callers having to know about scienceplots.

2. **No changes to existing `plot_catalog_comparison` / `plot_field_sources`
   internals** — they already accept `config`. We wrap them from the outside
   with `style_context()`. This keeps the diff minimal and non-breaking.

3. **Graceful mosaic-absent path** — `plot_flux_scale` with a missing mosaic
   produces a single-panel scatter plot (no image overlay). This makes tests
   trivial to write and keeps sim-mode CI clean.

4. **`--plots` default on** — production runs always produce diagnostics. Pass
   `--no-plots` for headless batch runs where speed matters.

5. **`plot_lightcurves.py` fix is one-line** — add `import scienceplots` and
   wrap `plt.subplots` in `with plt.style.context(["science", "notebook"]):`.
   No structural changes.
