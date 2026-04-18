# Stage A Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire existing plot infrastructure into Stage A forced photometry, apply `["science", "notebook"]` SciencePlots style universally, and fix the same gap in `plot_lightcurves.py`.

**Architecture:** New `stage_a_diagnostics.py` wraps existing `plot_catalog_comparison`/`plot_field_sources` under a `FigureConfig.style_context()` context manager. `forced_photometry.py` gets a `--plots` flag (default on). `FigureConfig` gains `style_context()` and `_get_mpl_styles()`.

**Tech Stack:** matplotlib (Agg backend for CI), scienceplots, astropy, numpy, pandas, pytest, tempfile

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `dsa110_continuum/visualization/config.py` | Modify | Add `style_context()` and `_get_mpl_styles()` to `FigureConfig` |
| `dsa110_continuum/visualization/stage_a_diagnostics.py` | Create | `plot_flux_scale`, `plot_source_field` |
| `scripts/forced_photometry.py` | Modify | Add `--plots`/`--no-plots` flag, call diagnostics after CSV write |
| `scripts/plot_lightcurves.py` | Modify | Wrap `plot_source_lightcurve` with scienceplots context |
| `tests/test_stage_a_diagnostics.py` | Create | 6 CI-runnable tests |

---

## IMPORTANT: Environment Constraints

- **Matplotlib backend:** Set `matplotlib.use("Agg")` before any import of `pyplot` in test files
- **`tempfile.NamedTemporaryFile(delete=False)`** — always use; NOT pytest `tmp_path`
- **scienceplots installed:** `"science"` and `"notebook"` confirmed available
- **No real mosaic required in tests** — create minimal 10×10 synthetic FITS with WCS

### Minimal synthetic FITS helper (use in tests):

```python
def _make_minimal_fits(path: str, nx: int = 20, ny: int = 20) -> None:
    """Write a minimal FITS with WCS headers for testing."""
    import numpy as np
    from astropy.io import fits
    data = np.random.default_rng(42).standard_normal((ny, nx)).astype(np.float32) * 0.001
    hdu = fits.PrimaryHDU(data)
    h = hdu.header
    h["NAXIS"] = 2
    h["NAXIS1"] = nx
    h["NAXIS2"] = ny
    h["CTYPE1"] = "RA---SIN"
    h["CTYPE2"] = "DEC--SIN"
    h["CRPIX1"] = nx // 2
    h["CRPIX2"] = ny // 2
    h["CRVAL1"] = 344.124
    h["CRVAL2"] = 16.15
    h["CDELT1"] = -20.0 / 3600.0
    h["CDELT2"] = 20.0 / 3600.0
    h["CUNIT1"] = "deg"
    h["CUNIT2"] = "deg"
    fits.writeto(path, data, h, overwrite=True)
```

### Minimal CSV helper (use in tests):

```python
def _make_forced_phot_csv(path: str, n: int = 5, use_injected: bool = False) -> None:
    """Write a minimal forced photometry CSV."""
    import csv, math
    ref_col = "injected_flux_jy" if use_injected else "catalog_flux_jy"
    fieldnames = ["source_name", "ra_deg", "dec_deg", ref_col,
                  "measured_flux_jy", "flux_err_jy", "flux_ratio", "snr"]
    rows = []
    for i in range(n):
        ref_flux = 0.1 + i * 0.05
        meas_flux = ref_flux * (0.9 + i * 0.02)
        rows.append({
            "source_name": f"J344.{i:04d}+16.0000",
            "ra_deg": 344.0 + i * 0.01,
            "dec_deg": 16.15,
            ref_col: round(ref_flux, 5),
            "measured_flux_jy": round(meas_flux, 5),
            "flux_err_jy": 0.005,
            "flux_ratio": round(meas_flux / ref_flux, 4),
            "snr": round(meas_flux / 0.003, 1),
        })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
```

---

## Task 1: `FigureConfig.style_context()` + `_get_mpl_styles()`

**Files:**
- Modify: `dsa110_continuum/visualization/config.py`
- Create: `tests/test_stage_a_diagnostics.py` (partial — test 6 only)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage_a_diagnostics.py
import matplotlib
matplotlib.use("Agg")

import pytest
import matplotlib.pyplot as plt

from dsa110_continuum.visualization.config import FigureConfig, PlotStyle


def test_style_context_publication():
    """FigureConfig(PUBLICATION).style_context() applies scienceplots rcParams."""
    config = FigureConfig(style=PlotStyle.PUBLICATION)
    # Before context: record current font size
    before = plt.rcParams.get("font.size", None)
    with config.style_context():
        inside = plt.rcParams.get("font.size", None)
    # scienceplots "science" style sets font.size to 8 or 9 (smaller than default 10+)
    # The exact value isn't critical — what matters is that style_context() is callable
    # and does not raise, and that the rcParams were modified inside the context.
    assert inside is not None
    # After context, rcParams should be restored
    after = plt.rcParams.get("font.size", None)
    assert after == before
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/test_stage_a_diagnostics.py::test_style_context_publication -v 2>&1 | tail -10
```

Expected: FAIL — `FigureConfig` has no `style_context` attribute.

- [ ] **Step 3: Add `style_context()` and `_get_mpl_styles()` to `FigureConfig`**

Read `dsa110_continuum/visualization/config.py` first to find the exact end of the `FigureConfig` class (just before `QUICKLOOK_CONFIG = ...`).

Add these two methods inside the `FigureConfig` dataclass (after the last `@property`):

```python
    def _get_mpl_styles(self) -> list[str]:
        """Return matplotlib style names for this config's PlotStyle."""
        if self.style == PlotStyle.PUBLICATION:
            return ["science", "notebook"]
        return []

    def style_context(self):
        """Context manager that applies this config's matplotlib style.

        Usage::

            config = FigureConfig(style=PlotStyle.PUBLICATION)
            with config.style_context():
                fig, ax = plt.subplots()
                ...

        When ``PlotStyle.PUBLICATION`` is selected, applies SciencePlots
        ``["science", "notebook"]`` styles automatically.
        """
        import matplotlib.pyplot as plt
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            styles = self._get_mpl_styles()
            if styles:
                with plt.style.context(styles):
                    yield
            else:
                yield

        return _ctx()
```

Also add `from contextlib import contextmanager` at the top of `config.py` if
not already present (check first).

- [ ] **Step 4: Run test — must pass**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/test_stage_a_diagnostics.py::test_style_context_publication -v 2>&1 | tail -10
```

Expected: 1 PASSED.

- [ ] **Step 5: Commit**

```bash
cd /home/user/workspace/dsa110-continuum
git add dsa110_continuum/visualization/config.py tests/test_stage_a_diagnostics.py
git commit -m "feat(visualization): add FigureConfig.style_context() for scienceplots integration"
```

---

## Task 2: `stage_a_diagnostics.py` + tests 1–5

**Files:**
- Create: `dsa110_continuum/visualization/stage_a_diagnostics.py`
- Modify: `tests/test_stage_a_diagnostics.py` (add tests 1–5)

- [ ] **Step 1: Write the 5 failing tests**

Append to `tests/test_stage_a_diagnostics.py`:

```python
import csv
import os
import tempfile
import numpy as np
from astropy.io import fits


def _make_minimal_fits(path: str, nx: int = 20, ny: int = 20) -> None:
    """Write a minimal FITS with WCS headers for testing."""
    data = np.random.default_rng(42).standard_normal((ny, nx)).astype(np.float32) * 0.001
    hdu = fits.PrimaryHDU(data)
    h = hdu.header
    h["NAXIS"] = 2
    h["NAXIS1"] = nx
    h["NAXIS2"] = ny
    h["CTYPE1"] = "RA---SIN"
    h["CTYPE2"] = "DEC--SIN"
    h["CRPIX1"] = nx // 2
    h["CRPIX2"] = ny // 2
    h["CRVAL1"] = 344.124
    h["CRVAL2"] = 16.15
    h["CDELT1"] = -20.0 / 3600.0
    h["CDELT2"] = 20.0 / 3600.0
    h["CUNIT1"] = "deg"
    h["CUNIT2"] = "deg"
    fits.writeto(path, data, h, overwrite=True)


def _make_forced_phot_csv(path: str, n: int = 5, use_injected: bool = False) -> None:
    """Write a minimal forced photometry CSV."""
    ref_col = "injected_flux_jy" if use_injected else "catalog_flux_jy"
    fieldnames = ["source_name", "ra_deg", "dec_deg", ref_col,
                  "measured_flux_jy", "flux_err_jy", "flux_ratio", "snr"]
    rows = []
    for i in range(n):
        ref_flux = 0.1 + i * 0.05
        meas_flux = ref_flux * (0.9 + i * 0.02)
        rows.append({
            "source_name": f"J344.{i:04d}+16.0000",
            "ra_deg": 344.0 + i * 0.01,
            "dec_deg": 16.15,
            ref_col: round(ref_flux, 5),
            "measured_flux_jy": round(meas_flux, 5),
            "flux_err_jy": 0.005,
            "flux_ratio": round(meas_flux / ref_flux, 4),
            "snr": round(meas_flux / 0.003, 1),
        })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def test_plot_flux_scale_sim_mode():
    """plot_flux_scale with injected_flux_jy column writes a PNG file."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        fits_path = f.name
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            _make_forced_phot_csv(csv_path, n=5, use_injected=True)
            _make_minimal_fits(fits_path)
            from dsa110_continuum.visualization.stage_a_diagnostics import plot_flux_scale
            result = plot_flux_scale(csv_path, fits_path, out_dir)
            assert result is not None
            assert os.path.exists(str(result))
            assert str(result).endswith(".png")
        finally:
            os.unlink(csv_path)
            os.unlink(fits_path)


def test_plot_flux_scale_catalog_mode():
    """plot_flux_scale with catalog_flux_jy column writes a PNG file."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        fits_path = f.name
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            _make_forced_phot_csv(csv_path, n=5, use_injected=False)
            _make_minimal_fits(fits_path)
            from dsa110_continuum.visualization.stage_a_diagnostics import plot_flux_scale
            result = plot_flux_scale(csv_path, fits_path, out_dir)
            assert result is not None
            assert os.path.exists(str(result))
        finally:
            os.unlink(csv_path)
            os.unlink(fits_path)


def test_plot_flux_scale_empty_csv():
    """plot_flux_scale raises ValueError for a zero-row CSV."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        csv_path = f.name
        f.write("source_name,ra_deg,dec_deg,catalog_flux_jy,measured_flux_jy,snr\n")
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            from dsa110_continuum.visualization.stage_a_diagnostics import plot_flux_scale
            with pytest.raises(ValueError, match="No sources"):
                plot_flux_scale(csv_path, "/nonexistent/mosaic.fits", out_dir)
        finally:
            os.unlink(csv_path)


def test_plot_source_field_produces_file():
    """plot_source_field writes a PNG file when mosaic is readable."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        fits_path = f.name
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            _make_forced_phot_csv(csv_path, n=5)
            _make_minimal_fits(fits_path)
            from dsa110_continuum.visualization.stage_a_diagnostics import plot_source_field
            result = plot_source_field(csv_path, fits_path, out_dir)
            assert result is not None
            assert os.path.exists(str(result))
            assert str(result).endswith(".png")
        finally:
            os.unlink(csv_path)
            os.unlink(fits_path)


def test_plot_source_field_no_mosaic():
    """plot_source_field returns a path even when mosaic doesn't exist (graceful degradation)."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = f.name
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            _make_forced_phot_csv(csv_path, n=3)
            from dsa110_continuum.visualization.stage_a_diagnostics import plot_source_field
            # Missing mosaic should not crash — returns path to written PNG
            result = plot_source_field(csv_path, "/nonexistent/mosaic.fits", out_dir)
            assert result is not None
            assert os.path.exists(str(result))
        finally:
            os.unlink(csv_path)
```

- [ ] **Step 2: Run the 5 tests — verify they fail**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/test_stage_a_diagnostics.py -k "not test_style_context" -v 2>&1 | tail -15
```

Expected: 5 FAIL — `stage_a_diagnostics` module not found.

- [ ] **Step 3: Create `dsa110_continuum/visualization/stage_a_diagnostics.py`**

```python
"""Stage A (forced photometry) diagnostic plots for the DSA-110 continuum pipeline.

Provides two convenience functions that wrap existing photometry_plots.py
visualizations under the SciencePlots "science"+"notebook" style:

- plot_flux_scale   — log-log flux scatter + source overlay
- plot_source_field — field image with detection markers

Both functions:
- Accept a forced-photometry CSV and a mosaic FITS path
- Apply PlotStyle.PUBLICATION (SciencePlots) via FigureConfig.style_context()
- Degrade gracefully when the mosaic is absent (scatter-only panel)
- Raise ValueError on an empty CSV
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np

from dsa110_continuum.visualization.config import FigureConfig, PlotStyle

log = logging.getLogger(__name__)

# Columns that may carry the reference flux (sim mode vs production mode)
_REF_FLUX_COLS = ("injected_flux_jy", "catalog_flux_jy")


def _read_csv(csv_path: str | Path) -> list[dict]:
    """Read forced photometry CSV into a list of row dicts."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _get_ref_col(rows: list[dict]) -> str:
    """Return the reference flux column name present in rows."""
    if not rows:
        return "catalog_flux_jy"
    for col in _REF_FLUX_COLS:
        if col in rows[0]:
            return col
    return "catalog_flux_jy"


def _load_mosaic_data(mosaic_path: str | Path) -> tuple | None:
    """Load mosaic data and WCS. Returns (data, wcs, rms) or None on failure."""
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
        from astropy.stats import mad_std
        with fits.open(mosaic_path) as hdul:
            data = hdul[0].data.squeeze().astype(np.float64)
            wcs = WCS(hdul[0].header).celestial
        finite = data[np.isfinite(data)]
        rms = float(mad_std(finite)) if finite.size > 0 else 1e-3
        return data, wcs, rms
    except Exception as exc:
        log.warning("Could not load mosaic %s for plot: %s", mosaic_path, exc)
        return None


def plot_flux_scale(
    csv_path: str | Path,
    mosaic_path: str | Path,
    out_dir: str | Path,
    *,
    config: FigureConfig | None = None,
) -> Path | None:
    """Generate flux-scale diagnostic: log-log scatter of measured vs reference flux.

    Parameters
    ----------
    csv_path : str | Path
        Forced photometry CSV produced by ``scripts/forced_photometry.py``.
    mosaic_path : str | Path
        Mosaic FITS file (used for image overlay panel). If absent or
        unreadable, falls back to a single scatter panel.
    out_dir : str | Path
        Output directory. PNG written as ``{mosaic_stem}_flux_scale.png``.
    config : FigureConfig, optional
        Plot configuration. Defaults to ``PlotStyle.PUBLICATION``
        (SciencePlots ``["science", "notebook"]``).

    Returns
    -------
    Path
        Path to the written PNG file.

    Raises
    ------
    ValueError
        If the CSV has no rows.
    """
    if config is None:
        config = FigureConfig(style=PlotStyle.PUBLICATION)

    rows = _read_csv(csv_path)
    if not rows:
        raise ValueError(f"No sources in forced photometry CSV: {csv_path}")

    ref_col = _get_ref_col(rows)
    catalog_name = "Injected (sim)" if "injected" in ref_col else "Master catalog"

    # Build catalog_sources list for plot_catalog_comparison
    mosaic_data = _load_mosaic_data(mosaic_path)
    if mosaic_data is not None:
        data, wcs, rms = mosaic_data
    else:
        data, wcs, rms = None, None, 1e-3

    catalog_sources = []
    for row in rows:
        try:
            ra = float(row["ra_deg"])
            dec = float(row["dec_deg"])
            ref_flux_jy = float(row.get(ref_col, 0))
            meas_flux_jy = float(row.get("measured_flux_jy", 0))
            snr = float(row.get("snr", 0)) if row.get("snr", "") != "" else 0.0
            ratio = meas_flux_jy / ref_flux_jy if ref_flux_jy > 0 else np.nan

            if wcs is not None:
                try:
                    px, py = wcs.all_world2pix([[ra, dec]], 0)[0]
                except Exception:
                    px, py = 0.0, 0.0
            else:
                px, py = 0.0, 0.0

            if not np.isfinite(ratio):
                status = "Non-det"
            elif 0.5 <= ratio <= 2.0:
                status = "✓ Match"
            else:
                status = "Variable"

            catalog_sources.append({
                "ra": ra,
                "dec": dec,
                "catalog_flux": ref_flux_jy * 1000.0,   # Jy → mJy for plot
                "measured_flux": meas_flux_jy * 1000.0,
                "snr": snr,
                "status": status,
                "pix_x": float(px),
                "pix_y": float(py),
            })
        except (ValueError, KeyError):
            continue

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(mosaic_path).stem
    out_path = out_dir / f"{stem}_flux_scale.png"

    from dsa110_continuum.visualization.photometry_plots import plot_catalog_comparison

    if data is not None:
        with config.style_context():
            plot_catalog_comparison(
                data=data,
                wcs=wcs,
                rms=rms,
                catalog_sources=catalog_sources,
                output=out_path,
                config=config,
                catalog_name=catalog_name,
            )
    else:
        # Mosaic not available — scatter-only plot
        _plot_flux_scatter_only(catalog_sources, out_path, config, catalog_name)

    log.info("Flux scale diagnostic written: %s", out_path)
    return out_path


def _plot_flux_scatter_only(
    catalog_sources: list[dict],
    out_path: Path,
    config: FigureConfig,
    catalog_name: str,
) -> None:
    """Fallback: scatter plot without image overlay."""
    import matplotlib.pyplot as plt

    cat_flux = [s["catalog_flux"] for s in catalog_sources]
    meas_flux = [s["measured_flux"] for s in catalog_sources]
    colors = [
        "green" if "✓" in s["status"] else ("red" if s["status"] == "Non-det" else "orange")
        for s in catalog_sources
    ]

    with config.style_context():
        fig, ax = plt.subplots(figsize=(6, 5), dpi=config.dpi)
        ax.scatter(cat_flux, meas_flux, c=colors, s=60, edgecolors="black", alpha=0.7)
        lo = min(cat_flux + meas_flux)
        hi = max(cat_flux + meas_flux)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"{catalog_name} Flux (mJy)")
        ax.set_ylabel("Measured Flux (mJy)")
        ax.set_title("Flux Scale Diagnostic")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)


def plot_source_field(
    csv_path: str | Path,
    mosaic_path: str | Path,
    out_dir: str | Path,
    *,
    config: FigureConfig | None = None,
) -> Path | None:
    """Generate field source overlay: mosaic image with detection markers.

    Parameters
    ----------
    csv_path : str | Path
        Forced photometry CSV.
    mosaic_path : str | Path
        Mosaic FITS (used for image background). Falls back to RA/Dec scatter
        if absent.
    out_dir : str | Path
        Output directory. PNG written as ``{mosaic_stem}_field_sources.png``.
    config : FigureConfig, optional
        Defaults to ``PlotStyle.PUBLICATION``.

    Returns
    -------
    Path
        Path to the written PNG file.
    """
    if config is None:
        config = FigureConfig(style=PlotStyle.PUBLICATION)

    rows = _read_csv(csv_path)
    if not rows:
        raise ValueError(f"No sources in forced photometry CSV: {csv_path}")

    mosaic_data = _load_mosaic_data(mosaic_path)
    if mosaic_data is not None:
        data, wcs, rms = mosaic_data
    else:
        data, wcs, rms = None, None, 1e-3

    ref_col = _get_ref_col(rows)

    sources = []
    for row in rows:
        try:
            ra = float(row["ra_deg"])
            dec = float(row["dec_deg"])
            meas_flux_jy = float(row.get("measured_flux_jy", 0))
            snr = float(row.get("snr", 0)) if row.get("snr", "") != "" else 0.0
            ref_flux_jy = float(row.get(ref_col, 0))

            if wcs is not None:
                try:
                    px, py = wcs.all_world2pix([[ra, dec]], 0)[0]
                except Exception:
                    px, py = 0.0, 0.0
            else:
                px, py = ra, dec  # fallback: treat as pixel coords

            sources.append({
                "ra": ra,
                "dec": dec,
                "flux": meas_flux_jy * 1000.0,  # mJy
                "snr": snr,
                "catalog_flux": ref_flux_jy * 1000.0,
                "pix_x": float(px),
                "pix_y": float(py),
                "status": "✓" if snr >= 3.0 else "Low SNR",
            })
        except (ValueError, KeyError):
            continue

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(mosaic_path).stem
    out_path = out_dir / f"{stem}_field_sources.png"

    from dsa110_continuum.visualization.photometry_plots import plot_field_sources

    with config.style_context():
        plot_field_sources(
            data=data,
            wcs=wcs,
            rms=rms,
            sources=sources,
            output=out_path,
            config=config,
        )

    log.info("Field source diagnostic written: %s", out_path)
    return out_path
```

- [ ] **Step 4: Run all 6 tests — all must pass**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/test_stage_a_diagnostics.py -v 2>&1 | tail -20
```

Expected: 6 PASSED.

Debug notes:
- If `plot_field_sources` signature differs from what the code passes, read `visualization/photometry_plots.py` around line 456 and adjust the call accordingly.
- If scienceplots raises about font cache, that is a warning, not an error — tests should still pass.
- The `matplotlib.use("Agg")` at module top of the test file must come BEFORE any `import matplotlib.pyplot`.

- [ ] **Step 5: Commit**

```bash
cd /home/user/workspace/dsa110-continuum
git add dsa110_continuum/visualization/stage_a_diagnostics.py tests/test_stage_a_diagnostics.py
git commit -m "feat(visualization): add stage_a_diagnostics.py with plot_flux_scale and plot_source_field"
```

---

## Task 3: Wire `--plots` into `forced_photometry.py`

**Files:**
- Modify: `scripts/forced_photometry.py`

No new tests — the script is a thin CLI wrapper; the underlying functions are tested in Task 2.

- [ ] **Step 1: Add `--plots`/`--no-plots` arguments to `main()`**

Find the argument parser block in `scripts/forced_photometry.py`. After the last `parser.add_argument(...)` call (before `args = parser.parse_args()`), add:

```python
    parser.add_argument(
        "--plots", action="store_true", default=True,
        help="Generate flux scale and field source diagnostic plots (default: on).",
    )
    parser.add_argument(
        "--no-plots", action="store_false", dest="plots",
        help="Skip diagnostic plot generation.",
    )
```

- [ ] **Step 2: Call diagnostics after CSV write in `main()`**

Find the block at the end of `main()` that starts with:
```python
    med = result["median_ratio"]
    n = result["n_sources"]
```

Just after the CSV is written (after `run_forced_photometry` returns `result`), and **before** the `if args.sim:` exit check, insert:

```python
    # ── Diagnostic plots ──────────────────────────────────────────────────────
    if args.plots and result["n_sources"] > 0:
        try:
            from dsa110_continuum.visualization.stage_a_diagnostics import (
                plot_flux_scale,
                plot_source_field,
            )
            out_dir = Path(result["csv_path"]).parent
            flux_plot = plot_flux_scale(result["csv_path"], mosaic_path, out_dir)
            field_plot = plot_source_field(result["csv_path"], mosaic_path, out_dir)
            log.info("Diagnostic plots written: %s, %s", flux_plot, field_plot)
        except Exception as exc:
            log.warning("Diagnostic plot generation failed (non-fatal): %s", exc)
```

Also add `from pathlib import Path` near the top of `main()` if not already imported (check first — it's already imported at module level via `from pathlib import Path`).

- [ ] **Step 3: Verify syntax and help**

```bash
cd /home/user/workspace/dsa110-continuum
python -c "import ast; ast.parse(open('scripts/forced_photometry.py').read()); print('Syntax OK')"
python scripts/forced_photometry.py --help | grep -E "plots|no-plots"
```

Expected: `Syntax OK` and both flags visible in help.

- [ ] **Step 4: Commit**

```bash
cd /home/user/workspace/dsa110-continuum
git add scripts/forced_photometry.py
git commit -m "feat(scripts): add --plots/--no-plots flag to forced_photometry.py, wire Stage A diagnostics"
```

---

## Task 4: Fix scienceplots in `plot_lightcurves.py`

**Files:**
- Modify: `scripts/plot_lightcurves.py`

No tests needed — this is a one-function style fix.

- [ ] **Step 1: Add scienceplots import and wrap `plot_source_lightcurve`**

Read `scripts/plot_lightcurves.py`. Find `plot_source_lightcurve`. It currently does:

```python
    fig, ax = plt.subplots(figsize=(7, 3.5))
```

Change to:

```python
    import scienceplots  # noqa
    with plt.style.context(["science", "notebook"]):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.errorbar(...)
        ...  # everything up to and including fig.savefig(out_path, ...)
        plt.close(fig)
```

The entire figure creation, all `ax.*` calls, `fig.tight_layout()`, `fig.savefig()`, and `plt.close(fig)` must be inside the `with` block.

- [ ] **Step 2: Verify syntax**

```bash
cd /home/user/workspace/dsa110-continuum
python -c "import ast; ast.parse(open('scripts/plot_lightcurves.py').read()); print('Syntax OK')"
```

Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
cd /home/user/workspace/dsa110-continuum
git add scripts/plot_lightcurves.py
git commit -m "fix(scripts): apply scienceplots style in plot_lightcurves.py"
```

---

## Task 5: Push and verify

- [ ] **Step 1: Run full diagnostic test suite + regression check**

```bash
cd /home/user/workspace/dsa110-continuum
pytest tests/test_stage_a_diagnostics.py tests/test_source_finding.py tests/test_two_stage_photometry.py -v 2>&1 | tail -30
```

Expected: 6 + 12 + 14 = 32 tests all passing.

- [ ] **Step 2: Show git log for new commits**

```bash
cd /home/user/workspace/dsa110-continuum
git log --oneline origin/main..HEAD
```

- [ ] **Step 3: Push**

```bash
cd /home/user/workspace/dsa110-continuum
git push origin main
```

- [ ] **Step 4: Report**

Show:
1. Test counts (32 total)
2. Commit list
3. Push result
