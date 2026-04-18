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

import numpy as np

from dsa110_continuum.visualization.config import FigureConfig, PlotStyle


def _setup_matplotlib() -> None:
    """Configure matplotlib for headless rendering (idempotent)."""
    import matplotlib
    matplotlib.use("Agg")


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
    # warn when falling back — neither expected column is present
    log.warning(
        "Neither %s found in CSV; defaulting to 'catalog_flux_jy'. "
        "All source fluxes will be zero.",
        " nor ".join(_REF_FLUX_COLS),
    )
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
    """Generate flux-scale diagnostic: log-log scatter of measured vs reference flux."""
    _setup_matplotlib()
    if config is None:
        config = FigureConfig(style=PlotStyle.PUBLICATION)

    rows = _read_csv(csv_path)
    if not rows:
        raise ValueError(f"No sources in forced photometry CSV: {csv_path}")

    ref_col = _get_ref_col(rows)
    catalog_name = "Injected (sim)" if "injected" in ref_col else "Master catalog"

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

    if data is not None:
        from dsa110_continuum.visualization.photometry_plots import plot_catalog_comparison
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

    # guard against empty list
    if not cat_flux:
        log.warning("No parseable sources for flux scatter plot; skipping.")
        return

    # filter out zeros for log scale
    safe = [(c, m) for c, m in zip(cat_flux, meas_flux) if c > 0 and m > 0]
    if not safe:
        log.warning("All sources have zero flux; skipping log-scale scatter plot.")
        return
    cat_flux = [c for c, _ in safe]
    meas_flux = [m for _, m in safe]
    colors = [
        "green" if "✓" in s["status"] else ("red" if s["status"] == "Non-det" else "orange")
        for s in catalog_sources if s["catalog_flux"] > 0 and s["measured_flux"] > 0
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


def _plot_field_scatter_only(
    sources: list[dict],
    out_path: Path,
    config: FigureConfig,
) -> None:
    """Fallback: simple scatter plot of source positions without image overlay."""
    import matplotlib.pyplot as plt

    with config.style_context():
        fig, ax = plt.subplots(figsize=(6, 5), dpi=config.dpi)
        detected = [(s["ra"], s["dec"]) for s in sources if s.get("status") == "✓"]
        low_snr  = [(s["ra"], s["dec"]) for s in sources if s.get("status") != "✓"]
        if detected:
            ax.scatter(*zip(*detected), c="green", s=40, label="SNR ≥ 3", zorder=3)
        if low_snr:
            ax.scatter(*zip(*low_snr),  c="orange", s=40, label="Low SNR", zorder=2)
        ax.legend()
        ax.set_xlabel("RA (deg)")
        ax.set_ylabel("Dec (deg)")
        ax.set_title("Field Sources (no mosaic — RA/Dec scatter)")
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
    """Generate field source overlay: mosaic image with detection markers."""
    _setup_matplotlib()
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
                px, py = 0.0, 0.0

            sources.append({
                "ra": ra,
                "dec": dec,
                "peak_flux": meas_flux_jy,       # Jy/beam — plot_field_sources expects peak_flux
                "flux": meas_flux_jy * 1000.0,   # mJy convenience copy
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

    if data is not None:
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
    else:
        # Mosaic not available — scatter-only fallback
        _plot_field_scatter_only(sources, out_path, config)

    log.info("Field source diagnostic written: %s", out_path)
    return out_path
