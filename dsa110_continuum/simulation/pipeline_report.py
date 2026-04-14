"""5-panel diagnostic report for SimulatedPipelineResult.

Generates a publication-quality PNG summarising all pipeline stages after a
``SimulatedPipeline.run()`` call.  Uses the SciencePlots ``["science",
"notebook"]`` style via ``apply_pipeline_style()``.

Panels
------
1. Gain corruption summary — per-antenna amplitude scatter shown as a text
   summary with colour-coded pass/fail indicator.
2. Calibration phase improvement — bar comparing pre- / post-calibration
   phase scatter (deg) for tile 0.
3. WSClean restored image — greyscale FITS display with source marker overlaid
   at each registered ground-truth position.
4. Epoch mosaic — greyscale FITS display with source marker.
5. Flux recovery — grouped bar chart of injected vs. recovered flux per source.

Usage
-----
>>> from dsa110_continuum.simulation.pipeline_report import generate_pipeline_report
>>> path = generate_pipeline_report(result, "pipeline_diagnostic.png")
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from dsa110_continuum.simulation.pipeline import SimulatedPipelineResult

logger = logging.getLogger(__name__)


def generate_pipeline_report(
    result: "SimulatedPipelineResult",
    output_path: Path | str,
    tile_restored_path: Path | None = None,
    source_positions: list[tuple[float, float]] | None = None,
) -> Path:
    """Write a 5-panel PNG diagnostic for a simulated pipeline run.

    Parameters
    ----------
    result:
        Output of ``SimulatedPipeline.run()``.
    output_path:
        Destination PNG path (created with parent directories).
    tile_restored_path:
        Path to the first tile's restored FITS (for panel 3).  If *None*,
        the method attempts to auto-locate it from ``result.work_dir``; if
        still not found, panel 3 shows a placeholder.
    source_positions:
        Optional list of ``(ra_deg, dec_deg)`` tuples for source markers
        overlaid on the image panels.  Defaults to positions extracted from
        ``result.source_results``.

    Returns
    -------
    Path
        Path to the written PNG file.
    """
    from dsa110_continuum.simulation.plot_style import apply_pipeline_style

    apply_pipeline_style()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Auto-locate tile-0 restored image if not supplied ─────────────────
    if tile_restored_path is None:
        candidate = result.work_dir / "tile_00" / "wsclean_out" / "wsclean-image.fits"
        if candidate.exists():
            tile_restored_path = candidate

    # ── Extract source positions for overlay ──────────────────────────────
    if source_positions is None:
        source_positions = [
            (r.ra_deg, r.dec_deg) for r in result.source_results
        ]

    # ── Figure layout ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(24, 6.5))
    fig.suptitle(
        "DSA-110 Simulated Pipeline Diagnostic",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    # ── Panel 1: Gain corruption summary ──────────────────────────────────
    ax1 = axes[0]
    ax1.set_title("(1) Gain Corruption", fontsize=12, fontweight="bold")
    ax1.axis("off")

    # Summary box with pipeline-run stats
    cal_sym = "PASS" if result.calibration_passed else "FAIL"
    img_sym = "PASS" if result.imaging_passed else "FAIL"
    lines = [
        r"$\sigma_{\rm amp}$ = 5%",
        r"$\sigma_{\phi}$ = 5\textdegree per antenna",
        "",
        f"Tiles:       {result.n_tiles}",
        f"Cal passed:  {cal_sym}",
        f"Img passed:  {img_sym}",
    ]
    # Use plain text version (no LaTeX) for robustness
    lines_plain = [
        "sigma_amp  = 5%",
        "sigma_phi  = 5 deg/antenna",
        "",
        f"Tiles:       {result.n_tiles}",
        f"Cal passed:  {cal_sym}",
        f"Img passed:  {img_sym}",
    ]
    text_body = "\n".join(lines_plain)
    ax1.text(
        0.5, 0.55, text_body,
        ha="center", va="center",
        transform=ax1.transAxes,
        fontsize=11,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="0.95", edgecolor="0.6", linewidth=1.5),
        linespacing=1.7,
    )
    # Errors list
    if result.errors:
        ax1.text(
            0.5, 0.06,
            f"{len(result.errors)} pipeline error(s)",
            ha="center", va="bottom",
            transform=ax1.transAxes,
            fontsize=9, color="firebrick", fontweight="bold",
        )

    # ── Panel 2: Calibration phase improvement ─────────────────────────────
    ax2 = axes[1]
    ax2.set_title("(2) Calibration", fontsize=12, fontweight="bold")

    # Try to read pre/post calibration phase stats from tile-0 MS
    pre_phase_std = post_phase_std = None
    try:
        import casacore.tables as ct
        ms_path = result.work_dir / "tile_00" / "tile_00.ms"
        if ms_path.exists():
            with ct.table(str(ms_path), readonly=True, ack=False) as t:
                raw  = t.getcol("DATA")
                corr = t.getcol("CORRECTED_DATA")
                ant1 = t.getcol("ANTENNA1")
                ant2 = t.getcol("ANTENNA2")
            cross = ant1 != ant2
            pre_phase_std  = float(np.degrees(np.angle(raw[cross, :, 0]).std()))
            post_phase_std = float(np.degrees(np.angle(corr[cross, :, 0]).std()))
    except Exception:
        pass

    if pre_phase_std is not None and post_phase_std is not None:
        bar_labels = ["Pre-cal\n(DATA)", "Post-cal\n(CORR)"]
        bar_vals   = [pre_phase_std, post_phase_std]
        bar_colors = ["#c0392b", "#27ae60"]
        bars = ax2.bar(
            bar_labels, bar_vals,
            color=bar_colors, width=0.5,
            edgecolor="black", linewidth=0.8,
        )
        ax2.set_ylabel("Phase scatter (deg)", fontsize=10)
        ax2.set_ylim(0, max(pre_phase_std * 1.3, 5))
        ax2.tick_params(labelsize=10)
        for bar, val in zip(bars, bar_vals):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + pre_phase_std * 0.05,
                f"{val:.1f} deg",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                color="black",
            )
    else:
        # Fallback text display
        status = "PASSED" if result.calibration_passed else "FAILED"
        color  = "#27ae60" if result.calibration_passed else "#c0392b"
        ax2.text(
            0.5, 0.5, status,
            ha="center", va="center",
            transform=ax2.transAxes,
            color=color, fontsize=16, fontweight="bold",
        )
        ax2.axis("off")

    # ── Panel 3: WSClean restored tile image ──────────────────────────────
    ax3 = axes[2]
    ax3.set_title("(3) WSClean Restored (tile 0)", fontsize=12, fontweight="bold")

    if tile_restored_path and Path(tile_restored_path).exists():
        _show_fits_image(ax3, tile_restored_path, source_positions, cmap="gray")
    else:
        ax3.text(
            0.5, 0.5, "Image not available",
            ha="center", va="center", transform=ax3.transAxes,
            fontsize=10, color="0.5",
        )
        ax3.axis("off")

    # ── Panel 4: Epoch mosaic ──────────────────────────────────────────────
    ax4 = axes[3]
    ax4.set_title("(4) Epoch Mosaic", fontsize=12, fontweight="bold")

    if result.mosaic_path and result.mosaic_path.exists():
        _show_fits_image(ax4, result.mosaic_path, source_positions, cmap="gray")
    else:
        ax4.text(
            0.5, 0.5, "Mosaic not available",
            ha="center", va="center", transform=ax4.transAxes,
            fontsize=10, color="0.5",
        )
        ax4.axis("off")

    # ── Panel 5: Flux recovery ────────────────────────────────────────────
    ax5 = axes[4]
    ax5.set_title("(5) Flux Recovery", fontsize=12, fontweight="bold")

    if result.source_results:
        ids       = [r.source_id for r in result.source_results]
        injected  = [r.injected_flux_jy for r in result.source_results]
        recovered = [
            r.recovered_flux_jy if not np.isnan(r.recovered_flux_jy) else 0.0
            for r in result.source_results
        ]
        passed_flags = [r.passed for r in result.source_results]

        x = np.arange(len(ids))
        w = 0.35
        ax5.bar(
            x - w / 2, injected, w,
            label="Injected", color="steelblue",
            edgecolor="black", linewidth=0.8,
        )
        ax5.bar(
            x + w / 2, recovered, w,
            label="Recovered",
            color=["#27ae60" if p else "#c0392b" for p in passed_flags],
            edgecolor="black", linewidth=0.8,
        )
        ax5.set_xticks(x)
        # Truncate long source IDs to keep labels readable
        short_ids = [sid[-8:] if len(sid) > 8 else sid for sid in ids]
        ax5.set_xticklabels(short_ids, rotation=30, ha="right", fontsize=10)
        ax5.set_ylabel("Flux (Jy)", fontsize=10)
        ax5.tick_params(labelsize=10)
        ax5.legend(fontsize=10, loc="upper right", framealpha=0.9)
        ax5.set_ylim(bottom=0)

        # Summary: N_recovered / N_total
        n_rec = result.n_recovered
        n_tot = len(result.source_results)
        summary_color = "#27ae60" if n_rec == n_tot else "#c0392b"
        ax5.text(
            0.05, 0.97,
            f"Recovered: {n_rec}/{n_tot}",
            ha="left", va="top", transform=ax5.transAxes,
            fontsize=12, fontweight="bold",
            color=summary_color,
        )
    else:
        ax5.text(
            0.5, 0.5, "No photometry results",
            ha="center", va="center", transform=ax5.transAxes,
            fontsize=10, color="0.5",
        )
        ax5.axis("off")

    # ── Finalise ──────────────────────────────────────────────────────────
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Pipeline diagnostic report written to %s", output_path)
    return output_path


def _show_fits_image(
    ax: plt.Axes,
    fits_path: Path | str,
    source_positions: list[tuple[float, float]],
    cmap: str = "gray",
) -> None:
    """Display a 2-D FITS image on *ax* with optional source overlays.

    Uses a symmetric ±99th-percentile colour scale.  Source positions are
    projected via the WCS and marked with a red '+' cross.
    """
    from astropy.io import fits as astrofits
    from astropy.wcs import WCS

    fits_path = Path(fits_path)
    with astrofits.open(str(fits_path)) as hdul:
        data = np.squeeze(hdul[0].data).astype(float)
        wcs = WCS(hdul[0].header).celestial

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        ax.text(0.5, 0.5, "empty image", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="0.5")
        ax.axis("off")
        return

    vmax = float(np.nanpercentile(np.abs(finite), 99.5))
    vmax = max(vmax, 1e-6)

    im = ax.imshow(
        data,
        origin="lower",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_xlabel("RA [px]", fontsize=10)
    ax.set_ylabel("Dec [px]", fontsize=10)
    ax.tick_params(labelsize=9)

    # Overlay source markers
    for ra_deg, dec_deg in source_positions:
        try:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
            px, py = wcs.world_to_pixel(c)
            px, py = float(px), float(py)
            if 0 <= px < data.shape[1] and 0 <= py < data.shape[0]:
                ax.plot(px, py, "+", color="red", ms=14, mew=2.5)
                ax.plot(px, py, "o", color="none", ms=18, mew=1.5,
                        markeredgecolor="red", alpha=0.7)
        except Exception:
            pass

    # Compact colour bar
    try:
        cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        cb.ax.tick_params(labelsize=9)
        cb.set_label("Jy/beam", fontsize=10)
    except Exception:
        pass
