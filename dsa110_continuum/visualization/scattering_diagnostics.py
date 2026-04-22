"""Scattering transform QA diagnostic plots for the DSA-110 continuum pipeline.

Provides two visualization functions:

- plot_scattering_overview  — spatial heatmap of per-patch scores for one mosaic
- plot_patch_coefficients   — lollipop + delta bar chart for a single flagged patch

Both functions:
- Accept outputs from dsa110_continuum.qa.scattering_qa (no torch/scattering dependency)
- Apply PlotStyle.PUBLICATION (SciencePlots ["science", "notebook"]) via FigureConfig
- Write PNG to the caller-supplied output path
- Degrade gracefully: scienceplots absent -> plain matplotlib style
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from dsa110_continuum.visualization.config import FigureConfig, PlotStyle

log = logging.getLogger(__name__)

# Threshold constants — mirrors scattering_qa._SCORE_WARN / _SCORE_FAIL
_SCORE_WARN: float = 0.85
_SCORE_FAIL: float = 0.70


def _setup_matplotlib() -> None:
    """Configure matplotlib for headless rendering (idempotent)."""
    import matplotlib
    matplotlib.use("Agg")


def plot_scattering_overview(
    result,
    output_path: str | Path,
    config: FigureConfig | None = None,
) -> None:
    """Write a spatial heatmap of per-patch scattering scores to *output_path*.

    Parameters
    ----------
    result : ScatteringQAResult
        Output of check_tile_scattering().
    output_path : str or Path
        Destination PNG path. Parent directory is created if absent.
    config : FigureConfig or None
        Plot configuration. Defaults to PlotStyle.PUBLICATION.
    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    if config is None:
        config = FigureConfig(style=PlotStyle.PUBLICATION)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    patches = result.patch_scores

    # Determine mosaic extent from patch bounds
    all_x_max = max(p.x_max for p in patches) if patches else 256
    all_y_max = max(p.y_max for p in patches) if patches else 256

    # Colormap: green (1.0) -> yellow (WARN) -> red (FAIL -> 0)
    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    with config.style_context():
        fig, ax = plt.subplots(figsize=(8, 5))

        for ps in patches:
            score = ps.score
            color = "lightgrey" if math.isnan(score) else cmap(norm(score))
            rect = mpatches.FancyBboxPatch(
                (ps.x_min, ps.y_min),
                ps.x_max - ps.x_min,
                ps.y_max - ps.y_min,
                boxstyle="round,pad=2",
                facecolor=color,
                edgecolor="white",
                linewidth=0.8,
            )
            ax.add_patch(rect)
            label = "NaN" if math.isnan(score) else f"{score:.3f}"
            ax.text(
                (ps.x_min + ps.x_max) / 2,
                (ps.y_min + ps.y_max) / 2,
                label,
                ha="center", va="center",
                fontsize=8,
                color="black" if not math.isnan(score) and score > 0.5 else "white",
            )

        # Threshold lines on colorbar via dummy ScalarMappable
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Scattering similarity score")
        cbar.ax.axhline(_SCORE_WARN, color="gold",    linewidth=1.5, linestyle="--", label=f"WARN ({_SCORE_WARN})")
        cbar.ax.axhline(_SCORE_FAIL, color="crimson", linewidth=1.5, linestyle="--", label=f"FAIL ({_SCORE_FAIL})")
        cbar.ax.legend(loc="lower left", fontsize=7, framealpha=0.7)

        gate_color = {"PASS": "green", "WARN": "orange", "FAIL": "red"}.get(result.gate, "grey")
        ax.set_title(
            f"Scattering QA — gate: {result.gate}  "
            f"(median={result.median_score:.3f}, min={result.min_score:.3f})",
            color=gate_color,
            fontsize=10,
        )
        ax.set_xlim(0, all_x_max)
        ax.set_ylim(0, all_y_max)
        ax.set_xlabel("Mosaic x (pixels)")
        ax.set_ylabel("Mosaic y (pixels)")
        ax.set_aspect("equal")
        ax.invert_yaxis()  # FITS convention: y=0 at top

        fig.tight_layout()
        fig.savefig(str(output_path), dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)

    log.info("Scattering overview saved to %s", output_path)


def plot_patch_coefficients(
    co_orig: np.ndarray,
    co_syn: np.ndarray,
    patch_score,
    output_path: str | Path,
    config: FigureConfig | None = None,
) -> None:
    """Write a two-panel coefficient diagnostic for one flagged patch.

    Left panel: lollipop chart of co_orig (data) vs co_syn (reference).
    Right panel: |co_orig - co_syn| delta bar chart, highlighting indices
    where delta > 2sigma in red.

    Parameters
    ----------
    co_orig : np.ndarray, shape (N,)
        Scattering covariance vector for the original patch.
    co_syn : np.ndarray, shape (N,)
        Scattering covariance vector for the synthesized reference.
    patch_score : PatchScore
        Used for the figure title (tile_name and score).
    output_path : str or Path
        Destination PNG path. Parent directory is created if absent.
    config : FigureConfig or None
        Plot configuration. Defaults to PlotStyle.PUBLICATION.
    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.PUBLICATION)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    indices = np.arange(len(co_orig))
    delta = np.abs(co_orig - co_syn)
    delta_mean = float(np.mean(delta))
    delta_std  = float(np.std(delta))
    anomalous = delta > (delta_mean + 2 * delta_std)

    with config.style_context():
        fig, (ax_coef, ax_delta) = plt.subplots(1, 2, figsize=(12, 4))

        # --- Left: lollipop chart ---
        ax_coef.vlines(indices, 0, co_orig, colors="steelblue",  linewidth=0.6, alpha=0.7, label=r"data ($c_\mathrm{orig}$)")
        ax_coef.vlines(indices, 0, co_syn,  colors="darkorange", linewidth=0.6, alpha=0.5, label=r"reference ($c_\mathrm{syn}$)")
        ax_coef.plot(indices, co_orig, "o", color="steelblue",  markersize=2, alpha=0.8)
        ax_coef.plot(indices, co_syn,  "o", color="darkorange", markersize=2, alpha=0.6)
        ax_coef.set_xlabel("Coefficient index")
        ax_coef.set_ylabel("Normalized value")
        ax_coef.set_title("Scattering covariance coefficients")
        ax_coef.legend(fontsize=8)
        ax_coef.axhline(0, color="grey", linewidth=0.5, linestyle=":")

        # --- Right: delta bar chart ---
        bar_colors = ["crimson" if a else "steelblue" for a in anomalous]
        ax_delta.bar(indices, delta, color=bar_colors, width=1.0, alpha=0.8)
        thresh_line = delta_mean + 2 * delta_std
        ax_delta.axhline(thresh_line, color="crimson",
                         linewidth=1.0, linestyle="--", label=r"mean + 2$\sigma$")
        ax_delta.set_xlabel("Coefficient index")
        ax_delta.set_ylabel(r"$|c_\mathrm{orig} - c_\mathrm{syn}|$")
        ax_delta.set_title(f"Coefficient delta  ({int(anomalous.sum())} anomalous)")
        ax_delta.legend(fontsize=8)

        score_str = f"{patch_score.score:.4f}" if not math.isnan(patch_score.score) else "NaN"
        fig.suptitle(
            f"Patch: {patch_score.tile_name}  —  score={score_str}  "
            f"[x={patch_score.x_min}:{patch_score.x_max}, y={patch_score.y_min}:{patch_score.y_max}]",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(str(output_path), dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)

    log.info("Patch coefficient diagnostic saved to %s", output_path)
