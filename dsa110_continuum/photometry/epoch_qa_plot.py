"""Diagnostic PNG for epoch QA results.

Three-panel figure:
  Left:   DSA/NVSS ratio histogram with [0.8, 1.2] acceptance band
  Centre: Detection completeness bar with 60% threshold
  Right:  Per-tile RMS bar chart with 17.1 mJy/beam limit line

Title shows overall PASS (green) or FAIL (red) verdict.
"""
from __future__ import annotations

import numpy as np
from dsa110_continuum.photometry.epoch_qa import (
    QA_COMPLETENESS_MIN,
    QA_RATIO_HIGH,
    QA_RATIO_LOW,
    QA_RMS_LIMIT_MJY,
    EpochQAResult,
)


def plot_epoch_qa(
    result: EpochQAResult,
    ratios: list[float],
    tile_rms_mjy: list[float],
    out_path: str,
    epoch_label: str = "",
) -> None:
    """Write a three-panel QA diagnostic PNG.

    Parameters
    ----------
    result:
        Output of ``measure_epoch_qa()``.
    ratios:
        Per-source DSA/NVSS flux ratios (used for histogram).
    tile_rms_mjy:
        Per-tile RMS values in mJy/beam (one entry per mosaic tile).
    out_path:
        Destination PNG file path.
    epoch_label:
        Human-readable epoch string shown in the figure title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    verdict_color = "#2ecc71" if result.qa_result == "PASS" else "#e74c3c"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#16213e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#4ec9b0")
        ax.tick_params(colors="#e0e0e0")
        ax.xaxis.label.set_color("#e0e0e0")
        ax.yaxis.label.set_color("#e0e0e0")
        ax.title.set_color("#4ec9b0")

    # --- Panel 1: Ratio histogram ---
    ax = axes[0]
    if ratios:
        ax.hist(ratios, bins=20, range=(0, 2), color="#4ec9b0", edgecolor="#1a1a2e", alpha=0.85)
        ax.axvspan(QA_RATIO_LOW, QA_RATIO_HIGH, alpha=0.15, color="#2ecc71", label="Pass band")
        ax.axvline(np.median(ratios), color="#f39c12", linewidth=2,
                   label=f"median={np.median(ratios):.2f}")
        ax.legend(fontsize=8, labelcolor="#e0e0e0", facecolor="#1a1a2e")
    else:
        ax.text(0.5, 0.5, "No detections", ha="center", va="center",
                color="#e74c3c", transform=ax.transAxes)
    ax.set_xlabel("DSA / NVSS flux ratio")
    ax.set_ylabel("N sources")
    ax.set_title(f"Flux scale  [{result.ratio_gate}]")

    # --- Panel 2: Detection completeness ---
    ax = axes[1]
    pct = result.completeness_frac * 100
    bar_color = "#2ecc71" if result.completeness_gate == "PASS" else (
        "#95a5a6" if result.completeness_gate == "SKIP" else "#e74c3c"
    )
    ax.bar(["Completeness"], [pct], color=bar_color, edgecolor="#1a1a2e")
    ax.axhline(QA_COMPLETENESS_MIN * 100, color="#f39c12", linewidth=2,
               linestyle="--", label=f"{QA_COMPLETENESS_MIN * 100:.0f}% threshold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("% recovered")
    ax.legend(fontsize=8, labelcolor="#e0e0e0", facecolor="#1a1a2e")
    n_str = f"{result.n_recovered}/{result.n_catalog}"
    ax.set_title(f"Completeness {n_str}  [{result.completeness_gate}]")

    # --- Panel 3: Per-tile RMS ---
    ax = axes[2]
    if tile_rms_mjy:
        colors = ["#2ecc71" if r <= QA_RMS_LIMIT_MJY else "#e74c3c" for r in tile_rms_mjy]
        ax.bar(range(len(tile_rms_mjy)), tile_rms_mjy, color=colors, edgecolor="#1a1a2e")
        ax.axhline(QA_RMS_LIMIT_MJY, color="#f39c12", linewidth=2,
                   linestyle="--", label=f"Limit {QA_RMS_LIMIT_MJY} mJy")
        ax.legend(fontsize=8, labelcolor="#e0e0e0", facecolor="#1a1a2e")
    else:
        ax.bar([0], [result.mosaic_rms_mjy], color="#4ec9b0", edgecolor="#1a1a2e")
        ax.axhline(QA_RMS_LIMIT_MJY, color="#f39c12", linewidth=2, linestyle="--")
    ax.set_xlabel("Tile index")
    ax.set_ylabel("RMS (mJy/beam)")
    ax.set_title(f"Noise floor  [{result.rms_gate}]")

    # --- Overall title ---
    verdict = result.qa_result
    title = f"Epoch QA — {epoch_label}   Overall: {verdict}"
    fig.suptitle(title, fontsize=13, color=verdict_color, fontweight="bold", y=1.02)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
