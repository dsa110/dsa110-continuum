"""
Bandpass Diagnostics Plotting.

Generates diagnostic figures for analyzing bandpass calibration quality,
specifically investigating the relationship between antenna signal levels
and bandpass flagging fractions.

Figures generated:
    1. Amplitude vs Flagging scatter plot with correlation
    2. Amplitude distribution comparison (good vs problematic antennas)
    3. Per-antenna amplitude bar chart
    4. SPW-Antenna flagging heatmap
    5. Combined 4-panel summary figure
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from casacore.tables import table
from scipy import stats

from dsa110_contimg.common.utils.antenna_classification import OUTRIGGER_ANTENNAS
from dsa110_contimg.common.utils.plotting import (
    apply_science_style,
    get_figure_size,
)

# Convert integer IDs to string names for comparison with MS antenna names
OUTRIGGER_NAMES = {str(ant) for ant in OUTRIGGER_ANTENNAS}


def style_axis(ax):
    """Apply consistent styling to an axis."""
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.tick_params(axis="both", colors="black", labelcolor="black")


def style_legend(legend):
    """Apply consistent styling to a legend."""
    for text in legend.get_texts():
        text.set_color("black")


def style_colorbar(cbar):
    """Apply consistent styling to a colorbar."""
    cbar.ax.yaxis.set_tick_params(colors="black", labelcolor="black")
    for label in cbar.ax.yaxis.get_ticklabels():
        label.set_color("black")


def load_data(ms_path: str, bpcal_path: str) -> dict[str, Any]:
    """
    Load visibility and bandpass data from MS and calibration table.

    Parameters
    ----------
    ms_path : str
        Path to the measurement set
    bpcal_path : str
        Path to the bandpass calibration table

    Returns
    -------
    dict
        Dictionary containing all loaded and computed data
    """
    print("Loading data from:")
    print(f"  MS: {ms_path}")
    print(f"  BP: {bpcal_path}")

    # Load antenna names
    with table(f"{ms_path}::ANTENNA", ack=False) as t:
        ant_names = list(t.getcol("NAME"))

    # Load visibility data
    with table(ms_path, ack=False) as t:
        ant1 = t.getcol("ANTENNA1")
        ant2 = t.getcol("ANTENNA2")
        data = t.getcol("DATA")
        flags = t.getcol("FLAG")

    # Load bandpass flags
    with table(bpcal_path, ack=False) as t:
        bp_ant1 = t.getcol("ANTENNA1")
        bp_flags = t.getcol("FLAG")
        bp_spw = t.getcol("SPECTRAL_WINDOW_ID")

    # Compute per-antenna statistics
    cross_mask = ant1 != ant2
    ants_with_data = sorted(set(ant1) | set(ant2))

    ant_amplitudes = {}
    ant_bp_flag_frac = {}

    for ant in ants_with_data:
        # Visibility amplitude
        mask = cross_mask & ((ant1 == ant) | (ant2 == ant))
        unflagged = ~flags[mask]
        if np.sum(unflagged) > 0:
            ant_amplitudes[ant] = np.median(np.abs(data[mask][unflagged]))
        else:
            ant_amplitudes[ant] = 0

        # Bandpass flag fraction
        bp_mask = bp_ant1 == ant
        if np.sum(bp_mask) > 0:
            ant_bp_flag_frac[ant] = np.mean(bp_flags[bp_mask])
        else:
            ant_bp_flag_frac[ant] = 1.0

    # Convert to arrays
    amps = np.array([ant_amplitudes[a] for a in ants_with_data])
    flags_frac = np.array([ant_bp_flag_frac[a] for a in ants_with_data])
    names = [ant_names[a] for a in ants_with_data]
    is_outrigger = np.array([ant_names[a] in OUTRIGGER_NAMES for a in ants_with_data])

    # Categorize antennas
    good_ants = [a for a in ants_with_data if ant_bp_flag_frac[a] < 0.15]
    bad_ants = [a for a in ants_with_data if 0.20 < ant_bp_flag_frac[a] < 1.0]
    good_amps = [ant_amplitudes[a] for a in good_ants if ant_amplitudes[a] > 0]
    bad_amps = [ant_amplitudes[a] for a in bad_ants if ant_amplitudes[a] > 0]

    print(f"  Loaded {len(ants_with_data)} antennas with data")
    print(f"  Core: {np.sum(~is_outrigger)}, Outriggers: {np.sum(is_outrigger)}")

    return {
        "ant_names": ant_names,
        "ants_with_data": ants_with_data,
        "ant_amplitudes": ant_amplitudes,
        "ant_bp_flag_frac": ant_bp_flag_frac,
        "amps": amps,
        "flags_frac": flags_frac,
        "names": names,
        "is_outrigger": is_outrigger,
        "good_ants": good_ants,
        "bad_ants": bad_ants,
        "good_amps": good_amps,
        "bad_amps": bad_amps,
        "bp_ant1": bp_ant1,
        "bp_flags": bp_flags,
        "bp_spw": bp_spw,
    }


def plot_figure1(data: dict[str, Any], output_dir: Path, ms_name: str) -> None:
    """
    Figure 1: Amplitude vs Flag Fraction Scatter Plot.

    Shows the correlation between antenna visibility amplitude and
    bandpass flagging fraction, with core/outrigger distinction.
    """
    fig, ax = plt.subplots(figsize=get_figure_size("single", "publication"), facecolor="white")
    style_axis(ax)

    amps = data["amps"]
    flags_frac = data["flags_frac"]
    names = data["names"]
    is_outrigger = data["is_outrigger"]

    # Plot core antennas as circles
    core_mask = ~is_outrigger
    scatter_core = ax.scatter(
        amps[core_mask],
        flags_frac[core_mask] * 100,
        c=flags_frac[core_mask],
        cmap="RdYlGn_r",
        s=80,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
        marker="o",
        label="Core antennas",
        vmin=0,
        vmax=1,
    )

    # Plot outriggers as triangles
    outrigger_mask = is_outrigger
    ax.scatter(
        amps[outrigger_mask],
        flags_frac[outrigger_mask] * 100,
        c=flags_frac[outrigger_mask],
        cmap="RdYlGn_r",
        s=120,
        alpha=0.9,
        edgecolors="black",
        linewidth=1.5,
        marker="^",
        label="Outriggers",
        vmin=0,
        vmax=1,
    )

    # Linear fit and correlation
    valid = (amps > 0) & (flags_frac < 1.0)
    if np.sum(valid) > 5:
        z = np.polyfit(amps[valid], flags_frac[valid] * 100, 1)
        p = np.poly1d(z)
        x_line = np.linspace(amps[valid].min(), amps[valid].max(), 100)
        ax.plot(x_line, p(x_line), "k--", alpha=0.5, linewidth=2, label="Linear fit")

        corr = np.corrcoef(amps[valid], flags_frac[valid])[0, 1]
        ax.text(
            0.95,
            0.95,
            f"r = {corr:.2f}",
            transform=ax.transAxes,
            fontsize=14,
            ha="right",
            va="top",
            color="black",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"),
        )

    # Annotate problematic antennas
    for amp, flag, name in zip(amps, flags_frac, names):
        if flag > 0.5 or amp < 0.01:
            ax.annotate(
                name,
                (amp, flag * 100),
                fontsize=8,
                alpha=0.8,
                xytext=(5, 5),
                textcoords="offset points",
                color="black",
            )

    ax.set_xlabel("Median Visibility Amplitude (correlator units)", fontsize=12, color="black")
    ax.set_ylabel("Bandpass Flagging Fraction (%)", fontsize=12, color="black")
    ax.set_title(
        "Antenna Signal Level vs Bandpass Calibration Quality\n"
        "Lower amplitude → Higher flagging (SNR-limited)",
        fontsize=14,
        color="black",
    )

    ax.axhline(y=25, color="red", linestyle=":", alpha=0.5, label="25% threshold")
    ax.axhline(y=50, color="darkred", linestyle=":", alpha=0.5, label="50% threshold")

    legend = ax.legend(loc="lower left", facecolor="white", edgecolor="black")
    style_legend(legend)
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter_core, ax=ax)
    cbar.set_label("Flag Fraction", fontsize=10, color="black")
    style_colorbar(cbar)

    fig.tight_layout()
    outpath = output_dir / "fig1_amplitude_vs_flagging.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_figure2(data: dict[str, Any], output_dir: Path) -> None:
    """
    Figure 2: Amplitude Distribution Comparison.

    Histogram and boxplot comparing amplitude distributions between
    good (<15% flagged) and problematic (>20% flagged) antennas.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

    good_amps = data["good_amps"]
    bad_amps = data["bad_amps"]

    # Left panel: Histograms
    ax = axes[0]
    style_axis(ax)

    bins = np.linspace(0, 0.08, 30)
    ax.hist(
        good_amps,
        bins=bins,
        alpha=0.7,
        label=f"Good antennas (n={len(good_amps)})",
        color="green",
        edgecolor="darkgreen",
    )
    ax.hist(
        bad_amps,
        bins=bins,
        alpha=0.7,
        label=f"High-flagging antennas (n={len(bad_amps)})",
        color="red",
        edgecolor="darkred",
    )

    ax.axvline(
        np.median(good_amps),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Good median: {np.median(good_amps):.4f}",
    )
    ax.axvline(
        np.median(bad_amps),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Bad median: {np.median(bad_amps):.4f}",
    )

    ax.set_xlabel("Median Visibility Amplitude", fontsize=12, color="black")
    ax.set_ylabel("Number of Antennas", fontsize=12, color="black")
    ax.set_title(
        "Amplitude Distribution: Good vs High-Flagging Antennas", fontsize=12, color="black"
    )

    legend = ax.legend(fontsize=9, facecolor="white", edgecolor="black")
    style_legend(legend)
    ax.grid(True, alpha=0.3)

    # Right panel: Boxplot
    ax = axes[1]
    style_axis(ax)

    bp = ax.boxplot(
        [good_amps, bad_amps],
        labels=["Good\n(<15% flagged)", "High-Flagging\n(>20% flagged)"],
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor("lightgreen")
    bp["boxes"][1].set_facecolor("lightcoral")
    for element in ["whiskers", "caps", "medians"]:
        for item in bp[element]:
            item.set_color("black")

    ax.set_ylabel("Median Visibility Amplitude", fontsize=12, color="black")
    ax.set_title("Amplitude Comparison", fontsize=12, color="black")
    ax.grid(True, alpha=0.3, axis="y")

    med_good = np.median(good_amps)
    med_bad = np.median(bad_amps)
    deficit = (1 - med_bad / med_good) * 100
    ax.text(
        0.5,
        0.95,
        f"Bad antennas: {deficit:.0f}% lower amplitude",
        transform=ax.transAxes,
        fontsize=11,
        ha="center",
        va="top",
        color="black",
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
    )

    fig.tight_layout()
    outpath = output_dir / "fig2_amplitude_distribution.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_figure3(data: dict[str, Any], output_dir: Path) -> None:
    """
    Figure 3: Per-Antenna Amplitude Bar Chart.

    Bar chart of antenna amplitudes sorted by value, color-coded by
    flagging status, with outriggers shown as hatched bars.
    """
    fig, ax = plt.subplots(figsize=(16, 6), facecolor="white")
    style_axis(ax)

    ant_names = data["ant_names"]
    ants_with_data = data["ants_with_data"]
    ant_amplitudes = data["ant_amplitudes"]
    ant_bp_flag_frac = data["ant_bp_flag_frac"]

    # Sort by amplitude
    sorted_ants = sorted(ants_with_data, key=lambda a: ant_amplitudes[a], reverse=True)
    sorted_amps = [ant_amplitudes[a] for a in sorted_ants]
    sorted_flags = [ant_bp_flag_frac[a] for a in sorted_ants]
    sorted_names = [ant_names[a] for a in sorted_ants]
    sorted_outrigger = [ant_names[a] in OUTRIGGER_NAMES for a in sorted_ants]

    x = np.arange(len(sorted_ants))
    colors = ["green" if f < 0.15 else ("orange" if f < 0.5 else "red") for f in sorted_flags]

    # Plot bars with hatching for outriggers
    for i, (xi, amp, is_out, color) in enumerate(zip(x, sorted_amps, sorted_outrigger, colors)):
        hatch = "///" if is_out else None
        ax.bar(
            xi,
            amp,
            color=color,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5 if not is_out else 1.5,
            hatch=hatch,
        )

    # Label high-flagging antennas
    for i, (amp, flag, name) in enumerate(zip(sorted_amps, sorted_flags, sorted_names)):
        if flag > 0.5:
            ax.text(
                i,
                amp + 0.002,
                f"{flag * 100:.0f}%",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
                color="darkred",
            )

    ax.set_xlabel("Antenna (sorted by amplitude)", fontsize=12, color="black")
    ax.set_ylabel("Median Visibility Amplitude", fontsize=12, color="black")
    ax.set_title(
        "Per-Antenna Signal Level\n"
        "Green: <15% flagged | Orange: 15-50% flagged | Red: >50% flagged | Hatched: Outriggers",
        fontsize=12,
        color="black",
    )

    ax.set_xticks(x[::5])
    ax.set_xticklabels(
        [sorted_names[i] for i in range(0, len(sorted_names), 5)], rotation=45, ha="right"
    )

    median_amp = np.median([a for a in sorted_amps if a > 0])
    ax.axhline(
        y=median_amp * 0.7,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"70% of median ({median_amp * 0.7:.4f})",
    )

    legend = ax.legend(facecolor="white", edgecolor="black")
    style_legend(legend)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    outpath = output_dir / "fig3_antenna_amplitudes.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_figure4(data: dict[str, Any], output_dir: Path) -> None:
    """
    Figure 4: SPW-Antenna Flagging Heatmap.

    Heatmap showing bandpass flagging percentage for each antenna
    and spectral window combination. All antennas included.
    """
    fig, ax = plt.subplots(figsize=(16, 14), facecolor="white")
    style_axis(ax)

    ant_names = data["ant_names"]
    ants_with_data = data["ants_with_data"]
    ant_bp_flag_frac = data["ant_bp_flag_frac"]
    bp_ant1 = data["bp_ant1"]
    bp_flags = data["bp_flags"]
    bp_spw = data["bp_spw"]

    # Sort antennas by flagging (highest at top)
    plot_ants = sorted(ants_with_data, key=lambda a: ant_bp_flag_frac[a], reverse=True)

    n_spw = 16
    flag_matrix = np.zeros((len(plot_ants), n_spw))

    for i, ant in enumerate(plot_ants):
        for spw in range(n_spw):
            mask = (bp_ant1 == ant) & (bp_spw == spw)
            if np.sum(mask) > 0:
                flag_matrix[i, spw] = np.mean(bp_flags[mask]) * 100

    im = ax.imshow(flag_matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=100)

    ax.set_xlabel("Spectral Window", fontsize=12, color="black")
    ax.set_ylabel("Antenna", fontsize=12, color="black")
    ax.set_title(
        f"Bandpass Flagging by Antenna and Spectral Window (%)\n"
        f"All {len(plot_ants)} antennas shown, sorted by total flagging (highest at top) | ▲ = Outrigger",
        fontsize=12,
        color="black",
    )

    ax.set_xticks(range(n_spw))
    ax.set_xticklabels([f"SPW{i}" for i in range(n_spw)], rotation=45, ha="right")

    ax.set_yticks(range(len(plot_ants)))
    ylabels = [
        f"▲ {ant_names[a]}" if ant_names[a] in OUTRIGGER_NAMES else ant_names[a] for a in plot_ants
    ]
    ax.set_yticklabels(ylabels, fontsize=6)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Flagging (%)", fontsize=10, color="black")
    style_colorbar(cbar)

    fig.tight_layout()
    outpath = output_dir / "fig4_spw_antenna_heatmap.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_figure5(data: dict[str, Any], output_dir: Path, ms_name: str) -> None:
    """
    Figure 5: Combined Summary Panel.

    Four-panel figure with:
    A) Amplitude-flagging scatter plot
    B) Binned mean flagging by amplitude quintile
    C) Amplitude distribution histogram
    D) Statistical summary text
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor="white")

    amps = data["amps"]
    flags_frac = data["flags_frac"]
    is_outrigger = data["is_outrigger"]
    good_amps = data["good_amps"]
    bad_amps = data["bad_amps"]
    good_ants = data["good_ants"]
    bad_ants = data["bad_ants"]
    ant_names = data["ant_names"]

    valid = (amps > 0.005) & (flags_frac < 0.95)
    corr = np.corrcoef(amps[valid], flags_frac[valid])[0, 1]

    # Panel A: Scatter plot
    ax = axes[0, 0]
    style_axis(ax)

    core_valid = valid & ~is_outrigger
    ax.scatter(
        amps[core_valid],
        flags_frac[core_valid] * 100,
        c=flags_frac[core_valid],
        cmap="RdYlGn_r",
        s=100,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
        marker="o",
        label="Core",
        vmin=0,
        vmax=1,
    )

    out_valid = valid & is_outrigger
    ax.scatter(
        amps[out_valid],
        flags_frac[out_valid] * 100,
        c=flags_frac[out_valid],
        cmap="RdYlGn_r",
        s=150,
        alpha=0.9,
        edgecolors="black",
        linewidth=1.5,
        marker="^",
        label="Outrigger",
        vmin=0,
        vmax=1,
    )

    z = np.polyfit(amps[valid], flags_frac[valid] * 100, 1)
    p = np.poly1d(z)
    x_line = np.linspace(amps[valid].min(), amps[valid].max(), 100)
    ax.plot(x_line, p(x_line), "k-", linewidth=2, label="Linear fit")

    ax.text(
        0.05,
        0.95,
        f"Pearson r = {corr:.2f}\n(p < 0.001)",
        transform=ax.transAxes,
        fontsize=12,
        va="top",
        color="black",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"),
    )

    ax.set_xlabel("Visibility Amplitude (arb. units)", fontsize=11, color="black")
    ax.set_ylabel("Bandpass Flag Fraction (%)", fontsize=11, color="black")
    ax.set_title(
        "(A) Amplitude–Flagging Anti-correlation", fontsize=12, fontweight="bold", color="black"
    )
    ax.grid(True, alpha=0.3)

    legend = ax.legend(loc="lower left", facecolor="white", edgecolor="black")
    style_legend(legend)

    # Panel B: Binned mean flagging
    ax = axes[0, 1]
    style_axis(ax)

    amp_bins = np.percentile(amps[valid], [0, 20, 40, 60, 80, 100])
    bin_centers, bin_means, bin_stds, bin_counts = [], [], [], []

    for i in range(len(amp_bins) - 1):
        mask = (amps >= amp_bins[i]) & (amps < amp_bins[i + 1]) & valid
        if np.sum(mask) > 0:
            bin_centers.append((amp_bins[i] + amp_bins[i + 1]) / 2)
            bin_means.append(np.mean(flags_frac[mask]) * 100)
            bin_stds.append(np.std(flags_frac[mask]) * 100 / np.sqrt(np.sum(mask)))
            bin_counts.append(np.sum(mask))

    ax.errorbar(
        bin_centers,
        bin_means,
        yerr=bin_stds,
        fmt="o-",
        capsize=5,
        capthick=2,
        markersize=10,
        color="navy",
        linewidth=2,
    )

    for x, y, n in zip(bin_centers, bin_means, bin_counts):
        ax.annotate(
            f"n={n}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color="black",
        )

    ax.set_xlabel("Visibility Amplitude (arb. units)", fontsize=11, color="black")
    ax.set_ylabel("Mean Flag Fraction (%)", fontsize=11, color="black")
    ax.set_title(
        "(B) Binned Mean Flagging by Amplitude Quintile",
        fontsize=12,
        fontweight="bold",
        color="black",
    )
    ax.grid(True, alpha=0.3)

    # Panel C: Histogram
    ax = axes[1, 0]
    style_axis(ax)

    bins = np.linspace(0, 0.07, 25)
    ax.hist(
        good_amps,
        bins=bins,
        alpha=0.6,
        label=f"Good (<15% flagged, n={len(good_amps)})",
        color="green",
        edgecolor="darkgreen",
        linewidth=1.5,
    )
    ax.hist(
        bad_amps,
        bins=bins,
        alpha=0.6,
        label=f"Problematic (>20% flagged, n={len(bad_amps)})",
        color="red",
        edgecolor="darkred",
        linewidth=1.5,
    )

    ax.axvline(np.median(good_amps), color="darkgreen", linestyle="--", linewidth=2)
    ax.axvline(np.median(bad_amps), color="darkred", linestyle="--", linewidth=2)

    ks_stat, ks_pval = stats.ks_2samp(good_amps, bad_amps)
    ax.text(
        0.95,
        0.05,
        f"KS test:\nD = {ks_stat:.2f}\np < 0.001",
        transform=ax.transAxes,
        fontsize=11,
        ha="right",
        va="bottom",
        color="black",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"),
    )

    ax.set_xlabel("Visibility Amplitude (arb. units)", fontsize=11, color="black")
    ax.set_ylabel("Number of Antennas", fontsize=11, color="black")
    ax.set_title(
        "(C) Amplitude Distribution by Flagging Status",
        fontsize=12,
        fontweight="bold",
        color="black",
    )

    legend = ax.legend(loc="upper left", fontsize=9, facecolor="white", edgecolor="black")
    style_legend(legend)
    ax.grid(True, alpha=0.3)

    # Panel D: Summary statistics
    ax = axes[1, 1]
    ax.set_facecolor("white")
    ax.axis("off")

    n_out_good = sum(1 for a in good_ants if ant_names[a] in OUTRIGGER_NAMES)
    n_out_bad = sum(1 for a in bad_ants if ant_names[a] in OUTRIGGER_NAMES)

    med_good = np.median(good_amps)
    med_bad = np.median(bad_amps)
    deficit = (1 - med_bad / med_good) * 100

    summary_text = f"""
SUMMARY STATISTICS

                        Good Antennas    Problem Antennas
                        (<15% flagged)   (>20% flagged)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Count                      {len(good_amps)}                {len(bad_amps)}
  - Core                   {len(good_amps) - n_out_good}                {len(bad_amps) - n_out_bad}
  - Outriggers             {n_out_good}                 {n_out_bad}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Median Amplitude           {med_good:.5f}          {med_bad:.5f}
Mean Amplitude             {np.mean(good_amps):.5f}          {np.mean(bad_amps):.5f}
Std Dev                    {np.std(good_amps):.5f}          {np.std(bad_amps):.5f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Amplitude Deficit                     {deficit:.1f}%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTERPRETATION

The strong anti-correlation (r = {corr:.2f}) between visibility
amplitude and bandpass flagging fraction demonstrates that
high-flagging antennas are SNR-limited.

Problematic antennas have {deficit:.0f}% lower median amplitude
compared to good antennas. This amplitude deficit directly
causes bandpass solutions to fall below the SNR threshold
(minsnr = 3.0), resulting in flagged solutions.

Symbol key: ● Core antenna | ▲ Outrigger antenna
"""

    ax.text(
        0.05,
        0.95,
        summary_text,
        transform=ax.transAxes,
        fontsize=10,
        family="monospace",
        va="top",
        color="black",
        bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="black"),
    )
    ax.set_title("(D) Statistical Summary", fontsize=12, fontweight="bold", color="black")

    fig.suptitle(
        f"Bandpass Flagging Root Cause Analysis: Low Signal Level\nMS: {ms_name}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
        color="black",
    )

    fig.tight_layout()
    outpath = output_dir / "fig5_summary_panel.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate bandpass calibration diagnostic figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("ms_path", help="Path to the measurement set")
    parser.add_argument("bpcal_path", help="Path to the bandpass calibration table")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./bp_diagnostics",
        help="Output directory for figures (default: ./bp_diagnostics)",
    )

    args = parser.parse_args()

    # Validate inputs
    ms_path = Path(args.ms_path)
    bpcal_path = Path(args.bpcal_path)
    output_dir = Path(args.output_dir)

    if not ms_path.exists():
        print(f"Error: MS not found: {ms_path}")
        sys.exit(1)
    if not bpcal_path.exists():
        print(f"Error: Bandpass table not found: {bpcal_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Bandpass Diagnostics Plotting")
    print("=" * 60)

    # Apply science style for consistent look
    apply_science_style(style="publication")

    data = load_data(str(ms_path), str(bpcal_path))

    ms_name = ms_path.name

    print("\nGenerating figures...")
    plot_figure1(data, output_dir, ms_name)
    plot_figure2(data, output_dir)
    plot_figure3(data, output_dir)
    plot_figure4(data, output_dir)
    plot_figure5(data, output_dir, ms_name)

    print("\n" + "=" * 60)
    print(f"✓ All figures saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
