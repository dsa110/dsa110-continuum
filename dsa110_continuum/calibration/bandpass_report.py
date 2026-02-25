"""
Bandpass Diagnostics HTML Report Generator.

Generates comprehensive HTML reports for bandpass calibration quality assessment,
including diagnostic figures, statistics, and recommendations.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np

from dsa110_contimg.common.utils.templates import render_template

logger = logging.getLogger(__name__)

# DSA-110 outrigger antennas (>1000m from array center)
OUTRIGGER_NAMES = {
    "105",
    "106",
    "107",
    "108",
    "109",
    "110",
    "111",
    "112",
    "113",
    "114",
    "115",
    "116",
    "117",
}


@dataclass
class BandpassReportData:
    """Container for all data needed to generate the bandpass report."""

    # Identification
    ms_path: str
    bpcal_path: str
    calibrator_name: str = "unknown"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Overall metrics
    flag_fraction: float = 0.0
    n_antennas_total: int = 0
    n_antennas_with_data: int = 0
    n_antennas_good: int = 0  # <15% flagged
    n_antennas_problem: int = 0  # >20% flagged
    n_spw: int = 16
    n_chan: int = 48  # Channels per SPW

    # Per-antenna data
    antenna_names: list[str] = field(default_factory=list)
    antenna_amplitudes: dict[int, float] = field(default_factory=dict)
    antenna_flag_fractions: dict[int, float] = field(default_factory=dict)
    antenna_is_outrigger: dict[int, bool] = field(default_factory=dict)
    antenna_positions: np.ndarray | None = None  # (n_ant, 3) ECEF positions

    # Statistics
    amplitude_flagging_correlation: float = 0.0
    good_antenna_median_amp: float = 0.0
    problem_antenna_median_amp: float = 0.0
    amplitude_deficit_pct: float = 0.0
    ks_statistic: float = 0.0
    ks_pvalue: float = 0.0

    # SPW-antenna flag matrix
    spw_antenna_flags: np.ndarray | None = None

    # Bandpass solutions: Dict[spw] -> Dict[ant] -> (amp_array, phase_array, flag_array)
    # Each array has shape (n_chan,) for the first polarization
    bandpass_solutions: dict[int, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]] = field(
        default_factory=dict
    )

    # Raw visibility spectra: Dict[spw] -> Dict[ant] -> (amp_array, flag_frac_array)
    # Each array has shape (n_chan,) showing median amplitude and flag fraction per channel
    raw_visibility_spectra: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = field(
        default_factory=dict
    )

    # SPW frequency information: Dict[spw] -> (freq_array_hz, chan_width_hz)
    # freq_array_hz has shape (n_chan,) with center frequency of each channel in Hz
    spw_frequencies: dict[int, tuple[np.ndarray, float]] = field(default_factory=dict)

    # Diagnostic results
    flagging_pattern: str = "unknown"
    root_cause: str = "unknown"
    severity: str = "unknown"
    recommendations: list[str] = field(default_factory=list)

    # Figure paths (if saved to disk) or base64 encoded images
    figure_paths: dict[str, str] = field(default_factory=dict)
    figure_base64: dict[str, str] = field(default_factory=dict)


def load_bandpass_report_data(
    ms_path: str,
    bpcal_path: str,
    calibrator_name: str = "unknown",
) -> BandpassReportData:
    """
    Load all data needed to generate the bandpass diagnostics report.

    Parameters
    ----------
    ms_path : str
        Path to the measurement set
    bpcal_path : str
        Path to the bandpass calibration table
    calibrator_name : str
        Name of the calibrator source

    Returns
    -------
    BandpassReportData
        Container with all report data
    """
    from casacore.tables import table
    from scipy import stats as scipy_stats

    report = BandpassReportData(
        ms_path=ms_path,
        bpcal_path=bpcal_path,
        calibrator_name=calibrator_name,
    )

    # Try to extract calibrator name from MS filename if not provided
    if calibrator_name == "unknown":
        # Format: YYYY-MM-DDTHH:MM:SS_cal.ms
        ms_basename = Path(ms_path).stem
        if ms_basename.endswith("_cal"):
            report.calibrator_name = ms_basename  # Use the timestamp as identifier
        else:
            report.calibrator_name = ms_basename

    # Load antenna names and positions
    with table(f"{ms_path}::ANTENNA", ack=False) as t:
        report.antenna_names = list(t.getcol("NAME"))
        report.n_antennas_total = len(report.antenna_names)
        report.antenna_positions = t.getcol("POSITION")  # (n_ant, 3) ECEF

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

    # Compute overall flag fraction
    report.flag_fraction = float(np.mean(bp_flags))

    # Get unique SPWs
    unique_spw = sorted(set(bp_spw))
    report.n_spw = len(unique_spw)

    # Load SPW frequency information
    with table(f"{ms_path}::SPECTRAL_WINDOW", ack=False) as t:
        chan_freq = t.getcol("CHAN_FREQ")  # (n_spw, n_chan) in Hz
        chan_width = t.getcol("CHAN_WIDTH")  # (n_spw, n_chan) in Hz

    for spw in unique_spw:
        if spw < chan_freq.shape[0]:
            # Store frequency array and typical channel width for this SPW
            freq_hz = chan_freq[spw, :]
            width_hz = np.abs(chan_width[spw, 0])  # Use first channel width
            report.spw_frequencies[spw] = (freq_hz, width_hz)

    # Compute per-antenna statistics
    cross_mask = ant1 != ant2
    ants_with_data = sorted(set(ant1) | set(ant2))
    report.n_antennas_with_data = len(ants_with_data)

    for ant in ants_with_data:
        # Visibility amplitude
        mask = cross_mask & ((ant1 == ant) | (ant2 == ant))
        unflagged = ~flags[mask]
        if np.sum(unflagged) > 0:
            report.antenna_amplitudes[ant] = float(np.median(np.abs(data[mask][unflagged])))
        else:
            report.antenna_amplitudes[ant] = 0.0

        # Bandpass flag fraction
        bp_mask = bp_ant1 == ant
        if np.sum(bp_mask) > 0:
            report.antenna_flag_fractions[ant] = float(np.mean(bp_flags[bp_mask]))
        else:
            report.antenna_flag_fractions[ant] = 1.0

        # Outrigger status
        report.antenna_is_outrigger[ant] = report.antenna_names[ant] in OUTRIGGER_NAMES

    # Build SPW-antenna flag matrix
    report.spw_antenna_flags = np.zeros((len(ants_with_data), report.n_spw))
    for i, ant in enumerate(ants_with_data):
        for j, spw in enumerate(unique_spw):
            mask = (bp_ant1 == ant) & (bp_spw == spw)
            if np.sum(mask) > 0:
                report.spw_antenna_flags[i, j] = float(np.mean(bp_flags[mask]))

    # Categorize antennas
    good_ants = [a for a in ants_with_data if report.antenna_flag_fractions[a] < 0.15]
    problem_ants = [a for a in ants_with_data if 0.20 < report.antenna_flag_fractions[a] < 1.0]
    report.n_antennas_good = len(good_ants)
    report.n_antennas_problem = len(problem_ants)

    # Compute statistics
    good_amps = [
        report.antenna_amplitudes[a] for a in good_ants if report.antenna_amplitudes[a] > 0
    ]
    problem_amps = [
        report.antenna_amplitudes[a] for a in problem_ants if report.antenna_amplitudes[a] > 0
    ]

    if good_amps:
        report.good_antenna_median_amp = float(np.median(good_amps))
    if problem_amps:
        report.problem_antenna_median_amp = float(np.median(problem_amps))

    if report.good_antenna_median_amp > 0 and report.problem_antenna_median_amp > 0:
        report.amplitude_deficit_pct = (
            1 - report.problem_antenna_median_amp / report.good_antenna_median_amp
        ) * 100

    # Correlation
    amps = np.array([report.antenna_amplitudes[a] for a in ants_with_data])
    flags_arr = np.array([report.antenna_flag_fractions[a] for a in ants_with_data])
    valid = (amps > 0.005) & (flags_arr < 0.95)

    if np.sum(valid) > 5:
        report.amplitude_flagging_correlation = float(
            np.corrcoef(amps[valid], flags_arr[valid])[0, 1]
        )

    # KS test
    if len(good_amps) > 2 and len(problem_amps) > 2:
        ks_result = scipy_stats.ks_2samp(good_amps, problem_amps)
        report.ks_statistic = float(ks_result.statistic)
        report.ks_pvalue = float(ks_result.pvalue)

    # Load raw visibility spectra per antenna/SPW
    # This shows the input data quality before bandpass calibration
    with table(ms_path, ack=False) as t:
        ms_ant1 = t.getcol("ANTENNA1")
        ms_ant2 = t.getcol("ANTENNA2")
        ms_data = t.getcol("DATA")  # (nrow, nchan, npol)
        ms_flags = t.getcol("FLAG")
        ms_ddid = t.getcol("DATA_DESC_ID")  # SPW index

    cross_mask = ms_ant1 != ms_ant2
    n_chan = ms_data.shape[1] if ms_data.ndim >= 2 else 48

    for spw in unique_spw:
        report.raw_visibility_spectra[spw] = {}
        spw_mask = (ms_ddid == spw) & cross_mask

        for ant in ants_with_data:
            ant_mask = spw_mask & ((ms_ant1 == ant) | (ms_ant2 == ant))
            n_baselines = np.sum(ant_mask)

            if n_baselines > 0:
                # Get data and flags for this antenna/spw
                ant_data = ms_data[ant_mask]  # (n_bl, nchan, npol)
                ant_flags = ms_flags[ant_mask]

                # Compute median amplitude per channel (first pol, unflagged)
                amp_per_chan = np.zeros(n_chan)
                flag_frac_per_chan = np.zeros(n_chan)

                for ch in range(n_chan):
                    ch_data = ant_data[:, ch, 0]  # First polarization
                    ch_flags = ant_flags[:, ch, 0]
                    unflagged = ~ch_flags
                    flag_frac_per_chan[ch] = 1.0 - np.mean(unflagged)

                    if np.sum(unflagged) > 0:
                        amp_per_chan[ch] = np.median(np.abs(ch_data[unflagged]))
                    else:
                        amp_per_chan[ch] = np.nan

                report.raw_visibility_spectra[spw][ant] = (amp_per_chan, flag_frac_per_chan)

    # Load actual bandpass solutions (amplitude, phase, flags per channel)
    with table(bpcal_path, ack=False) as t:
        bp_cparam = t.getcol("CPARAM")  # Complex gains: shape (nrow, nchan, npol)
        bp_flag = t.getcol("FLAG")  # Flags: shape (nrow, nchan, npol)
        bp_ant = t.getcol("ANTENNA1")
        bp_spw = t.getcol("SPECTRAL_WINDOW_ID")

    # Determine number of channels
    if bp_cparam.ndim == 3:
        report.n_chan = bp_cparam.shape[1]
    else:
        report.n_chan = 48  # Default

    # Organize by SPW -> antenna -> (amp, phase, flag) for first polarization
    for spw in unique_spw:
        report.bandpass_solutions[spw] = {}
        for ant in ants_with_data:
            mask = (bp_ant == ant) & (bp_spw == spw)
            if np.sum(mask) > 0:
                # Get the first matching row (should be only one per ant/spw)
                idx = np.where(mask)[0][0]
                cparam = bp_cparam[idx, :, 0]  # First polarization
                flag = bp_flag[idx, :, 0]

                amp = np.abs(cparam)
                phase = np.angle(cparam, deg=True)

                report.bandpass_solutions[spw][ant] = (amp, phase, flag)

    # Determine flagging pattern and recommendations
    report.flagging_pattern, report.root_cause, report.severity, report.recommendations = (
        _analyze_flagging_and_recommend(report)
    )

    return report


def _analyze_flagging_and_recommend(report: BandpassReportData) -> tuple[str, str, str, list[str]]:
    """Analyze flagging pattern and generate recommendations."""
    pattern = "unknown"
    root_cause = "unknown"
    severity = "low"
    recommendations = []

    flag_pct = report.flag_fraction * 100

    # Determine severity
    if flag_pct < 5:
        severity = "low"
    elif flag_pct < 15:
        severity = "medium"
    elif flag_pct < 30:
        severity = "high"
    else:
        severity = "critical"

    # Analyze correlation
    if report.amplitude_flagging_correlation < -0.5:
        pattern = "amplitude_correlated"
        root_cause = "Low signal level causing SNR-limited flagging"
        recommendations.append("Apply pre-bandpass phase correction to improve coherence")
        recommendations.append("Check that data is coherently phased to calibrator position")
        recommendations.append(
            f"Consider flagging antennas with >50% flagging ({report.n_antennas_problem} problematic)"
        )

    # Check for antenna-specific issues
    n_100pct_flagged = sum(1 for f in report.antenna_flag_fractions.values() if f >= 0.99)
    if n_100pct_flagged > 0:
        if pattern == "unknown":
            pattern = "antenna_specific"
            root_cause = f"{n_100pct_flagged} antennas with 100% flagging (offline or no signal)"
        recommendations.append(f"Flag {n_100pct_flagged} offline antennas before imaging")

    # Check outrigger performance
    outrigger_flags = [
        report.antenna_flag_fractions[a]
        for a in report.antenna_flag_fractions
        if report.antenna_is_outrigger.get(a, False)
    ]
    if outrigger_flags:
        outrigger_mean_flag = np.mean(outrigger_flags)
        if outrigger_mean_flag > 0.5:
            recommendations.append(
                f"Outriggers have {outrigger_mean_flag * 100:.0f}% mean flagging - consider core-only imaging"
            )

    # SPW analysis
    if report.spw_antenna_flags is not None:
        spw_mean_flags = np.mean(report.spw_antenna_flags, axis=0)
        high_flag_spws = np.where(spw_mean_flags > 0.5)[0]
        if len(high_flag_spws) > 0:
            if pattern == "unknown":
                pattern = "spw_specific"
                root_cause = f"SPWs {list(high_flag_spws)} have >50% flagging"
            recommendations.append(
                f"Consider flagging SPWs {list(high_flag_spws)} (edge/RFI affected)"
            )

    if not recommendations:
        if flag_pct < 5:
            recommendations.append("Calibration quality is good - no action needed")
        else:
            recommendations.append("Review calibration setup and data quality")

    return pattern, root_cause, severity, recommendations


def generate_figures_base64(
    report: BandpassReportData,
    save_pngs_dir: str | None = None,
    image_prefix: str = "bandpass",
) -> dict[str, str]:
    """
    Generate all diagnostic figures and return as base64-encoded PNGs.

    This embeds figures directly in the HTML report without requiring
    separate image files.

    Parameters
    ----------
    report : BandpassReportData
        Report data container
    save_pngs_dir : Optional[str]
        If provided, save PNG files to this directory in addition to embedding
    image_prefix : str
        Prefix for saved PNG filenames (default: "bandpass")

    Returns
    -------
    Dict[str, str]
        Dictionary mapping figure names to base64-encoded PNG data
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Apply science + notebook style (auto-imported from package initialization)
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "notebook"])
    except (ImportError, OSError):
        logger.debug("SciencePlots not available; using matplotlib defaults")

    figures = {}

    # Create PNG output directory if specified
    if save_pngs_dir:
        png_dir = Path(save_pngs_dir)
        png_dir.mkdir(parents=True, exist_ok=True)

    # Get arrays for plotting
    ants_with_data = sorted(report.antenna_amplitudes.keys())
    amps = np.array([report.antenna_amplitudes[a] for a in ants_with_data])
    flags_frac = np.array([report.antenna_flag_fractions[a] for a in ants_with_data])
    names = [report.antenna_names[a] for a in ants_with_data]
    is_outrigger = np.array([report.antenna_is_outrigger.get(a, False) for a in ants_with_data])

    # Helper to convert figure to base64 (and optionally save to file)
    def fig_to_base64(fig, figure_name: str = ""):
        # Save to file if directory specified
        if save_pngs_dir and figure_name:
            png_path = png_dir / f"{image_prefix}_{figure_name}.png"
            fig.savefig(png_path, format="png", dpi=120, bbox_inches="tight", facecolor="white")
            logger.info("Saved PNG: %s", png_path)

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return b64

    def style_axis(ax):
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_color("black")
        ax.tick_params(axis="both", colors="black", labelcolor="black")

    def style_legend(legend):
        for text in legend.get_texts():
            text.set_color("black")

    # =========================================================================
    # Figure 1: Amplitude vs Flagging Scatter
    # =========================================================================
    fig1, ax = plt.subplots(figsize=(9, 7), facecolor="white")
    style_axis(ax)

    core_mask = ~is_outrigger
    scatter_core = ax.scatter(
        amps[core_mask],
        flags_frac[core_mask] * 100,
        c=flags_frac[core_mask],
        cmap="RdYlGn_r",
        s=70,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
        marker="o",
        label="Core antennas",
        vmin=0,
        vmax=1,
    )

    outrigger_mask = is_outrigger
    ax.scatter(
        amps[outrigger_mask],
        flags_frac[outrigger_mask] * 100,
        c=flags_frac[outrigger_mask],
        cmap="RdYlGn_r",
        s=100,
        alpha=0.9,
        edgecolors="black",
        linewidth=1.5,
        marker="^",
        label="Outriggers",
        vmin=0,
        vmax=1,
    )

    valid = (amps > 0) & (flags_frac < 1.0)
    if np.sum(valid) > 5:
        z = np.polyfit(amps[valid], flags_frac[valid] * 100, 1)
        p = np.poly1d(z)
        x_line = np.linspace(amps[valid].min(), amps[valid].max(), 100)
        ax.plot(x_line, p(x_line), "k--", alpha=0.5, linewidth=2, label="Linear fit")
        ax.text(
            0.95,
            0.95,
            f"r = {report.amplitude_flagging_correlation:.2f}",
            transform=ax.transAxes,
            fontsize=12,
            ha="right",
            va="top",
            color="black",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"),
        )

    for amp, flag, name in zip(amps, flags_frac, names):
        if flag > 0.5 or amp < 0.01:
            ax.annotate(
                name,
                (amp, flag * 100),
                fontsize=7,
                alpha=0.8,
                xytext=(3, 3),
                textcoords="offset points",
                color="black",
            )

    ax.set_xlabel("Median Visibility Amplitude", fontsize=11, color="black")
    ax.set_ylabel("Bandpass Flagging (%)", fontsize=11, color="black")
    ax.set_title("Antenna Signal Level vs Calibration Quality", fontsize=12, color="black")
    ax.axhline(y=25, color="red", linestyle=":", alpha=0.5, label="25% threshold")
    ax.axhline(y=50, color="darkred", linestyle=":", alpha=0.5, label="50% threshold")
    legend = ax.legend(loc="lower left", facecolor="white", edgecolor="black", fontsize=9)
    style_legend(legend)
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter_core, ax=ax)
    cbar.set_label("Flag Fraction", fontsize=9, color="black")
    cbar.ax.yaxis.set_tick_params(colors="black", labelcolor="black")

    figures["amplitude_vs_flagging"] = fig_to_base64(fig1, "amplitude_vs_flagging")

    # =========================================================================
    # Figure 2: Amplitude Distribution
    # =========================================================================
    good_amps = [
        report.antenna_amplitudes[a]
        for a in ants_with_data
        if report.antenna_flag_fractions[a] < 0.15 and report.antenna_amplitudes[a] > 0
    ]
    bad_amps = [
        report.antenna_amplitudes[a]
        for a in ants_with_data
        if 0.20 < report.antenna_flag_fractions[a] < 1.0 and report.antenna_amplitudes[a] > 0
    ]

    fig2, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="white")

    ax = axes[0]
    style_axis(ax)
    bins = np.linspace(0, max(amps) * 1.1, 25)
    ax.hist(
        good_amps,
        bins=bins,
        alpha=0.7,
        label=f"Good (n={len(good_amps)})",
        color="green",
        edgecolor="darkgreen",
    )
    ax.hist(
        bad_amps,
        bins=bins,
        alpha=0.7,
        label=f"Problematic (n={len(bad_amps)})",
        color="red",
        edgecolor="darkred",
    )
    if good_amps:
        ax.axvline(np.median(good_amps), color="green", linestyle="--", linewidth=2)
    if bad_amps:
        ax.axvline(np.median(bad_amps), color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Visibility Amplitude", fontsize=10, color="black")
    ax.set_ylabel("Count", fontsize=10, color="black")
    ax.set_title("Amplitude Distribution", fontsize=11, color="black")
    legend = ax.legend(fontsize=9, facecolor="white", edgecolor="black")
    style_legend(legend)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    style_axis(ax)
    if good_amps and bad_amps:
        bp = ax.boxplot(
            [good_amps, bad_amps], labels=["Good\n(<15%)", "Problem\n(>20%)"], patch_artist=True
        )
        bp["boxes"][0].set_facecolor("lightgreen")
        bp["boxes"][1].set_facecolor("lightcoral")
        for element in ["whiskers", "caps", "medians"]:
            for item in bp[element]:
                item.set_color("black")
        ax.text(
            0.5,
            0.95,
            f"{report.amplitude_deficit_pct:.0f}% amplitude deficit",
            transform=ax.transAxes,
            fontsize=10,
            ha="center",
            va="top",
            color="black",
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
        )
    ax.set_ylabel("Visibility Amplitude", fontsize=10, color="black")
    ax.set_title("Amplitude Comparison", fontsize=11, color="black")
    ax.grid(True, alpha=0.3, axis="y")

    fig2.tight_layout()
    figures["amplitude_distribution"] = fig_to_base64(fig2, "amplitude_distribution")

    # =========================================================================
    # Figure 3: Per-Antenna Bar Chart
    # =========================================================================
    fig3, ax = plt.subplots(figsize=(14, 5), facecolor="white")
    style_axis(ax)

    sorted_ants = sorted(ants_with_data, key=lambda a: report.antenna_amplitudes[a], reverse=True)
    sorted_amps = [report.antenna_amplitudes[a] for a in sorted_ants]
    sorted_flags = [report.antenna_flag_fractions[a] for a in sorted_ants]
    sorted_names = [report.antenna_names[a] for a in sorted_ants]
    sorted_outrigger = [report.antenna_is_outrigger.get(a, False) for a in sorted_ants]

    x = np.arange(len(sorted_ants))
    colors = ["green" if f < 0.15 else ("orange" if f < 0.5 else "red") for f in sorted_flags]

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

    ax.set_xlabel("Antenna (sorted by amplitude)", fontsize=10, color="black")
    ax.set_ylabel("Visibility Amplitude", fontsize=10, color="black")
    ax.set_title(
        "Per-Antenna Signal Level (Green:<15% | Orange:15-50% | Red:>50% flagged | Hatched:Outrigger)",
        fontsize=10,
        color="black",
    )
    ax.set_xticks(x[::5])
    ax.set_xticklabels(
        [sorted_names[i] for i in range(0, len(sorted_names), 5)], rotation=45, ha="right"
    )
    ax.grid(True, alpha=0.3, axis="y")

    fig3.tight_layout()
    figures["antenna_amplitudes"] = fig_to_base64(fig3, "antenna_amplitudes")

    # =========================================================================
    # Figure 4: SPW-Antenna Heatmap
    # =========================================================================
    if report.spw_antenna_flags is not None:
        # Sort by flagging
        sorted_indices = np.argsort([report.antenna_flag_fractions[a] for a in ants_with_data])[
            ::-1
        ]
        sorted_matrix = report.spw_antenna_flags[sorted_indices, :]
        sorted_ant_names = [report.antenna_names[ants_with_data[i]] for i in sorted_indices]
        sorted_is_out = [
            report.antenna_is_outrigger.get(ants_with_data[i], False) for i in sorted_indices
        ]

        fig4, ax = plt.subplots(figsize=(12, 10), facecolor="white")
        style_axis(ax)

        im = ax.imshow(sorted_matrix * 100, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=100)
        ax.set_xlabel("Spectral Window", fontsize=10, color="black")
        ax.set_ylabel("Antenna", fontsize=10, color="black")
        ax.set_title(
            f"Bandpass Flagging (%) - All {len(ants_with_data)} Antennas",
            fontsize=11,
            color="black",
        )

        ax.set_xticks(range(report.n_spw))
        ax.set_xticklabels([f"SPW{i}" for i in range(report.n_spw)], rotation=45, ha="right")
        ax.set_yticks(range(len(sorted_ant_names)))
        ylabels = [
            f"[O] {n}" if is_out else n for n, is_out in zip(sorted_ant_names, sorted_is_out)
        ]
        ax.set_yticklabels(ylabels, fontsize=5)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Flagging (%)", fontsize=9, color="black")
        cbar.ax.yaxis.set_tick_params(colors="black", labelcolor="black")

        fig4.tight_layout()
        figures["spw_antenna_heatmap"] = fig_to_base64(fig4, "spw_antenna_heatmap")

    # =========================================================================
    # Figure 5: Bandpass Solutions Overview (4x4 grid, one panel per SPW)
    # =========================================================================
    if report.bandpass_solutions:
        n_spw = min(report.n_spw, 16)  # Max 16 panels
        ncols = 4
        nrows = (n_spw + ncols - 1) // ncols

        fig5, axes = plt.subplots(nrows, ncols, figsize=(16, 3 * nrows), facecolor="white")
        axes = np.atleast_2d(axes)

        # Identify good antennas for reference
        good_ants = [a for a in ants_with_data if report.antenna_flag_fractions.get(a, 1.0) < 0.15]

        spw_list = sorted(report.bandpass_solutions.keys())[:n_spw]
        channels = np.arange(report.n_chan)

        for idx, spw in enumerate(spw_list):
            row, col = idx // ncols, idx % ncols
            ax = axes[row, col]
            style_axis(ax)

            spw_data = report.bandpass_solutions[spw]

            # Collect all amplitudes to find median reference
            all_amps = []
            for ant, (amp, phase, flag) in spw_data.items():
                if ant in good_ants and np.sum(~flag) > 0:
                    all_amps.append(amp)

            # Calculate reference (median of good antennas)
            if all_amps:
                ref_amp = np.nanmedian(np.array(all_amps), axis=0)
            else:
                ref_amp = None

            # Track flagged channels for shading
            channel_flag_count = np.zeros(report.n_chan)

            # Plot each antenna
            for ant, (amp, phase, flag) in spw_data.items():
                flag_frac = report.antenna_flag_fractions.get(ant, 0)
                is_out = report.antenna_is_outrigger.get(ant, False)

                # Color by quality
                if flag_frac < 0.15:
                    color = "green"
                    alpha = 0.4
                    zorder = 1
                elif flag_frac < 0.50:
                    color = "orange"
                    alpha = 0.5
                    zorder = 2
                else:
                    color = "red"
                    alpha = 0.6
                    zorder = 3

                # Mask flagged channels
                amp_plot = amp.copy()
                amp_plot[flag] = np.nan
                channel_flag_count += flag.astype(int)

                # Plot line
                lw = 1.5 if is_out else 0.8
                ls = "--" if is_out else "-"
                ax.plot(
                    channels,
                    amp_plot,
                    color=color,
                    alpha=alpha,
                    linewidth=lw,
                    linestyle=ls,
                    zorder=zorder,
                )

            # Shade universally flagged channels (100% of antennas flagged = no valid data)
            flag_frac_per_chan = channel_flag_count / max(len(spw_data), 1)
            for ch in range(report.n_chan):
                if flag_frac_per_chan[ch] >= 0.999:  # Essentially 100%
                    ax.axvspan(ch - 0.5, ch + 0.5, color="red", alpha=0.3, zorder=0)

            # Plot reference line
            if ref_amp is not None:
                ax.plot(
                    channels,
                    ref_amp,
                    "k-",
                    linewidth=2,
                    alpha=0.8,
                    zorder=10,
                    label="Median (good)",
                )

            ax.set_xlim(0, report.n_chan - 1)

            # Get frequency range for title
            if spw in report.spw_frequencies:
                freq_hz, _ = report.spw_frequencies[spw]
                freq_min_mhz = freq_hz[0] / 1e6
                freq_max_mhz = freq_hz[-1] / 1e6
                ax.set_title(
                    f"SPW {spw} ({freq_min_mhz:.0f}-{freq_max_mhz:.0f} MHz)",
                    fontsize=9,
                    fontweight="bold",
                    color="black",
                )
            else:
                ax.set_title(f"SPW {spw}", fontsize=10, fontweight="bold", color="black")

            if row == nrows - 1:
                ax.set_xlabel("Channel", fontsize=9, color="black")
            if col == 0:
                ax.set_ylabel("BP Amplitude", fontsize=9, color="black")

            # Add secondary frequency axis on top row
            if row == 0 and spw in report.spw_frequencies:
                freq_hz, _ = report.spw_frequencies[spw]
                ax2 = ax.twiny()
                ax2.set_xlim(freq_hz[0] / 1e6, freq_hz[-1] / 1e6)
                ax2.set_xlabel("Frequency (MHz)", fontsize=8, color="gray")
                ax2.tick_params(axis="x", labelsize=7, colors="gray")

            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for idx in range(len(spw_list), nrows * ncols):
            row, col = idx // ncols, idx % ncols
            axes[row, col].set_visible(False)

        # Add legend to first subplot
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="green", alpha=0.7, label="Good (<15% flag)"),
            Line2D([0], [0], color="orange", alpha=0.7, label="Moderate (15-50%)"),
            Line2D([0], [0], color="red", alpha=0.7, label="Poor (>50% flag)"),
            Line2D([0], [0], color="black", linewidth=2, label="Reference median"),
            Line2D([0], [0], color="green", linestyle="--", label="Outrigger"),
        ]
        axes[0, -1].legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=8,
            facecolor="white",
            edgecolor="black",
        )

        fig5.suptitle(
            "Bandpass Solutions by SPW (red shading = all antennas flagged)",
            fontsize=12,
            fontweight="bold",
            color="black",
            y=1.02,
        )
        fig5.tight_layout()
        figures["bandpass_solutions"] = fig_to_base64(fig5, "bandpass_solutions")

    # =========================================================================
    # Figure 6: Raw Visibility Spectra (input data before calibration)
    # =========================================================================
    if report.raw_visibility_spectra:
        n_spw = len(report.raw_visibility_spectra)
        ncols = 4
        nrows = (n_spw + ncols - 1) // ncols

        fig6, axes = plt.subplots(nrows, ncols, figsize=(16, 3 * nrows), facecolor="white")
        axes = np.atleast_2d(axes)

        # Identify good antennas for reference
        good_ants = [a for a in ants_with_data if report.antenna_flag_fractions.get(a, 1.0) < 0.15]

        spw_list = sorted(report.raw_visibility_spectra.keys())[:n_spw]
        channels = np.arange(report.n_chan)

        for idx, spw in enumerate(spw_list):
            row, col = idx // ncols, idx % ncols
            ax = axes[row, col]
            style_axis(ax)

            spw_data = report.raw_visibility_spectra.get(spw, {})

            # Collect all amplitudes to find median reference
            all_amps = []
            for ant, (amp, flag_frac) in spw_data.items():
                if ant in good_ants and np.sum(~np.isnan(amp)) > 0:
                    all_amps.append(amp)

            # Calculate reference (median of good antennas)
            if all_amps:
                ref_amp = np.nanmedian(np.array(all_amps), axis=0)
            else:
                ref_amp = None

            # Track flagged channels for shading
            channel_flag_fracs = []

            # Plot each antenna
            for ant, (amp, flag_frac) in spw_data.items():
                ant_flag_frac = report.antenna_flag_fractions.get(ant, 0)
                is_out = report.antenna_is_outrigger.get(ant, False)

                # Color by quality (based on bandpass solution quality)
                if ant_flag_frac < 0.15:
                    color = "green"
                    alpha = 0.4
                    zorder = 1
                elif ant_flag_frac < 0.50:
                    color = "orange"
                    alpha = 0.5
                    zorder = 2
                else:
                    color = "red"
                    alpha = 0.6
                    zorder = 3

                channel_flag_fracs.append(flag_frac)

                # Plot line
                lw = 1.5 if is_out else 0.8
                ls = "--" if is_out else "-"
                ax.plot(
                    channels,
                    amp,
                    color=color,
                    alpha=alpha,
                    linewidth=lw,
                    linestyle=ls,
                    zorder=zorder,
                )

            # Shade heavily flagged channels
            if channel_flag_fracs:
                mean_flag_frac = np.nanmean(np.array(channel_flag_fracs), axis=0)
                for ch in range(report.n_chan):
                    if mean_flag_frac[ch] >= 0.8:  # >80% flagged
                        ax.axvspan(ch - 0.5, ch + 0.5, color="red", alpha=0.3, zorder=0)

            # Plot reference line
            if ref_amp is not None:
                ax.plot(
                    channels,
                    ref_amp,
                    "k-",
                    linewidth=2,
                    alpha=0.8,
                    zorder=10,
                    label="Median (good)",
                )

            ax.set_xlim(0, report.n_chan - 1)

            # Get frequency range for title
            if spw in report.spw_frequencies:
                freq_hz, _ = report.spw_frequencies[spw]
                freq_min_mhz = freq_hz[0] / 1e6
                freq_max_mhz = freq_hz[-1] / 1e6
                ax.set_title(
                    f"SPW {spw} ({freq_min_mhz:.0f}-{freq_max_mhz:.0f} MHz)",
                    fontsize=9,
                    fontweight="bold",
                    color="black",
                )
            else:
                ax.set_title(f"SPW {spw}", fontsize=10, fontweight="bold", color="black")

            if row == nrows - 1:
                ax.set_xlabel("Channel", fontsize=9, color="black")
            if col == 0:
                ax.set_ylabel("Raw Amplitude", fontsize=9, color="black")

            # Add secondary frequency axis on top row
            if row == 0 and spw in report.spw_frequencies:
                freq_hz, _ = report.spw_frequencies[spw]
                ax2 = ax.twiny()
                ax2.set_xlim(freq_hz[0] / 1e6, freq_hz[-1] / 1e6)
                ax2.set_xlabel("Frequency (MHz)", fontsize=8, color="gray")
                ax2.tick_params(axis="x", labelsize=7, colors="gray")

            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for idx in range(len(spw_list), nrows * ncols):
            row, col = idx // ncols, idx % ncols
            axes[row, col].set_visible(False)

        # Add legend to first subplot
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="green", alpha=0.7, label="Good ant"),
            Line2D([0], [0], color="orange", alpha=0.7, label="Moderate ant"),
            Line2D([0], [0], color="red", alpha=0.7, label="Poor ant"),
            Line2D([0], [0], color="black", linewidth=2, label="Reference median"),
            Line2D([0], [0], color="green", linestyle="--", label="Outrigger"),
        ]
        axes[0, -1].legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=8,
            facecolor="white",
            edgecolor="black",
        )

        fig6.suptitle(
            "Raw Visibility Amplitude by SPW (red shading = >80% flagged)",
            fontsize=12,
            fontweight="bold",
            color="black",
            y=1.02,
        )
        fig6.tight_layout()
        figures["raw_visibility_spectra"] = fig_to_base64(fig6, "raw_visibility_spectra")

    # =========================================================================
    # Figure 7: Antenna Map (birds-eye view)
    # =========================================================================
    if report.antenna_positions is not None and len(ants_with_data) > 0:
        fig7, ax = plt.subplots(figsize=(10, 10), facecolor="white")
        style_axis(ax)

        # Convert ECEF to local ENU (East-North-Up) coordinates
        # Use array center as reference
        pos = report.antenna_positions
        ref_pos = np.mean(pos, axis=0)

        # Simplified local tangent plane conversion
        # At OVRO latitude (~37 deg), approximate conversion
        lat_rad = np.radians(37.2339)  # OVRO latitude

        # Compute local offsets (approximate East-North)
        dx = pos[:, 0] - ref_pos[0]
        dy = pos[:, 1] - ref_pos[1]
        dz = pos[:, 2] - ref_pos[2]

        # Rotate to local ENU
        east = -np.sin(np.radians(-118.2818)) * dx + np.cos(np.radians(-118.2818)) * dy
        north = (
            -np.sin(lat_rad) * np.cos(np.radians(-118.2818)) * dx
            - np.sin(lat_rad) * np.sin(np.radians(-118.2818)) * dy
            + np.cos(lat_rad) * dz
        )

        # Plot each antenna
        for ant in range(len(report.antenna_names)):
            if ant not in report.antenna_flag_fractions:
                color = "gray"
                marker = "x"
                size = 50
                alpha = 0.3
            else:
                flag_frac = report.antenna_flag_fractions[ant]
                is_out = report.antenna_is_outrigger.get(ant, False)

                if flag_frac < 0.15:
                    color = "green"
                elif flag_frac < 0.50:
                    color = "orange"
                else:
                    color = "red"

                marker = "^" if is_out else "o"
                size = 120 if is_out else 80
                alpha = 0.8

            if marker == "x":
                ax.scatter(
                    east[ant],
                    north[ant],
                    c=color,
                    marker=marker,
                    s=size,
                    alpha=alpha,
                    linewidth=1.5,
                )
            else:
                ax.scatter(
                    east[ant],
                    north[ant],
                    c=color,
                    marker=marker,
                    s=size,
                    alpha=alpha,
                    edgecolors="black",
                    linewidth=0.5,
                )

            # Label antennas
            name = report.antenna_names[ant]
            ax.annotate(
                name,
                (east[ant], north[ant]),
                fontsize=6,
                alpha=0.7,
                xytext=(2, 2),
                textcoords="offset points",
                color="black",
            )

        ax.set_xlabel("East (m)", fontsize=11, color="black")
        ax.set_ylabel("North (m)", fontsize=11, color="black")
        ax.set_title("Antenna Array Layout - Calibration Quality", fontsize=12, color="black")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Add legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="green",
                markersize=10,
                label="Good (<15% flagged)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="orange",
                markersize=10,
                label="Moderate (15-50%)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=10,
                label="Poor (>50%)",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="gray",
                markersize=10,
                label="Outrigger",
            ),
            Line2D([0], [0], marker="x", color="gray", markersize=8, label="No data"),
        ]
        legend = ax.legend(
            handles=legend_elements,
            loc="upper left",
            fontsize=9,
            facecolor="white",
            edgecolor="black",
        )
        style_legend(legend)

        figures["antenna_map"] = fig_to_base64(fig7, "antenna_map")

    return figures


def generate_altair_figures(report: BandpassReportData) -> dict[str, str]:
    """
    Generate interactive Altair/Vega-Lite figures as embeddable HTML snippets.

    Returns a dict mapping figure name to an HTML div with embedded Vega-Lite spec.
    Each figure gets a unique div ID for proper rendering.
    """
    import altair as alt
    import pandas as pd

    # Disable max rows limit for large spectral datasets
    alt.data_transformers.disable_max_rows()

    figures = {}

    # Shared legend configuration for larger, easier-to-click text
    legend_config = alt.Legend(
        labelFontSize=14, titleFontSize=14, symbolSize=200, labelLimit=300, symbolStrokeWidth=2
    )

    def chart_to_embed_html(chart, div_id):
        """Convert Altair chart to embeddable HTML snippet with vega-embed."""
        spec_json = chart.to_json()
        return f'''<div id="{div_id}" style="width: 100%;"></div>
<script type="text/javascript">
(function() {{
    var spec = {spec_json};
    vegaEmbed('#{div_id}', spec, {{"renderer": "svg", "actions": true}})
        .catch(console.error);
}})();
</script>'''

    # Prepare data for plotting
    ants_with_data = sorted(report.antenna_amplitudes.keys())

    df = pd.DataFrame(
        {
            "antenna_id": ants_with_data,
            "antenna_name": [report.antenna_names[a] for a in ants_with_data],
            "amplitude": [report.antenna_amplitudes[a] for a in ants_with_data],
            "flag_fraction": [report.antenna_flag_fractions[a] for a in ants_with_data],
            "flag_pct": [report.antenna_flag_fractions[a] * 100 for a in ants_with_data],
            "is_outrigger": [report.antenna_is_outrigger.get(a, False) for a in ants_with_data],
            "antenna_type": [
                "Outrigger" if report.antenna_is_outrigger.get(a, False) else "Core"
                for a in ants_with_data
            ],
        }
    )

    # Quality category for coloring
    def get_quality(flag_frac):
        if flag_frac < 0.15:
            return "Good (<15%)"
        elif flag_frac < 0.50:
            return "Moderate (15-50%)"
        else:
            return "Poor (>50%)"

    df["quality"] = df["flag_fraction"].apply(get_quality)

    # Color scale matching our matplotlib colors
    quality_color_scale = alt.Scale(
        domain=["Good (<15%)", "Moderate (15-50%)", "Poor (>50%)"],
        range=["#28a745", "#fd7e14", "#dc3545"],
    )

    # =========================================================================
    # Interactive Antenna Map
    # =========================================================================
    if report.antenna_positions is not None:
        pos = report.antenna_positions
        ref_pos = np.mean(pos, axis=0)

        # Approximate local ENU conversion (OVRO location)
        lat_rad = np.radians(37.2339)
        lon_rad = np.radians(-118.2818)

        dx = pos[:, 0] - ref_pos[0]
        dy = pos[:, 1] - ref_pos[1]
        dz = pos[:, 2] - ref_pos[2]

        east = -np.sin(lon_rad) * dx + np.cos(lon_rad) * dy
        north = (
            -np.sin(lat_rad) * np.cos(lon_rad) * dx
            - np.sin(lat_rad) * np.sin(lon_rad) * dy
            + np.cos(lat_rad) * dz
        )

        # Build dataframe with positions for all antennas
        map_data = []
        for ant in range(len(report.antenna_names)):
            name = report.antenna_names[ant]
            is_out = name in OUTRIGGER_NAMES

            if ant in report.antenna_flag_fractions:
                flag_frac = report.antenna_flag_fractions[ant]
                amp = report.antenna_amplitudes.get(ant, 0)
                quality = get_quality(flag_frac)
                has_data = True
            else:
                flag_frac = 1.0
                amp = 0.0
                quality = "No data"
                has_data = False

            map_data.append(
                {
                    "antenna_id": ant,
                    "antenna_name": name,
                    "east": east[ant],
                    "north": north[ant],
                    "amplitude": amp,
                    "flag_fraction": flag_frac,
                    "flag_pct": flag_frac * 100,
                    "antenna_type": "Outrigger" if is_out else "Core",
                    "quality": quality,
                    "has_data": has_data,
                }
            )

        map_df = pd.DataFrame(map_data)

        # Extended color scale including "No data"
        map_color_scale = alt.Scale(
            domain=["Good (<15%)", "Moderate (15-50%)", "Poor (>50%)", "No data"],
            range=["#28a745", "#fd7e14", "#dc3545", "#6c757d"],
        )

        # Quality filter - click legend to toggle filter (click to select, click again to deselect)
        map_quality_selection = alt.selection_point(
            name="map_quality",
            fields=["quality"],
            bind="legend",
            toggle="true",  # Vega expression: always toggle on click (not just shift+click)
        )

        # Antenna type filter - click legend to toggle filter
        map_type_selection = alt.selection_point(
            name="map_type",
            fields=["antenna_type"],
            bind="legend",
            toggle="true",  # Vega expression: always toggle on click
        )

        # Vega expression: show if selection empty OR datum matches selection
        # This enables proper toggle behavior with multi-select
        map_opacity_expr = (
            "(length(data('map_quality_store')) == 0 || vlSelectionTest('map_quality_store', datum)) && "
            "(length(data('map_type_store')) == 0 || vlSelectionTest('map_type_store', datum))"
        )

        map_chart = (
            alt.Chart(map_df)
            .mark_point(filled=True, stroke="black", strokeWidth=0.5)
            .encode(
                x=alt.X("east:Q", title="East (m)", scale=alt.Scale(zero=False)),
                y=alt.Y("north:Q", title="North (m)", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "quality:N",
                    title="Calibration Quality",
                    scale=map_color_scale,
                    sort=["Good (<15%)", "Moderate (15-50%)", "Poor (>50%)", "No data"],
                    legend=legend_config,
                ),
                shape=alt.Shape(
                    "antenna_type:N",
                    title="Antenna Type",
                    scale=alt.Scale(domain=["Core", "Outrigger"], range=["circle", "triangle-up"]),
                    legend=legend_config,
                ),
                size=alt.condition(
                    alt.datum.antenna_type == "Outrigger", alt.value(150), alt.value(100)
                ),
                opacity=alt.condition(map_opacity_expr, alt.value(1.0), alt.value(0.15)),
                tooltip=[
                    alt.Tooltip("antenna_name:N", title="Antenna"),
                    alt.Tooltip("antenna_type:N", title="Type"),
                    alt.Tooltip("amplitude:Q", title="Amplitude", format=".5f"),
                    alt.Tooltip("flag_pct:Q", title="Flagging %", format=".1f"),
                    alt.Tooltip("quality:N", title="Quality"),
                ],
            )
            .add_params(map_quality_selection, map_type_selection)
            .properties(
                width=500,
                height=500,
                title="Antenna Array Layout - Click legend to filter by Quality or Antenna Type",
            )
            .interactive()
        )

        figures["antenna_map"] = chart_to_embed_html(map_chart, "vega-antenna-map")

    # =========================================================================
    # Interactive Amplitude vs Flagging Scatter
    # =========================================================================
    valid_df = df[df["amplitude"] > 0].copy()

    scatter = (
        alt.Chart(valid_df)
        .mark_point(filled=True, stroke="black", strokeWidth=0.5)
        .encode(
            x=alt.X(
                "amplitude:Q", title="Median Visibility Amplitude", scale=alt.Scale(zero=False)
            ),
            y=alt.Y("flag_pct:Q", title="Bandpass Flagging (%)", scale=alt.Scale(domain=[0, 105])),
            color=alt.Color(
                "quality:N",
                title="Calibration Quality",
                scale=quality_color_scale,
                sort=["Good (<15%)", "Moderate (15-50%)", "Poor (>50%)"],
                legend=legend_config,
            ),
            shape=alt.Shape(
                "antenna_type:N",
                title="Antenna Type",
                scale=alt.Scale(domain=["Core", "Outrigger"], range=["circle", "triangle-up"]),
                legend=legend_config,
            ),
            size=alt.condition(
                alt.datum.antenna_type == "Outrigger", alt.value(150), alt.value(80)
            ),
            tooltip=[
                alt.Tooltip("antenna_name:N", title="Antenna"),
                alt.Tooltip("antenna_type:N", title="Type"),
                alt.Tooltip("amplitude:Q", title="Amplitude", format=".5f"),
                alt.Tooltip("flag_pct:Q", title="Flagging %", format=".1f"),
                alt.Tooltip("quality:N", title="Quality"),
            ],
        )
    )

    # Add threshold lines
    threshold_25 = (
        alt.Chart(pd.DataFrame({"y": [25]}))
        .mark_rule(strokeDash=[5, 5], color="orange", strokeWidth=1.5)
        .encode(y="y:Q")
    )

    threshold_50 = (
        alt.Chart(pd.DataFrame({"y": [50]}))
        .mark_rule(strokeDash=[5, 5], color="red", strokeWidth=1.5)
        .encode(y="y:Q")
    )

    # Add correlation annotation
    corr_text = (
        alt.Chart(
            pd.DataFrame(
                {
                    "x": [valid_df["amplitude"].max() * 0.95],
                    "y": [5],
                    "text": [f"r = {report.amplitude_flagging_correlation:.2f}"],
                }
            )
        )
        .mark_text(align="right", fontSize=14, fontWeight="bold")
        .encode(x="x:Q", y="y:Q", text="text:N")
    )

    amp_flag_chart = (
        (scatter + threshold_25 + threshold_50 + corr_text)
        .properties(width=550, height=400, title="Antenna Signal Level vs Calibration Quality")
        .interactive()
    )

    figures["amplitude_vs_flagging"] = chart_to_embed_html(amp_flag_chart, "vega-amp-vs-flag")

    # =========================================================================
    # Interactive Per-Antenna Bar Chart
    # =========================================================================
    sorted_df = df.sort_values("amplitude", ascending=True).copy()
    sorted_df["rank"] = range(len(sorted_df))

    bars = (
        alt.Chart(sorted_df)
        .mark_bar()
        .encode(
            x=alt.X("amplitude:Q", title="Median Visibility Amplitude"),
            y=alt.Y(
                "antenna_name:N",
                title="Antenna",
                sort=alt.EncodingSortField(field="amplitude", order="ascending"),
            ),
            color=alt.Color(
                "quality:N",
                title="Calibration Quality",
                scale=quality_color_scale,
                sort=["Good (<15%)", "Moderate (15-50%)", "Poor (>50%)"],
            ),
            tooltip=[
                alt.Tooltip("antenna_name:N", title="Antenna"),
                alt.Tooltip("antenna_type:N", title="Type"),
                alt.Tooltip("amplitude:Q", title="Amplitude", format=".5f"),
                alt.Tooltip("flag_pct:Q", title="Flagging %", format=".1f"),
            ],
        )
        .properties(
            width=500,
            height=max(400, len(sorted_df) * 8),
            title="Per-Antenna Signal Levels (sorted by amplitude)",
        )
        .interactive()
    )

    figures["antenna_amplitudes"] = chart_to_embed_html(bars, "vega-antenna-amps")

    # =========================================================================
    # Interactive SPW Bandpass Browser with Filtering
    # =========================================================================
    if report.bandpass_solutions and report.spw_frequencies:
        # Build a long-form dataframe with bandpass data for all SPWs
        bp_rows = []
        ants_with_data = sorted(report.antenna_amplitudes.keys())
        good_ants = [a for a in ants_with_data if report.antenna_flag_fractions.get(a, 1.0) < 0.15]

        # Include all SPWs for full interactivity
        all_spws = sorted(report.bandpass_solutions.keys())

        for spw in all_spws:
            if spw not in report.spw_frequencies:
                continue

            spw_data = report.bandpass_solutions.get(spw, {})
            freq_hz, _ = report.spw_frequencies[spw]
            freq_mhz = freq_hz / 1e6

            for ant, (amp, phase, flag) in spw_data.items():
                ant_name = report.antenna_names[ant]
                ant_flag_frac = report.antenna_flag_fractions.get(ant, 0)
                is_out = report.antenna_is_outrigger.get(ant, False)
                quality = get_quality(ant_flag_frac)
                ant_type = "Outrigger" if is_out else "Core"

                for ch in range(len(amp)):
                    if not flag[ch]:  # Only unflagged channels
                        bp_rows.append(
                            {
                                "spw": spw,
                                "channel": ch,
                                "frequency_mhz": freq_mhz[ch],
                                "antenna_id": ant,
                                "antenna_name": ant_name,
                                "amplitude": amp[ch],
                                "phase_deg": phase[ch],
                                "quality": quality,
                                "antenna_type": ant_type,
                                "is_reference": ant in good_ants,
                            }
                        )

        if bp_rows:
            bp_df = pd.DataFrame(bp_rows)

            # Calculate reference median per SPW/channel and add to main dataframe
            ref_df = (
                bp_df[bp_df["is_reference"]]
                .groupby(["spw", "channel", "frequency_mhz"])
                .agg({"amplitude": "median"})
                .reset_index()
                .rename(columns={"amplitude": "ref_amplitude"})
            )

            # Create reference line data with its own 'layer' field for legend
            ref_line_df = ref_df.copy()
            ref_line_df["layer"] = "Reference Median"

            # Add layer field to main data
            bp_df["layer"] = "Antenna Data"

            # SPW radio buttons on the side - include all SPWs with frequency labels
            spw_list = sorted(bp_df["spw"].unique())
            spw_labels = []
            for s in spw_list:
                if s in report.spw_frequencies:
                    freq_hz, _ = report.spw_frequencies[s]
                    spw_labels.append(
                        f"SPW {s} ({freq_hz[0] / 1e6:.0f}-{freq_hz[-1] / 1e6:.0f} MHz)"
                    )
                else:
                    spw_labels.append(f"SPW {s}")

            spw_radio = alt.binding_radio(
                options=spw_list, labels=spw_labels, name="Spectral Window: "
            )
            spw_selection = alt.selection_point(
                name="bp_spw", fields=["spw"], bind=spw_radio, value=spw_list[0]
            )

            # Quality filter - click legend to toggle filter (click to select, click again to deselect)
            quality_selection = alt.selection_point(
                name="bp_quality",
                fields=["quality"],
                bind="legend",
                toggle="true",  # Vega expression: always toggle on click
            )

            # Antenna type filter - click legend to toggle filter
            type_selection = alt.selection_point(
                name="bp_type",
                fields=["antenna_type"],
                bind="legend",
                toggle="true",  # Vega expression: always toggle on click
            )

            # Reference median toggle - click legend to show/hide
            ref_selection = alt.selection_point(
                name="bp_ref",
                fields=["layer"],
                bind="legend",
                toggle="true",  # Vega expression: always toggle on click
            )

            # Build the filtered bandpass plot
            bp_base = alt.Chart(bp_df).transform_filter(spw_selection)

            # Vega expression for compound selection with empty=true behavior
            # Each term: show if selection is empty OR datum passes selection test
            bp_opacity_expr = (
                "(length(data('bp_quality_store')) == 0 || vlSelectionTest('bp_quality_store', datum)) && "
                "(length(data('bp_type_store')) == 0 || vlSelectionTest('bp_type_store', datum))"
            )

            # Lines for each antenna
            bp_lines = bp_base.mark_line(strokeWidth=1).encode(
                x=alt.X(
                    "frequency_mhz:Q",
                    title="Frequency (MHz)",
                    scale=alt.Scale(zero=False),
                    axis=alt.Axis(format=".1f"),
                ),
                y=alt.Y("amplitude:Q", title="Bandpass Amplitude", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "quality:N",
                    title="Calibration Quality",
                    scale=quality_color_scale,
                    sort=["Good (<15%)", "Moderate (15-50%)", "Poor (>50%)"],
                    legend=legend_config,
                ),
                strokeDash=alt.StrokeDash(
                    "antenna_type:N",
                    title="Antenna Type",
                    scale=alt.Scale(domain=["Core", "Outrigger"], range=[[1, 0], [4, 4]]),
                    legend=legend_config,
                ),
                opacity=alt.condition(bp_opacity_expr, alt.value(0.6), alt.value(0.05)),
                detail="antenna_name:N",
                tooltip=[
                    alt.Tooltip("antenna_name:N", title="Antenna"),
                    alt.Tooltip("antenna_type:N", title="Antenna Type"),
                    alt.Tooltip("quality:N", title="Calibration Quality"),
                    alt.Tooltip("frequency_mhz:Q", title="Frequency (MHz)", format=".2f"),
                    alt.Tooltip("amplitude:Q", title="Amplitude", format=".4f"),
                ],
            )

            # Reference median line with its own legend entry
            ref_base = alt.Chart(ref_line_df).transform_filter(spw_selection)
            ref_line = ref_base.mark_line(strokeWidth=3).encode(
                x="frequency_mhz:Q",
                y="ref_amplitude:Q",
                color=alt.Color(
                    "layer:N",
                    title="Reference",
                    scale=alt.Scale(domain=["Reference Median"], range=["black"]),
                    legend=legend_config,
                ),
                opacity=alt.condition(
                    ref_selection,
                    alt.value(0.9),
                    alt.value(0.0),
                    empty=True,  # Show when nothing selected
                ),
                tooltip=[
                    alt.Tooltip("layer:N", title="Layer"),
                    alt.Tooltip("frequency_mhz:Q", title="Frequency (MHz)", format=".2f"),
                    alt.Tooltip("ref_amplitude:Q", title="Ref Amplitude", format=".4f"),
                ],
            )

            # Combine and add selections
            bp_chart = (
                (bp_lines + ref_line)
                .add_params(spw_selection, quality_selection, type_selection, ref_selection)
                .properties(
                    width=700,
                    height=400,
                    title="Interactive Bandpass Browser - Click legend to filter; Use radio buttons to select SPW",
                )
            )

            figures["bandpass_browser"] = chart_to_embed_html(bp_chart, "vega-bp-browser")

    # =========================================================================
    # Interactive Raw Visibility Browser with Filtering
    # =========================================================================
    if report.raw_visibility_spectra and report.spw_frequencies:
        # Build a long-form dataframe with raw visibility data for all SPWs
        raw_rows = []
        ants_with_data = sorted(report.antenna_amplitudes.keys())
        good_ants = [a for a in ants_with_data if report.antenna_flag_fractions.get(a, 1.0) < 0.15]

        # Include all SPWs for full interactivity
        all_spws = sorted(report.raw_visibility_spectra.keys())

        for spw in all_spws:
            if spw not in report.spw_frequencies:
                continue

            spw_data = report.raw_visibility_spectra.get(spw, {})
            freq_hz, _ = report.spw_frequencies[spw]
            freq_mhz = freq_hz / 1e6

            for ant, (amp, flag_frac) in spw_data.items():
                ant_name = report.antenna_names[ant]
                ant_flag_frac = report.antenna_flag_fractions.get(ant, 0)
                is_out = report.antenna_is_outrigger.get(ant, False)
                quality = get_quality(ant_flag_frac)
                ant_type = "Outrigger" if is_out else "Core"

                for ch in range(len(amp)):
                    if not np.isnan(amp[ch]):  # Only valid data
                        raw_rows.append(
                            {
                                "spw": spw,
                                "channel": ch,
                                "frequency_mhz": freq_mhz[ch],
                                "antenna_id": ant,
                                "antenna_name": ant_name,
                                "amplitude": amp[ch],
                                "quality": quality,
                                "antenna_type": ant_type,
                                "is_reference": ant in good_ants,
                            }
                        )

        if raw_rows:
            raw_df = pd.DataFrame(raw_rows)

            # Calculate reference median per SPW/channel and add layer field
            raw_ref_df = (
                raw_df[raw_df["is_reference"]]
                .groupby(["spw", "channel", "frequency_mhz"])
                .agg({"amplitude": "median"})
                .reset_index()
                .rename(columns={"amplitude": "ref_amplitude"})
            )

            # Create reference line data with its own 'layer' field for legend
            raw_ref_line_df = raw_ref_df.copy()
            raw_ref_line_df["layer"] = "Reference Median"

            # Add layer field to main data
            raw_df["layer"] = "Antenna Data"

            # SPW radio buttons on the side - include all SPWs with frequency labels
            raw_spw_list = sorted(raw_df["spw"].unique())
            raw_spw_labels = []
            for s in raw_spw_list:
                if s in report.spw_frequencies:
                    freq_hz, _ = report.spw_frequencies[s]
                    raw_spw_labels.append(
                        f"SPW {s} ({freq_hz[0] / 1e6:.0f}-{freq_hz[-1] / 1e6:.0f} MHz)"
                    )
                else:
                    raw_spw_labels.append(f"SPW {s}")

            raw_spw_radio = alt.binding_radio(
                options=raw_spw_list, labels=raw_spw_labels, name="Spectral Window: "
            )
            raw_spw_selection = alt.selection_point(
                name="raw_spw", fields=["spw"], bind=raw_spw_radio, value=raw_spw_list[0]
            )

            # Quality filter - click legend to toggle filter (click to select, click again to deselect)
            raw_quality_selection = alt.selection_point(
                name="raw_quality",
                fields=["quality"],
                bind="legend",
                toggle="true",  # Vega expression: always toggle on click
            )

            # Antenna type filter - click legend to toggle filter
            raw_type_selection = alt.selection_point(
                name="raw_type",
                fields=["antenna_type"],
                bind="legend",
                toggle="true",  # Vega expression: always toggle on click
            )

            # Reference median toggle - click legend to show/hide
            raw_ref_selection = alt.selection_point(
                name="raw_ref",
                fields=["layer"],
                bind="legend",
                toggle="true",  # Vega expression: always toggle on click
            )

            # Build the filtered raw visibility plot
            raw_base = alt.Chart(raw_df).transform_filter(raw_spw_selection)

            # Vega expression for compound selection with empty=true behavior
            # Each term: show if selection is empty OR datum passes selection test
            raw_opacity_expr = (
                "(length(data('raw_quality_store')) == 0 || vlSelectionTest('raw_quality_store', datum)) && "
                "(length(data('raw_type_store')) == 0 || vlSelectionTest('raw_type_store', datum))"
            )

            # Lines for each antenna
            raw_lines = raw_base.mark_line(strokeWidth=1).encode(
                x=alt.X(
                    "frequency_mhz:Q",
                    title="Frequency (MHz)",
                    scale=alt.Scale(zero=False),
                    axis=alt.Axis(format=".1f"),
                ),
                y=alt.Y(
                    "amplitude:Q", title="Raw Visibility Amplitude", scale=alt.Scale(zero=False)
                ),
                color=alt.Color(
                    "quality:N",
                    title="Calibration Quality",
                    scale=quality_color_scale,
                    sort=["Good (<15%)", "Moderate (15-50%)", "Poor (>50%)"],
                    legend=legend_config,
                ),
                strokeDash=alt.StrokeDash(
                    "antenna_type:N",
                    title="Antenna Type",
                    scale=alt.Scale(domain=["Core", "Outrigger"], range=[[1, 0], [4, 4]]),
                    legend=legend_config,
                ),
                opacity=alt.condition(raw_opacity_expr, alt.value(0.6), alt.value(0.05)),
                detail="antenna_name:N",
                tooltip=[
                    alt.Tooltip("antenna_name:N", title="Antenna"),
                    alt.Tooltip("antenna_type:N", title="Antenna Type"),
                    alt.Tooltip("quality:N", title="Calibration Quality"),
                    alt.Tooltip("frequency_mhz:Q", title="Frequency (MHz)", format=".2f"),
                    alt.Tooltip("amplitude:Q", title="Amplitude", format=".6f"),
                ],
            )

            # Reference median line with its own legend entry
            raw_ref_base = alt.Chart(raw_ref_line_df).transform_filter(raw_spw_selection)
            raw_ref_line = raw_ref_base.mark_line(strokeWidth=3).encode(
                x="frequency_mhz:Q",
                y="ref_amplitude:Q",
                color=alt.Color(
                    "layer:N",
                    title="Reference",
                    scale=alt.Scale(domain=["Reference Median"], range=["black"]),
                    legend=legend_config,
                ),
                opacity=alt.condition(
                    raw_ref_selection,
                    alt.value(0.9),
                    alt.value(0.0),
                    empty=True,  # Show when nothing selected
                ),
                tooltip=[
                    alt.Tooltip("layer:N", title="Layer"),
                    alt.Tooltip("frequency_mhz:Q", title="Frequency (MHz)", format=".2f"),
                    alt.Tooltip("ref_amplitude:Q", title="Ref Amplitude", format=".6f"),
                ],
            )

            # Combine and add selections
            raw_chart = (
                (raw_lines + raw_ref_line)
                .add_params(
                    raw_spw_selection, raw_quality_selection, raw_type_selection, raw_ref_selection
                )
                .properties(
                    width=700,
                    height=400,
                    title="Interactive Raw Visibility Browser - Click legend to filter; Use radio buttons to select SPW",
                )
            )

            figures["raw_browser"] = chart_to_embed_html(raw_chart, "vega-raw-browser")

    return figures


def generate_html_report(
    report: BandpassReportData,
    output_path: str | None = None,
    embed_figures: bool = True,
    interactive: bool = True,
    save_pngs_dir: str | None = None,
    image_prefix: str = "bandpass",
) -> str:
    """
    Generate a complete HTML report for bandpass diagnostics.

    Parameters
    ----------
    report : BandpassReportData
        Report data container
    output_path : str, optional
        Path to save the HTML report. If None, returns HTML string only.
    embed_figures : bool
        If True, embed figures as base64. If False, reference external files.
    interactive : bool
        If True, use interactive Altair/Vega-Lite figures where available.
        Static matplotlib figures are used as fallback for complex plots.
    save_pngs_dir : Optional[str]
        If provided, save PNG files to this directory
    image_prefix : str
        Prefix for saved PNG filenames (default: "bandpass")

    Returns
    -------
    str
        HTML content of the report
    """
    # Generate figures
    if embed_figures:
        figures = generate_figures_base64(
            report, save_pngs_dir=save_pngs_dir, image_prefix=image_prefix
        )
    else:
        figures = report.figure_paths

    # Generate interactive figures (Altair/Vega-Lite)
    interactive_figures = {}
    if interactive:
        try:
            interactive_figures = generate_altair_figures(report)
        except Exception as e:
            logger.warning(f"Failed to generate interactive figures: {e}")

    # Determine status color and icon
    flag_pct = report.flag_fraction * 100
    if flag_pct < 5:
        status_color = "#28a745"  # green
        status_icon = "[OK]"
        status_text = "GOOD"
    elif flag_pct < 15:
        status_color = "#ffc107"  # yellow
        status_icon = "[!]"
        status_text = "MODERATE"
    elif flag_pct < 30:
        status_color = "#fd7e14"  # orange
        status_icon = "[!]"
        status_text = "HIGH"
    else:
        status_color = "#dc3545"  # red
        status_icon = "[X]"
        status_text = "CRITICAL"

    # Build figure HTML - static PNG fallback
    def img_tag(fig_key, alt_text):
        if embed_figures and fig_key in figures:
            return f'<img src="data:image/png;base64,{figures[fig_key]}" alt="{alt_text}" style="max-width:100%; height:auto;">'
        elif fig_key in figures:
            return f'<img src="{figures[fig_key]}" alt="{alt_text}" style="max-width:100%; height:auto;">'
        return f'<p style="color:gray;">Figure not available: {alt_text}</p>'

    # Build interactive figure HTML - uses Altair/Vega-Lite if available
    def interactive_fig(fig_key, alt_text, fallback_key=None):
        if fig_key in interactive_figures:
            return f'<div class="interactive-figure">{interactive_figures[fig_key]}</div>'
        # Fallback to static image
        return img_tag(fallback_key or fig_key, alt_text)

    # Build recommendations HTML
    recommendations_html = ""
    for rec in report.recommendations:
        recommendations_html += f"<li>{rec}</li>\n"

    # Build antenna table
    ants_with_data = sorted(report.antenna_amplitudes.keys())
    antenna_rows = ""
    for ant in sorted(ants_with_data, key=lambda a: report.antenna_flag_fractions[a], reverse=True)[
        :20
    ]:
        name = report.antenna_names[ant]
        amp = report.antenna_amplitudes[ant]
        flag = report.antenna_flag_fractions[ant] * 100
        is_out = "[O]" if report.antenna_is_outrigger.get(ant, False) else ""

        if flag < 15:
            row_class = "good"
        elif flag < 50:
            row_class = "warning"
        else:
            row_class = "bad"

        antenna_rows += f'''
        <tr class="{row_class}">
            <td>{is_out} {name}</td>
            <td>{amp:.5f}</td>
            <td>{flag:.1f}%</td>
        </tr>
        '''

    # Pre-calculate template variables
    timestamp_str = report.timestamp[:19].replace("T", " ")
    corr_strength = "strong" if abs(report.amplitude_flagging_correlation) > 0.5 else "weak"
    corr_sign = "negative" if report.amplitude_flagging_correlation < 0 else "positive"
    ks_result = (
        "distributions significantly different"
        if report.ks_pvalue < 0.05
        else "no significant difference"
    )

    html = render_template(
        "bandpass_report.html",
        report=report,
        status_color=status_color,
        status_icon=status_icon,
        status_text=status_text,
        flag_pct=flag_pct,
        recommendations_html=recommendations_html,
        antenna_rows=antenna_rows,
        timestamp_str=timestamp_str,
        corr_strength=corr_strength,
        corr_sign=corr_sign,
        ks_result=ks_result,
        fig_antenna_map=interactive_fig("antenna_map", "Antenna Map"),
        fig_raw_browser=interactive_fig("raw_browser", "Interactive Raw Visibility Browser"),
        fig_raw_visibility_spectra=img_tag("raw_visibility_spectra", "Raw Visibility Spectra"),
        fig_bandpass_browser=interactive_fig("bandpass_browser", "Interactive Bandpass Browser"),
        fig_bandpass_solutions=img_tag("bandpass_solutions", "Bandpass Solutions"),
        fig_amplitude_vs_flagging=interactive_fig("amplitude_vs_flagging", "Amplitude vs Flagging"),
        fig_amplitude_distribution=img_tag("amplitude_distribution", "Amplitude Distribution"),
        fig_antenna_amplitudes=interactive_fig("antenna_amplitudes", "Antenna Amplitudes"),
        fig_spw_antenna_heatmap=img_tag("spw_antenna_heatmap", "SPW-Antenna Heatmap"),
    )

    if output_path:
        Path(output_path).write_text(html, encoding="utf-8")
        logger.info("Bandpass diagnostics report saved to: %s", output_path)

    return html


def generate_bandpass_report(
    ms_path: str,
    bpcal_path: str,
    output_dir: str,
    calibrator_name: str = "unknown",
    save_pngs: bool = True,
) -> str:
    """
    High-level function to generate complete bandpass diagnostics report.

    This is the main entry point for the pipeline integration.

    Parameters
    ----------
    ms_path : str
        Path to the measurement set
    bpcal_path : str
        Path to the bandpass calibration table
    output_dir : str
        Directory to save the report
    calibrator_name : str
        Name of the calibrator source
    save_pngs : bool
        Whether to save PNG files alongside the HTML report (default True)

    Returns
    -------
    str
        Path to the generated HTML report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading bandpass diagnostics data from %s", ms_path)
    report_data = load_bandpass_report_data(ms_path, bpcal_path, calibrator_name)

    # Generate report filename based on MS name
    ms_name = Path(ms_path).stem
    report_filename = f"bandpass_diagnostics_{ms_name}.html"
    report_path = output_dir / report_filename

    # Determine PNG save directory
    save_pngs_dir = str(output_dir) if save_pngs else None

    # Generate HTML report (and optionally save PNGs)
    logger.info("Generating bandpass diagnostics report: %s", report_path)
    generate_html_report(
        report_data,
        str(report_path),
        embed_figures=True,
        save_pngs_dir=save_pngs_dir,
        image_prefix=f"bandpass_{ms_name}",
    )

    return str(report_path)
