"""
Calibration stability plotting for DSA-110 continuum imaging pipeline.

Visualizes gain amplitude and phase stability metrics from CASA calibration tables.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from dsa110_contimg.core.visualization.config import FigureConfig, PlotStyle


def _setup_matplotlib() -> None:
    """Configure matplotlib for headless operation."""
    import matplotlib

    matplotlib.use("Agg")


def plot_calibration_stability(
    antenna_results: list[dict],
    summary: dict,
    output_path: str | Path,
    config: FigureConfig | None = None,
) -> None:
    """Generate multi-panel plot showing calibration stability metrics.

    Parameters
    ----------
    antenna_results :
        List of per-antenna stability results.
    summary :
        Summary statistics dictionary.
    output_path :
        Path to save the plot.
    config :
        Figure configuration.
    antenna_results: List[Dict] :

    """
    _setup_matplotlib()

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    if not antenna_results:
        raise ValueError("No antenna results to plot")

    # Extract data
    antenna_ids = [r["antenna_id"] for r in antenna_results]
    amp_stds = [r["amplitude"]["fractional_std_percent"] for r in antenna_results]
    phase_stds = [r["phase"]["std_deg"] for r in antenna_results]
    phase_rms = [r["phase"]["wrapped_rms_deg"] for r in antenna_results]
    drift_rates = [r["temporal"]["drift_rate_per_hour"] for r in antenna_results]

    with plt.rc_context(config.to_mpl_params()):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Row 1, Col 1: Amplitude std per antenna
        axes[0, 0].plot(antenna_ids, amp_stds, "o-", alpha=0.7)
        axes[0, 0].axhline(
            summary["amplitude_stability"]["mean_fractional_std_percent"],
            color="r",
            linestyle="--",
            label="Mean",
        )
        axes[0, 0].axhline(10, color="orange", linestyle=":", label="Sim default (10%)")
        axes[0, 0].set_xlabel("Antenna ID")
        axes[0, 0].set_ylabel("Amplitude Std (%)")
        axes[0, 0].set_title("Gain Amplitude Stability per Antenna")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Row 1, Col 2: Phase std per antenna
        axes[0, 1].plot(antenna_ids, phase_stds, "o-", alpha=0.7, color="orange")
        axes[0, 1].axhline(
            summary["phase_stability"]["mean_std_deg"],
            color="r",
            linestyle="--",
            label="Mean",
        )
        axes[0, 1].axhline(10, color="orange", linestyle=":", label="Sim default (10°)")
        axes[0, 1].set_xlabel("Antenna ID")
        axes[0, 1].set_ylabel("Phase Std (degrees)")
        axes[0, 1].set_title("Gain Phase Stability per Antenna")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Row 1, Col 3: Phase wrapped RMS
        axes[0, 2].plot(antenna_ids, phase_rms, "o-", alpha=0.7, color="green")
        axes[0, 2].axhline(
            summary["phase_stability"]["mean_wrapped_rms_deg"],
            color="r",
            linestyle="--",
            label="Mean",
        )
        axes[0, 2].set_xlabel("Antenna ID")
        axes[0, 2].set_ylabel("Phase Wrapped RMS (degrees)")
        axes[0, 2].set_title("Phase Wrapped RMS per Antenna")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Row 2, Col 1: Amplitude std histogram
        axes[1, 0].hist(amp_stds, bins=20, alpha=0.7, edgecolor="black")
        axes[1, 0].axvline(
            summary["amplitude_stability"]["mean_fractional_std_percent"],
            color="r",
            linestyle="--",
            label="Mean",
        )
        axes[1, 0].axvline(10, color="orange", linestyle=":", label="Sim default")
        axes[1, 0].set_xlabel("Amplitude Std (%)")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].set_title("Amplitude Std Distribution")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Row 2, Col 2: Phase std histogram
        axes[1, 1].hist(phase_stds, bins=20, alpha=0.7, edgecolor="black", color="orange")
        axes[1, 1].axvline(
            summary["phase_stability"]["mean_std_deg"],
            color="r",
            linestyle="--",
            label="Mean",
        )
        axes[1, 1].axvline(10, color="orange", linestyle=":", label="Sim default")
        axes[1, 1].set_xlabel("Phase Std (degrees)")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_title("Phase Std Distribution")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Row 2, Col 3: Amplitude drift rate
        axes[1, 2].plot(antenna_ids, drift_rates, "o-", alpha=0.7, color="purple")
        axes[1, 2].axhline(0, color="black", linestyle="-", linewidth=0.5)
        axes[1, 2].set_xlabel("Antenna ID")
        axes[1, 2].set_ylabel("Drift Rate (per hour)")
        axes[1, 2].set_title("Amplitude Drift Rate per Antenna")
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
