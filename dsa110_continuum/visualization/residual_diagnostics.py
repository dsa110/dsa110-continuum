"""
Visibility residual diagnostics for calibration quality assessment.

Computes and visualizes post-calibration residuals (CORRECTED_DATA - MODEL_DATA)
to verify calibration quality. Key diagnostics:

- Residual amplitude vs baseline length (should be flat for good calibration)
- Residual phase vs time (should be scatter around zero)
- Residual histograms (should be Gaussian with mean ≈ 0)
- Per-antenna residual statistics

References
----------
- Thompson, Moran, Swenson: "Interferometry and Synthesis in Radio Astronomy"
- CASA visstat task patterns
- eht-imaging residual analysis

Note: These are VISIBILITY residuals (DATA - MODEL in UV space), distinct from
IMAGE residuals (dirty image - clean model) in qa_plots.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

from dsa110_contimg.core.visualization.config import FigureConfig, PlotStyle
from dsa110_contimg.core.visualization.plot_context import (
    PlotContext,
    should_generate_interactive,
)

logger = logging.getLogger(__name__)


def _setup_matplotlib() -> None:
    """Configure matplotlib for headless operation."""
    import matplotlib

    matplotlib.use("Agg")


@dataclass
class ResidualData:
    """Container for visibility residual data extracted from MS."""

    residuals: NDArray  # Complex, shape varies by extraction mode
    uvdist: NDArray  # UV distance in wavelengths
    time: NDArray  # Time in seconds from start
    antenna1: NDArray
    antenna2: NDArray
    baseline_lengths_m: NDArray
    frequencies_hz: NDArray
    weights: NDArray | None = None
    flags: NDArray | None = None
    polarization_labels: list[str] = field(default_factory=lambda: ["XX", "YY"])
    n_baselines: int = 0
    n_times: int = 0
    n_channels: int = 0
    n_pols: int = 0


@dataclass
class ResidualStatistics:
    """Summary statistics for visibility residuals.

    All statistics are computed on unflagged data only.

    """

    # Global statistics
    n_unflagged: int
    n_total: int
    flag_fraction: float

    # Amplitude statistics
    mean_amplitude: float
    median_amplitude: float
    rms_amplitude: float
    std_amplitude: float

    # Phase statistics (degrees)
    mean_phase_deg: float
    std_phase_deg: float
    circular_mean_phase_deg: float  # Proper circular mean

    # Real/Imag statistics (should be ~0 for good cal)
    mean_real: float
    mean_imag: float
    std_real: float
    std_imag: float

    # Per-antenna statistics
    per_antenna_rms: dict[int, float] | None = None

    # Outlier detection
    outlier_fraction: float = 0.0  # Fraction > 5*median
    outlier_threshold: float = 5.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_unflagged": self.n_unflagged,
            "n_total": self.n_total,
            "flag_fraction": self.flag_fraction,
            "amplitude": {
                "mean": self.mean_amplitude,
                "median": self.median_amplitude,
                "rms": self.rms_amplitude,
                "std": self.std_amplitude,
            },
            "phase_deg": {
                "mean": self.mean_phase_deg,
                "std": self.std_phase_deg,
                "circular_mean": self.circular_mean_phase_deg,
            },
            "real_imag": {
                "mean_real": self.mean_real,
                "mean_imag": self.mean_imag,
                "std_real": self.std_real,
                "std_imag": self.std_imag,
            },
            "outlier_fraction": self.outlier_fraction,
            "per_antenna_rms": self.per_antenna_rms,
        }


def extract_residuals_from_ms(
    ms_path: str | Path,
    data_column: str = "CORRECTED_DATA",
    model_column: str = "MODEL_DATA",
    average_channels: bool = True,
    channel_range: tuple[int, int] | None = None,
    time_range: tuple[float, float] | None = None,
    exclude_autocorr: bool = True,
    apply_weights: bool = True,
) -> ResidualData:
    """Extract visibility residuals from a Measurement Set.

    Computes residuals as: data_column - model_column

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    data_column :
        Data column name (CORRECTED_DATA, DATA)
    model_column :
        Model column name (MODEL_DATA)
    average_channels :
        If True, average over frequency channels
    channel_range :
        Optional (start, end) channel indices to use
    time_range :
        Optional (start_mjd, end_mjd) to select
    exclude_autocorr :
        If True, exclude auto-correlations
    apply_weights :
        If True, apply WEIGHT_SPECTRUM or WEIGHT to residuals
    ms_path : Union[str, Path]

    Returns
    -------
        ResidualData container with extracted residuals and metadata

    Raises
    ------
    FileNotFoundError
        If MS does not exist
    ValueError
        If required columns are missing
    RuntimeError
        If casacore is not available

    """
    try:
        from casacore.tables import table
    except ImportError as e:
        raise RuntimeError(
            "casacore.tables is required for residual extraction. "
            "Ensure you are in the casa6 environment."
        ) from e

    ms_path = Path(ms_path)
    if not ms_path.exists():
        raise FileNotFoundError(f"MS not found: {ms_path}")

    logger.info(f"Extracting residuals from {ms_path}")
    logger.info(f"  Data column: {data_column}, Model column: {model_column}")

    with table(str(ms_path), readonly=True) as tb:
        # Verify columns exist
        available_cols = tb.colnames()
        if data_column not in available_cols:
            raise ValueError(
                f"Data column '{data_column}' not found. "
                f"Available: {[c for c in available_cols if 'DATA' in c]}"
            )
        if model_column not in available_cols:
            raise ValueError(
                f"Model column '{model_column}' not found. "
                f"Available: {[c for c in available_cols if 'DATA' in c or 'MODEL' in c]}"
            )

        # Read data
        data = tb.getcol(data_column)  # shape: (nrow, nchan, npol)
        model = tb.getcol(model_column)
        flags = tb.getcol("FLAG")
        uvw = tb.getcol("UVW")  # shape: (nrow, 3) in meters
        time = tb.getcol("TIME")  # MJD seconds
        ant1 = tb.getcol("ANTENNA1")
        ant2 = tb.getcol("ANTENNA2")

        # Try to get weights
        weights = None
        if apply_weights:
            if "WEIGHT_SPECTRUM" in available_cols:
                weights = tb.getcol("WEIGHT_SPECTRUM")
            elif "WEIGHT" in available_cols:
                # Expand WEIGHT to match data shape
                w = tb.getcol("WEIGHT")  # shape: (nrow, npol)
                weights = np.broadcast_to(w[:, np.newaxis, :], data.shape).copy()

    # Get frequency info from SPECTRAL_WINDOW subtable
    with table(str(ms_path / "SPECTRAL_WINDOW"), readonly=True) as spw_tb:
        chan_freq = spw_tb.getcol("CHAN_FREQ")[0]  # Assume single SPW

    # Get polarization labels from POLARIZATION subtable
    pol_labels = ["XX", "YY"]  # Default for DSA-110
    try:
        with table(str(ms_path / "POLARIZATION"), readonly=True) as pol_tb:
            corr_type = pol_tb.getcol("CORR_TYPE")[0]
            # Map correlation types to labels
            corr_map = {
                5: "RR",
                6: "RL",
                7: "LR",
                8: "LL",
                9: "XX",
                10: "XY",
                11: "YX",
                12: "YY",
                1: "I",
                2: "Q",
                3: "U",
                4: "V",
            }
            pol_labels = [corr_map.get(c, f"P{c}") for c in corr_type]
    except Exception as e:
        logger.warning(f"Could not read polarization info: {e}")

    # Exclude auto-correlations
    if exclude_autocorr:
        cross_mask = ant1 != ant2
        data = data[cross_mask]
        model = model[cross_mask]
        flags = flags[cross_mask]
        uvw = uvw[cross_mask]
        time = time[cross_mask]
        ant1 = ant1[cross_mask]
        ant2 = ant2[cross_mask]
        if weights is not None:
            weights = weights[cross_mask]

    # Apply channel selection
    if channel_range is not None:
        ch_start, ch_end = channel_range
        data = data[:, ch_start:ch_end, :]
        model = model[:, ch_start:ch_end, :]
        flags = flags[:, ch_start:ch_end, :]
        chan_freq = chan_freq[ch_start:ch_end]
        if weights is not None:
            weights = weights[:, ch_start:ch_end, :]

    # Compute residuals
    residuals = data - model

    # Apply flags (set flagged data to NaN for statistics)
    residuals = np.where(flags, np.nan + 0j, residuals)

    # Compute UV distance in wavelengths (use central frequency)
    central_freq = np.median(chan_freq)
    wavelength = 2.998e8 / central_freq  # meters
    baseline_lengths_m = np.sqrt(uvw[:, 0] ** 2 + uvw[:, 1] ** 2)
    uvdist = baseline_lengths_m / wavelength

    # Channel averaging if requested
    if average_channels:
        # Weighted average over channels
        if weights is not None:
            # Mask flagged weights
            w_masked = np.where(flags, 0.0, weights)
            w_sum = np.nansum(w_masked, axis=1, keepdims=True)
            w_sum = np.where(w_sum == 0, 1.0, w_sum)  # Avoid division by zero
            residuals = np.nansum(residuals * w_masked, axis=1) / w_sum.squeeze(axis=1)
            weights = np.nansum(w_masked, axis=1)
        else:
            residuals = np.nanmean(residuals, axis=1)  # shape: (nrow, npol)
        flags = np.all(flags, axis=1)  # Fully flagged if all channels flagged
        chan_freq = np.array([central_freq])

    # Normalize time to start
    time_offset = time - time.min()

    # Count unique baselines and times
    unique_baselines = set(zip(ant1, ant2))
    unique_times = len(np.unique(time))

    return ResidualData(
        residuals=residuals,
        uvdist=uvdist,
        time=time_offset,
        antenna1=ant1,
        antenna2=ant2,
        baseline_lengths_m=baseline_lengths_m,
        frequencies_hz=chan_freq,
        weights=weights,
        flags=flags,
        polarization_labels=pol_labels,
        n_baselines=len(unique_baselines),
        n_times=unique_times,
        n_channels=len(chan_freq),
        n_pols=residuals.shape[-1],
    )


def compute_residual_statistics(
    data: ResidualData,
    outlier_threshold: float = 5.0,
) -> ResidualStatistics:
    """Compute comprehensive statistics on visibility residuals.

    Parameters
    ----------
    data :
        ResidualData from extract_residuals_from_ms()
    outlier_threshold :
        Multiple of median for outlier detection

    Returns
    -------
        ResidualStatistics with summary metrics

    """
    residuals = data.residuals.flatten()

    # Handle flags/NaNs
    valid_mask = ~np.isnan(residuals)
    valid_residuals = residuals[valid_mask]

    n_total = len(residuals)
    n_unflagged = len(valid_residuals)
    flag_fraction = 1.0 - (n_unflagged / n_total) if n_total > 0 else 0.0

    if n_unflagged == 0:
        logger.warning("No valid (unflagged) residuals to compute statistics")
        return ResidualStatistics(
            n_unflagged=0,
            n_total=n_total,
            flag_fraction=1.0,
            mean_amplitude=np.nan,
            median_amplitude=np.nan,
            rms_amplitude=np.nan,
            std_amplitude=np.nan,
            mean_phase_deg=np.nan,
            std_phase_deg=np.nan,
            circular_mean_phase_deg=np.nan,
            mean_real=np.nan,
            mean_imag=np.nan,
            std_real=np.nan,
            std_imag=np.nan,
            outlier_fraction=np.nan,
        )

    # Amplitude statistics
    amplitudes = np.abs(valid_residuals)
    mean_amp = float(np.mean(amplitudes))
    median_amp = float(np.median(amplitudes))
    rms_amp = float(np.sqrt(np.mean(amplitudes**2)))
    std_amp = float(np.std(amplitudes))

    # Phase statistics (in degrees)
    phases = np.angle(valid_residuals, deg=True)
    mean_phase = float(np.mean(phases))
    std_phase = float(np.std(phases))

    # Circular mean for phases (proper handling of wrap-around)
    phases_rad = np.angle(valid_residuals)
    circular_mean_rad = np.arctan2(np.mean(np.sin(phases_rad)), np.mean(np.cos(phases_rad)))
    circular_mean_deg = float(np.degrees(circular_mean_rad))

    # Real/Imaginary statistics
    real_parts = np.real(valid_residuals)
    imag_parts = np.imag(valid_residuals)
    mean_real = float(np.mean(real_parts))
    mean_imag = float(np.mean(imag_parts))
    std_real = float(np.std(real_parts))
    std_imag = float(np.std(imag_parts))

    # Outlier detection
    outlier_mask = amplitudes > (outlier_threshold * median_amp)
    outlier_fraction = float(np.mean(outlier_mask))

    # Per-antenna RMS
    per_antenna_rms: dict[int, float] = {}
    all_antennas = np.unique(np.concatenate([data.antenna1, data.antenna2]))

    for ant in all_antennas:
        # Get residuals involving this antenna
        ant_mask = (data.antenna1 == ant) | (data.antenna2 == ant)
        if data.residuals.ndim == 2:
            ant_residuals = data.residuals[ant_mask, :].flatten()
        else:
            ant_residuals = data.residuals[ant_mask].flatten()

        ant_valid = ant_residuals[~np.isnan(ant_residuals)]
        if len(ant_valid) > 0:
            per_antenna_rms[int(ant)] = float(np.sqrt(np.mean(np.abs(ant_valid) ** 2)))

    return ResidualStatistics(
        n_unflagged=n_unflagged,
        n_total=n_total,
        flag_fraction=flag_fraction,
        mean_amplitude=mean_amp,
        median_amplitude=median_amp,
        rms_amplitude=rms_amp,
        std_amplitude=std_amp,
        mean_phase_deg=mean_phase,
        std_phase_deg=std_phase,
        circular_mean_phase_deg=circular_mean_deg,
        mean_real=mean_real,
        mean_imag=mean_imag,
        std_real=std_real,
        std_imag=std_imag,
        per_antenna_rms=per_antenna_rms,
        outlier_fraction=outlier_fraction,
        outlier_threshold=outlier_threshold,
    )


def plot_residual_amplitude_vs_baseline(
    data: ResidualData,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Residual Amplitude vs Baseline Length",
    polarization: int | None = None,
    use_hexbin: bool = True,
    gridsize: int = 50,
    show_median_line: bool = True,
    context: PlotContext | None = None,
    interactive: bool | None = None,
) -> Figure:
    """Plot residual amplitude vs UV distance (baseline length).

    For well-calibrated data, residuals should be flat with no trend
    vs baseline length.

    Parameters
    ----------
    data :
        ResidualData from extract_residuals_from_ms()
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    polarization :
        Polarization index to plot (None = average all)
    use_hexbin :
        Use hexbin for dense data, else scatter
    gridsize :
        Hexbin grid size
    show_median_line :
        Show running median line
    context :
        Plot context for format selection
    interactive :
        Override interactive format selection

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    # Extract data
    uvdist = data.uvdist
    residuals = data.residuals

    # Handle polarization selection
    if residuals.ndim == 2:
        if polarization is not None:
            amps = np.abs(residuals[:, polarization])
            pol_label = data.polarization_labels[polarization]
        else:
            # Average over polarizations
            amps = np.nanmean(np.abs(residuals), axis=1)
            pol_label = "avg"
    else:
        amps = np.abs(residuals)
        pol_label = ""

    # Remove NaN values
    valid = ~np.isnan(amps)
    uvdist_valid = uvdist[valid]
    amps_valid = amps[valid]

    # Convert to kilolambda
    uvdist_klambda = uvdist_valid / 1e3

    # Check if interactive mode
    use_interactive = should_generate_interactive(context, interactive)

    if use_interactive:
        from dsa110_contimg.core.visualization.vega_specs import save_vega_spec

        # Subsample for interactive (Vega-Lite can be slow with many points)
        max_points = 5000
        if len(uvdist_klambda) > max_points:
            indices = np.random.choice(len(uvdist_klambda), max_points, replace=False)
            uvdist_sub = uvdist_klambda[indices]
            amps_sub = amps_valid[indices]
        else:
            uvdist_sub = uvdist_klambda
            amps_sub = amps_valid

        values = [
            {"uvdist_klambda": float(u), "amplitude": float(a)}
            for u, a in zip(uvdist_sub, amps_sub)
        ]

        spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "description": "Residual amplitude vs baseline length",
            "title": title,
            "width": 600,
            "height": 400,
            "data": {"values": values},
            "mark": {"type": "circle", "opacity": 0.3, "size": 10},
            "encoding": {
                "x": {
                    "field": "uvdist_klambda",
                    "type": "quantitative",
                    "title": "UV Distance (kλ)",
                },
                "y": {
                    "field": "amplitude",
                    "type": "quantitative",
                    "title": "Residual Amplitude",
                },
                "tooltip": [
                    {"field": "uvdist_klambda", "type": "quantitative", "format": ".1f"},
                    {"field": "amplitude", "type": "quantitative", "format": ".4e"},
                ],
            },
        }

        if output:
            json_path = Path(str(output).replace(".png", ".vega.json"))
            save_vega_spec(spec, json_path)
            logger.info(f"Saved interactive residual plot: {json_path}")

        return None  # No matplotlib figure for interactive

    # Static matplotlib plot
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    if use_hexbin and len(uvdist_klambda) > 1000:
        hb = ax.hexbin(
            uvdist_klambda,
            amps_valid,
            gridsize=gridsize,
            cmap="viridis",
            mincnt=1,
            reduce_C_function=np.median,
        )
        fig.colorbar(hb, ax=ax, label="Median count")
    else:
        ax.scatter(
            uvdist_klambda,
            amps_valid,
            s=1,
            alpha=0.3,
            c="steelblue",
        )

    # Running median line
    if show_median_line:
        # Bin the data
        n_bins = 20
        bin_edges = np.linspace(uvdist_klambda.min(), uvdist_klambda.max(), n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_medians = []

        for i in range(n_bins):
            mask = (uvdist_klambda >= bin_edges[i]) & (uvdist_klambda < bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_medians.append(np.median(amps_valid[mask]))
            else:
                bin_medians.append(np.nan)

        ax.plot(bin_centers, bin_medians, "r-", linewidth=2, label="Running median")
        ax.legend()

    ax.set_xlabel("UV Distance (kλ)", fontsize=config.effective_label_size)
    ax.set_ylabel("Residual Amplitude", fontsize=config.effective_label_size)
    ax.set_title(
        f"{title} [{pol_label}]" if pol_label else title, fontsize=config.effective_title_size
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved residual vs baseline plot: {output}")
        plt.close(fig)

    return fig


def plot_residual_phase_vs_time(
    data: ResidualData,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Residual Phase vs Time",
    polarization: int | None = None,
    color_by_baseline: bool = False,
    max_baselines: int = 10,
    context: PlotContext | None = None,
    interactive: bool | None = None,
) -> Figure:
    """Plot residual phase vs time.

    For well-calibrated data, residual phases should scatter around zero
    with no temporal trends.

    Parameters
    ----------
    data :
        ResidualData from extract_residuals_from_ms()
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    polarization :
        Polarization index to plot (None = average)
    color_by_baseline :
        Color-code different baselines
    max_baselines :
        Max baselines to show if color_by_baseline
    context :
        Plot context for format selection
    interactive :
        Override interactive format selection

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    # Extract data
    time_min = data.time / 60.0  # Convert to minutes
    residuals = data.residuals

    # Handle polarization selection
    if residuals.ndim == 2:
        if polarization is not None:
            phases = np.angle(residuals[:, polarization], deg=True)
            pol_label = data.polarization_labels[polarization]
        else:
            # Circular mean over polarizations
            phases = np.degrees(
                np.arctan2(
                    np.nanmean(np.sin(np.angle(residuals)), axis=1),
                    np.nanmean(np.cos(np.angle(residuals)), axis=1),
                )
            )
            pol_label = "avg"
    else:
        phases = np.angle(residuals, deg=True)
        pol_label = ""

    # Remove NaN values
    valid = ~np.isnan(phases)
    time_valid = time_min[valid]
    phases_valid = phases[valid]

    # Check if interactive mode
    use_interactive = should_generate_interactive(context, interactive)

    if use_interactive:
        from dsa110_contimg.core.visualization.vega_specs import save_vega_spec

        # Subsample for interactive
        max_points = 5000
        if len(time_valid) > max_points:
            indices = np.random.choice(len(time_valid), max_points, replace=False)
            time_sub = time_valid[indices]
            phases_sub = phases_valid[indices]
        else:
            time_sub = time_valid
            phases_sub = phases_valid

        values = [
            {"time_min": float(t), "phase_deg": float(p)} for t, p in zip(time_sub, phases_sub)
        ]

        spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "description": "Residual phase vs time",
            "title": title,
            "width": 600,
            "height": 400,
            "data": {"values": values},
            "layer": [
                {
                    "mark": {"type": "circle", "opacity": 0.3, "size": 10},
                    "encoding": {
                        "x": {
                            "field": "time_min",
                            "type": "quantitative",
                            "title": "Time (minutes)",
                        },
                        "y": {
                            "field": "phase_deg",
                            "type": "quantitative",
                            "title": "Residual Phase (deg)",
                            "scale": {"domain": [-180, 180]},
                        },
                    },
                },
                {
                    "mark": {"type": "rule", "strokeDash": [4, 4], "color": "gray"},
                    "encoding": {"y": {"datum": 0}},
                },
            ],
        }

        if output:
            json_path = Path(str(output).replace(".png", ".vega.json"))
            save_vega_spec(spec, json_path)
            logger.info(f"Saved interactive phase plot: {json_path}")

        return None

    # Static matplotlib plot
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    if color_by_baseline:
        # Get unique baselines
        baselines = list(set(zip(data.antenna1[valid], data.antenna2[valid])))
        baselines = baselines[:max_baselines]

        for bl in baselines:
            mask = (data.antenna1[valid] == bl[0]) & (data.antenna2[valid] == bl[1])
            if np.sum(mask) > 0:
                ax.scatter(
                    time_valid[mask],
                    phases_valid[mask],
                    s=2,
                    alpha=0.5,
                    label=f"{bl[0]}-{bl[1]}",
                )
        ax.legend(fontsize=8, ncol=2)
    else:
        ax.scatter(
            time_valid,
            phases_valid,
            s=1,
            alpha=0.3,
            c="steelblue",
        )

    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylim(-180, 180)
    ax.set_xlabel("Time (minutes)", fontsize=config.effective_label_size)
    ax.set_ylabel("Residual Phase (deg)", fontsize=config.effective_label_size)
    ax.set_title(
        f"{title} [{pol_label}]" if pol_label else title, fontsize=config.effective_title_size
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved residual phase plot: {output}")
        plt.close(fig)

    return fig


def plot_residual_histogram(
    data: ResidualData,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Residual Distribution",
    bins: int = 100,
    show_gaussian: bool = True,
    plot_real_imag: bool = True,
    context: PlotContext | None = None,
    interactive: bool | None = None,
) -> Figure:
    """Plot histogram of visibility residuals.

    For well-calibrated data with thermal noise only, residuals should
    follow a Gaussian distribution centered at zero.

    Parameters
    ----------
    data :
        ResidualData from extract_residuals_from_ms()
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    bins :
        Number of histogram bins
    show_gaussian :
        Overlay Gaussian fit
    plot_real_imag :
        Plot separate real/imag histograms (2 panels)
    context :
        Plot context for format selection
    interactive :
        Override interactive format selection

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from scipy import stats

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    residuals = data.residuals.flatten()
    valid = ~np.isnan(residuals)
    residuals_valid = residuals[valid]

    # Check if interactive mode
    use_interactive = should_generate_interactive(context, interactive)

    if use_interactive:
        from dsa110_contimg.core.visualization.vega_specs import (
            create_residual_histogram_spec,
            save_vega_spec,
        )

        # Use amplitude for interactive histogram
        amps = np.abs(residuals_valid)
        spec = create_residual_histogram_spec(
            residuals=amps,
            bin_count=bins,
            title=f"{title} (Amplitude)",
            gaussian_fit={"mean": float(np.mean(amps)), "std": float(np.std(amps))},
        )

        if output:
            json_path = Path(str(output).replace(".png", ".vega.json"))
            save_vega_spec(spec, json_path)
            logger.info(f"Saved interactive histogram: {json_path}")

        return None

    # Static matplotlib plot
    if plot_real_imag:
        fig, axes = plt.subplots(1, 2, figsize=(config.figsize[0] * 2, config.figsize[1]))

        for ax, part, label in [
            (axes[0], np.real(residuals_valid), "Real"),
            (axes[1], np.imag(residuals_valid), "Imaginary"),
        ]:
            counts, bin_edges, _ = ax.hist(
                part,
                bins=bins,
                density=True,
                alpha=0.7,
                color="steelblue",
                edgecolor="black",
                linewidth=0.5,
            )

            if show_gaussian:
                # Fit Gaussian
                mu, sigma = stats.norm.fit(part)
                x = np.linspace(part.min(), part.max(), 200)
                pdf = stats.norm.pdf(x, mu, sigma)
                ax.plot(x, pdf, "r-", linewidth=2, label=f"Gaussian (μ={mu:.2e}, σ={sigma:.2e})")
                ax.legend(fontsize=8)

            ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
            ax.set_xlabel(f"{label} Residual", fontsize=config.effective_label_size)
            ax.set_ylabel("Density", fontsize=config.effective_label_size)
            ax.set_title(f"{label} Part", fontsize=config.effective_title_size)
            ax.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=config.effective_title_size + 2)

    else:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

        amps = np.abs(residuals_valid)
        counts, bin_edges, _ = ax.hist(
            amps,
            bins=bins,
            density=True,
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
            linewidth=0.5,
        )

        if show_gaussian:
            # Rayleigh distribution for amplitude of complex Gaussian
            from scipy.stats import rayleigh

            loc, scale = rayleigh.fit(amps)
            x = np.linspace(0, amps.max(), 200)
            pdf = rayleigh.pdf(x, loc, scale)
            ax.plot(x, pdf, "r-", linewidth=2, label=f"Rayleigh fit (σ={scale:.2e})")
            ax.legend()

        ax.set_xlabel("Residual Amplitude", fontsize=config.effective_label_size)
        ax.set_ylabel("Density", fontsize=config.effective_label_size)
        ax.set_title(title, fontsize=config.effective_title_size)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved residual histogram: {output}")
        plt.close(fig)

    return fig


def plot_residual_complex_scatter(
    data: ResidualData,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Residual Complex Scatter",
    max_points: int = 10000,
    context: PlotContext | None = None,
    interactive: bool | None = None,
) -> Figure:
    """Plot residuals in the complex plane (Real vs Imaginary).

    For well-calibrated data, residuals should be centered at (0, 0)
    with circular Gaussian scatter.

    Parameters
    ----------
    data :
        ResidualData from extract_residuals_from_ms()
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    max_points :
        Maximum points to plot (subsample if needed)
    context :
        Plot context for format selection
    interactive :
        Override interactive format selection

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    residuals = data.residuals.flatten()
    valid = ~np.isnan(residuals)
    residuals_valid = residuals[valid]

    # Subsample if too many points
    if len(residuals_valid) > max_points:
        indices = np.random.choice(len(residuals_valid), max_points, replace=False)
        residuals_sub = residuals_valid[indices]
    else:
        residuals_sub = residuals_valid

    real_parts = np.real(residuals_sub)
    imag_parts = np.imag(residuals_sub)

    # Check if interactive mode
    use_interactive = should_generate_interactive(context, interactive)

    if use_interactive:
        from dsa110_contimg.core.visualization.vega_specs import (
            create_scatter_spec,
            save_vega_spec,
        )

        spec = create_scatter_spec(
            x_data=real_parts,
            y_data=imag_parts,
            x_label="Real",
            y_label="Imaginary",
            title=title,
        )

        if output:
            json_path = Path(str(output).replace(".png", ".vega.json"))
            save_vega_spec(spec, json_path)
            logger.info(f"Saved interactive complex scatter: {json_path}")

        return None

    # Static matplotlib plot
    fig, ax = plt.subplots(figsize=(config.figsize[0], config.figsize[0]), dpi=config.dpi)

    ax.scatter(
        real_parts,
        imag_parts,
        s=1,
        alpha=0.2,
        c="steelblue",
    )

    # Mark origin
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

    # Add circles at 1σ, 2σ, 3σ
    std_val = np.std(np.abs(residuals_sub))
    for n_sigma in [1, 2, 3]:
        circle = plt.Circle(
            (0, 0),
            n_sigma * std_val,
            fill=False,
            linestyle="--",
            color="red",
            alpha=0.5,
            label=f"{n_sigma}σ" if n_sigma == 1 else None,
        )
        ax.add_patch(circle)

    # Mark centroid
    centroid_real = np.mean(real_parts)
    centroid_imag = np.mean(imag_parts)
    ax.scatter(
        [centroid_real],
        [centroid_imag],
        s=100,
        marker="x",
        c="red",
        linewidths=2,
        label=f"Centroid ({centroid_real:.2e}, {centroid_imag:.2e})",
    )

    ax.set_xlabel("Real", fontsize=config.effective_label_size)
    ax.set_ylabel("Imaginary", fontsize=config.effective_label_size)
    ax.set_title(title, fontsize=config.effective_title_size)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved complex scatter plot: {output}")
        plt.close(fig)

    return fig


def plot_residual_per_antenna(
    data: ResidualData,
    stats: ResidualStatistics,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Per-Antenna Residual RMS",
    highlight_threshold: float = 2.0,
    context: PlotContext | None = None,
    interactive: bool | None = None,
) -> Figure:
    """Plot per-antenna residual RMS to identify problematic antennas.

    Parameters
    ----------
    data :
        ResidualData from extract_residuals_from_ms()
    stats :
        ResidualStatistics from compute_residual_statistics()
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    highlight_threshold :
        Highlight antennas with RMS > threshold * median
    context :
        Plot context for format selection
    interactive :
        Override interactive format selection

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    if stats.per_antenna_rms is None or len(stats.per_antenna_rms) == 0:
        logger.warning("No per-antenna RMS data available")
        return None

    antennas = sorted(stats.per_antenna_rms.keys())
    rms_values = [stats.per_antenna_rms[ant] for ant in antennas]
    median_rms = np.median(rms_values)

    # Check if interactive mode
    use_interactive = should_generate_interactive(context, interactive)

    if use_interactive:
        from dsa110_contimg.core.visualization.vega_specs import save_vega_spec

        values = [
            {
                "antenna": int(ant),
                "rms": float(rms),
                "above_threshold": rms > highlight_threshold * median_rms,
            }
            for ant, rms in zip(antennas, rms_values)
        ]

        spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "description": "Per-antenna residual RMS",
            "title": title,
            "width": 600,
            "height": 400,
            "data": {"values": values},
            "mark": {"type": "bar", "tooltip": True},
            "encoding": {
                "x": {
                    "field": "antenna",
                    "type": "ordinal",
                    "title": "Antenna",
                },
                "y": {
                    "field": "rms",
                    "type": "quantitative",
                    "title": "Residual RMS",
                },
                "color": {
                    "field": "above_threshold",
                    "type": "nominal",
                    "scale": {"domain": [False, True], "range": ["steelblue", "red"]},
                    "legend": {"title": "Above threshold"},
                },
            },
        }

        if output:
            json_path = Path(str(output).replace(".png", ".vega.json"))
            save_vega_spec(spec, json_path)
            logger.info(f"Saved interactive per-antenna plot: {json_path}")

        return None

    # Static matplotlib plot
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    # Color bars based on threshold
    colors = [
        "red" if rms > highlight_threshold * median_rms else "steelblue" for rms in rms_values
    ]

    ax.bar(range(len(antennas)), rms_values, color=colors, alpha=0.7, edgecolor="black")
    ax.set_xticks(range(len(antennas)))
    ax.set_xticklabels(antennas, fontsize=8)

    # Threshold line
    ax.axhline(
        median_rms,
        color="gray",
        linestyle="-",
        linewidth=1,
        label=f"Median RMS = {median_rms:.2e}",
    )
    ax.axhline(
        highlight_threshold * median_rms,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"{highlight_threshold}× median threshold",
    )

    ax.set_xlabel("Antenna", fontsize=config.effective_label_size)
    ax.set_ylabel("Residual RMS", fontsize=config.effective_label_size)
    ax.set_title(title, fontsize=config.effective_title_size)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved per-antenna residual plot: {output}")
        plt.close(fig)

    return fig


def generate_residual_diagnostic_report(
    ms_path: str | Path,
    output_dir: str | Path,
    config: FigureConfig | None = None,
    data_column: str = "CORRECTED_DATA",
    model_column: str = "MODEL_DATA",
    interactive: bool = False,
) -> dict[str, any]:
    """Generate comprehensive residual diagnostic plots and statistics.

    This is the main entry point for generating all residual diagnostics
    for a calibrated Measurement Set.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    output_dir :
        Directory for output plots
    config :
        Figure configuration
    data_column :
        Data column name
    model_column :
        Model column name
    interactive :
        Generate interactive Vega-Lite specs
    ms_path : Union[str, Path]
    output_dir : Union[str, config: Optional[FigureConfig]
         (Default value = None)

    Returns
    -------
    Dictionary with

    - statistics
        ResidualStatistics dict
    - plots
        List of generated plot paths
    - quality_assessment
        Summary assessment

    """
    ms_path = Path(ms_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ms_name = ms_path.stem

    logger.info(f"Generating residual diagnostics for {ms_path}")

    # Extract residuals
    data = extract_residuals_from_ms(
        ms_path,
        data_column=data_column,
        model_column=model_column,
    )

    # Compute statistics
    stats = compute_residual_statistics(data)

    generated_plots = []

    # Generate plots
    context = PlotContext.API if interactive else PlotContext.PIPELINE

    # 1. Amplitude vs baseline
    plot_path = output_dir / f"{ms_name}_residual_vs_baseline.png"
    plot_residual_amplitude_vs_baseline(
        data, output=plot_path, config=config, context=context, interactive=interactive
    )
    generated_plots.append(str(plot_path))

    # 2. Phase vs time
    plot_path = output_dir / f"{ms_name}_residual_phase_vs_time.png"
    plot_residual_phase_vs_time(
        data, output=plot_path, config=config, context=context, interactive=interactive
    )
    generated_plots.append(str(plot_path))

    # 3. Histogram (real/imag)
    plot_path = output_dir / f"{ms_name}_residual_histogram.png"
    plot_residual_histogram(
        data, output=plot_path, config=config, context=context, interactive=interactive
    )
    generated_plots.append(str(plot_path))

    # 4. Complex scatter
    plot_path = output_dir / f"{ms_name}_residual_complex.png"
    plot_residual_complex_scatter(
        data, output=plot_path, config=config, context=context, interactive=interactive
    )
    generated_plots.append(str(plot_path))

    # 5. Per-antenna RMS
    plot_path = output_dir / f"{ms_name}_residual_per_antenna.png"
    plot_residual_per_antenna(
        data, stats, output=plot_path, config=config, context=context, interactive=interactive
    )
    generated_plots.append(str(plot_path))

    # Quality assessment
    quality_assessment = _assess_residual_quality(stats)

    return {
        "statistics": stats.to_dict(),
        "plots": generated_plots,
        "quality_assessment": quality_assessment,
    }


def _assess_residual_quality(stats: ResidualStatistics) -> dict[str, any]:
    """Assess calibration quality based on residual statistics.

    Parameters
    ----------

    Returns
    -------
        Dictionary with quality assessment and recommendations

    """
    issues = []
    warnings = []
    quality_score = 100  # Start at 100, deduct for issues

    # Check centroid offset (should be near 0)
    centroid_offset = np.sqrt(stats.mean_real**2 + stats.mean_imag**2)
    if centroid_offset > 0.1 * stats.rms_amplitude:
        issues.append(
            f"Residual centroid offset ({centroid_offset:.2e}) is >10% of RMS - "
            "possible systematic calibration error"
        )
        quality_score -= 20

    # Check for outliers
    if stats.outlier_fraction > 0.05:
        issues.append(
            f"High outlier fraction ({stats.outlier_fraction:.1%}) - "
            "possible bad data or calibration failures"
        )
        quality_score -= 15
    elif stats.outlier_fraction > 0.01:
        warnings.append(f"Moderate outlier fraction ({stats.outlier_fraction:.1%})")
        quality_score -= 5

    # Check flag fraction
    if stats.flag_fraction > 0.5:
        warnings.append(
            f"High flag fraction ({stats.flag_fraction:.1%}) - limited data for residual analysis"
        )
        quality_score -= 10

    # Check phase scatter
    if stats.std_phase_deg > 30:
        warnings.append(
            f"High phase scatter ({stats.std_phase_deg:.1f}°) - possible phase calibration issues"
        )
        quality_score -= 10

    # Check per-antenna RMS spread
    if stats.per_antenna_rms:
        rms_values = list(stats.per_antenna_rms.values())
        rms_spread = np.max(rms_values) / np.min(rms_values) if np.min(rms_values) > 0 else np.inf
        if rms_spread > 3:
            bad_antennas = [
                ant for ant, rms in stats.per_antenna_rms.items() if rms > 2 * np.median(rms_values)
            ]
            issues.append(
                f"Large per-antenna RMS spread (max/min = {rms_spread:.1f}). "
                f"Problem antennas: {bad_antennas}"
            )
            quality_score -= 15

    quality_score = max(0, quality_score)

    if quality_score >= 90:
        overall = "EXCELLENT"
    elif quality_score >= 70:
        overall = "GOOD"
    elif quality_score >= 50:
        overall = "ACCEPTABLE"
    else:
        overall = "POOR"

    return {
        "overall": overall,
        "score": quality_score,
        "issues": issues,
        "warnings": warnings,
        "recommendation": (
            "Calibration looks good, proceed with imaging."
            if overall in ["EXCELLENT", "GOOD"]
            else "Review calibration - consider re-running with different parameters."
        ),
    }
