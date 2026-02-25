"""
Closure phase plotting utilities.

Closure phases are robust calibration-independent observables formed from
phase measurements around closed triangles of baselines. They are essential
diagnostics for:
- Self-calibration validation (closure phases should be zero for point sources)
- Detecting complex source structure
- Identifying problematic antennas
- Validating phase calibration quality

Closure phase for a triangle of antennas (i, j, k):
    CP_ijk = φ_ij + φ_jk + φ_ki

where φ_ij is the visibility phase on baseline i-j.

DSA-110 Array:
- 117 antenna indices in data files
- 96 active antennas used in observations
- 4560 unique baselines (96 * 95 / 2)
- ~140k possible closure triangles

References
----------
- Thompson, Moran, Swenson: "Interferometry and Synthesis in Radio Astronomy"
- eht-imaging closure phase analysis patterns
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from dsa110_contimg.core.visualization.config import FigureConfig, PlotStyle

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _setup_matplotlib() -> None:
    """Configure matplotlib for headless operation."""
    import matplotlib

    matplotlib.use("Agg")


def compute_closure_phases(
    visibility: NDArray,
    antenna1: NDArray,
    antenna2: NDArray,
    triangles: list[tuple[int, int, int]] | None = None,
    n_antennas: int | None = None,
) -> dict[tuple[int, int, int], NDArray]:
    """Compute closure phases for antenna triangles.

    Parameters
    ----------
    visibility :
        Complex visibility array (Nvis,) or (Nvis, Nfreq) or (Nvis, Nfreq, Npol)
    antenna1 :
        Antenna 1 index for each visibility (Nvis,)
    antenna2 :
        Antenna 2 index for each visibility (Nvis,)
    triangles :
        List of antenna triangles (i, j, k). If None, auto-generate.
    n_antennas :
        Number of antennas (auto-detected if None)

    Returns
    -------
        Dictionary mapping (ant_i, ant_j, ant_k) -> closure phases array

    """
    vis = np.asarray(visibility)
    ant1 = np.asarray(antenna1).astype(int)
    ant2 = np.asarray(antenna2).astype(int)

    # Handle multi-dimensional visibilities (average over freq/pol)
    if vis.ndim > 1:
        # Average over all but first dimension.
        # Avoid `np.nanmean` on the full array since it forces a large copy via
        # numpy's NaN replacement path; instead compute a fast mean and only
        # fall back to `nanmean` for rows that actually contain NaNs.
        vis2 = vis.reshape(vis.shape[0], -1)
        row_mean = vis2.mean(axis=1)

        nan_rows = np.isnan(row_mean)
        if np.any(nan_rows):
            row_mean[nan_rows] = np.nanmean(vis2[nan_rows], axis=1)

        vis = row_mean

    # Extract phases
    phases = np.angle(vis)  # radians

    # Build baseline lookup: (ant1, ant2) -> indices
    if n_antennas is None:
        n_antennas = max(ant1.max(), ant2.max()) + 1

    baseline_phases: dict[tuple[int, int], NDArray] = {}
    for i in range(len(ant1)):
        a1, a2 = ant1[i], ant2[i]
        key = (min(a1, a2), max(a1, a2))
        if key not in baseline_phases:
            baseline_phases[key] = []
        # Store with sign convention: phase(a1, a2) = -phase(a2, a1)
        sign = 1 if a1 < a2 else -1
        baseline_phases[key].append(sign * phases[i])

    # Convert to arrays
    for key in baseline_phases:
        baseline_phases[key] = np.array(baseline_phases[key])

    # Generate triangles if not provided
    if triangles is None:
        triangles = []
        antennas = sorted(set(ant1) | set(ant2))
        # Generate all possible triangles
        for i, a in enumerate(antennas):
            for j, b in enumerate(antennas[i + 1 :], i + 1):
                for c in antennas[j + 1 :]:
                    triangles.append((a, b, c))
        # Limit to manageable number
        if len(triangles) > 100:
            triangles = triangles[:100]

    # Compute closure phases for each triangle
    closure_phases = {}
    for tri in triangles:
        i, j, k = tri

        # Get baseline phases with proper sign convention
        bl_ij = (min(i, j), max(i, j))
        bl_jk = (min(j, k), max(j, k))
        bl_ki = (min(k, i), max(k, i))

        if (
            bl_ij not in baseline_phases
            or bl_jk not in baseline_phases
            or bl_ki not in baseline_phases
        ):
            continue

        # Get phases (need to handle sign convention)
        phi_ij = baseline_phases[bl_ij] * (1 if i < j else -1)
        phi_jk = baseline_phases[bl_jk] * (1 if j < k else -1)
        phi_ki = baseline_phases[bl_ki] * (1 if k < i else -1)

        # Need same length arrays
        min_len = min(len(phi_ij), len(phi_jk), len(phi_ki))
        if min_len == 0:
            continue

        cp = phi_ij[:min_len] + phi_jk[:min_len] + phi_ki[:min_len]
        # Wrap to [-π, π]
        cp = np.arctan2(np.sin(cp), np.cos(cp))
        closure_phases[tri] = cp

    return closure_phases


def plot_closure_phase_histogram(
    closure_phases: dict[tuple[int, int, int], NDArray],
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Closure Phase Distribution",
    bins: int = 72,
    show_gaussian: bool = True,
) -> Figure:
    """Plot histogram of all closure phases.

    For a well-calibrated point source, closure phases should be tightly
    clustered around zero.

    Parameters
    ----------
    closure_phases : dict
        Dictionary from compute_closure_phases().
    output : str or Path, optional
        Output file path.
    config : FigureConfig, optional
        Figure configuration.
    title : str
        Plot title.
    bins : int
        Number of histogram bins.
    show_gaussian : bool
        Overlay Gaussian fit.

    Returns
    -------
    Figure
        matplotlib Figure object.
    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from scipy import stats

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    # Concatenate all closure phases
    all_cp = np.concatenate([cp for cp in closure_phases.values()])
    all_cp_deg = np.degrees(all_cp)

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    # Histogram
    ax.hist(
        all_cp_deg,
        bins=bins,
        range=(-180, 180),
        color="steelblue",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
        density=True,
    )

    # Statistics
    mean_cp = np.mean(all_cp_deg)
    std_cp = np.std(all_cp_deg)
    median_cp = np.median(all_cp_deg)

    # Gaussian fit overlay
    if show_gaussian:
        x = np.linspace(-180, 180, 200)
        gaussian = stats.norm.pdf(x, mean_cp, std_cp)
        ax.plot(x, gaussian, "r-", linewidth=2, label=f"Gaussian (σ={std_cp:.1f}°)")

    ax.axvline(0, color="gray", linestyle="--", alpha=0.7, linewidth=1)
    ax.axvline(
        mean_cp, color="red", linestyle="-", alpha=0.7, linewidth=1.5, label=f"Mean={mean_cp:.1f}°"
    )

    ax.set_xlabel("Closure Phase (degrees)", fontsize=config.effective_label_size)
    ax.set_ylabel("Probability Density", fontsize=config.effective_label_size)
    ax.set_title(title, fontsize=config.effective_title_size)
    ax.set_xlim(-180, 180)

    # Statistics annotation
    stats_text = (
        f"N triangles: {len(closure_phases)}\n"
        f"N samples: {len(all_cp):,}\n"
        f"Mean: {mean_cp:.2f}°\n"
        f"Median: {median_cp:.2f}°\n"
        f"Std: {std_cp:.2f}°"
    )
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=config.effective_tick_size,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    ax.legend(fontsize=config.effective_tick_size, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved closure phase histogram: {output}")
        plt.close(fig)

    return fig


def plot_closure_phase_vs_time(
    times: NDArray,
    closure_phases: dict[tuple[int, int, int], NDArray],
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Closure Phase vs Time",
    max_triangles: int = 10,
    antenna_names: list[str] | None = None,
) -> Figure:
    """Plot closure phases as function of time for selected triangles.

    Parameters
    ----------
    times :
        Time array (MJD or seconds)
    closure_phases :
        Dictionary from compute_closure_phases()
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    max_triangles :
        Maximum number of triangles to display
    antenna_names :
        List of antenna names for labels

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    times = np.asarray(times)
    t_min = (times - times.min()) * 24 * 60 if times.max() > 1000 else times / 60

    # Select triangles with most data
    triangles = sorted(closure_phases.keys(), key=lambda t: len(closure_phases[t]), reverse=True)[
        :max_triangles
    ]

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    colors = plt.cm.tab10(np.linspace(0, 1, len(triangles)))

    for idx, tri in enumerate(triangles):
        cp = closure_phases[tri]
        cp_deg = np.degrees(cp)

        # Match time array length
        t_plot = t_min[: len(cp)] if len(t_min) >= len(cp) else t_min

        if antenna_names is not None:
            label = f"{antenna_names[tri[0]]}-{antenna_names[tri[1]]}-{antenna_names[tri[2]]}"
        else:
            label = f"{tri[0]}-{tri[1]}-{tri[2]}"

        ax.scatter(t_plot[: len(cp_deg)], cp_deg, s=3, alpha=0.5, c=[colors[idx]], label=label)

    ax.set_xlabel("Time (minutes)", fontsize=config.effective_label_size)
    ax.set_ylabel("Closure Phase (degrees)", fontsize=config.effective_label_size)
    ax.set_title(title, fontsize=config.effective_title_size)
    ax.set_ylim(-180, 180)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=config.effective_tick_size - 2, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved closure phase vs time: {output}")
        plt.close(fig)

    return fig


def plot_closure_phase_per_triangle(
    closure_phases: dict[tuple[int, int, int], NDArray],
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Closure Phase by Triangle",
    max_triangles: int = 20,
    antenna_names: list[str] | None = None,
    threshold_deg: float = 10.0,
) -> Figure:
    """Plot closure phase statistics per triangle as bar chart.

    Useful for identifying triangles with anomalous closure phases,
    which may indicate antenna or baseline problems.

    Parameters
    ----------
    closure_phases : dict
        Dictionary from compute_closure_phases().
    output : str or Path, optional
        Output file path.
    config : FigureConfig, optional
        Figure configuration.
    title : str
        Plot title.
    max_triangles : int
        Maximum triangles to show.
    antenna_names : list of str, optional
        List of antenna names.
    threshold_deg : float
        Threshold for flagging high RMS (degrees).

    Returns
    -------
    Figure
        matplotlib Figure object.
    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    # Compute statistics per triangle
    stats = []
    for tri, cp in closure_phases.items():
        cp_deg = np.degrees(cp)
        stats.append(
            {
                "triangle": tri,
                "mean": np.mean(cp_deg),
                "std": np.std(cp_deg),
                "median": np.median(cp_deg),
                "n": len(cp),
            }
        )

    # Sort by RMS (std)
    stats = sorted(stats, key=lambda x: x["std"], reverse=True)[:max_triangles]

    fig, ax = plt.subplots(figsize=(max(config.figsize[0], len(stats) * 0.5), config.figsize[1]))

    x = np.arange(len(stats))
    stds = [s["std"] for s in stats]

    # Color by whether RMS exceeds threshold
    colors = ["red" if s > threshold_deg else "steelblue" for s in stds]

    ax.bar(x, stds, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)

    # Triangle labels
    if antenna_names is not None:
        labels = [
            f"{antenna_names[s['triangle'][0]]}-{antenna_names[s['triangle'][1]]}-{antenna_names[s['triangle'][2]]}"
            for s in stats
        ]
    else:
        labels = [f"{s['triangle'][0]}-{s['triangle'][1]}-{s['triangle'][2]}" for s in stats]

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=config.effective_tick_size - 2)

    ax.axhline(
        threshold_deg, color="red", linestyle="--", alpha=0.7, label=f"Threshold ({threshold_deg}°)"
    )

    ax.set_xlabel("Triangle", fontsize=config.effective_label_size)
    ax.set_ylabel("Closure Phase RMS (degrees)", fontsize=config.effective_label_size)
    ax.set_title(title, fontsize=config.effective_title_size)
    ax.legend(fontsize=config.effective_tick_size)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved closure phase per triangle: {output}")
        plt.close(fig)

    return fig


def plot_closure_phase_antenna_contribution(
    closure_phases: dict[tuple[int, int, int], NDArray],
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Antenna Contribution to Closure Phase RMS",
    n_antennas: int | None = None,
    antenna_names: list[str] | None = None,
    threshold_deg: float = 15.0,
) -> Figure:
    """Plot per-antenna contribution to closure phase scatter.

    Antennas appearing in many high-RMS triangles are likely problematic.

    Parameters
    ----------
    closure_phases : dict
        Dictionary from compute_closure_phases().
    output : str or Path, optional
        Output file path.
    config : FigureConfig, optional
        Figure configuration.
    title : str
        Plot title.
    n_antennas : int, optional
        Number of antennas.
    antenna_names : list of str, optional
        List of antenna names.
    threshold_deg : float
        Threshold for flagging.

    Returns
    -------
    Figure
        matplotlib Figure object.
    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    # Compute RMS contribution per antenna
    if n_antennas is None:
        all_ants = set()
        for tri in closure_phases.keys():
            all_ants.update(tri)
        n_antennas = max(all_ants) + 1

    ant_rms_sum = np.zeros(n_antennas)
    ant_count = np.zeros(n_antennas)

    for tri, cp in closure_phases.items():
        rms = np.std(np.degrees(cp))
        for ant in tri:
            ant_rms_sum[ant] += rms
            ant_count[ant] += 1

    # Average RMS per antenna
    ant_rms_avg = np.divide(
        ant_rms_sum, ant_count, where=ant_count > 0, out=np.zeros_like(ant_rms_sum)
    )

    # Only show antennas with data
    active_ants = np.where(ant_count > 0)[0]

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    colors = ["red" if ant_rms_avg[a] > threshold_deg else "steelblue" for a in active_ants]

    ax.bar(
        range(len(active_ants)),
        ant_rms_avg[active_ants],
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )

    if antenna_names is not None:
        labels = [antenna_names[a] if a < len(antenna_names) else f"Ant {a}" for a in active_ants]
    else:
        labels = [f"Ant {a}" for a in active_ants]

    ax.set_xticks(range(len(active_ants)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=config.effective_tick_size - 2)

    ax.axhline(
        threshold_deg,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Flag threshold ({threshold_deg}°)",
    )

    ax.set_xlabel("Antenna", fontsize=config.effective_label_size)
    ax.set_ylabel("Mean Closure Phase RMS (degrees)", fontsize=config.effective_label_size)
    ax.set_title(title, fontsize=config.effective_title_size)
    ax.legend(fontsize=config.effective_tick_size)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved antenna closure phase contribution: {output}")
        plt.close(fig)

    return fig


def extract_closure_phases_from_ms(
    ms_path: str | Path,
    *,
    max_rows: int | None = None,
    row_stride: int = 1,
    max_channels: int | None = None,
    max_pols: int | None = None,
) -> dict:
    """Extract data needed for closure phase computation from Measurement Set.

    Parameters
    ----------
    ms_path : str or Path
        Path to Measurement Set.
    max_rows : int, optional
        Maximum number of rows to read.
    row_stride : int
        Row stride for reading data. Default 1.
    max_channels : int, optional
        Maximum number of channels to read.
    max_pols : int, optional
        Maximum number of polarizations to read.

    Returns
    -------
    dict
        Dictionary with visibility, antenna1, antenna2, time arrays.
    """
    try:
        from casacore.tables import table
    except ImportError as exc:
        raise ImportError("casacore required. Install with: pip install python-casacore") from exc

    ms_path = Path(ms_path)
    result = {}

    with table(str(ms_path), readonly=True, ack=False) as tb:
        if row_stride < 1:
            raise ValueError("row_stride must be >= 1")

        nrow = tb.nrows()
        if max_rows is not None:
            nrow = min(nrow, int(max_rows))

        getcol_kwargs = {"startrow": 0, "nrow": nrow, "rowincr": int(row_stride)}

        result["antenna1"] = tb.getcol("ANTENNA1", **getcol_kwargs)
        result["antenna2"] = tb.getcol("ANTENNA2", **getcol_kwargs)
        result["time"] = tb.getcol("TIME", **getcol_kwargs)

        if "CORRECTED_DATA" in tb.colnames():
            vis_col = "CORRECTED_DATA"
        elif "DATA" in tb.colnames():
            vis_col = "DATA"
        else:
            raise ValueError("No visibility data column found in MS")

        if max_channels is None and max_pols is None:
            result["visibility"] = tb.getcol(vis_col, **getcol_kwargs)
        else:
            # Read a slice of the per-row visibility cell to avoid loading the full
            # frequency/polarization cube for large MS files.
            example_cell = tb.getcell(vis_col, 0)
            if example_cell.ndim == 1:
                n_dim0 = example_cell.shape[0]
                trc = [min(max_channels or n_dim0, n_dim0) - 1]
                blc = [0]
                inc = [1]
            else:
                n_dim0, n_dim1 = example_cell.shape[0], example_cell.shape[1]
                trc = [
                    min(max_channels or n_dim0, n_dim0) - 1,
                    min(max_pols or n_dim1, n_dim1) - 1,
                ]
                blc = [0, 0]
                inc = [1, 1]

            result["visibility"] = tb.getcolslice(
                vis_col,
                blc=blc,
                trc=trc,
                inc=inc,
                **getcol_kwargs,
            )

    # Get antenna names
    with table(str(ms_path / "ANTENNA"), readonly=True, ack=False) as ant_tb:
        result["antenna_names"] = list(ant_tb.getcol("NAME"))

    return result
