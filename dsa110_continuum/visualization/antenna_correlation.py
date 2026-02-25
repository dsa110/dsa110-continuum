"""
Antenna gain correlation analysis and visualization.

Provides tools for computing and visualizing correlations between antenna gains,
which helps identify:
- Systematic calibration errors affecting groups of antennas
- Hardware issues creating correlated behavior
- Environmental effects (temperature, humidity)
- Reference antenna stability

Key visualizations:
- Gain correlation matrix (heatmap showing pairwise correlations)
- Correlation network graph (showing significant correlations)
- Temporal correlation evolution (how correlations change over time)
- Anomaly detection (identifying unusually correlated antenna pairs)

Usage:
    from dsa110_contimg.core.visualization.antenna_correlation import (
        extract_gains_from_caltable,
        compute_gain_correlation_matrix,
        plot_gain_correlation_matrix,
        plot_correlation_network,
        identify_correlated_groups,
    )

    # Load gains
    gains = extract_gains_from_caltable("calibration.bcal")

    # Compute correlations
    corr_matrix, stats = compute_gain_correlation_matrix(gains)

    # Visualize
    plot_gain_correlation_matrix(corr_matrix, output="correlation_matrix.png")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

from dsa110_contimg.core.visualization.config import FigureConfig, PlotStyle

logger = logging.getLogger(__name__)


def _setup_matplotlib() -> None:
    """Configure matplotlib for headless operation."""
    import matplotlib

    matplotlib.use("Agg")


@dataclass
class AntennaGainData:
    """Container for antenna gain data."""

    antenna_ids: list[int]
    times: NDArray[np.floating]
    frequencies: NDArray[np.floating]
    gains_amp: NDArray[np.floating]
    gains_phase: NDArray[np.floating]
    flags: NDArray[np.bool_]
    polarizations: list[str] = field(default_factory=lambda: ["XX", "YY"])
    source_file: str | None = None

    @property
    def n_antennas(self) -> int:
        return len(self.antenna_ids)

    @property
    def n_times(self) -> int:
        return len(self.times)

    @property
    def n_freqs(self) -> int:
        return len(self.frequencies)

    @property
    def n_pols(self) -> int:
        return self.gains_amp.shape[-1] if self.gains_amp.ndim > 3 else 1


@dataclass
class CorrelationStatistics:
    """Statistics from gain correlation analysis."""

    mean_correlation: float
    median_correlation: float
    max_correlation: float
    min_correlation: float
    n_significant_pairs: int
    significant_pairs: list[tuple[int, int, float]]
    clustered_groups: list[list[int]]
    reference_antenna: int
    problematic_antennas: list[int]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean_correlation": self.mean_correlation,
            "median_correlation": self.median_correlation,
            "max_correlation": self.max_correlation,
            "min_correlation": self.min_correlation,
            "n_significant_pairs": self.n_significant_pairs,
            "significant_pairs": [
                {"ant1": a1, "ant2": a2, "correlation": c} for a1, a2, c in self.significant_pairs
            ],
            "clustered_groups": self.clustered_groups,
            "reference_antenna": self.reference_antenna,
            "problematic_antennas": self.problematic_antennas,
        }


def extract_gains_from_caltable(
    caltable_path: str | Path,
    gain_type: str = "complex",
) -> AntennaGainData:
    """Extract gain data from a CASA calibration table.

    Parameters
    ----------
    caltable_path :
        Path to calibration table (.bcal, .gcal, etc.)
    gain_type :
        Type of gains ("complex", "amplitude", "phase")
    caltable_path : Union[str, Path]

    Returns
    -------
        AntennaGainData container

    """
    from casacore.tables import table

    caltable_path = str(caltable_path)

    with table(caltable_path, readonly=True, ack=False) as tb:
        gains = tb.getcol("CPARAM")  # Complex gains
        flags = tb.getcol("FLAG")
        times = tb.getcol("TIME")
        antenna1 = tb.getcol("ANTENNA1")

    # Get antenna info
    with table(f"{caltable_path}/ANTENNA", readonly=True, ack=False) as ant_tb:
        n_antennas = ant_tb.nrows()
        antenna_ids = list(range(n_antennas))

    # Get spectral window info
    with table(f"{caltable_path}/SPECTRAL_WINDOW", readonly=True, ack=False) as spw_tb:
        frequencies = spw_tb.getcol("CHAN_FREQ").flatten()

    # Reorganize data by antenna
    unique_times = np.unique(times)
    n_times = len(unique_times)
    n_freqs = gains.shape[1]
    n_pols = gains.shape[2]

    # Initialize arrays
    gains_amp = np.zeros((n_antennas, n_times, n_freqs, n_pols))
    gains_phase = np.zeros((n_antennas, n_times, n_freqs, n_pols))
    flags_arr = np.ones((n_antennas, n_times, n_freqs, n_pols), dtype=bool)

    # Fill arrays
    time_to_idx = {t: i for i, t in enumerate(unique_times)}
    for i in range(len(antenna1)):
        ant = antenna1[i]
        t_idx = time_to_idx[times[i]]
        gains_amp[ant, t_idx, :, :] = np.abs(gains[i])
        gains_phase[ant, t_idx, :, :] = np.angle(gains[i], deg=True)
        flags_arr[ant, t_idx, :, :] = flags[i]

    # Determine polarization names
    pols = ["XX", "YY"] if n_pols == 2 else ["I"]
    if n_pols == 4:
        pols = ["XX", "XY", "YX", "YY"]

    return AntennaGainData(
        antenna_ids=antenna_ids,
        times=unique_times,
        frequencies=frequencies,
        gains_amp=gains_amp,
        gains_phase=gains_phase,
        flags=flags_arr,
        polarizations=pols,
        source_file=caltable_path,
    )


def compute_gain_correlation_matrix(
    data: AntennaGainData,
    component: str = "amplitude",
    polarization: int = 0,
    freq_avg: bool = True,
    significance_threshold: float = 0.5,
) -> tuple[NDArray[np.floating], CorrelationStatistics]:
    """Compute pairwise correlation matrix between antenna gains.

    Parameters
    ----------
    data :
        AntennaGainData container
    component :
        "amplitude" or "phase"
    polarization :
        Polarization index to use
    freq_avg :
        Average over frequency before computing correlation
    significance_threshold :
        Threshold for significant correlations

    Returns
    -------
        Tuple of (correlation_matrix, CorrelationStatistics)

    """
    n_ant = data.n_antennas

    # Select component
    if component == "amplitude":
        gains = data.gains_amp[:, :, :, polarization]
    else:
        gains = data.gains_phase[:, :, :, polarization]

    flags = data.flags[:, :, :, polarization]

    # Mask flagged data
    gains = np.ma.array(gains, mask=flags)

    # Average over frequency if requested
    if freq_avg:
        gains = np.ma.mean(gains, axis=2)  # Shape: (n_ant, n_time)
    else:
        # Flatten freq and time dimensions
        gains = gains.reshape(n_ant, -1)

    # Compute correlation matrix
    corr_matrix = np.zeros((n_ant, n_ant))

    for i in range(n_ant):
        for j in range(i, n_ant):
            g1 = gains[i].compressed()
            g2 = gains[j].compressed()

            # Need at least some overlapping valid points
            if len(g1) < 3 or len(g2) < 3:
                corr_matrix[i, j] = np.nan
                corr_matrix[j, i] = np.nan
                continue

            # Handle case where arrays have different lengths
            min_len = min(len(g1), len(g2))
            if min_len < 3:
                corr_matrix[i, j] = np.nan
                corr_matrix[j, i] = np.nan
                continue

            # Compute Pearson correlation
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                corr = np.corrcoef(g1[:min_len], g2[:min_len])[0, 1]
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

    # Compute statistics
    off_diag = corr_matrix[np.triu_indices(n_ant, k=1)]
    off_diag_valid = off_diag[~np.isnan(off_diag)]

    # Find significant pairs
    significant_pairs = []
    for i in range(n_ant):
        for j in range(i + 1, n_ant):
            if not np.isnan(corr_matrix[i, j]) and abs(corr_matrix[i, j]) > significance_threshold:
                significant_pairs.append(
                    (data.antenna_ids[i], data.antenna_ids[j], corr_matrix[i, j])
                )

    # Find clustered groups using simple threshold-based clustering
    clustered_groups = _find_correlated_groups(
        corr_matrix, data.antenna_ids, threshold=significance_threshold
    )

    # Find reference antenna (lowest mean correlation with others)
    mean_corr_per_ant = np.nanmean(np.abs(corr_matrix), axis=1)
    # Exclude diagonal
    for i in range(n_ant):
        mean_corr_per_ant[i] = np.nanmean(np.abs(np.delete(corr_matrix[i], i)))

    ref_ant_idx = np.argmin(mean_corr_per_ant)
    reference_antenna = data.antenna_ids[ref_ant_idx]

    # Find problematic antennas (unusually high correlations)
    problematic_threshold = np.nanmedian(mean_corr_per_ant) + 2 * np.nanstd(mean_corr_per_ant)
    problematic_antennas = [
        data.antenna_ids[i] for i in range(n_ant) if mean_corr_per_ant[i] > problematic_threshold
    ]

    stats = CorrelationStatistics(
        mean_correlation=float(np.nanmean(off_diag_valid)) if len(off_diag_valid) > 0 else 0.0,
        median_correlation=float(np.nanmedian(off_diag_valid)) if len(off_diag_valid) > 0 else 0.0,
        max_correlation=float(np.nanmax(off_diag_valid)) if len(off_diag_valid) > 0 else 0.0,
        min_correlation=float(np.nanmin(off_diag_valid)) if len(off_diag_valid) > 0 else 0.0,
        n_significant_pairs=len(significant_pairs),
        significant_pairs=significant_pairs,
        clustered_groups=clustered_groups,
        reference_antenna=reference_antenna,
        problematic_antennas=problematic_antennas,
    )

    return corr_matrix, stats


def _find_correlated_groups(
    corr_matrix: NDArray[np.floating],
    antenna_ids: list[int],
    threshold: float = 0.5,
) -> list[list[int]]:
    """Find groups of antennas with high mutual correlation.

    Uses simple connectivity-based grouping.

    Parameters
    ----------
    corr_matrix :
        Correlation matrix
    antenna_ids :
        List of antenna IDs
    threshold :
        Correlation threshold for grouping
    corr_matrix: "NDArray[np.floating]" :

    antenna_ids: List[int] :

    Returns
    -------
        List of antenna ID groups

    """
    n_ant = len(antenna_ids)

    # Build adjacency based on high correlation
    adj = np.abs(corr_matrix) > threshold
    np.fill_diagonal(adj, False)

    # Simple connected components
    visited = set()
    groups = []

    for i in range(n_ant):
        if i in visited:
            continue

        # BFS from this node
        group = []
        queue = [i]

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            group.append(antenna_ids[node])

            # Add neighbors
            for j in range(n_ant):
                if adj[node, j] and j not in visited:
                    queue.append(j)

        if len(group) > 1:
            groups.append(sorted(group))

    return groups


def plot_gain_correlation_matrix(
    corr_matrix: NDArray[np.floating],
    antenna_ids: list[int] | None = None,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Antenna Gain Correlation Matrix",
    cmap: str = "RdBu_r",
    annotate: bool = True,
    highlight_threshold: float = 0.5,
    interactive: bool = False,
) -> Figure | dict[str, Any]:
    """Plot antenna gain correlation matrix as a heatmap.

    Parameters
    ----------
    corr_matrix :
        Correlation matrix (n_ant x n_ant)
    antenna_ids :
        List of antenna IDs for labels
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    cmap :
        Colormap name
    annotate :
        Show correlation values in cells
    highlight_threshold :
        Threshold to highlight significant correlations
    interactive :
        Return Vega-Lite spec instead of matplotlib Figure
    corr_matrix: "NDArray[np.floating]" :

    antenna_ids: Optional[List[int]] :
         (Default value = None)
    output : Optional[Union[str, Path]]
         (Default value = None)
    config: Optional[FigureConfig] :
         (Default value = None)

    Returns
    -------
        matplotlib Figure or Vega-Lite spec dict

    """
    if interactive:
        return _create_correlation_matrix_vega_spec(
            corr_matrix, antenna_ids=antenna_ids, title=title
        )

    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    n_ant = corr_matrix.shape[0]
    if antenna_ids is None:
        antenna_ids = list(range(n_ant))

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    # Create heatmap
    im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect="equal")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Correlation", fontsize=config.effective_label_size)

    # Set ticks
    ax.set_xticks(range(n_ant))
    ax.set_yticks(range(n_ant))
    ax.set_xticklabels(antenna_ids, fontsize=config.effective_tick_size - 1)
    ax.set_yticklabels(antenna_ids, fontsize=config.effective_tick_size - 1)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
    if annotate and n_ant <= 20:
        for i in range(n_ant):
            for j in range(n_ant):
                val = corr_matrix[i, j]
                if np.isnan(val):
                    continue

                # Choose text color based on background
                text_color = "white" if abs(val) > 0.5 else "black"

                # Highlight significant correlations
                fontweight = "bold" if abs(val) > highlight_threshold and i != j else "normal"

                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=config.effective_tick_size - 2,
                    fontweight=fontweight,
                )

    # Highlight diagonal
    for i in range(n_ant):
        ax.add_patch(
            plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2)
        )

    ax.set_xlabel("Antenna", fontsize=config.effective_label_size)
    ax.set_ylabel("Antenna", fontsize=config.effective_label_size)
    ax.set_title(title, fontsize=config.effective_title_size)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved correlation matrix: {output}")
        plt.close(fig)

    return fig


def plot_correlation_network(
    corr_matrix: NDArray[np.floating],
    antenna_ids: list[int] | None = None,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Antenna Correlation Network",
    threshold: float = 0.5,
    layout: str = "spring",
) -> Figure:
    """Plot antenna correlations as a network graph.

    Edges connect antenna pairs with correlation above threshold.
    Edge thickness and color indicate correlation strength.

    Parameters
    ----------
    corr_matrix :
        Correlation matrix
    antenna_ids :
        List of antenna IDs
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    threshold :
        Minimum correlation to show edge
    layout :
        Network layout ("spring", "circular", "shell")
    corr_matrix: "NDArray[np.floating]" :

    antenna_ids: Optional[List[int]] :
         (Default value = None)
    output : Optional[Union[str, Path]]
         (Default value = None)
    config: Optional[FigureConfig] :
         (Default value = None)

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx not available, falling back to simple visualization")
        return _plot_correlation_network_simple(
            corr_matrix, antenna_ids, output, config, title, threshold
        )

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    n_ant = corr_matrix.shape[0]
    if antenna_ids is None:
        antenna_ids = list(range(n_ant))

    # Build graph
    G = nx.Graph()
    G.add_nodes_from(antenna_ids)

    edges = []
    for i in range(n_ant):
        for j in range(i + 1, n_ant):
            corr = corr_matrix[i, j]
            if not np.isnan(corr) and abs(corr) > threshold:
                edges.append(
                    (antenna_ids[i], antenna_ids[j], {"weight": abs(corr), "sign": np.sign(corr)})
                )

    G.add_edges_from(edges)

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=500, node_color="lightblue", edgecolors="black")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=config.effective_tick_size)

    # Draw edges with varying width and color
    if edges:
        weights = [G[u][v]["weight"] for u, v in G.edges()]
        signs = [G[u][v]["sign"] for u, v in G.edges()]
        edge_colors = ["red" if s > 0 else "blue" for s in signs]
        widths = [w * 3 for w in weights]

        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            width=widths,
            edge_color=edge_colors,
            alpha=0.7,
        )

    # Add legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="red", linewidth=2, label="Positive correlation"),
        Line2D([0], [0], color="blue", linewidth=2, label="Negative correlation"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=config.effective_tick_size)

    ax.set_title(f"{title}\n(threshold = {threshold})", fontsize=config.effective_title_size)
    ax.axis("off")

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved correlation network: {output}")
        plt.close(fig)

    return fig


def _plot_correlation_network_simple(
    corr_matrix: NDArray[np.floating],
    antenna_ids: list[int] | None,
    output: str | Path | None,
    config: FigureConfig | None,
    title: str,
    threshold: float,
) -> Figure:
    """Simple network plot without networkx.

    Parameters
    ----------
    corr_matrix: "NDArray[np.floating]" :

    antenna_ids: Optional[List[int]] :

    output : Optional[Union[str, Path]]
    config: Optional[FigureConfig] :

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    n_ant = corr_matrix.shape[0]
    if antenna_ids is None:
        antenna_ids = list(range(n_ant))

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    # Arrange nodes in a circle
    angles = np.linspace(0, 2 * np.pi, n_ant, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)

    # Draw edges first
    for i in range(n_ant):
        for j in range(i + 1, n_ant):
            corr = corr_matrix[i, j]
            if not np.isnan(corr) and abs(corr) > threshold:
                color = "red" if corr > 0 else "blue"
                width = abs(corr) * 3
                ax.plot([x[i], x[j]], [y[i], y[j]], color=color, linewidth=width, alpha=0.5)

    # Draw nodes
    ax.scatter(x, y, s=500, c="lightblue", edgecolors="black", zorder=5)

    # Add labels
    for i, ant_id in enumerate(antenna_ids):
        ax.annotate(
            str(ant_id),
            (x[i], y[i]),
            ha="center",
            va="center",
            fontsize=config.effective_tick_size,
            zorder=6,
        )

    ax.set_title(f"{title}\n(threshold = {threshold})", fontsize=config.effective_title_size)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved correlation network: {output}")
        plt.close(fig)

    return fig


def plot_temporal_correlation_evolution(
    data: AntennaGainData,
    ant_pair: tuple[int, int],
    component: str = "amplitude",
    polarization: int = 0,
    window_size: int = 10,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str | None = None,
) -> Figure:
    """Plot how correlation between two antennas evolves over time.

    Uses rolling window correlation.

    Parameters
    ----------
    data :
        AntennaGainData container
    ant_pair :
        Tuple of antenna IDs to analyze
    component :
        "amplitude" or "phase"
    polarization :
        Polarization index
    window_size :
        Rolling window size (number of time points)
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    ant1_idx = data.antenna_ids.index(ant_pair[0])
    ant2_idx = data.antenna_ids.index(ant_pair[1])

    # Select component
    if component == "amplitude":
        gains = data.gains_amp[:, :, :, polarization]
    else:
        gains = data.gains_phase[:, :, :, polarization]

    flags = data.flags[:, :, :, polarization]

    # Average over frequency
    gains1 = np.ma.array(gains[ant1_idx], mask=flags[ant1_idx])
    gains2 = np.ma.array(gains[ant2_idx], mask=flags[ant2_idx])

    gains1_avg = np.ma.mean(gains1, axis=1)
    gains2_avg = np.ma.mean(gains2, axis=1)

    # Compute rolling correlation
    n_times = len(data.times)
    correlations = []
    time_centers = []

    for i in range(n_times - window_size + 1):
        g1 = gains1_avg[i : i + window_size].compressed()
        g2 = gains2_avg[i : i + window_size].compressed()

        if len(g1) >= 3 and len(g2) >= 3:
            min_len = min(len(g1), len(g2))
            corr = np.corrcoef(g1[:min_len], g2[:min_len])[0, 1]
        else:
            corr = np.nan

        correlations.append(corr)
        time_centers.append(data.times[i + window_size // 2])

    correlations = np.array(correlations)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=config.figsize, sharex=True)

    # Top panel: gains
    ax1.plot(data.times, gains1_avg, "-", label=f"Ant {ant_pair[0]}", alpha=0.7)
    ax1.plot(data.times, gains2_avg, "-", label=f"Ant {ant_pair[1]}", alpha=0.7)
    ax1.set_ylabel(f"Gain {component.title()}", fontsize=config.effective_label_size)
    ax1.legend(fontsize=config.effective_tick_size)
    ax1.grid(True, alpha=0.3)

    # Bottom panel: rolling correlation
    ax2.plot(time_centers, correlations, "o-", color="purple", markersize=4)
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.fill_between(
        time_centers,
        -1,
        1,
        where=np.abs(correlations) > 0.5,
        alpha=0.2,
        color="red",
        label="|corr| > 0.5",
    )
    ax2.set_ylabel("Correlation", fontsize=config.effective_label_size)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_xlabel("Time (MJD)", fontsize=config.effective_label_size)
    ax2.legend(fontsize=config.effective_tick_size)
    ax2.grid(True, alpha=0.3)

    if title is None:
        title = f"Temporal Correlation: Ant {ant_pair[0]} vs {ant_pair[1]}"
    fig.suptitle(title, fontsize=config.effective_title_size)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved temporal correlation plot: {output}")
        plt.close(fig)

    return fig


def plot_correlation_summary(
    corr_matrix: NDArray[np.floating],
    stats: CorrelationStatistics,
    antenna_ids: list[int] | None = None,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Correlation Analysis Summary",
) -> Figure:
    """Create summary figure with correlation matrix and statistics.

    Parameters
    ----------
    corr_matrix :
        Correlation matrix
    stats :
        CorrelationStatistics object
    antenna_ids :
        List of antenna IDs
    output :
        Output file path
    config :
        Figure configuration
    title :
        Overall title
    corr_matrix: "NDArray[np.floating]" :

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    n_ant = corr_matrix.shape[0]
    if antenna_ids is None:
        antenna_ids = list(range(n_ant))

    fig = plt.figure(figsize=(config.figsize[0] * 1.5, config.figsize[1]))
    gs = GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])

    # Correlation matrix (left, spanning both rows)
    ax_matrix = fig.add_subplot(gs[:, 0])
    im = ax_matrix.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax_matrix.set_xticks(range(n_ant))
    ax_matrix.set_yticks(range(n_ant))
    ax_matrix.set_xticklabels(antenna_ids, fontsize=config.effective_tick_size - 2)
    ax_matrix.set_yticklabels(antenna_ids, fontsize=config.effective_tick_size - 2)
    plt.setp(ax_matrix.get_xticklabels(), rotation=45, ha="right")
    ax_matrix.set_xlabel("Antenna", fontsize=config.effective_label_size)
    ax_matrix.set_ylabel("Antenna", fontsize=config.effective_label_size)
    ax_matrix.set_title("Correlation Matrix", fontsize=config.effective_label_size)
    fig.colorbar(im, ax=ax_matrix, shrink=0.6, label="Correlation")

    # Histogram of correlations (top right)
    ax_hist = fig.add_subplot(gs[0, 1])
    off_diag = corr_matrix[np.triu_indices(n_ant, k=1)]
    off_diag_valid = off_diag[~np.isnan(off_diag)]
    ax_hist.hist(off_diag_valid, bins=20, color="steelblue", edgecolor="black", alpha=0.7)
    ax_hist.axvline(
        stats.mean_correlation,
        color="red",
        linestyle="--",
        label=f"Mean: {stats.mean_correlation:.2f}",
    )
    ax_hist.axvline(
        stats.median_correlation,
        color="orange",
        linestyle="--",
        label=f"Median: {stats.median_correlation:.2f}",
    )
    ax_hist.set_xlabel("Correlation", fontsize=config.effective_label_size)
    ax_hist.set_ylabel("Count", fontsize=config.effective_label_size)
    ax_hist.set_title("Distribution", fontsize=config.effective_label_size)
    ax_hist.legend(fontsize=config.effective_tick_size - 2)

    # Statistics text (bottom right)
    ax_text = fig.add_subplot(gs[1, 1])
    ax_text.axis("off")

    stats_text = f"""Summary Statistics
─────────────────
Mean correlation: {stats.mean_correlation:.3f}
Median correlation: {stats.median_correlation:.3f}
Max correlation: {stats.max_correlation:.3f}
Min correlation: {stats.min_correlation:.3f}

Significant pairs (|r| > 0.5): {stats.n_significant_pairs}
Reference antenna: {stats.reference_antenna}

Clustered groups: {len(stats.clustered_groups)}"""

    if stats.problematic_antennas:
        stats_text += f"\nProblematic antennas: {stats.problematic_antennas}"

    ax_text.text(
        0.05,
        0.95,
        stats_text,
        transform=ax_text.transAxes,
        fontsize=config.effective_tick_size,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(facecolor="lightyellow", alpha=0.8, edgecolor="gray"),
    )

    fig.suptitle(title, fontsize=config.effective_title_size)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved correlation summary: {output}")
        plt.close(fig)

    return fig


def identify_correlated_groups(
    data: AntennaGainData,
    component: str = "amplitude",
    polarization: int = 0,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Identify groups of antennas with correlated behavior.

    Parameters
    ----------
    data :
        AntennaGainData container
    component :
        "amplitude" or "phase"
    polarization :
        Polarization index
    threshold :
        Correlation threshold for grouping

    Returns
    -------
        Dictionary with grouping results

    """
    corr_matrix, stats = compute_gain_correlation_matrix(
        data,
        component=component,
        polarization=polarization,
        significance_threshold=threshold,
    )

    return {
        "groups": stats.clustered_groups,
        "n_groups": len(stats.clustered_groups),
        "reference_antenna": stats.reference_antenna,
        "problematic_antennas": stats.problematic_antennas,
        "significant_pairs": stats.significant_pairs,
        "mean_correlation": stats.mean_correlation,
        "component": component,
        "threshold": threshold,
    }


def generate_correlation_diagnostic_report(
    caltable_path: str | Path,
    output_dir: str | Path,
    title_prefix: str = "Antenna Correlation",
    interactive: bool = False,
) -> dict[str, Any]:
    """Generate comprehensive correlation diagnostic report.

    Creates multiple plots and returns summary statistics.

    Parameters
    ----------
    caltable_path :
        Path to calibration table
    output_dir :
        Output directory for plots
    title_prefix :
        Prefix for plot titles
    interactive :
        Generate Vega-Lite specs for interactive plots
    caltable_path : Union[str, Path]
    output_dir: Union[str :

    Returns
    -------
        Dictionary with paths to generated files and statistics

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract gains
    data = extract_gains_from_caltable(caltable_path)

    results = {
        "source_file": str(caltable_path),
        "n_antennas": data.n_antennas,
        "n_times": data.n_times,
        "plots": {},
        "statistics": {},
    }

    # Amplitude correlation
    amp_corr, amp_stats = compute_gain_correlation_matrix(
        data, component="amplitude", polarization=0
    )
    results["statistics"]["amplitude"] = amp_stats.to_dict()

    amp_matrix_path = output_dir / "correlation_matrix_amplitude.png"
    plot_gain_correlation_matrix(
        amp_corr,
        antenna_ids=data.antenna_ids,
        output=amp_matrix_path,
        title=f"{title_prefix} - Amplitude",
    )
    results["plots"]["amplitude_matrix"] = str(amp_matrix_path)

    # Phase correlation
    phase_corr, phase_stats = compute_gain_correlation_matrix(
        data, component="phase", polarization=0
    )
    results["statistics"]["phase"] = phase_stats.to_dict()

    phase_matrix_path = output_dir / "correlation_matrix_phase.png"
    plot_gain_correlation_matrix(
        phase_corr,
        antenna_ids=data.antenna_ids,
        output=phase_matrix_path,
        title=f"{title_prefix} - Phase",
    )
    results["plots"]["phase_matrix"] = str(phase_matrix_path)

    # Network plot
    network_path = output_dir / "correlation_network.png"
    plot_correlation_network(
        amp_corr,
        antenna_ids=data.antenna_ids,
        output=network_path,
        title=f"{title_prefix} Network",
    )
    results["plots"]["network"] = str(network_path)

    # Summary plot
    summary_path = output_dir / "correlation_summary.png"
    plot_correlation_summary(
        amp_corr,
        amp_stats,
        antenna_ids=data.antenna_ids,
        output=summary_path,
        title=f"{title_prefix} Summary",
    )
    results["plots"]["summary"] = str(summary_path)

    # Interactive specs if requested
    if interactive:
        amp_spec = _create_correlation_matrix_vega_spec(
            amp_corr, antenna_ids=data.antenna_ids, title=f"{title_prefix} - Amplitude"
        )
        results["interactive"] = {"amplitude_matrix": amp_spec}

    logger.info(f"Generated correlation diagnostic report in {output_dir}")
    return results


def _create_correlation_matrix_vega_spec(
    corr_matrix: NDArray[np.floating],
    antenna_ids: list[int] | None = None,
    title: str = "Antenna Correlation Matrix",
) -> dict[str, Any]:
    """Create Vega-Lite spec for interactive correlation matrix.

    Parameters
    ----------
    corr_matrix :
        Correlation matrix
    antenna_ids :
        List of antenna IDs
    title :
        Plot title
    corr_matrix: "NDArray[np.floating]" :

    antenna_ids: Optional[List[int]] :
         (Default value = None)

    Returns
    -------
        Vega-Lite specification dictionary

    """
    n_ant = corr_matrix.shape[0]
    if antenna_ids is None:
        antenna_ids = list(range(n_ant))

    # Convert to tabular format
    records = []
    for i in range(n_ant):
        for j in range(n_ant):
            val = corr_matrix[i, j]
            if not np.isnan(val):
                records.append(
                    {
                        "antenna1": antenna_ids[i],
                        "antenna2": antenna_ids[j],
                        "correlation": float(val),
                    }
                )

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": title,
        "description": "Interactive antenna gain correlation matrix",
        "width": 400,
        "height": 400,
        "data": {"values": records},
        "mark": "rect",
        "encoding": {
            "x": {
                "field": "antenna1",
                "type": "ordinal",
                "title": "Antenna",
            },
            "y": {
                "field": "antenna2",
                "type": "ordinal",
                "title": "Antenna",
            },
            "color": {
                "field": "correlation",
                "type": "quantitative",
                "scale": {"domain": [-1, 1], "scheme": "redblue"},
                "title": "Correlation",
            },
            "tooltip": [
                {"field": "antenna1", "type": "ordinal", "title": "Antenna 1"},
                {"field": "antenna2", "type": "ordinal", "title": "Antenna 2"},
                {
                    "field": "correlation",
                    "type": "quantitative",
                    "title": "Correlation",
                    "format": ".3f",
                },
            ],
        },
    }
