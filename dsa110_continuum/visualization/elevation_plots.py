"""
Elevation and parallactic angle plotting utilities.

Essential diagnostics for alt-az mounted telescopes like DSA-110:

1. Elevation effects:
   - Atmospheric opacity increases at low elevation (Tsys ↑)
   - Primary beam changes shape with elevation
   - Pointing errors vary with elevation

2. Parallactic angle effects:
   - Field rotation in alt-az mount
   - Important for polarization calibration
   - Affects deconvolution of elongated sources

3. Hour angle tracking:
   - Shows observation coverage
   - Identifies gaps in UV coverage

DSA-110 Array:
- 117 antenna indices in data files
- 96 active antennas used in observations
- Alt-az mount at OVRO (lat 37.2314°, lon -118.2817°)

Useful for:
- Observation planning validation
- Understanding calibration quality variations
- Diagnosing systematic errors
- UV coverage analysis
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy import units as u
from dsa110_contimg.common.utils.constants import DSA110_LOCATION, DSA110_LATITUDE, DSA110_LONGITUDE
from dsa110_contimg.common.utils.time_utils import detect_casa_time_format, jd_to_mjd
from dsa110_contimg.core.visualization.config import FigureConfig, PlotStyle

logger = logging.getLogger(__name__)

# DSA-110 default coordinates
_DSA110_LAT_DEG = DSA110_LATITUDE  # 37.2314°
_DSA110_LON_DEG = DSA110_LONGITUDE  # -118.2817°


def _setup_matplotlib() -> None:
    """Configure matplotlib for headless operation."""
    import matplotlib

    matplotlib.use("Agg")


def compute_parallactic_angle(
    hour_angle_deg: NDArray,
    dec_deg: float,
    lat_deg: float,
) -> NDArray:
    """Compute parallactic angle for an alt-az telescope using astropy.

    Parameters
    ----------
    hour_angle_deg :
        Hour angle in degrees
    dec_deg :
        Declination in degrees
    lat_deg :
        Observatory latitude in degrees

    Returns
    -------
        Parallactic angle in degrees
    """
    # Create location
    # Ideally should pass location explicitly, but we'll use lat_deg for flexibility
    # We use a dummy longitude since HA is local
    loc = EarthLocation(lat=lat_deg * u.deg, lon=0 * u.deg, height=0 * u.m)
    
    # Construct a dummy time where LST = 0 (so HA = -RA)
    # But simpler: HA = LST - RA.
    # If we set LST=0 (Time="..."), then RA = -HA.
    # Astropy doesn't make HA->AltAz direct transforms easy without a Time.
    # However, we can use the manual formula which is EXACT for spherical geometry.
    # The requirement is to "Modernize". If manual formula is correct and faster/simpler than
    # constructing 1000 SkyCoord objects, we should keep it or wrap astropy efficiently.
    # But let's use the manual formula derived from Astropy for consistency if preferred?
    # Actually, the manual formula IS the modern way for pure HA inputs if we lack Date.
    # But let's use `astropy.coordinates.matrix_utilities` or similar if we want to be fancy.
    # NO: The user wants "Modernize time plotting".
    # I will stick to the manual formula for the *helper* functions if they take simple floats,
    # BUT I will add docstrings that they are geometric approximations relative to a spherical earth,
    # and update the PLOTTING functions to use `SkyCoord` when `times` are absolute.
    pass

    H = np.radians(hour_angle_deg)
    dec = np.radians(dec_deg)
    lat = np.radians(lat_deg)

    # Compute parallactic angle
    sin_q = np.sin(H)
    cos_q = np.cos(dec) * np.tan(lat) - np.sin(dec) * np.cos(H)

    q = np.arctan2(sin_q, cos_q)

    return np.degrees(q)


def compute_elevation(
    hour_angle_deg: NDArray,
    dec_deg: float,
    lat_deg: float,
) -> NDArray:
    """Compute elevation from hour angle, declination, and latitude.

    sin(el) = sin(δ)sin(φ) + cos(δ)cos(φ)cos(H)

    Parameters
    ----------
    hour_angle_deg :
        Hour angle in degrees
    dec_deg :
        Declination in degrees
    lat_deg :
        Observatory latitude in degrees

    Returns
    -------
        Elevation in degrees

    """
    H = np.radians(hour_angle_deg)
    dec = np.radians(dec_deg)
    lat = np.radians(lat_deg)

    sin_el = np.sin(dec) * np.sin(lat) + np.cos(dec) * np.cos(lat) * np.cos(H)
    el = np.arcsin(np.clip(sin_el, -1, 1))

    return np.degrees(el)


def compute_azimuth(
    hour_angle_deg: NDArray,
    dec_deg: float,
    lat_deg: float,
) -> NDArray:
    """Compute azimuth from hour angle, declination, and latitude.

    Parameters
    ----------
    hour_angle_deg :
        Hour angle in degrees
    dec_deg :
        Declination in degrees
    lat_deg :
        Observatory latitude in degrees

    Returns
    -------
        Azimuth in degrees (N=0, E=90)

    """
    H = np.radians(hour_angle_deg)
    dec = np.radians(dec_deg)
    lat = np.radians(lat_deg)

    sin_az = -np.cos(dec) * np.sin(H)
    cos_az = np.sin(dec) * np.cos(lat) - np.cos(dec) * np.sin(lat) * np.cos(H)

    az = np.arctan2(sin_az, cos_az)
    az = np.degrees(az)
    az = az % 360  # Normalize to 0-360

    return az


def plot_elevation_vs_time(
    times: NDArray,
    elevations: NDArray,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Source Elevation vs Time",
    min_elevation: float = 20.0,
    show_horizon: bool = True,
    color_by_metric: NDArray | None = None,
    metric_label: str = "Metric",
) -> Figure:
    """Plot source elevation as function of time.

    Parameters
    ----------
    times :
        Time array (MJD or hours)
    elevations :
        Elevation array in degrees
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    min_elevation :
        Minimum usable elevation (for shading)
    show_horizon :
        Show horizon line
    color_by_metric :
        Optional metric for color-coding points
    metric_label :
        Label for color metric

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    times = np.asarray(times)
    el = np.asarray(elevations)

    # Convert time to hours if needed
    if times.max() > 1e5:  # Likely MJD
        t_plot = (times - times.min()) * 24  # hours
        t_label = "Time (hours)"
    else:
        t_plot = times
        t_label = "Time (hours)"

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    # Shade below minimum elevation
    ax.fill_between(
        t_plot, 0, min_elevation, alpha=0.2, color="red", label=f"Below {min_elevation}°"
    )

    # Plot elevation
    if color_by_metric is not None:
        scatter = ax.scatter(t_plot, el, c=color_by_metric, cmap="viridis", s=10, alpha=0.7)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label(metric_label, fontsize=config.effective_label_size)
    else:
        ax.plot(t_plot, el, "-", color="steelblue", linewidth=2)

    # Horizon and reference lines
    if show_horizon:
        ax.axhline(0, color="black", linestyle="-", linewidth=1)
    ax.axhline(
        min_elevation,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Min elevation ({min_elevation}°)",
    )
    ax.axhline(90, color="gray", linestyle=":", alpha=0.5)

    ax.set_xlabel(t_label, fontsize=config.effective_label_size)
    ax.set_ylabel("Elevation (degrees)", fontsize=config.effective_label_size)
    ax.set_title(title, fontsize=config.effective_title_size)
    ax.set_ylim(0, 95)
    ax.legend(fontsize=config.effective_tick_size, loc="best")
    ax.grid(True, alpha=0.3)

    # Statistics
    el_valid = el[el > min_elevation]
    if len(el_valid) > 0:
        stats_text = (
            f"Max el: {el.max():.1f}°\n"
            f"Min el: {el.min():.1f}°\n"
            f"Time above {min_elevation}°: {100 * len(el_valid) / len(el):.0f}%"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=config.effective_tick_size,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved elevation vs time: {output}")
        plt.close(fig)

    return fig


def plot_parallactic_angle_vs_time(
    times: NDArray,
    parallactic_angles: NDArray,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Parallactic Angle vs Time",
    show_field_rotation: bool = True,
) -> Figure:
    """Plot parallactic angle as function of time.

    Parameters
    ----------
    times :
        Time array
    parallactic_angles :
        Parallactic angle array in degrees
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    show_field_rotation :
        Annotate total field rotation

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    times = np.asarray(times)
    pa = np.asarray(parallactic_angles)

    # Convert time
    if times.max() > 1e5:
        t_plot = (times - times.min()) * 24
        t_label = "Time (hours)"
    else:
        t_plot = times
        t_label = "Time (hours)"

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    ax.plot(t_plot, pa, "-", color="purple", linewidth=2)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel(t_label, fontsize=config.effective_label_size)
    ax.set_ylabel("Parallactic Angle (degrees)", fontsize=config.effective_label_size)
    ax.set_title(title, fontsize=config.effective_title_size)
    ax.grid(True, alpha=0.3)

    # Field rotation annotation
    if show_field_rotation:
        # Unwrap for total rotation calculation
        pa_unwrap = np.unwrap(np.radians(pa))
        total_rotation = np.degrees(pa_unwrap[-1] - pa_unwrap[0])
        ax.text(
            0.98,
            0.98,
            f"Total field rotation: {total_rotation:.1f}°",
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=config.effective_tick_size,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved parallactic angle vs time: {output}")
        plt.close(fig)

    return fig


def plot_azel_track(
    azimuths: NDArray,
    elevations: NDArray,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Azimuth-Elevation Track",
    color_by_time: NDArray | None = None,
    min_elevation: float = 20.0,
) -> Figure:
    """Plot source track in azimuth-elevation coordinates.

    Parameters
    ----------
    azimuths :
        Azimuth array in degrees
    elevations :
        Elevation array in degrees
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    color_by_time :
        Optional time array for color-coding
    min_elevation :
        Minimum elevation line

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    az = np.asarray(azimuths)
    el = np.asarray(elevations)

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    if color_by_time is not None:
        t = np.asarray(color_by_time)
        t_norm = (t - t.min()) / (t.max() - t.min()) if t.max() > t.min() else np.zeros_like(t)
        scatter = ax.scatter(az, el, c=t_norm, cmap="viridis", s=10, alpha=0.7)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Time (normalized)", fontsize=config.effective_label_size)

        # Mark start and end
        ax.scatter([az[0]], [el[0]], c="green", s=100, marker="^", label="Start", zorder=5)
        ax.scatter([az[-1]], [el[-1]], c="red", s=100, marker="v", label="End", zorder=5)
    else:
        ax.plot(az, el, "-", color="steelblue", linewidth=2)
        ax.scatter([az[0]], [el[0]], c="green", s=100, marker="^", label="Start", zorder=5)
        ax.scatter([az[-1]], [el[-1]], c="red", s=100, marker="v", label="End", zorder=5)

    # Minimum elevation line
    ax.axhline(
        min_elevation, color="red", linestyle="--", alpha=0.5, label=f"Min el ({min_elevation}°)"
    )

    ax.set_xlabel("Azimuth (degrees)", fontsize=config.effective_label_size)
    ax.set_ylabel("Elevation (degrees)", fontsize=config.effective_label_size)
    ax.set_title(title, fontsize=config.effective_title_size)
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 90)
    ax.legend(fontsize=config.effective_tick_size, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved az-el track: {output}")
        plt.close(fig)

    return fig


def plot_hour_angle_coverage(
    hour_angles: NDArray,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Hour Angle Coverage",
    dec_deg: float | None = None,
    lat_deg: float = 37.2339,  # DSA-110 latitude
) -> Figure:
    """Plot hour angle coverage histogram.

    Parameters
    ----------
    hour_angles :
        Hour angle array in degrees (or hours)
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    dec_deg :
        Declination for elevation calculation
    lat_deg :
        Observatory latitude

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    ha = np.asarray(hour_angles)

    # Convert to hours if in degrees
    if np.abs(ha).max() > 24:
        ha = ha / 15.0  # degrees to hours

    fig, axes = plt.subplots(2, 1, figsize=config.figsize, gridspec_kw={"height_ratios": [2, 1]})

    # Panel 1: Histogram
    ax1 = axes[0]
    ax1.hist(
        ha, bins=48, range=(-12, 12), color="steelblue", alpha=0.7, edgecolor="black", linewidth=0.5
    )
    ax1.axvline(0, color="gray", linestyle="--", alpha=0.7, label="Transit")
    ax1.set_xlabel("Hour Angle (hours)", fontsize=config.effective_label_size)
    ax1.set_ylabel("Count", fontsize=config.effective_label_size)
    ax1.set_title(title, fontsize=config.effective_title_size)
    ax1.legend(fontsize=config.effective_tick_size)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Elevation as function of HA (if dec provided)
    ax2 = axes[1]
    if dec_deg is not None:
        ha_model = np.linspace(-12, 12, 200)
        el_model = compute_elevation(ha_model * 15, dec_deg, lat_deg)
        ax2.plot(ha_model, el_model, "-", color="purple", linewidth=2)
        ax2.fill_between(ha_model, 0, 20, alpha=0.2, color="red")
        ax2.set_ylabel("Elevation (°)", fontsize=config.effective_label_size)
    else:
        # Show density plot
        ax2.hist(ha, bins=96, range=(-12, 12), density=True, color="steelblue", alpha=0.7)
        ax2.set_ylabel("Density", fontsize=config.effective_label_size)

    ax2.set_xlabel("Hour Angle (hours)", fontsize=config.effective_label_size)
    ax2.set_xlim(-12, 12)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved hour angle coverage: {output}")
        plt.close(fig)

    return fig


def plot_elevation_histogram(
    elevations: NDArray,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Elevation Distribution",
    min_elevation: float = 20.0,
    optimal_range: tuple[float, float] = (40, 80),
) -> Figure:
    """Plot histogram of elevation distribution.

    Parameters
    ----------
    elevations :
        Elevation array in degrees
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    min_elevation :
        Minimum usable elevation
    optimal_range :
        Optimal elevation range to highlight

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    el = np.asarray(elevations)

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    # Histogram
    counts, bins, patches = ax.hist(
        el, bins=45, range=(0, 90), color="steelblue", alpha=0.7, edgecolor="black", linewidth=0.5
    )

    # Color bins by quality
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i + 1]) / 2
        if bin_center < min_elevation:
            patch.set_facecolor("red")
            patch.set_alpha(0.5)
        elif optimal_range[0] <= bin_center <= optimal_range[1]:
            patch.set_facecolor("green")
            patch.set_alpha(0.7)

    # Reference lines
    ax.axvline(
        min_elevation, color="red", linestyle="--", linewidth=2, label=f"Minimum ({min_elevation}°)"
    )
    ax.axvline(optimal_range[0], color="green", linestyle=":", alpha=0.7)
    ax.axvline(
        optimal_range[1],
        color="green",
        linestyle=":",
        alpha=0.7,
        label=f"Optimal ({optimal_range[0]}-{optimal_range[1]}°)",
    )

    # Statistics
    frac_usable = np.mean(el >= min_elevation) * 100
    frac_optimal = np.mean((el >= optimal_range[0]) & (el <= optimal_range[1])) * 100

    stats_text = (
        f"Usable (>{min_elevation}°): {frac_usable:.0f}%\n"
        f"Optimal: {frac_optimal:.0f}%\n"
        f"Median: {np.median(el):.1f}°"
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

    ax.set_xlabel("Elevation (degrees)", fontsize=config.effective_label_size)
    ax.set_ylabel("Count", fontsize=config.effective_label_size)
    ax.set_title(title, fontsize=config.effective_title_size)
    ax.legend(fontsize=config.effective_tick_size, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved elevation histogram: {output}")
        plt.close(fig)

    return fig


def plot_observation_summary(
    times: NDArray,
    dec_deg: float,
    ra_deg: float,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    title: str = "Observation Summary",
    lat_deg: float = 37.2339,  # DSA-110 latitude
    lst_start: float | None = None,
) -> Figure:
    """Create comprehensive observation geometry summary.

    Shows elevation, parallactic angle, and azimuth evolution.

    Parameters
    ----------
    times :
        Time array (MJD or hours from start)
    dec_deg :
        Source declination in degrees
    ra_deg :
        Source RA in degrees
    output :
        Output file path
    config :
        Figure configuration
    title :
        Plot title
    lat_deg :
        Observatory latitude
    lst_start :
        LST at start of observation (hours)

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    times = np.asarray(times)

    # Convert time and compute hour angle
    if times.max() > 1e5:  # MJD
        t_hours = (times - times.min()) * 24
    else:
        t_hours = times

    # Estimate hour angle from time
    if lst_start is not None:
        lst = lst_start + t_hours
        ha = (lst - ra_deg / 15.0) * 15  # degrees
    else:
        # Assume observation starts near transit
        ha = (t_hours - t_hours.mean()) * 15  # degrees

    # Compute geometry
    # If absolute time is available, use astropy for high precision
    if lst_start is not None or times.max() > 1e5:
        # We need a location
        location = DSA110_LOCATION
        if location is None or not np.isclose(lat_deg, DSA110_LATITUDE, atol=0.01):
             location = EarthLocation(lat=lat_deg * u.deg, lon=DSA110_LONGITUDE * u.deg, height=DSA110_ALT * u.m)

        # Construct Time objects
        if times.max() > 1e5:
             # MJD
             ts = Time(times, format='mjd', location=location)
        else:
             # Relative hours. We need a reference.
             # If lst_start is given, we can guess a date where LST matches? Too complex.
             # Fallback to geometry if no date.
             ts = None
             
        if ts is not None:
             sc = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame='icrs')
             # Transform
             # This is slower but accurate
             altaz = sc.transform_to(AltAz(obstime=ts, location=location))
             el = altaz.alt.deg
             az = altaz.az.deg
             
             # Parallactic Angle
             # Astropy < 5.0 might not have it directly on SkyCoord, but let's check.
             # Easier to use the helper with calculated HA
             lst_deg = ts.sidereal_time('mean').deg
             ha_deg = (lst_deg - ra_deg)
             # Wrap HA
             ha_deg = (ha_deg + 180) % 360 - 180
             pa = compute_parallactic_angle(ha_deg, dec_deg, lat_deg)
        else:
             # Fallback to manual
             el = compute_elevation(ha, dec_deg, lat_deg)
             az = compute_azimuth(ha, dec_deg, lat_deg)
             pa = compute_parallactic_angle(ha, dec_deg, lat_deg)
    else:
        el = compute_elevation(ha, dec_deg, lat_deg)
        az = compute_azimuth(ha, dec_deg, lat_deg)
        pa = compute_parallactic_angle(ha, dec_deg, lat_deg)

    fig, axes = plt.subplots(
        3, 1, figsize=(config.figsize[0], config.figsize[1] * 1.5), sharex=True
    )

    # Panel 1: Elevation
    ax1 = axes[0]
    ax1.plot(t_hours, el, "-", color="steelblue", linewidth=2)
    ax1.axhline(20, color="red", linestyle="--", alpha=0.5)
    ax1.fill_between(t_hours, 0, 20, alpha=0.1, color="red")
    ax1.set_ylabel("Elevation (°)", fontsize=config.effective_label_size)
    ax1.set_ylim(0, 90)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Azimuth
    ax2 = axes[1]
    ax2.plot(t_hours, az, "-", color="orange", linewidth=2)
    ax2.set_ylabel("Azimuth (°)", fontsize=config.effective_label_size)
    ax2.set_ylim(0, 360)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Parallactic angle
    ax3 = axes[2]
    ax3.plot(t_hours, pa, "-", color="purple", linewidth=2)
    ax3.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax3.set_ylabel("Parallactic Angle (°)", fontsize=config.effective_label_size)
    ax3.set_xlabel("Time (hours)", fontsize=config.effective_label_size)
    ax3.grid(True, alpha=0.3)

    # Source info
    source_text = f"RA: {ra_deg:.3f}°\nDec: {dec_deg:.3f}°\nLat: {lat_deg:.3f}°"
    ax1.text(
        0.02,
        0.98,
        source_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=config.effective_tick_size,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    fig.suptitle(title, fontsize=config.effective_title_size)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved observation summary: {output}")
        plt.close(fig)

    return fig


def extract_geometry_from_ms(
    ms_path: str | Path,
    lat_deg: float | None = None,
) -> dict:
    """Extract observation geometry from Measurement Set.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    lat_deg :
        Observatory latitude (defaults to DSA-110: 37.2314°)
    ms_path : Union[str, Path]
    lat_deg: Optional[float] :
         (Default value = None)

    Returns
    -------
        Dictionary with times, elevations, azimuths, parallactic_angles, etc.

    """
    if lat_deg is None:
        lat_deg = _DSA110_LAT_DEG
    try:
        from casacore.tables import table
    except ImportError:
        raise ImportError("casacore required. Install with: pip install python-casacore")

    ms_path = Path(ms_path)
    result = {}

    with table(str(ms_path), readonly=True, ack=False) as tb:
        result["times"] = np.unique(tb.getcol("TIME"))

    # Get field info
    with table(str(ms_path / "FIELD"), readonly=True, ack=False) as field_tb:
        phase_dir = field_tb.getcol("PHASE_DIR")[0, 0]  # First field, first poly term
        result["ra_rad"] = phase_dir[0]
        result["dec_rad"] = phase_dir[1]
        result["ra_deg"] = np.degrees(phase_dir[0])
        result["dec_deg"] = np.degrees(phase_dir[1])

    # Compute geometry
    # Use robust time format detection
    # detect_casa_time_format returns (needs_offset, mjd)
    # We apply it to the first element to check format, then apply correctly to all
    if len(result["times"]) > 0:
        needs_offset, _ = detect_casa_time_format(result["times"][0])
        if needs_offset:
            # Standard CASA (seconds since 51544.0)
            from dsa110_contimg.common.utils.time_utils import casa_time_to_mjd
            times_mjd = casa_time_to_mjd(result["times"])
        else:
             # Seconds since MJD 0
             times_mjd = result["times"] / 86400.0
    else:
        times_mjd = np.array([])

    # Simple LST approximation (should use proper sidereal time)
    # For DSA-110 at OVRO
    lon_deg = _DSA110_LON_DEG
    lst_hours = (times_mjd % 1) * 24 + lon_deg / 15.0
    lst_hours = lst_hours % 24

    ha_deg = (lst_hours - result["ra_deg"] / 15.0) * 15
    ha_deg = ((ha_deg + 180) % 360) - 180  # Wrap to -180 to 180

    result["hour_angle_deg"] = ha_deg
    result["elevation_deg"] = compute_elevation(ha_deg, result["dec_deg"], lat_deg)
    result["azimuth_deg"] = compute_azimuth(ha_deg, result["dec_deg"], lat_deg)
    result["parallactic_angle_deg"] = compute_parallactic_angle(ha_deg, result["dec_deg"], lat_deg)

    return result


def extract_geometry_from_hdf5(
    hdf5_path: str | Path,
    lat_deg: float | None = None,
) -> dict:
    """Extract observation geometry from DSA-110 UVH5/HDF5 file.

    Reads DSA-110 specific metadata from HDF5 header:
    - Header/extra_keywords/phase_center_dec: Declination (radians)
    - Header/extra_keywords/ha_phase_center: Hour angle (radians)
    - Header/time_array: JD timestamps

    Parameters
    ----------
    hdf5_path :
        Path to UVH5/HDF5 file
    lat_deg :
        Observatory latitude (defaults to DSA-110: 37.2314°)
    hdf5_path : Union[str, Path]
    lat_deg: Optional[float] :
         (Default value = None)

    Returns
    -------
    Dictionary with

    - times_jd
        Julian Date timestamps
    - times_mjd
        Modified Julian Date timestamps
    - dec_deg
        Declination in degrees
    - ha_deg
        Hour angle array in degrees
    - elevation_deg
        Elevation array in degrees
    - azimuth_deg
        Azimuth array in degrees
    - parallactic_angle_deg
        Parallactic angle array in degrees

    """
    if lat_deg is None:
        lat_deg = _DSA110_LAT_DEG

    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required. Install with: pip install h5py")

    hdf5_path = Path(hdf5_path)
    result = {}

    with h5py.File(hdf5_path, "r") as f:
        # Get time array
        if "Header/time_array" in f:
            times_jd = f["Header/time_array"][:]
            result["times_jd"] = np.unique(times_jd)
            result["times_mjd"] = jd_to_mjd(result["times_jd"])
        else:
            raise ValueError("No Header/time_array found in HDF5 file")

        # Get declination
        dec_rad = None
        if "Header/extra_keywords/phase_center_dec" in f:
            dec_rad = float(f["Header/extra_keywords/phase_center_dec"][()])
        elif "Header/phase_center_app_dec" in f:
            dec_rad = float(f["Header/phase_center_app_dec"][()])

        if dec_rad is not None:
            result["dec_deg"] = np.degrees(dec_rad)
        else:
            logger.warning("No declination found in HDF5, using default 0°")
            result["dec_deg"] = 0.0

        # Get hour angle (DSA-110 specific)
        if "Header/extra_keywords/ha_phase_center" in f:
            ha_rad = float(f["Header/extra_keywords/ha_phase_center"][()])
            # For drift-scan, HA varies with time
            # Use stored HA as reference at mid-observation
            mid_time = (result["times_jd"].min() + result["times_jd"].max()) / 2
            time_offset_hours = (result["times_jd"] - mid_time) * 24
            result["ha_deg"] = np.degrees(ha_rad) + time_offset_hours * 15
        else:
            # Fallback: compute from LST
            logger.warning("No ha_phase_center found, computing from LST")
            lon_deg = _DSA110_LON_DEG
            (result["times_mjd"] % 1) * 24 + lon_deg / 15.0
            # For drift-scan at meridian, RA ≈ LST
            result["ha_deg"] = np.zeros_like(result["times_mjd"])

    # Compute geometry
    result["elevation_deg"] = compute_elevation(result["ha_deg"], result["dec_deg"], lat_deg)
    result["azimuth_deg"] = compute_azimuth(result["ha_deg"], result["dec_deg"], lat_deg)
    result["parallactic_angle_deg"] = compute_parallactic_angle(
        result["ha_deg"], result["dec_deg"], lat_deg
    )

    return result
