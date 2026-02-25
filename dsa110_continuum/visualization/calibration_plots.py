"""
Calibration diagnostic plotting utilities.

Provides functions for:
- Bandpass solutions (amplitude, phase vs frequency)
- Gain solutions (amplitude, phase vs time)
- Delay solutions
- Dynamic spectra / waterfalls

Adapted from:
- dsa110-calib/dsacalib/plotting.py (Dana Simard)
- dsa110_contimg/calibration/plotting.py (existing CASA plotbandpass wrapper)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from dsa110_contimg.core.visualization.config import FigureConfig, PlotStyle
from dsa110_contimg.core.visualization.plot_context import PlotContext, should_generate_interactive
from dsa110_contimg.core.visualization.vega_specs import save_vega_spec

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _setup_matplotlib() -> None:
    """Configure matplotlib for headless operation."""
    import matplotlib

    matplotlib.use("Agg")


def plot_bandpass(
    caltable: str | Path,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    plot_amplitude: bool = True,
    plot_phase: bool = True,
    antenna: str | None = None,
    spw: int | None = None,
    smooth_overlay: bool = True,
    smooth_poly_order: int = 3,
) -> list[Path]:
    """Plot bandpass calibration solutions.

    Generates per-SPW plots showing amplitude and phase vs frequency.

    Parameters
    ----------
    caltable :
        Path to CASA bandpass calibration table
    output :
        Output directory or file prefix
    config :
        Figure configuration
    plot_amplitude :
        Generate amplitude plot
    plot_phase :
        Generate phase plot
    antenna :
        Specific antenna to plot (default: all)
    spw :
        Specific spectral window (default: all)
    smooth_overlay :
        Overlay low-order polynomial fit for spectral smoothness
    smooth_poly_order :
        Polynomial order for smoothness fit
    caltable : Union[str, Path]
    output : Optional[Union[str, Path]]
         (Default value = None)
    config: Optional[FigureConfig] :
         (Default value = None)

    Returns
    -------
        List of generated plot file paths

    """
    _setup_matplotlib()

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    caltable = Path(caltable)
    if output is None:
        output = caltable.parent / f"{caltable.name}_plots"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    generated = []

    # Helper to call plotbandpass with log environment protection
    def _call_plotbandpass(**kwargs):
        try:
            from dsa110_contimg.common.utils.casa_init import casa_log_environment

            with casa_log_environment():
                from casatasks import plotbandpass

                return plotbandpass(**kwargs)
        except ImportError:
            from casatasks import plotbandpass

            return plotbandpass(**kwargs)

    try:
        if plot_amplitude:
            amp_plot = str(output / f"{caltable.name}_amp")
            _call_plotbandpass(
                caltable=str(caltable),
                xaxis="freq",
                yaxis="amp",
                figfile=amp_plot,
                interactive=False,
                showflagged=False,
                overlay="antenna" if antenna is None else "",
                antenna=antenna or "",
                spw=str(spw) if spw is not None else "",
            )
            # Find generated files
            for f in output.glob(f"{caltable.name}_amp*.png"):
                generated.append(f)

        if plot_phase:
            phase_plot = str(output / f"{caltable.name}_phase")
            _call_plotbandpass(
                caltable=str(caltable),
                xaxis="freq",
                yaxis="phase",
                figfile=phase_plot,
                interactive=False,
                showflagged=False,
                overlay="antenna" if antenna is None else "",
                antenna=antenna or "",
                spw=str(spw) if spw is not None else "",
                plotrange=[0, 0, -180, 180],
            )
            for f in output.glob(f"{caltable.name}_phase*.png"):
                generated.append(f)

    except ImportError:
        logger.warning("CASA not available, using fallback plotting")
        generated = _plot_bandpass_fallback(
            caltable,
            output,
            config,
            plot_amplitude,
            plot_phase,
            smooth_overlay=smooth_overlay,
            smooth_poly_order=smooth_poly_order,
        )
    except Exception as e:
        logger.error(f"plotbandpass failed: {e}, using fallback")
        generated = _plot_bandpass_fallback(
            caltable,
            output,
            config,
            plot_amplitude,
            plot_phase,
            smooth_overlay=smooth_overlay,
            smooth_poly_order=smooth_poly_order,
        )

    logger.info(f"Generated {len(generated)} bandpass plots in {output}")
    return generated


def _plot_bandpass_fallback(
    caltable: Path,
    output: Path,
    config: FigureConfig,
    plot_amplitude: bool,
    plot_phase: bool,
    smooth_overlay: bool = True,
    smooth_poly_order: int = 3,
) -> list[Path]:
    """Fallback bandpass plotting using casacore directly.

    Parameters
    ----------
    """
    import matplotlib.pyplot as plt

    generated = []

    try:
        from casacore.tables import table

        with table(str(caltable), readonly=True) as tb:
            cparam = tb.getcol("CPARAM")  # Complex gains
            freq = tb.getcol("CHAN_FREQ") if "CHAN_FREQ" in tb.colnames() else None
            antenna1 = tb.getcol("ANTENNA1")

        nant = len(np.unique(antenna1))
        nchan = cparam.shape[1]
        _npol = cparam.shape[2]  # Stored for potential future use

        # Frequency axis
        if freq is None:
            freq = np.arange(nchan)
            freq_label = "Channel"
        else:
            freq = freq[0] / 1e9  # Convert to GHz
            freq_label = "Frequency (GHz)"

        smoothness_rms: list[float] = []
        # Plot amplitude
        if plot_amplitude:
            fig, axes = plt.subplots(
                2, 1, figsize=(config.figsize[0], config.figsize[1] * 2), sharex=True
            )

            for pol_idx, pol_label in enumerate(["Pol A", "Pol B"]):
                ax = axes[pol_idx]
                for ant in range(min(nant, 10)):  # Limit to first 10 antennas
                    amp = np.abs(cparam[ant, :, pol_idx])
                    ax.plot(freq, amp, alpha=0.7, label=f"Ant {ant}")
                    if smooth_overlay and len(freq) > smooth_poly_order:
                        try:
                            coeffs = np.polyfit(freq, amp, smooth_poly_order)
                            fit = np.polyval(coeffs, freq)
                            ax.plot(freq, fit, linestyle="--", alpha=0.5)
                            residual = amp - fit
                            smoothness_rms.append(float(np.sqrt(np.nanmean(residual**2))))
                        except Exception as fit_err:
                            logger.debug(
                                "Smoothness fit failed for ant %s pol %s: %s",
                                ant,
                                pol_label,
                                fit_err,
                            )
                ax.set_ylabel("Amplitude")
                ax.set_title(pol_label)
                ax.legend(ncol=5, fontsize=8)

            axes[-1].set_xlabel(freq_label)
            fig.suptitle(f"Bandpass: {caltable.name}")
            fig.tight_layout()

            amp_path = output / f"{caltable.name}_amp_fallback.png"
            fig.savefig(amp_path, dpi=config.dpi, bbox_inches="tight")
            plt.close(fig)
            generated.append(amp_path)
            if smoothness_rms:
                logger.info(
                    "Bandpass smoothness RMS (median over plotted antennas/pols): %.4f",
                    float(np.median(smoothness_rms)),
                )

        # Plot phase
        if plot_phase:
            fig, axes = plt.subplots(
                2, 1, figsize=(config.figsize[0], config.figsize[1] * 2), sharex=True
            )

            for pol_idx, pol_label in enumerate(["Pol A", "Pol B"]):
                ax = axes[pol_idx]
                for ant in range(min(nant, 10)):
                    phase = np.angle(cparam[ant, :, pol_idx], deg=True)
                    ax.plot(freq, phase, alpha=0.7, label=f"Ant {ant}")
                ax.set_ylabel("Phase (deg)")
                ax.set_ylim(-180, 180)
                ax.set_title(pol_label)
                ax.legend(ncol=5, fontsize=8)

            axes[-1].set_xlabel(freq_label)
            fig.suptitle(f"Bandpass Phase: {caltable.name}")
            fig.tight_layout()

            phase_path = output / f"{caltable.name}_phase_fallback.png"
            fig.savefig(phase_path, dpi=config.dpi, bbox_inches="tight")
            plt.close(fig)
            generated.append(phase_path)

    except Exception as e:
        logger.error(f"Fallback bandpass plotting failed: {e}")

    return generated


def plot_gains(
    caltable: str | Path,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    plot_amplitude: bool = True,
    plot_phase: bool = True,
) -> list[Path]:
    """Plot gain calibration solutions vs time.

    Parameters
    ----------
    caltable :
        Path to CASA gain calibration table
    output :
        Output directory or file prefix
    config :
        Figure configuration
    plot_amplitude :
        Generate amplitude plot
    plot_phase :
        Generate phase plot
    caltable : Union[str, Path]
    output : Optional[Union[str, Path]]
         (Default value = None)
    config: Optional[FigureConfig] :
         (Default value = None)

    Returns
    -------
        List of generated plot file paths

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    caltable = Path(caltable)
    if output is None:
        output = caltable.parent / f"{caltable.name}_plots"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    generated = []

    try:
        from casacore.tables import table

        with table(str(caltable), readonly=True) as tb:
            cparam = tb.getcol("CPARAM")
            time = tb.getcol("TIME")
            antenna1 = tb.getcol("ANTENNA1")

        # Convert time to minutes from start
        time_min = (time - time.min()) / 60.0

        nant = len(np.unique(antenna1))
        _npol = cparam.shape[-1]  # Stored for potential future use

        # Plot amplitude
        if plot_amplitude:
            fig, axes = plt.subplots(
                2, 1, figsize=(config.figsize[0], config.figsize[1] * 2), sharex=True
            )

            for pol_idx, pol_label in enumerate(["Pol A", "Pol B"]):
                ax = axes[pol_idx]
                for ant in range(min(nant, 10)):
                    mask = antenna1 == ant
                    amp = np.abs(cparam[mask, 0, pol_idx])
                    t = time_min[mask]
                    ax.plot(t, amp, ".", alpha=0.7, label=f"Ant {ant}")
                ax.set_ylabel("Amplitude")
                ax.set_title(pol_label)
                ax.legend(ncol=5, fontsize=8)

            axes[-1].set_xlabel("Time (minutes)")
            fig.suptitle(f"Gains: {caltable.name}")
            fig.tight_layout()

            amp_path = output / f"{caltable.name}_gain_amp.png"
            fig.savefig(amp_path, dpi=config.dpi, bbox_inches="tight")
            plt.close(fig)
            generated.append(amp_path)

        # Plot phase
        if plot_phase:
            fig, axes = plt.subplots(
                2, 1, figsize=(config.figsize[0], config.figsize[1] * 2), sharex=True
            )

            for pol_idx, pol_label in enumerate(["Pol A", "Pol B"]):
                ax = axes[pol_idx]
                for ant in range(min(nant, 10)):
                    mask = antenna1 == ant
                    phase = np.angle(cparam[mask, 0, pol_idx], deg=True)
                    t = time_min[mask]
                    ax.plot(t, phase, ".", alpha=0.7, label=f"Ant {ant}")
                ax.set_ylabel("Phase (deg)")
                ax.set_ylim(-180, 180)
                ax.set_title(pol_label)
                ax.legend(ncol=5, fontsize=8)

            axes[-1].set_xlabel("Time (minutes)")
            fig.suptitle(f"Gain Phase: {caltable.name}")
            fig.tight_layout()

            phase_path = output / f"{caltable.name}_gain_phase.png"
            fig.savefig(phase_path, dpi=config.dpi, bbox_inches="tight")
            plt.close(fig)
            generated.append(phase_path)

    except Exception as e:
        logger.error(f"Gain plotting failed: {e}")

    logger.info(f"Generated {len(generated)} gain plots")
    return generated


def _find_ms_for_caltable(caltable: Path, ms_path: str | Path | None) -> Path | None:
    if ms_path is not None:
        candidate = Path(ms_path)
        return candidate if candidate.exists() else None
    ms_files = sorted(caltable.parent.glob("*.ms"))
    return ms_files[0] if ms_files else None


def plot_kcal_delays(
    caltable: str | Path,
    ms_path: str | Path | None = None,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    default_ref_frequency_hz: float = 1400e6,
) -> list[Path]:
    """Plot K-calibration (delay) solutions from a CASA calibration table.

    This reads complex per-antenna solutions from the caltable's CPARAM column
    and estimates delays from the phase at a reference frequency.

    Parameters
    ----------
    caltable :
        Path to CASA K calibration table (.kcal or .K)
    ms_path :
        Optional MS path to read REF_FREQUENCY from SPECTRAL_WINDOW
    output :
        Output directory (defaults to <caltable>_plots)
    config :
        Figure configuration
    default_ref_frequency_hz :
        Fallback ref frequency when MS is unavailable
    caltable : Union[str, Path]
    ms_path : Optional[Union[str, Path]]
         (Default value = None)
    output : Optional[Union[str, config: Optional[FigureConfig]
         (Default value = None)

    Returns
    -------
        List of generated plot file paths

    """
    _setup_matplotlib()
    from dsa110_contimg.core.visualization.kcal_delay_plots import (
        plot_kcal_delays as _plot_kcal_delays,
    )

    return _plot_kcal_delays(
        caltable=caltable,
        ms_path=ms_path,
        output=output,
        config=config,
        default_ref_frequency_hz=default_ref_frequency_hz,
    )


def plot_delays(
    delay_data: NDArray,
    delay_axis: NDArray,
    baseline_names: list[str],
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    labels: list[str] | None = None,
) -> Figure:
    """Plot visibility amplitude vs delay (fringe search).

    Adapted from dsacalib/plotting.py plot_delays().

    Parameters
    ----------
    delay_data :
        Complex visibilities FFT'd along freq axis, shape (nvis, nbl, ndelay, npol)
    delay_axis :
        Delay values in nanoseconds
    baseline_names :
        List of baseline labels
    output :
        Output file path
    config :
        Figure configuration
    labels :
        Labels for each visibility type

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    nvis = delay_data.shape[0]
    nbl = delay_data.shape[1]
    npol = delay_data.shape[-1]

    if labels is None:
        labels = [f"Vis {i}" for i in range(nvis)]

    # Compute peak delays
    delays = delay_axis[np.argmax(np.abs(delay_data), axis=2)]

    # Layout
    nx = min(nbl, 5)
    ny = (nbl + nx - 1) // nx

    alpha = 0.5 if nvis > 2 else 1.0

    for pol_idx in range(npol):
        pol_label = "B" if pol_idx else "A"

        fig, axes = plt.subplots(
            ny, nx, figsize=(config.figsize[0] * nx / 2, config.figsize[1] * ny / 2), sharex=True
        )
        axes = np.atleast_2d(axes).flatten()

        for bl_idx in range(nbl):
            ax = axes[bl_idx]
            ax.axvline(0, color="gray", linestyle="--", alpha=0.5)

            for vis_idx in range(nvis):
                ax.plot(
                    delay_axis,
                    np.log10(np.abs(delay_data[vis_idx, bl_idx, :, pol_idx]) + 1e-10),
                    label=labels[vis_idx],
                    alpha=alpha,
                )
                ax.axvline(delays[vis_idx, bl_idx, pol_idx], color="red", alpha=0.5)

            ax.text(
                0.05,
                0.95,
                f"{baseline_names[bl_idx]}: {pol_label}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
            )

        axes[0].legend(fontsize=8)

        for ax in axes[-(nx):]:
            ax.set_xlabel("Delay (ns)")

        fig.suptitle(f"Delay Search - Pol {pol_label}")
        fig.tight_layout()

        if output:
            out_path = Path(output).with_suffix(f".pol{pol_label}.png")
            fig.savefig(out_path, dpi=config.dpi, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved delay plot: {out_path}")

    return fig


def plot_dynamic_spectrum(
    vis: NDArray,
    freq_ghz: NDArray,
    mjd: NDArray,
    baseline_names: list[str],
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    normalize: bool = False,
    vmin: float = -100,
    vmax: float = 100,
) -> Figure:
    """Plot dynamic spectrum (waterfall) of visibilities.

    Adapted from dsacalib/plotting.py plot_dyn_spec().

    Parameters
    ----------
    vis :
        Visibilities, shape (nbl, ntime, nfreq, npol)
    freq_ghz :
        Frequency array in GHz
    mjd :
        Time array in MJD
    baseline_names :
        List of baseline labels
    output :
        Output file path
    config :
        Figure configuration
    normalize :
        Normalize visibilities by amplitude
    vmin :
        Minimum display value
    vmax :
        Maximum display value

    Returns
    -------
        matplotlib Figure object

    """
    _setup_matplotlib()
    import astropy.units as u
    import matplotlib.pyplot as plt

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    nbl, nt, nf, npol = vis.shape

    # Layout
    nx = min(nbl, 5)
    ny = (nbl * 2 + nx - 1) // nx

    # Rebin if needed
    if nt > 125:
        vis_plot = np.nanmean(vis[:, : nt // 125 * 125, ...].reshape(nbl, 125, -1, nf, npol), 2)
        t_plot = mjd[: nt // 125 * 125].reshape(125, -1).mean(-1)
    else:
        vis_plot = vis.copy()
        t_plot = mjd

    if nf > 125:
        vis_plot = np.nanmean(
            vis_plot[:, :, : nf // 125 * 125, :].reshape(nbl, vis_plot.shape[1], 125, -1, npol), 3
        )
        f_plot = freq_ghz[: nf // 125 * 125].reshape(125, -1).mean(-1)
    else:
        f_plot = freq_ghz

    # Normalize
    dplot = vis_plot.real
    norm_factor = dplot.reshape(nbl, -1, npol).mean(axis=1)[:, np.newaxis, np.newaxis, :]
    dplot = dplot / np.where(norm_factor != 0, norm_factor, 1)

    if normalize:
        dplot = dplot / np.abs(dplot)
        vmin, vmax = -1, 1

    dplot = dplot - 1

    # Time axis in minutes
    t_min = ((t_plot - t_plot[0]) * u.d).to_value(u.min)

    fig, axes = plt.subplots(ny, nx, figsize=(8 * nx, 8 * ny))
    axes = np.atleast_2d(axes).flatten()

    for bl_idx in range(nbl):
        for pol_idx in range(npol):
            ax_idx = pol_idx * nbl + bl_idx
            ax = axes[ax_idx]

            ax.imshow(
                dplot[bl_idx, :, :, pol_idx].T,
                origin="lower",
                interpolation="none",
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
                extent=[t_min[0], t_min[-1], f_plot[0], f_plot[-1]],
                cmap="RdBu_r",
            )
            ax.text(
                0.05,
                0.95,
                f"{baseline_names[bl_idx]}, pol {'B' if pol_idx else 'A'}",
                transform=ax.transAxes,
                fontsize=14,
                color="white",
                verticalalignment="top",
            )

    # Axis labels
    for ax in axes[-(nx):]:
        ax.set_xlabel("Time (min)")
    for ax in axes[::nx]:
        ax.set_ylabel("Freq (GHz)")

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight")
        logger.info(f"Saved dynamic spectrum: {output}")
        plt.close(fig)

    return fig


def plot_flagging_diagnostics(
    caltable: str | Path,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    time_bin_sec: float = 120.0,
    context: PlotContext | None = None,
    interactive: bool | None = None,
) -> list[Path]:
    """Plot per-antenna flag fractions and time-resolved occupancy heatmap.

    Parameters
    ----------
    caltable : Union[str, Path]
    output : Optional[Union[str, Path]]
         (Default value = None)
    config: Optional[FigureConfig] :
         (Default value = None)
    """
    from dsa110_contimg.core.qa.calibration_quality import compute_flag_statistics

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    caltable = Path(caltable)
    if output is None:
        output = caltable.parent / f"{caltable.name}_flagdiag"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    use_interactive = should_generate_interactive(context, interactive)
    stats = compute_flag_statistics(str(caltable), time_bin_sec=time_bin_sec)

    generated: list[Path] = []

    if use_interactive:
        # Bar chart spec
        bar_spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "description": "Per-antenna flag fraction",
            "mark": {"type": "bar", "tooltip": True},
            "data": {"values": stats["per_antenna"]},
            "encoding": {
                "x": {"field": "antenna", "type": "ordinal", "title": "Antenna"},
                "y": {
                    "field": "flag_fraction",
                    "type": "quantitative",
                    "title": "Flagged Fraction",
                    "scale": {"domain": [0, 1]},
                },
                "tooltip": [
                    {"field": "antenna", "type": "ordinal", "title": "Antenna"},
                    {
                        "field": "flag_fraction",
                        "type": "quantitative",
                        "title": "Flag fraction",
                        "format": ".2%",
                    },
                    {"field": "n_rows", "type": "quantitative", "title": "Solutions"},
                ],
            },
            "config": {"axis": {"grid": True, "gridOpacity": 0.2}},
        }
        bar_path = output.with_suffix(".vega.json")
        save_vega_spec(bar_spec, bar_path)
        generated.append(bar_path)

        # Heatmap spec (optional)
        heatmap = stats.get("heatmap")
        time_axis = stats.get("time_axis")
        if heatmap is not None and time_axis is not None:
            values = []
            for ant_idx, row in enumerate(heatmap):
                for ti, frac in enumerate(row):
                    if np.isnan(frac):
                        continue
                    values.append(
                        {
                            "antenna": ant_idx,
                            "time_min": float(time_axis[ti] / 60.0),
                            "flag_fraction": float(frac),
                        }
                    )
            heat_spec = {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "description": "Flag occupancy heatmap",
                "mark": "rect",
                "data": {"values": values},
                "encoding": {
                    "x": {"field": "time_min", "type": "quantitative", "title": "Time (min)"},
                    "y": {"field": "antenna", "type": "ordinal", "title": "Antenna"},
                    "color": {
                        "field": "flag_fraction",
                        "type": "quantitative",
                        "title": "Flagged",
                        "scale": {"domain": [0, 1], "scheme": "reds"},
                    },
                },
            }
            heat_path = output.with_name(f"{output.name}_heatmap.vega.json")
            save_vega_spec(heat_spec, heat_path)
            generated.append(heat_path)
        return generated

    # Static plots
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    with plt.rc_context(config.to_mpl_params()):
        fig, axes = plt.subplots(1, 2, figsize=(config.figsize[0] * 2, config.figsize[1]))

        # Bar chart
        antennas = [d["antenna"] for d in stats["per_antenna"]]
        frac = [d["flag_fraction"] for d in stats["per_antenna"]]
        axes[0].bar(antennas, frac, color="tab:red", alpha=0.7)
        axes[0].set_xlabel("Antenna")
        axes[0].set_ylabel("Flagged Fraction")
        axes[0].set_ylim(0, 1)
        axes[0].set_title("Flagged Fraction per Antenna")
        axes[0].grid(True, alpha=0.3)

        # Heatmap
        heatmap = stats.get("heatmap")
        time_axis = stats.get("time_axis")
        if heatmap is not None and time_axis is not None:
            im = axes[1].imshow(
                heatmap,
                origin="lower",
                aspect="auto",
                interpolation="none",
                vmin=0,
                vmax=1,
                extent=[0, (len(time_axis) * time_bin_sec) / 60.0, 0, heatmap.shape[0]],
                cmap="Reds",
            )
            axes[1].set_xlabel("Time (min)")
            axes[1].set_ylabel("Antenna")
            axes[1].set_title("Flag Occupancy (Antenna × Time)")
            fig.colorbar(im, ax=axes[1], label="Flagged Fraction")
        else:
            axes[1].axis("off")
            axes[1].set_title("No time axis available")

        fig.suptitle(f"Flagging Diagnostics: {caltable.name}")
        fig.tight_layout()

        out_path = output.with_suffix(".png")
        fig.savefig(out_path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
        generated.append(out_path)

    return generated


def plot_gain_snr(
    caltable: str | Path,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    context: PlotContext | None = None,
    interactive: bool | None = None,
) -> list[Path]:
    """Plot per-antenna SNR distributions and SNR vs time.

    Parameters
    ----------
    caltable : Union[str, Path]
    output : Optional[Union[str, Path]]
         (Default value = None)
    config: Optional[FigureConfig] :
         (Default value = None)
    context: Optional[PlotContext] :
         (Default value = None)
    interactive: Optional[bool] :
         (Default value = None)

    """
    from dsa110_contimg.core.qa.calibration_quality import extract_gain_snr

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    caltable = Path(caltable)
    if output is None:
        output = caltable.parent / f"{caltable.name}_snr"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    snr = extract_gain_snr(str(caltable))
    use_interactive = should_generate_interactive(context, interactive)
    generated: list[Path] = []

    if use_interactive:
        values = []
        for ant_data in snr["per_antenna"]:
            snr_time = ant_data.get("snr_time")
            if snr_time is None or (hasattr(snr_time, "__len__") and len(snr_time) == 0):
                continue
            for s, t in zip(ant_data["snr_values"], snr_time):
                values.append(
                    {
                        "antenna": ant_data["antenna"],
                        "snr": float(s),
                        "time_min": float((t - snr_time[0]) / 60.0),
                    }
                )
        scatter_spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "description": "SNR vs time per antenna",
            "data": {"values": values},
            "mark": {"type": "point", "tooltip": True, "opacity": 0.6},
            "encoding": {
                "x": {"field": "time_min", "type": "quantitative", "title": "Time (min)"},
                "y": {"field": "snr", "type": "quantitative", "title": "SNR"},
                "color": {"field": "antenna", "type": "nominal", "legend": {"title": "Antenna"}},
            },
        }
        hist_spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "description": "SNR distribution",
            "data": {"values": [{"snr": float(v)} for v in snr["snr_flat"]]},
            "mark": "bar",
            "encoding": {
                "x": {"field": "snr", "type": "quantitative", "bin": True, "title": "SNR"},
                "y": {"aggregate": "count", "type": "quantitative", "title": "Count"},
            },
        }
        scatter_path = output.with_suffix(".vega.json")
        hist_path = output.with_name(f"{output.name}_hist.vega.json")
        save_vega_spec(scatter_spec, scatter_path)
        save_vega_spec(hist_spec, hist_path)
        generated.extend([scatter_path, hist_path])
        return generated

    _setup_matplotlib()
    import matplotlib.pyplot as plt

    with plt.rc_context(config.to_mpl_params()):
        fig, axes = plt.subplots(1, 2, figsize=(config.figsize[0] * 2, config.figsize[1]))

        # Histogram
        axes[0].hist(snr["snr_flat"], bins=30, alpha=0.7, edgecolor="black")
        axes[0].set_xlabel("SNR")
        axes[0].set_ylabel("Count")
        axes[0].set_title("SNR Distribution")
        axes[0].axvline(snr["summary"]["median"], color="r", linestyle="--", label="Median")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Time series
        time_axis = snr.get("time_min")
        if time_axis is not None:
            for ant_data in snr["per_antenna"]:
                t = (
                    (ant_data.get("snr_time") - ant_data.get("snr_time")[0]) / 60.0
                    if ant_data.get("snr_time") is not None
                    else None
                )
                if t is None:
                    continue
                axes[1].plot(
                    t, ant_data["snr_values"], ".", alpha=0.6, label=f"Ant {ant_data['antenna']}"
                )
            axes[1].set_xlabel("Time (min)")
            axes[1].set_ylabel("SNR")
            axes[1].set_title("SNR vs Time")
            axes[1].legend(ncol=2, fontsize=8)
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].axis("off")
            axes[1].set_title("No time axis available")

        fig.suptitle(f"SNR Diagnostics: {caltable.name}")
        fig.tight_layout()
        out_path = output.with_suffix(".png")
        fig.savefig(out_path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
        generated.append(out_path)

    return generated


def plot_dterm_scatter(
    caltable: str | Path,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    context: PlotContext | None = None,
    interactive: bool | None = None,
) -> list[Path]:
    """Plot D-term leakage estimates on the complex plane and amplitude/phase histograms.

    Parameters
    ----------
    caltable : Union[str, Path]
    output : Optional[Union[str, Path]]
         (Default value = None)
    config: Optional[FigureConfig] :
         (Default value = None)
    context: Optional[PlotContext] :
         (Default value = None)
    interactive: Optional[bool] :
         (Default value = None)

    """
    from dsa110_contimg.core.qa.calibration_quality import extract_dterms

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    caltable = Path(caltable)
    if output is None:
        output = caltable.parent / f"{caltable.name}_dterms"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    dterms = extract_dterms(str(caltable))
    use_interactive = should_generate_interactive(context, interactive)
    generated: list[Path] = []

    if use_interactive:
        values = []
        for ant in dterms["per_antenna"]:
            for entry in ant["dterms"]:
                values.append(
                    {
                        "antenna": ant["antenna"],
                        "pol": entry["pol"],
                        "real": entry["real"],
                        "imag": entry["imag"],
                        "amp": entry["amp"],
                        "phase_deg": entry["phase_deg"],
                    }
                )
        scatter_spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "description": "D-term complex scatter",
            "data": {"values": values},
            "mark": {"type": "point", "tooltip": True, "opacity": 0.7},
            "encoding": {
                "x": {"field": "real", "type": "quantitative", "title": "Real"},
                "y": {"field": "imag", "type": "quantitative", "title": "Imag"},
                "color": {"field": "pol", "type": "nominal", "title": "Pol"},
                "shape": {"field": "antenna", "type": "nominal", "title": "Antenna"},
            },
        }
        scatter_path = output.with_suffix(".vega.json")
        save_vega_spec(scatter_spec, scatter_path)
        generated.append(scatter_path)
        return generated

    _setup_matplotlib()
    import matplotlib.pyplot as plt

    with plt.rc_context(config.to_mpl_params()):
        fig, axes = plt.subplots(1, 2, figsize=(config.figsize[0] * 2, config.figsize[1]))

        # Complex plane
        for ant in dterms["per_antenna"]:
            for entry in ant["dterms"]:
                axes[0].scatter(
                    entry["real"],
                    entry["imag"],
                    alpha=0.7,
                    label=f"Ant {ant['antenna']} {entry['pol']}",
                )
        axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.5)
        axes[0].axvline(0, color="gray", linestyle="--", linewidth=0.5)
        axes[0].set_xlabel("Real")
        axes[0].set_ylabel("Imag")
        axes[0].set_title("D-term Complex Scatter")
        axes[0].legend(fontsize=8, ncol=2)
        axes[0].grid(True, alpha=0.3)

        # Amplitude histogram
        amps = [entry["amp"] for ant in dterms["per_antenna"] for entry in ant["dterms"]]
        axes[1].hist(amps, bins=30, alpha=0.7, edgecolor="black")
        axes[1].set_xlabel("Amplitude")
        axes[1].set_ylabel("Count")
        axes[1].set_title("D-term Amplitude Distribution")
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(f"D-term Diagnostics: {caltable.name}")
        fig.tight_layout()
        out_path = output.with_suffix(".png")
        fig.savefig(out_path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
        generated.append(out_path)

    return generated


def plot_gain_comparison(
    caltable_a: str | Path,
    caltable_b: str | Path,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
) -> list[Path]:
    """Compare two calibration tables by plotting per-antenna median amplitudes and phases.

    Parameters
    ----------
    caltable_a : Union[str, Path]
    caltable_b: Union[str :

    output : Optional[Union[str, Path]]
         (Default value = None)
    config: Optional[FigureConfig] :
         (Default value = None)

    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    try:
        from casacore.tables import table
    except ImportError as exc:
        raise RuntimeError("casacore.tables is required for gain comparison plots") from exc

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    caltable_a = Path(caltable_a)
    caltable_b = Path(caltable_b)
    if output is None:
        output = caltable_a.parent / f"{caltable_a.name}_vs_{caltable_b.name}"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def _load_stats(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with table(str(path), readonly=True) as tb:
            cparam = tb.getcol("CPARAM")
            ants = (
                tb.getcol("ANTENNA1") if "ANTENNA1" in tb.colnames() else np.arange(cparam.shape[0])
            )
        amp = np.abs(cparam).reshape(cparam.shape[0], -1)
        phase = np.degrees(np.angle(cparam)).reshape(cparam.shape[0], -1)
        med_amp = np.array([np.nanmedian(amp[ants == ant]) for ant in np.unique(ants)])
        med_phase = np.array([np.nanmedian(phase[ants == ant]) for ant in np.unique(ants)])
        return np.unique(ants), med_amp, med_phase

    ants_a, amp_a, phase_a = _load_stats(caltable_a)
    ants_b, amp_b, phase_b = _load_stats(caltable_b)
    ants = np.unique(np.concatenate([ants_a, ants_b]))

    # Align amplitudes by antenna
    amp_map_a = {int(a): v for a, v in zip(ants_a, amp_a)}
    amp_map_b = {int(a): v for a, v in zip(ants_b, amp_b)}
    phase_map_a = {int(a): v for a, v in zip(ants_a, phase_a)}
    phase_map_b = {int(a): v for a, v in zip(ants_b, phase_b)}

    amp_vals_a = [amp_map_a.get(int(a), np.nan) for a in ants]
    amp_vals_b = [amp_map_b.get(int(a), np.nan) for a in ants]
    phase_vals_a = [phase_map_a.get(int(a), np.nan) for a in ants]
    phase_vals_b = [phase_map_b.get(int(a), np.nan) for a in ants]

    fig, axes = plt.subplots(1, 2, figsize=(config.figsize[0] * 2, config.figsize[1]))

    axes[0].plot(ants, amp_vals_a, "o-", label=caltable_a.name)
    axes[0].plot(ants, amp_vals_b, "o-", label=caltable_b.name)
    axes[0].set_xlabel("Antenna")
    axes[0].set_ylabel("Median Amplitude")
    axes[0].set_title("Gain Amplitude Comparison")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(ants, phase_vals_a, "o-", label=caltable_a.name)
    axes[1].plot(ants, phase_vals_b, "o-", label=caltable_b.name)
    axes[1].set_xlabel("Antenna")
    axes[1].set_ylabel("Median Phase (deg)")
    axes[1].set_title("Gain Phase Comparison")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Calibration Comparison")
    fig.tight_layout()
    out_path = output.with_suffix(".png")
    fig.savefig(out_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)

    return [out_path]
