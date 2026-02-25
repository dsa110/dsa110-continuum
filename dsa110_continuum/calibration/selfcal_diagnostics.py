"""
Self-calibration diagnostics generation.

This module provides automatic diagnostic plot generation after self-calibration
completes. It integrates with the convergence_plots and closure_phase_plots
visualization modules.

Usage:
    from dsa110_contimg.core.calibration.selfcal_diagnostics import (
        generate_selfcal_diagnostics,
    )

    # After selfcal_ms() completes:
    success, result = selfcal_ms(ms_path, output_dir, config)
    if success:
        generate_selfcal_diagnostics(result, output_dir, ms_path)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def generate_selfcal_diagnostics(
    result: dict[str, Any],
    output_dir: str | Path,
    ms_path: str | Path | None = None,
    include_closure_phases: bool = True,
) -> dict[str, Path]:
    """Generate diagnostic plots from self-calibration results.

    Creates:
    - Convergence plot (SNR, chi-squared, RMS vs iteration)
    - Antenna solution quality plot
    - Chi-squared improvement plot
    - Closure phase histogram (if MS provided)

    Parameters
    ----------
    result : Dict[str, Any]
        Dictionary from selfcal_ms() (SelfCalResult as dict).
    output_dir : Union[str, Path]
        Directory to save diagnostic plots.
    ms_path : Optional[Union[str, Path]], optional
        Optional MS path for closure phase analysis (default is None).
    include_closure_phases : bool, optional
        Whether to compute closure phases (slower, default is True).

    Returns
    -------
    Dict[str, Path]
        Dictionary mapping plot name to output path.
    """
    from dsa110_contimg.core.visualization.convergence_plots import (
        ConvergenceData,
        plot_antenna_solution_quality,
        plot_chi_squared_improvement,
        plot_selfcal_convergence,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generated_plots: dict[str, Path] = {}

    # Extract iteration data
    iterations = result.get("iterations", [])
    if not iterations:
        logger.warning("No iteration data available for diagnostics")
        return generated_plots

    # Build ConvergenceData from result dict
    data = ConvergenceData(
        iterations=[it.get("iteration", i) for i, it in enumerate(iterations)],
        chi_squared=[it.get("chi_squared", 0.0) for it in iterations],
        snr=[it.get("snr", 0.0) for it in iterations],
        rms=[it.get("rms", 0.0) for it in iterations],
        peak_flux=[it.get("peak_flux", 0.0) for it in iterations],
        antenna_snr_median=[it.get("antenna_snr_median", 0.0) for it in iterations],
        antenna_snr_min=[it.get("antenna_snr_min", 0.0) for it in iterations],
        phase_scatter=[it.get("phase_scatter_deg", 0.0) for it in iterations],
        amp_scatter=[it.get("amp_scatter_frac", 0.0) for it in iterations],
        mode=[it.get("mode", "unknown") for it in iterations],
        solint=[it.get("solint", "") for it in iterations],
    )

    # 1. Main convergence plot (4-panel)
    try:
        convergence_path = output_path / "selfcal_convergence.png"
        plot_selfcal_convergence(data, output=convergence_path)
        generated_plots["convergence"] = convergence_path
        logger.info(f"Generated convergence plot: {convergence_path}")
    except Exception as e:
        logger.warning(f"Failed to generate convergence plot: {e}")

    # 2. Antenna solution quality
    if any(data.antenna_snr_median or []):
        try:
            antenna_path = output_path / "selfcal_antenna_quality.png"
            plot_antenna_solution_quality(data, output=antenna_path)
            generated_plots["antenna_quality"] = antenna_path
            logger.info(f"Generated antenna quality plot: {antenna_path}")
        except Exception as e:
            logger.warning(f"Failed to generate antenna quality plot: {e}")

    # 3. Chi-squared improvement
    initial_chi_sq = result.get("initial_chi_squared", 0.0)
    chi_squared_values = [it.get("chi_squared", 0.0) for it in iterations]

    if initial_chi_sq > 0 and any(chi_squared_values):
        try:
            chi_sq_path = output_path / "selfcal_chi_squared.png"
            plot_chi_squared_improvement(
                initial_chi_sq=initial_chi_sq,
                chi_squared_per_iteration=[initial_chi_sq] + chi_squared_values,
                output=chi_sq_path,
            )
            generated_plots["chi_squared"] = chi_sq_path
            logger.info(f"Generated chi-squared plot: {chi_sq_path}")
        except Exception as e:
            logger.warning(f"Failed to generate chi-squared plot: {e}")

    # 4. Closure phases (if MS available and requested)
    if ms_path and include_closure_phases:
        try:
            from dsa110_contimg.core.visualization.closure_phase_plots import (
                compute_closure_phases,
                extract_closure_phases_from_ms,
                plot_closure_phase_antenna_contribution,
                plot_closure_phase_histogram,
            )

            ms_path = Path(ms_path)
            if ms_path.exists():
                logger.info("Extracting closure phases from MS...")
                ms_data = extract_closure_phases_from_ms(ms_path)

                cp = compute_closure_phases(
                    ms_data["visibility"],
                    ms_data["antenna1"],
                    ms_data["antenna2"],
                )

                # Histogram
                cp_hist_path = output_path / "selfcal_closure_phases.png"
                plot_closure_phase_histogram(cp, output=cp_hist_path)
                generated_plots["closure_phases"] = cp_hist_path
                logger.info(f"Generated closure phase histogram: {cp_hist_path}")

                # Antenna contribution
                cp_ant_path = output_path / "selfcal_closure_phase_antennas.png"
                plot_closure_phase_antenna_contribution(
                    cp,
                    output=cp_ant_path,
                    antenna_names=ms_data.get("antenna_names"),
                )
                generated_plots["closure_phase_antennas"] = cp_ant_path
                logger.info(f"Generated closure phase antenna plot: {cp_ant_path}")
        except ImportError:
            logger.debug("casacore not available, skipping closure phases")
        except Exception as e:
            logger.warning(f"Failed to generate closure phase plots: {e}")

    logger.info(f"Generated {len(generated_plots)} diagnostic plots in {output_path}")
    return generated_plots


def generate_observation_diagnostics(
    ms_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Generate observation geometry diagnostics from an MS.

    Creates:
    - Elevation vs time plot
    - Observation summary (el, az, parallactic angle)
    - UV coverage plot

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    output_dir :
        Directory to save diagnostic plots
    ms_path : Union[str, Path]
    output_dir: Union[str :


    Returns
    -------
        Dictionary mapping plot name to output path

    """
    from dsa110_contimg.core.visualization.elevation_plots import (
        extract_geometry_from_ms,
        plot_elevation_vs_time,
        plot_observation_summary,
    )
    from dsa110_contimg.core.visualization.uv_plots import (
        extract_uv_from_ms,
        plot_uv_coverage,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ms_path = Path(ms_path)

    generated_plots: dict[str, Path] = {}

    # 1. Observation geometry
    try:
        geom = extract_geometry_from_ms(ms_path)
        times_hours = (geom["times"] - geom["times"].min()) / 3600

        # Elevation plot
        el_path = output_path / "obs_elevation.png"
        plot_elevation_vs_time(times_hours, geom["elevation_deg"], output=el_path)
        generated_plots["elevation"] = el_path

        # Full summary
        summary_path = output_path / "obs_summary.png"
        plot_observation_summary(
            times_hours,
            geom["dec_deg"],
            geom["ra_deg"],
            output=summary_path,
            title=f"Observation: {ms_path.name}",
        )
        generated_plots["obs_summary"] = summary_path

        logger.info("Generated observation geometry plots")
    except Exception as e:
        logger.warning(f"Failed to generate geometry plots: {e}")

    # 2. UV coverage
    try:
        uv_data = extract_uv_from_ms(ms_path)
        uv_path = output_path / "uv_coverage.png"
        plot_uv_coverage(
            uv_data["u"],
            uv_data["v"],
            output=uv_path,
            title=f"UV Coverage: {ms_path.name}",
        )
        generated_plots["uv_coverage"] = uv_path
        logger.info("Generated UV coverage plot")
    except Exception as e:
        logger.warning(f"Failed to generate UV coverage plot: {e}")

    return generated_plots


__all__ = [
    "generate_selfcal_diagnostics",
    "generate_observation_diagnostics",
]
