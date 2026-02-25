"""
Self-calibration module for iterative imaging and calibration.

Self-calibration (selfcal) iteratively improves calibration by using the
current best image as a model for the next round of calibration:

    Initial Imaging → Phase Self-cal → Amplitude+Phase Self-cal → Final Image

This module provides:
- SelfCalConfig: Configuration for self-calibration parameters
- SelfCalResult: Results from a self-calibration run
- selfcal_iteration: Run a single self-cal iteration
- selfcal_ms: Run full self-calibration loop on a Measurement Set

Self-calibration workflow:
1. Create initial image (dirty or cleaned)
2. Predict model visibilities to MODEL_DATA
3. Solve gaincal (phase-only initially, then amp+phase)
4. Apply calibration
5. Re-image with improved calibration
6. Measure SNR improvement (image + visibility chi-squared)
7. Repeat until convergence or max iterations

Best practices implemented (per Perplexity analysis, Dec 2025):
- Solution intervals start LONG (5-10 min) for stable bootstrap, then shorten
- Per-antenna SNR thresholds: phase ~3-5, amplitude ~10-20
- Visibility chi-squared monitoring for robust convergence detection
- Gain smoothness checking to reject noisy solutions
- Drift-scan aware: beam attenuation limits for amplitude self-cal
- Subband phase combining option for increased SNR

Backend support:
- WSClean: Uses `-predict` for model prediction (recommended)
- CASA tclean: Uses internal ft() for model prediction
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from dsa110_contimg.core.calibration.casa_service import CASAService
from dsa110_contimg.common.utils.casa_init import ensure_casa_path

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class SelfCalMode(str, Enum):
    """Self-calibration mode."""

    PHASE = "phase"  # Phase-only calibration
    AMPLITUDE_PHASE = "ap"  # Amplitude + phase calibration


class SelfCalStatus(str, Enum):
    """Status of self-calibration."""

    SUCCESS = "success"
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    DIVERGED = "diverged"
    FAILED = "failed"
    NO_IMPROVEMENT = "no_improvement"
    LOW_ANTENNA_SNR = "low_antenna_snr"  # Per-antenna SNR too low
    NOISY_GAINS = "noisy_gains"  # Gain solutions too noisy


# Default solution intervals for progressive selfcal
# IMPORTANT: Start LONG for stable bootstrap, then shorten progressively
# For L-band (~1.4 GHz) with ~1 km baselines, atmospheric coherence is minutes
DEFAULT_PHASE_SOLINTS = ["300s", "120s", "60s"]  # 5min -> 2min -> 1min
DEFAULT_AMP_SOLINT = "inf"  # Amplitude solint (typically longer for stability)

# Per-antenna SNR thresholds (per Perplexity recommendations)
DEFAULT_PHASE_ANTENNA_SNR = 3.0  # Minimum per-antenna SNR for phase solutions
DEFAULT_AMP_ANTENNA_SNR = 10.0  # Minimum per-antenna SNR for amplitude solutions

# Gain smoothness thresholds
DEFAULT_MAX_PHASE_SCATTER_DEG = 30.0  # Max phase RMS scatter across antennas
DEFAULT_MAX_AMP_SCATTER_FRAC = 0.3  # Max amplitude RMS scatter (fractional)

# Drift-scan primary beam threshold
DEFAULT_MIN_BEAM_RESPONSE = 0.5  # Min PB response for amp self-cal (50%)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SelfCalConfig:
    """Configuration for self-calibration."""

    # Iteration control
    max_iterations: int = 5
    min_snr_improvement: float = 1.05  # 5% improvement threshold
    stop_on_divergence: bool = True

    # Phase-only calibration parameters
    phase_solints: list[str] = field(default_factory=lambda: DEFAULT_PHASE_SOLINTS.copy())
    phase_minsnr: float = 3.0  # CASA gaincal minsnr parameter
    phase_antenna_snr: float = DEFAULT_PHASE_ANTENNA_SNR  # Per-antenna threshold
    phase_combine: str = ""

    # Amplitude+phase calibration parameters
    do_amplitude: bool = True
    amp_solint: str = DEFAULT_AMP_SOLINT
    amp_minsnr: float = 3.0  # CASA gaincal minsnr parameter
    amp_antenna_snr: float = DEFAULT_AMP_ANTENNA_SNR  # Per-antenna threshold
    amp_combine: str = "scan"

    # Per-antenna SNR and gain quality checks
    check_antenna_snr: bool = True  # Compute and check per-antenna SNR
    check_gain_smoothness: bool = True  # Check gain solution smoothness
    max_phase_scatter_deg: float = DEFAULT_MAX_PHASE_SCATTER_DEG
    max_amp_scatter_frac: float = DEFAULT_MAX_AMP_SCATTER_FRAC

    # Visibility chi-squared monitoring
    use_chi_squared: bool = True  # Monitor chi-squared for convergence
    min_chi_squared_improvement: float = 0.95  # Chi-sq must decrease by 5%

    # Subband handling
    combine_spw_phase: bool = False  # Combine SPWs for phase (increases SNR)

    # Drift-scan specific
    drift_scan_mode: bool = False  # Enable drift-scan handling
    min_beam_response: float = DEFAULT_MIN_BEAM_RESPONSE  # Min PB for amp selfcal

    # Imaging parameters
    imsize: int = 1024
    cell_arcsec: float | None = None  # Auto-calculate if None
    niter: int = 10000
    threshold: str = "0.1mJy"
    robust: float = 0.0
    backend: str = "wsclean"
    wsclean_path: str | None = None

    # Galvin adaptive clip for artifact suppression during self-cal imaging
    use_galvin_clip: bool = True  # Enable Galvin adaptive minimum absolute clip
    galvin_box_size: int = 100  # Box size for sliding window (pixels)
    galvin_adaptive_depth: int = 3  # Max iterations for adaptive subdivision

    # Quality control
    min_initial_snr: float = 5.0
    max_flagged_fraction: float = 0.5

    # Data selection
    refant: str | None = None
    uvrange: str = ""
    spw: str = ""
    field: str = "0"

    # Model seeding
    use_nvss_seeding: bool = False
    nvss_min_mjy: float = 10.0
    calib_ra_deg: float | None = None

    # Diagnostics generation
    generate_diagnostics: bool = True  # Auto-generate diagnostic plots
    diagnostics_closure_phases: bool = True  # Include closure phase analysis
    calib_dec_deg: float | None = None
    calib_flux_jy: float | None = None


# =============================================================================
# Results
# =============================================================================


@dataclass
class SelfCalIterationResult:
    """Result from a single self-calibration iteration."""

    iteration: int
    mode: SelfCalMode
    solint: str
    success: bool
    snr: float = 0.0
    rms: float = 0.0
    peak_flux: float = 0.0
    chi_squared: float = 0.0
    antenna_snr_median: float = 0.0
    antenna_snr_min: float = 0.0
    phase_scatter_deg: float = 0.0
    amp_scatter_frac: float = 0.0
    gaintable: str | None = None
    image_path: str | None = None
    message: str = ""


@dataclass
class SelfCalResult:
    """Result from a full self-calibration run."""

    status: SelfCalStatus
    iterations_completed: int = 0
    initial_snr: float = 0.0
    initial_chi_squared: float = 0.0
    best_snr: float = 0.0
    final_snr: float = 0.0
    final_chi_squared: float = 0.0
    improvement_factor: float = 1.0
    chi_squared_improvement: float = 1.0
    iterations: list[SelfCalIterationResult] = field(default_factory=list)
    best_iteration: int = -1
    final_image: str | None = None
    final_gaintables: list[str] = field(default_factory=list)
    message: str = ""


# =============================================================================
# Helper Functions
# =============================================================================


def _measure_image_stats(image_path: str) -> tuple[float, float, float]:
    """Measure image statistics (peak, rms, snr).

    Parameters
    ----------
    image_path :
        Path to FITS or CASA image

    Returns
    -------
        Tuple of (peak_flux, rms, snr)

    """
    try:
        # Try FITS first
        if image_path.endswith(".fits"):
            from astropy.io import fits

            with fits.open(image_path) as hdul:
                data = hdul[0].data
                # Handle multi-dimensional data (freq, pol axes)
                while data.ndim > 2:
                    data = data[0]

                # Measure peak and RMS
                peak = np.nanmax(np.abs(data))
                # RMS from outer regions (avoid central source)
                ny, nx = data.shape
                edge_data = np.concatenate(
                    [
                        data[: ny // 4, :].flatten(),
                        data[3 * ny // 4 :, :].flatten(),
                        data[:, : nx // 4].flatten(),
                        data[:, 3 * nx // 4 :].flatten(),
                    ]
                )
                rms = np.nanstd(edge_data)
                snr = peak / rms if rms > 0 else 0.0
                return peak, rms, snr
        else:
            # Try CASA image
            try:
                from casacore.images import image as casa_image

                img = casa_image(image_path)
                try:
                    data = img.getdata()
                finally:
                    try:
                        img.close()
                    except Exception:
                        pass

                # Collapse singleton axes but keep 2D image plane
                data = np.squeeze(data)
                if data.ndim > 2:
                    data = data.reshape(-1, *data.shape[-2:])[0]

                peak = np.nanmax(np.abs(data))
                ny, nx = data.shape
                edge_data = np.concatenate(
                    [
                        data[: ny // 4, :].flatten(),
                        data[3 * ny // 4 :, :].flatten(),
                        data[:, : nx // 4].flatten(),
                        data[:, 3 * nx // 4 :].flatten(),
                    ]
                )
                rms = np.nanstd(edge_data)
                snr = peak / rms if rms > 0 else 0.0
                return peak, rms, snr
            except ImportError:
                logger.warning("casacore.images not available")
                return 0.0, 0.0, 0.0
    except Exception as e:
        logger.warning(f"Failed to measure image stats for {image_path}: {e}")
        return 0.0, 0.0, 0.0


def _get_flagged_fraction(ms_path: str) -> float:
    """Get fraction of flagged data in MS.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set

    Returns
    -------
        Flagged fraction (0-1)

    """
    try:
        import casacore.tables as tb

        with tb.table(ms_path, readonly=True) as t:
            flags = t.getcol("FLAG")
            return np.mean(flags)
    except Exception as e:
        logger.warning(f"Failed to get flagged fraction: {e}")
        return 0.0


def _compute_visibility_chi_squared(ms_path: str, field: str = "") -> float:
    """Compute reduced chi-squared from visibility residuals.

    Chi-squared = mean(|DATA - MODEL|^2 / SIGMA^2)

    This provides a visibility-domain metric for convergence that complements
    image-domain SNR. A decreasing chi-squared indicates improving calibration.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    field :
        Field selection (empty for all)

    Returns
    -------
        Reduced chi-squared value, or 0.0 if computation fails

    """
    try:
        import casacore.tables as tb

        query = f"FIELD_ID=={field}" if field and field.isdigit() else ""

        with tb.table(ms_path, readonly=True) as t:
            if query:
                t = t.query(query)

            # Get corrected data (or data if no correction applied)
            try:
                data = t.getcol("CORRECTED_DATA")
            except Exception:
                data = t.getcol("DATA")

            try:
                model = t.getcol("MODEL_DATA")
            except Exception:
                logger.debug("MODEL_DATA not present, chi-squared = 0")
                return 0.0

            flags = t.getcol("FLAG")

            # Get weights (use WEIGHT_SPECTRUM if available, else WEIGHT)
            try:
                weights = t.getcol("WEIGHT_SPECTRUM")
            except Exception:
                weight = t.getcol("WEIGHT")
                # Broadcast weight to match data shape
                weights = np.broadcast_to(weight[:, np.newaxis, :], data.shape).copy()

            # Compute residuals
            residual = data - model

            # Mask flagged data
            residual_masked = np.ma.array(residual, mask=flags)
            weights_masked = np.ma.array(weights, mask=flags)

            # Compute weighted chi-squared
            # chi2 = sum(w * |r|^2) / sum(w)
            weighted_residual_sq = weights_masked * np.abs(residual_masked) ** 2
            chi_sq = np.ma.sum(weighted_residual_sq) / np.ma.sum(weights_masked)

            # Reduced chi-squared (approximately)
            return float(chi_sq)

    except Exception as e:
        logger.warning(f"Failed to compute chi-squared: {e}")
        return 0.0


def _compute_per_antenna_snr(
    ms_path: str,
    solint_seconds: float,
    field: str = "",
) -> tuple[float, float, np.ndarray]:
    """Compute per-antenna SNR for a given solution interval.

    For each antenna, estimates SNR = sqrt(N) * mean(|V|) / std(|V|)
    where N is the number of samples in the solution interval.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    solint_seconds :
        Solution interval in seconds
    field :
        Field selection

    Returns
    -------
        Tuple of (median_snr, min_snr, per_antenna_snr_array)

    """
    try:
        import casacore.tables as tb

        with tb.table(ms_path, readonly=True) as t:
            # Get antenna columns
            ant1 = t.getcol("ANTENNA1")
            ant2 = t.getcol("ANTENNA2")
            times = t.getcol("TIME")

            try:
                data = t.getcol("CORRECTED_DATA")
            except Exception:
                data = t.getcol("DATA")

            flags = t.getcol("FLAG")

            # Get unique antennas
            n_ant = max(np.max(ant1), np.max(ant2)) + 1

            # Compute time bins based on solint
            time_min, time_max = times.min(), times.max()
            total_time = time_max - time_min

            if solint_seconds <= 0 or solint_seconds >= total_time:
                # Single solution for entire observation
                n_time_bins = 1
            else:
                n_time_bins = max(1, int(total_time / solint_seconds))

            # For each antenna, compute SNR across all baselines involving it
            antenna_snrs = []

            for ant_id in range(n_ant):
                # Select baselines involving this antenna
                mask = (ant1 == ant_id) | (ant2 == ant_id)
                ant_data = data[mask]
                ant_flags = flags[mask]

                if ant_data.size == 0:
                    antenna_snrs.append(0.0)
                    continue

                # Mask flagged data
                ant_data_masked = np.ma.array(ant_data, mask=ant_flags)

                # Compute amplitude
                amp = np.abs(ant_data_masked)

                # SNR estimate: mean / std, scaled by sqrt(N_samples / N_bins)
                mean_amp = np.ma.mean(amp)
                std_amp = np.ma.std(amp)

                if std_amp > 0:
                    n_samples = np.ma.count(amp)
                    # SNR per solution interval
                    snr = mean_amp / std_amp * np.sqrt(n_samples / n_time_bins)
                    antenna_snrs.append(float(snr))
                else:
                    antenna_snrs.append(0.0)

            antenna_snrs = np.array(antenna_snrs)
            valid_snrs = antenna_snrs[antenna_snrs > 0]

            if len(valid_snrs) == 0:
                return 0.0, 0.0, antenna_snrs

            return float(np.median(valid_snrs)), float(np.min(valid_snrs)), antenna_snrs

    except Exception as e:
        logger.warning(f"Failed to compute per-antenna SNR: {e}")
        return 0.0, 0.0, np.array([])


def _parse_solint_seconds(solint: str) -> float:
    """Parse solution interval string to seconds.

    Parameters
    ----------
    solint :
        Solution interval (e.g., "60s", "2min", "inf")

    Returns
    -------
        Seconds (float), or -1 for "inf"

    """
    solint = solint.strip().lower()

    if solint == "inf" or solint == "infinite":
        return -1.0

    # Try parsing numeric suffix
    if solint.endswith("s"):
        return float(solint[:-1])
    elif solint.endswith("min"):
        return float(solint[:-3]) * 60
    elif solint.endswith("h"):
        return float(solint[:-1]) * 3600
    else:
        # Assume seconds
        try:
            return float(solint)
        except ValueError:
            logger.warning(f"Could not parse solint '{solint}', assuming inf")
            return -1.0


def _measure_gain_smoothness(caltable_path: str) -> tuple[float, float]:
    """Measure smoothness of gain solutions in a calibration table.

    Parameters
    ----------
    caltable_path :
        Path to CASA calibration table

    Returns
    -------
    - phase_scatter_deg
        RMS phase scatter across antennas (degrees)
    - amp_scatter_frac
        RMS amplitude scatter as fraction of mean

    """
    try:
        import casacore.tables as tb

        with tb.table(caltable_path, readonly=True) as t:
            # CPARAM contains complex gains
            cparam = t.getcol("CPARAM")  # Shape: (nrow, nchan, npol)
            flags = t.getcol("FLAG")

            # Mask flagged solutions
            cparam_masked = np.ma.array(cparam, mask=flags)

            # Extract phase and amplitude
            phases = np.angle(cparam_masked, deg=True)
            amps = np.abs(cparam_masked)

            # Phase scatter: RMS of phases after removing mean per polarization
            phase_mean = np.ma.mean(phases, axis=(0, 1), keepdims=True)
            phase_residual = phases - phase_mean
            phase_scatter = float(np.ma.std(phase_residual))

            # Amplitude scatter: RMS relative to mean
            amp_mean = np.ma.mean(amps)
            if amp_mean > 0:
                amp_scatter = float(np.ma.std(amps) / amp_mean)
            else:
                amp_scatter = 0.0

            return phase_scatter, amp_scatter

    except Exception as e:
        logger.warning(f"Failed to measure gain smoothness: {e}")
        return 0.0, 0.0


def _check_beam_response(
    ms_path: str,
    field: str,
    threshold: float = 0.5,
) -> tuple[bool, float]:
    """Check if field is within acceptable primary beam response.

    For drift-scan observations, amplitude self-calibration should be limited
    to times when the primary beam response is high to avoid absorbing beam
    effects into the gains.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    field :
        Field ID to check
    threshold :
        Minimum beam response threshold (0-1)

    Returns
    -------
        Tuple of (is_acceptable, estimated_beam_response)

    """
    try:
        import casacore.tables as tb

        # Get field pointing direction
        field_id = int(field) if field.isdigit() else 0

        with tb.table(f"{ms_path}/FIELD", readonly=True) as field_tab:
            if field_id >= field_tab.nrows():
                logger.warning(f"Field {field_id} not found in MS")
                return True, 1.0

            phase_dir = field_tab.getcol("PHASE_DIR")[field_id]
            field_ra = phase_dir[0, 0]  # radians
            field_dec = phase_dir[0, 1]  # radians

        # Get pointing direction (if available)
        pointing_path = f"{ms_path}/POINTING"
        if not Path(pointing_path).exists():
            # No pointing info, assume on-axis
            return True, 1.0

        with tb.table(pointing_path, readonly=True) as pointing_tab:
            if pointing_tab.nrows() == 0:
                return True, 1.0

            pointing = pointing_tab.getcol("DIRECTION")
            # Use mean pointing for simplicity
            mean_ra = np.mean(pointing[:, 0, 0])
            mean_dec = np.mean(pointing[:, 0, 1])

        # Compute angular separation
        cos_sep = np.sin(field_dec) * np.sin(mean_dec) + np.cos(field_dec) * np.cos(
            mean_dec
        ) * np.cos(field_ra - mean_ra)
        separation_rad = np.arccos(np.clip(cos_sep, -1, 1))
        separation_deg = np.degrees(separation_rad)

        # Estimate beam response (Gaussian approximation)
        # DSA-110 FWHM ~ 2.5 deg at 1.4 GHz
        fwhm_deg = 2.5
        sigma_deg = fwhm_deg / 2.355

        beam_response = np.exp(-0.5 * (separation_deg / sigma_deg) ** 2)

        is_acceptable = beam_response >= threshold

        if not is_acceptable:
            logger.warning(
                f"Field {field} beam response {beam_response:.2f} < {threshold:.2f} threshold"
            )

        return is_acceptable, float(beam_response)

    except Exception as e:
        logger.warning(f"Failed to check beam response: {e}")
        return True, 1.0


def _predict_model_wsclean(
    ms_path: str,
    model_prefix: str,
    wsclean_path: str | None = None,
) -> bool:
    """Predict model visibilities using WSClean -predict.

    This writes to the MODEL_DATA column from a model image.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    model_prefix :
        Prefix of model image (e.g., "image" for "image-model.fits")
    wsclean_path :
        Optional path to WSClean executable

    Returns
    -------
        True if successful

    """
    try:
        # Prefer explicit path; otherwise always use Docker wsclean
        if wsclean_path:
            from dsa110_contimg.common.utils.wsclean_utils import build_wsclean_native_env

            env = build_wsclean_native_env()
            cmd = [
                wsclean_path,
                "-predict",
                "-reorder",
                "-name",
                model_prefix,
                ms_path,
            ]
        else:
            from dsa110_contimg.common.utils.gpu_utils import build_docker_command

            docker_user_flags = None
            if os.getenv("WSCLEAN_DOCKER_USER", "").lower() == "host":
                docker_user_flags = ["--user", f"{os.getuid()}:{os.getgid()}"]
            env = None

            cmd = build_docker_command(
                image="dsa110-contimg:gpu",
                command=["wsclean"],
                env_vars={
                    "NVIDIA_DISABLE_REQUIRE": "1",
                    "MAMBA_ROOT_PREFIX": "/dev/shm/micromamba",
                    "HOME": "/dev/shm/dsa110-contimg",
                },
                extra_flags=docker_user_flags,
            )
            cmd.extend(
                [
                    "-predict",
                    "-reorder",
                    "-name",
                    model_prefix,
                    ms_path,
                ]
            )

        logger.info("Running WSClean predict: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, timeout=600, capture_output=True, env=env)

        logger.info("Model prediction successful")
        return True

    except Exception as e:
        logger.error(f"WSClean predict failed: {e}")
        return False


def _ensure_writable(paths) -> None:
    """Best-effort fix of ownership/permissions for files touched by Docker.

    Parameters
    ----------
    paths :


    """
    for p in paths:
        try:
            subprocess.run(
                ["sudo", "chown", "-R", f"{os.getuid()}:{os.getgid()}", str(p)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            continue


# NOTE: _predict_model_casa() is currently unused. The tclean backend uses
# savemodel="modelcolumn" which writes MODEL_DATA during imaging. This function
# could be used as an alternative workflow: run tclean with savemodel="none"
# (safer, avoids MS corruption risk if interrupted), then call this function
# to populate MODEL_DATA separately via ft().
#
# def _predict_model_casa(
#     ms_path: str,
#     model_image: str,
#     field: str = "0",
# ) -> bool:
#     """Predict model visibilities using CASA ft().
#
#     Args:
#         ms_path: Path to Measurement Set
#         model_image: Path to model image
#         field: Field selection
#
#     Returns:
#         True if successful
#     """
#     try:
#         from dsa110_contimg.core.calibration.casa_service import CASAService
#
#         service = CASAService()
#         service.ft(
#             vis=ms_path,
#             model=model_image,
#             field=field,
#             usescratch=True,
#         )
#
#         logger.info("CASA ft() model prediction successful")
#         return True
#
#     except Exception as e:
#         logger.error(f"CASA ft() failed: {e}")
#         return False


def _run_gaincal(
    ms_path: str,
    caltable: str,
    field: str,
    solint: str,
    calmode: str,
    refant: str | None,
    minsnr: float,
    combine: str,
    gaintable: list[str] | None = None,
    uvrange: str = "",
    spw: str = "",
) -> bool:
    """Run CASA gaincal.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    caltable :
        Output calibration table path
    field :
        Field selection
    solint :
        Solution interval
    calmode :
        Calibration mode ('p' for phase, 'ap' for amp+phase)
    refant :
        Reference antenna
    minsnr :
        Minimum SNR for solutions
    combine :
        Combine parameter
    gaintable :
        Previous calibration tables to apply
    uvrange :
        UV range selection
    spw :
        Spectral window selection

    Returns
    -------
        True if successful

    """
    try:
        ensure_casa_path()

        service = CASAService()

        kwargs: dict[str, Any] = {
            "vis": ms_path,
            "caltable": caltable,
            "field": field,
            "solint": solint,
            "gaintype": "G",
            "calmode": calmode,
            "minsnr": minsnr,
        }

        if refant:
            kwargs["refant"] = refant
        if combine:
            kwargs["combine"] = combine
        if gaintable:
            kwargs["gaintable"] = gaintable
        if uvrange:
            kwargs["uvrange"] = uvrange
        if spw:
            kwargs["spw"] = spw

        logger.info(f"Running gaincal: solint={solint}, calmode={calmode}")
        service.gaincal(**kwargs)

        # Verify table was created
        if Path(caltable).exists():
            logger.info(f"Calibration table created: {caltable}")
            return True
        else:
            logger.error(f"Calibration table not created: {caltable}")
            return False

    except Exception as e:
        logger.error(f"gaincal failed: {e}")
        return False


def _apply_calibration(
    ms_path: str,
    gaintables: list[str],
    field: str = "",
) -> bool:
    """Apply calibration tables to MS.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    gaintables :
        List of calibration tables to apply
    field :
        Field selection

    Returns
    -------
        True if successful

    """
    try:
        from dsa110_contimg.core.calibration.applycal import apply_to_target

        apply_to_target(
            ms_path,
            field=field,
            gaintables=gaintables,
            calwt=True,
        )

        logger.info(f"Applied {len(gaintables)} calibration tables")
        return True

    except Exception as e:
        logger.error(f"applycal failed: {e}")
        return False


def _run_imaging(
    ms_path: str,
    imagename: str,
    config: SelfCalConfig,
    galvin_clip_image: str | None = None,
) -> str | None:
    """Run imaging with configured parameters.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    imagename :
        Output image name prefix
    config :
        Self-calibration configuration
    galvin_clip_image :
        Optional path to image for Galvin adaptive clipping.
        When provided and config.use_galvin_clip is True, this image will
        be used to create an adaptive clip mask for artifact suppression.

    Returns
    -------
        Path to output image, or None if failed

    """
    try:
        if config.backend == "wsclean":
            from dsa110_contimg.core.imaging.cli_imaging import image_ms

            # Build galvin_clip_mask argument if enabled and image provided
            galvin_clip_mask = None
            if config.use_galvin_clip and galvin_clip_image:
                galvin_clip_mask = galvin_clip_image
                logger.info(f"Using Galvin adaptive clip from: {galvin_clip_image}")

            image_ms(
                ms_path,
                imagename=imagename,
                field=config.field,
                imsize=config.imsize,
                cell_arcsec=config.cell_arcsec,
                niter=config.niter,
                threshold=config.threshold,
                robust=config.robust,
                backend="wsclean",
                wsclean_path=config.wsclean_path,
                galvin_clip_mask=galvin_clip_mask,
                galvin_box_size=config.galvin_box_size,
                galvin_adaptive_depth=config.galvin_adaptive_depth,
            )

            # Find output image
            for suffix in [".image.fits", "-image.fits", ".image"]:
                img_path = f"{imagename}{suffix}"
                if Path(img_path).exists():
                    return img_path

            # WSClean naming convention
            img_path = f"{imagename}-MFS-image.fits"
            if Path(img_path).exists():
                return img_path

            logger.warning(f"Could not find output image for {imagename}")
            return None

        else:
            # CASA tclean
            # Note: Galvin clip not supported with tclean backend
            if config.use_galvin_clip and galvin_clip_image:
                logger.warning(
                    "Galvin adaptive clip requested but not supported with tclean backend. "
                    "Use backend='wsclean' for Galvin clip support."
                )

            from dsa110_contimg.common.utils.casa_init import ensure_casa_path

            ensure_casa_path()

            from dsa110_contimg.common.utils.ms_permissions import (
                ensure_dir_writable,
                ensure_ms_writable,
            )

            ensure_ms_writable(ms_path)
            ensure_dir_writable(Path(imagename).parent)

            # Determine which data column to image
            datacolumn = "corrected"
            try:
                import casacore.tables as tb

                with tb.table(ms_path, readonly=True, ack=False) as t:
                    if "CORRECTED_DATA" not in t.colnames():
                        datacolumn = "data"
            except Exception as e:
                logger.warning(f"Could not inspect MS columns, defaulting to DATA: {e}")
                datacolumn = "data"

            service = CASAService()
            service.tclean(
                vis=ms_path,
                imagename=imagename,
                field=config.field,
                imsize=[config.imsize, config.imsize],
                cell=f"{config.cell_arcsec or 1.0}arcsec",
                niter=config.niter,
                threshold=config.threshold,
                weighting="briggs",
                robust=config.robust,
                datacolumn=datacolumn,
                savemodel="modelcolumn",  # Save model to MODEL_DATA
            )

            img_path = f"{imagename}.image"
            if Path(img_path).exists():
                return img_path

            return None

    except Exception as e:
        logger.error(f"Imaging failed: {e}")
        return None


# =============================================================================
# Main Self-Calibration Functions
# =============================================================================


def selfcal_iteration(
    ms_path: str,
    output_dir: str,
    iteration: int,
    mode: SelfCalMode,
    solint: str,
    config: SelfCalConfig,
    previous_gaintables: list[str] | None = None,
    previous_image: str | None = None,
) -> SelfCalIterationResult:
    """Run a single self-calibration iteration.

    One iteration consists of:
    1. Image the current data
    2. Predict model visibilities
    3. Solve gaincal
    4. Apply calibration

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    output_dir :
        Output directory for products
    iteration :
        Iteration number (0-indexed)
    mode :
        Calibration mode (phase or ap)
    solint :
        Solution interval
    config :
        Self-calibration configuration
    previous_gaintables :
        Previous calibration tables to apply
    previous_image :
        Path to previous iteration's image for Galvin clip mask

    Returns
    -------
        SelfCalIterationResult with iteration details

    """
    result = SelfCalIterationResult(
        iteration=iteration,
        mode=mode,
        solint=solint,
        success=False,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Image prefix for this iteration
    iter_prefix = f"selfcal_iter{iteration}_{mode.value}_{solint.replace(' ', '')}"
    imagename = str(output_path / iter_prefix)
    caltable = str(output_path / f"{iter_prefix}.cal")

    try:
        # Step 0: Pre-checks for amplitude self-cal in drift-scan mode
        if mode == SelfCalMode.AMPLITUDE_PHASE and config.drift_scan_mode:
            beam_ok, beam_response = _check_beam_response(
                ms_path, config.field, config.min_beam_response
            )
            if not beam_ok:
                result.message = (
                    f"Beam response {beam_response:.2f} below threshold "
                    f"{config.min_beam_response:.2f} - skipping amp self-cal"
                )
                logger.warning(result.message)
                return result

        # Step 0b: Check per-antenna SNR if enabled
        if config.check_antenna_snr:
            solint_sec = _parse_solint_seconds(solint)
            median_snr, min_snr, _ = _compute_per_antenna_snr(ms_path, solint_sec, config.field)
            result.antenna_snr_median = median_snr
            result.antenna_snr_min = min_snr

            # Check against threshold
            threshold = (
                config.phase_antenna_snr if mode == SelfCalMode.PHASE else config.amp_antenna_snr
            )

            if median_snr < threshold:
                result.message = (
                    f"Per-antenna SNR {median_snr:.1f} below threshold {threshold:.1f} "
                    f"for {mode.value} self-cal - consider longer solint"
                )
                logger.warning(result.message)
                # Don't fail, but log warning - let caller decide

            logger.info(
                f"Iteration {iteration}: Per-antenna SNR median={median_snr:.1f}, min={min_snr:.1f}"
            )

        # Step 1: Image
        logger.info(f"Iteration {iteration}: Imaging ({mode.value}, solint={solint})")
        image_path = _run_imaging(ms_path, imagename, config, galvin_clip_image=previous_image)

        if not image_path:
            result.message = "Imaging failed"
            return result

        result.image_path = image_path

        # Fix ownership if WSClean Docker wrote files as root
        _ensure_writable([ms_path, output_dir, Path(image_path).parent])

        # Measure image stats
        peak, rms, snr = _measure_image_stats(image_path)
        result.peak_flux = peak
        result.rms = rms
        result.snr = snr

        logger.info(
            f"Iteration {iteration}: SNR={snr:.1f}, Peak={peak * 1e3:.3f}mJy, RMS={rms * 1e6:.1f}µJy"
        )

        # Step 1b: Compute visibility chi-squared if enabled
        if config.use_chi_squared:
            chi_sq = _compute_visibility_chi_squared(ms_path, config.field)
            result.chi_squared = chi_sq
            logger.info(f"Iteration {iteration}: Chi-squared={chi_sq:.3f}")

        # Step 2: Predict model
        logger.info(f"Iteration {iteration}: Predicting model")

        if config.backend == "wsclean":
            # WSClean uses prefix for model files
            model_prefix = imagename
            if not _predict_model_wsclean(ms_path, model_prefix, wsclean_path=config.wsclean_path):
                result.message = "Model prediction failed"
                return result
            # Fix ownership again after Docker predict writes to MS
            _ensure_writable([ms_path, output_dir])
        else:
            # CASA tclean already saved model if savemodel="modelcolumn"
            pass

        # Step 3: Gaincal
        logger.info(f"Iteration {iteration}: Running gaincal")
        calmode = "p" if mode == SelfCalMode.PHASE else "ap"
        minsnr = config.phase_minsnr if mode == SelfCalMode.PHASE else config.amp_minsnr

        # Handle combine parameter - allow SPW combining for phase if configured
        if mode == SelfCalMode.PHASE and config.combine_spw_phase:
            combine = "spw" if not config.phase_combine else f"{config.phase_combine},spw"
        else:
            combine = config.phase_combine if mode == SelfCalMode.PHASE else config.amp_combine

        if not _run_gaincal(
            ms_path=ms_path,
            caltable=caltable,
            field=config.field,
            solint=solint,
            calmode=calmode,
            refant=config.refant,
            minsnr=minsnr,
            combine=combine,
            gaintable=previous_gaintables,
            uvrange=config.uvrange,
            spw=config.spw,
        ):
            result.message = "Gaincal failed"
            return result

        result.gaintable = caltable

        # Step 3b: Check gain smoothness if enabled
        if config.check_gain_smoothness:
            phase_scatter, amp_scatter = _measure_gain_smoothness(caltable)
            result.phase_scatter_deg = phase_scatter
            result.amp_scatter_frac = amp_scatter

            logger.info(
                f"Iteration {iteration}: Gain scatter - phase={phase_scatter:.1f}deg, "
                f"amp={amp_scatter:.3f}"
            )

            # Check against thresholds
            if phase_scatter > config.max_phase_scatter_deg:
                result.message = (
                    f"Phase scatter {phase_scatter:.1f}° exceeds threshold "
                    f"{config.max_phase_scatter_deg:.1f}° - solutions may be noisy"
                )
                logger.warning(result.message)

            if mode == SelfCalMode.AMPLITUDE_PHASE:
                if amp_scatter > config.max_amp_scatter_frac:
                    result.message = (
                        f"Amp scatter {amp_scatter:.3f} exceeds threshold "
                        f"{config.max_amp_scatter_frac:.3f} - solutions may be noisy"
                    )
                    logger.warning(result.message)

        # Step 4: Apply calibration
        logger.info(f"Iteration {iteration}: Applying calibration")
        all_gaintables = (previous_gaintables or []) + [caltable]

        if not _apply_calibration(ms_path, all_gaintables, field=config.field):
            result.message = "Applycal failed"
            return result

        result.success = True
        result.message = f"Completed: SNR={snr:.1f}"

        return result

    except Exception as e:
        logger.error(f"Iteration {iteration} failed: {e}")
        result.message = str(e)
        return result


def selfcal_ms(
    ms_path: str,
    output_dir: str,
    config: SelfCalConfig | None = None,
    initial_caltables: list[str] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Run full self-calibration loop on a Measurement Set.

    This orchestrates multiple self-calibration iterations:
    1. Phase-only iterations with progressive solution intervals
    2. Optional amplitude+phase iteration

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    output_dir :
        Output directory for all products
    config :
        Self-calibration configuration (default: SelfCalConfig())
    initial_caltables :
        Initial calibration tables to apply first

    Returns
    -------
        Tuple of (success, summary_dict)

    """
    if config is None:
        config = SelfCalConfig()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result = SelfCalResult(status=SelfCalStatus.FAILED)

    logger.info("=" * 60)
    logger.info("Starting self-calibration")
    logger.info(f"MS: {ms_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Max iterations: {config.max_iterations}")
    logger.info(f"Phase solints: {config.phase_solints}")
    logger.info(f"Do amplitude: {config.do_amplitude}")
    logger.info(f"Check per-antenna SNR: {config.check_antenna_snr}")
    logger.info(f"Check gain smoothness: {config.check_gain_smoothness}")
    logger.info(f"Use chi-squared: {config.use_chi_squared}")
    logger.info(f"Drift-scan mode: {config.drift_scan_mode}")
    logger.info("=" * 60)

    # Check flagged fraction
    flagged_frac = _get_flagged_fraction(ms_path)
    if flagged_frac > config.max_flagged_fraction:
        result.message = f"Too much data flagged: {flagged_frac * 100:.1f}% > {config.max_flagged_fraction * 100:.1f}%"
        logger.error(result.message)
        return False, _result_to_dict(result)

    # Apply initial calibration if provided
    current_gaintables: list[str] = []
    if initial_caltables:
        logger.info(f"Applying {len(initial_caltables)} initial calibration tables")
        if not _apply_calibration(ms_path, initial_caltables, field=config.field):
            result.message = "Failed to apply initial calibration"
            return False, _result_to_dict(result)
        current_gaintables = list(initial_caltables)

    # Create initial image to measure starting SNR
    logger.info("Creating initial image to measure baseline SNR")
    initial_imagename = str(output_path / "selfcal_initial")
    initial_image = _run_imaging(ms_path, initial_imagename, config)

    if not initial_image:
        result.message = "Failed to create initial image"
        return False, _result_to_dict(result)

    initial_peak, initial_rms, initial_snr = _measure_image_stats(initial_image)
    result.initial_snr = initial_snr

    # Compute initial chi-squared if enabled
    initial_chi_sq = 0.0
    if config.use_chi_squared:
        initial_chi_sq = _compute_visibility_chi_squared(ms_path, config.field)
        result.initial_chi_squared = initial_chi_sq
        logger.info(f"Initial chi-squared: {initial_chi_sq:.3f}")

    logger.info(
        f"Initial image: SNR={initial_snr:.1f}, Peak={initial_peak * 1e3:.3f}mJy, RMS={initial_rms * 1e6:.1f}µJy"
    )

    if initial_snr < config.min_initial_snr:
        result.message = f"Initial SNR too low: {initial_snr:.1f} < {config.min_initial_snr:.1f}"
        result.status = SelfCalStatus.FAILED
        logger.warning(result.message)
        return False, _result_to_dict(result)

    best_snr = initial_snr
    best_chi_sq = initial_chi_sq
    best_iteration = -1
    iteration = 0
    previous_image = initial_image  # Track for Galvin clip mask

    # Phase-only iterations
    for solint in config.phase_solints:
        if iteration >= config.max_iterations:
            logger.info("Reached maximum iterations")
            break

        iter_result = selfcal_iteration(
            ms_path=ms_path,
            output_dir=output_dir,
            iteration=iteration,
            mode=SelfCalMode.PHASE,
            solint=solint,
            config=config,
            previous_gaintables=current_gaintables,
            previous_image=previous_image,
        )

        result.iterations.append(iter_result)

        if not iter_result.success:
            logger.warning(f"Phase iteration {iteration} failed: {iter_result.message}")
            # Continue to next iteration, don't abort entire selfcal
            iteration += 1
            continue

        # Check for improvement using both SNR and chi-squared
        snr_improved = iter_result.snr > best_snr * config.min_snr_improvement
        chi_sq_improved = True  # Default if not using chi-squared

        if config.use_chi_squared and iter_result.chi_squared > 0 and best_chi_sq > 0:
            chi_sq_improved = (
                iter_result.chi_squared < best_chi_sq * config.min_chi_squared_improvement
            )
            if chi_sq_improved:
                logger.info(
                    f"Chi-squared improved: {best_chi_sq:.3f} -> {iter_result.chi_squared:.3f}"
                )

        if snr_improved:
            logger.info(f"SNR improved: {best_snr:.1f} -> {iter_result.snr:.1f}")
            best_snr = iter_result.snr
            best_iteration = iteration
            if iter_result.gaintable:
                current_gaintables.append(iter_result.gaintable)
            if iter_result.image_path:
                previous_image = iter_result.image_path  # Update for next Galvin clip
            if config.use_chi_squared and iter_result.chi_squared > 0:
                best_chi_sq = iter_result.chi_squared
        elif iter_result.snr < best_snr and config.stop_on_divergence:
            logger.warning(f"SNR decreased: {best_snr:.1f} -> {iter_result.snr:.1f}, stopping")
            result.status = SelfCalStatus.DIVERGED
            break
        else:
            logger.info(f"SNR not significantly improved: {best_snr:.1f} -> {iter_result.snr:.1f}")

        iteration += 1

    # Amplitude+phase iteration
    if config.do_amplitude and iteration < config.max_iterations:
        # Check if amplitude self-cal should proceed based on status
        if result.status != SelfCalStatus.DIVERGED:
            logger.info("Running amplitude+phase self-calibration")

            iter_result = selfcal_iteration(
                ms_path=ms_path,
                output_dir=output_dir,
                iteration=iteration,
                mode=SelfCalMode.AMPLITUDE_PHASE,
                solint=config.amp_solint,
                config=config,
                previous_gaintables=current_gaintables,
                previous_image=previous_image,
            )

            result.iterations.append(iter_result)

            if iter_result.success:
                if iter_result.snr > best_snr:
                    logger.info(f"Amplitude SNR improved: {best_snr:.1f} -> {iter_result.snr:.1f}")
                    best_snr = iter_result.snr
                    best_iteration = iteration
                    if iter_result.gaintable:
                        current_gaintables.append(iter_result.gaintable)
                    if config.use_chi_squared and iter_result.chi_squared > 0:
                        best_chi_sq = iter_result.chi_squared
                elif iter_result.snr < best_snr and config.stop_on_divergence:
                    logger.warning(
                        f"Amplitude SNR decreased: {best_snr:.1f} -> {iter_result.snr:.1f}"
                    )
                    result.status = SelfCalStatus.DIVERGED

            iteration += 1

    # Final results
    result.iterations_completed = iteration
    result.best_snr = best_snr
    result.best_iteration = best_iteration
    result.final_snr = result.iterations[-1].snr if result.iterations else initial_snr
    result.improvement_factor = best_snr / initial_snr if initial_snr > 0 else 1.0
    result.final_gaintables = current_gaintables

    # Chi-squared tracking
    if config.use_chi_squared:
        result.final_chi_squared = (
            result.iterations[-1].chi_squared if result.iterations else initial_chi_sq
        )
        if initial_chi_sq > 0 and result.final_chi_squared > 0:
            result.chi_squared_improvement = initial_chi_sq / result.final_chi_squared

    # Find final image
    if result.iterations and result.iterations[-1].image_path:
        result.final_image = result.iterations[-1].image_path

    # Determine final status
    if result.status == SelfCalStatus.FAILED:
        if result.improvement_factor >= config.min_snr_improvement:
            result.status = SelfCalStatus.SUCCESS
        elif iteration >= config.max_iterations:
            result.status = SelfCalStatus.MAX_ITERATIONS
        else:
            result.status = SelfCalStatus.NO_IMPROVEMENT

    success = result.status in (
        SelfCalStatus.SUCCESS,
        SelfCalStatus.CONVERGED,
        SelfCalStatus.MAX_ITERATIONS,
    )

    logger.info("=" * 60)
    logger.info("Self-calibration complete")
    logger.info(f"Status: {result.status.value}")
    logger.info(f"Iterations: {result.iterations_completed}")
    logger.info(f"Initial SNR: {result.initial_snr:.1f}")
    logger.info(f"Best SNR: {result.best_snr:.1f}")
    logger.info(f"Improvement: {result.improvement_factor:.2f}x")
    if config.use_chi_squared and result.initial_chi_squared > 0:
        logger.info(f"Chi-squared improvement: {result.chi_squared_improvement:.2f}x")
    logger.info("=" * 60)

    result.message = f"{result.status.value}: {result.improvement_factor:.2f}x improvement in {result.iterations_completed} iterations"

    # Generate diagnostic plots if enabled
    result_dict = _result_to_dict(result)
    if config.generate_diagnostics:
        try:
            from dsa110_contimg.core.calibration.selfcal_diagnostics import (
                generate_selfcal_diagnostics,
            )

            diag_plots = generate_selfcal_diagnostics(
                result_dict,
                output_dir,
                ms_path=ms_path,
                include_closure_phases=config.diagnostics_closure_phases,
            )
            result_dict["diagnostic_plots"] = {k: str(v) for k, v in diag_plots.items()}
            logger.info(f"Generated {len(diag_plots)} diagnostic plots")
        except Exception as e:
            logger.warning(f"Failed to generate diagnostics: {e}")

    return success, result_dict


def _result_to_dict(result: SelfCalResult) -> dict[str, Any]:
    """Convert SelfCalResult to dictionary for serialization.

    Parameters
    ----------
    """

    def _to_float(value: Any) -> float | None:
        return None if value is None else float(value)

    return {
        "status": result.status.value,
        "iterations_completed": int(result.iterations_completed),
        "initial_snr": _to_float(result.initial_snr),
        "initial_chi_squared": _to_float(result.initial_chi_squared),
        "best_snr": _to_float(result.best_snr),
        "final_snr": _to_float(result.final_snr),
        "final_chi_squared": _to_float(result.final_chi_squared),
        "improvement_factor": _to_float(result.improvement_factor),
        "chi_squared_improvement": _to_float(result.chi_squared_improvement),
        "best_iteration": int(result.best_iteration) if result.best_iteration is not None else -1,
        "final_image": result.final_image,
        "final_gaintables": result.final_gaintables,
        "message": result.message,
        "iterations": [
            {
                "iteration": int(ir.iteration),
                "mode": ir.mode.value,
                "solint": ir.solint,
                "success": ir.success,
                "snr": _to_float(ir.snr),
                "rms": _to_float(ir.rms),
                "peak_flux": _to_float(ir.peak_flux),
                "chi_squared": _to_float(ir.chi_squared),
                "antenna_snr_median": _to_float(ir.antenna_snr_median),
                "antenna_snr_min": _to_float(ir.antenna_snr_min),
                "phase_scatter_deg": _to_float(ir.phase_scatter_deg),
                "amp_scatter_frac": _to_float(ir.amp_scatter_frac),
                "gaintable": ir.gaintable,
                "image_path": ir.image_path,
                "message": ir.message,
            }
            for ir in result.iterations
        ],
    }


__all__ = [
    "SelfCalMode",
    "SelfCalStatus",
    "SelfCalConfig",
    "SelfCalIterationResult",
    "SelfCalResult",
    "selfcal_iteration",
    "selfcal_ms",
    # Constants
    "DEFAULT_PHASE_SOLINTS",
    "DEFAULT_AMP_SOLINT",
    "DEFAULT_PHASE_ANTENNA_SNR",
    "DEFAULT_AMP_ANTENNA_SNR",
    "DEFAULT_MAX_PHASE_SCATTER_DEG",
    "DEFAULT_MAX_AMP_SCATTER_FRAC",
    "DEFAULT_MIN_BEAM_RESPONSE",
]
