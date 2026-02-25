"""Enhanced visibility models for realistic synthetic data generation.

This module provides functions to generate visibilities with:
- Extended source models (Gaussian, disk)
- Thermal noise
- Calibration errors

Parameters can be loaded from dsa110_measured_parameters.yaml for rigor.
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import yaml

from dsa110_contimg.common.utils.paths import get_repo_root
from dsa110_contimg.common.utils.stability import Stability, stability

logger = logging.getLogger(__name__)

# Cache for loaded parameters
_PARAMETER_CACHE = None


def load_measured_parameters(config_path: Path | None = None) -> dict:
    """Load DSA-110 measured parameters from YAML config file.

    Parameters
    ----------
    config_path : Path, optional
        Path to dsa110_measured_parameters.yaml
        If None, uses default location in simulation/config/
    config_path: Optional[Path] :
         (Default value = None)

    """
    global _PARAMETER_CACHE

    if _PARAMETER_CACHE is not None:
        return _PARAMETER_CACHE

    if config_path is None:
        # Default location: simulations/config/
        # visibility_models.py is in backend/src/dsa110_contimg/simulation/
        # Go up to repo root, then to simulations/config/
        repo_root = get_repo_root(Path(__file__))
        config_path = repo_root / "simulations" / "config" / "dsa110_measured_parameters.yaml"

    if not config_path.exists():
        warnings.warn(
            f"Parameter file not found: {config_path}. Using hardcoded defaults. "
            "Run scripts/characterize_dsa110_system.py to generate measured parameters.",
            UserWarning,
        )
        _PARAMETER_CACHE = {}
        return _PARAMETER_CACHE

    try:
        from dsa110_contimg.common.utils.yaml_loader import load_yaml_with_env

        params = load_yaml_with_env(config_path, expand_vars=True)
        _PARAMETER_CACHE = params
        logger.info("Loaded measured parameters from %s", config_path)
        return params
    except Exception as e:
        warnings.warn(
            f"Failed to load parameter file {config_path}: {e}. Using hardcoded defaults.",
            UserWarning,
        )
        _PARAMETER_CACHE = {}
        return _PARAMETER_CACHE


def get_parameter(
    param_category: str,
    param_name: str,
    default_value: float,
    warn_if_assumed: bool = True,
) -> float:
    """Get parameter value with validation status checking.

    Parameters
    ----------
    param_category : str
        Category in YAML (e.g., 'system_parameters', 'calibration_errors')
    param_name : str
        Parameter name (e.g., 'system_temperature', 'gain_amplitude_std')
    default_value : float
        Fallback value if parameter not measured
    warn_if_assumed : bool, optional
        Emit warning if using assumed/default value
    """
    params = load_measured_parameters()

    if not params or param_category not in params:
        if warn_if_assumed:
            warnings.warn(
                f"Using default value for {param_name}: {default_value}. "
                "Parameter not measured. Run characterize_dsa110_system.py for rigorous values.",
                UserWarning,
            )
        return default_value

    param_data = params[param_category].get(param_name)
    if param_data is None:
        if warn_if_assumed:
            warnings.warn(
                f"Using default value for {param_name}: {default_value}. "
                "Parameter not in config file.",
                UserWarning,
            )
        return default_value

    value = param_data.get("value")
    validation_status = param_data.get("validation_status", "unknown")

    if validation_status == "assumed" and warn_if_assumed:
        warnings.warn(
            f"Parameter {param_name} is ASSUMED (not measured): {value}. "
            "Consider running characterize_dsa110_system.py for measured values.",
            UserWarning,
        )
    elif validation_status in ["measured", "validated"]:
        logger.debug("Using measured parameter %s: %s", param_name, value)

    return value


@stability(
    Stability.EXPERIMENTAL,
    reason="Uses ideal Gaussian noise; lacks RFI and frequency-dependent T_sys.",
)
def calculate_thermal_noise_rms(
    integration_time_sec: float,
    channel_width_hz: float,
    system_temperature_k: float | None = None,
    efficiency: float | None = None,
    frequency_hz: float = 1.4e9,
    use_measured_params: bool = True,
) -> float:
    """Calculate RMS thermal noise for a single visibility.

    Uses the radiometer equation:
    sigma = T_sys / (eta * sqrt(2 * delta_nu * delta_t))

    Parameters
    ----------
    integration_time_sec :
        Integration time in seconds
    channel_width_hz :
        Channel width in Hz
    system_temperature_k :
        System temperature in Kelvin
        If None and use_measured_params=True, loads from config
        Otherwise defaults to 50K
    efficiency :
        System efficiency
        If None and use_measured_params=True, loads from config
        Otherwise defaults to 0.7
    frequency_hz :
        Observing frequency in Hz (default: 1.4 GHz for DSA-110)
    use_measured_params :
        If True, attempt to load from dsa110_measured_parameters.yaml

    Returns
    -------
        RMS noise in Jy

    """
    # Load parameters from config if requested
    if use_measured_params:
        params = load_measured_parameters()
        if system_temperature_k is None:
            system_temperature_k = (
                params.get("thermal_noise", {}).get("system_temperature", {}).get("value_k", 50.0)
            )
            if (
                params.get("thermal_noise", {})
                .get("system_temperature", {})
                .get("validation_status")
                == "assumed"
            ):
                logger.warning("Using assumed T_sys (not measured from real data)")
        if efficiency is None:
            efficiency = (
                params.get("telescope_efficiency", {})
                .get("aperture_efficiency", {})
                .get("value", 0.7)
            )
            if (
                params.get("telescope_efficiency", {})
                .get("aperture_efficiency", {})
                .get("validation_status")
                == "assumed"
            ):
                logger.warning("Using assumed efficiency (not measured from real data)")
    else:
        # Use provided values or hardcoded defaults
        if system_temperature_k is None:
            system_temperature_k = 50.0
        if efficiency is None:
            efficiency = 0.7

    # Convert system temperature to Jy
    # The conversion factor is frequency-dependent: S = 2*k*T / A_eff
    # For DSA-110 interferometer, the conversion scales approximately as (freq/1.4GHz)²
    # At 1.4 GHz: ~2.0 Jy/K (from config or default)
    # General: conversion_factor ≈ 2.0 * (1.4e9 / frequency_hz)²
    reference_freq_hz = 1.4e9

    if use_measured_params:
        params = load_measured_parameters()
        base_conversion = (
            params.get("thermal_noise", {}).get("conversion_factor", {}).get("value_jy_per_k", 2.0)
        )
        if (
            params.get("thermal_noise", {}).get("conversion_factor", {}).get("validation_status")
            == "assumed"
        ):
            logger.warning("Using assumed Jy/K conversion (not measured from real data)")
    else:
        base_conversion = 2.0

    conversion_factor = base_conversion * (reference_freq_hz / frequency_hz) ** 2
    t_sys_jy = system_temperature_k * conversion_factor

    # Radiometer equation
    # sigma = T_sys / (eta * sqrt(2 * delta_nu * delta_t))
    delta_nu = channel_width_hz
    delta_t = integration_time_sec

    rms_jy = t_sys_jy / (efficiency * np.sqrt(2.0 * delta_nu * delta_t))

    return rms_jy


def add_thermal_noise(
    visibilities: np.ndarray,
    integration_time_sec: float,
    channel_width_hz: float,
    system_temperature_k: float = 50.0,
    efficiency: float = 0.7,
    frequency_hz: float = 1.4e9,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add realistic thermal noise to visibilities.

    Parameters
    ----------
    visibilities :
        Complex visibility array (shape: Nblts, Nspws, Nfreqs, Npols)
    integration_time_sec :
        Integration time per visibility
    channel_width_hz :
        Channel width in Hz
    system_temperature_k :
        System temperature in Kelvin
    efficiency :
        System efficiency
    frequency_hz :
        Observing frequency in Hz (default: 1.4 GHz for DSA-110)
    rng :
        Random number generator (for reproducibility)
    visibilities: np.ndarray :

    Returns
    -------
        Visibilities with thermal noise added

    """
    if rng is None:
        rng = np.random.default_rng()

    # Calculate RMS noise per visibility
    rms_jy = calculate_thermal_noise_rms(
        integration_time_sec,
        channel_width_hz,
        system_temperature_k,
        efficiency,
        frequency_hz,
    )

    # Generate complex Gaussian noise
    # Real and imaginary parts are independent, each with sigma = rms/sqrt(2)
    noise_real = rng.normal(0.0, rms_jy / np.sqrt(2.0), visibilities.shape)
    noise_imag = rng.normal(0.0, rms_jy / np.sqrt(2.0), visibilities.shape)
    noise = (noise_real + 1j * noise_imag).astype(visibilities.dtype)

    return visibilities + noise




def add_calibration_errors(
    visibilities: np.ndarray,
    nants: int,
    gain_std: float | None = None,
    phase_std_deg: float | None = None,
    bandpass_std: float | None = None,
    use_measured_params: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add realistic calibration errors to visibilities.

    Applies antenna-based gain and phase errors, and frequency-dependent
    bandpass variations.

    Parameters
    ----------
    visibilities :
        Complex visibility array (shape: Nblts, Nspws, Nfreqs, Npols)
    nants :
        Number of antennas
    gain_std :
        Standard deviation of gain errors (fractional)
        If None and use_measured_params=True, loads from config
        Otherwise defaults to 0.1 (10%)
    phase_std_deg :
        Standard deviation of phase errors in degrees
        If None and use_measured_params=True, loads from config
        Otherwise defaults to 10 deg
    bandpass_std :
        Standard deviation of bandpass variations (fractional)
        If None and use_measured_params=True, loads from config
        Otherwise defaults to 0.05 (5%)
    use_measured_params :
        If True, attempt to load from dsa110_measured_parameters.yaml
    rng :
        Random number generator (for reproducibility)
    visibilities: np.ndarray :

    Returns
    -------
        Tuple of (corrected_visibilities, antenna_gains, antenna_phases)
        where gains and phases are arrays of shape (nants, Nfreqs, Npols)

    """
    if rng is None:
        rng = np.random.default_rng()

    # Load parameters from config if requested
    if use_measured_params:
        params = load_measured_parameters()
        if gain_std is None:
            gain_std = (
                params.get("calibration_errors", {})
                .get("antenna_gains", {})
                .get("rms_fractional", 0.1)
            )
            if (
                params.get("calibration_errors", {})
                .get("antenna_gains", {})
                .get("validation_status")
                == "assumed"
            ):
                logger.warning("Using assumed gain_std (not measured from real data)")
        if phase_std_deg is None:
            phase_std_deg = (
                params.get("calibration_errors", {})
                .get("antenna_phases", {})
                .get("rms_degrees", 10.0)
            )
            if (
                params.get("calibration_errors", {})
                .get("antenna_phases", {})
                .get("validation_status")
                == "assumed"
            ):
                logger.warning("Using assumed phase_std (not measured from real data)")
        if bandpass_std is None:
            bandpass_std = (
                params.get("calibration_errors", {})
                .get("bandpass_stability", {})
                .get("rms_fractional", 0.05)
            )
            if (
                params.get("calibration_errors", {})
                .get("bandpass_stability", {})
                .get("validation_status")
                == "assumed"
            ):
                logger.warning("Using assumed bandpass_std (not measured from real data)")
    else:
        # Use provided values or hardcoded defaults
        if gain_std is None:
            gain_std = 0.1
        if phase_std_deg is None:
            phase_std_deg = 10.0
        if bandpass_std is None:
            bandpass_std = 0.05

    nblts, nspws, nfreqs, npols = visibilities.shape

    # Generate antenna-based gains and phases
    # Shape: (nants, nfreqs, npols)
    gain_errors = rng.normal(1.0, gain_std, (nants, nfreqs, npols))
    phase_errors_deg = rng.normal(0.0, phase_std_deg, (nants, nfreqs, npols))
    phase_errors_rad = np.deg2rad(phase_errors_deg)

    # Add frequency-dependent bandpass variations
    bandpass_errors = rng.normal(1.0, bandpass_std, (nants, nfreqs, npols))

    # Combine: total gain = gain_error * bandpass_error
    total_gains = gain_errors * bandpass_errors
    total_phases = phase_errors_rad

    # Convert to complex gains: g = |g| * exp(i * phi)
    complex_gains = total_gains * np.exp(1j * total_phases)

    # Apply to visibilities
    # For baseline (i, j): V_corr = V_true * g_i * conj(g_j)
    # We need to extract antenna indices from baseline-time array
    # This is simplified - in practice we'd need ant_1_array and ant_2_array

    # For now, return gains/phases and let caller apply them
    # This is because we need the antenna arrays to properly apply

    return visibilities, complex_gains, total_phases


def load_real_calibration_solutions(
    caltable_path: Path,
    time_avg: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load real calibration solutions from a CASA caltable.

    This provides the most rigorous approach to simulating calibration errors:
    instead of generating random gains/phases, use actual measured solutions
    from a real observation.

    Parameters
    ----------
    caltable_path : Path
        Path to CASA calibration table (e.g., 'observation_gcal')
    time_avg : bool, optional
        If True, average solutions over time to get single gain per antenna
        If False, keep time-dependent solutions (default: True)

    Returns
    -------
    complex_gains : np.ndarray
        Complex gains (shape: nants, nfreqs, npols) if time_avg=True
        or (nants, ntimes, nfreqs, npols) if time_avg=False
    antenna_indices : np.ndarray
        Antenna indices corresponding to gain solutions

    Raises
    ------
    ImportError
        If casacore is not available
    FileNotFoundError
        If caltable doesn't exist

    Examples
    --------

    Notes
    -----
    This approach is superior to synthetic calibration errors because:
    - Captures real antenna-to-antenna gain variations
    - Includes actual frequency-dependent effects
    - Preserves correlations between antennas
    - No assumptions about error distributions needed
    >>> # Load gains from a real observation
    >>> gains, ants = load_real_calibration_solutions('obs_gcal')
    >>> # Use these gains in simulation
    >>> vis_corrupted = apply_calibration_errors_to_visibilities(
    ...     visibilities, ant_1_array, ant_2_array, gains
    ... )
    """
    try:
        from casacore.tables import table
    except ImportError:
        raise ImportError(
            "casacore is required to read caltables. Install with: pip install python-casacore"
        )

    if not Path(caltable_path).exists():
        raise FileNotFoundError(f"Caltable not found: {caltable_path}")

    logger.info("Loading calibration solutions from %s", caltable_path)

    # Open caltable
    with table(str(caltable_path)) as tb:
        # Read data
        cparam = tb.getcol("CPARAM")  # Complex gains (npol, nchan, nrow)
        antenna = tb.getcol("ANTENNA1")  # Antenna indices
        flag = tb.getcol("FLAG")  # Flags (npol, nchan, nrow)

        # Get unique antennas
        unique_ants = np.unique(antenna)
        nants = len(unique_ants)

        # Get dimensions
        npol, nchan, nrow = cparam.shape

        if time_avg:
            # Average over time for each antenna
            gains = np.zeros((nants, nchan, npol), dtype=complex)

            for i, ant in enumerate(unique_ants):
                ant_mask = antenna == ant
                ant_data = cparam[:, :, ant_mask]  # (npol, nchan, ntimes)
                ant_flags = flag[:, :, ant_mask]  # (npol, nchan, ntimes)

                # Mask flagged data
                ant_data_ma = np.ma.masked_where(ant_flags, ant_data)

                # Average over time: (npol, nchan, ntimes) -> (npol, nchan)
                gains_time_avg = np.ma.mean(ant_data_ma, axis=2)

                # Transpose to (nchan, npol)
                gains[i, :, :] = gains_time_avg.T

            logger.info(
                "Loaded time-averaged gains for %d antennas, %d channels, %d polarizations",
                nants,
                nchan,
                npol,
            )
        else:
            # Keep time dimension
            # This is more complex - would need to track time indices
            # For now, raise not implemented
            raise NotImplementedError(
                "Time-dependent gain loading not yet implemented. Use time_avg=True for now."
            )

    return gains, unique_ants


def apply_calibration_errors_to_visibilities(
    visibilities: np.ndarray,
    ant_1_array: np.ndarray,
    ant_2_array: np.ndarray,
    complex_gains: np.ndarray,
) -> np.ndarray:
    """Apply calibration errors to visibilities given antenna arrays.

    Parameters
    ----------
    visibilities : np.ndarray
        Complex visibility array (shape: Nblts, Nspws, Nfreqs, Npols)
    ant_1_array : np.ndarray
        Antenna 1 indices (shape: Nblts)
    ant_2_array : np.ndarray
        Antenna 2 indices (shape: Nblts)
    complex_gains : np.ndarray
        Complex gains (shape: nants, Nfreqs, Npols)

    Returns
    -------
        np.ndarray
        Visibilities with calibration errors applied

    Notes
    -----
        The complex_gains can be either:
        - Synthetically generated using add_calibration_errors()
        - Real gains loaded from caltables using load_real_calibration_solutions()

        The latter approach is recommended for rigorous simulations as it uses
        actual measured calibration solutions from real observations.
    """
    nblts, nspws, nfreqs, npols = visibilities.shape

    # Apply gains: V_corr = V_true * g_i * conj(g_j)
    corrected = visibilities.copy()

    for blt_idx in range(nblts):
        ant1 = int(ant_1_array[blt_idx])
        ant2 = int(ant_2_array[blt_idx])

        # Get gains for this baseline
        g1 = complex_gains[ant1, :, :]  # (Nfreqs, Npols)
        g2 = complex_gains[ant2, :, :]  # (Nfreqs, Npols)

        # Apply: V_corr = V * g1 * conj(g2)
        # For each spectral window
        for spw in range(nspws):
            # g1 and g2 are (Nfreqs, Npols), need to broadcast with (Nfreqs, Npols) visibility
            corrected[blt_idx, spw, :, :] = visibilities[blt_idx, spw, :, :] * g1 * np.conj(g2)

    return corrected
