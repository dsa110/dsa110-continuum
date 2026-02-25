"""Noise model validation for DSA-110 images.

This module provides theoretical noise prediction and validation against measured
RMS noise in continuum images. It uses radiometer equation calculations based on
system parameters and observing configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def calculate_theoretical_rms(
    ms_path: str,
    bandwidth_hz: float = 250e6,  # DSA-110 continuum bandwidth (250 MHz)
    integration_time_s: float | None = None,  # Extracted from MS (typical: 12.88s drift-scan)
    num_antennas: int = 117,  # DSA-110 antennas (117 total)
    sefd_per_element_jy: float = 5800.0,  # Measured from T_sys = 25 K (see dsa110_measured_parameters.yaml)
    efficiency: float = 0.7,
) -> float:
    """Calculate theoretical RMS noise from radiometer equation.

    The radiometer equation for continuum imaging:
        σ = (SEFD_element) / (η * sqrt(N_pol * N_ant * (N_ant - 1) * Δν * t_int))

    Where:
        - SEFD_element = System Equivalent Flux Density per element (Jy)
        - η = system efficiency (includes correlator losses, flagging, etc.)
        - N_pol = number of polarizations (2 for DSA-110)
        - N_ant = number of antennas
        - Δν = total bandwidth (Hz)
        - t_int = integration time (s)

    **NOTE**: This formula calculates the theoretical thermal noise for perfect
    imaging. Real image RMS may be higher due to:
        - Dynamic range limitations (sidelobe noise from bright sources)
        - Calibration errors (residual gain/phase errors)
        - Deconvolution errors (CLEAN artifacts)
        - RFI residuals
    Typical degradation factor: 1.2-2x theoretical noise.

    Args:
        ms_path: Path to measurement set (used to extract integration time if not provided)
        bandwidth_hz: Total bandwidth in Hz (default: 250 MHz for continuum)
        integration_time_s: Total integration time in seconds (extracted from MS if None,
                           typical DSA-110 drift-scan: 12.88s)
        num_antennas: Number of antennas (default: 117 for DSA-110)
        sefd_per_element_jy: SEFD per element in Janskys (default: 5800 Jy,
                            measured from DSA-110 observations with T_sys = 25 K)
        efficiency: System efficiency factor (default: 0.7)

    Returns
    -------
        Theoretical RMS noise in mJy/beam

    Raises
    ------
        FileNotFoundError: If MS file doesn't exist and integration time not provided
    """
    # Get integration time from MS if not provided
    if integration_time_s is None:
        integration_time_s = _extract_integration_time(ms_path)

    # Radiometer equation for interferometer
    # σ = SEFD_element / (η * sqrt(N_pol * N_ant * (N_ant - 1) * Δν * t_int))
    # For DSA-110: N_pol = 2 (dual polarization)
    n_pol = 2

    rms_jy = sefd_per_element_jy / (
        efficiency
        * np.sqrt(n_pol * num_antennas * (num_antennas - 1) * bandwidth_hz * integration_time_s)
    )

    # Convert Jy to mJy
    rms_mjy = float(rms_jy * 1000.0)

    return rms_mjy


def _extract_integration_time(ms_path: str) -> float:
    """Extract total integration time from Measurement Set.

    Args:
        ms_path: Path to measurement set

    Returns
    -------
        Total integration time in seconds

    Raises
    ------
        FileNotFoundError: If MS doesn't exist
        RuntimeError: If MS cannot be read
    """
    ms_path_obj = Path(ms_path)
    if not ms_path_obj.exists():
        raise FileNotFoundError(f"Measurement set not found: {ms_path}")

    try:
        # Try using casatools if available
        from casatools import table as tbtool

        tb = tbtool()
        tb.open(str(ms_path_obj))

        # Get exposure time column
        exposure = tb.getcol("EXPOSURE")
        interval = tb.getcol("INTERVAL")
        tb.close()

        # Total integration time (sum of all integrations)
        # Use EXPOSURE if available, otherwise INTERVAL
        if len(exposure) > 0:
            total_time = np.sum(exposure)
        else:
            total_time = np.sum(interval)

        return float(total_time)

    except ImportError:
        # Fallback: estimate from file timestamps or default
        # For DSA-110, typical snapshot is ~10 seconds
        return 10.0
    except Exception as e:
        raise RuntimeError(f"Failed to read MS {ms_path}: {e}") from e


def validate_noise_prediction(
    image_path: str,
    ms_path: str,
    measured_rms: float,
    tolerance_sigma: float = 3.0,
    bandwidth_hz: float = 250e6,
    **radiometer_kwargs,
) -> dict[str, Any]:
    """Validate measured RMS against theoretical prediction.

    Compares measured image RMS noise to theoretical prediction from radiometer
    equation. Large deviations indicate potential calibration or imaging issues.

    Args:
        image_path: Path to FITS image (for validation, not currently used)
        ms_path: Path to measurement set
        measured_rms: Measured RMS noise in mJy/beam from image
        tolerance_sigma: Threshold for flagging deviations (default: 3.0 sigma)
        bandwidth_hz: Total bandwidth in Hz (default: 250 MHz)
        **radiometer_kwargs: Additional arguments passed to calculate_theoretical_rms

    Returns
    -------
        Dictionary with validation results:
            - predicted_rms (float): Theoretical RMS in mJy/beam
            - measured_rms (float): Measured RMS in mJy/beam (echoed)
            - deviation_sigma (float): Deviation in sigma units
            - passed (bool): True if deviation < tolerance_sigma
            - message (str): Human-readable validation message

    Example:
        >>> result = validate_noise_prediction(
        ...     image_path="/path/to/image.fits",
        ...     ms_path="/path/to/data.ms",
        ...     measured_rms=0.15
        ... )
        >>> print(result)
        {
            'predicted_rms': 0.14,
            'measured_rms': 0.15,
            'deviation_sigma': 0.5,
            'passed': True,
            'message': 'Noise within 0.5σ of prediction'
        }
    """
    # Calculate theoretical RMS
    predicted_rms = calculate_theoretical_rms(
        ms_path=ms_path,
        bandwidth_hz=bandwidth_hz,
        **radiometer_kwargs,
    )

    # Calculate deviation
    # Assume uncertainty in prediction is ~20% (conservative estimate)
    prediction_uncertainty = predicted_rms * 0.2
    deviation = abs(measured_rms - predicted_rms)
    deviation_sigma = deviation / prediction_uncertainty

    # Determine if passed
    passed = deviation_sigma < tolerance_sigma

    # Generate message
    if passed:
        message = f"Noise within {deviation_sigma:.1f}σ of prediction"
    else:
        direction = "higher" if measured_rms > predicted_rms else "lower"
        message = (
            f"Noise {direction} than predicted by {deviation_sigma:.1f}σ "
            f"(measured: {measured_rms:.3f} mJy, predicted: {predicted_rms:.3f} mJy)"
        )

    return {
        "predicted_rms": float(predicted_rms),
        "measured_rms": float(measured_rms),
        "deviation_sigma": float(deviation_sigma),
        "passed": bool(passed),
        "message": message,
    }


def validate_noise_scaling(
    image_rms_values: list[float],
    integration_times: list[float],
) -> dict[str, Any]:
    """Validate that noise scales as 1/sqrt(t_int) as expected.

    For a series of images with different integration times, verify that
    RMS noise scales according to radiometer equation.

    Args:
        image_rms_values: List of measured RMS values (mJy/beam)
        integration_times: Corresponding integration times (seconds)

    Returns
    -------
        Dictionary with scaling validation results:
            - slope (float): Fitted power law index (should be ~-0.5)
            - expected_slope (float): Expected value (-0.5)
            - deviation_sigma (float): How many sigma from expected
            - passed (bool): True if slope within 2σ of expected
            - message (str): Human-readable result

    Example:
        >>> result = validate_noise_scaling(
        ...     image_rms_values=[0.2, 0.14, 0.1],
        ...     integration_times=[10, 20, 40]
        ... )
    """
    if len(image_rms_values) < 3:
        return {
            "slope": None,
            "expected_slope": -0.5,
            "deviation_sigma": None,
            "passed": False,
            "message": "Insufficient data points for scaling validation (need >= 3)",
        }

    # Fit power law: rms = A * t^α
    # In log space: log(rms) = log(A) + α * log(t)
    log_t = np.log10(integration_times)
    log_rms = np.log10(image_rms_values)

    # Linear fit
    coeffs = np.polyfit(log_t, log_rms, 1)
    slope = float(coeffs[0])  # This is α (should be -0.5 for radiometer equation)

    # Expected slope for radiometer equation
    expected_slope = -0.5

    # Estimate uncertainty in slope (conservative)
    slope_uncertainty = 0.1

    # Calculate deviation
    deviation_sigma = float(abs(slope - expected_slope) / slope_uncertainty)

    # Check if passed (within 2σ)
    passed = bool(deviation_sigma < 2.0)

    # Generate message
    if passed:
        message = (
            f"Noise scaling matches expectation (slope: {slope:.2f}, expected: {expected_slope})"
        )
    else:
        message = (
            f"Noise scaling anomalous: slope={slope:.2f} "
            f"(expected {expected_slope}, {deviation_sigma:.1f}σ deviation)"
        )

    return {
        "slope": slope,
        "expected_slope": expected_slope,
        "deviation_sigma": deviation_sigma,
        "passed": passed,
        "message": message,
    }
