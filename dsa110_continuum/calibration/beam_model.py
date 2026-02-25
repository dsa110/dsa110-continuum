"""Unified primary beam model interface.

This module provides a consistent interface for primary beam calculations
across the pipeline. Uses EveryBeam via Docker (dsa110-contimg:gpu)
as the primary method, with graceful fallback to Airy disk model.

The approach prioritizes:
  1. Docker-based EveryBeam (when Docker available)
  2. Airy disk model (always available, matches EveryBeam generic dish)

The Airy disk model matches EveryBeam's implementation for generic dishes:
  PB(theta) = (2 * J1(x) / x)²
  where x = π * D * sin(theta) / λ
  J1 is the first-order Bessel function
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy.special import j1

logger = logging.getLogger(__name__)

# Note: Native EveryBeam Python bindings (everybeam_py) are not installed.
# We use Docker-based EveryBeam via dsa110-contimg:gpu container instead.
_EVERYBEAM_NATIVE_AVAILABLE = False


@dataclass
class BeamConfig:
    """Configuration for primary beam calculations.

    This dataclass consolidates beam model parameters, making it easier to
    pass beam configuration across the pipeline and reducing function signature
    complexity.

    Returns
    -------
    >>> config = BeamConfig(
        ...     frequency_ghz=1.4,
        ...     antenna_ra=0.1,
        ...     antenna_dec=0.2,
        ... )
        >>> response = primary_beam_response(config, src_ra=0.3, src_dec=0.4)
    """

    frequency_ghz: float
    antenna_ra: float
    antenna_dec: float
    dish_diameter_m: float = 4.7
    ms_path: str | None = None
    field_id: int = 0
    beam_mode: str = "analytic"
    use_docker: bool = True

    def __post_init__(self):
        """Validate beam configuration."""
        if self.frequency_ghz <= 0:
            raise ValueError(f"frequency_ghz must be positive, got {self.frequency_ghz}")
        if self.dish_diameter_m <= 0:
            raise ValueError(f"dish_diameter_m must be positive, got {self.dish_diameter_m}")
        if self.field_id < 0:
            raise ValueError(f"field_id must be non-negative, got {self.field_id}")
        valid_modes = {"analytic", "full", "numeric", "element", "array", "none"}
        if self.beam_mode not in valid_modes:
            raise ValueError(f"beam_mode must be one of {valid_modes}, got {self.beam_mode}")


def primary_beam_response(
    src_ra: float,
    src_dec: float,
    *,
    config: BeamConfig,
) -> float:
    """Calculate primary beam response using unified model.

        This function provides a consistent beam model across the pipeline.
        Priority order:
        1. Docker-based EveryBeam (if config.use_docker=True and Docker available)
        2. Airy disk model (fallback)

        The Docker approach is preferred as it avoids GLIBC compatibility issues
        on systems with older libc versions.

    Parameters
    ----------
    src_ra : float
        Source RA in radians.
    src_dec : float
        Source Dec in radians.
    config : BeamConfig
        BeamConfig object with antenna and frequency parameters.

    Notes
    -----
        When using Docker EveryBeam, the response is extracted from the Jones
        matrix by averaging the diagonal elements. This provides a scalar
        response consistent with the Airy model interface.

    Examples
    --------
        >>> config = BeamConfig(frequency_ghz=1.4, antenna_ra=0.1, antenna_dec=0.2)
        >>> response = primary_beam_response(src_ra=0.3, src_dec=0.4, config=config)
    """
    freq_hz = config.frequency_ghz * 1e9
    src_ra_deg = np.rad2deg(src_ra)
    src_dec_deg = np.rad2deg(src_dec)

    # Priority 1: Try Docker-based EveryBeam if MS path provided and Docker available
    if config.use_docker and config.ms_path is not None:
        try:
            from dsa110_contimg.core.calibration.beam_docker import (
                _check_docker_available,
                _check_image_available,
                evaluate_beam_docker,
            )

            if _check_docker_available() and _check_image_available():
                logger.debug(
                    f"Using Docker-based EveryBeam for beam evaluation "
                    f"at RA={src_ra_deg:.2f}°, Dec={src_dec_deg:.2f}°"
                )
                return evaluate_beam_docker(
                    config.ms_path,
                    src_ra_deg,
                    src_dec_deg,
                    freq_hz,
                    field_id=config.field_id,
                )
        except ImportError:
            logger.debug("beam_docker module not available")
        except Exception as e:
            logger.debug(f"Docker-based EveryBeam evaluation failed: {e}")

    # Priority 2: Fall back to Airy disk model (always available)
    logger.debug("Using Airy disk model for beam evaluation")
    return _airy_primary_beam_response(config, src_ra, src_dec)


def _airy_primary_beam_response(
    config: BeamConfig,
    src_ra: float,
    src_dec: float,
) -> float:
    """Calculate primary beam response using Airy disk pattern.

    This matches EveryBeam's generic dish model for circular apertures.
    Formula: PB(theta) = (2 * J1(x) / x)²
    where J1 is the first-order Bessel function of the first kind.

    Parameters
    ----------
    config :
        BeamConfig with antenna_ra, antenna_dec, frequency_ghz, dish_diameter_m
    src_ra :
        Source RA in radians
    src_dec :
        Source Dec in radians

    Returns
    -------
        Primary beam response in [0, 1]

    """
    # Offset angle approximation on the sky
    dra = (src_ra - config.antenna_ra) * np.cos(config.antenna_dec)
    ddec = src_dec - config.antenna_dec
    theta = np.sqrt(dra * dra + ddec * ddec)

    # Handle zero separation case (source at phase center)
    # Use np.where to handle both scalars and arrays
    theta_safe = np.where(np.abs(theta) < 1e-10, 1e-10, theta)

    # Airy disk: x = π * D * sin(theta) / λ
    # where λ = c / f
    c_mps = 299792458.0
    lam_m = c_mps / (config.frequency_ghz * 1e9)
    x = np.pi * config.dish_diameter_m * np.sin(theta_safe) / lam_m

    # Avoid division by zero
    x_safe = np.where(np.abs(x) < 1e-10, 1e-10, x)

    # Airy pattern: (2 * J1(x) / x)²
    # Using scipy's first-order Bessel function of the first kind
    resp = (2.0 * j1(x_safe) / x_safe) ** 2

    # Set response to 1.0 where theta was originally near zero
    resp = np.where(np.abs(theta) < 1e-10, 1.0, resp)

    # Clamp numeric noise
    return np.clip(resp, 0.0, 1.0)
