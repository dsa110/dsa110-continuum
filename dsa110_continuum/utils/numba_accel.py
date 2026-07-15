"""Numba-accelerated functions for DSA-110 pipeline.

OPTIMIZATION 3: Uses Numba JIT compilation to accelerate numerical
computations in the pipeline, particularly:
- Angular separation calculations
- UVW coordinate transformations
- Coordinate system conversions

Performance gains are most significant for large arrays and
repeated calculations (e.g., processing many baselines/times).

Usage:
    from dsa110_continuum.utils.numba_accel import (
        angular_separation_jit,
        rotate_uvw_jit,
        jd_to_mjd_jit,
    )
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Try to import numba, fall back to pure numpy if unavailable
try:
    # Explicitly disable debug flags before import
    import os

    os.environ["NUMBA_DUMP_SSA"] = "0"
    os.environ["NUMBA_DEBUG"] = "0"

    import numba.core.config

    numba.core.config.DEBUG = False
    numba.core.config.DUMP_SSA = False

    from numba import jit, prange

    NUMBA_AVAILABLE = True
    logger.debug("Numba JIT compilation available")
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not available, using pure numpy fallback")

    # Create no-op decorator for fallback
    def jit(*args, **kwargs):
        """No-op decorator when numba is not available."""

        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    def prange(*args):
        """Fallback to regular range when numba is not available."""
        return range(*args)


# =============================================================================
# Angular Separation (spherical law of cosines)
# =============================================================================


@jit(nopython=True, cache=True, fastmath=True)
def angular_separation_jit(
    ra1: np.ndarray,
    dec1: np.ndarray,
    ra2: np.ndarray,
    dec2: np.ndarray,
) -> np.ndarray:
    """Compute angular separation using the spherical law of cosines (JIT-compiled).

        This is significantly faster than astropy's angular_separation for
        large arrays due to JIT compilation and avoiding Python overhead.

        Unlike the haversine form, arccos conditioning imposes a ~2.1e-8 rad
        (4.3 mas) floor on the smallest resolvable separations — negligible
        for this pipeline's arcsecond-level cross-matching.

    Parameters
    ----------
    ra1 : array_like
        Right ascension of first point(s) in radians
    dec1 : array_like
        Declination of first point(s) in radians
    ra2 : array_like
        Right ascension of second point(s) in radians
    dec2 : array_like
        Declination of second point(s) in radians

    Returns
    -------
        array_like
        Angular separation in radians
    """
    sin_dec1 = np.sin(dec1)
    sin_dec2 = np.sin(dec2)
    cos_dec1 = np.cos(dec1)
    cos_dec2 = np.cos(dec2)
    cos_dra = np.cos(ra1 - ra2)

    cos_sep = sin_dec1 * sin_dec2 + cos_dec1 * cos_dec2 * cos_dra

    # Clamp to [-1, 1] to avoid numerical issues with arccos
    # Note: Use np.maximum/np.minimum instead of np.clip for 0-d array compatibility in numba
    cos_sep = np.maximum(np.minimum(cos_sep, 1.0), -1.0)

    return np.arccos(cos_sep)


@jit(nopython=True, cache=True, fastmath=True)
def angular_separation_scalar_jit(
    ra1: float,
    dec1: float,
    ra2: float,
    dec2: float,
) -> float:
    """Compute angular separation for scalar inputs (JIT-compiled).

    Parameters
    ----------
    ra1 : float
        Right ascension of first point in radians
    dec1 : float
        Declination of first point in radians
    ra2 : float
        Right ascension of second point in radians
    dec2 : float
        Declination of second point in radians

    Returns
    -------
        float
        Angular separation in radians
    """
    cos_sep = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    # Clip to [-1, 1]
    if cos_sep > 1.0:
        cos_sep = 1.0
    elif cos_sep < -1.0:
        cos_sep = -1.0
    return np.arccos(cos_sep)


# =============================================================================
# UVW Rotation Matrix
# =============================================================================


@jit(nopython=True, cache=True, fastmath=True)
def compute_uvw_rotation_matrix(
    ha: float,
    dec: float,
) -> np.ndarray:
    """Compute UVW rotation matrix for given hour angle and declination.

        The UVW coordinate system is defined relative to the phase center:
        - U: points East
        - V: points North
        - W: points toward the phase center

    Parameters
    ----------
    ha : float
        Hour angle in radians
    dec : float
        Declination in radians

    Returns
    -------
        ndarray
        3x3 rotation matrix
    """
    sin_ha = np.sin(ha)
    cos_ha = np.cos(ha)
    sin_dec = np.sin(dec)
    cos_dec = np.cos(dec)

    # Rotation matrix from XYZ to UVW
    R = np.array(
        [
            [sin_ha, cos_ha, 0.0],
            [-sin_dec * cos_ha, sin_dec * sin_ha, cos_dec],
            [cos_dec * cos_ha, -cos_dec * sin_ha, sin_dec],
        ]
    )
    return R


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def rotate_xyz_to_uvw_jit(
    xyz: np.ndarray,
    ha_array: np.ndarray,
    dec: float,
) -> np.ndarray:
    """Rotate XYZ baseline vectors to UVW coordinates (JIT-compiled, parallel).

        This function computes UVW coordinates for many baselines at different
        hour angles, using parallel execution for large arrays.

    Parameters
    ----------
    xyz : ndarray
        Baseline vectors in XYZ, shape (N, 3)
    ha_array : ndarray
        Hour angles in radians, shape (N,)
    dec : float
        Declination in radians (scalar, same for all)

    Returns
    -------
        ndarray
        UVW coordinates, shape (N, 3)
    """
    n = xyz.shape[0]
    uvw = np.empty((n, 3), dtype=np.float64)

    sin_dec = np.sin(dec)
    cos_dec = np.cos(dec)

    for i in prange(n):
        ha = ha_array[i]
        sin_ha = np.sin(ha)
        cos_ha = np.cos(ha)

        x, y, z = xyz[i, 0], xyz[i, 1], xyz[i, 2]

        # U = x*sin(ha) + y*cos(ha)
        uvw[i, 0] = x * sin_ha + y * cos_ha
        # V = -x*sin(dec)*cos(ha) + y*sin(dec)*sin(ha) + z*cos(dec)
        uvw[i, 1] = -x * sin_dec * cos_ha + y * sin_dec * sin_ha + z * cos_dec
        # W = x*cos(dec)*cos(ha) - y*cos(dec)*sin(ha) + z*sin(dec)
        uvw[i, 2] = x * cos_dec * cos_ha - y * cos_dec * sin_ha + z * sin_dec

    return uvw


# =============================================================================
# Time Conversions
# =============================================================================


@jit(nopython=True, cache=True, fastmath=True)
def jd_to_mjd_jit(jd: np.ndarray) -> np.ndarray:
    """Convert Julian Date to Modified Julian Date (JIT-compiled).

        MJD = JD - 2400000.5

    Parameters
    ----------
    jd : array_like
        Julian Date array

    Returns
    -------
        array_like
        Modified Julian Date array
    """
    return jd - 2400000.5


@jit(nopython=True, cache=True, fastmath=True)
def mjd_to_jd_jit(mjd: np.ndarray) -> np.ndarray:
    """Convert Modified Julian Date to Julian Date (JIT-compiled).

        JD = MJD + 2400000.5

    Parameters
    ----------
    mjd : array_like
        Modified Julian Date array

    Returns
    -------
        array_like
        Julian Date array
    """
    return mjd + 2400000.5


# =============================================================================
# LST Calculation (approximation for speed)
# =============================================================================


@jit(nopython=True, cache=True, fastmath=True)
def approx_lst_jit(
    mjd: np.ndarray,
    longitude_rad: float,
) -> np.ndarray:
    """Approximate Local Sidereal Time calculation (JIT-compiled).

        This is an approximation suitable for phase center tracking where
        sub-arcsecond precision is not required. For high-precision work,
        use astropy.

        Based on Meeus, "Astronomical Algorithms" (1991).

    Parameters
    ----------
    mjd : array_like
        Modified Julian Date array
    longitude_rad : float
        Observatory longitude in radians (East positive)

    Returns
    -------
        array_like
        Local Sidereal Time in radians
    """
    # Days since J2000.0
    D = mjd - 51544.5

    # Greenwich Mean Sidereal Time (radians)
    # GMST = 18.697374558 + 24.06570982441908 * D (in hours)
    # Convert to radians: 1 hour = pi/12 radians
    GMST_rad = (18.697374558 + 24.06570982441908 * D) * (np.pi / 12.0)

    # Local Sidereal Time
    LST = GMST_rad + longitude_rad

    # Normalize to [0, 2*pi)
    two_pi = 2.0 * np.pi
    LST = LST % two_pi

    return LST


# =============================================================================
# Batch Operations
# =============================================================================


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def compute_phase_corrections_jit(
    uvw: np.ndarray,
    freq_hz: np.ndarray,
    w_offset: np.ndarray,
) -> np.ndarray:
    """Compute phase corrections for visibility data (JIT-compiled, parallel).

        Computes exp(-2*pi*i * w_offset * freq / c) for each baseline-time-freq.

    Parameters
    ----------
    uvw : ndarray
        UVW coordinates, shape (Nblts, 3)
    freq_hz : ndarray
        Frequencies in Hz, shape (Nfreq,)
    w_offset : ndarray
        W-coordinate offset to apply, shape (Nblts,)

    Returns
    -------
        ndarray
        Complex phase corrections, shape (Nblts, Nfreq)
    """
    c_light = 299792458.0  # Speed of light in m/s
    nblts = uvw.shape[0]
    nfreq = freq_hz.shape[0]

    corrections = np.empty((nblts, nfreq), dtype=np.complex128)

    for i in prange(nblts):
        w_diff = w_offset[i]
        for j in range(nfreq):
            phase = -2.0 * np.pi * w_diff * freq_hz[j] / c_light
            corrections[i, j] = np.cos(phase) + 1j * np.sin(phase)

    return corrections


# =============================================================================
# Utility Functions
# =============================================================================


def is_numba_available() -> bool:
    """Check if numba JIT compilation is available.

    Returns
    -------
        True if numba is installed and functional
    """
    return NUMBA_AVAILABLE


def warm_up_jit() -> None:
    """Warm up JIT-compiled functions by calling them with small arrays.

    This forces compilation before the first real use, avoiding
    compilation overhead during time-critical operations.
    """
    if not NUMBA_AVAILABLE:
        return

    # Small test arrays
    ra = np.array([0.0, 1.0], dtype=np.float64)
    dec = np.array([0.5, 0.5], dtype=np.float64)
    mjd = np.array([60000.0, 60000.5], dtype=np.float64)
    xyz = np.array([[100.0, 200.0, 50.0], [150.0, 250.0, 75.0]], dtype=np.float64)
    ha = np.array([0.0, 0.1], dtype=np.float64)

    # Trigger compilation
    _ = angular_separation_jit(ra, dec, ra, dec)
    _ = angular_separation_scalar_jit(0.0, 0.5, 0.1, 0.5)
    _ = jd_to_mjd_jit(mjd + 2400000.5)
    _ = approx_lst_jit(mjd, -2.0)
    _ = rotate_xyz_to_uvw_jit(xyz, ha, 0.5)

    logger.debug("JIT functions warmed up")


__all__ = [
    "NUMBA_AVAILABLE",
    "angular_separation_jit",
    "angular_separation_scalar_jit",
    "compute_uvw_rotation_matrix",
    "rotate_xyz_to_uvw_jit",
    "jd_to_mjd_jit",
    "mjd_to_jd_jit",
    "approx_lst_jit",
    "compute_phase_corrections_jit",
    "is_numba_available",
    "warm_up_jit",
]
