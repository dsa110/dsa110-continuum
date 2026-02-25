"""Fast UVW coordinate computation using astropy instead of CASA.

This provides a 100-1000x speedup over the CASA measures tool for synthetic data generation.

Important: UVW coordinates depend on Earth rotation (time). Any implementation must
incorporate the observation time when rotating Earth-fixed (ITRF/ITRS) baseline vectors
into the sky-fixed UVW frame.
"""

import astropy.units as u
import numpy as np
from astropy.coordinates import ITRS, EarthLocation, SkyCoord
from astropy.time import Time

from dsa110_contimg.common.utils.constants import DSA110_LOCATION


def fast_uvw_from_baselines(
    baseline_vectors_itrf: np.ndarray,
    times_mjd: np.ndarray,
    phase_center_ra_deg: float,
    phase_center_dec_deg: float,
    location: EarthLocation = DSA110_LOCATION,
) -> np.ndarray:
    """Compute UVW coordinates using pure astropy (100x faster than CASA).

    Parameters
    ----------
    baseline_vectors_itrf : np.ndarray
        ITRF baseline vectors, shape (nbls, 3) in meters
    times_mjd : np.ndarray
        Observation times in MJD, shape (ntimes,)
    phase_center_ra_deg : float
        Phase center RA in degrees (J2000)
    phase_center_dec_deg : float
        Phase center Dec in degrees (J2000)
    location : EarthLocation
        Observatory location (default: DSA110_LOCATION)

    Returns
    -------
    uvw : np.ndarray
        UVW coordinates, shape (nbls * ntimes, 3) in meters
    """
    nbls = baseline_vectors_itrf.shape[0]
    ntimes = len(times_mjd)

    # Create phase center
    phase_center = SkyCoord(
        ra=phase_center_ra_deg * u.deg, dec=phase_center_dec_deg * u.deg, frame="icrs"
    )

    # Pre-allocate output
    uvw = np.zeros((nbls * ntimes, 3), dtype=np.float64)

    # For each time, rotate baselines into UVW frame
    for tidx, mjd in enumerate(times_mjd):
        obstime = Time(mjd, format="mjd", scale="utc")

        # Get rotation matrix from ITRS (Earth-fixed) to UVW at this time.
        # The ITRS basis of the UVW frame changes with time because the phase
        # center direction rotates in the Earth-fixed frame.
        rotation = _get_uvw_rotation_matrix(phase_center, obstime, location)

        # Apply rotation to all baselines at once (vectorized)
        start_idx = tidx * nbls
        end_idx = start_idx + nbls
        uvw[start_idx:end_idx] = baseline_vectors_itrf @ rotation.T

    return uvw


def _get_uvw_rotation_matrix(
    phase_center: SkyCoord,
    obstime: Time,
    location: EarthLocation,
) -> np.ndarray:
    """Compute rotation matrix from ITRS (Earth-fixed) to UVW coordinates.

    The UVW coordinate system is defined as:
    - W: Points toward phase center
    - V: Points toward NCP in plane perpendicular to W
    - U: Completes right-handed system (U = V × W)

    Returns
    -------
    rotation : np.ndarray
        3x3 rotation matrix from ITRF to UVW
    """
    # NOTE: The input baseline vectors are in an Earth-fixed frame (ITRF/ITRS).
    # To project them into UVW, we need the UVW basis vectors expressed in the
    # same Earth-fixed frame at this time.
    _ = location  # retained for API compatibility; not required for ITRS transforms

    # W axis: unit vector toward the phase center, expressed in ITRS at obstime.
    phase_itrs = phase_center.transform_to(ITRS(obstime=obstime))
    w_hat = phase_itrs.cartesian.xyz.to_value(u.one)
    w_norm = np.linalg.norm(w_hat)
    if w_norm == 0:
        raise ValueError("Invalid phase center direction (zero vector)")
    w_hat = w_hat / w_norm

    # NCP direction in ITRS at obstime.
    # Using ITRS (not a fixed [0,0,1]) is essential; Earth rotation changes the
    # NCP direction in the Earth-fixed frame.
    ncp_icrs = SkyCoord(ra=0.0 * u.deg, dec=90.0 * u.deg, frame="icrs")
    ncp_itrs = ncp_icrs.transform_to(ITRS(obstime=obstime))
    ncp_hat = ncp_itrs.cartesian.xyz.to_value(u.one)
    ncp_hat = ncp_hat / np.linalg.norm(ncp_hat)

    # V axis: component of NCP perpendicular to W.
    v_hat = ncp_hat - np.dot(ncp_hat, w_hat) * w_hat
    v_norm = np.linalg.norm(v_hat)
    if v_norm == 0:
        raise ValueError("Invalid UVW basis: NCP parallel to phase center")
    v_hat = v_hat / v_norm

    # U axis: completes right-handed system.
    u_hat = np.cross(v_hat, w_hat)

    # Rotation matrix: rows are basis vectors (u, v, w) in ITRS coordinates.
    # For an ITRS vector x, UVW = x @ rotation.T
    return np.array([u_hat, v_hat, w_hat])


def fast_build_uvw_grid(
    baseline_vectors_itrf: np.ndarray,
    times_mjd: np.ndarray,
    phase_center_ra_deg: float,
    phase_center_dec_deg: float,
) -> np.ndarray:
    """Build full UVW grid for all baseline-time combinations.

    This is optimized for synthetic data generation where we need UVW
    for the full grid of (baseline, time) pairs.

    Parameters
    ----------
    baseline_vectors_itrf : np.ndarray
        ITRF baseline vectors, shape (nbls, 3) in meters
    times_mjd : np.ndarray
        Observation times in MJD, shape (ntimes,)
    phase_center_ra_deg : float
        Phase center RA in degrees (J2000)
    phase_center_dec_deg : float
        Phase center Dec in degrees (J2000)

    Returns
    -------
    uvw : np.ndarray
        UVW coordinates, shape (nbls * ntimes, 3) in meters
    """
    return fast_uvw_from_baselines(
        baseline_vectors_itrf,
        times_mjd,
        phase_center_ra_deg,
        phase_center_dec_deg,
        DSA110_LOCATION,
    )
