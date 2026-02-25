"""
Condon (1997) error calculations for radio source measurements.

Implements flux and positional error formulas from Condon (1997):
"Errors in Elliptical Gaussian Fits"
DOI: 10.1086/133871

These provide more accurate uncertainties than simple thermal noise estimates
by accounting for source morphology, SNR, and beam characteristics.

Reference: askap-vast/vast-pipeline image/utils.py::calc_condon_flux_errors()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# Default Condon (1997) exponent values
# These control how uncertainties scale with source size relative to beam
DEFAULT_ALPHA_MAJ1 = 1.5  # For flux and size errors
DEFAULT_ALPHA_MIN1 = 1.5
DEFAULT_ALPHA_MAJ2 = 2.5  # For position errors
DEFAULT_ALPHA_MIN2 = 0.5


@dataclass
class CondonFluxErrors:
    """Flux and size errors calculated using Condon (1997) methodology.

    Attributes
    ----------
    flux_peak_err : float
        Peak flux error (Jy/beam)
    flux_int_err : float
        Integrated flux error (Jy)
    major_err_arcsec : float
        Major axis error (arcsec)
    minor_err_arcsec : float
        Minor axis error (arcsec)
    pa_err_deg : float
        Position angle error (degrees)
    """

    flux_peak_err: float
    flux_int_err: float
    major_err_arcsec: float
    minor_err_arcsec: float
    pa_err_deg: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "flux_peak_err": self.flux_peak_err,
            "flux_int_err": self.flux_int_err,
            "major_err_arcsec": self.major_err_arcsec,
            "minor_err_arcsec": self.minor_err_arcsec,
            "pa_err_deg": self.pa_err_deg,
        }


@dataclass
class CondonPositionErrors:
    """Positional errors calculated using Condon (1997) methodology.

    Attributes
    ----------
    ra_err_arcsec : float
        RA error (arcsec)
    dec_err_arcsec : float
        Dec error (arcsec)
    error_radius_arcsec : float
        Combined error radius (arcsec)
    """

    ra_err_arcsec: float
    dec_err_arcsec: float
    error_radius_arcsec: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "ra_err_arcsec": self.ra_err_arcsec,
            "dec_err_arcsec": self.dec_err_arcsec,
            "error_radius_arcsec": self.error_radius_arcsec,
        }


@dataclass
class CondonErrors:
    """Complete Condon (1997) errors for a source.

    Attributes
    ----------
    flux : CondonFluxErrors
        Flux and size errors
    position : CondonPositionErrors
        Positional errors
    systematic_ra_arcsec : float
        Systematic RA error added (arcsec)
    systematic_dec_arcsec : float
        Systematic Dec error added (arcsec)
    """

    flux: CondonFluxErrors
    position: CondonPositionErrors
    systematic_ra_arcsec: float = 0.0
    systematic_dec_arcsec: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "flux": self.flux.to_dict(),
            "position": self.position.to_dict(),
            "systematic_ra_arcsec": self.systematic_ra_arcsec,
            "systematic_dec_arcsec": self.systematic_dec_arcsec,
        }


def calc_condon_flux_errors(
    flux_peak: float,
    flux_int: float,
    snr: float,
    local_rms: float,
    major_arcsec: float,
    minor_arcsec: float,
    pa_deg: float,
    beam_major_arcsec: float,
    beam_minor_arcsec: float,
    *,
    alpha_maj1: float = DEFAULT_ALPHA_MAJ1,
    alpha_min1: float = DEFAULT_ALPHA_MIN1,
    frac_flux_cal_error: float = 0.0,
) -> CondonFluxErrors:
    """Calculate flux and size errors using Condon (1997) equations.

        Adapted from VAST Pipeline: image/utils.py::calc_condon_flux_errors()

        The Condon (1997) formulas provide more accurate flux uncertainties
        by accounting for:
        - Source size relative to beam (resolved vs unresolved)
        - SNR of the detection
        - Correlation between fitted parameters

    Parameters
    ----------
    flux_peak : float
        Peak flux (Jy/beam)
    flux_int : float
        Integrated flux (Jy)
    snr : float
        Signal-to-noise ratio
    local_rms : float
        Local RMS noise (Jy/beam)
    major_arcsec : float
        Fitted major axis (arcsec)
    minor_arcsec : float
        Fitted minor axis (arcsec)
    pa_deg : float
        Position angle (degrees)
    beam_major_arcsec : float
        Beam major axis (arcsec)
    beam_minor_arcsec : float
        Beam minor axis (arcsec)
    alpha_maj1 : float, optional
        Exponent for major axis scaling (default 1.5)
    alpha_min1 : float, optional
        Exponent for minor axis scaling (default 1.5)
    frac_flux_cal_error : float, optional
        Fractional flux calibration error (default 0)

    Returns
    -------
        CondonFluxErrors
        All flux and size uncertainties

    Raises
    ------
        ValueError
        If any inputs are invalid (non-positive where required)
    """
    # Validate inputs
    if snr <= 0:
        raise ValueError("SNR must be positive")
    if local_rms <= 0:
        raise ValueError("Local RMS must be positive")
    if beam_major_arcsec <= 0 or beam_minor_arcsec <= 0:
        raise ValueError("Beam dimensions must be positive")

    # Convert to consistent units (arcsec -> degrees internally)
    major = major_arcsec / 3600.0
    minor = minor_arcsec / 3600.0
    theta_B = beam_major_arcsec / 3600.0
    theta_b = beam_minor_arcsec / 3600.0

    # Handle degenerate cases
    if major <= 0 or minor <= 0:
        # Point source: use beam as proxy
        major = theta_B
        minor = theta_b

    # Calculate rho_sq term (Eq. 21 in Condon 1997)
    # This is the SNR-weighted beam-to-source area ratio
    rho_sq = (
        (major * minor / (4.0 * theta_B * theta_b))
        * (1.0 + (theta_B / major) ** 2) ** alpha_maj1
        * (1.0 + (theta_b / minor) ** 2) ** alpha_min1
        * snr**2
    )

    # Ensure rho_sq is positive and finite
    if not (np.isfinite(rho_sq) and rho_sq > 0):
        # Fallback to simple noise-based errors
        return CondonFluxErrors(
            flux_peak_err=local_rms,
            flux_int_err=local_rms
            * np.sqrt(major_arcsec * minor_arcsec / (beam_major_arcsec * beam_minor_arcsec)),
            major_err_arcsec=beam_major_arcsec / snr,
            minor_err_arcsec=beam_minor_arcsec / snr,
            pa_err_deg=90.0 / snr,  # Rough estimate
        )

    rho = np.sqrt(rho_sq)

    # Peak flux error (Eq. 22)
    flux_peak_err = abs(flux_peak) / rho

    # Add flux calibration error in quadrature
    if frac_flux_cal_error > 0:
        flux_cal_err = abs(flux_peak) * frac_flux_cal_error
        flux_peak_err = np.sqrt(flux_peak_err**2 + flux_cal_err**2)

    # Integrated flux error
    # For a 2D Gaussian: S_int = S_peak * (a * b) / (beam_a * beam_b)
    # Error propagation gives larger errors for resolved sources
    beam_ratio = (major * minor) / (theta_B * theta_b)
    flux_int_err = flux_peak_err * np.sqrt(beam_ratio) if beam_ratio >= 1 else flux_peak_err

    # Major axis error (Eq. 23-25)
    major_err = (2.0 * major) / rho

    # Minor axis error
    minor_err = (2.0 * minor) / rho

    # Position angle error (Eq. 26)
    # PA error depends on axis ratio; circular sources have undefined PA
    axis_ratio = major / minor if minor > 0 else 1.0
    if axis_ratio > 1.1:  # Only meaningful for elliptical sources
        pa_err = np.degrees(2.0 / (rho * (axis_ratio**2 - 1)))
    else:
        pa_err = 180.0  # Effectively undefined

    return CondonFluxErrors(
        flux_peak_err=float(flux_peak_err),
        flux_int_err=float(flux_int_err),
        major_err_arcsec=float(major_err * 3600),
        minor_err_arcsec=float(minor_err * 3600),
        pa_err_deg=float(min(pa_err, 180.0)),
    )


def calc_condon_position_errors(
    snr: float,
    major_arcsec: float,
    minor_arcsec: float,
    pa_deg: float,
    beam_major_arcsec: float,
    beam_minor_arcsec: float,
    *,
    alpha_maj2: float = DEFAULT_ALPHA_MAJ2,
    alpha_min2: float = DEFAULT_ALPHA_MIN2,
    systematic_ra_arcsec: float = 0.0,
    systematic_dec_arcsec: float = 0.0,
) -> CondonPositionErrors:
    """Calculate positional errors using Condon (1997) equations.

        Adapted from VAST Pipeline: image/utils.py::calc_condon_flux_errors()

        Positional errors scale as ~beam / SNR for point sources, but more
        slowly for resolved sources due to the alpha exponents.

    Parameters
    ----------
    snr : float
        Signal-to-noise ratio
    major_arcsec : float
        Fitted major axis (arcsec)
    minor_arcsec : float
        Fitted minor axis (arcsec)
    pa_deg : float
        Position angle (degrees, E of N)
    beam_major_arcsec : float
        Beam major axis (arcsec)
    beam_minor_arcsec : float
        Beam minor axis (arcsec)
    alpha_maj2 : float, optional
        Exponent for major axis scaling (default 2.5)
    alpha_min2 : float, optional
        Exponent for minor axis scaling (default 0.5)
    systematic_ra_arcsec : float
        Systematic RA error to add in quadrature
    systematic_dec_arcsec : float
        Systematic Dec error to add in quadrature

    Returns
    -------
        CondonPositionErrors
        RA/Dec uncertainties
    """
    if snr <= 0:
        raise ValueError("SNR must be positive")
    if beam_major_arcsec <= 0 or beam_minor_arcsec <= 0:
        raise ValueError("Beam dimensions must be positive")

    # Convert to degrees
    major = major_arcsec / 3600.0
    minor = minor_arcsec / 3600.0
    theta_B = beam_major_arcsec / 3600.0
    theta_b = beam_minor_arcsec / 3600.0
    theta_rad = np.radians(pa_deg)

    # Handle point sources
    if major <= 0 or minor <= 0:
        major = theta_B
        minor = theta_b

    # Calculate rho_sq for positional errors (uses alpha_maj2, alpha_min2)
    rho_sq_pos = (
        (major * minor / (4.0 * theta_B * theta_b))
        * (1.0 + (theta_B / major) ** 2) ** alpha_maj2
        * (1.0 + (theta_b / minor) ** 2) ** alpha_min2
        * snr**2
    )

    if not (np.isfinite(rho_sq_pos) and rho_sq_pos > 0):
        # Fallback to simple estimate
        simple_err = beam_major_arcsec / (2.0 * snr)
        return CondonPositionErrors(
            ra_err_arcsec=float(np.sqrt(simple_err**2 + systematic_ra_arcsec**2)),
            dec_err_arcsec=float(np.sqrt(simple_err**2 + systematic_dec_arcsec**2)),
            error_radius_arcsec=float(np.sqrt(2) * simple_err),
        )

    rho_pos = np.sqrt(rho_sq_pos)

    # Position errors in major/minor axis directions (Eq. 27)
    err_major = major / rho_pos
    err_minor = minor / rho_pos

    # Project onto RA/Dec axes
    # Note: RA error needs cos(dec) correction when applied to sky positions
    # Here we give the error in arcsec (on-sky distance)
    cos_pa = np.cos(theta_rad)
    sin_pa = np.sin(theta_rad)

    # RA error (predominantly along minor axis for PA ~ 0)
    ra_err_arcsec = np.sqrt((err_major * 3600 * sin_pa) ** 2 + (err_minor * 3600 * cos_pa) ** 2)

    # Dec error (predominantly along major axis for PA ~ 0)
    dec_err_arcsec = np.sqrt((err_major * 3600 * cos_pa) ** 2 + (err_minor * 3600 * sin_pa) ** 2)

    # Add systematic errors in quadrature
    ra_err_total = np.sqrt(ra_err_arcsec**2 + systematic_ra_arcsec**2)
    dec_err_total = np.sqrt(dec_err_arcsec**2 + systematic_dec_arcsec**2)

    # Combined error radius (for matching purposes)
    error_radius = np.sqrt(ra_err_total**2 + dec_err_total**2)

    return CondonPositionErrors(
        ra_err_arcsec=float(ra_err_total),
        dec_err_arcsec=float(dec_err_total),
        error_radius_arcsec=float(error_radius),
    )


def calc_condon_errors(
    flux_peak: float,
    flux_int: float,
    snr: float,
    local_rms: float,
    major_arcsec: float,
    minor_arcsec: float,
    pa_deg: float,
    beam_major_arcsec: float,
    beam_minor_arcsec: float,
    *,
    systematic_ra_arcsec: float = 0.0,
    systematic_dec_arcsec: float = 0.0,
    frac_flux_cal_error: float = 0.0,
) -> CondonErrors:
    """Calculate complete Condon (1997) errors for a source.

        This is the main entry point combining flux and positional errors.

    Parameters
    ----------
    flux_peak : float
        Peak flux (Jy/beam)
    flux_int : float
        Integrated flux (Jy)
    snr : float
        Signal-to-noise ratio
    local_rms : float
        Local RMS noise (Jy/beam)
    major_arcsec : float
        Fitted major axis (arcsec)
    minor_arcsec : float
        Fitted minor axis (arcsec)
    pa_deg : float
        Position angle (degrees)
    beam_major_arcsec : float
        Beam major axis (arcsec)
    beam_minor_arcsec : float
        Beam minor axis (arcsec)
    systematic_ra_arcsec : float
        Systematic RA error (arcsec)
    systematic_dec_arcsec : float
        Systematic Dec error (arcsec)
    frac_flux_cal_error : float
        Fractional flux calibration error

    Returns
    -------
        CondonErrors
        Complete uncertainty information
    """
    flux_errors = calc_condon_flux_errors(
        flux_peak=flux_peak,
        flux_int=flux_int,
        snr=snr,
        local_rms=local_rms,
        major_arcsec=major_arcsec,
        minor_arcsec=minor_arcsec,
        pa_deg=pa_deg,
        beam_major_arcsec=beam_major_arcsec,
        beam_minor_arcsec=beam_minor_arcsec,
        frac_flux_cal_error=frac_flux_cal_error,
    )

    position_errors = calc_condon_position_errors(
        snr=snr,
        major_arcsec=major_arcsec,
        minor_arcsec=minor_arcsec,
        pa_deg=pa_deg,
        beam_major_arcsec=beam_major_arcsec,
        beam_minor_arcsec=beam_minor_arcsec,
        systematic_ra_arcsec=systematic_ra_arcsec,
        systematic_dec_arcsec=systematic_dec_arcsec,
    )

    return CondonErrors(
        flux=flux_errors,
        position=position_errors,
        systematic_ra_arcsec=systematic_ra_arcsec,
        systematic_dec_arcsec=systematic_dec_arcsec,
    )


def simple_position_error(
    beam_arcsec: float,
    snr: float,
    systematic_arcsec: float = 0.0,
) -> float:
    """Simple positional error estimate: beam / (2 * SNR).

        This is a commonly used approximation when detailed source
        parameters are not available.

    Parameters
    ----------
    beam_arcsec : float
        Beam size (arcsec) - typically geometric mean of axes
    snr : float
        Signal-to-noise ratio
    systematic_arcsec : float, optional
        Systematic error to add in quadrature

    Returns
    -------
        float
        Position error in arcsec
    """
    if snr <= 0:
        raise ValueError("SNR must be positive")

    fit_error = beam_arcsec / (2.0 * snr)
    return float(np.sqrt(fit_error**2 + systematic_arcsec**2))


def estimate_systematic_errors(
    catalog_name: str,
) -> tuple[float, float]:
    """Get typical systematic positional errors for common surveys.

        These are empirically determined systematic errors that should
        be added in quadrature to fit-based uncertainties.

    Parameters
    ----------
    catalog_name : str
        Name of survey/catalog (case-insensitive)

    Returns
    -------
        tuple of float
        Tuple of (ra_err_arcsec, dec_err_arcsec)
    """
    # Typical systematic errors from VAST and other radio surveys
    systematics = {
        "nvss": (1.0, 1.0),  # NVSS: ~1 arcsec systematic
        "first": (0.5, 0.5),  # FIRST: ~0.5 arcsec systematic
        "vlass": (0.5, 0.5),  # VLASS: ~0.5 arcsec systematic
        "sumss": (1.5, 1.5),  # SUMSS: ~1.5 arcsec systematic
        "racs": (0.5, 0.5),  # RACS: ~0.5 arcsec systematic
        "askap": (0.5, 0.5),  # Generic ASKAP: ~0.5 arcsec
        "dsa-110": (1.0, 1.0),  # DSA-110: conservative ~1 arcsec
        "default": (1.0, 1.0),  # Default: ~1 arcsec
    }

    key = catalog_name.lower().replace("-", "").replace("_", "")

    for name, errors in systematics.items():
        if name.replace("-", "").replace("_", "") in key:
            return errors

    return systematics["default"]
