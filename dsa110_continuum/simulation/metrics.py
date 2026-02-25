"""Validation metrics for synthetic data pipeline testing.

    This module provides metrics for quantifying how well the pipeline recovers
    known ground truth from synthetic data. Metrics include flux recovery accuracy,
    astrometric precision, detection completeness, and variability recovery.

    These metrics enable automated validation of pipeline performance and
    regression testing.

    Example
-------
    >>> from dsa110_contimg.core.simulation.metrics import (
    ...     flux_recovery_error,
    ...     astrometric_offset,
    ...     detection_completeness,
    ... )
    >>>
    >>> # Flux recovery
    >>> injected_flux = 2.5  # Jy
    >>> measured_flux = 2.4  # Jy
    >>> error_pct = flux_recovery_error(measured_flux, injected_flux)
    >>> print(f"Flux error: {error_pct:.1f}%")  # -4.0%
    >>>
    >>> # Astrometry
    >>> offset_arcsec = astrometric_offset(
    ...     measured_ra=188.001, measured_dec=42.001,
    ...     true_ra=188.000, true_dec=42.000
    ... )
    >>> print(f"Position error: {offset_arcsec:.2f} arcsec")
    >>>
    >>> # Detection rate
    >>> completeness = detection_completeness(
    ...     n_detected=48, n_injected=50
    ... )
    >>> print(f"Detection completeness: {completeness:.1%}")  # 96.0%
"""

from __future__ import annotations

import logging
import math

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from dsa110_contimg.common.unified_config import settings
from dsa110_contimg.common.utils.gpu_utils import get_array_module

logger = logging.getLogger(__name__)


def flux_recovery_error(
    measured_flux_jy: float,
    injected_flux_jy: float,
) -> float:
    """Compute flux recovery error as percentage.

    Parameters
    ----------
    measured_flux_jy : float
        Measured flux from pipeline (Jy)
    injected_flux_jy : float
        Injected ground truth flux (Jy)

    Returns
    -------
    float
        Percentage error calculated as (measured - injected) / injected * 100.
        Positive values indicate overestimation, negative values indicate underestimation.

    Examples
    --------
    >>> flux_recovery_error(2.4, 2.5)  # 4% underestimation
    -4.0
    >>> flux_recovery_error(2.6, 2.5)  # 4% overestimation
    4.0
    """
    if injected_flux_jy == 0:
        logger.warning("Injected flux is zero, cannot compute recovery error")
        return np.nan

    return ((measured_flux_jy - injected_flux_jy) / injected_flux_jy) * 100.0


def flux_recovery_ratio(
    measured_flux_jy: float,
    injected_flux_jy: float,
) -> float:
    """Compute flux recovery ratio.

    Parameters
    ----------
    measured_flux_jy : float
        Measured flux from pipeline (Jy)
    injected_flux_jy : float
        Injected ground truth flux (Jy)

    Returns
    -------
        float
        Ratio of measured to injected flux.
        Values near 1.0 indicate perfect recovery.

    Examples
    --------
        >>> flux_recovery_ratio(2.4, 2.5)
        0.96
    """
    if injected_flux_jy == 0:
        return np.nan
    return measured_flux_jy / injected_flux_jy


def astrometric_offset(
    measured_ra_deg: float,
    measured_dec_deg: float,
    true_ra_deg: float,
    true_dec_deg: float,
) -> float:
    """Compute astrometric offset in arcseconds.

    Uses small-angle approximation for speed. For large separations
    (>1 degree), consider using astropy.coordinates.

    Parameters
    ----------
    measured_ra_deg : float
        Measured RA (degrees)
    measured_dec_deg : float
        Measured Dec (degrees)
    true_ra_deg : float
        True RA (degrees)
    true_dec_deg : float
        True Dec (degrees)

    Returns
    -------
    float
        Separation in arcseconds.

    Examples
    --------
    >>> astrometric_offset(188.001, 42.000, 188.000, 42.000)
    2.7...  # ~2.7 arcsec offset in RA
    """
    # Small-angle approximation
    c1 = SkyCoord(ra=measured_ra_deg * u.deg, dec=measured_dec_deg * u.deg)
    c2 = SkyCoord(ra=true_ra_deg * u.deg, dec=true_dec_deg * u.deg)
    return c1.separation(c2).arcsec


def detection_completeness(
    n_detected: int,
    n_injected: int,
) -> float:
    """Compute detection completeness (fraction of sources detected).

    Parameters
    ----------
    n_detected : int
        Number of injected sources that were detected
    n_injected : int
        Total number of injected sources

    Returns
    -------
        float
        Fraction between 0 and 1 representing completeness.

    Examples
    --------
        >>> detection_completeness(48, 50)
        0.96
    """
    if n_injected == 0:
        return np.nan
    return n_detected / n_injected


def false_positive_rate(
    n_false_detections: int,
    n_total_detections: int,
) -> float:
    """Compute false positive rate.

    Parameters
    ----------
    n_false_detections : int
        Number of detections with no injected source
    n_total_detections : int
        Total number of detections

    Returns
    -------
        float
        Fraction between 0 and 1 representing false positive rate.

    Examples
    --------
        >>> false_positive_rate(2, 50)
        0.04
    """
    if n_total_detections == 0:
        return 0.0
    return n_false_detections / n_total_detections


def variability_detection_score(
    detected_variable: bool,
    is_actually_variable: bool,
) -> int:
    """Score variability detection (1=correct, 0=incorrect).

    Parameters
    ----------
    detected_variable : bool
        Pipeline flagged source as variable
    is_actually_variable : bool
        Source has variability model in ground truth

    Returns
    -------
        int
        1 if detection is correct, 0 otherwise.

    Examples
    --------
        >>> variability_detection_score(True, True)  # True positive
        1
        >>> variability_detection_score(False, True)  # Missed variable
        0
    """
    return 1 if detected_variable == is_actually_variable else 0


def rms_flux_error(
    measured_fluxes: np.ndarray,
    injected_fluxes: np.ndarray,
) -> float:
    """Compute RMS flux error across multiple measurements.

    Parameters
    ----------
    measured_fluxes : np.ndarray
        Array of measured fluxes (Jy)
    injected_fluxes : np.ndarray
        Array of injected fluxes (Jy)

    Returns
    -------
        float
        RMS error in Jy.

    Examples
    --------
        >>> measured = np.array([2.4, 2.5, 2.6])
        >>> injected = np.array([2.5, 2.5, 2.5])
        >>> rms_flux_error(measured, injected)
        0.081...
    """
    if len(measured_fluxes) != len(injected_fluxes):
        raise ValueError("Arrays must have same length")

    residuals = measured_fluxes - injected_fluxes
    return float(np.sqrt(np.mean(residuals**2)))


def normalized_rms_error(
    measured_fluxes: np.ndarray,
    injected_fluxes: np.ndarray,
) -> float:
    """Compute normalized RMS error (as fraction of mean flux).

    Parameters
    ----------
    measured_fluxes : np.ndarray
        Array of measured fluxes (Jy)
    injected_fluxes : np.ndarray
        Array of injected fluxes (Jy)

    Returns
    -------
        float
        Normalized RMS error (unitless).

    Examples
    --------
        >>> measured = np.array([2.4, 2.5, 2.6])
        >>> injected = np.array([2.5, 2.5, 2.5])
        >>> normalized_rms_error(measured, injected)
        0.032...  # ~3.2% RMS error
    """
    rms = rms_flux_error(measured_fluxes, injected_fluxes)
    mean_flux = np.mean(injected_fluxes)

    if mean_flux == 0:
        return np.nan

    return rms / mean_flux


def mean_absolute_percentage_error(
    measured_fluxes: np.ndarray,
    injected_fluxes: np.ndarray,
) -> float:
    """Compute mean absolute percentage error (MAPE).

    Parameters
    ----------
    measured_fluxes : np.ndarray
        Array of measured fluxes (Jy)
    injected_fluxes : np.ndarray
        Array of injected fluxes (Jy)

    Returns
    -------
        float
        MAPE as percentage (0-100).

    Examples
    --------
        >>> measured = np.array([2.4, 2.6])
        >>> injected = np.array([2.5, 2.5])
        >>> mean_absolute_percentage_error(measured, injected)
        4.0
    """
    if len(measured_fluxes) != len(injected_fluxes):
        raise ValueError("Arrays must have same length")

    # Avoid division by zero
    mask = injected_fluxes != 0
    if not np.any(mask):
        return np.nan

    percentage_errors = (
        np.abs((measured_fluxes[mask] - injected_fluxes[mask]) / injected_fluxes[mask]) * 100.0
    )

    return float(np.mean(percentage_errors))


def compute_variability_metrics(
    measured_fluxes: np.ndarray,
    measured_errors: np.ndarray,
    *,
    prefer_gpu: bool | None = None,
    min_elements: int | None = None,
) -> tuple[float, float, float]:
    """Compute variability metrics from flux measurements.

        Returns the same metrics used by the production pipeline:
        - η (eta): Weighted variance metric
        - V: Coefficient of variation (std/mean)
        - χ²/ν: Reduced chi-squared

    Parameters
    ----------
    measured_fluxes : np.ndarray
        Array of flux measurements (Jy)
    measured_errors : np.ndarray
        Array of flux uncertainties (Jy)
    prefer_gpu : bool or None, optional
        Whether to prefer GPU computation (default is None)
    min_elements : int or None, optional
        Minimum number of elements required (default is None)

    Returns
    -------
        tuple
        Tuple of (eta, V, chi2_nu).

    Examples
    --------
        >>> fluxes = np.array([2.5, 2.7, 2.3])
        >>> errors = np.array([0.1, 0.1, 0.1])
        >>> eta, V, chi2_nu = compute_variability_metrics(fluxes, errors)
    """
    prefer_gpu = settings.gpu.prefer_gpu if prefer_gpu is None else prefer_gpu
    min_elements = settings.gpu.min_array_size if min_elements is None else min_elements

    xp, is_gpu = get_array_module(prefer_gpu=prefer_gpu, min_elements=min_elements)
    if is_gpu and measured_fluxes.size >= min_elements and measured_errors.size >= min_elements:
        fluxes = xp.asarray(measured_fluxes)
        errors = xp.asarray(measured_errors)
    else:
        fluxes = measured_fluxes
        errors = measured_errors
        xp = np
        is_gpu = False

    n = fluxes.size
    if n < 2:
        return np.nan, np.nan, np.nan

    # Check for zero or negative errors (invalid)
    if xp.any(errors <= 0):
        # Mean and std still computable
        mean_flux = xp.mean(fluxes)
        std_flux = xp.std(fluxes, ddof=1)
        V = std_flux / mean_flux if mean_flux != 0 else xp.nan
        # Weighted metrics require valid errors
        eta = xp.nan
        chi2_nu = xp.nan
        to_host = xp.asnumpy if hasattr(xp, "asnumpy") else np.asarray
        return float(to_host(eta)), float(to_host(V)), float(to_host(chi2_nu))

    # Mean and std
    mean_flux = xp.mean(fluxes)
    std_flux = xp.std(fluxes, ddof=1)

    # Coefficient of variation
    V = std_flux / mean_flux if mean_flux != 0 else xp.nan

    # Weighted variance (eta metric)
    weights = 1.0 / (errors**2)
    weighted_mean = xp.sum(weights * fluxes) / xp.sum(weights)
    weighted_variance = xp.sum(weights * (fluxes - weighted_mean) ** 2) / xp.sum(weights)
    expected_variance = n / xp.sum(weights)  # Expected from noise
    eta = (weighted_variance - expected_variance) / (n - 1) if n > 1 else xp.nan

    # Reduced chi-squared
    chi2 = xp.sum(((fluxes - mean_flux) / errors) ** 2)
    chi2_nu = chi2 / (n - 1) if n > 1 else xp.nan

    to_host = xp.asnumpy if hasattr(xp, "asnumpy") else np.asarray
    return float(to_host(eta)), float(to_host(V)), float(to_host(chi2_nu))


def match_sources_by_position(
    measured_positions: list[tuple[float, float]],
    true_positions: list[tuple[float, float]],
    match_radius_arcsec: float = 5.0,
) -> list[tuple[int, int, float]]:
    """Match measured sources to ground truth by position.

    Parameters
    ----------
    measured_positions : list of tuple of float
        List of (ra_deg, dec_deg) for detections
    true_positions : list of tuple of float
        List of (ra_deg, dec_deg) for injected sources
    match_radius_arcsec : float, optional
        Maximum match radius in arcseconds (default is 5.0)

    Returns
    -------
        list of tuple
        List of (measured_idx, true_idx, separation_arcsec) for matches.

    Examples
    --------
        >>> measured = [(188.001, 42.000), (189.0, 42.0)]
        >>> true = [(188.000, 42.000)]
        >>> matches = match_sources_by_position(measured, true, match_radius_arcsec=10.0)
        >>> len(matches)  # One match
        1
    """
    matches = []
    used_measured = set()
    used_true = set()

    # Simple greedy matching (for small catalogs)
    # For large catalogs, use KD-tree
    for i, (meas_ra, meas_dec) in enumerate(measured_positions):
        best_match = None
        best_separation = match_radius_arcsec

        for j, (true_ra, true_dec) in enumerate(true_positions):
            if j in used_true:
                continue

            sep = astrometric_offset(meas_ra, meas_dec, true_ra, true_dec)
            if sep < best_separation:
                best_separation = sep
                best_match = j

        if best_match is not None:
            matches.append((i, best_match, best_separation))
            used_measured.add(i)
            used_true.add(best_match)

    return matches
