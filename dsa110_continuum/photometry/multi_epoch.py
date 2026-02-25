"""
Multi-epoch source statistics and aggregations.

Provides statistics computed across multiple observations of the same source,
adopted from VAST Pipeline methodology for transient/variable source tracking.

Reference: askap-vast/vast-pipeline pipeline/utils.py, models.py Source
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WeightedPositionStats:
    """Weighted average position statistics.

    Attributes
    ----------
    wavg_ra : float
        Weighted average RA (degrees).
    wavg_dec : float
        Weighted average Dec (degrees).
    wavg_uncertainty_ew : float
        Weighted uncertainty in RA/EW direction (degrees).
    wavg_uncertainty_ns : float
        Weighted uncertainty in Dec/NS direction (degrees).
    n_measurements : int
        Number of measurements used.
    """

    wavg_ra: float
    wavg_dec: float
    wavg_uncertainty_ew: float
    wavg_uncertainty_ns: float
    n_measurements: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "wavg_ra_deg": self.wavg_ra,
            "wavg_dec_deg": self.wavg_dec,
            "wavg_uncertainty_ew_arcsec": self.wavg_uncertainty_ew * 3600,
            "wavg_uncertainty_ns_arcsec": self.wavg_uncertainty_ns * 3600,
            "n_measurements": self.n_measurements,
        }


@dataclass
class FluxAggregateStats:
    """Aggregate flux statistics across epochs.

    Attributes
    ----------
    avg_flux_int : float
        Average integrated flux (Jy)
    avg_flux_peak : float
        Average peak flux (Jy/beam)
    min_flux_int : float
        Minimum integrated flux
    max_flux_int : float
        Maximum integrated flux
    min_flux_peak : float
        Minimum peak flux
    max_flux_peak : float
        Maximum peak flux
    std_flux_int : float
        Standard deviation of integrated flux
    std_flux_peak : float
        Standard deviation of peak flux
    n_detections : int
        Number of detections
    n_forced : int
        Number of forced photometry measurements
    """

    avg_flux_int: float
    avg_flux_peak: float
    min_flux_int: float
    max_flux_int: float
    min_flux_peak: float
    max_flux_peak: float
    std_flux_int: float
    std_flux_peak: float
    n_detections: int
    n_forced: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "avg_flux_int": self.avg_flux_int,
            "avg_flux_peak": self.avg_flux_peak,
            "min_flux_int": self.min_flux_int,
            "max_flux_int": self.max_flux_int,
            "min_flux_peak": self.min_flux_peak,
            "max_flux_peak": self.max_flux_peak,
            "std_flux_int": self.std_flux_int,
            "std_flux_peak": self.std_flux_peak,
            "n_detections": self.n_detections,
            "n_forced": self.n_forced,
        }


@dataclass
class SNRAggregateStats:
    """Aggregate SNR statistics across epochs.

    Attributes
    ----------
    min_snr : float
        Minimum SNR
    max_snr : float
        Maximum SNR
    avg_snr : float
        Average SNR
    median_snr : float
        Median SNR
    """

    min_snr: float
    max_snr: float
    avg_snr: float
    median_snr: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_snr": self.min_snr,
            "max_snr": self.max_snr,
            "avg_snr": self.avg_snr,
            "median_snr": self.median_snr,
        }


@dataclass
class NewSourceMetrics:
    """Metrics for evaluating new source significance.

        Adopted from VAST Pipeline for assessing whether a newly detected
        source is genuinely new (transient) vs an artifact or barely-detected.

    Attributes
    ----------
    new_high_sigma : float
        Highest significance (sigma) the source would have had if placed in previous images where it was not detected
    is_new : bool
        Whether this is a genuinely new source
    first_detection_mjd : float
        MJD of first detection
    first_detection_snr : float
        SNR of first detection
    n_images_missed : int
        Number of images where source was below threshold
    """

    new_high_sigma: float
    is_new: bool
    first_detection_mjd: float | None = None
    first_detection_snr: float | None = None
    n_images_missed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "new_high_sigma": self.new_high_sigma,
            "is_new": self.is_new,
            "first_detection_mjd": self.first_detection_mjd,
            "first_detection_snr": self.first_detection_snr,
            "n_images_missed": self.n_images_missed,
        }


@dataclass
class MultiEpochSourceStats:
    """Complete multi-epoch statistics for a source.

        Combines all aggregate metrics into a single container.

    Attributes
    ----------
    source_id : any
        Source identifier
    position : any
        Weighted average position
    flux : FluxAggregateStats
        Flux aggregate statistics
    snr : SNRAggregateStats
        SNR aggregate statistics
    new_source : NewSourceMetrics
        New source validation metrics
    n_measurements : int
        Total number of measurements
    n_selavy : int
        Number from source finder (Selavy/Aegean)
    n_forced : int
        Number from forced photometry
    n_relations : int
        Number of related sources
    n_siblings : int
        Number of siblings (multi-component)
    """

    source_id: str
    position: WeightedPositionStats
    flux: FluxAggregateStats
    snr: SNRAggregateStats
    new_source: NewSourceMetrics | None = None
    n_measurements: int = 0
    n_selavy: int = 0
    n_forced: int = 0
    n_relations: int = 0
    n_siblings: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "position": self.position.to_dict(),
            "flux": self.flux.to_dict(),
            "snr": self.snr.to_dict(),
            "new_source": self.new_source.to_dict() if self.new_source else None,
            "n_measurements": self.n_measurements,
            "n_selavy": self.n_selavy,
            "n_forced": self.n_forced,
            "n_relations": self.n_relations,
            "n_siblings": self.n_siblings,
        }


def calc_weighted_average_position(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    ra_err_deg: np.ndarray,
    dec_err_deg: np.ndarray,
) -> WeightedPositionStats:
    """Calculate weighted average position.

        Adopted from VAST Pipeline: Source.wavg_ra, wavg_dec

        Weights are inverse-variance: w = 1 / sigma^2

    Parameters
    ----------
    ra_deg : array_like
        Array of RA values (degrees)
    dec_deg : array_like
        Array of Dec values (degrees)
    ra_err_deg : array_like
        Array of RA uncertainties (degrees)
    dec_err_deg : array_like
        Array of Dec uncertainties (degrees)

    Returns
    -------
        WeightedPositionStats
    """
    # Filter valid measurements
    valid = (
        np.isfinite(ra_deg)
        & np.isfinite(dec_deg)
        & np.isfinite(ra_err_deg)
        & np.isfinite(dec_err_deg)
        & (ra_err_deg > 0)
        & (dec_err_deg > 0)
    )

    if not np.any(valid):
        raise ValueError("No valid measurements for weighted average")

    ra = ra_deg[valid]
    dec = dec_deg[valid]
    ra_err = ra_err_deg[valid]
    dec_err = dec_err_deg[valid]

    # Weights = 1 / sigma^2
    w_ra = 1.0 / (ra_err**2)
    w_dec = 1.0 / (dec_err**2)

    # Weighted averages
    wavg_ra = np.sum(w_ra * ra) / np.sum(w_ra)
    wavg_dec = np.sum(w_dec * dec) / np.sum(w_dec)

    # Weighted uncertainties
    wavg_ra_err = 1.0 / np.sqrt(np.sum(w_ra))
    wavg_dec_err = 1.0 / np.sqrt(np.sum(w_dec))

    return WeightedPositionStats(
        wavg_ra=float(wavg_ra),
        wavg_dec=float(wavg_dec),
        wavg_uncertainty_ew=float(wavg_ra_err),
        wavg_uncertainty_ns=float(wavg_dec_err),
        n_measurements=int(np.sum(valid)),
    )


def calc_flux_aggregates(
    flux_int: np.ndarray,
    flux_peak: np.ndarray,
    is_forced: np.ndarray | None = None,
) -> FluxAggregateStats:
    """Calculate flux aggregate statistics.

        Adopted from VAST Pipeline: Source flux statistics

    Parameters
    ----------
    flux_int : array_like
        Array of integrated fluxes (Jy)
    flux_peak : array_like
        Array of peak fluxes (Jy/beam)
    is_forced : array_like, optional
        Boolean array indicating forced photometry measurements

    Returns
    -------
        FluxAggregateStats
    """
    # Filter valid measurements
    valid_int = np.isfinite(flux_int)
    valid_peak = np.isfinite(flux_peak)
    valid = valid_int & valid_peak

    if not np.any(valid):
        raise ValueError("No valid flux measurements")

    f_int = flux_int[valid]
    f_peak = flux_peak[valid]

    # Count forced vs detection
    n_total = len(f_int)
    if is_forced is not None:
        forced = is_forced[valid]
        n_forced = int(np.sum(forced))
        n_detections = n_total - n_forced
    else:
        n_forced = 0
        n_detections = n_total

    return FluxAggregateStats(
        avg_flux_int=float(np.mean(f_int)),
        avg_flux_peak=float(np.mean(f_peak)),
        min_flux_int=float(np.min(f_int)),
        max_flux_int=float(np.max(f_int)),
        min_flux_peak=float(np.min(f_peak)),
        max_flux_peak=float(np.max(f_peak)),
        std_flux_int=float(np.std(f_int)) if len(f_int) > 1 else 0.0,
        std_flux_peak=float(np.std(f_peak)) if len(f_peak) > 1 else 0.0,
        n_detections=n_detections,
        n_forced=n_forced,
    )


def calc_snr_aggregates(snr: np.ndarray) -> SNRAggregateStats:
    """Calculate SNR aggregate statistics.

    Parameters
    ----------
    snr : array_like
        Array of SNR values

    Returns
    -------
        SNRAggregateStats
    """
    valid = np.isfinite(snr) & (snr > 0)

    if not np.any(valid):
        raise ValueError("No valid SNR measurements")

    s = snr[valid]

    return SNRAggregateStats(
        min_snr=float(np.min(s)),
        max_snr=float(np.max(s)),
        avg_snr=float(np.mean(s)),
        median_snr=float(np.median(s)),
    )


def calc_new_source_significance(
    source_flux_peak: float,
    previous_rms_values: np.ndarray,
    previous_mjd_values: np.ndarray | None = None,
    detection_threshold_sigma: float = 5.0,
) -> NewSourceMetrics:
    """Calculate new source significance metrics.

        Adopted from VAST Pipeline: Source.new_high_sigma

        For a newly detected source, calculate how significant it would have
        been if placed in previous images (where it was not detected).

    Parameters
    ----------
    source_flux_peak : float
        Peak flux of the source (Jy/beam)
    previous_rms_values : array_like
        RMS values of images where source was not detected
    previous_mjd_values : array_like, optional
        MJD timestamps of previous images
    detection_threshold_sigma : float, optional
        Detection threshold (default 5σ)

    Returns
    -------
        NewSourceMetrics
    """
    if len(previous_rms_values) == 0:
        # No previous images - this is genuinely the first observation
        return NewSourceMetrics(
            new_high_sigma=float("inf"),
            is_new=True,
            n_images_missed=0,
        )

    # Filter valid RMS values
    valid = np.isfinite(previous_rms_values) & (previous_rms_values > 0)

    if not np.any(valid):
        return NewSourceMetrics(
            new_high_sigma=float("inf"),
            is_new=True,
            n_images_missed=0,
        )

    rms = previous_rms_values[valid]

    # Calculate sigma in each previous image
    sigma_values = source_flux_peak / rms

    # New high sigma is the maximum significance in previous images
    new_high_sigma = float(np.max(sigma_values))

    # Count images where source would have been below threshold
    n_missed = int(np.sum(sigma_values < detection_threshold_sigma))

    # Source is "new" if it was below threshold in all previous images
    is_new = new_high_sigma < detection_threshold_sigma

    return NewSourceMetrics(
        new_high_sigma=new_high_sigma,
        is_new=is_new,
        n_images_missed=n_missed,
    )


def compute_multi_epoch_stats(
    source_id: str,
    measurements: list[dict[str, Any]],
    previous_image_rms: np.ndarray | None = None,
) -> MultiEpochSourceStats:
    """Compute complete multi-epoch statistics for a source.

        This is the main entry point for multi-epoch analysis.

    Parameters
    ----------
    source_id : any
        Source identifier
    measurements : list of dict
        List of measurement dicts with keys:
        - ra_deg, dec_deg: Position
        - ra_err_deg, dec_err_deg: Position uncertainties (optional)
        - flux_int, flux_peak: Fluxes
        - local_rms or snr: Either SNR or RMS for SNR calculation
        - is_forced: Boolean for forced photometry (optional)
        - mjd: Observation timestamp (optional)
    previous_image_rms : array_like, optional
        RMS of images where source was not detected (for new source analysis)

    Returns
    -------
        MultiEpochSourceStats
    """
    if not measurements:
        raise ValueError("No measurements provided")

    # Extract arrays
    n = len(measurements)
    ra = np.array([m["ra_deg"] for m in measurements])
    dec = np.array([m["dec_deg"] for m in measurements])

    # Position errors (default to 1 arcsec if not provided)
    default_err = 1.0 / 3600  # 1 arcsec in degrees
    ra_err = np.array([m.get("ra_err_deg", default_err) for m in measurements])
    dec_err = np.array([m.get("dec_err_deg", default_err) for m in measurements])

    # Fluxes
    flux_int = np.array([m.get("flux_int", m.get("flux_peak", 0)) for m in measurements])
    flux_peak = np.array([m.get("flux_peak", m.get("flux_int", 0)) for m in measurements])

    # SNR - either provided or calculated from flux/rms
    snr = []
    for m in measurements:
        if "snr" in m:
            snr.append(m["snr"])
        elif "local_rms" in m and m["local_rms"] > 0:
            snr.append(m["flux_peak"] / m["local_rms"])
        else:
            snr.append(np.nan)
    snr = np.array(snr)

    # Forced photometry flag
    is_forced = np.array([m.get("is_forced", False) for m in measurements])

    # Calculate position stats
    position = calc_weighted_average_position(ra, dec, ra_err, dec_err)

    # Calculate flux stats
    flux = calc_flux_aggregates(flux_int, flux_peak, is_forced)

    # Calculate SNR stats
    snr_stats = calc_snr_aggregates(snr)

    # New source analysis
    new_source = None
    if previous_image_rms is not None and len(previous_image_rms) > 0:
        # Use the brightest measurement for new source analysis
        brightest_flux = float(np.max(flux_peak[np.isfinite(flux_peak)]))
        new_source = calc_new_source_significance(
            source_flux_peak=brightest_flux,
            previous_rms_values=previous_image_rms,
        )

    # Counts
    n_forced = int(np.sum(is_forced))
    n_selavy = n - n_forced

    return MultiEpochSourceStats(
        source_id=source_id,
        position=position,
        flux=flux,
        snr=snr_stats,
        new_source=new_source,
        n_measurements=n,
        n_selavy=n_selavy,
        n_forced=n_forced,
    )


def calc_two_epoch_pair_metrics(
    measurements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Calculate two-epoch metrics for all measurement pairs.

        Adopted from VAST Pipeline: pipeline/pairs.py

        For each unique pair of measurements, calculate Vs and m metrics.
        Returns the most significant pair metrics.

    Parameters
    ----------
    measurements : list of dict
        List of measurement dicts with flux_int, flux_peak,
        flux_int_err, flux_peak_err

    Returns
    -------
        list of dict
        List of pair metric dicts with vs_int, vs_peak, m_int, m_peak
    """
    from dsa110_contimg.core.photometry.variability import calculate_m_metric, calculate_vs_metric

    if len(measurements) < 2:
        return []

    pairs = []
    n = len(measurements)

    for i in range(n):
        for j in range(i + 1, n):
            m1 = measurements[i]
            m2 = measurements[j]

            pair = {
                "idx_a": i,
                "idx_b": j,
            }

            # Integrated flux metrics
            try:
                pair["vs_int"] = calculate_vs_metric(
                    m1["flux_int"],
                    m2["flux_int"],
                    m1.get("flux_int_err", m1.get("flux_err", 0.001)),
                    m2.get("flux_int_err", m2.get("flux_err", 0.001)),
                )
                pair["m_int"] = calculate_m_metric(m1["flux_int"], m2["flux_int"])
            except (ValueError, ZeroDivisionError):
                pair["vs_int"] = np.nan
                pair["m_int"] = np.nan

            # Peak flux metrics
            try:
                pair["vs_peak"] = calculate_vs_metric(
                    m1["flux_peak"],
                    m2["flux_peak"],
                    m1.get("flux_peak_err", m1.get("flux_err", 0.001)),
                    m2.get("flux_peak_err", m2.get("flux_err", 0.001)),
                )
                pair["m_peak"] = calculate_m_metric(m1["flux_peak"], m2["flux_peak"])
            except (ValueError, ZeroDivisionError):
                pair["vs_peak"] = np.nan
                pair["m_peak"] = np.nan

            pairs.append(pair)

    return pairs


def get_most_significant_pair(
    pairs: list[dict[str, Any]],
    min_abs_vs: float = 4.3,
) -> dict[str, Any] | None:
    """Get the most significant two-epoch pair.

        Adopted from VAST Pipeline: Source.vs_abs_significant_max_*

        Returns the pair with highest |Vs| that exceeds the threshold.

    Parameters
    ----------
    pairs : list of dict
        List of pair metrics from calc_two_epoch_pair_metrics
    min_abs_vs : float, optional
        Minimum |Vs| threshold (default 4.3)

    Returns
    -------
        dict or None
        Most significant pair dict, or None if none exceed threshold
    """
    if not pairs:
        return None

    significant = [
        p
        for p in pairs
        if (np.isfinite(p.get("vs_int", np.nan)) and abs(p["vs_int"]) >= min_abs_vs)
        or (np.isfinite(p.get("vs_peak", np.nan)) and abs(p["vs_peak"]) >= min_abs_vs)
    ]

    if not significant:
        return None

    # Sort by maximum |Vs| (either int or peak)
    def max_vs(p: dict) -> float:
        vs_int = abs(p.get("vs_int", 0)) if np.isfinite(p.get("vs_int", np.nan)) else 0
        vs_peak = abs(p.get("vs_peak", 0)) if np.isfinite(p.get("vs_peak", np.nan)) else 0
        return max(vs_int, vs_peak)

    return max(significant, key=max_vs)
