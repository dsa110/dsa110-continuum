"""Catalog-based validation utilities for DSA-110 imaging pipeline.

    This module provides validation functions that compare detected sources in
    pipeline-generated images against reference catalogs (NVSS, FIRST, VLASS, etc.)
    to verify flux scale accuracy, astrometric precision, and source completeness.

    Based on VAST survey validation patterns.

    Functions
---------
    validate_flux_scale
    Validate flux scale against reference catalog.
    run_full_validation
    Run all validation types (astrometry, flux, counts).
    extract_sources_from_image
    Extract source positions from FITS image.

    Example
-------
    >>> from dsa110_contimg.core.qa.catalog_validation import validate_flux_scale
    >>> result = validate_flux_scale("image.fits", catalog="nvss", min_snr=5.0)
    >>> print(f"Matched {result.n_matched} sources")
    >>> print(f"Mean flux ratio: {result.mean_flux_ratio:.3f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.io import fits
from astropy.stats import mad_std, sigma_clipped_stats
from astropy.wcs import WCS

from dsa110_contimg.common.utils.fits_utils import get_2d_data_and_wcs

logger = logging.getLogger(__name__)


# =============================================================================
# Result Dataclasses
# =============================================================================


@dataclass
class FluxScaleResult:
    """Result from flux scale validation against reference catalog."""

    n_matched: int = 0
    mean_flux_ratio: float = 1.0
    rms_flux_ratio: float = 0.0
    flux_scale_error: float = 0.0
    has_issues: bool = False
    issues: list[str] = field(default_factory=list)
    has_warnings: bool = False
    warnings: list[str] = field(default_factory=list)
    median_flux_ratio: float = 1.0
    matched_sources: pd.DataFrame | None = None


@dataclass
class AstrometryResult:
    """Result from astrometric validation against reference catalog."""

    n_matched: int = 0
    rms_offset_arcsec: float = 0.0
    median_offset_arcsec: float = 0.0
    ra_offset_arcsec: float = 0.0
    dec_offset_arcsec: float = 0.0
    has_issues: bool = False
    issues: list[str] = field(default_factory=list)
    has_warnings: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class SourceCountsResult:
    """Result from source count / completeness validation."""

    n_detected: int = 0
    n_expected: int = 0
    n_matched: int = 0
    completeness: float = 0.0
    false_positive_rate: float = 0.0
    has_issues: bool = False
    issues: list[str] = field(default_factory=list)
    has_warnings: bool = False
    warnings: list[str] = field(default_factory=list)


# =============================================================================
# Source Extraction
# =============================================================================


def extract_sources_from_image(
    image_path: str | Path,
    min_snr: float = 5.0,
    max_sources: int = 500,
    rms_box_size: int = 100,
) -> pd.DataFrame:
    """Extract sources from a FITS image using simple peak detection.

        This is a lightweight source finder that identifies local maxima above
        the noise threshold. For production use, consider using Aegean or PyBDSF.

    Parameters
    ----------
    image_path : Union[str, Path]
        Path to FITS image.
    min_snr : float, optional
        Minimum signal-to-noise ratio for detection. Default is 5.0.
    max_sources : int, optional
        Maximum number of sources to return. Default is 500.
    rms_box_size : int, optional
        Box size for local RMS estimation (pixels). Default is 100.

    Returns
    -------
        list
        List of detected sources.

    Examples
    --------
        >>> sources = extract_sources_from_image("image.fits", min_snr=5.0)
        >>> print(f"Found {len(sources)} sources")
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load FITS data using canonical utility (memmap, float64, WCS fallback)
    data, wcs, _header_info = get_2d_data_and_wcs(image_path)

    # Estimate global RMS using MAD
    finite_data = data[np.isfinite(data)]
    if len(finite_data) == 0:
        logger.warning("No finite pixels in image")
        return pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_jy", "snr", "x_pix", "y_pix"])

    _, _, rms = sigma_clipped_stats(finite_data, sigma=3.0)
    if rms <= 0:
        rms = mad_std(finite_data)

    threshold = min_snr * rms

    # Simple peak detection using scipy
    try:
        from scipy.ndimage import label, maximum_filter
    except ImportError:
        logger.warning("scipy not available - returning empty source list")
        return pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_jy", "snr", "x_pix", "y_pix"])

    # Find local maxima
    # Replace NaN with -inf for peak detection
    data_clean = np.where(np.isfinite(data), data, -np.inf)

    # Apply maximum filter
    local_max = maximum_filter(data_clean, size=5)
    is_peak = (data_clean == local_max) & (data_clean > threshold)

    # Label connected regions
    labeled, n_regions = label(is_peak)

    if n_regions == 0:
        logger.info(f"No sources above {min_snr}σ threshold found")
        return pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_jy", "snr", "x_pix", "y_pix"])

    # Extract peak positions and fluxes
    sources = []
    for region_id in range(1, min(n_regions + 1, max_sources + 1)):
        mask = labeled == region_id
        if not np.any(mask):
            continue

        # Find peak pixel in this region
        region_data = np.where(mask, data_clean, -np.inf)
        peak_idx = np.unravel_index(np.argmax(region_data), region_data.shape)
        y_pix, x_pix = peak_idx

        flux = data[y_pix, x_pix]
        if not np.isfinite(flux):
            continue

        snr = flux / rms

        # Convert pixel to world coordinates
        try:
            ra, dec = wcs.pixel_to_world_values(x_pix, y_pix)
            if not (np.isfinite(ra) and np.isfinite(dec)):
                continue
        except Exception:
            continue

        sources.append(
            {
                "ra_deg": float(ra),
                "dec_deg": float(dec),
                "flux_jy": float(flux),
                "snr": float(snr),
                "x_pix": float(x_pix),
                "y_pix": float(y_pix),
            }
        )

    df = pd.DataFrame(sources)

    # Sort by flux (brightest first) and limit
    if len(df) > 0:
        df = df.sort_values("flux_jy", ascending=False).head(max_sources).reset_index(drop=True)

    logger.info(f"Extracted {len(df)} sources from {image_path.name}")
    return df


# =============================================================================
# Flux Scale Validation
# =============================================================================


def validate_flux_scale(
    image_path: str | Path,
    catalog: str = "nvss",
    min_snr: float = 5.0,
    flux_range_jy: tuple[float, float] = (0.01, 10.0),
    max_flux_ratio_error: float = 0.2,
    match_radius_arcsec: float = 10.0,
    detected_sources: pd.DataFrame | None = None,
    catalog_sources: pd.DataFrame | None = None,
) -> FluxScaleResult:
    """Validate flux scale by comparing detected sources with reference catalog.

        Cross-matches sources in the image with a reference catalog and computes
        the flux ratio to assess calibration accuracy.

    Parameters
    ----------
    image_path : Union[str, Path]
        Path to FITS image (used for WCS and source extraction).
    catalog : str, optional
        Reference catalog name ("nvss", "first", "vlass", "racs"). Default is "nvss".
    min_snr : float, optional
        Minimum SNR for source detection. Default is 5.0.
    flux_range_jy : Tuple[float, float], optional
        Flux range (min, max) in Jy to consider for matching. Default is (0.01, 10.0).
    max_flux_ratio_error : float, optional
        Maximum acceptable flux ratio error (0.2 = 20%). Default is 0.2.
    match_radius_arcsec : float, optional
        Cross-match radius in arcseconds. Default is 10.0.
    detected_sources : Optional[pd.DataFrame], optional
        Pre-extracted sources (extracted if None). Default is None.
    catalog_sources : Optional[pd.DataFrame], optional
        Pre-queried catalog sources (queried if None). Default is None.

    Returns
    -------
        object
        Result object containing flux scale validation metrics.

    Examples
    --------
        >>> result = validate_flux_scale("image.fits", catalog="nvss", min_snr=5.0)
        >>> print(f"Flux scale error: {result.flux_scale_error * 100:.1f}%")
    """
    image_path = Path(image_path)
    result = FluxScaleResult()

    # Extract sources if not provided
    if detected_sources is None:
        try:
            detected_sources = extract_sources_from_image(
                image_path, min_snr=min_snr, max_sources=500
            )
        except Exception as e:
            logger.warning(f"Failed to extract sources: {e}")
            result.has_issues = True
            result.issues.append(f"Source extraction failed: {e}")
            return result

    if len(detected_sources) == 0:
        logger.warning("No sources extracted from image")
        result.has_warnings = True
        result.warnings.append("No sources extracted from image")
        return result

    # Query catalog if not provided
    if catalog_sources is None:
        try:
            catalog_sources = _query_catalog_for_image(image_path, catalog)
        except Exception as e:
            logger.warning(f"Failed to query catalog: {e}")
            result.has_issues = True
            result.issues.append(f"Catalog query failed: {e}")
            return result

    if len(catalog_sources) == 0:
        logger.warning(f"No {catalog.upper()} sources found in field")
        result.has_warnings = True
        result.warnings.append(f"No {catalog.upper()} sources in field")
        return result

    # Filter by flux range
    min_flux, max_flux = flux_range_jy
    detected_filtered = detected_sources[
        (detected_sources["flux_jy"] >= min_flux) & (detected_sources["flux_jy"] <= max_flux)
    ].copy()

    # Determine catalog flux column
    flux_col = "flux_mjy" if "flux_mjy" in catalog_sources.columns else "flux_jy"
    if flux_col == "flux_mjy":
        catalog_flux_jy = catalog_sources[flux_col] / 1000.0
    else:
        catalog_flux_jy = catalog_sources[flux_col]

    catalog_filtered = catalog_sources[
        (catalog_flux_jy >= min_flux) & (catalog_flux_jy <= max_flux)
    ].copy()

    if len(detected_filtered) == 0 or len(catalog_filtered) == 0:
        result.has_warnings = True
        result.warnings.append(f"No sources in flux range {min_flux:.3f}-{max_flux:.3f} Jy")
        return result

    # Cross-match
    detected_coords = SkyCoord(
        ra=detected_filtered["ra_deg"].values * u.deg,
        dec=detected_filtered["dec_deg"].values * u.deg,
    )
    catalog_coords = SkyCoord(
        ra=catalog_filtered["ra_deg"].values * u.deg,
        dec=catalog_filtered["dec_deg"].values * u.deg,
    )

    idx, sep2d, _ = match_coordinates_sky(detected_coords, catalog_coords)

    # Filter by match radius
    match_radius = match_radius_arcsec * u.arcsec
    matched_mask = sep2d < match_radius

    n_matched = int(np.sum(matched_mask))
    result.n_matched = n_matched

    if n_matched < 3:
        result.has_warnings = True
        result.warnings.append(f"Only {n_matched} matched sources (need ≥3 for robust statistics)")
        if n_matched == 0:
            return result

    # Compute flux ratios for matched sources
    detected_flux = detected_filtered.iloc[matched_mask]["flux_jy"].values
    catalog_idx = idx[matched_mask]

    if flux_col == "flux_mjy":
        catalog_flux = catalog_filtered.iloc[catalog_idx]["flux_mjy"].values / 1000.0
    else:
        catalog_flux = catalog_filtered.iloc[catalog_idx]["flux_jy"].values

    # Avoid division by zero
    valid_flux = catalog_flux > 0
    if not np.any(valid_flux):
        result.has_issues = True
        result.issues.append("No valid catalog fluxes for comparison")
        return result

    flux_ratios = detected_flux[valid_flux] / catalog_flux[valid_flux]

    # Compute statistics (using sigma-clipped to reject outliers)
    try:
        mean_ratio, median_ratio, std_ratio = sigma_clipped_stats(flux_ratios, sigma=3.0)
    except Exception:
        mean_ratio = np.nanmean(flux_ratios)
        median_ratio = np.nanmedian(flux_ratios)
        std_ratio = np.nanstd(flux_ratios)

    result.mean_flux_ratio = float(mean_ratio)
    result.median_flux_ratio = float(median_ratio)
    result.rms_flux_ratio = float(std_ratio)
    result.flux_scale_error = abs(1.0 - mean_ratio)

    # Check for issues
    if result.flux_scale_error > max_flux_ratio_error:
        result.has_issues = True
        result.issues.append(
            f"Flux scale error ({result.flux_scale_error * 100:.1f}%) exceeds "
            f"threshold ({max_flux_ratio_error * 100:.0f}%)"
        )
    elif result.flux_scale_error > max_flux_ratio_error / 2:
        result.has_warnings = True
        result.warnings.append(
            f"Flux scale error ({result.flux_scale_error * 100:.1f}%) approaching threshold"
        )

    if std_ratio > 0.3:
        result.has_warnings = True
        result.warnings.append(f"High flux ratio scatter (RMS={std_ratio:.2f})")

    # Store matched sources for detailed analysis
    matched_df = pd.DataFrame(
        {
            "ra_deg": detected_filtered.iloc[matched_mask]["ra_deg"].values,
            "dec_deg": detected_filtered.iloc[matched_mask]["dec_deg"].values,
            "detected_flux_jy": detected_flux,
            "catalog_flux_jy": catalog_flux,
            "flux_ratio": flux_ratios if len(flux_ratios) == len(detected_flux) else np.nan,
            "separation_arcsec": sep2d[matched_mask].arcsec,
        }
    )
    result.matched_sources = matched_df

    logger.info(
        f"Flux scale validation: {n_matched} matched, "
        f"mean ratio={mean_ratio:.3f}, error={result.flux_scale_error * 100:.1f}%"
    )

    return result


# =============================================================================
# Astrometry Validation
# =============================================================================


def _validate_astrometry(
    detected_sources: pd.DataFrame,
    catalog_sources: pd.DataFrame,
    match_radius_arcsec: float = 10.0,
    max_rms_arcsec: float = 5.0,
) -> AstrometryResult:
    """Validate astrometry by computing positional offsets.

    Parameters
    ----------
    detected_sources : pd.DataFrame
        DataFrame with ra_deg, dec_deg columns for detected sources.
    catalog_sources : pd.DataFrame
        Reference catalog DataFrame.
    match_radius_arcsec : float, optional
        Cross-match radius in arcseconds. Default is 10.0.
    max_rms_arcsec : float, optional
        Maximum acceptable RMS offset. Default is 5.0.

    Returns
    -------
        None
    """
    result = AstrometryResult()

    if len(detected_sources) == 0 or len(catalog_sources) == 0:
        result.has_warnings = True
        result.warnings.append("Insufficient sources for astrometry validation")
        return result

    # Cross-match
    detected_coords = SkyCoord(
        ra=detected_sources["ra_deg"].values * u.deg,
        dec=detected_sources["dec_deg"].values * u.deg,
    )
    catalog_coords = SkyCoord(
        ra=catalog_sources["ra_deg"].values * u.deg,
        dec=catalog_sources["dec_deg"].values * u.deg,
    )

    idx, sep2d, _ = match_coordinates_sky(detected_coords, catalog_coords)

    match_radius = match_radius_arcsec * u.arcsec
    matched_mask = sep2d < match_radius
    n_matched = int(np.sum(matched_mask))

    result.n_matched = n_matched

    if n_matched < 5:
        result.has_warnings = True
        result.warnings.append(f"Only {n_matched} matched sources (need ≥5 for robust astrometry)")
        if n_matched == 0:
            return result

    # Compute offsets
    matched_detected = detected_sources.iloc[matched_mask]
    matched_catalog = catalog_sources.iloc[idx[matched_mask]]

    # RA offset (corrected for cos(dec))
    cos_dec = np.cos(np.radians(matched_detected["dec_deg"].values))
    ra_offset_deg = (matched_detected["ra_deg"].values - matched_catalog["ra_deg"].values) * cos_dec
    dec_offset_deg = matched_detected["dec_deg"].values - matched_catalog["dec_deg"].values

    ra_offset_arcsec = ra_offset_deg * 3600.0
    dec_offset_arcsec = dec_offset_deg * 3600.0

    # Total offset
    total_offset_arcsec = np.sqrt(ra_offset_arcsec**2 + dec_offset_arcsec**2)

    # Statistics
    result.ra_offset_arcsec = float(np.nanmean(ra_offset_arcsec))
    result.dec_offset_arcsec = float(np.nanmean(dec_offset_arcsec))
    result.median_offset_arcsec = float(np.nanmedian(total_offset_arcsec))
    result.rms_offset_arcsec = float(np.sqrt(np.nanmean(total_offset_arcsec**2)))

    # Check for issues
    if result.rms_offset_arcsec > max_rms_arcsec:
        result.has_issues = True
        result.issues.append(
            f'RMS offset ({result.rms_offset_arcsec:.2f}") exceeds threshold ({max_rms_arcsec}")'
        )
    elif result.rms_offset_arcsec > max_rms_arcsec / 2:
        result.has_warnings = True
        result.warnings.append(
            f'RMS offset ({result.rms_offset_arcsec:.2f}") approaching threshold'
        )

    # Check for systematic offset
    systematic_offset = np.sqrt(result.ra_offset_arcsec**2 + result.dec_offset_arcsec**2)
    if systematic_offset > result.max_separation_arcsec:
        result.has_warnings = True
        result.warnings.append(
            f'Systematic offset ({systematic_offset:.2f}") exceeds threshold ({result.max_separation_arcsec:.2f}"): '
            f'ΔRA={result.ra_offset_arcsec:.2f}", ΔDec={result.dec_offset_arcsec:.2f}"'
        )
    elif systematic_offset > 2.0:
        result.has_warnings = True
        result.warnings.append(
            f'Systematic offset detected: ΔRA={result.ra_offset_arcsec:.2f}", '
            f'ΔDec={result.dec_offset_arcsec:.2f}"'
        )

    logger.info(
        f"Astrometry validation: {n_matched} matched, "
        f'RMS={result.rms_offset_arcsec:.2f}", '
        f'median={result.median_offset_arcsec:.2f}"'
    )

    return result


# =============================================================================
# Source Counts / Completeness Validation
# =============================================================================


def _validate_source_counts(
    detected_sources: pd.DataFrame,
    catalog_sources: pd.DataFrame,
    min_flux_jy: float = 0.01,
    match_radius_arcsec: float = 10.0,
    min_completeness: float = 0.7,
) -> SourceCountsResult:
    """Validate source counts and completeness.

    Parameters
    ----------
    detected_sources : pd.DataFrame
        DataFrame with detected sources.
    catalog_sources : pd.DataFrame
        Reference catalog DataFrame.
    min_flux_jy : float, optional
        Minimum flux for completeness calculation. Default is 0.01.
    match_radius_arcsec : float, optional
        Cross-match radius in arcseconds. Default is 10.0.
    min_completeness : float, optional
        Minimum acceptable completeness fraction. Default is 0.7.

    Returns
    -------
        None
    """
    result = SourceCountsResult()

    # Filter by flux
    detected_filtered = (
        detected_sources[detected_sources["flux_jy"] >= min_flux_jy]
        if "flux_jy" in detected_sources.columns
        else detected_sources
    )

    # Determine catalog flux column
    flux_col = "flux_mjy" if "flux_mjy" in catalog_sources.columns else "flux_jy"
    if flux_col == "flux_mjy":
        flux_threshold = min_flux_jy * 1000  # Convert to mJy
    else:
        flux_threshold = min_flux_jy

    catalog_filtered = catalog_sources[catalog_sources[flux_col] >= flux_threshold]

    result.n_detected = len(detected_filtered)
    result.n_expected = len(catalog_filtered)

    if result.n_expected == 0:
        result.has_warnings = True
        result.warnings.append("No catalog sources above flux threshold")
        return result

    if result.n_detected == 0:
        result.has_issues = True
        result.issues.append("No sources detected above flux threshold")
        return result

    # Cross-match to find completeness
    detected_coords = SkyCoord(
        ra=detected_filtered["ra_deg"].values * u.deg,
        dec=detected_filtered["dec_deg"].values * u.deg,
    )
    catalog_coords = SkyCoord(
        ra=catalog_filtered["ra_deg"].values * u.deg,
        dec=catalog_filtered["dec_deg"].values * u.deg,
    )

    # Match catalog → detected (for completeness)
    _, sep_cat, _ = match_coordinates_sky(catalog_coords, detected_coords)
    match_radius = match_radius_arcsec * u.arcsec
    catalog_matched = sep_cat < match_radius

    result.n_matched = int(np.sum(catalog_matched))
    result.completeness = result.n_matched / result.n_expected if result.n_expected > 0 else 0.0

    # Match detected → catalog (for false positives)
    _, sep_det, _ = match_coordinates_sky(detected_coords, catalog_coords)
    detected_matched = sep_det < match_radius
    n_false_positives = int(np.sum(~detected_matched))
    result.false_positive_rate = (
        n_false_positives / result.n_detected if result.n_detected > 0 else 0.0
    )

    # Check for issues
    if result.completeness < min_completeness:
        result.has_issues = True
        result.issues.append(
            f"Completeness ({result.completeness * 100:.1f}%) below threshold "
            f"({min_completeness * 100:.0f}%)"
        )
    elif result.completeness < min_completeness + 0.1:
        result.has_warnings = True
        result.warnings.append(
            f"Completeness ({result.completeness * 100:.1f}%) approaching threshold"
        )

    if result.false_positive_rate > 0.3:
        result.has_warnings = True
        result.warnings.append(
            f"High false positive rate ({result.false_positive_rate * 100:.1f}%)"
        )

    logger.info(
        f"Source counts validation: detected={result.n_detected}, "
        f"expected={result.n_expected}, completeness={result.completeness * 100:.1f}%"
    )

    return result


# =============================================================================
# Full Validation Pipeline
# =============================================================================


def run_full_validation(
    image_path: str | Path,
    catalog: str = "nvss",
    validation_types: list[str] | None = None,
    generate_html: bool = False,
    html_output_path: str | None = None,
    min_snr: float = 5.0,
    match_radius_arcsec: float = 10.0,
    catalog_radius_deg: float = 1.5,
    max_astrometry_rms_arcsec: float = 5.0,
) -> tuple[AstrometryResult | None, FluxScaleResult | None, SourceCountsResult | None]:
    """Run full validation suite on a pipeline-generated image.

        Performs astrometry, flux scale, and source count validation against
        a reference catalog.

    Parameters
    ----------
    image_path : Union[str, Path]
        Path to FITS image.
    catalog : str
        Reference catalog ("nvss", "first", "vlass", "racs").
        Default is "nvss".
    validation_types : Optional[List[str]]
        List of validation types to run. Options: ["astrometry", "flux_scale", "source_counts"].
        Default is all types.
    generate_html : bool
        Whether to generate HTML report. Default is False.
    html_output_path : Optional[str]
        Path for HTML report (required if generate_html=True). Default is None.
    min_snr : float
        Minimum SNR for source extraction. Default is 5.0.
    match_radius_arcsec : float
        Cross-match radius in arcseconds. Default is 10.0.
    catalog_radius_deg : float
        Catalog query radius around image center in degrees. Default is 1.5.
    max_astrometry_rms_arcsec : float
        Maximum acceptable astrometric RMS offset in arcseconds. Default is 5.0.

    Returns
    -------
        tuple
        Results of the validation: astrometry, flux, counts.

    Examples
    --------
        >>> astrometry, flux, counts = run_full_validation(
        ...     "image.fits",
        ...     catalog="nvss",
        ...     validation_types=["astrometry", "flux_scale"],
        ... )
        >>> if flux and flux.flux_scale_error < 0.1:
        ...     print("Flux scale OK!")
    """
    image_path = Path(image_path)

    if validation_types is None:
        validation_types = ["astrometry", "flux_scale", "source_counts"]

    logger.info(
        f"Running validation on {image_path.name}: "
        f"types={validation_types}, catalog={catalog.upper()}"
    )

    astrometry_result: AstrometryResult | None = None
    flux_scale_result: FluxScaleResult | None = None
    source_counts_result: SourceCountsResult | None = None

    # Extract sources once for all validation types
    try:
        detected_sources = extract_sources_from_image(image_path, min_snr=min_snr, max_sources=500)
    except Exception as e:
        logger.error(f"Source extraction failed: {e}")
        # Return empty results with issues flagged
        if "astrometry" in validation_types:
            astrometry_result = AstrometryResult(
                has_issues=True, issues=[f"Source extraction failed: {e}"]
            )
        if "flux_scale" in validation_types:
            flux_scale_result = FluxScaleResult(
                has_issues=True, issues=[f"Source extraction failed: {e}"]
            )
        if "source_counts" in validation_types:
            source_counts_result = SourceCountsResult(
                has_issues=True, issues=[f"Source extraction failed: {e}"]
            )
        return astrometry_result, flux_scale_result, source_counts_result

    # Query catalog once for all validation types
    try:
        catalog_sources = _query_catalog_for_image(
            image_path,
            catalog,
            radius_deg=catalog_radius_deg,
        )
    except Exception as e:
        logger.error(f"Catalog query failed: {e}")
        if "astrometry" in validation_types:
            astrometry_result = AstrometryResult(
                has_issues=True, issues=[f"Catalog query failed: {e}"]
            )
        if "flux_scale" in validation_types:
            flux_scale_result = FluxScaleResult(
                has_issues=True, issues=[f"Catalog query failed: {e}"]
            )
        if "source_counts" in validation_types:
            source_counts_result = SourceCountsResult(
                has_issues=True, issues=[f"Catalog query failed: {e}"]
            )
        return astrometry_result, flux_scale_result, source_counts_result

    # Run requested validations
    if "astrometry" in validation_types:
        try:
            astrometry_result = _validate_astrometry(
                detected_sources,
                catalog_sources,
                match_radius_arcsec=match_radius_arcsec,
                max_rms_arcsec=max_astrometry_rms_arcsec,
            )
        except Exception as e:
            logger.warning(f"Astrometry validation failed: {e}")
            astrometry_result = AstrometryResult(
                has_issues=True, issues=[f"Validation failed: {e}"]
            )

    if "flux_scale" in validation_types:
        try:
            flux_scale_result = validate_flux_scale(
                image_path,
                catalog=catalog,
                min_snr=min_snr,
                match_radius_arcsec=match_radius_arcsec,
                detected_sources=detected_sources,
                catalog_sources=catalog_sources,
            )
        except Exception as e:
            logger.warning(f"Flux scale validation failed: {e}")
            flux_scale_result = FluxScaleResult(has_issues=True, issues=[f"Validation failed: {e}"])

    if "source_counts" in validation_types:
        try:
            source_counts_result = _validate_source_counts(
                detected_sources, catalog_sources, match_radius_arcsec=match_radius_arcsec
            )
        except Exception as e:
            logger.warning(f"Source counts validation failed: {e}")
            source_counts_result = SourceCountsResult(
                has_issues=True, issues=[f"Validation failed: {e}"]
            )

    # Generate HTML report if requested
    if generate_html and html_output_path:
        try:
            _generate_html_report(
                html_output_path,
                image_path,
                astrometry_result,
                flux_scale_result,
                source_counts_result,
            )
            logger.info(f"HTML report generated: {html_output_path}")
        except Exception as e:
            logger.warning(f"Failed to generate HTML report: {e}")

    return astrometry_result, flux_scale_result, source_counts_result


# =============================================================================
# Helper Functions
# =============================================================================


def _query_catalog_for_image(
    image_path: str | Path,
    catalog: str,
    radius_deg: float = 1.5,
) -> pd.DataFrame:
    """Query catalog sources within the image field of view.

    Parameters
    ----------
    image_path : Union[str, Path]
        Path to FITS image.
    catalog : str
        Catalog name ("nvss", "first", "vlass", "racs").
    radius_deg : float, optional
        Search radius in degrees. Default is 1.5.

    Returns
    -------
        None
    """
    image_path = Path(image_path)

    # Get image center from WCS
    with fits.open(image_path) as hdul:
        header = hdul[0].header
        wcs = WCS(header, naxis=2)

    # Get image center
    naxis1 = header.get("NAXIS1", 1)
    naxis2 = header.get("NAXIS2", 1)
    center_pix = (naxis1 / 2, naxis2 / 2)
    ra_center, dec_center = wcs.pixel_to_world_values(*center_pix)

    # Query catalog
    from dsa110_contimg.core.calibration.catalog_registry import query_catalog

    catalog_sources = query_catalog(
        catalog=catalog,
        ra_deg=float(ra_center),
        dec_deg=float(dec_center),
        radius_deg=radius_deg,
    )

    logger.debug(
        f"Queried {len(catalog_sources)} sources from {catalog.upper()} "
        f"around RA={ra_center:.3f}, Dec={dec_center:.3f}"
    )

    return catalog_sources


def _generate_html_report(
    output_path: str,
    image_path: Path,
    astrometry_result: AstrometryResult | None,
    flux_scale_result: FluxScaleResult | None,
    source_counts_result: SourceCountsResult | None,
) -> None:
    """Generate HTML validation report.

    Parameters
    ----------
    output_path :
        Path to write HTML file
    image_path :
        Path to the validated image
    astrometry_result :
        Astrometry validation results
    flux_scale_result :
        Flux scale validation results
    source_counts_result :
        Source counts validation results
    """
    from datetime import datetime

    # Build HTML content
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<title>Validation Report</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; }",
        "h1 { color: #333; }",
        "h2 { color: #666; border-bottom: 1px solid #ccc; }",
        ".pass { color: green; }",
        ".warn { color: orange; }",
        ".fail { color: red; }",
        "table { border-collapse: collapse; margin: 10px 0; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "th { background-color: #f2f2f2; }",
        ".issue { background-color: #ffe0e0; }",
        ".warning { background-color: #fff3cd; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>Validation Report: {image_path.name}</h1>",
        f"<p>Generated: {datetime.now().isoformat()}</p>",
    ]

    # Astrometry section
    if astrometry_result:
        status_class = (
            "fail"
            if astrometry_result.has_issues
            else ("warn" if astrometry_result.has_warnings else "pass")
        )
        status_text = (
            "FAIL"
            if astrometry_result.has_issues
            else ("WARNING" if astrometry_result.has_warnings else "PASS")
        )
        html_parts.extend(
            [
                "<h2>Astrometry Validation</h2>",
                f"<p>Status: <span class='{status_class}'>{status_text}</span></p>",
                "<table>",
                f"<tr><th>Sources Matched</th><td>{astrometry_result.n_matched}</td></tr>",
                f'<tr><th>RMS Offset</th><td>{astrometry_result.rms_offset_arcsec:.2f}"</td></tr>',
                f'<tr><th>Median Offset</th><td>{astrometry_result.median_offset_arcsec:.2f}"</td></tr>',
                f'<tr><th>RA Offset</th><td>{astrometry_result.ra_offset_arcsec:.2f}"</td></tr>',
                f'<tr><th>Dec Offset</th><td>{astrometry_result.dec_offset_arcsec:.2f}"</td></tr>',
                "</table>",
            ]
        )
        for issue in astrometry_result.issues:
            html_parts.append(f"<p class='issue'> {issue}</p>")
        for warning in astrometry_result.warnings:
            html_parts.append(f"<p class='warning'> {warning}</p>")

    # Flux scale section
    if flux_scale_result:
        status_class = (
            "fail"
            if flux_scale_result.has_issues
            else ("warn" if flux_scale_result.has_warnings else "pass")
        )
        status_text = (
            "FAIL"
            if flux_scale_result.has_issues
            else ("WARNING" if flux_scale_result.has_warnings else "PASS")
        )
        html_parts.extend(
            [
                "<h2>Flux Scale Validation</h2>",
                f"<p>Status: <span class='{status_class}'>{status_text}</span></p>",
                "<table>",
                f"<tr><th>Sources Matched</th><td>{flux_scale_result.n_matched}</td></tr>",
                f"<tr><th>Mean Flux Ratio</th><td>{flux_scale_result.mean_flux_ratio:.3f}</td></tr>",
                f"<tr><th>Median Flux Ratio</th><td>{flux_scale_result.median_flux_ratio:.3f}</td></tr>",
                f"<tr><th>RMS Scatter</th><td>{flux_scale_result.rms_flux_ratio:.3f}</td></tr>",
                f"<tr><th>Flux Scale Error</th><td>{flux_scale_result.flux_scale_error * 100:.1f}%</td></tr>",
                "</table>",
            ]
        )
        for issue in flux_scale_result.issues:
            html_parts.append(f"<p class='issue'> {issue}</p>")
        for warning in flux_scale_result.warnings:
            html_parts.append(f"<p class='warning'> {warning}</p>")

    # Source counts section
    if source_counts_result:
        status_class = (
            "fail"
            if source_counts_result.has_issues
            else ("warn" if source_counts_result.has_warnings else "pass")
        )
        status_text = (
            "FAIL"
            if source_counts_result.has_issues
            else ("WARNING" if source_counts_result.has_warnings else "PASS")
        )
        html_parts.extend(
            [
                "<h2>Source Counts Validation</h2>",
                f"<p>Status: <span class='{status_class}'>{status_text}</span></p>",
                "<table>",
                f"<tr><th>Sources Detected</th><td>{source_counts_result.n_detected}</td></tr>",
                f"<tr><th>Sources Expected</th><td>{source_counts_result.n_expected}</td></tr>",
                f"<tr><th>Sources Matched</th><td>{source_counts_result.n_matched}</td></tr>",
                f"<tr><th>Completeness</th><td>{source_counts_result.completeness * 100:.1f}%</td></tr>",
                f"<tr><th>False Positive Rate</th><td>{source_counts_result.false_positive_rate * 100:.1f}%</td></tr>",
                "</table>",
            ]
        )
        for issue in source_counts_result.issues:
            html_parts.append(f"<p class='issue'> {issue}</p>")
        for warning in source_counts_result.warnings:
            html_parts.append(f"<p class='warning'> {warning}</p>")

    html_parts.extend(
        [
            "</body>",
            "</html>",
        ]
    )

    # Write HTML file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as html_file:
        html_file.write("\n".join(html_parts))
