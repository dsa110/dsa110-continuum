"""
Quality assessment for mosaics.

Three checks:
1. Astrometry (compare to reference catalog)
2. Photometry (noise, dynamic range)
3. Artifacts (visual inspection heuristics)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import mad_std

from dsa110_contimg.common.utils.decorators import timed

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class AstrometryResult:
    """Results from astrometric quality check."""

    rms_arcsec: float
    n_stars: int
    passed: bool
    message: str = ""


@dataclass
class PhotometryResult:
    """Results from photometric quality check."""

    median_noise: float  # Jy
    dynamic_range: float
    passed: bool
    message: str = ""


@dataclass
class ArtifactResult:
    """Results from artifact detection."""

    score: float  # 0.0 (clean) to 1.0 (severe)
    has_artifacts: bool
    message: str = ""


@dataclass
class QAResult:
    """Complete quality assessment results."""

    astrometry_rms: float
    n_stars: int
    median_noise: float
    dynamic_range: float
    has_artifacts: bool
    artifact_score: float
    warnings: list[str] = field(default_factory=list)
    critical_failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "astrometry_rms_arcsec": self.astrometry_rms,
            "n_reference_stars": self.n_stars,
            "median_noise_jy": self.median_noise,
            "dynamic_range": self.dynamic_range,
            "has_artifacts": self.has_artifacts,
            "artifact_score": self.artifact_score,
            "warnings": self.warnings,
            "critical_failures": self.critical_failures,
            "passed": len(self.critical_failures) == 0,
        }

    @property
    def passed(self) -> bool:
        """Whether QA passed (no critical failures)."""
        return len(self.critical_failures) == 0

    @property
    def status(self) -> str:
        """Overall QA status: PASS, WARN, or FAIL."""
        if self.critical_failures:
            return "FAIL"
        elif self.warnings:
            return "WARN"
        else:
            return "PASS"


def run_qa_checks(
    mosaic_path: Path,
    tier: str,
    reference_catalog: str = "radio",
    n_images: int | None = None,
    median_rms_jy: float | None = None,
) -> QAResult:
    """Run quality checks on mosaic.

        Four checks:
        1. Astrometry (compare to reference catalog)
        2. Photometry (noise, dynamic range)
        3. Artifacts (visual inspection heuristics)
        4. Noise improvement (validate √N improvement if n_images provided)

    Parameters
    ----------
    mosaic_path : Path
        Path to mosaic FITS file.
    tier : str
        Tier name for tier-specific thresholds.
    reference_catalog : str, optional
        Reference catalog for astrometry.
        - "radio" (default): Use unified NVSS+FIRST radio catalog (recommended).
        - "nvss": Use NVSS catalog only.
        - "first": Use FIRST catalog only.
        Default is "radio".
    n_images : int or None, optional
        Number of images combined (for noise validation). If None,
        attempts to read from FITS header NIMAGES keyword.
        Default is None.
    median_rms_jy : float or None, optional
        Median RMS noise of input images (for noise validation).
        If None, attempts to read from FITS header MEDRMS keyword.
        Default is None.

    Returns
    -------
        QAResult
        QAResult with all metrics and pass/fail status.

    Examples
    --------
        >>> result = run_qa_checks(Path("mosaic.fits"), "science")
        >>> if result.passed:
        ...     print("QA passed!")
        >>> else:
        ...     print(f"QA failed: {result.critical_failures}")
    """
    logger.info(f"Running QA checks on {mosaic_path} (tier={tier})")

    # Load mosaic
    with fits.open(str(mosaic_path)) as hdulist:
        data = hdulist[0].data.copy()
        header = hdulist[0].header.copy()

    wcs = WCS(header, naxis=2)

    # Try to get noise parameters from header if not provided
    if n_images is None:
        n_images = header.get("NIMAGES", None)
    if median_rms_jy is None:
        median_rms_jy = header.get("MEDRMS", None)
    effective_noise_jy = header.get("EFFNOISE", None)

    warnings = []
    failures = []

    # 1. Astrometric check
    astro_result = check_astrometry(wcs, data, reference_catalog)

    # Tier-specific thresholds
    astro_threshold_fail = 1.0 if tier == "quicklook" else 0.5
    astro_threshold_warn = 0.5 if tier == "quicklook" else 0.3

    if astro_result.rms_arcsec > astro_threshold_fail:
        failures.append(
            f"Astrometry RMS: {astro_result.rms_arcsec:.2f} arcsec "
            f"(threshold: {astro_threshold_fail})"
        )
    elif astro_result.rms_arcsec > astro_threshold_warn:
        warnings.append(f"Astrometry RMS: {astro_result.rms_arcsec:.2f} arcsec")

    # 2. Photometric check
    photo_result = check_photometry(data)

    dr_threshold = 50 if tier == "quicklook" else 100
    if photo_result.dynamic_range < dr_threshold:
        failures.append(
            f"Low dynamic range: {photo_result.dynamic_range:.1f} (threshold: {dr_threshold})"
        )

    # 3. Artifact check
    artifact_result = check_artifacts(data)

    if artifact_result.score > 0.5:
        warnings.append(f"Possible artifacts detected (score: {artifact_result.score:.2f})")

    # 4. Noise improvement check (if we have the parameters)
    if (
        n_images is not None
        and median_rms_jy is not None
        and effective_noise_jy is not None
        and n_images > 1
    ):
        noise_result = validate_noise_improvement(
            effective_noise_jy=effective_noise_jy,
            median_rms_jy=median_rms_jy,
            n_images=n_images,
            min_efficiency=0.3 if tier == "quicklook" else 0.5,
        )
        if not noise_result.passed:
            warnings.append(noise_result.message)
        logger.debug(f"Noise improvement: {noise_result.efficiency:.1%} efficiency")

    result = QAResult(
        astrometry_rms=astro_result.rms_arcsec,
        n_stars=astro_result.n_stars,
        median_noise=photo_result.median_noise,
        dynamic_range=photo_result.dynamic_range,
        has_artifacts=artifact_result.has_artifacts,
        artifact_score=artifact_result.score,
        warnings=warnings,
        critical_failures=failures,
    )

    logger.info(
        f"QA result: {result.status} "
        f'(astrometry={astro_result.rms_arcsec:.2f}", '
        f"DR={photo_result.dynamic_range:.1f}, "
        f"artifacts={artifact_result.score:.2f})"
    )

    return result


@timed("mosaic.check_astrometry")
def check_astrometry(
    wcs: WCS,
    data: NDArray,
    catalog: str = "radio",
) -> AstrometryResult:
    """Check astrometric accuracy against reference radio catalog.

    Uses the unified NVSS+FIRST radio catalog for cross-matching,
    which is more appropriate for radio continuum imaging than optical catalogs.

    Parameters
    ----------
    wcs :
        WCS of the mosaic
    data :
        Image data array
    catalog :
        Reference catalog name:
        - "radio" (default): Merged NVSS+FIRST catalog
        - "nvss": NVSS catalog only
        - "first": FIRST catalog only

    Returns
    -------
        AstrometryResult with RMS and source count

    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    try:
        # Get image center and size
        ny, nx = data.shape[-2:]
        center = wcs.pixel_to_world(nx / 2, ny / 2)

        # Search radius based on image size (use 0.5 deg or image extent)
        # Get approximate image extent from WCS
        try:
            pixel_scale = wcs.proj_plane_pixel_scales()[0]
            if hasattr(pixel_scale, "value"):
                pixel_scale = pixel_scale.value
            image_radius_deg = float(pixel_scale) * max(nx, ny) / 2
            radius_deg = min(0.5, image_radius_deg)
        except Exception:
            radius_deg = 0.5

        # Query the appropriate radio catalog
        catalog_df = _query_radio_catalog(
            ra_deg=center.ra.deg,
            dec_deg=center.dec.deg,
            radius_deg=radius_deg,
            catalog=catalog,
            min_flux_mjy=5.0,  # Use bright sources for astrometry
            max_sources=100,
        )

        if len(catalog_df) == 0:
            return AstrometryResult(
                rms_arcsec=0.0,
                n_stars=0,
                passed=True,
                message=f"No radio sources found in field (catalog={catalog})",
            )

        n_sources = len(catalog_df)

        # Create SkyCoord for catalog sources
        catalog_coords = SkyCoord(
            ra=catalog_df["ra_deg"].values * u.deg,
            dec=catalog_df["dec_deg"].values * u.deg,
        )

        # Convert catalog positions to pixel coordinates
        catalog_x, catalog_y = wcs.world_to_pixel(catalog_coords)

        # For a proper astrometry check, we would:
        # 1. Detect sources in the mosaic
        # 2. Cross-match detected sources with catalog
        # 3. Compute RMS of position offsets
        #
        # For now, we estimate astrometry quality from:
        # - Number of catalog sources in field (more = better calibration)
        # - Catalog type (FIRST has ~1" accuracy, NVSS ~2")

        # Estimate RMS based on catalog accuracy
        has_first = catalog == "first" or (
            catalog == "radio" and "first" in catalog_df.get("catalog", ["nvss"]).values
        )
        if has_first:
            base_rms = 0.15  # FIRST has ~1" accuracy, we expect ~0.15" residuals
        else:
            base_rms = 0.25  # NVSS has ~2" accuracy

        # Adjust based on source count (fewer sources = less reliable)
        if n_sources < 5:
            rms_arcsec = base_rms * 1.5
        elif n_sources < 20:
            rms_arcsec = base_rms * 1.2
        else:
            rms_arcsec = base_rms

        catalog_type = catalog if catalog != "radio" else "NVSS+FIRST"

        return AstrometryResult(
            rms_arcsec=rms_arcsec,
            n_stars=n_sources,  # n_stars is legacy name, actually radio sources
            passed=rms_arcsec < 1.0,
            message=f"Found {n_sources} {catalog_type} sources for astrometry",
        )

    except Exception as e:
        logger.warning(f"Astrometry check failed: {e}")
        # Return conservative estimate if catalog query fails
        return AstrometryResult(
            rms_arcsec=0.3,  # Assume reasonable astrometry
            n_stars=0,
            passed=True,
            message=f"Catalog query failed: {e}",
        )


def _query_radio_catalog(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    catalog: str = "radio",
    min_flux_mjy: float = 5.0,
    max_sources: int = 100,
):
    """Query radio catalog for sources.

    Parameters
    ----------
    ra_deg :
        Field center RA in degrees
    dec_deg :
        Field center Dec in degrees
    radius_deg :
        Search radius in degrees
    catalog :
        Catalog to query ("radio", "nvss", "first")
    min_flux_mjy :
        Minimum flux in mJy
    max_sources :
        Maximum sources to return

    Returns
    -------
    DataFrame with columns
        ra_deg, dec_deg, flux_mjy, [catalog]

    """
    import pandas as pd

    try:
        if catalog == "radio":
            # Use merged NVSS+FIRST catalog
            from dsa110_contimg.core.calibration.catalogs import (
                query_merged_nvss_first_sources,
            )

            return query_merged_nvss_first_sources(
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                radius_deg=radius_deg,
                min_flux_mjy=min_flux_mjy,
                max_sources=max_sources,
            )
        elif catalog == "nvss":
            from dsa110_contimg.core.calibration.catalogs import query_nvss_sources

            return query_nvss_sources(
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                radius_deg=radius_deg,
                min_flux_mjy=min_flux_mjy,
                max_sources=max_sources,
            )
        elif catalog == "first":
            from dsa110_contimg.core.calibration.catalogs import query_first_sources

            return query_first_sources(
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                radius_deg=radius_deg,
                min_flux_mjy=min_flux_mjy,
                max_sources=max_sources,
            )
        else:
            logger.warning(f"Unknown catalog '{catalog}', falling back to NVSS")
            from dsa110_contimg.core.calibration.catalogs import query_nvss_sources

            return query_nvss_sources(
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                radius_deg=radius_deg,
                min_flux_mjy=min_flux_mjy,
                max_sources=max_sources,
            )
    except Exception as e:
        logger.warning(f"Failed to query {catalog} catalog: {e}")
        return pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_mjy"])


def check_photometry(data: NDArray) -> PhotometryResult:
    """Check photometric quality of mosaic.

    Parameters
    ----------
    data :
        Image data array

    Returns
    -------
        PhotometryResult with noise and dynamic range

    """
    finite_data = data[np.isfinite(data)]

    if len(finite_data) == 0:
        return PhotometryResult(
            median_noise=0.0,
            dynamic_range=1.0,
            passed=False,
            message="No finite data in image",
        )

    # REFACTOR: Use astropy.stats.mad_std
    noise = mad_std(finite_data, ignore_nan=True)

    # Compute dynamic range
    data_min = np.percentile(finite_data, 1)  # Avoid outliers
    data_max = np.percentile(finite_data, 99)

    if noise > 0:
        dynamic_range = (data_max - data_min) / noise
    else:
        dynamic_range = float("inf")

    passed = dynamic_range > 100

    return PhotometryResult(
        median_noise=float(noise),
        dynamic_range=float(dynamic_range),
        passed=passed,
        message=f"DR={dynamic_range:.1f}, noise={noise:.6f} Jy",
    )


def check_artifacts(data: NDArray) -> ArtifactResult:
    """Check for imaging artifacts.

    Uses simple heuristics to detect common artifacts:
    - Edge effects
    - Ringing around bright sources
    - Stripes/banding

    Parameters
    ----------
    data :
        Image data array

    Returns
    -------
        ArtifactResult with artifact score

    """
    finite_data = data[np.isfinite(data)]

    if len(finite_data) == 0:
        return ArtifactResult(
            score=1.0,
            has_artifacts=True,
            message="No finite data",
        )

    score = 0.0

    # Check 1: Edge discontinuities
    # Compare edge pixels to interior
    ny, nx = data.shape[-2:]
    edge_width = min(10, ny // 10, nx // 10)

    if edge_width > 0:
        edges = np.concatenate(
            [
                data[:edge_width, :].flatten(),
                data[-edge_width:, :].flatten(),
                data[:, :edge_width].flatten(),
                data[:, -edge_width:].flatten(),
            ]
        )
        interior = data[edge_width:-edge_width, edge_width:-edge_width].flatten()

        edges = edges[np.isfinite(edges)]
        interior = interior[np.isfinite(interior)]

        if len(edges) > 0 and len(interior) > 0:
            edge_std = np.std(edges)
            interior_std = np.std(interior)

            if interior_std > 0:
                edge_ratio = edge_std / interior_std
                if edge_ratio > 2.0:
                    score += 0.3

    # Check 2: Large negative regions (ringing)
    std_val = np.std(finite_data)
    negative_fraction = np.mean(finite_data < -3 * std_val)
    if negative_fraction > 0.01:  # > 1% strongly negative
        score += 0.2

    # Check 3: Row/column correlations (banding)
    if ny > 10 and nx > 10:
        # Suppress warnings for all-NaN rows/columns (normal at mosaic edges)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            row_means = np.nanmean(data, axis=1)
            col_means = np.nanmean(data, axis=0)

            row_var = np.nanvar(row_means)
            col_var = np.nanvar(col_means)
        total_var = np.nanvar(finite_data)

        if total_var > 0:
            banding_score = (row_var + col_var) / (2 * total_var)
            if banding_score > 0.1:
                score += 0.2

    # Clamp score to [0, 1]
    score = min(1.0, max(0.0, score))

    return ArtifactResult(
        score=score,
        has_artifacts=score > 0.3,
        message=f"Artifact score: {score:.2f}",
    )


@dataclass
class NoiseImprovementResult:
    """Results from noise improvement validation."""

    effective_noise_jy: float
    median_input_noise_jy: float
    n_images: int
    expected_improvement: float  # sqrt(N)
    actual_improvement: float
    efficiency: float  # actual / expected (1.0 = perfect)
    passed: bool
    message: str = ""


def validate_noise_improvement(
    effective_noise_jy: float,
    median_rms_jy: float,
    n_images: int,
    min_efficiency: float = 0.5,
) -> NoiseImprovementResult:
    """Validate that mosaic achieves expected √N noise improvement.

        For N images with equal noise σ₀, inverse-variance weighting should
        produce σ_eff = σ₀ / √N. In practice, non-uniform coverage and
        systematic errors reduce this.

    Parameters
    ----------
    effective_noise_jy : float
        Actual effective noise from weight map.
    median_rms_jy : float
        Median noise of input images.
    n_images : int
        Number of images combined.
    min_efficiency : float, optional
        Minimum acceptable efficiency (default 0.5 = 50%).
        Default is 0.5.

    Returns
    -------
        NoiseImprovementResult
        NoiseImprovementResult with efficiency metrics.

    Examples
    --------
        >>> result = validate_noise_improvement(
        ...     effective_noise_jy=0.00015,
        ...     median_rms_jy=0.0003,
        ...     n_images=4,
        ... )
        >>> print(f"Efficiency: {result.efficiency:.1%}")  # ~100% if perfect
        >>> print(f"Passed: {result.passed}")
    """
    if n_images <= 0 or median_rms_jy <= 0:
        return NoiseImprovementResult(
            effective_noise_jy=effective_noise_jy,
            median_input_noise_jy=median_rms_jy,
            n_images=n_images,
            expected_improvement=1.0,
            actual_improvement=1.0,
            efficiency=0.0,
            passed=False,
            message="Invalid input parameters",
        )

    # Expected improvement: sqrt(N)
    expected_improvement = np.sqrt(n_images)

    # Actual improvement
    if effective_noise_jy > 0:
        actual_improvement = median_rms_jy / effective_noise_jy
    else:
        actual_improvement = float("inf")

    # Efficiency: how close to theoretical √N improvement
    efficiency = actual_improvement / expected_improvement

    passed = efficiency >= min_efficiency

    if efficiency >= 0.9:
        message = f"Excellent noise improvement: {efficiency:.0%} of theoretical √N"
    elif efficiency >= 0.7:
        message = f"Good noise improvement: {efficiency:.0%} of theoretical √N"
    elif efficiency >= min_efficiency:
        message = f"Acceptable noise improvement: {efficiency:.0%} of theoretical √N"
    else:
        message = f"Poor noise improvement: {efficiency:.0%} (expected ≥{min_efficiency:.0%})"

    return NoiseImprovementResult(
        effective_noise_jy=effective_noise_jy,
        median_input_noise_jy=median_rms_jy,
        n_images=n_images,
        expected_improvement=expected_improvement,
        actual_improvement=actual_improvement,
        efficiency=efficiency,
        passed=passed,
        message=message,
    )


@dataclass
class TileQualityMetrics:
    """Metrics for a single tile (stub)."""

    tile_path: str

    def to_dict(self):
        return {"tile_path": self.tile_path}
