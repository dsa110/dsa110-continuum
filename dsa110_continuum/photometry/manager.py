"""PhotometryManager: Centralized photometry workflow coordination.

This manager consolidates scattered photometry functionality into a unified
interface, similar to StreamingMosaicManager for mosaic workflows.

Workflow:
    1. Query catalog sources for field
    2. Measure photometry (forced/adaptive/Aegean)
    3. Normalize measurements (optional)
    4. Detect ESE candidates (optional)
    5. Store results in database
    6. Link to data registry
"""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from dsa110_contimg.interfaces.api.batch import create_batch_photometry_job
from dsa110_contimg.infrastructure.database import (
    ensure_pipeline_db,
    photometry_insert,
)
from dsa110_contimg.infrastructure.database.data_registry import (
    ensure_data_registry_db,
    link_photometry_to_data,
)
from dsa110_contimg.core.photometry.ese_pipeline import auto_detect_ese_for_new_measurements
from dsa110_contimg.core.photometry.forced import ForcedPhotometryResult, measure_many
from dsa110_contimg.core.photometry.helpers import (
    query_sources_for_fits,
    query_sources_for_mosaic,
)
from dsa110_contimg.common.utils.decorators import timed

logger = logging.getLogger(__name__)


class PhotometryConfig:
    """Configuration for photometry workflow."""

    def __init__(
        self,
        catalog: str = "nvss",
        radius_deg: float | None = None,
        ra_radius_deg: float | None = None,
        dec_radius_deg: float | None = None,
        min_flux_mjy: float | None = None,
        max_sources: int | None = None,
        method: str = "peak",
        normalize: bool = False,
        detect_ese: bool = False,
        catalog_path: Path | None = None,
        auto_compute_extent: bool = True,
    ):
        """Initialize photometry configuration.

        Parameters
        ----------
        catalog : str
            Catalog type ("nvss", "first", "rax", "vlass", "master").
        radius_deg : float or None
            Search radius in degrees (circular, 0.5 for images, 1.0 for mosaics).
            If None and ra_radius_deg/dec_radius_deg not provided, will be computed
            from FITS extent if auto_compute_extent=True.
        ra_radius_deg : float or None
            RA search radius in degrees (for elongated mosaics).
            If None, uses radius_deg or computes from extent.
        dec_radius_deg : float or None
            Dec search radius in degrees (for elongated mosaics).
            If None, uses radius_deg or computes from extent.
        min_flux_mjy : float
            Minimum flux in mJy for catalog sources.
        max_sources : int
            Maximum number of sources to measure.
        method : str
            Photometry method ("peak", "adaptive", "aegean").
        normalize : bool
            Enable normalization using reference sources.
        detect_ese : bool
            Automatically detect ESE candidates after measurement.
        catalog_path : str or Path or None
            Optional path to catalog database file.
        auto_compute_extent : bool
            If True and radii not provided, compute from FITS extent.
        """
        self.catalog = catalog
        self.radius_deg = radius_deg
        self.ra_radius_deg = ra_radius_deg
        self.dec_radius_deg = dec_radius_deg
        self.min_flux_mjy = min_flux_mjy
        self.max_sources = max_sources
        self.method = method
        self.normalize = normalize
        self.detect_ese = detect_ese
        self.catalog_path = catalog_path
        self.auto_compute_extent = auto_compute_extent

        self._validate()

    def _validate(self):
        """Validate configuration parameters."""
        valid_catalogs = {"nvss", "first", "rax", "vlass", "master"}
        if self.catalog not in valid_catalogs:
            raise ValueError(f"Invalid catalog '{self.catalog}'. Must be one of {valid_catalogs}")

        if self.radius_deg is not None and self.radius_deg <= 0:
            raise ValueError(f"radius_deg must be positive, got {self.radius_deg}")

        if self.min_flux_mjy is not None and self.min_flux_mjy <= 0:
            raise ValueError(f"min_flux_mjy must be positive, got {self.min_flux_mjy}")

        if self.max_sources is not None and self.max_sources <= 0:
            raise ValueError(f"max_sources must be positive, got {self.max_sources}")

        valid_methods = {"peak", "adaptive", "aegean"}
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method '{self.method}'. Must be one of {valid_methods}")

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "PhotometryConfig":
        """Create config from dictionary.

        Parameters
        ----------
        config_dict : dict
            Configuration dictionary.

        Returns
        -------
            PhotometryConfig
            Configuration object created from dictionary.
        """
        return cls(**config_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "catalog": self.catalog,
            "radius_deg": self.radius_deg,
            "ra_radius_deg": self.ra_radius_deg,
            "dec_radius_deg": self.dec_radius_deg,
            "min_flux_mjy": self.min_flux_mjy,
            "max_sources": self.max_sources,
            "method": self.method,
            "normalize": self.normalize,
            "detect_ese": self.detect_ese,
            "catalog_path": str(self.catalog_path) if self.catalog_path else None,
            "auto_compute_extent": self.auto_compute_extent,
        }


class PhotometryResult:
    """Result of photometry measurement workflow."""

    def __init__(
        self,
        fits_path: Path,
        sources_queried: int,
        measurements_successful: int,
        measurements_total: int,
        batch_job_id: int | None = None,
        results: list[ForcedPhotometryResult] | None = None,
    ):
        """Initialize photometry result.

        Parameters
        ----------
        fits_path : Path
            Path to FITS file that was measured.
        sources_queried : int
            Number of sources found in catalog.
        measurements_successful : int
            Number of successful measurements.
        measurements_total : int
            Total number of measurement attempts.
        batch_job_id : str or None, optional
            Batch job ID if created asynchronously.
        results : list or None, optional
            List of measurement results if executed synchronously.
        """
        self.fits_path = fits_path
        self.sources_queried = sources_queried
        self.measurements_successful = measurements_successful
        self.measurements_total = measurements_total
        self.batch_job_id = batch_job_id
        self.results = results

    @property
    def success_rate(self) -> float:
        """Calculate success rate of measurements."""
        if self.measurements_total == 0:
            return 0.0
        return self.measurements_successful / self.measurements_total


class PhotometryManager:
    """Manages photometry workflow for images and mosaics.

        This class consolidates scattered photometry functionality into a unified
        interface. It handles:
        - Catalog source querying
        - Photometry measurement (forced/adaptive/Aegean)
        - Normalization (optional)
        - ESE detection (optional)
        - Database storage
        - Data registry linking

    Examples
    --------
        >>> manager = PhotometryManager(
        ...     products_db_path=Path("state/db/pipeline.sqlite3"),
        ...     data_registry_db_path=Path("state/db/pipeline.sqlite3"),
        ... )
        >>> result = manager.measure_for_fits(
        ...     fits_path=Path("image.fits"),
        ...     config=PhotometryConfig(catalog="nvss", radius_deg=0.5),
        ... )
        >>> print(f"Measured {result.measurements_successful} sources")
    """

    def __init__(
        self,
        products_db_path: Path,
        data_registry_db_path: Path | None = None,
        default_config: PhotometryConfig | None = None,
    ):
        """Initialize photometry manager.

        Parameters
        ----------
        products_db_path : Path
            Path to products database.
        data_registry_db_path : Path or None, optional
            Optional path to data registry database.
        default_config : PhotometryConfig or None, optional
            Default photometry configuration.
        """
        self.products_db_path = products_db_path
        self.data_registry_db_path = data_registry_db_path
        self.default_config = default_config or PhotometryConfig()

    @timed("photometry.measure_for_fits")
    def measure_for_fits(
        self,
        fits_path: Path,
        config: PhotometryConfig | None = None,
        create_batch_job: bool = True,
        data_id: str | None = None,
        group_id: str | None = None,
        dry_run: bool = False,
    ) -> PhotometryResult | None:
        """Run complete photometry workflow for a FITS image.

        Parameters
        ----------
        fits_path : Path
            Path to FITS image file.
        config : PhotometryConfig or None, optional
            Photometry configuration (uses default if None).
        create_batch_job : bool, optional
            If True, create batch job for async execution.
            If False, execute synchronously. Default is True.
        data_id : str or None, optional
            Optional data ID for data registry linking.
        group_id : str or None, optional
            Optional group ID for tracking.
        dry_run : bool, optional
            If True, simulate workflow without creating jobs or measurements.

        Returns
        -------
            PhotometryResult
            Result of the photometry measurement.
        """
        if not fits_path.exists():
            logger.warning(f"FITS image not found: {fits_path}")
            return None

        config = config or self.default_config

        try:
            # Determine search radii (support elongated mosaics)
            ra_radius, dec_radius = self._get_search_radii(fits_path, config)

            # Query sources for the image field with separate RA/Dec radii for elongated mosaics
            sources = query_sources_for_fits(
                fits_path,
                catalog=config.catalog,
                radius_deg=config.radius_deg,
                min_flux_mjy=config.min_flux_mjy,
                max_sources=config.max_sources,
                catalog_path=config.catalog_path,
                ra_radius_deg=ra_radius,
                dec_radius_deg=dec_radius,
            )

            if not sources:
                logger.info(f"No sources found for photometry in {fits_path}")
                return PhotometryResult(
                    fits_path=fits_path,
                    sources_queried=0,
                    measurements_successful=0,
                    measurements_total=0,
                )

            # Extract coordinates from sources
            coordinates = [
                {
                    "ra_deg": float(src.get("ra", src.get("ra_deg", 0.0))),
                    "dec_deg": float(src.get("dec", src.get("dec_deg", 0.0))),
                    "source_id": src.get("source_id", src.get("name", None)),
                    "nvss_flux_mjy": src.get("flux_peak", src.get("flux_int", None)),
                }
                for src in sources
            ]

            logger.info(f"Found {len(coordinates)} sources for photometry in {fits_path.name}")

            if dry_run:
                logger.info(f"DRY-RUN MODE: Would measure {len(coordinates)} sources")
                return PhotometryResult(
                    fits_path=fits_path,
                    sources_queried=len(sources),
                    measurements_successful=0,
                    measurements_total=len(coordinates),
                    batch_job_id=None,
                )

            if create_batch_job:
                # Create batch job for async execution
                batch_job_id = self._create_batch_job(
                    fits_paths=[fits_path],
                    coordinates=coordinates,
                    config=config,
                    data_id=data_id or fits_path.stem,
                    dry_run=dry_run,
                )
                if batch_job_id:
                    # Link to data registry if available
                    if self.data_registry_db_path and data_id:
                        self._link_to_data_registry(data_id, str(batch_job_id))

                    return PhotometryResult(
                        fits_path=fits_path,
                        sources_queried=len(sources),
                        measurements_successful=0,  # Unknown until job completes
                        measurements_total=len(coordinates),
                        batch_job_id=batch_job_id,
                    )
                return None
            else:
                # Execute synchronously
                # measure_many expects list of tuples
                coords_tuples = [(c["ra_deg"], c["dec_deg"]) for c in coordinates]
                results = measure_many(str(fits_path), coords_tuples)
                successful = len([r for r in results if r.success])

                if self.products_db_path:
                    conn = ensure_pipeline_db()
                    try:
                        # Store results
                        for i, res in enumerate(results):
                            # Assuming results match input coordinates order
                            if i >= len(coordinates):
                                break

                            source_id = coordinates[i].get("source_id")
                            nvss_flux = coordinates[i].get("nvss_flux_mjy")

                            # Insert into photometry table
                            if res.success:
                                measured_ts = time.time()
                                photometry_insert(
                                    conn,
                                    image_path=str(fits_path),
                                    ra_deg=res.ra_deg,
                                    dec_deg=res.dec_deg,
                                    nvss_flux_mjy=float(nvss_flux) if nvss_flux else None,
                                    peak_jyb=res.peak_jyb,
                                    peak_err_jyb=res.peak_err_jyb,
                                    flux_jy=res.peak_jyb,
                                    flux_err_jy=res.peak_err_jyb,
                                    normalized_flux_jy=res.peak_jyb,
                                    normalized_flux_err_jy=res.peak_err_jyb,
                                    measured_at=measured_ts,
                                    source_id=source_id,
                                )

                                # Detect ESE if configured
                                if config.detect_ese and source_id:
                                    # We call this per source, but we might want to batch it later
                                    # For now, simple sequential calls
                                    auto_detect_ese_for_new_measurements(
                                        products_db=self.products_db_path, source_id=source_id
                                    )

                        conn.commit()
                    except Exception as e:
                        logger.error(f"Failed to store results/detect ESE: {e}")
                    finally:
                        conn.close()

                return PhotometryResult(
                    fits_path=fits_path,
                    sources_queried=len(sources),
                    measurements_successful=successful,
                    measurements_total=len(coordinates),
                    results=results,
                )

        except Exception as e:
            logger.error(
                f"Failed to run photometry for {fits_path}: {e}",
                exc_info=True,
            )
            return None

    @timed("photometry.measure_for_mosaic")
    def measure_for_mosaic(
        self,
        mosaic_path: Path,
        config: PhotometryConfig | None = None,
        create_batch_job: bool = True,
        data_id: str | None = None,
        group_id: str | None = None,
        dry_run: bool = False,
    ) -> PhotometryResult | None:
        """Run complete photometry workflow for a mosaic FITS file.

            Similar to `measure_for_fits()` but uses larger default search radius
            (1.0 deg) for mosaics.

        Parameters
        ----------
        mosaic_path : Path
            Path to mosaic FITS file.
        config : PhotometryConfig or None, optional
            Photometry configuration (uses default with radius=1.0 if None).
        create_batch_job : bool, optional
            If True, create batch job for async execution.
        data_id : str or None, optional
            Optional data ID for data registry linking.
        group_id : str or None, optional
            Optional group ID for tracking.
        dry_run : bool, optional
            If True, simulate workflow without creating jobs or measurements.

        Returns
        -------
            PhotometryResult
            Result of the photometry measurement.
        """
        if not mosaic_path.exists():
            logger.warning(f"Mosaic FITS file not found: {mosaic_path}")
            return None

        # Use mosaic-appropriate defaults if config not provided
        if config is None:
            config = PhotometryConfig(
                catalog=self.default_config.catalog,
                radius_deg=1.0,  # Larger radius for mosaics
                min_flux_mjy=self.default_config.min_flux_mjy,
                max_sources=self.default_config.max_sources,
                method=self.default_config.method,
                normalize=self.default_config.normalize,
                detect_ese=self.default_config.detect_ese,
                catalog_path=self.default_config.catalog_path,
            )

        try:
            # Determine search radii (support elongated mosaics)
            ra_radius, dec_radius = self._get_search_radii(mosaic_path, config)

            # Query sources for the mosaic field with separate RA/Dec radii for elongated mosaics
            sources = query_sources_for_mosaic(
                mosaic_path,
                catalog=config.catalog,
                radius_deg=config.radius_deg,
                min_flux_mjy=config.min_flux_mjy,
                max_sources=config.max_sources,
                catalog_path=config.catalog_path,
                ra_radius_deg=ra_radius,
                dec_radius_deg=dec_radius,
            )

            if not sources:
                logger.info(f"No sources found for photometry in {mosaic_path}")
                return PhotometryResult(
                    fits_path=mosaic_path,
                    sources_queried=0,
                    measurements_successful=0,
                    measurements_total=0,
                )

            # Extract coordinates from sources
            coordinates = [
                {
                    "ra_deg": float(src.get("ra", src.get("ra_deg", 0.0))),
                    "dec_deg": float(src.get("dec", src.get("dec_deg", 0.0))),
                    "source_id": src.get("source_id", src.get("name", None)),
                    "nvss_flux_mjy": src.get("flux_peak", src.get("flux_int", None)),
                }
                for src in sources
            ]

            logger.info(f"Found {len(coordinates)} sources for photometry in {mosaic_path.name}")

            if dry_run:
                logger.info(f"DRY-RUN MODE: Would measure {len(coordinates)} sources")
                return PhotometryResult(
                    fits_path=mosaic_path,
                    sources_queried=len(sources),
                    measurements_successful=0,
                    measurements_total=len(coordinates),
                    batch_job_id=None,
                )

            if create_batch_job:
                # Create batch job for async execution
                batch_job_id = self._create_batch_job(
                    fits_paths=[mosaic_path],
                    coordinates=coordinates,
                    config=config,
                    data_id=data_id or mosaic_path.stem,
                    dry_run=dry_run,
                )
                if batch_job_id:
                    # Link to data registry if available
                    if self.data_registry_db_path and data_id:
                        self._link_to_data_registry(data_id, str(batch_job_id))

                    return PhotometryResult(
                        fits_path=mosaic_path,
                        sources_queried=len(sources),
                        measurements_successful=0,  # Unknown until job completes
                        measurements_total=len(coordinates),
                        batch_job_id=batch_job_id,
                    )
                return None
            else:
                # Execute synchronously
                # measure_many expects list of tuples
                coords_tuples = [(c["ra_deg"], c["dec_deg"]) for c in coordinates]
                results = measure_many(str(mosaic_path), coords_tuples)
                successful = len([r for r in results if r.success])

                if self.products_db_path:
                    conn = ensure_pipeline_db()
                    try:
                        # Store results
                        for i, res in enumerate(results):
                            if i >= len(coordinates):
                                break

                            source_id = coordinates[i].get("source_id")
                            nvss_flux = coordinates[i].get("nvss_flux_mjy")

                            if res.success:
                                measured_ts = time.time()
                                photometry_insert(
                                    conn,
                                    image_path=str(mosaic_path),
                                    ra_deg=res.ra_deg,
                                    dec_deg=res.dec_deg,
                                    nvss_flux_mjy=float(nvss_flux) if nvss_flux else None,
                                    peak_jyb=res.peak_jyb,
                                    peak_err_jyb=res.peak_err_jyb,
                                    flux_jy=res.peak_jyb,
                                    flux_err_jy=res.peak_err_jyb,
                                    normalized_flux_jy=res.peak_jyb,
                                    normalized_flux_err_jy=res.peak_err_jyb,
                                    measured_at=measured_ts,
                                    source_id=source_id,
                                    mosaic_path=str(mosaic_path),
                                )

                                if config.detect_ese and source_id:
                                    auto_detect_ese_for_new_measurements(
                                        products_db=self.products_db_path, source_id=source_id
                                    )

                        conn.commit()
                    except Exception as e:
                        logger.error(f"Failed to store results/detect ESE: {e}")
                    finally:
                        conn.close()

                return PhotometryResult(
                    fits_path=mosaic_path,
                    sources_queried=len(sources),
                    measurements_successful=successful,
                    measurements_total=len(coordinates),
                    results=results,
                )

        except Exception as e:
            logger.error(
                f"Failed to run photometry for {mosaic_path}: {e}",
                exc_info=True,
            )
            return None

    def _create_batch_job(
        self,
        fits_paths: list[Path],
        coordinates: list[dict[str, float]],
        config: PhotometryConfig,
        data_id: str | None = None,
        dry_run: bool = False,
    ) -> int | None:
        """Create a batch photometry job.

        Parameters
        ----------
        fits_paths : list of Path
            List of FITS file paths.
        coordinates : list of dict
            List of coordinate dictionaries.
        config : PhotometryConfig
            Photometry configuration.
        data_id : str or None, optional
            Optional data ID for linking.
        dry_run : bool, optional
            If True, simulate without creating job.

        Returns
        -------
            BatchJob
            Created batch photometry job object.
        """
        if dry_run:
            logger.info(f"DRY-RUN MODE: Would create batch job for {len(fits_paths)} file(s)")
            return None

        try:
            conn = ensure_pipeline_db()

            # Prepare batch job parameters
            params = {
                "method": config.method,
                "normalize": config.normalize,
            }

            # Create batch photometry job
            batch_job_id = create_batch_photometry_job(
                conn=conn,
                job_type="batch_photometry",
                fits_paths=[str(p) for p in fits_paths],
                coordinates=coordinates,
                params=params,
                data_id=data_id,
            )

            logger.info(
                f"Created photometry batch job {batch_job_id} for {len(fits_paths)} file(s)"
            )
            return batch_job_id

        except Exception as e:
            logger.error(f"Failed to create batch photometry job: {e}", exc_info=True)
            return None

    def _link_to_data_registry(self, data_id: str, photometry_job_id: str) -> bool:
        """Link photometry job to data registry.

        Parameters
        ----------
        data_id : str
            Data product ID.
        photometry_job_id : str
            Photometry job ID.

        Returns
        -------
            None
        """
        if not self.data_registry_db_path:
            return False

        try:
            conn = ensure_data_registry_db(self.data_registry_db_path)
            success = link_photometry_to_data(conn, data_id, photometry_job_id)
            if success:
                logger.debug(f"Linked photometry job {photometry_job_id} to data_id {data_id}")
            else:
                logger.debug(f"Could not link photometry job (data_id {data_id} may not exist)")
            return success
        except Exception as e:
            logger.debug(f"Failed to link photometry to data registry: {e}")
            return False

    def _get_search_radii(self, fits_path: Path, config: PhotometryConfig) -> tuple[float, float]:
        """Get RA and Dec search radii for a FITS file.

            Supports elongated mosaics by computing extent from FITS header
            or using explicit radii from config.

        Parameters
        ----------
        fits_path : Path
            Path to FITS file.
        config : PhotometryConfig
            Photometry configuration.

        Returns
        -------
            tuple of float
            RA and Dec search radii in degrees.
        """
        # Use explicit radii if provided
        if config.ra_radius_deg is not None and config.dec_radius_deg is not None:
            return config.ra_radius_deg, config.dec_radius_deg

        # Use single radius if provided (circular search)
        if config.radius_deg is not None:
            return config.radius_deg, config.radius_deg

        # Auto-compute from FITS extent if enabled
        if config.auto_compute_extent:
            try:
                ra_extent, dec_extent = self._compute_field_extent(fits_path)
                # Add 10% buffer for edge cases
                ra_radius = ra_extent / 2 * 1.1
                dec_radius = dec_extent / 2 * 1.1
                logger.debug(
                    f"Computed search radii from FITS extent: "
                    f"RA={ra_radius:.3f}°, Dec={dec_radius:.3f}° "
                    f"(extent: {ra_extent:.3f}° × {dec_extent:.3f}°)"
                )
                return ra_radius, dec_radius
            except Exception as e:
                logger.warning(f"Failed to compute extent from {fits_path}, using defaults: {e}")

        # Fallback to defaults
        default_radius = 0.5  # For images
        return default_radius, default_radius

    def _compute_field_extent(self, fits_path: Path) -> tuple[float, float]:
        """Compute RA and Dec extent of FITS image from header.

        Parameters
        ----------
        fits_path : Path
            Path to FITS file.

        Returns
        -------
            tuple of float
            RA and Dec extent in degrees.
        """
        with fits.open(fits_path) as hdul:
            hdr = hdul[0].header
            wcs = WCS(hdr)

            if not wcs.has_celestial:
                raise ValueError("FITS file has no celestial WCS")

            naxis1 = hdr.get("NAXIS1", 0)
            naxis2 = hdr.get("NAXIS2", 0)

            if naxis1 == 0 or naxis2 == 0:
                raise ValueError("Invalid image dimensions")

            # Get corners in pixel coordinates
            corners_pix = np.array(
                [
                    [0, 0],
                    [naxis1, 0],
                    [naxis1, naxis2],
                    [0, naxis2],
                ]
            )

            # Convert to world coordinates
            corners_world = wcs.all_pix2world(corners_pix, 0)
            ra_corners = corners_world[:, 0]
            dec_corners = corners_world[:, 1]

            # Handle RA wrap-around (if crossing 0/360)
            ra_diff = np.max(ra_corners) - np.min(ra_corners)
            if ra_diff > 180:
                # Wrap around - add 360 to negative values
                ra_corners = np.where(ra_corners < 180, ra_corners + 360, ra_corners)

            ra_extent = float(np.max(ra_corners) - np.min(ra_corners))
            dec_extent = float(np.max(dec_corners) - np.min(dec_corners))

            return ra_extent, dec_extent
