"""
Mosaic Orchestrator - Bridge between legacy adapter and mosaic pipeline.

This module provides the MosaicOrchestrator class that the legacy adapter
expects for mosaic creation tasks. It wraps the underlying pipeline functions
with an interface suitable for async task execution.

Key responsibilities:
- Translate group IDs and image paths to pipeline inputs
- Manage database connections and output directories
- Coordinate optional photometry after mosaic creation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import statistics

from dsa110_contimg.common.utils.decorators import timed

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for MosaicOrchestrator."""

    products_db_path: Path | None = None
    hdf5_db_path: Path | None = None
    mosaic_dir: Path = field(default_factory=lambda: Path("/data/mosaics"))
    enable_photometry: bool = True
    photometry_config: dict[str, Any] | None = None


class MosaicOrchestrator:
    """Orchestrates mosaic creation from legacy tasks.

        This class is the interface expected by the legacy adapter's
        execute_create_mosaic function. It provides methods for:
        - Creating mosaics from a group ID (auto-discover images)
        - Creating mosaics from explicit image paths
        - Processing observations with optional photometry

    Examples
    --------
        >>> orchestrator = MosaicOrchestrator(
        ...     products_db_path=Path("pipeline.sqlite3"),
        ...     enable_photometry=True,
        ... )
        >>> result = orchestrator.create_mosaic_for_group(
        ...     group_id="2025-06-01_12:00:00",
        ...     center_ra_deg=180.0,
        ... )
    """

    def __init__(
        self,
        products_db_path: Path | str | None = None,
        hdf5_db_path: Path | str | None = None,
        mosaic_dir: Path | str | None = None,
        enable_photometry: bool = True,
        photometry_config: dict[str, Any] | None = None,
    ):
        """Initialize orchestrator.

        Parameters
        ----------
        products_db_path : Path
            Path to products database
        hdf5_db_path : Path, optional
            Path to HDF5 database (optional)
        mosaic_dir : Path
            Directory for output mosaics
        enable_photometry : bool
            Whether to run photometry after mosaic
        photometry_config : dict
            Configuration dict for photometry
        """
        self.products_db_path = Path(products_db_path) if products_db_path else None
        self.hdf5_db_path = Path(hdf5_db_path) if hdf5_db_path else None
        self.mosaic_dir = Path(mosaic_dir) if mosaic_dir else Path("/data/mosaics")
        self.enable_photometry = enable_photometry
        self.photometry_config = photometry_config or {}

        # Ensure output directory exists
        self.mosaic_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(
            f"MosaicOrchestrator initialized: db={self.products_db_path}, "
            f"mosaic_dir={self.mosaic_dir}, photometry={self.enable_photometry}"
        )

    @timed("mosaic.create_mosaic_for_group")
    def create_mosaic_for_group(
        self,
        group_id: str,
        center_ra_deg: float | None = None,
        span_minutes: float = 60,
        tier: str = "science",
    ) -> dict[str, Any]:
        """Create mosaic from a group ID.

            This method discovers images associated with the group and
            creates a mosaic using the appropriate tier settings.

        Parameters
        ----------
        group_id : str
            Group identifier (e.g., "2025-06-01_12:00:00")
        center_ra_deg : float or None, optional
            Optional RA center override in degrees (default is None)
        span_minutes : float, optional
            Time span for image selection (default is 60)
        tier : str, optional
            Mosaic tier to use (default is "science")
        """
        from .pipeline import MosaicPipelineConfig, run_on_demand_mosaic

        logger.info(f"Creating mosaic for group: {group_id}")

        # Parse group_id to get time range
        start_time, end_time = self._parse_group_id(group_id, span_minutes)

        # Create pipeline config
        config = MosaicPipelineConfig(
            database_path=self.products_db_path or Path("pipeline.sqlite3"),
            mosaic_dir=self.mosaic_dir,
        )

        # Generate unique mosaic name from group
        mosaic_name = f"group_{group_id.replace(':', '-').replace(' ', '_')}"

        # Run pipeline
        result = run_on_demand_mosaic(
            config=config,
            name=mosaic_name,
            start_time=start_time,
            end_time=end_time,
            tier=tier,
        )

        if not result.success:
            logger.error(f"Mosaic pipeline failed for group {group_id}: {result.errors}")
            return {
                "error": result.message,
                "errors": result.errors,
            }

        # Run photometry if enabled and mosaic succeeded
        photometry_results = None
        if self.enable_photometry and result.mosaic_path:
            photometry_results = self._run_photometry(result.mosaic_path)

        num_tiles = getattr(result, "num_tiles", None)
        if not isinstance(num_tiles, int):
            num_tiles = getattr(result, "n_images", None)
        if not isinstance(num_tiles, int):
            num_tiles = 0

        return {
            "mosaic_path": result.mosaic_path,
            "metadata": {
                "plan_id": result.plan_id,
                "mosaic_id": result.mosaic_id,
                "qa_status": result.qa_status,
                "tier": tier,
                "group_id": group_id,
            },
            "photometry": photometry_results,
            "num_tiles": num_tiles,
        }

    @timed("mosaic.process_observation")
    def process_observation(
        self,
        center_ra_deg: float | None = None,
        group_id: str | None = None,
        dec_deg: float | None = None,
        time_range_hours: float = 24,
    ) -> dict[str, Any]:
        """Process an observation to create a mosaic.

        This is a more flexible interface that can work with either
        a group_id or sky coordinates.

        Parameters
        ----------
        center_ra_deg :
            Center RA in degrees
        group_id :
            Optional group identifier
        dec_deg :
            Center Dec in degrees
        time_range_hours :
            Time range for image selection
        center_ra_deg : float | None :
            (Default value = None)
        group_id : str | None :
            (Default value = None)
        dec_deg : float | None :
            (Default value = None)
        time_range_hours : float :
            (Default value = 24)
        center_ra_deg : float | None :
            (Default value = None)
        group_id : str | None :
            (Default value = None)
        dec_deg : float | None :
            (Default value = None)
        time_range_hours : float :
            (Default value = 24)
        center_ra_deg: float | None :
             (Default value = None)
        group_id: str | None :
             (Default value = None)
        dec_deg: float | None :
             (Default value = None)
        """
        from .pipeline import MosaicPipelineConfig, run_on_demand_mosaic
        from .tiers import select_tier_for_request

        logger.info(
            f"Processing observation: ra={center_ra_deg}, "
            f"group={group_id}, time_range={time_range_hours}h"
        )

        # Determine time range
        now = datetime.now(UTC)
        end_time = int(now.timestamp())
        start_time = end_time - int(time_range_hours * 3600)

        # Auto-select tier
        tier = select_tier_for_request(time_range_hours)

        # Generate unique name
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        if group_id:
            mosaic_name = f"obs_{group_id.replace(':', '-')}_{timestamp}"
        elif center_ra_deg is not None:
            mosaic_name = f"obs_ra{center_ra_deg:.1f}_{timestamp}"
        else:
            mosaic_name = f"obs_{timestamp}"

        # Create pipeline config
        config = MosaicPipelineConfig(
            database_path=self.products_db_path or Path("pipeline.sqlite3"),
            mosaic_dir=self.mosaic_dir,
        )

        # Run pipeline
        result = run_on_demand_mosaic(
            config=config,
            name=mosaic_name,
            start_time=start_time,
            end_time=end_time,
            tier=tier.value,
        )

        if not result.success:
            return None

        # Run photometry if enabled
        photometry_results = None
        if self.enable_photometry and result.mosaic_path:
            photometry_results = self._run_photometry(result.mosaic_path)

        return {
            "mosaic_path": result.mosaic_path,
            "metadata": {
                "plan_id": result.plan_id,
                "mosaic_id": result.mosaic_id,
                "qa_status": result.qa_status,
                "tier": tier.value,
            },
            "photometry": photometry_results,
            "num_tiles": result.num_tiles or 0,
        }

    @timed("mosaic.create_mosaic_from_images")
    def create_mosaic_from_images(
        self,
        image_paths: list[str | Path],
        output_path: Path | str | None = None,
        tier: str = "science",
    ) -> dict[str, Any]:
        """Create mosaic from explicit list of images.

            This bypasses the database query and creates a mosaic
            directly from the provided image files.

        Parameters
        ----------
        image_paths : list of str or Path
            List of FITS image paths
        output_path : Path, str, or None, optional
            Optional output path (auto-generated if not provided)
        tier : str, optional
            Tier to use for quality settings (default is "science")
        """
        from .builder import build_mosaic
        from .qa import run_qa_checks
        from .tiers import TIER_CONFIGS, MosaicTier

        logger.info(f"Creating mosaic from {len(image_paths)} images")

        # Validate and convert paths
        paths = []
        for p in image_paths:
            path = Path(p)
            if path.exists():
                paths.append(path)
            else:
                logger.warning(f"Image not found: {p}")

        if len(paths) < 2:
            return {"error": f"Need at least 2 valid images, found {len(paths)}"}

        # Determine output path
        if output_path:
            out_path = Path(output_path)
        else:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            out_path = self.mosaic_dir / f"mosaic_{timestamp}_n{len(paths)}.fits"

        # Get tier config
        tier_enum = MosaicTier(tier)
        tier_config = TIER_CONFIGS[tier_enum]

        # Build mosaic
        try:
            mosaic_result = build_mosaic(
                image_paths=paths,
                output_path=out_path,
                alignment_order=tier_config.alignment_order,
                timeout_minutes=tier_config.timeout_minutes,
            )
        except Exception as e:
            logger.exception(f"Mosaic build failed: {e}")
            return {"error": str(e)}

        # Run QA
        try:
            qa_result = run_qa_checks(
                mosaic_path=mosaic_result.output_path,
                tier=tier,
            )
            qa_status = "PASS" if qa_result.passed else ("WARN" if qa_result.warnings else "FAIL")
        except Exception as e:
            logger.warning(f"QA check failed: {e}")
            qa_status = "UNKNOWN"

        # Run photometry if enabled
        photometry_results = None
        if self.enable_photometry:
            photometry_results = self._run_photometry(str(mosaic_result.output_path))

        return {
            "mosaic_path": str(mosaic_result.output_path),
            "metadata": {
                "n_images": mosaic_result.n_images,
                "median_rms": mosaic_result.median_rms,
                "coverage_sq_deg": mosaic_result.coverage_sq_deg,
                "qa_status": qa_status,
                "tier": tier,
            },
            "photometry": photometry_results,
            "num_tiles": len(paths),
        }

    def _parse_group_id(
        self,
        group_id: str,
        span_minutes: float,
    ) -> tuple[int, int]:
        """Parse group ID to get time range.

            Group IDs are typically in format "YYYY-MM-DD_HH:MM:SS" or similar.

        Parameters
        ----------
        group_id : str
            Group identifier string
        span_minutes : float
            Time span in minutes
        """
        # Try common formats
        formats = [
            "%Y-%m-%d_%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y%m%d_%H%M%S",
            "%Y%m%d%H%M%S",
        ]

        center_time = None
        for fmt in formats:
            try:
                center_time = datetime.strptime(group_id, fmt)
                center_time = center_time.replace(tzinfo=UTC)
                break
            except ValueError:
                continue

        if center_time is None:
            # Fall back to current time
            logger.warning(f"Could not parse group_id '{group_id}', using current time")
            center_time = datetime.now(UTC)

        # Calculate time range
        half_span_seconds = (span_minutes * 60) / 2
        center_unix = center_time.timestamp()

        start_time = int(center_unix - half_span_seconds)
        end_time = int(center_unix + half_span_seconds)

        return start_time, end_time

    def _run_photometry(self, mosaic_path: str) -> dict[str, Any] | None:
        """Run photometry on a mosaic.

        Parameters
        ----------
        mosaic_path : str
            Path to mosaic FITS file
        """
        try:
            from dsa110_contimg.core.photometry import run_photometry

            result = run_photometry(
                image_path=Path(mosaic_path),
                **self.photometry_config,
            )
            return result
        except ImportError:
            logger.debug("Photometry module not available")
            return None
        except Exception as e:
            logger.warning(f"Photometry failed: {e}")
            return None

    @timed("mosaic.create_sliding_window_mosaic")
    def create_sliding_window_mosaic(
        self,
        tile_image_ids: list[int],
        tier: str = "science",
    ) -> dict[str, Any]:
        """Create a mosaic from tiles specified by the sliding-window trigger.

        This method is called when the sliding-window trigger determines
        that a mosaic should be created. It takes the exact list of tile
        image IDs and creates the mosaic.

        Parameters
        ----------
        tile_image_ids : list[int]
            List of image database IDs to include in the mosaic.
        tier : str
            Mosaic tier to use (default: "science").

        Returns
        -------
        dict
            Result containing mosaic_path, metadata, photometry, and num_tiles.
        """
        from .trigger import SlidingWindowTrigger

        logger.info(f"Creating sliding-window mosaic with {len(tile_image_ids)} tiles")

        # Get image paths from database
        from dsa110_contimg.infrastructure.database.unified import Database, get_pipeline_db_path

        db_path = self.products_db_path or get_pipeline_db_path()
        db = Database(db_path)

        placeholders = ",".join("?" * len(tile_image_ids))
        rows = db.query(
            f"""
            SELECT id, path, center_dec_deg
            FROM images
            WHERE id IN ({placeholders})
            ORDER BY created_at ASC
            """,
            tuple(tile_image_ids),
        )

        if not rows:
            return {
                "error": "No valid images found for tile IDs",
                "tile_image_ids": tile_image_ids,
            }

        image_paths = [row["path"] for row in rows]
        avg_dec = statistics.mean(r["center_dec_deg"] or 0 for r in rows)

        # Create mosaic using existing method
        result = self.create_mosaic_from_images(
            image_paths=image_paths,
            tier=tier,
        )

        if result.get("error"):
            return result

        # Get mosaic ID from database for trigger callback
        mosaic_path = result.get("mosaic_path")
        mosaic_id = None
        mosaic_mjd = None

        if mosaic_path:
            # Query mosaic ID and MJD
            mosaic_row = db.query(
                "SELECT id, mid_mjd FROM mosaics WHERE path = ?",
                (mosaic_path,),
            )
            if mosaic_row:
                mosaic_id = mosaic_row[0]["id"]
                mosaic_mjd = mosaic_row[0]["mid_mjd"]

        # Update trigger state
        if mosaic_id:
            try:
                trigger = SlidingWindowTrigger(db_path=db_path)
                trigger.mark_mosaic_complete(
                    mosaic_id=mosaic_id,
                    tile_image_ids=tile_image_ids,
                    mosaic_mjd=mosaic_mjd or 0.0,
                )
            except Exception as e:
                logger.warning(f"Failed to update trigger state: {e}")

        # Add sliding-window metadata
        result["metadata"] = result.get("metadata", {})
        result["metadata"]["trigger_type"] = "sliding-window"
        result["metadata"]["tile_image_ids"] = tile_image_ids
        result["metadata"]["avg_dec_deg"] = avg_dec

        return result
