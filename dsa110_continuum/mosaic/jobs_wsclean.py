"""
WSClean-based job implementations for mosaicking pipeline.

These jobs use visibility-domain joint deconvolution instead of
image-domain stacking. The key differences from the legacy jobs:

1. MosaicMSPlanningJob - Queries ms_index table (not images)
2. MosaicWSCleanBuildJob - Uses WSClean with IDG + beam correction

The pipeline workflow:
1. Planning: Select MS files by time range from ms_index
2. Calibration: Phase-shift transit MS to calibrator, solve BP/gains
3. Apply calibration to all MS copies
4. Mosaic: Phase-shift to mean meridian, run WSClean joint deconvolution
5. QA: Same as legacy (astrometry, photometry, artifacts)
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from astropy.time import Time

from dsa110_contimg.workflow.pipeline import Job, JobResult, register_job

from .wsclean_mosaic import (
    WSCleanMosaicConfig,
    build_wsclean_mosaic,
    cleanup_scratch,
)

logger = logging.getLogger(__name__)


@dataclass
class MosaicWSCleanJobConfig:
    """Configuration for WSClean mosaic jobs.

    Attributes
    ----------
    database_path : Path
        Path to the unified database
    mosaic_dir : Path
        Directory for output mosaics
    scratch_dir : Path
        Directory for MS copies (preserves originals)
    ms_table : str
        Name of the MS index table (default: ms_index)
    """

    database_path: Path
    mosaic_dir: Path
    scratch_dir: Path = Path("/dev/shm/mosaic")
    ms_table: str = "ms_index"


@register_job
@dataclass
class MosaicMSPlanningJob(Job):
    """Select MS files for mosaicking based on time range.

    Unlike the legacy MosaicPlanningJob which queries images,
    this job queries the ms_index table for MS files.

    Inputs:
        - start_time, end_time (Unix timestamps)
        - tier (quicklook/science/deep)
        - mosaic_name (unique identifier)
        - calibrator_name (optional, for auto-detection of transit MS)
        - calibrator_ra_deg, calibrator_dec_deg (optional, for phase centering)

    Outputs:
        - plan_id (database row)
        - ms_paths (list of MS file paths)
        - n_ms_files (count)
        - calibrator_ms_idx (index of transit MS for calibration)
        - mean_ra_deg (computed mean meridian)
        - dec_deg (declination)
    """

    job_type: str = "mosaic_ms_planning"

    start_time: int = 0
    end_time: int = 0
    tier: str = "science"
    mosaic_name: str = ""
    calibrator_name: str | None = None
    calibrator_ra_deg: float | None = None
    calibrator_dec_deg: float | None = None
    config: MosaicWSCleanJobConfig | None = None

    def validate(self) -> tuple[bool, str | None]:
        """Validate job parameters."""
        if not self.config:
            return False, "No configuration provided"
        if not self.mosaic_name:
            return False, "No mosaic name provided"
        if self.start_time >= self.end_time:
            return False, "start_time must be before end_time"
        return True, None

    def execute(self) -> JobResult:
        """Execute the MS planning job."""
        import numpy as np

        from .schema import ensure_mosaic_tables
        from .tiers import TIER_CONFIGS, MosaicTier

        # Validate first
        is_valid, error = self.validate()
        if not is_valid:
            return JobResult.fail(error or "Validation failed")

        logger.info(
            f"Planning WSClean mosaic '{self.mosaic_name}' "
            f"(tier={self.tier}, range={self.start_time}-{self.end_time})"
        )

        try:
            tier_enum = MosaicTier(self.tier)
            tier_config = TIER_CONFIGS[tier_enum]
        except (ValueError, KeyError):
            return JobResult.fail(f"Invalid tier: {self.tier}")

        try:
            conn = sqlite3.connect(str(self.config.database_path))
            conn.row_factory = sqlite3.Row

            # Ensure mosaic tables exist
            ensure_mosaic_tables(conn)

            # Query MS files in time range
            # Filter by status = 'calibrated' or 'imaged' (processed MS files)
            query = f"""
                SELECT path, mid_mjd, ra_deg, dec_deg, status
                FROM {self.config.ms_table}
                WHERE mid_mjd BETWEEN ? AND ?
                  AND status IN ('calibrated', 'imaged', 'converted')
                ORDER BY mid_mjd ASC
            """

            # Convert Unix timestamps to MJD
            start_mjd = Time(self.start_time, format="unix").mjd
            end_mjd = Time(self.end_time, format="unix").mjd

            cursor = conn.execute(query, (start_mjd, end_mjd))
            ms_rows = cursor.fetchall()

            if len(ms_rows) == 0:
                conn.close()
                return JobResult.fail("No MS files found in time range")

            ms_paths = [row["path"] for row in ms_rows]
            ra_values = [row["ra_deg"] for row in ms_rows if row["ra_deg"] is not None]
            dec_values = [row["dec_deg"] for row in ms_rows if row["dec_deg"] is not None]

            if not ra_values or not dec_values:
                conn.close()
                return JobResult.fail("MS files missing RA/Dec coordinates")

            # Compute mean meridian
            mean_ra = float(np.mean(ra_values))
            dec = float(np.mean(dec_values))

            # Find calibrator transit MS (closest to calibrator RA if provided)
            calibrator_ms_idx = None
            if self.calibrator_ra_deg is not None:
                # Find MS closest to calibrator RA
                ra_diffs = [abs(r - self.calibrator_ra_deg) for r in ra_values]
                calibrator_ms_idx = int(np.argmin(ra_diffs))
                logger.info(
                    f"Selected MS {calibrator_ms_idx} for calibration "
                    f"(RA offset: {ra_diffs[calibrator_ms_idx]:.4f}°)"
                )

            # Calculate coverage
            coverage = {
                "ra_min_deg": min(ra_values),
                "ra_max_deg": max(ra_values),
                "dec_min_deg": min(dec_values),
                "dec_max_deg": max(dec_values),
            }

            # Insert plan into database
            cursor = conn.execute(
                """
                INSERT INTO mosaic_plans
                    (name, tier, start_time, end_time, image_ids, n_images,
                     ra_min_deg, ra_max_deg, dec_min_deg, dec_max_deg,
                     created_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                """,
                (
                    self.mosaic_name,
                    self.tier,
                    self.start_time,
                    self.end_time,
                    json.dumps(ms_paths),  # Store MS paths instead of image IDs
                    len(ms_paths),
                    coverage["ra_min_deg"],
                    coverage["ra_max_deg"],
                    coverage["dec_min_deg"],
                    coverage["dec_max_deg"],
                    int(time.time()),
                ),
            )
            plan_id = cursor.lastrowid
            conn.commit()
            conn.close()

            logger.info(f"Created MS plan {plan_id} with {len(ms_paths)} MS files")

            return JobResult.ok(
                outputs={
                    "plan_id": plan_id,
                    "ms_paths": ms_paths,
                    "n_ms_files": len(ms_paths),
                    "calibrator_ms_idx": calibrator_ms_idx,
                    "mean_ra_deg": mean_ra,
                    "dec_deg": dec,
                    "calibrator_name": self.calibrator_name,
                    "calibrator_ra_deg": self.calibrator_ra_deg,
                    "calibrator_dec_deg": self.calibrator_dec_deg,
                },
                message=f"Selected {len(ms_paths)} MS files for {self.tier} mosaic",
            )

        except Exception as e:
            logger.exception(f"MS planning job failed: {e}")
            return JobResult.fail(str(e))


@register_job
@dataclass
class MosaicWSCleanBuildJob(Job):
    """Build mosaic using WSClean joint deconvolution.

    This replaces the legacy MosaicBuildJob which used image-domain
    "linear mosaicking" with scipy.

    Inputs:
        - plan_id (from planning job)

    Outputs:
        - mosaic_id (database row)
        - mosaic_path (FITS file path - PB corrected)
        - flat_noise_path (FITS file path - flat noise)
    """

    job_type: str = "mosaic_wsclean_build"

    plan_id: int = 0
    config: MosaicWSCleanJobConfig | None = None

    def validate(self) -> tuple[bool, str | None]:
        """Validate job parameters."""
        if not self.config:
            return False, "No configuration provided"
        if not self.plan_id:
            return False, "No plan_id provided"
        return True, None

    def execute(self) -> JobResult:
        """Execute the WSClean build job."""
        from .schema import ensure_mosaic_tables
        from .tiers import TIER_CONFIGS, MosaicTier

        # Validate first
        is_valid, error = self.validate()
        if not is_valid:
            return JobResult.fail(error or "Validation failed")

        logger.info(f"Building WSClean mosaic for plan {self.plan_id}")

        try:
            conn = sqlite3.connect(str(self.config.database_path))
            conn.row_factory = sqlite3.Row

            # Ensure mosaic tables exist
            ensure_mosaic_tables(conn)

            # Get plan details
            cursor = conn.execute("SELECT * FROM mosaic_plans WHERE id = ?", (self.plan_id,))
            plan = cursor.fetchone()

            if not plan:
                conn.close()
                return JobResult.fail(f"Plan {self.plan_id} not found")

            # Update status to building
            conn.execute(
                "UPDATE mosaic_plans SET status = 'building' WHERE id = ?", (self.plan_id,)
            )
            conn.commit()

            # Get MS paths from plan (stored as JSON)
            ms_paths = [Path(p) for p in json.loads(plan["image_ids"])]

            # Get tier configuration
            tier_config = TIER_CONFIGS[MosaicTier(plan["tier"])]

            # Configure WSClean
            wsclean_config = WSCleanMosaicConfig(
                scratch_dir=self.config.scratch_dir,
                output_dir=self.config.mosaic_dir,
                size=4096,
                scale="1asec",
                niter=50000 if tier_config.timeout_minutes >= 30 else 10000,
                mgain=0.6,
                auto_threshold=3.0,
                idg_mode="cpu",
                parallel_deconvolution=2000,
                local_rms=True,
            )

            # Build mosaic
            try:
                result = build_wsclean_mosaic(
                    ms_paths=ms_paths,
                    output_name=plan["name"],
                    config=wsclean_config,
                )
            except Exception as e:
                conn.execute(
                    "UPDATE mosaic_plans SET status = 'failed' WHERE id = ?", (self.plan_id,)
                )
                conn.commit()
                conn.close()
                return JobResult.fail(f"WSClean mosaic build failed: {e}")

            # Compute approximate coverage (from plan)
            coverage_sq_deg = (
                (plan["ra_max_deg"] - plan["ra_min_deg"])
                * (plan["dec_max_deg"] - plan["dec_min_deg"])
                * abs(math.cos(math.radians(plan["dec_min_deg"])))
            )

            # Register mosaic in database
            cursor = conn.execute(
                """
                INSERT INTO mosaics
                    (plan_id, path, tier, n_images, median_rms_jy,
                     coverage_sq_deg, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.plan_id,
                    str(result.pb_corrected_path),
                    plan["tier"],
                    len(ms_paths),
                    result.median_rms_jy,
                    coverage_sq_deg,
                    int(time.time()),
                ),
            )
            mosaic_id = cursor.lastrowid

            conn.execute(
                "UPDATE mosaic_plans SET status = 'completed' WHERE id = ?", (self.plan_id,)
            )
            conn.commit()
            conn.close()

            logger.info(f"Built WSClean mosaic {mosaic_id}: {result.pb_corrected_path}")

            # Cleanup scratch copies
            cleanup_scratch(wsclean_config.scratch_dir, plan["name"])

            return JobResult.ok(
                outputs={
                    "mosaic_id": mosaic_id,
                    "mosaic_path": str(result.pb_corrected_path),
                    "flat_noise_path": str(result.output_path),
                    "n_ms_files": result.n_ms_files,
                    "phase_center_ra_deg": result.phase_center_ra_deg,
                    "phase_center_dec_deg": result.phase_center_dec_deg,
                },
                message=f"Built WSClean mosaic: {result.pb_corrected_path}",
            )

        except Exception as e:
            logger.exception(f"WSClean build job failed: {e}")
            return JobResult.fail(str(e))
