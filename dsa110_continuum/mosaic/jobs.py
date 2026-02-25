"""
Dagster job implementations for mosaicking pipeline.

Three jobs:
1. MosaicPlanningJob - Query images, select tier, validate
2. MosaicBuildJob - Run linear mosaicking, combine, write FITS
3. MosaicQAJob - Astrometry, photometry, artifact detection

These jobs inherit from the generic pipeline.Job base class.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from dsa110_contimg.workflow.pipeline import Job, JobResult, register_job

logger = logging.getLogger(__name__)


@dataclass
class MosaicJobConfig:
    """Configuration for mosaic jobs.

    Attributes
    ----------
    database_path : Path
        Path to the unified database
    mosaic_dir : Path
        Directory for output mosaics
    images_table : str
        Name of the images table
    """

    database_path: Path
    mosaic_dir: Path
    images_table: str = "images"


@register_job
@dataclass
class MosaicPlanningJob(Job):
    """Select images for mosaicking based on time range and tier.

    Inputs:
        - start_time, end_time (Unix timestamps)
        - tier (quicklook/science/deep)
        - mosaic_name (unique identifier)

    Outputs:
        - plan_id (database row)
        - image_ids (list of selected images)
        - n_images (count)
    """

    job_type: str = "mosaic_planning"

    start_time: int = 0
    end_time: int = 0
    tier: str = "science"
    mosaic_name: str = ""
    config: MosaicJobConfig | None = None

    def validate(self) -> tuple[bool, str | None]:
        """Validate job parameters."""
        if not self.config:
            return False, "No configuration provided"
        if not self.mosaic_name:
            return False, "No mosaic name provided"
        return True, None

    def execute(self) -> JobResult:
        """Execute the planning job."""
        from .schema import ensure_mosaic_tables
        from .tiers import TIER_CONFIGS, MosaicTier

        # Validate first
        is_valid, error = self.validate()
        if not is_valid:
            return JobResult.fail(error or "Validation failed")

        logger.info(
            f"Planning mosaic '{self.mosaic_name}' "
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

            # Query images in time range with quality filter
            # Note: No LIMIT - all images in time range are included
            # (coverage is determined by time range, not artificial caps)
            query = f"""
                SELECT id, path, noise_jy, center_ra_deg, center_dec_deg
                FROM {self.config.images_table}
                WHERE created_at BETWEEN ? AND ?
                  AND noise_jy < ?
                  AND noise_jy IS NOT NULL
                ORDER BY noise_jy ASC
            """

            cursor = conn.execute(
                query,
                (
                    self.start_time,
                    self.end_time,
                    tier_config.rms_threshold_jy,
                ),
            )
            images = cursor.fetchall()

            if len(images) == 0:
                conn.close()
                return JobResult.fail("No images found in time range")

            # Calculate coverage statistics
            ra_values = [img["center_ra_deg"] for img in images if img["center_ra_deg"] is not None]
            dec_values = [
                img["center_dec_deg"] for img in images if img["center_dec_deg"] is not None
            ]

            coverage = {
                "ra_min_deg": min(ra_values) if ra_values else None,
                "ra_max_deg": max(ra_values) if ra_values else None,
                "dec_min_deg": min(dec_values) if dec_values else None,
                "dec_max_deg": max(dec_values) if dec_values else None,
            }

            # Insert plan into database
            image_ids = [img["id"] for img in images]

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
                    json.dumps(image_ids),
                    len(images),
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

            logger.info(f"Created plan {plan_id} with {len(images)} images")

            return JobResult.ok(
                outputs={
                    "plan_id": plan_id,
                    "image_ids": image_ids,
                    "n_images": len(images),
                },
                message=f"Selected {len(images)} images for {self.tier} mosaic",
            )

        except Exception as e:
            logger.exception(f"Planning job failed: {e}")
            return JobResult.fail(str(e))


@register_job
@dataclass
class MosaicBuildJob(Job):
    """Build mosaic from planned images using linear mosaicking.

    Inputs:
        - plan_id (from planning job)

    Outputs:
        - mosaic_id (database row)
        - mosaic_path (FITS file path)
    """

    job_type: str = "mosaic_build"

    plan_id: int = 0
    config: MosaicJobConfig | None = None

    def validate(self) -> tuple[bool, str | None]:
        """Validate job parameters."""
        if not self.config:
            return False, "No configuration provided"
        if not self.plan_id:
            return False, "No plan_id provided"
        return True, None

    def execute(self) -> JobResult:
        """Execute the build job."""
        from .builder import build_mosaic
        from .schema import ensure_mosaic_tables
        from .tiers import TIER_CONFIGS, MosaicTier

        # Validate first
        is_valid, error = self.validate()
        if not is_valid:
            return JobResult.fail(error or "Validation failed")

        logger.info(f"Building mosaic for plan {self.plan_id}")

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

            # Get image paths
            image_ids = json.loads(plan["image_ids"])
            placeholders = ",".join("?" * len(image_ids))
            cursor = conn.execute(
                f"SELECT path FROM {self.config.images_table} WHERE id IN ({placeholders})",
                tuple(image_ids),
            )
            images = cursor.fetchall()
            image_paths = [Path(img["path"]) for img in images]

            # Get tier configuration
            tier_config = TIER_CONFIGS[MosaicTier(plan["tier"])]

            # Build output path
            self.config.mosaic_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.config.mosaic_dir / f"{plan['name']}.fits"

            # Build mosaic
            try:
                result = build_mosaic(
                    image_paths=image_paths,
                    output_path=output_path,
                    alignment_order=tier_config.alignment_order,
                    timeout_minutes=tier_config.timeout_minutes,
                )
            except Exception as e:
                conn.execute(
                    "UPDATE mosaic_plans SET status = 'failed' WHERE id = ?", (self.plan_id,)
                )
                conn.commit()
                conn.close()
                return JobResult.fail(f"Mosaic build failed: {e}")

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
                    str(output_path),
                    plan["tier"],
                    len(image_paths),
                    result.median_rms,
                    result.coverage_sq_deg,
                    int(time.time()),
                ),
            )
            mosaic_id = cursor.lastrowid

            conn.execute(
                "UPDATE mosaic_plans SET status = 'completed' WHERE id = ?", (self.plan_id,)
            )
            conn.commit()
            conn.close()

            logger.info(f"Built mosaic {mosaic_id}: {output_path}")

            return JobResult.ok(
                outputs={
                    "mosaic_id": mosaic_id,
                    "mosaic_path": str(output_path),
                    "n_images": result.n_images,
                    "median_rms": result.median_rms,
                },
                message=f"Built mosaic: {output_path}",
            )

        except Exception as e:
            logger.exception(f"Build job failed: {e}")
            return JobResult.fail(str(e))


@register_job
@dataclass
class MosaicQAJob(Job):
    """Run quality checks on completed mosaic.

    Inputs:
        - mosaic_id (from build job)

    Outputs:
        - qa_status (PASS/WARN/FAIL)
        - qa_metrics (dict)
    """

    job_type: str = "mosaic_qa"

    mosaic_id: int = 0
    config: MosaicJobConfig | None = None

    def validate(self) -> tuple[bool, str | None]:
        """Validate job parameters."""
        if not self.config:
            return False, "No configuration provided"
        if not self.mosaic_id:
            return False, "No mosaic_id provided"
        return True, None

    def execute(self) -> JobResult:
        """Execute the QA job."""
        from .qa import run_qa_checks
        from .schema import ensure_mosaic_tables

        # Validate first
        is_valid, error = self.validate()
        if not is_valid:
            return JobResult.fail(error or "Validation failed")

        logger.info(f"Running QA for mosaic {self.mosaic_id}")

        try:
            conn = sqlite3.connect(str(self.config.database_path))
            conn.row_factory = sqlite3.Row

            # Ensure mosaic tables exist
            ensure_mosaic_tables(conn)

            # Get mosaic details
            cursor = conn.execute("SELECT * FROM mosaics WHERE id = ?", (self.mosaic_id,))
            mosaic = cursor.fetchone()

            if not mosaic:
                conn.close()
                return JobResult.fail(f"Mosaic {self.mosaic_id} not found")

            # Run QA checks
            qa_result = run_qa_checks(
                mosaic_path=Path(mosaic["path"]),
                tier=mosaic["tier"],
            )

            # Determine overall status
            qa_status = qa_result.status

            # Store QA results
            conn.execute(
                """
                INSERT INTO mosaic_qa
                    (mosaic_id, astrometry_rms_arcsec, n_reference_stars,
                     median_noise_jy, dynamic_range, has_artifacts,
                     artifact_score, passed, warnings, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.mosaic_id,
                    qa_result.astrometry_rms,
                    qa_result.n_stars,
                    qa_result.median_noise,
                    qa_result.dynamic_range,
                    int(qa_result.has_artifacts),
                    qa_result.artifact_score,
                    int(qa_result.passed),
                    json.dumps(qa_result.warnings),
                    int(time.time()),
                ),
            )

            # Update mosaic record
            conn.execute(
                """
                UPDATE mosaics
                SET qa_status = ?, qa_details = ?
                WHERE id = ?
                """,
                (qa_status, json.dumps(qa_result.to_dict()), self.mosaic_id),
            )
            conn.commit()
            conn.close()

            logger.info(f"QA complete for mosaic {self.mosaic_id}: {qa_status}")

            return JobResult.ok(
                outputs={
                    "qa_status": qa_status,
                    "qa_metrics": qa_result.to_dict(),
                },
                message=f"QA complete: {qa_status}",
            )

        except Exception as e:
            logger.exception(f"QA job failed: {e}")
            return JobResult.fail(str(e))
