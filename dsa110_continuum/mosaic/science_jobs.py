"""
Jobs for the rigorous Science Mosaic workflow.

These jobs bridge the legacy Pipeline/Job architecture with the new
Dagster-based science workflow.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass

from dsa110_contimg.workflow.pipeline import Job, JobResult, register_job
from dsa110_contimg.core.mosaic.jobs import MosaicJobConfig
from dsa110_contimg.workflow.dagster.jobs.science_mosaic import science_mosaic_workflow

logger = logging.getLogger(__name__)


@register_job
@dataclass
class SciencePlanningJob(Job):
    """Create a mosaic plan for the Science workflow.

    Unlike MosaicPlanningJob, this does NOT query existing images,
    as the Science workflow generates new images from MS data.
    """

    job_type: str = "science_planning"

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
        from dsa110_contimg.core.mosaic.schema import ensure_mosaic_tables

        is_valid, error = self.validate()
        if not is_valid:
            return JobResult.fail(error or "Validation failed")

        logger.info(f"Planning Science mosaic '{self.mosaic_name}'")

        try:
            with contextlib.closing(sqlite3.connect(str(self.config.database_path))) as conn:
                conn.row_factory = sqlite3.Row

                ensure_mosaic_tables(conn)

                # Insert plan with empty image_ids
                # The workflow will populate n_images later
                cursor = conn.execute(
                    """
                    INSERT INTO mosaic_plans
                        (name, tier, start_time, end_time, image_ids, n_images,
                         created_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
                    """,
                    (
                        self.mosaic_name,
                        self.tier,
                        self.start_time,
                        self.end_time,
                        json.dumps([]),  # No images yet
                        0,
                        int(time.time()),
                    ),
                )
                plan_id = cursor.lastrowid
                conn.commit()

            logger.info(f"Created pending Science plan {plan_id}")

            return JobResult.ok(
                outputs={
                    "plan_id": plan_id,
                    "image_ids": [],
                    "n_images": 0,
                },
                message="Created pending Science plan",
            )

        except Exception as e:
            logger.exception(f"Science planning failed: {e}")
            return JobResult.fail(str(e))


@register_job
@dataclass
class ScienceMosaicBridgeJob(Job):
    """Bridge job to execute the Dagster Science Mosaic workflow."""

    job_type: str = "science_mosaic_bridge"

    plan_id: int = 0
    config: MosaicJobConfig | None = None

    # Need access to start/end time/name to configure the Dagster job
    # Passed from pipeline params
    start_time: int = 0
    end_time: int = 0
    mosaic_name: str = ""

    def validate(self) -> tuple[bool, str | None]:
        """Validate job parameters."""
        if not self.config:
            return False, "No configuration provided"
        if not self.plan_id:
            return False, "No plan_id provided"
        return True, None

    def execute(self) -> JobResult:
        """Execute the Dagster workflow."""
        from datetime import datetime, timezone

        is_valid, error = self.validate()
        if not is_valid:
            return JobResult.fail(error or "Validation failed")

        logger.info(f"Executing Science Mosaic workflow for plan {self.plan_id}")

        # Update status to building
        self._update_status("building")

        try:
            # Convert timestamps to ISO8601
            start_iso = datetime.fromtimestamp(self.start_time, tz=timezone.utc).isoformat()
            end_iso = datetime.fromtimestamp(self.end_time, tz=timezone.utc).isoformat()

            # Construct op config
            op_config = {
                "config": {
                    "start_time": start_iso,
                    "end_time": end_iso,
                    "mosaic_name": self.mosaic_name,
                    "database_path": str(self.config.database_path),
                    "images_table": self.config.images_table,
                    "calibrator_name": "0834+555",  # Hardcoded for now, or could be param
                }
            }

            # Simplify run_config construction to avoid repetition
            dagster_ops = [
                "find_and_convert_data",
                "select_transit_tile",
                "solve_transit_calibration",
                "image_transit",
                "image_others",
                "create_science_mosaic",
                "register_mosaic_result",
                "apply_calibration",
                "image_vis",
                "apply_calibration_with_solve_result",
            ]

            # We need to ensure the resource config matches the environment
            # `DSA110PipelineResource` expects config that matches `get_pipeline_resource_config()` in definitions.py
            # But here we are constructing it manually.
            from dsa110_contimg.common.utils.paths import resolve_paths
            paths = resolve_paths()

            run_config = {
                "ops": {op_name: op_config for op_name in dagster_ops},
                "resources": {
                    "dsa110_pipeline": {
                        "config": {
                            "input_dir": str(paths.input_dir),  # Default
                            "output_dir": str(paths.ms_dir),  # Output of conversion is MS
                            "scratch_dir": str(paths.tmpfs_dir),
                            "state_dir": str(paths.state_dir),
                        }
                    }
                }
            }

            # Execute in process
            result = science_mosaic_workflow.execute_in_process(run_config=run_config)

            if not result.success:
                raise RuntimeError("Dagster execution failed")

            # Result registration handled by register_mosaic_result op
            # But we should verify.

            # We can check database or output
            # For JobResult, we want to return what MosaicBuildJob returns

            mosaic = None
            with contextlib.closing(sqlite3.connect(str(self.config.database_path))) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM mosaics WHERE plan_id = ?", (self.plan_id,))
                mosaic = cursor.fetchone()

            if mosaic:
                return JobResult.ok(
                    outputs={
                        "mosaic_id": mosaic["id"],
                        "mosaic_path": mosaic["path"],
                        "n_images": mosaic["n_images"],
                        "median_rms": mosaic["median_rms_jy"],
                    },
                    message=f"Built Science mosaic: {mosaic['path']}",
                )
            else:
                # Op might have failed to register
                raise RuntimeError("Mosaic created but not registered in database")

        except Exception as e:
            logger.exception(f"Science mosaic execution failed: {e}")
            self._update_status("failed")
            return JobResult.fail(str(e))

    def _update_status(self, status: str) -> None:
        """Update plan status in database."""
        try:
            with contextlib.closing(sqlite3.connect(str(self.config.database_path))) as conn:
                conn.execute(
                    "UPDATE mosaic_plans SET status = ? WHERE id = ?", (status, self.plan_id)
                )
                conn.commit()
        except Exception:
            pass
