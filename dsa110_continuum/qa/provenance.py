"""Pipeline provenance and run manifest.

Records calibration quality, per-tile status, and per-epoch results
into a single JSON manifest alongside pipeline products. When a mosaic
looks bad, open the manifest to immediately see what went wrong.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RunManifest:
    """Accumulates provenance during a pipeline run and serializes to JSON."""

    # Identity
    git_sha: str = ""
    started_at: str = ""
    finished_at: str | None = None
    wall_time_sec: float | None = None
    command_line: list[str] = field(default_factory=list)
    hostname: str = ""

    # Inputs
    date: str = ""
    cal_date: str = ""
    bp_table: str = ""
    g_table: str = ""
    epoch_g_table: str | None = None
    ms_files: list[str] = field(default_factory=list)

    # Calibration quality (from compute_calibration_metrics)
    cal_quality: dict[str, Any] = field(default_factory=dict)

    # Per-tile records
    tiles: list[dict[str, Any]] = field(default_factory=list)

    # Per-epoch records
    epochs: list[dict[str, Any]] = field(default_factory=list)

    # Overall
    gaincal_status: str = ""

    @classmethod
    def start(
        cls,
        date: str,
        cal_date: str,
        argv: list[str] | None = None,
    ) -> RunManifest:
        """Create a manifest capturing initial run metadata."""
        git_sha = ""
        try:
            git_sha = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        except Exception:
            pass

        return cls(
            git_sha=git_sha,
            started_at=datetime.now(timezone.utc).isoformat(),
            command_line=list(argv or sys.argv),
            hostname=socket.gethostname(),
            date=date,
            cal_date=cal_date,
        )

    def assess_cal_quality(self, bp_path: str, g_path: str) -> None:
        """Compute and store calibration quality metrics.

        Calls ``compute_calibration_metrics`` from
        ``dsa110_continuum.calibration.qa`` and logs warnings for poor
        quality indicators.
        """
        from dsa110_continuum.calibration.qa import compute_calibration_metrics

        self.bp_table = bp_path
        self.g_table = g_path

        for label, path in [("bp", bp_path), ("g", g_path)]:
            try:
                metrics = compute_calibration_metrics(path)
                d = metrics.to_dict()
                self.cal_quality[label] = d

                if metrics.extraction_error:
                    logger.warning(
                        "Cal QA %s: extraction error — %s",
                        label.upper(),
                        metrics.extraction_error,
                    )
                    continue

                if metrics.flag_fraction > 0.3:
                    logger.warning(
                        "Cal QA %s: high flag fraction %.1f%%",
                        label.upper(),
                        metrics.flag_fraction * 100,
                    )
                if metrics.phase_scatter_deg > 30.0:
                    logger.warning(
                        "Cal QA %s: high phase scatter %.1f°",
                        label.upper(),
                        metrics.phase_scatter_deg,
                    )
            except Exception as exc:
                self.cal_quality[label] = {"error": str(exc)}
                logger.warning("Cal QA %s failed: %s", label.upper(), exc)

    def record_tile(
        self,
        ms_path: str,
        fits_path: str | None,
        status: str,
        elapsed_sec: float,
        error: str | None = None,
    ) -> None:
        """Record the outcome of a single tile."""
        rec: dict[str, Any] = {
            "ms_path": ms_path,
            "fits_path": fits_path,
            "status": status,
            "elapsed_sec": round(elapsed_sec, 1),
        }
        if error is not None:
            rec["error"] = error
        self.tiles.append(rec)

    def record_epoch(
        self,
        hour: int,
        epoch_result: dict[str, Any],
        epoch_qa: Any | None = None,
    ) -> None:
        """Record the outcome of an epoch mosaic."""
        rec: dict[str, Any] = {
            "hour": hour,
            "n_tiles": epoch_result.get("n_tiles"),
            "status": epoch_result.get("status"),
            "mosaic_path": epoch_result.get("mosaic_path"),
            "peak": epoch_result.get("peak"),
            "rms": epoch_result.get("rms"),
            "n_sources": epoch_result.get("n_sources"),
            "median_ratio": epoch_result.get("median_ratio"),
            "gaincal_status": epoch_result.get("gaincal_status"),
            "qa_result": epoch_result.get("qa_result"),
        }
        if epoch_qa is not None:
            try:
                rec["rms_mjy"] = epoch_qa.mosaic_rms_mjy
                rec["completeness_frac"] = epoch_qa.completeness_frac
            except AttributeError:
                pass
        self.epochs.append(rec)

    def finalize(self, wall_time_sec: float) -> None:
        """Mark the run as finished."""
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.wall_time_sec = round(wall_time_sec, 1)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return asdict(self)

    def save(self, output_dir: str) -> str:
        """Write manifest JSON to *output_dir*/{date}_manifest.json."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{self.date}_manifest.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info("Manifest written: %s", path)
        return path
