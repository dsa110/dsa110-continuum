"""Simple background worker to execute queued batch photometry jobs."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

from dsa110_contimg.interfaces.api.job_adapters import run_batch_photometry_job
from dsa110_contimg.infrastructure.database import ensure_pipeline_db

logger = logging.getLogger(__name__)


class PhotometryBatchWorker:
    """Polls the products DB for pending batch photometry jobs and executes them."""

    def __init__(
        self, products_db_path: Path, poll_interval: float = 10.0, max_workers: int | None = None
    ):
        self.products_db_path = products_db_path
        self.poll_interval = poll_interval
        self.max_workers = max_workers
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_status_log = 0.0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run_loop, name="photometry-batch-worker", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def _claim_next_job(self) -> tuple[int, list[str], list[dict[str, float]], dict] | None:
        conn = ensure_pipeline_db()
        conn.row_factory = sqlite3.Row  # type: ignore[attr-defined]
        try:
            row = conn.execute(
                """
                SELECT id, params
                FROM batch_jobs
                WHERE status = 'pending' AND type = 'batch_photometry'
                ORDER BY created_at
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None

            # Attempt to claim the job so only one worker processes it
            updated = conn.execute(
                "UPDATE batch_jobs SET status = 'running' WHERE id = ? AND status = 'pending'",
                (row["id"],),
            )
            conn.commit()
            if updated.rowcount == 0:
                return None

            params = {}
            if row["params"]:
                try:
                    params = json.loads(row["params"])
                except (json.JSONDecodeError, TypeError):
                    params = {}

            items = conn.execute(
                "SELECT ms_path FROM batch_job_items WHERE batch_id = ?",
                (row["id"],),
            ).fetchall()

            fits_paths = []
            coordinates: list[dict[str, float]] = []
            coord_keys = set()
            for item in items:
                parts = item["ms_path"].split(":")
                if len(parts) < 3:
                    continue
                try:
                    ra = float(parts[-2])
                    dec = float(parts[-1])
                    coord_key = (ra, dec)
                    if coord_key not in coord_keys:
                        coord_keys.add(coord_key)
                        coordinates.append({"ra_deg": ra, "dec_deg": dec})
                    fits_path = ":".join(parts[:-2])
                    if fits_path not in fits_paths:
                        fits_paths.append(fits_path)
                except ValueError:
                    continue

            return row["id"], fits_paths, coordinates, params
        finally:
            conn.close()

    def status(self) -> dict[str, int]:
        """Return counts of photometry batch jobs by status for observability."""
        conn = ensure_pipeline_db()
        try:
            counts = conn.execute(
                """
                SELECT status, COUNT(*) as cnt
                FROM batch_jobs
                WHERE type = 'batch_photometry'
                GROUP BY status
                """
            ).fetchall()
            return {row[0]: row[1] for row in counts}
        except sqlite3.Error:
            logger.debug("Failed to fetch photometry batch status", exc_info=True)
            return {}
        finally:
            conn.close()

    def _log_status_periodically(self) -> None:
        now = time.time()
        if now - self._last_status_log < max(self.poll_interval, 30.0):
            return
        self._last_status_log = now
        summary = self.status()
        logger.info("Photometry batch worker status", extra={"summary": summary})

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self._log_status_periodically()
            job = self._claim_next_job()
            if not job:
                logger.debug("Photometry worker idle - no pending jobs")
                time.sleep(self.poll_interval)
                continue

            batch_id, fits_paths, coordinates, params = job
            if self.max_workers and "max_workers" not in params:
                params["max_workers"] = self.max_workers

            logger.info(
                "Starting batch photometry job",
                extra={
                    "batch_id": batch_id,
                    "fits": len(fits_paths),
                    "coords": len(coordinates),
                    "params": params,
                },
            )
            start_time = time.time()
            try:
                run_batch_photometry_job(
                    batch_id, fits_paths, coordinates, params, self.products_db_path
                )
            except (OSError, ValueError, RuntimeError):
                logger.exception(f"Batch photometry job {batch_id} failed")
            duration = time.time() - start_time
            logger.info(
                "Finished batch photometry job",
                extra={
                    "batch_id": batch_id,
                    "duration_sec": round(duration, 2),
                    "status": self.status(),
                },
            )
