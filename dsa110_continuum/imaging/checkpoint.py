"""
Checkpointing support for long-running imaging operations.

Provides resume capability for WSClean and tclean imaging runs that may
take hours to complete. If a crash occurs at hour 4 of a 5-hour run,
the checkpoint allows resuming from the last major cycle.

Key Features:
- Track imaging progress in database
- Save intermediate state to disk
- Resume from last checkpoint on restart
- Automatic cleanup of stale checkpoints

Usage:
    from dsa110_contimg.core.imaging.checkpoint import ImagingCheckpoint

    checkpoint = ImagingCheckpoint(ms_path, output_dir)

    # Check for existing progress
    if checkpoint.can_resume():
        resume_args = checkpoint.get_resume_args()
        # Continue from last checkpoint
    else:
        # Start fresh

    # During imaging
    for iteration in range(n_iter):
        # Do work...
        if iteration % checkpoint_interval == 0:
            checkpoint.save(
                iteration=iteration,
                peak_residual=peak,
                model_path=model_file,
            )

    # On completion
    checkpoint.complete()
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Checkpoint save interval (save every N iterations)
DEFAULT_CHECKPOINT_INTERVAL = 500

# Stale checkpoint threshold (hours)
STALE_CHECKPOINT_HOURS = 24


@dataclass
class CheckpointState:
    """State of an imaging checkpoint."""

    ms_path: str
    output_dir: str
    started_at: float
    updated_at: float
    iteration: int = 0
    total_iterations: int = 0
    peak_residual: float | None = None
    model_path: str | None = None
    psf_path: str | None = None
    residual_path: str | None = None
    backend: str = "wsclean"
    status: str = "in_progress"  # in_progress, completed, failed
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointState:
        """Create from dictionary."""
        # Handle missing fields for backward compatibility
        data.setdefault("metadata", {})
        data.setdefault("status", "in_progress")
        data.setdefault("error_message", None)
        return cls(**data)


class ImagingCheckpoint:
    """
    Checkpoint manager for long-running imaging operations.

    Saves progress to both filesystem and database for durability.
    Filesystem checkpoint allows resume even if database is unavailable.

    Attributes
    ----------
        ms_path: Path to measurement set being imaged
        output_dir: Directory for imaging outputs
        checkpoint_file: Path to JSON checkpoint file
    """

    def __init__(
        self,
        ms_path: str | Path,
        output_dir: str | Path,
        backend: str = "wsclean",
        checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    ):
        """
        Initialize checkpoint manager.

        Parameters
        ----------
        ms_path : str or Path
            Path to measurement set being imaged
        output_dir : str or Path
            Directory for imaging outputs
        backend : str
            Imaging backend ('wsclean' or 'tclean')
        checkpoint_interval : int
            Save checkpoint every N iterations
        """
        self.ms_path = Path(ms_path)
        self.output_dir = Path(output_dir)
        self.backend = backend
        self.checkpoint_interval = checkpoint_interval

        # Checkpoint file path
        self.checkpoint_file = self.output_dir / ".imaging_checkpoint.json"

        # Current state
        self._state: CheckpointState | None = None

    def initialize(self, total_iterations: int, **metadata: Any) -> None:
        """
        Initialize a new checkpoint for fresh imaging run.

        Parameters
        ----------
        total_iterations : int
            Total iterations planned
        **metadata : Any
            Additional metadata to store (cell_size, imsize, etc.)
        """
        now = time.time()
        self._state = CheckpointState(
            ms_path=str(self.ms_path),
            output_dir=str(self.output_dir),
            started_at=now,
            updated_at=now,
            iteration=0,
            total_iterations=total_iterations,
            backend=self.backend,
            metadata=metadata,
        )
        self._save_to_file()
        self._save_to_database()
        logger.info(
            "Initialized imaging checkpoint for %s (%d iterations)",
            self.ms_path.name,
            total_iterations,
        )

    def can_resume(self) -> bool:
        """
        Check if there's a valid checkpoint to resume from.

        Returns
        -------
        bool
            True if resumable checkpoint exists
        """
        if not self.checkpoint_file.exists():
            return False

        try:
            state = self._load_from_file()
            if state is None:
                return False

            # Check if checkpoint is stale
            hours_old = (time.time() - state.updated_at) / 3600
            if hours_old > STALE_CHECKPOINT_HOURS:
                logger.warning(
                    "Checkpoint is stale (%.1f hours old), starting fresh",
                    hours_old,
                )
                return False

            # Check if status allows resume
            if state.status != "in_progress":
                return False

            # Verify model files still exist
            if state.model_path and not Path(state.model_path).exists():
                logger.warning(
                    "Model file missing (%s), cannot resume", state.model_path
                )
                return False

            self._state = state
            return True

        except Exception as e:
            logger.warning("Could not load checkpoint: %s", e)
            return False

    def get_resume_args(self) -> dict[str, Any]:
        """
        Get arguments needed to resume imaging from checkpoint.

        Returns
        -------
        dict
            Arguments for imaging backend's continue/resume mode
        """
        if self._state is None:
            raise ValueError("No checkpoint loaded - call can_resume() first")

        if self.backend == "wsclean":
            return self._get_wsclean_resume_args()
        elif self.backend == "tclean":
            return self._get_tclean_resume_args()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _get_wsclean_resume_args(self) -> dict[str, Any]:
        """Get WSClean-specific resume arguments."""
        return {
            "continue": True,
            "start_iteration": self._state.iteration,
            "model_prefix": self._state.model_path,
            # WSClean reads existing model.fits and continues
        }

    def _get_tclean_resume_args(self) -> dict[str, Any]:
        """Get tclean-specific resume arguments."""
        return {
            "calcres": False,  # Don't recalculate residuals
            "calcpsf": False,  # Don't recalculate PSF
            "restart": True,  # Resume from model
            "niter": self._state.total_iterations - self._state.iteration,
        }

    def save(
        self,
        iteration: int,
        peak_residual: float | None = None,
        model_path: str | None = None,
        psf_path: str | None = None,
        residual_path: str | None = None,
        **metadata: Any,
    ) -> None:
        """
        Save checkpoint at current iteration.

        Parameters
        ----------
        iteration : int
            Current iteration number
        peak_residual : float, optional
            Current peak residual value
        model_path : str, optional
            Path to current model image
        psf_path : str, optional
            Path to PSF image
        residual_path : str, optional
            Path to residual image
        **metadata : Any
            Additional metadata to update
        """
        if self._state is None:
            raise ValueError("Checkpoint not initialized - call initialize() first")

        # Don't save every iteration, only at intervals
        if iteration % self.checkpoint_interval != 0:
            return

        self._state.iteration = iteration
        self._state.updated_at = time.time()
        if peak_residual is not None:
            self._state.peak_residual = peak_residual
        if model_path is not None:
            self._state.model_path = str(model_path)
        if psf_path is not None:
            self._state.psf_path = str(psf_path)
        if residual_path is not None:
            self._state.residual_path = str(residual_path)
        self._state.metadata.update(metadata)

        self._save_to_file()
        self._save_to_database()

        elapsed = self._state.updated_at - self._state.started_at
        progress = iteration / max(self._state.total_iterations, 1) * 100
        logger.debug(
            "Checkpoint saved: iteration %d/%d (%.1f%%), elapsed %.1fs",
            iteration,
            self._state.total_iterations,
            progress,
            elapsed,
        )

    def complete(self) -> None:
        """Mark imaging as completed and clean up checkpoint."""
        if self._state is None:
            return

        self._state.status = "completed"
        self._state.updated_at = time.time()
        self._save_to_database()

        # Remove checkpoint file on successful completion
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()

        elapsed = self._state.updated_at - self._state.started_at
        logger.info(
            "Imaging completed for %s: %d iterations in %.1f seconds",
            self.ms_path.name,
            self._state.iteration,
            elapsed,
        )

    def fail(self, error_message: str) -> None:
        """Mark imaging as failed with error message."""
        if self._state is None:
            return

        self._state.status = "failed"
        self._state.error_message = error_message
        self._state.updated_at = time.time()
        self._save_to_file()  # Keep checkpoint for debugging
        self._save_to_database()

        logger.error(
            "Imaging failed for %s at iteration %d: %s",
            self.ms_path.name,
            self._state.iteration,
            error_message,
        )

    def _save_to_file(self) -> None:
        """Save checkpoint to JSON file."""
        if self._state is None:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Write to temp file first, then rename for atomicity
        temp_file = self.checkpoint_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(self._state.to_dict(), f, indent=2)
        shutil.move(temp_file, self.checkpoint_file)

    def _load_from_file(self) -> CheckpointState | None:
        """Load checkpoint from JSON file."""
        if not self.checkpoint_file.exists():
            return None

        try:
            with open(self.checkpoint_file) as f:
                data = json.load(f)
            return CheckpointState.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Could not parse checkpoint file: %s", e)
            return None

    def _save_to_database(self) -> None:
        """Save checkpoint state to pipeline database."""
        if self._state is None:
            return

        try:
            from dsa110_contimg.infrastructure.database.unified import get_db

            db = get_db()

            # Upsert into imaging_checkpoints table
            db.execute(
                """
                INSERT INTO imaging_checkpoints (
                    ms_path, output_dir, started_at, updated_at,
                    iteration, total_iterations, peak_residual,
                    model_path, backend, status, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ms_path) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    iteration = excluded.iteration,
                    peak_residual = excluded.peak_residual,
                    model_path = excluded.model_path,
                    status = excluded.status,
                    error_message = excluded.error_message
                """,
                (
                    str(self._state.ms_path),
                    str(self._state.output_dir),
                    self._state.started_at,
                    self._state.updated_at,
                    self._state.iteration,
                    self._state.total_iterations,
                    self._state.peak_residual,
                    self._state.model_path,
                    self._state.backend,
                    self._state.status,
                    self._state.error_message,
                ),
            )
        except Exception as e:
            # Don't fail imaging just because DB write failed
            logger.warning("Could not save checkpoint to database: %s", e)


def get_stale_checkpoints(max_age_hours: float = STALE_CHECKPOINT_HOURS) -> list[dict]:
    """
    Get list of stale (abandoned) checkpoints.

    Returns
    -------
    list of dict
        Stale checkpoint records from database
    """
    try:
        from dsa110_contimg.infrastructure.database.unified import get_db

        db = get_db()

        cutoff = time.time() - (max_age_hours * 3600)
        rows = db.query(
            """
            SELECT * FROM imaging_checkpoints
            WHERE status = 'in_progress' AND updated_at < ?
            ORDER BY updated_at ASC
            """,
            (cutoff,),
        )
        return rows
    except Exception as e:
        logger.warning("Could not query stale checkpoints: %s", e)
        return []


def cleanup_stale_checkpoints(max_age_hours: float = STALE_CHECKPOINT_HOURS) -> int:
    """
    Clean up stale checkpoints by marking them as failed.

    Returns
    -------
    int
        Number of checkpoints cleaned up
    """
    stale = get_stale_checkpoints(max_age_hours)
    if not stale:
        return 0

    try:
        from dsa110_contimg.infrastructure.database.unified import get_db

        db = get_db()

        for record in stale:
            db.execute(
                """
                UPDATE imaging_checkpoints
                SET status = 'failed', error_message = 'Stale checkpoint - no progress in ? hours'
                WHERE ms_path = ?
                """,
                (max_age_hours, record["ms_path"]),
            )

        logger.info("Cleaned up %d stale checkpoints", len(stale))
        return len(stale)
    except Exception as e:
        logger.warning("Could not clean up stale checkpoints: %s", e)
        return 0
