"""
CASA Service Adapter.

This module provides a unified service for interacting with CASA tasks
in a safe, logged environment. It handles:
1. Lazy importing of CASA tasks to prevent premature log file creation
2. Log redirection to the dedicated CASA logs directory
3. Typed interfaces for common CASA tasks
4. Command logging for debugging

Usage:
    from dsa110_contimg.core.calibration.casa_service import CASAService

    service = CASAService()

    # Run gaincal
    service.gaincal(
        vis="my.ms",
        caltable="my.G",
        ...
    )
"""

import logging
import os
from contextlib import contextmanager
from typing import Any

import numpy as np

from dsa110_contimg.common.utils.casa_init import (
    casa_log_environment,
    ensure_casa_path,
    get_casa_task,
)

# Ensure environment is set up on module import
ensure_casa_path()

logger = logging.getLogger(__name__)


class CalibrationProtectionError(Exception):
    """Raised when an operation would destroy applied calibration."""

    pass


def _detect_applied_calibration(ms_path: str, threshold: float = 0.01) -> bool:
    """Detect if CORRECTED_DATA contains applied calibration.

    Compares CORRECTED_DATA to DATA and returns True if they differ
    by more than the threshold (indicating calibration has been applied).

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set
    threshold : float, default=0.01
        Relative difference threshold. If median(|CORRECTED - DATA| / |DATA|)
        exceeds this value, calibration is considered applied.

    Returns
    -------
    bool
        True if CORRECTED_DATA appears to contain applied calibration
    """
    try:
        import casacore.tables as ct

        with ct.table(ms_path, readonly=True) as t:
            colnames = set(t.colnames())

            # If no CORRECTED_DATA, no calibration to protect
            if "CORRECTED_DATA" not in colnames or "DATA" not in colnames:
                return False

            # Sample a subset of rows for efficiency
            nrows = t.nrows()
            sample_size = min(10000, nrows)
            step = max(1, nrows // sample_size)

            data = t.getcol("DATA", startrow=0, nrow=sample_size, rowincr=step)
            corrected = t.getcol("CORRECTED_DATA", startrow=0, nrow=sample_size, rowincr=step)

        # Check if they're identical (no calibration applied)
        if np.allclose(data, corrected, rtol=1e-6):
            logger.debug(f"CORRECTED_DATA == DATA in {ms_path}, no calibration detected")
            return False

        # Calculate relative difference
        data_amp = np.abs(data)
        diff_amp = np.abs(corrected - data)

        # Avoid division by zero
        valid = data_amp > 1e-10
        if not valid.any():
            return False

        rel_diff = np.median(diff_amp[valid] / data_amp[valid])

        if rel_diff > threshold:
            logger.info(
                f"Calibration detected in {ms_path}: "
                f"median relative difference = {rel_diff:.2%}"
            )
            return True

        logger.debug(
            f"CORRECTED_DATA ~ DATA in {ms_path}: "
            f"median relative difference = {rel_diff:.2%} < {threshold:.2%}"
        )
        return False

    except Exception as e:
        logger.debug(f"Could not check for calibration in {ms_path}: {e}")
        # If we can't check, assume no calibration to be safe
        return False


class CASAService:
    """Service for interacting with CASA tasks in a safe, logged environment."""

    def __init__(self, use_process_isolation: bool | None = None):
        """Initialize the CASA service.

        Parameters
        ----------
        use_process_isolation : bool, optional
            Whether to run CASA tasks in isolated processes.
            If None, checks DSA110_CASA_PROCESS_ISOLATION env var.
        """
        if use_process_isolation is None:
            self.use_process_isolation = (
                os.environ.get("DSA110_CASA_PROCESS_ISOLATION", "false").lower() == "true"
            )
        else:
            self.use_process_isolation = use_process_isolation

        if self.use_process_isolation:
            from dsa110_contimg.core.calibration.casa_process import CASAProcessExecutor

            self.process_executor = CASAProcessExecutor()
            logger.info("CASAService initialized with process isolation ENABLED")
        else:
            logger.debug("CASAService initialized with process isolation DISABLED")

    @contextmanager
    def log_environment(self):
        """Context manager for CASA operations that need log directory.

        This delegates to the utility function but provides it through the service.
        """
        with casa_log_environment() as log_dir:
            yield log_dir

    def run_task(self, task_name: str, **kwargs) -> Any:
        """Run a CASA task with log redirection.

        Parameters
        ----------
        task_name : str
            Name of the CASA task (e.g., "gaincal", "bandpass")
        **kwargs :
            Arguments to pass to the task

        Returns
        -------
        Any
            Result of the CASA task execution
        """
        # Build command string for logging
        cmd_str = self.build_command_string(task_name, kwargs)

        if self.use_process_isolation:
            logger.info(f"Running CASA task (isolated): {cmd_str}")
            return self.process_executor.run_task(task_name, **kwargs)

        # Legacy thread-based execution
        task = get_casa_task(task_name)
        if not task:
            raise ImportError(f"CASA task '{task_name}' could not be imported.")

        logger.info(f"Running CASA task: {cmd_str}")

        # Run task in protected logging environment
        with casa_log_environment():
            return task(**kwargs)

    def gaincal(self, **kwargs) -> Any:
        """Run gaincal task."""
        return self.run_task("gaincal", **kwargs)

    def bandpass(self, **kwargs) -> Any:
        """Run bandpass task."""
        return self.run_task("bandpass", **kwargs)

    def smoothcal(self, **kwargs) -> Any:
        """Run smoothcal task."""
        return self.run_task("smoothcal", **kwargs)

    def setjy(self, **kwargs) -> Any:
        """Run setjy task."""
        return self.run_task("setjy", **kwargs)

    def fluxscale(self, **kwargs) -> Any:
        """Run fluxscale task."""
        return self.run_task("fluxscale", **kwargs)

    def applycal(self, **kwargs) -> Any:
        """Run applycal task."""
        return self.run_task("applycal", **kwargs)

    def flagdata(self, **kwargs) -> Any:
        """Run flagdata task."""
        return self.run_task("flagdata", **kwargs)

    def tclean(self, **kwargs) -> Any:
        """Run tclean task."""
        return self.run_task("tclean", **kwargs)

    def ft(self, **kwargs) -> Any:
        """Run ft task."""
        return self.run_task("ft", **kwargs)

    def flagmanager(self, **kwargs) -> Any:
        """Run flagmanager task."""
        return self.run_task("flagmanager", **kwargs)

    def concat(self, **kwargs) -> Any:
        """Run concat task."""
        return self.run_task("concat", **kwargs)

    def initweights(self, **kwargs) -> Any:
        """Run initweights task."""
        return self.run_task("initweights", **kwargs)

    def phaseshift(self, **kwargs) -> Any:
        """Run phaseshift task."""
        return self.run_task("phaseshift", **kwargs)

    def gencal(self, **kwargs) -> Any:
        """Run gencal task."""
        return self.run_task("gencal", **kwargs)

    def clearcal(
        self,
        *,
        protect_calibration: bool = True,
        **kwargs,
    ) -> Any:
        """Run clearcal task with optional calibration protection.

        WARNING: clearcal resets CORRECTED_DATA to DATA, destroying any
        applied calibration! Use protect_calibration=True (default) to
        prevent accidental removal of calibration.

        Parameters
        ----------
        protect_calibration : bool, default=True
            If True, check whether CORRECTED_DATA contains applied calibration
            (differs from DATA by more than 1%) before running clearcal.
            Raises CalibrationProtectionError if calibration would be destroyed.
            Set to False only if you explicitly want to clear calibration.
        **kwargs
            Arguments passed to CASA clearcal task (vis, field, spw, etc.)

        Raises
        ------
        CalibrationProtectionError
            If protect_calibration=True and CORRECTED_DATA contains valid
            calibration that would be destroyed.
        """
        vis = kwargs.get("vis")
        if protect_calibration and vis:
            if _detect_applied_calibration(vis):
                raise CalibrationProtectionError(
                    f"CORRECTED_DATA in {vis} contains applied calibration. "
                    f"clearcal would destroy this calibration! "
                    f"If you really want to clear calibration, use protect_calibration=False."
                )
        return self.run_task("clearcal", **kwargs)

    def exportfits(self, **kwargs) -> Any:
        """Run exportfits task."""
        return self.run_task("exportfits", **kwargs)

    def plotbandpass(self, **kwargs) -> Any:
        """Run plotbandpass task."""
        return self.run_task("plotbandpass", **kwargs)

    def plotcal(self, **kwargs) -> Any:
        """Run plotcal task."""
        return self.run_task("plotcal", **kwargs)

    def split(self, **kwargs) -> Any:
        """Run split task."""
        return self.run_task("split", **kwargs)

    def mstransform(self, **kwargs) -> Any:
        """Run mstransform task."""
        return self.run_task("mstransform", **kwargs)

    def build_command_string(self, task_name: str, kwargs: dict[str, Any]) -> str:
        """Build a human-readable command string for logging.

        Parameters
        ----------
        task_name : str
            CASA task name
        kwargs : dict
            Task arguments

        Returns
        -------
        str
            Formatted command string
        """
        # Filter out None values
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}

        # Format parameters
        params = []
        for key, value in sorted(filtered_kwargs.items()):
            if isinstance(value, str):
                params.append(f"{key}='{value}'")
            elif isinstance(value, (list, tuple)):
                params.append(f"{key}={list(value)}")
            else:
                params.append(f"{key}={value}")

        return f"{task_name}({', '.join(params)})"

    def get_version(self) -> str | None:
        """Get CASA version string.

        Returns
        -------
        Optional[str]
            CASA version string (e.g., "6.7.2"), or None if unavailable
        """
        try:
            import casatools  # type: ignore

            if hasattr(casatools, "version"):
                version = casatools.version()
                if isinstance(version, str):
                    return version
                elif isinstance(version, (list, tuple)):
                    return ".".join(str(v) for v in version)
                else:
                    return str(version)

            # Fallback
            import casatasks  # type: ignore

            if hasattr(casatasks, "version"):
                version = casatasks.version()
                if isinstance(version, str):
                    return version
                elif isinstance(version, (list, tuple)):
                    return ".".join(str(v) for v in version)
                else:
                    return str(version)
        except ImportError:
            pass

        return None
