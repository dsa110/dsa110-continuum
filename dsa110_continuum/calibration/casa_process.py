"""
CASA Process Executor.

This module provides a mechanism to run CASA tasks in isolated processes.
This eliminates race conditions related to the Current Working Directory (CWD),
which CASA uses for logging (casalog).

By running each task in a fresh process with its own temporary CWD, we achieve:
1. True parallelism (no GIL, no global CWD lock needed).
2. Clean log isolation.
3. Crash resilience (if CASA segfaults, it doesn't kill the main pipeline).
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _casa_worker(
    task_name: str,
    kwargs: dict[str, Any],
    result_queue: multiprocessing.Queue,
    temp_dir: str,
) -> None:
    """
    Worker function to run a CASA task in a separate process.

    Parameters
    ----------
    task_name : str
        Name of the CASA task to run.
    kwargs : dict
        Arguments for the task.
    result_queue : multiprocessing.Queue
        Queue to send the result or exception back to the parent.
    temp_dir : str
        Path to the temporary directory to use as CWD.
    """
    try:
        # Change CWD to the temporary directory to isolate casalog
        os.chdir(temp_dir)

        # Import casatasks here, inside the process
        # This ensures a fresh initialization if possible, though mostly
        # we rely on the process boundary for isolation.
        try:
            import casatasks
        except ImportError:
            result_queue.put(("error", (ImportError("Could not import casatasks"), None)))
            return

        task_func = getattr(casatasks, task_name, None)
        if task_func is None:
            result_queue.put(
                ("error", (AttributeError(f"Task '{task_name}' not found in casatasks"), None))
            )
            return

        # Run the task
        result = task_func(**kwargs)
        result_queue.put(("success", result))

    except Exception as e:
        # Capture full traceback
        tb = traceback.format_exc()
        result_queue.put(("error", (e, tb)))


class CASAProcessExecutor:
    """Executes CASA tasks in isolated processes."""

    def __init__(self, timeout: float = 3600.0):
        """
        Initialize the executor.

        Parameters
        ----------
        timeout : float
            Maximum time (in seconds) to wait for a task to complete.
            Default is 1 hour.
        """
        self.timeout = timeout

    def run_task(self, task_name: str, **kwargs) -> Any:
        """
        Run a CASA task in a separate process.

        Parameters
        ----------
        task_name : str
            Name of the CASA task.
        **kwargs :
            Arguments for the task.

        Returns
        -------
        Any
            The return value of the CASA task.

        Raises
        ------
        RuntimeError
            If the task fails or times out.
        """
        # Create a temporary directory for this process execution
        # We use a context manager to ensure it's cleaned up
        with tempfile.TemporaryDirectory(prefix=f"casa_proc_{task_name}_") as temp_dir:
            result_queue = multiprocessing.Queue()

            # Replaced ProcessGuard with manual process management to support parallelism.
            # ProcessGuard kills all children/process group, which breaks concurrent tasks.
            process = multiprocessing.Process(
                target=_casa_worker,
                args=(task_name, kwargs, result_queue, temp_dir),
                daemon=True,  # Daemon processes are killed if parent dies
            )

            process.start()

            try:
                # Wait for result
                try:
                    status, payload = result_queue.get(timeout=self.timeout)
                except multiprocessing.queues.Empty:
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                    raise RuntimeError(f"CASA task '{task_name}' timed out after {self.timeout}s")

                process.join(timeout=5)

                if status == "success":
                    # Copy log file back if needed?
                    # For now, we assume users check the task result.
                    # If we want to preserve casalog, we should copy it from temp_dir
                    # to the main log dir.
                    self._preserve_log(temp_dir, task_name)
                    return payload
                else:
                    # Error case
                    exception, tb = payload
                    logger.error(f"CASA task '{task_name}' failed in worker process:\n{tb}")
                    raise exception

            except Exception as e:
                # Ensure process is killed on any other error (e.g. KeyboardInterrupt)
                if process.is_alive():
                    process.terminate()
                    process.join()
                raise e
            finally:
                # Always ensure the specific child process is dead
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1)

    def _preserve_log(self, temp_dir: str, task_name: str) -> None:
        """
        Copy the casalog from the temp dir to the centralized log directory.

        Adds unique process ID to log filename to prevent collisions when
        multiple CASA tasks run simultaneously.

        Example: casa-20260121-120000.log -> casa-20260121-120000-pid12345.log
        """
        try:
            temp_path = Path(temp_dir)
            log_files = list(temp_path.glob("casalog*.log"))
            if not log_files:
                return

            # Target directory
            # We try to use the standard location if available
            # Or just let it be deleted with the temp dir if we don't care (but we usually care)

            # For now, let's copy to dsa110_contimg logs if defined
            # We can import this dynamically to avoid circular imports
            from dsa110_contimg.common.utils.casa_init import CASA_LOG_DIR

            target_dir = Path(CASA_LOG_DIR)
            target_dir.mkdir(parents=True, exist_ok=True)

            # Add unique process ID to prevent filename collisions
            pid = os.getpid()

            for log_file in log_files:
                # Insert PID before file extension to ensure uniqueness
                # casa-20260121-120000.log -> casa-20260121-120000-pid12345.log
                stem = log_file.stem
                suffix = log_file.suffix
                new_name = f"{stem}-pid{pid}{suffix}"
                target_path = target_dir / new_name

                shutil.copy2(log_file, target_path)
                logger.debug(f"Preserved CASA log: {task_name} -> {new_name}")

        except Exception as e:
            logger.warning(f"Failed to preserve CASA log from worker: {e}")
