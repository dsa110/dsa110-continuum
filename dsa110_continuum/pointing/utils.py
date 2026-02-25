"""Utilities for reading pointing information from MS and UVH5 files."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.time import Time

# Ensure CASAPATH is set before importing CASA modules
from dsa110_contimg.common.utils.casa_init import ensure_casa_path

ensure_casa_path()

import casacore.tables as casatables  # noqa: E402

table = casatables.table  # noqa: N816

from dsa110_contimg.core.calibration.schedule import DSA110_LOCATION  # noqa: E402

logger = logging.getLogger(__name__)


def _time_from_seconds(seconds: np.ndarray | None) -> Time | None:
    """Convert seconds to astropy Time object with automatic format detection.

    Automatically detects whether seconds are relative to MJD 0 or MJD 51544.0
    by validating the resulting date. This handles both formats:
    - Seconds since MJD 0 (pyuvdata format)
    - Seconds since MJD 51544.0 (CASA standard)

    Parameters
    ----------
    seconds : array-like or None
        Time in seconds (format auto-detected)
    seconds: Optional[np.ndarray] :


    """
    if seconds is None or len(seconds) == 0:
        return None
    from dsa110_contimg.common.utils.time_utils import (
        DEFAULT_YEAR_RANGE,
        detect_casa_time_format,
    )

    time_sec = float(np.mean(seconds))
    _, mjd = detect_casa_time_format(time_sec, DEFAULT_YEAR_RANGE)
    return Time(mjd, format="mjd", scale="utc")


def load_pointing(path: str | Path, field_id: int | None = None) -> dict[str, Any]:
    """Return pointing info for an MS or UVH5 file.

    Parameters
    ----------
    path : str or Path
        Measurement Set ``*.ms`` directory or UVH5 ``*.hdf5`` file.
    field_id : int, optional
        When reading an MS, select this FIELD_ID; defaults to the FIELD with
        the largest number of rows.
    path: str | Path :

    field_id: Optional[int] :
         (Default value = None)

    Returns
    -------
    dict
        Dictionary containing pointing information with keys:
        - source_type: 'ms' or 'uvh5'
        - ra_deg: Right ascension in degrees
        - dec_deg: Declination in degrees
        - mid_time: Observation mid-time as Time object
        - fields: List of field information (MS only)
        - selected_field_id: Selected field ID
        - ms_path: Path to MS (if applicable)


    """
    # Input validation
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    # Validate field_id if provided
    if field_id is not None and (not isinstance(field_id, int) or field_id < 0):
        raise ValueError(f"field_id must be a non-negative integer, got: {field_id}")

    info: dict[str, Any] = {
        "input_path": str(path),
        "source_type": None,
        "ra_deg": None,
        "dec_deg": None,
        "mid_time": None,
        "fields": None,
        "selected_field_id": None,
        "ms_path": None,
    }

    if path.suffix == ".ms" or path.name.endswith(".ms"):
        if not path.is_dir():
            raise ValueError(f"MS path must be a directory: {path}")
        info["source_type"] = "ms"
        info["ms_path"] = str(path)

        try:
            with table(str(path) + "::FIELD") as tf:
                phase_dir = tf.getcol("PHASE_DIR")
                if phase_dir.ndim < 3:
                    raise ValueError("PHASE_DIR has unexpected shape")
                ra_list = np.degrees(phase_dir[:, 0, 0])
                dec_list = np.degrees(phase_dir[:, 0, 1])

            with table(str(path)) as tb:
                field_ids = tb.getcol("FIELD_ID")
                times = tb.getcol("TIME")

            unique_ids = np.unique(field_ids)
            fields: list[dict[str, Any]] = []
            for fid in unique_ids:
                idx = np.where(field_ids == fid)[0]
                if int(fid) >= len(ra_list):
                    logger.warning("Field %s index out of range for PHASE_DIR", fid)
                    continue
                fields.append(
                    {
                        "field_id": int(fid),
                        "rows": int(idx.size),
                        "ra_deg": float(ra_list[int(fid)]),
                        "dec_deg": float(dec_list[int(fid)]),
                    }
                )
            info["fields"] = fields

            if not fields:
                raise RuntimeError("No valid fields found in MS")

            if field_id is None:
                fid = max(fields, key=lambda x: x["rows"])["field_id"]
            else:
                fid = field_id
                if fid not in [f["field_id"] for f in fields]:
                    available = [f["field_id"] for f in fields]
                    raise ValueError(
                        f"Field {fid} not present in MS {path}. Available fields: {available}"
                    )

            info["selected_field_id"] = int(fid)
            info["ra_deg"] = float(ra_list[int(fid)])
            info["dec_deg"] = float(dec_list[int(fid)])
            info["mid_time"] = _time_from_seconds(times[field_ids == fid])
            return info

        except Exception as e:
            logger.error("Failed to read MS %s: %s", path, e)
            raise RuntimeError(f"Error reading MS {path}: {e}") from e

    if path.suffix == ".hdf5" and path.exists():
        if not path.is_file():
            raise ValueError(f"UVH5 path must be a file: {path}")
        info["source_type"] = "uvh5"

        try:
            from dsa110_contimg.common.utils.hdf5_io import open_uvh5_metadata

            with open_uvh5_metadata(path) as f:
                header = f.get("Header")
                if header is None:
                    raise ValueError("No Header group found in UVH5 file")

                time_arr = np.asarray(header["time_array"]) if "time_array" in header else None
                info["mid_time"] = _time_from_seconds(time_arr)

                dec_val = None
                ha_val = None
                if "extra_keywords" in header:
                    ek = header["extra_keywords"]
                    if "phase_center_dec" in ek:
                        dec_val = float(np.asarray(ek["phase_center_dec"]))
                    if "ha_phase_center" in ek:
                        ha_val = float(np.asarray(ek["ha_phase_center"]))

                if dec_val is not None:
                    info["dec_deg"] = np.degrees(dec_val)
                else:
                    logger.warning("No phase_center_dec found in UVH5 extra_keywords")

                if info["mid_time"] is not None and ha_val is not None:
                    lst = info["mid_time"].sidereal_time("apparent", longitude=DSA110_LOCATION.lon)
                    ra = (lst - ha_val * u.rad).wrap_at(360 * u.deg)
                    info["ra_deg"] = float(ra.deg)
                else:
                    logger.warning("Cannot compute RA: missing mid_time or ha_phase_center")

            return info

        except Exception as e:
            logger.error("Failed to read UVH5 %s: %s", path, e)
            raise RuntimeError(f"Error reading UVH5 {path}: {e}") from e

    raise ValueError(f"Unsupported file format: {path}. Expected .ms directory or .hdf5 file")


# =============================================================================
# Shared Pointing Detection Functions
# =============================================================================
# These functions are used by both:
# - Streaming pipeline (PointingTracker) - real-time change detection
# - CLI discovery commands - historical analysis of observations
#
# The algorithms are aligned to ensure consistent behavior across both use cases.
# =============================================================================

# Shared constants - same defaults for streaming and CLI
DEFAULT_DEC_CHANGE_THRESHOLD = 0.1  # degrees - minimum change to detect
DEFAULT_TOLERANCE = 0.1  # degrees - for grouping into runs
DEFAULT_STEP_SIZE = 100  # files (~8 hours at 5 min/file)


@dataclass
class DecRun:
    """A contiguous run of observations at one declination.

    Used by both CLI discovery and streaming pipeline for representing
    periods where the telescope pointed at a consistent declination.

    """

    dec_deg: float
    start_file: Path
    end_file: Path
    file_count: int = 0

    @property
    def start_time(self) -> str:
        """Extract timestamp from start file name."""
        return self.start_file.stem.split("_sb")[0]

    @property
    def end_time(self) -> str:
        """Extract timestamp from end file name."""
        return self.end_file.stem.split("_sb")[0]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "dec_deg": self.dec_deg,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "start_file": str(self.start_file),
            "end_file": str(self.end_file),
            "file_count": self.file_count,
        }


def read_uvh5_dec_fast(path: Path) -> float | None:
    """Read declination from UVH5 file metadata (fast path).

    Uses direct h5py access to avoid loading full UVData object.
    This is the canonical implementation used by both streaming and CLI.

    Parameters
    ----------
    path :
        Path to UVH5/HDF5 file

    Returns
    -------
        Declination in degrees, or None if not found

    """
    try:
        import h5py

        with h5py.File(path, "r") as f:
            # phase_center_dec is stored in radians in extra_keywords
            if "Header/extra_keywords/phase_center_dec" in f:
                dec_rad = f["Header/extra_keywords/phase_center_dec"][()]
                return float(np.degrees(dec_rad))
            # Fallback: try direct phase_center_dec key
            if "Header/phase_center_dec" in f:
                dec_rad = f["Header/phase_center_dec"][()]
                return float(np.degrees(dec_rad))
    except Exception as e:
        logger.debug(f"Could not read Dec from {path}: {e}")
    return None


def detect_dec_change(
    new_dec: float,
    current_dec: float | None,
    threshold: float = DEFAULT_DEC_CHANGE_THRESHOLD,
) -> bool:
    """Detect if a declination change is significant.

    This is the fundamental change detection used by both streaming
    and CLI modes.

    Parameters
    ----------
    new_dec :
        New declination in degrees
    current_dec :
        Current/previous declination in degrees (None if first)
    threshold :
        Minimum change to consider significant

    Returns
    -------
        True if change is significant, False otherwise

    """
    if current_dec is None:
        return True  # First observation is always a "change"
    return abs(new_dec - current_dec) >= threshold


def find_dec_runs_fine(
    files: list[Path],
    tolerance: float = DEFAULT_TOLERANCE,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[DecRun]:
    """Find declination runs by checking every file (finer-sampling mode).

    This uses the same logic as the streaming PointingTracker,
    guaranteeing no runs are missed regardless of length.

    Complexity: O(N) file reads

    Parameters
    ----------
    files :
        Sorted list of HDF5 file paths
    tolerance :
        Max declination difference to consider same pointing
    progress_callback :
        Optional callback(files_read, total_files)
    files: List[Path] :

    Returns
    -------
        List of DecRun objects

    """
    if not files:
        return []

    runs: list[DecRun] = []
    current_dec: float | None = None
    run_start_idx: int = 0
    files_read = 0

    for i, f in enumerate(files):
        files_read += 1
        if progress_callback and files_read % 100 == 0:
            progress_callback(files_read, len(files))

        dec = read_uvh5_dec_fast(f)
        if dec is None:
            continue

        if detect_dec_change(dec, current_dec, tolerance):
            # Record previous run if exists
            if current_dec is not None and i > run_start_idx:
                runs.append(
                    DecRun(
                        dec_deg=round(current_dec, 1),
                        start_file=files[run_start_idx],
                        end_file=files[i - 1],
                        file_count=i - run_start_idx,
                    )
                )
            # Start new run
            current_dec = dec
            run_start_idx = i

    # Record final run
    if current_dec is not None:
        runs.append(
            DecRun(
                dec_deg=round(current_dec, 1),
                start_file=files[run_start_idx],
                end_file=files[-1],
                file_count=len(files) - run_start_idx,
            )
        )

    if progress_callback:
        progress_callback(len(files), len(files))

    return runs


def find_dec_runs_fast(
    files: list[Path],
    tolerance: float = DEFAULT_TOLERANCE,
    step_size: int = DEFAULT_STEP_SIZE,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[DecRun]:
    """Find declination runs using sampling with binary search (fast mode).

        Probes every `step_size` files and uses binary search to find exact
        change boundaries. Much faster than finer-sampling mode but may miss runs
        shorter than `step_size` files.

    Complexity: O(N/step + k*log(step)) file reads, where k = number of changes

    Parameters
    ----------
    files : list of Path
        Sorted list of HDF5 file paths
    tolerance : float
        Max declination difference to consider same pointing
    step_size : int
        Files to skip between probes (trades accuracy for speed)
    progress_callback : callable or None
        Optional callback(files_read, total_files)

    Returns
    -------
        list of DecRun
        List of DecRun objects

    Notes
    -----
        Runs shorter than `step_size` files may be missed if they occur
        between two runs of the same declination. Use finer-sampling mode for
        guaranteed accuracy.
    """
    if not files:
        return []

    runs: list[DecRun] = []
    files_read = 0

    def track_read(filepath: Path) -> float | None:
        nonlocal files_read
        files_read += 1
        if progress_callback:
            progress_callback(files_read, len(files))
        return read_uvh5_dec_fast(filepath)

    def find_change_point(left: int, right: int, left_dec: float) -> int:
        """Binary search for where declination changes.

        Parameters
        ----------
        """
        while right - left > 1:
            mid = (left + right) // 2
            mid_dec = track_read(files[mid])
            if mid_dec is None or abs(mid_dec - left_dec) < tolerance:
                left = mid
            else:
                right = mid
        return right

    # Start with first file
    i = 0
    current_dec = track_read(files[0])
    if current_dec is None:
        i = 1
        while i < len(files) and (current_dec := track_read(files[i])) is None:
            i += 1

    if current_dec is None:
        return []

    run_start = i

    while i < len(files):
        # Probe ahead by step_size
        probe = min(i + step_size, len(files) - 1)
        probe_dec = track_read(files[probe])

        if probe_dec is not None and abs(probe_dec - current_dec) >= tolerance:
            # Dec changed somewhere between i and probe - binary search
            change_idx = find_change_point(i, probe, current_dec)
            runs.append(
                DecRun(
                    dec_deg=round(current_dec, 1),
                    start_file=files[run_start],
                    end_file=files[change_idx - 1],
                    file_count=change_idx - run_start,
                )
            )
            run_start = change_idx
            current_dec = track_read(files[change_idx])
            if current_dec is None:
                # Skip bad files
                while (
                    run_start < len(files) and (current_dec := track_read(files[run_start])) is None
                ):
                    run_start += 1
                if current_dec is None:
                    break
            i = change_idx
        else:
            # No change in this window, jump ahead
            i = probe + 1

    # Record final run
    if current_dec is not None:
        runs.append(
            DecRun(
                dec_deg=round(current_dec, 1),
                start_file=files[run_start],
                end_file=files[-1],
                file_count=len(files) - run_start,
            )
        )

    return runs


def find_dec_runs(
    files: list[Path],
    tolerance: float = DEFAULT_TOLERANCE,
    step_size: int | None = DEFAULT_STEP_SIZE,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[DecRun]:
    """Find declination runs in a file list.

    This is the unified entry point that selects the appropriate algorithm:
    - If step_size is None: finer-sampling mode (checks every file, O(N))
    - If step_size is set: fast mode (sampling + binary search, O(N/step))

    Parameters
    ----------
    files :
        Sorted list of HDF5 file paths
    tolerance :
        Max declination difference to consider same pointing
    step_size :
        If None, use finer-sampling mode. Otherwise, sample every N files.
    progress_callback :
        Optional callback(files_read, total_files)
    files: List[Path] :

    Returns
    -------
        List of DecRun objects

    """
    if step_size is None:
        return find_dec_runs_fine(files, tolerance, progress_callback)
    else:
        return find_dec_runs_fast(files, tolerance, step_size, progress_callback)


def iter_dec_changes(
    files: Iterator[Path],
    tolerance: float = DEFAULT_TOLERANCE,
) -> Iterator[tuple[Path, float, bool]]:
    """Iterate over files, yielding declination and change status.

    This provides streaming-compatible change detection, yielding
    (file, dec, is_change) tuples as files are processed.

    Parameters
    ----------
    files :
        Iterator of HDF5 file paths (can be infinite stream)
    tolerance :
        Max declination difference to consider same pointing
    files: Iterator[Path] :

    """
    current_dec: float | None = None

    for f in files:
        dec = read_uvh5_dec_fast(f)
        if dec is None:
            continue

        is_change = detect_dec_change(dec, current_dec, tolerance)
        if is_change:
            current_dec = dec

        yield f, dec, is_change
