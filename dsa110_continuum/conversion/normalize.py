"""
Subband filename normalization utilities.

This module provides functions for normalizing subband filenames to ensure
all subbands in a group share the same canonical timestamp. This eliminates
the need for fuzzy time-based clustering in queries.

The Problem:
    The correlator writes subband files with slightly different timestamps:
    - 2025-01-15T12:00:00_sb00.hdf5
    - 2025-01-15T12:00:01_sb01.hdf5  (1 second later)
    - 2025-01-15T12:00:00_sb02.hdf5
    - 2025-01-15T12:00:02_sb03.hdf5  (2 seconds later)

The Solution:
    When a subband arrives, if it clusters with an existing group, rename the
    file to use the canonical group_id:
    - 2025-01-15T12:00:00_sb00.hdf5  (first arrival = canonical)
    - 2025-01-15T12:00:00_sb01.hdf5  (renamed from T12:00:01)
    - 2025-01-15T12:00:00_sb02.hdf5  (already matches)
    - 2025-01-15T12:00:00_sb03.hdf5  (renamed from T12:00:02)

Benefits:
    - Exact matching: GROUP BY group_id just works
    - Self-documenting: Filesystem shows true group membership
    - Simpler queries: No fuzzy time-window clustering needed
    - Idempotent: Re-running normalizer is safe

Used by:
    - Dagster ingestion assets for normalize-group operations
    - Batch normalization of historical data
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dsa110_contimg.infrastructure.database.hdf5_index import (
    parse_subband_filename,
)

logger = logging.getLogger(__name__)

# Default tolerance for grouping subbands (seconds)
DEFAULT_CLUSTER_TOLERANCE_S = 120.0


def parse_subband_info(path: Path) -> tuple[str, int] | None:
    """Extract (group_id, subband_idx) from a filename, or None if not matched.

    Thin wrapper around :func:`parse_subband_filename` that accepts a :class:`Path`.

    Parameters
    ----------
    path : Path
        Path to HDF5 subband file.

    Returns
    -------
    tuple or None
        Tuple of (group_id, subband_index) or None if filename doesn't match pattern.

    Examples
    --------
    >>> parse_subband_info(Path("/data/2025-10-02T00:12:00_sb05.hdf5"))
    ('2025-10-02T00:12:00', 5)
    """
    return parse_subband_filename(path.name)


def build_subband_filename(group_id: str, subband_idx: int) -> str:
    """Build a canonical subband filename from group_id and index.

    Parameters
    ----------
    group_id : str
        Canonical timestamp (YYYY-MM-DDTHH:MM:SS).
    subband_idx : int
        Subband index (0-15).

    Returns
    -------
    str
        Filename string like "2025-01-15T12:00:00_sb05.hdf5".

    Examples
    --------
    >>> build_subband_filename("2025-01-15T12:00:00", 5)
    '2025-01-15T12:00:00_sb05.hdf5'
    """
    return f"{group_id}_sb{subband_idx:02d}.hdf5"


def normalize_subband_path(
    path: Path,
    canonical_group_id: str,
    dry_run: bool = False,
) -> tuple[Path, bool]:
    """Rename a subband file to use the canonical group_id.

    If the file already has the correct name, no action is taken.

    Parameters
    ----------
    path : Path
        Current path to the subband file.
    canonical_group_id : str
        The canonical group_id to use (YYYY-MM-DDTHH:MM:SS).
    dry_run : bool, optional
        If True, don't actually rename, just return what would happen (default is False).

    Returns
    -------
    tuple
        - new_path : Path
            Path after normalization (may be same as input if no rename needed).
        - was_renamed : bool
            True if file was (or would be) renamed.

    Raises
    ------
    FileNotFoundError
        If the source file doesn't exist.
    ValueError
        If the path doesn't match subband filename pattern.
    OSError
        If rename fails (e.g., permission denied, target exists).

    Examples
    --------
    >>> path = Path("/data/2025-01-15T12:00:01_sb05.hdf5")
    >>> new_path, renamed = normalize_subband_path(path, "2025-01-15T12:00:00")
    >>> print(new_path)
    /data/2025-01-15T12:00:00_sb05.hdf5
    >>> print(renamed)
    True
    """
    # Parse current filename
    info = parse_subband_info(path)
    if info is None:
        raise ValueError(f"Path does not match subband pattern: {path}")

    current_group_id, subband_idx = info

    # Check if already normalized
    if current_group_id == canonical_group_id:
        logger.debug("File already normalized: %s", path.name)
        return path, False

    # Build new filename
    new_name = build_subband_filename(canonical_group_id, subband_idx)
    new_path = path.parent / new_name

    # Check source exists
    if not path.exists():
        raise FileNotFoundError(f"Source file does not exist: {path}")

    # Check target doesn't already exist (shouldn't happen in normal operation)
    if new_path.exists() and not path.samefile(new_path):
        raise OSError(f"Target file already exists: {new_path}")

    if dry_run:
        logger.info("Would rename: %s -> %s", path.name, new_name)
        return new_path, True

    # Perform atomic rename
    # Note: Path.rename() is atomic on POSIX filesystems
    try:
        path.rename(new_path)
        logger.info("Normalized: %s -> %s", path.name, new_name)
        return new_path, True
    except OSError as err:
        logger.error("Failed to rename %s -> %s: %s", path.name, new_name, err)
        raise


def normalize_subband_on_ingest(
    path: Path,
    target_group_id: str,
    source_group_id: str,
) -> Path:
    """Normalize a subband file during ingest.

    This is the main entry point called during streaming ingest. It handles:
    - Checking if rename is needed
    - Performing atomic rename

    Parameters
    ----------
    path :
        Path to the incoming subband file
    target_group_id :
        The canonical group_id (from clustering)
    source_group_id :
        The original group_id from the filename

    Returns
    -------
        Path after normalization (original or renamed)

    """
    if source_group_id == target_group_id:
        return path

    new_path, _ = normalize_subband_path(path, target_group_id, dry_run=False)
    return new_path


def normalize_directory(
    directory: Path,
    cluster_tolerance_s: float = DEFAULT_CLUSTER_TOLERANCE_S,
    dry_run: bool = True,
    files: list[Path] | None = None,
):
    """
    Normalize subband filenames in a directory to use a canonical group_id.

    This ensures all subbands for the same observation share the exact same
    timestamp in their filename, allowing for easy grouping by filename alone.

    Parameters
    ----------
    directory : Path
        Directory containing HDF5 subband files
    cluster_tolerance_s : float
        Tolerance in seconds for grouping timestamps (default: 120.0)
    dry_run : bool
        If True, only report what would be done (default: True for safety)
    files : Optional[list[Path]]
        Optional list of pre-scanned files to avoid re-scanning the directory.
        If provided, only these files will be considered.

    Returns
    -------
        dict
        Dictionary with statistics:
        - files_scanned: Total HDF5 files processed
        - files_renamed: Number of files renamed (or would be renamed)
        - groups_found: Number of observation groups detected
        - errors: Number of files that failed to parse/rename
    """
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    # Collect all subband files
    files_by_group: dict[str, dict[int, Path]] = defaultdict(dict)
    errors = 0

    # Use provided files or scan directory
    source_files = files if files is not None else directory.glob("*_sb*.hdf5")

    for hdf5_file in source_files:
        info = parse_subband_info(hdf5_file)
        if info is None:
            if files is None:  # Only log if we were scanning the whole directory
                logger.warning("Could not parse filename: %s", hdf5_file.name)
            errors += 1
            continue

        group_id, subband_idx = info
        files_by_group[group_id][subband_idx] = hdf5_file

    # Cluster groups within tolerance
    sorted_groups = sorted(files_by_group.keys())
    canonical_map: dict[str, str] = {}  # original_group_id -> canonical_group_id
    canonical_timestamps: list[str] = []  # Chronological list of canonical timestamps

    for group_id in sorted_groups:
        # Parse timestamp
        try:
            ts = datetime.fromisoformat(group_id)
        except ValueError:
            logger.warning("Invalid timestamp format: %s", group_id)
            canonical_map[group_id] = group_id
            canonical_timestamps.append(group_id)
            continue

        # Find or create canonical group
        # Performance optimization: Search BACKWARDS from the end of canonical_timestamps
        # since we are processing groups chronologically, the match will be near the end.
        found_canonical = False
        for i in range(len(canonical_timestamps) - 1, -1, -1):
            canonical_ts_str = canonical_timestamps[i]
            try:
                canonical_ts = datetime.fromisoformat(canonical_ts_str)
                diff = (ts - canonical_ts).total_seconds()

                if abs(diff) <= cluster_tolerance_s:
                    canonical_map[group_id] = canonical_ts_str
                    found_canonical = True
                    break

                # Since canonical_timestamps is sorted, if we are more than tolerance_s
                # away and moving further away (diff > tolerance_s), we can stop.
                if diff > cluster_tolerance_s:
                    break
            except ValueError:
                continue

        if not found_canonical:
            # This is a new canonical timestamp
            canonical_map[group_id] = group_id
            canonical_timestamps.append(group_id)

    # Normalize files
    files_renamed = 0
    files_scanned = 0

    for group_id, subbands in files_by_group.items():
        canonical = canonical_map.get(group_id, group_id)

        for subband_idx, path in subbands.items():
            files_scanned += 1

            if group_id != canonical:
                try:
                    _, was_renamed = normalize_subband_path(path, canonical, dry_run=dry_run)
                    if was_renamed:
                        files_renamed += 1
                except Exception as e:
                    logger.error("Failed to normalize %s: %s", path, e)
                    errors += 1

    # Count unique canonical groups
    groups_found = len(set(canonical_map.values()))

    return {
        "files_scanned": files_scanned,
        "files_renamed": files_renamed,
        "groups_found": groups_found,
        "errors": errors,
    }
