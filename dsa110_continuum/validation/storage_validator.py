"""
Storage path validation for DSA-110 pipeline.

Validates that output paths are appropriate for the operation type.
The DSA-110 system has different storage tiers:
- /data/ (HDD) - Source code, databases, raw HDF5 (slow)
- /stage/ (NVMe SSD) - Output MS files, working data (fast)
- /dev/shm/ (NVMe SSD) - Temp files, builds (fast)
- /dev/shm/ (tmpfs) - In-memory staging for conversion

I/O-intensive operations should avoid /data/ to prevent performance degradation.
"""

import warnings
from pathlib import Path
from typing import Literal

# Storage tier classifications (site-specific constants for DSA-110 system)
# These hardcoded paths are intentional and define the storage architecture
SLOW_STORAGE = ["/data"]  # nosec B108 # noqa: S108
FAST_STORAGE = ["/stage", "/dev/shm/dsa110-contimg", "/dev/shm"]  # nosec B108 # noqa: S108
ALLOWED_SLOW_OPS = ["read", "database", "catalog"]

# Operation types that require fast storage
IO_INTENSIVE_OPS = [
    "conversion",  # UVH5 → MS conversion
    "imaging",     # WSClean/tclean imaging
    "calibration", # CASA calibration
    "mosaic",      # Mosaic generation
    "build",       # Frontend/docs builds
]

OperationType = Literal[
    "conversion",
    "imaging",
    "calibration",
    "mosaic",
    "build",
    "read",
    "database",
    "catalog",
    "other",
]


class PerformanceWarning(UserWarning):
    """Warning for performance-impacting storage choices."""
    pass


def validate_output_path(
    path: Path | str,
    operation: OperationType,
    *,
    raise_error: bool = False,
) -> bool:
    """
    Validate that an output path is appropriate for the operation type.

    Args:
        path: Output path to validate
        operation: Type of operation being performed
        raise_error: If True, raise ValueError instead of warning

    Returns:
        True if path is appropriate, False otherwise

    Raises:
        ValueError: If raise_error=True and path is inappropriate

    Examples:
        >>> from pathlib import Path
        >>> # Good: MS output to fast storage
        >>> validate_output_path("/stage/ms/file.ms", "conversion")
        True

        >>> # Warning: MS output to slow storage
        >>> validate_output_path("/data/ms/file.ms", "conversion")
        PerformanceWarning: Writing conversion output to slow storage: /data/ms/file.ms
        False

        >>> # OK: Database on /data/
        >>> validate_output_path("/data/pipeline.sqlite3", "database")
        True
    """
    if isinstance(path, str):
        path = Path(path)

    path_str = str(path.resolve())

    # Check if operation is I/O-intensive
    if operation in IO_INTENSIVE_OPS:
        # Check if writing to slow storage
        on_slow_storage = any(path_str.startswith(slow) for slow in SLOW_STORAGE)

        if on_slow_storage:
            msg = (
                f"Writing {operation} output to slow storage: {path}\n"
                f"Consider using fast storage for better performance:\n"
                f"  - /stage/ for MS files and working data\n"
                f"  - /dev/shm/ for temporary files and builds\n"
                f"  - /dev/shm/ for in-memory staging"
            )

            if raise_error:
                raise ValueError(msg)
            else:
                warnings.warn(msg, PerformanceWarning, stacklevel=2)
                return False

    return True


def get_recommended_storage(operation: OperationType) -> str:
    """
    Get recommended storage location for an operation type.

    Args:
        operation: Type of operation

    Returns:
        Recommended storage path prefix

    Examples:
        >>> get_recommended_storage("conversion")
        '/stage'

        >>> get_recommended_storage("build")
        '/dev/shm/dsa110-contimg'

        >>> get_recommended_storage("database")
        '/data'
    """
    recommendations = {
        "conversion": "/stage",
        "imaging": "/stage",
        "calibration": "/stage",
        "mosaic": "/stage",
        "build": "/dev/shm/dsa110-contimg",
        "read": "/data",
        "database": "/data",
        "catalog": "/data",
        "other": "/stage",
    }

    return recommendations.get(operation, "/stage")


def check_available_space(path: Path | str, required_gb: float = 10.0) -> bool:
    """
    Check if path has sufficient available space.

    Args:
        path: Path to check
        required_gb: Minimum required space in GB

    Returns:
        True if sufficient space available

    Raises:
        RuntimeError: If insufficient space available
    """
    if isinstance(path, str):
        path = Path(path)

    import shutil

    # Get disk usage stats for the mount point
    try:
        stats = shutil.disk_usage(path.parent if path.is_file() else path)
    except (OSError, FileNotFoundError):
        # Path doesn't exist yet, check parent
        stats = shutil.disk_usage(path.parent)

    available_gb = stats.free / (1024**3)

    if available_gb < required_gb:
        raise RuntimeError(
            f"Insufficient disk space on {path}\n"
            f"Available: {available_gb:.1f} GB, Required: {required_gb:.1f} GB"
        )

    return True


def suggest_alternative_path(current_path: Path | str, operation: OperationType) -> Path:
    """
    Suggest an alternative path for better performance.

    Args:
        current_path: Current (suboptimal) path
        operation: Operation type

    Returns:
        Suggested alternative path

    Examples:
        >>> suggest_alternative_path("/data/ms/file.ms", "conversion")
        PosixPath('/stage/ms/file.ms')

        >>> suggest_alternative_path("/data/temp/build", "build")
        PosixPath('/dev/shm/temp/build')
    """
    if isinstance(current_path, str):
        current_path = Path(current_path)

    recommended = get_recommended_storage(operation)

    # Replace the slow storage prefix with recommended
    path_str = str(current_path)
    for slow in SLOW_STORAGE:
        if path_str.startswith(slow):
            return Path(path_str.replace(slow, recommended, 1))

    return current_path


# Example usage in pipeline code
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m dsa110_contimg.core.validation.storage_validator <path> <operation>")
        print("Operations: conversion, imaging, calibration, mosaic, build")
        sys.exit(1)

    path = Path(sys.argv[1])
    operation = sys.argv[2]

    is_valid = validate_output_path(path, operation)  # type: ignore

    if is_valid:
        print(f"✓ Path is appropriate: {path}")
    else:
        suggested = suggest_alternative_path(path, operation)  # type: ignore
        print(f"⚠ Consider using: {suggested}")
