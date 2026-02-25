"""
Disk space checks for imaging/mosaic tasks.

Provides a minimal check_disk_space used by imaging CLI. If insufficient space
is detected and fatal=True, a RuntimeError is raised; otherwise a warning
message is returned.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def check_disk_space(
    target_path: str | Path,
    *,
    required_bytes: int,
    operation: str = "",
    fatal: bool = False,
) -> tuple[bool, str]:
    """Check if there is enough free disk space for an operation.

    Parameters
    ----------
    target_path :
        File or directory path where data will be written.
    required_bytes :
        Estimated bytes required.
    operation :
        Description used in messages.
    fatal :
        If True, raise RuntimeError when space is insufficient.
    target_path: str | Path :

    * :

    Returns
    -------
        (ok, message) where ok indicates sufficient space.

    """
    path = Path(target_path)
    # Use parent if target is not yet created
    if not path.exists():
        path = path.parent
    if not path.exists():
        # Fallback to stage root for disk space check
        from dsa110_contimg.infrastructure.database import data_config

        path = (
            data_config.STAGE_BASE if data_config.STAGE_BASE.exists() else data_config.get_pid_dir()
        )

    usage = shutil.disk_usage(str(path))
    free_bytes = usage.free

    msg = (
        f"Disk space check for {operation or 'operation'}: "
        f"required={required_bytes / 1e9:.2f}GB, available={free_bytes / 1e9:.2f}GB"
    )

    if free_bytes < required_bytes:
        msg = "Insufficient " + msg
        if fatal:
            raise RuntimeError(msg)
        return False, msg

    return True, msg
