"""Shared group ID parsing helpers for ops pipeline scripts.

.. deprecated::
    Use ``parse_subband_filename`` from
    :mod:`dsa110_contimg.infrastructure.database.hdf5_index` instead.
    ``group_id_from_path(p)`` is equivalent to ``parse_subband_filename(os.path.basename(p))[0]``.
"""

import os
import warnings


def group_id_from_path(path: str) -> str:
    """Extract group ID from file path.

    .. deprecated::
        Use ``parse_subband_filename(os.path.basename(path))[0]`` instead.

    Group ID is the base filename without the subband suffix (e.g., _sb00).

    Args:
        path: File path (e.g., "/path/to/group_sb00.hdf5")

    Returns
    -------
        Group ID (e.g., "group")

    Example:
        >>> group_id_from_path("/data/0834_555_sb00.hdf5")
        '0834_555'

    """
    warnings.warn(
        "group_id_from_path() is deprecated. "
        "Use parse_subband_filename(os.path.basename(path))[0] from "
        "dsa110_contimg.infrastructure.database.hdf5_index instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    base = os.path.basename(path)
    return base.split("_sb", 1)[0]
