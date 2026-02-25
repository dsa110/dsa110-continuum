# backend/src/dsa110_contimg/conversion/__init__.py

"""
DSA-110 Continuum Imaging Pipeline - Conversion Module.

.. note::
    For new code, prefer using the public API which provides a simpler interface:

        from dsa110_contimg.interfaces.public_api import convert_uvh5_to_ms

    This module is primarily for internal use and advanced customization.

This module provides functionality for converting UVH5 subband files to
Measurement Sets (MS).

Entry Points:
    Batch conversion:
        from dsa110_contimg.core.conversion import convert_subband_groups_to_ms

    Dagster-based ingestion (replaces old streaming pipeline):
        # See docs/guides/ingestion.md for Dagster asset setup

Writers:
    DirectSubbandWriter - Main MS writer for production use

    For explicit file-list conversion (bypassing auto-discovery), use
    DirectSubbandWriter directly::

        from dsa110_contimg.core.conversion.writers import get_writer
        import pyuvdata

        writer_cls = get_writer("direct-subband")
        uvdata = pyuvdata.UVData()  # Empty - DirectSubbandWriter reads files directly
        writer = writer_cls(uvdata, ms_path, file_list=file_list, max_workers=4)
        writer.write()
"""

from . import helpers_coordinates  # Make coordinate helpers accessible via the package

# Downsampling utilities
from .downsample_uvh5 import (
    downsample_uvh5,
    get_downsampling_info,
)

# Flattened exports - main conversion API
from .conversion_orchestrator import convert_subband_groups_to_ms

# File normalization utilities (formerly in streaming submodule)
from .normalize import (
    build_subband_filename,
    normalize_directory,
    normalize_subband_on_ingest,
    normalize_subband_path,
)

# Writers
from .writers import (
    DirectSubbandWriter,
    MSWriter,
    get_writer,
)

# Calibrator transit-based MS generation
from .calibrator_ms_generator import (
    CalibratorInfo,
    CalibratorMSGenerator,
    CalibratorMSResult,
    TransitInfo,
)

__all__ = [
    # Submodules
    "helpers_coordinates",
    # Batch conversion
    "convert_subband_groups_to_ms",
    # Normalization
    "build_subband_filename",
    "normalize_directory",
    "normalize_subband_on_ingest",
    "normalize_subband_path",
    # Writers
    "MSWriter",
    "DirectSubbandWriter",
    "get_writer",
    # Downsampling
    "downsample_uvh5",
    "get_downsampling_info",
    # Calibrator MS generation
    "CalibratorMSGenerator",
    "CalibratorMSResult",
    "CalibratorInfo",
    "TransitInfo",
]
