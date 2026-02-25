"""Helper utilities for UVH5 :arrow_right: CASA Measurement Set conversion.

This module provides backward-compatible imports from specialized helper modules.
All functions have been split into logical modules for better organization:
- helpers_antenna.py: Antenna position functions
- helpers_coordinates.py: Coordinate and phase functions
- helpers_model.py: Model and UVW functions
- helpers_validation.py: Validation functions
- helpers_telescope.py: Telescope utility functions
"""

import logging

# Provide a patchable casacore table symbol for tests and submodules
from dsa110_contimg.common.utils.casa_init import ensure_casa_path

ensure_casa_path()
import casacore.tables as casatables  # type: ignore

# Expose as module attribute so tests can patch dsa110_contimg.conversion.helpers.table
table = casatables.table  # noqa: N816

logger = logging.getLogger("dsa110_contimg.conversion.helpers")

# Import all functions from specialized modules for backward compatibility
from .helpers_antenna import (
    _ensure_antenna_diameters,
    set_antenna_positions,
)
from .helpers_coordinates import (
    compute_and_set_uvw,
    get_meridian_coords,
    phase_to_meridian,
)
from .helpers_model import (
    amplitude_sky_model,
    primary_beam_response,
    set_model_column,
)
from .helpers_telescope import (
    cleanup_casa_file_handles,
    set_telescope_identity,
)
from .helpers_validation import (
    validate_antenna_positions,
    validate_model_data_quality,
    validate_ms_frequency_order,
    validate_phase_center_coherence,
    validate_reference_antenna_stability,
    validate_uvw_precision,
)

__all__ = [
    "get_meridian_coords",
    "set_antenna_positions",
    "_ensure_antenna_diameters",
    "set_model_column",
    "amplitude_sky_model",
    "primary_beam_response",
    "phase_to_meridian",
    "validate_ms_frequency_order",
    "cleanup_casa_file_handles",
    "validate_phase_center_coherence",
    "validate_uvw_precision",
    "validate_antenna_positions",
    "validate_model_data_quality",
    "validate_reference_antenna_stability",
    "set_telescope_identity",
    "compute_and_set_uvw",
]
