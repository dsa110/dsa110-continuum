# backend/src/dsa110_contimg/imaging/__init__.py

"""Imaging module for DSA-110 continuum pipeline.

.. note::
    For new code, prefer using the public API which provides a simpler interface:

        from dsa110_contimg.interfaces.public_api import image_ms

    This module is primarily for internal use and advanced customization.

This module provides tools for creating FITS images from calibrated
Measurement Sets using WSClean or CASA tclean, as well as catalog-based
mask and overlay generation.

GPU-accelerated visibility prediction (degridding) is available for
multi-source sky model calibration.
"""

from dsa110_contimg.core.imaging.catalog_tools import (
    create_catalog_fits_mask,
    create_catalog_mask,
    create_catalog_overlay,
)
from dsa110_contimg.core.imaging.gpu_gridding import (
    DegridConfig,
    DegridResult,
    GriddingConfig,
    GriddingResult,
    clear_w_kernel_cache,
    cpu_grid_visibilities,
    gpu_degrid_visibilities,
    gpu_grid_visibilities,
)
from dsa110_contimg.core.imaging.gpu_predict import (
    CatalogSourceAdapter,
    PredictConfig,
    PredictResult,
    SourceModel,
    predict_model_for_ms,
    predict_model_from_catalog,
    predict_visibilities_gpu,
    render_sources_to_image,
)
from dsa110_contimg.core.imaging.params import (
    ImagingParams,
    image_ms_with_params,
)

__all__ = [
    # Catalog tools
    "create_catalog_fits_mask",
    "create_catalog_mask",
    "create_catalog_overlay",
    # Imaging parameters
    "ImagingParams",
    "image_ms_with_params",
    # GPU gridding
    "GriddingConfig",
    "GriddingResult",
    "gpu_grid_visibilities",
    "cpu_grid_visibilities",
    # GPU degridding (prediction)
    "DegridConfig",
    "DegridResult",
    "gpu_degrid_visibilities",
    "clear_w_kernel_cache",
    # Sky model prediction
    "SourceModel",
    "PredictConfig",
    "PredictResult",
    "CatalogSourceAdapter",
    "render_sources_to_image",
    "predict_visibilities_gpu",
    "predict_model_for_ms",
    "predict_model_from_catalog",
]
