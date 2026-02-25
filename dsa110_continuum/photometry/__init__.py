"""Photometry utilities for DSA-110 (forced photometry on FITS images)."""

from dsa110_contimg.core.photometry.condon_errors import (
    CondonErrors,
    CondonFluxErrors,
    CondonPositionErrors,
    calc_condon_errors,
    calc_condon_flux_errors,
    calc_condon_position_errors,
    simple_position_error,
)
from dsa110_contimg.core.photometry.forced import (
    ForcedPhotometryResult,
    inject_source,
    measure_forced_peak,
    measure_many,
)

# Image and source QA metrics (VAST-style)
from dsa110_contimg.core.photometry.image_qa import (
    ImageQAMetrics,
    ImageRMSMetrics,
    compute_image_qa_metrics,
    get_local_rms_at_position,
    get_rms_noise_image_values,
)
from dsa110_contimg.core.photometry.manager import (
    PhotometryConfig,
    PhotometryManager,
    PhotometryResult,
)
from dsa110_contimg.core.photometry.multi_epoch import (
    FluxAggregateStats,
    MultiEpochSourceStats,
    NewSourceMetrics,
    SNRAggregateStats,
    WeightedPositionStats,
    calc_flux_aggregates,
    calc_new_source_significance,
    calc_two_epoch_pair_metrics,
    compute_multi_epoch_stats,
    get_most_significant_pair,
)
from dsa110_contimg.core.photometry.source_metrics import (
    IslandMetrics,
    SourceMorphologyMetrics,
    SourceQAMetrics,
    SpatialMetrics,
    batch_compute_source_metrics,
    calculate_compactness,
    calculate_snr,
    compute_source_qa_metrics,
)

__all__: list[str] = [
    # Forced photometry
    "ForcedPhotometryResult",
    "measure_forced_peak",
    "measure_many",
    "inject_source",
    "PhotometryManager",
    "PhotometryConfig",
    "PhotometryResult",
    # Image QA metrics
    "ImageQAMetrics",
    "ImageRMSMetrics",
    "compute_image_qa_metrics",
    "get_local_rms_at_position",
    "get_rms_noise_image_values",
    # Source metrics
    "IslandMetrics",
    "SourceMorphologyMetrics",
    "SourceQAMetrics",
    "SpatialMetrics",
    "batch_compute_source_metrics",
    "calculate_compactness",
    "calculate_snr",
    "compute_source_qa_metrics",
    # Condon errors
    "CondonErrors",
    "CondonFluxErrors",
    "CondonPositionErrors",
    "calc_condon_errors",
    "calc_condon_flux_errors",
    "calc_condon_position_errors",
    "simple_position_error",
    # Multi-epoch stats
    "FluxAggregateStats",
    "MultiEpochSourceStats",
    "NewSourceMetrics",
    "SNRAggregateStats",
    "WeightedPositionStats",
    "calc_flux_aggregates",
    "calc_new_source_significance",
    "calc_two_epoch_pair_metrics",
    "compute_multi_epoch_stats",
    "get_most_significant_pair",
]
