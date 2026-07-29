"""Quality Assurance and validation utilities for DSA-110 imaging pipeline.

    This module provides catalog-based validation for verifying flux scale accuracy,
    astrometric precision, and source completeness in pipeline-generated images.

    Main Functions:
validate_flux_scale: Validate flux scale against reference catalog
run_full_validation: Run all validation types (astrometry, flux, counts)
extract_sources_from_image: Extract source positions from FITS image

    Pipeline Hooks:
hook_ese_detection_complete: Post-ESE-detection hook

    Example
-------
    >>> from dsa110_continuum.qa import validate_flux_scale
    >>> result = validate_flux_scale("image.fits", catalog="nvss", min_snr=5.0)
    >>> print(f"Flux scale error: {result.flux_scale_error * 100:.1f}%")
"""

try:
    from dsa110_continuum.qa.calibration_stability_tracker import (
        AntennaGainSnapshot,
        AntennaTrendAnalysis,
        CalibrationStabilityReport,
        CalibrationStabilityTracker,
        get_global_tracker,
        reset_global_tracker,
    )
    from dsa110_continuum.qa.catalog_validation import (
        AstrometryResult,
        FluxScaleResult,
        SourceCountsResult,
        extract_sources_from_image,
        run_full_validation,
        validate_flux_scale,
    )
    from dsa110_continuum.qa.delay_validation import (
        DelayValidationResult,
        check_delay_solutions,
        compute_geometric_delay_limits,
        validate_delay_solutions,
    )
    from dsa110_continuum.qa.pipeline_hooks import hook_ese_detection_complete
    from dsa110_continuum.qa.pipeline_quality import (
        check_calibration_quality,
        check_image_quality,
        check_ms_after_conversion,
    )
    from dsa110_continuum.qa.uvw_validation import (
        UVWValidationResult,
        check_uvw_after_phaseshift,
        compare_uvw_before_after,
        validate_uvw_geometry,
    )
except ImportError:
    pass  # optional deps of the target module absent (cloud/test env)

__all__ = [
    # Catalog validation
    "AstrometryResult",
    "FluxScaleResult",
    "SourceCountsResult",
    "extract_sources_from_image",
    "run_full_validation",
    "validate_flux_scale",
    # Pipeline hooks
    "hook_ese_detection_complete",
    # Calibration stability tracking (ring buffer)
    "AntennaGainSnapshot",
    "AntennaTrendAnalysis",
    "CalibrationStabilityReport",
    "CalibrationStabilityTracker",
    "get_global_tracker",
    "reset_global_tracker",
    # Pipeline quality wrappers
    "check_calibration_quality",
    "check_image_quality",
    "check_ms_after_conversion",
    # Delay validation
    "DelayValidationResult",
    "check_delay_solutions",
    "compute_geometric_delay_limits",
    "validate_delay_solutions",
    # UVW validation
    "UVWValidationResult",
    "check_uvw_after_phaseshift",
    "compare_uvw_before_after",
    "validate_uvw_geometry",
]
