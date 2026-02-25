"""Simulation utilities for generating synthetic DSA-110 datasets.

This package provides tools for:
- Generating synthetic UVH5 visibility data
- Injecting time-variable sources (flares, ESE, periodic)
- Ground truth tracking for validation
- Validation of pipeline outputs
- Multi-epoch time-domain simulation

Main modules:
    - time_domain: Multi-epoch generation with variability
    - variability_models: Time-varying flux models
    - ground_truth: Track injected sources for validation
    - validation: Validate pipeline outputs against ground truth
    - metrics: Quantitative validation metrics
    - pyuvsim_adapter: Integration with pyuvsim for accurate visibility simulation

pyuvsim Integration
-------------------
The simulation toolkit can use pyuvsim (Lanman et al., 2019) for high-precision
visibility simulation. pyuvsim provides:
- Validated visibility calculations (tested against analytic solutions)
- Full polarization and beam model support
- MPI parallelization for large simulations

To use pyuvsim for visibility generation:
    >>> from dsa110_contimg.core.simulation import simulate_visibilities
    >>> uvdata_sim = simulate_visibilities(uvdata, sources)

"""

# Time-domain simulation
# Ground truth tracking
from dsa110_contimg.core.simulation.ground_truth import (
    GroundTruthRegistry,
    GroundTruthSource,
)

# Metrics
from dsa110_contimg.core.simulation.metrics import (
    astrometric_offset,
    compute_variability_metrics,
    detection_completeness,
    false_positive_rate,
    flux_recovery_error,
    match_sources_by_position,
    mean_absolute_percentage_error,
    rms_flux_error,
)
from dsa110_contimg.core.simulation.time_domain import (
    EpochData,
    MultiEpochResult,
    generate_multi_epoch_uvh5,
)

# Validation
from dsa110_contimg.core.simulation.validation import (
    ValidationReport,
    validate_all,
    validate_ese_detection,
    validate_lightcurve,
    validate_photometry,
)

# Variability models
from dsa110_contimg.core.simulation.variability_models import (
    ConstantFlux,
    ESEScattering,
    FlareModel,
    PeriodicVariation,
    VariabilityModel,
    compute_flux_at_time,
    create_variability_model,
)

# pyuvsim integration for accurate visibility simulation
from dsa110_contimg.core.simulation.pyuvsim_adapter import (
    check_mpi_available,
    check_pyuvsim_available,
    create_dsa110_beam,
    simulate_visibilities,
    sources_to_skymodel,
)

# Configuration and constants
from dsa110_contimg.core.simulation.simulation_config import (
    DSA110_CHANNEL_WIDTH_HZ,
    DSA110_CHANNELS_PER_SUBBAND,
    DSA110_FREQ_MAX_HZ,
    DSA110_FREQ_MIN_HZ,
    DSA110_INTEGRATION_TIME_SEC,
    DSA110_NUM_ANTENNAS,
    DSA110_NUM_ANTENNAS_TEST,
    DSA110_NUM_INTEGRATIONS,
    DSA110_NUM_POLARIZATIONS,
    DSA110_NUM_SUBBANDS,
    DSA110_REFERENCE_FREQ_HZ,
    DSA110_TOTAL_BANDWIDTH_HZ,
    DSA110_TOTAL_CHANNELS,
    SimulationConfig,
    get_config,
    get_test_config,
)

__all__ = [
    # Configuration
    "SimulationConfig",
    "get_config",
    "get_test_config",
    # Constants
    "DSA110_NUM_ANTENNAS",
    "DSA110_NUM_ANTENNAS_TEST",
    "DSA110_INTEGRATION_TIME_SEC",
    "DSA110_NUM_INTEGRATIONS",
    "DSA110_NUM_SUBBANDS",
    "DSA110_CHANNELS_PER_SUBBAND",
    "DSA110_TOTAL_CHANNELS",
    "DSA110_CHANNEL_WIDTH_HZ",
    "DSA110_TOTAL_BANDWIDTH_HZ",
    "DSA110_FREQ_MIN_HZ",
    "DSA110_FREQ_MAX_HZ",
    "DSA110_REFERENCE_FREQ_HZ",
    "DSA110_NUM_POLARIZATIONS",
    # Time-domain
    "generate_multi_epoch_uvh5",
    "EpochData",
    "MultiEpochResult",
    # Variability models
    "VariabilityModel",
    "ConstantFlux",
    "FlareModel",
    "ESEScattering",
    "PeriodicVariation",
    "compute_flux_at_time",
    "create_variability_model",
    # Ground truth
    "GroundTruthSource",
    "GroundTruthRegistry",
    # Validation
    "ValidationReport",
    "validate_photometry",
    "validate_lightcurve",
    "validate_ese_detection",
    "validate_all",
    # Metrics
    "flux_recovery_error",
    "astrometric_offset",
    "detection_completeness",
    "false_positive_rate",
    "rms_flux_error",
    "mean_absolute_percentage_error",
    "compute_variability_metrics",
    "match_sources_by_position",
    # pyuvsim integration
    "simulate_visibilities",
    "sources_to_skymodel",
    "create_dsa110_beam",
    "check_pyuvsim_available",
    "check_mpi_available",
]
