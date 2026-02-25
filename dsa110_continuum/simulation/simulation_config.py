"""DSA-110 Simulation Configuration Module.

This module provides a centralized configuration system for DSA-110 simulations,
loading parameters from the authoritative dsa110_measured_parameters.yaml file.

The configuration is loaded once and cached for performance. All simulation
code should import parameters from this module rather than hardcoding values.

Examples
--------
>>> from dsa110_contimg.core.simulation.simulation_config import get_config
>>> config = get_config()
>>> print(f"Integration time: {config.integration_time_sec} s")
>>> print(f"Number of antennas: {config.num_antennas}")

>>> # Or use individual constants
>>> from dsa110_contimg.core.simulation.simulation_config import (
...     DSA110_NUM_ANTENNAS,
...     DSA110_INTEGRATION_TIME_SEC,
...     DSA110_NUM_SUBBANDS,
... )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)

# =============================================================================
# DSA-110 Constants (Production Values)
# These are the authoritative values from dsa110_measured_parameters.yaml
# =============================================================================

# Array Configuration
DSA110_NUM_ANTENNAS: int = 117
"""Number of antennas in the full DSA-110 array."""

DSA110_NUM_ANTENNAS_TEST: int = 8
"""Reduced antenna count for fast unit tests (28 baselines)."""

# Temporal Parameters
DSA110_INTEGRATION_TIME_SEC: float = 12.884902000427246
"""Integration time per visibility sample (seconds). High precision from correlator."""

DSA110_NUM_INTEGRATIONS: int = 24
"""Number of time integrations in a standard 5-minute tile."""

DSA110_TOTAL_DURATION_SEC: float = 309.12
"""Total observation duration for a tile (seconds). ~5 minutes."""

# Spectral Parameters
DSA110_NUM_SUBBANDS: int = 16
"""Number of spectral subbands output by the correlator."""

DSA110_CHANNELS_PER_SUBBAND: int = 48
"""Number of frequency channels per subband."""

DSA110_TOTAL_CHANNELS: int = 768
"""Total number of channels across all subbands (16 × 48)."""

DSA110_CHANNEL_WIDTH_HZ: float = 244140.625
"""Width of each frequency channel (Hz). = 15.625 MHz / 48 channels per raw subband."""

DSA110_SUBBAND_WIDTH_HZ: float = 11718750.0
"""Bandwidth of one subband (Hz). = 48 × 244140.625 Hz."""

DSA110_TOTAL_BANDWIDTH_HZ: float = 187500000.0
"""Total bandwidth across all subbands (Hz). = 16 × 11.71875 MHz."""

DSA110_FREQ_MIN_HZ: float = 1311400000.0
"""Minimum frequency (Hz). Lower edge of subband 15."""

DSA110_FREQ_MAX_HZ: float = 1498600000.0
"""Maximum frequency (Hz). Upper edge of subband 00."""

DSA110_REFERENCE_FREQ_HZ: float = 1405000000.0
"""Reference frequency for spectral index calculations (Hz). Center of L-band."""

# Site Parameters
DSA110_LATITUDE_DEG: float = 37.2314
"""Observatory latitude (degrees)."""

DSA110_LONGITUDE_DEG: float = -118.2817
"""Observatory longitude (degrees)."""

DSA110_ALTITUDE_M: float = 1222.0
"""Observatory altitude (meters)."""

# Polarization
DSA110_POLARIZATIONS: tuple[int, int] = (-5, -6)
"""Polarization codes for XX (-5) and YY (-6)."""

DSA110_NUM_POLARIZATIONS: int = 2
"""Number of polarizations (XX, YY)."""

# Antenna Parameters
DSA110_DISH_DIAMETER_M: float = 4.65
"""Antenna dish diameter (meters)."""

# System Parameters (assumed values, pending measurement)
DSA110_SYSTEM_TEMP_K: float = 50.0
"""System temperature (K). ASSUMED - awaiting measurement."""

DSA110_APERTURE_EFFICIENCY: float = 0.7
"""Aperture efficiency. ASSUMED - typical value for small dishes."""


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for DSA-110 simulations.
    
    This dataclass holds all parameters needed for generating synthetic
    DSA-110 data. Values are loaded from dsa110_measured_parameters.yaml
    or use the constants defined above as defaults.
    
    Attributes
    ----------
    num_antennas : int
        Number of antennas in the array (default: 117 for production)
    integration_time_sec : float
        Integration time per visibility sample (seconds)
    num_subbands : int
        Number of spectral subbands
    channels_per_subband : int
        Number of frequency channels per subband
    channel_width_hz : float
        Width of each frequency channel (Hz)
    freq_min_hz : float
        Minimum frequency (Hz)
    freq_max_hz : float
        Maximum frequency (Hz)
    reference_freq_hz : float
        Reference frequency for spectral calculations (Hz)
    num_integrations : int
        Number of time integrations per tile
    num_polarizations : int
        Number of polarizations (typically 2: XX, YY)
    latitude_deg : float
        Observatory latitude (degrees)
    longitude_deg : float
        Observatory longitude (degrees)
    altitude_m : float
        Observatory altitude (meters)
    dish_diameter_m : float
        Antenna dish diameter (meters)
    system_temp_k : float
        System temperature (K)
    
    Examples
    --------
    >>> config = SimulationConfig()
    >>> print(f"Full array: {config.num_antennas} antennas")
    Full array: 117 antennas
    
    >>> # Create config for fast testing
    >>> test_config = SimulationConfig(num_antennas=8)
    >>> print(f"Test array: {test_config.num_antennas} antennas")
    Test array: 8 antennas
    """
    
    # Array configuration
    num_antennas: int = DSA110_NUM_ANTENNAS
    
    # Temporal parameters
    integration_time_sec: float = DSA110_INTEGRATION_TIME_SEC
    num_integrations: int = DSA110_NUM_INTEGRATIONS
    total_duration_sec: float = DSA110_TOTAL_DURATION_SEC
    
    # Spectral parameters
    num_subbands: int = DSA110_NUM_SUBBANDS
    channels_per_subband: int = DSA110_CHANNELS_PER_SUBBAND
    channel_width_hz: float = DSA110_CHANNEL_WIDTH_HZ
    subband_width_hz: float = DSA110_SUBBAND_WIDTH_HZ
    total_bandwidth_hz: float = DSA110_TOTAL_BANDWIDTH_HZ
    freq_min_hz: float = DSA110_FREQ_MIN_HZ
    freq_max_hz: float = DSA110_FREQ_MAX_HZ
    reference_freq_hz: float = DSA110_REFERENCE_FREQ_HZ
    
    # Site parameters
    latitude_deg: float = DSA110_LATITUDE_DEG
    longitude_deg: float = DSA110_LONGITUDE_DEG
    altitude_m: float = DSA110_ALTITUDE_M
    
    # Polarization
    num_polarizations: int = DSA110_NUM_POLARIZATIONS
    polarizations: tuple[int, int] = DSA110_POLARIZATIONS
    
    # Antenna parameters
    dish_diameter_m: float = DSA110_DISH_DIAMETER_M
    
    # System parameters
    system_temp_k: float = DSA110_SYSTEM_TEMP_K
    aperture_efficiency: float = DSA110_APERTURE_EFFICIENCY
    
    # Derived properties
    @property
    def total_channels(self) -> int:
        """Total number of channels across all subbands."""
        return self.num_subbands * self.channels_per_subband
    
    @property
    def num_baselines(self) -> int:
        """Number of cross-correlation baselines (excluding autocorrelations)."""
        return self.num_antennas * (self.num_antennas - 1) // 2
    
    @property
    def num_baselines_with_auto(self) -> int:
        """Number of baselines including autocorrelations."""
        return self.num_antennas * (self.num_antennas + 1) // 2
    
    @classmethod
    def from_yaml(cls, yaml_path: Path | None = None) -> SimulationConfig:
        """Load configuration from YAML file.
        
        Parameters
        ----------
        yaml_path : Path, optional
            Path to the YAML configuration file. If not provided,
            uses the default dsa110_measured_parameters.yaml.
        
        Returns
        -------
        SimulationConfig
            Configuration loaded from YAML.
        
        Examples
        --------
        >>> config = SimulationConfig.from_yaml()
        >>> print(config.num_antennas)
        117
        """
        if yaml_path is None:
            yaml_path = (
                Path(__file__).parent / "config" / "dsa110_measured_parameters.yaml"
            )
        
        try:
            from dsa110_contimg.common.utils.yaml_loader import load_yaml_with_env
            params = load_yaml_with_env(yaml_path)
        except ImportError:
            # Fallback to standard yaml if yaml_loader not available
            import yaml
            with open(yaml_path) as f:
                params = yaml.safe_load(f)
        
        # Extract relevant parameters
        spectral = params.get("spectral", {})
        temporal = params.get("temporal", {})
        site = params.get("site", {})
        antenna_params = params.get("antenna_parameters", {})
        system_params = params.get("system_parameters", {})
        observing_params = params.get("observing_parameters", {})
        
        return cls(
            # Array configuration
            num_antennas=_get_param_value(observing_params, "num_antennas", DSA110_NUM_ANTENNAS),
            
            # Temporal parameters
            integration_time_sec=temporal.get("integration_time_sec", DSA110_INTEGRATION_TIME_SEC),
            num_integrations=DSA110_NUM_INTEGRATIONS,  # Not in YAML
            total_duration_sec=temporal.get("total_duration_sec", DSA110_TOTAL_DURATION_SEC),
            
            # Spectral parameters
            num_subbands=spectral.get("num_subbands", DSA110_NUM_SUBBANDS),
            channels_per_subband=spectral.get("channels_per_subband", DSA110_CHANNELS_PER_SUBBAND),
            channel_width_hz=spectral.get("channel_width_hz", DSA110_CHANNEL_WIDTH_HZ),
            freq_min_hz=spectral.get("freq_min_hz", DSA110_FREQ_MIN_HZ),
            freq_max_hz=spectral.get("freq_max_hz", DSA110_FREQ_MAX_HZ),
            reference_freq_hz=spectral.get("reference_frequency_hz", DSA110_REFERENCE_FREQ_HZ),
            
            # Site parameters
            latitude_deg=site.get("latitude_deg", DSA110_LATITUDE_DEG),
            longitude_deg=site.get("longitude_deg", DSA110_LONGITUDE_DEG),
            altitude_m=site.get("altitude_m", DSA110_ALTITUDE_M),
            
            # Polarization
            num_polarizations=_get_param_value(observing_params, "num_polarizations", DSA110_NUM_POLARIZATIONS),
            
            # Antenna parameters
            dish_diameter_m=_get_param_value(antenna_params, "antenna_diameter", DSA110_DISH_DIAMETER_M),
            
            # System parameters
            system_temp_k=_get_param_value(system_params, "system_temperature", DSA110_SYSTEM_TEMP_K),
            aperture_efficiency=_get_param_value(system_params, "aperture_efficiency", DSA110_APERTURE_EFFICIENCY),
        )
    
    @classmethod
    def for_testing(cls, num_antennas: int = DSA110_NUM_ANTENNAS_TEST) -> SimulationConfig:
        """Create a configuration suitable for unit testing.
        
        Uses a reduced antenna count (default 8) for faster test execution
        while maintaining all other production parameters.
        
        Parameters
        ----------
        num_antennas : int, optional
            Number of antennas for testing (default: 8)
        
        Returns
        -------
        SimulationConfig
            Configuration with reduced antenna count.
        
        Examples
        --------
        >>> test_config = SimulationConfig.for_testing()
        >>> print(f"Test with {test_config.num_antennas} antennas ({test_config.num_baselines} baselines)")
        Test with 8 antennas (28 baselines)
        """
        return cls(num_antennas=num_antennas)


def _get_param_value(params: dict[str, Any], key: str, default: Any) -> Any:
    """Extract parameter value from YAML structure.
    
    Handles both simple values and the structured format with 'value' key.
    """
    if key not in params:
        return default
    
    value = params[key]
    if isinstance(value, dict) and "value" in value:
        return value["value"] if value["value"] is not None else default
    return value


@lru_cache(maxsize=1)
def get_config(yaml_path: str | None = None) -> SimulationConfig:
    """Get the cached simulation configuration.
    
    This function loads the configuration once and caches it for performance.
    Subsequent calls return the cached configuration.
    
    Parameters
    ----------
    yaml_path : str, optional
        Path to YAML configuration file. If not provided, uses the default
        dsa110_measured_parameters.yaml file.
    
    Returns
    -------
    SimulationConfig
        The cached simulation configuration.
    
    Examples
    --------
    >>> config = get_config()
    >>> print(config.num_antennas)
    117
    """
    path = Path(yaml_path) if yaml_path else None
    return SimulationConfig.from_yaml(path)


def get_test_config(num_antennas: int = DSA110_NUM_ANTENNAS_TEST) -> SimulationConfig:
    """Get a configuration suitable for unit testing.
    
    This is a convenience function that creates a SimulationConfig with
    a reduced antenna count for faster test execution.
    
    Parameters
    ----------
    num_antennas : int, optional
        Number of antennas for testing (default: 8)
    
    Returns
    -------
    SimulationConfig
        Configuration with reduced antenna count.
    """
    return SimulationConfig.for_testing(num_antennas)


# Export all public symbols
__all__ = [
    # Primary configuration
    "SimulationConfig",
    "get_config",
    "get_test_config",
    
    # Constants - Array
    "DSA110_NUM_ANTENNAS",
    "DSA110_NUM_ANTENNAS_TEST",
    
    # Constants - Temporal
    "DSA110_INTEGRATION_TIME_SEC",
    "DSA110_NUM_INTEGRATIONS",
    "DSA110_TOTAL_DURATION_SEC",
    
    # Constants - Spectral
    "DSA110_NUM_SUBBANDS",
    "DSA110_CHANNELS_PER_SUBBAND",
    "DSA110_TOTAL_CHANNELS",
    "DSA110_CHANNEL_WIDTH_HZ",
    "DSA110_SUBBAND_WIDTH_HZ",
    "DSA110_TOTAL_BANDWIDTH_HZ",
    "DSA110_FREQ_MIN_HZ",
    "DSA110_FREQ_MAX_HZ",
    "DSA110_REFERENCE_FREQ_HZ",
    
    # Constants - Site
    "DSA110_LATITUDE_DEG",
    "DSA110_LONGITUDE_DEG",
    "DSA110_ALTITUDE_M",
    
    # Constants - Polarization
    "DSA110_POLARIZATIONS",
    "DSA110_NUM_POLARIZATIONS",
    
    # Constants - Antenna
    "DSA110_DISH_DIAMETER_M",
    
    # Constants - System
    "DSA110_SYSTEM_TEMP_K",
    "DSA110_APERTURE_EFFICIENCY",
]
