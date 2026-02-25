"""
Calibration configuration for DSA-110.

This module provides the calibration configuration for DSA-110. There is ONE
configuration optimized for maximum precision based on empirical validation.

Usage:
    from dsa110_contimg.core.calibration.presets import CalibrationPreset, DEFAULT_PRESET

    # Use the default preset
    params = DEFAULT_PRESET.to_dict()

    # Customize if needed (rare)
    custom = DEFAULT_PRESET.with_overrides(refant="105", calibrator_name="3C454.3")

    # Pass to pipeline stage
    context.inputs["calibration_params"] = custom.to_dict()

DSA-110 Calibration Notes:
- K-calibration (delay) is NOT needed - DSA-110 has stable delays
- Empirical validation (2026-02-02) shows phase slopes <3° across band
- Pre-bandpass phase solve improves bandpass quality
- Full amplitude+phase gain calibration provides best results
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass
class CalibrationPreset:
    """Calibration configuration preset.

    Encapsulates all calibration parameters used by CalibrationSolveStage.
    Provides a single source of truth for calibration configurations.

    """

    # General parameters
    field: str = "0~23"  # All 24 fields by default (DSA-110 drift-scan)
    refant: str = "auto"  # Auto-select best outrigger based on MS antenna health
    model_source: str = "catalog"  # Use VLA calibrator catalog
    calibrator_name: str | None = None  # Required for catalog mode

    # Stage control flags
    solve_delay: bool = False  # K calibration (typically skip for DSA-110)
    solve_bandpass: bool = True  # BP calibration (essential)
    solve_gains: bool = True  # G calibration (essential)

    # Phaseshift support (DSA-110 specific)
    do_phaseshift: bool = True  # Always phaseshift for DSA-110

    # K (delay) parameters
    k_combine_spw: bool = False
    k_t_slow: str = "inf"
    k_t_fast: str = "60s"
    k_uvrange: str = ""
    k_minsnr: float = 3.0
    k_skip_slow: bool = False

    # Pre-bandpass phase parameters
    prebp_phase: bool = False  # Enable if data has strong phase drifts
    prebp_uvrange: str = ""
    prebp_minsnr: float = 3.0

    # BP (bandpass) parameters
    bp_combine_field: bool = True  # Combine all fields for max SNR
    bp_combine_spw: bool = True  # Recommended: combine SPWs
    bp_model_standard: str = "Perley-Butler 2017"
    bp_minsnr: float = 3.0
    bp_uvrange: str = ""
    bp_smooth_type: str | None = None  # "median", "mean", or None
    bp_smooth_window: int | None = None

    # G (gain) parameters
    gain_solint: str = "inf"  # Infinite solution interval (solve per scan)
    gain_calmode: str = "ap"  # Amplitude + phase by default
    gain_t_short: str = "60s"
    gain_minsnr: float = 3.0
    gain_uvrange: str = ""

    # Flagging parameters
    do_flagging: bool = True
    flag_autocorr: bool = True
    use_adaptive_flagging: bool = True
    use_gpu_rfi: bool = True  # GPU-accelerated RFI detection (CuPy/CUDA)

    # Fast mode
    fast: bool = False

    def __post_init__(self):
        """Validate parameters after initialization."""
        if not self.refant:
            raise ValueError("Reference antenna (refant) must be specified (use 'auto' for automatic selection)")

        # Validate numeric ranges
        for field_name in ["bp_minsnr", "gain_minsnr", "k_minsnr", "prebp_minsnr"]:
            val = getattr(self, field_name)
            if val <= 0:
                raise ValueError(f"{field_name} must be positive, got {val}")

        # Validate gain calibration mode
        valid_modes = {"p", "a", "ap"}
        if self.gain_calmode not in valid_modes:
            raise ValueError(
                f"gain_calmode must be one of {valid_modes}, got '{self.gain_calmode}'"
            )

        # Validate smoothing type
        if self.bp_smooth_type and self.bp_smooth_type not in {"median", "mean"}:
            raise ValueError(
                f"bp_smooth_type must be 'median', 'mean' or None, got '{self.bp_smooth_type}'"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert preset to dictionary suitable for pipeline stage inputs."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:  # Skip None values
                result[key] = value
        return result

    def with_overrides(self, **kwargs: Any) -> CalibrationPreset:
        """Create a new preset with specified parameters overridden.

        Parameters
        ----------
        **kwargs : Any
            Parameters to override (e.g., refant="105", gain_solint="60s")

        Returns
        -------
        CalibrationPreset
            New CalibrationPreset instance with overrides applied.

        Examples
        --------
        custom = PRESET_STANDARD.with_overrides(
            refant="105",
            gain_solint="60s",
            calibrator_name="3C286"
        )
        """
        return replace(self, **kwargs)


# =============================================================================
# DSA-110 Calibration Configuration
# =============================================================================

# The single, most-precise calibration configuration for DSA-110.
#
# This configuration is based on empirical validation (2026-02-02) on 3C454.3 data:
#   - K-calibration (delay) is NOT needed: phase slopes <3° across band
#   - Overall phase std ~27° after calibration (acceptable for DSA-110)
#   - Baselines 0-10, 0-50, 0-90 all show <3° total drift
#   - DSA-110 correlator and cables have stable delays
#
# Configuration choices for maximum precision:
#   - Pre-bandpass phase solve (prebp_phase=True) improves BP quality
#   - Full amplitude+phase gain calibration (gain_calmode="ap")
#   - Combine all fields and SPWs for maximum SNR
#   - GPU-accelerated RFI flagging (CuPy/CUDA) for faster processing
#   - Adaptive flagging enabled for RFI mitigation
DEFAULT_PRESET = CalibrationPreset(
    field="0~23",  # All 24 fields (5-minute observation)
    refant="103",  # Reference antenna (override with "auto" for automatic selection)
    calibrator_name=None,  # Must be specified at runtime
    solve_delay=False,  # K-cal NOT needed for DSA-110 (stable delays)
    solve_bandpass=True,  # Essential: frequency-dependent gains
    solve_gains=True,  # Essential: time-dependent gains
    gain_calmode="ap",  # Full amplitude + phase for best precision
    gain_solint="inf",  # One solution per scan (sufficient for DSA-110)
    prebp_phase=True,  # Pre-BP phase solve for improved BP quality
    prebp_minsnr=3.0,
    bp_minsnr=5.0,  # Bandpass SNR threshold
    gain_minsnr=3.0,  # Gain SNR threshold
    bp_combine_field=True,  # Combine all fields for max SNR
    bp_combine_spw=True,  # Combine SPWs for max SNR
    do_flagging=True,  # Enable flagging
    flag_autocorr=True,  # Flag autocorrelations
    use_adaptive_flagging=True,  # Adaptive RFI flagging
    use_gpu_rfi=True,  # GPU-accelerated RFI detection (CuPy/CUDA)
)

# Backwards compatibility aliases
PRESET_STANDARD = DEFAULT_PRESET
PRESET_DEFAULT = DEFAULT_PRESET


# =============================================================================
# Preset Registry (simplified)
# =============================================================================

# Single preset for DSA-110
PRESETS: dict[str, CalibrationPreset] = {
    "default": DEFAULT_PRESET,
    "standard": DEFAULT_PRESET,  # Alias for backwards compatibility
}


def get_preset(name: str = "default") -> CalibrationPreset:
    """Get the DSA-110 calibration preset.

    Parameters
    ----------
    name : str, optional
        Preset name. Only "default" and "standard" are valid (both return
        the same configuration). Default is "default".

    Returns
    -------
    CalibrationPreset
        The DSA-110 calibration configuration.

    Raises
    ------
    KeyError
        If preset name not found.

    Examples
    --------
    >>> preset = get_preset()
    >>> preset = get_preset("default")
    """
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[name]


def list_presets() -> list[str]:
    """List available preset names."""
    return list(PRESETS.keys())
