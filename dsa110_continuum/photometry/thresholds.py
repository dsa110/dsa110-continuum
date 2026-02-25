"""Configurable threshold presets for ESE detection.

This module provides predefined threshold presets (conservative, moderate, sensitive)
for ESE detection, allowing users to easily select appropriate sensitivity levels.
"""

from __future__ import annotations

from enum import Enum


class ThresholdPreset(str, Enum):
    """Enumeration of available threshold presets."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    SENSITIVE = "sensitive"


# Threshold preset definitions
PRESET_THRESHOLDS = {
    ThresholdPreset.CONSERVATIVE: {
        "min_sigma": 5.0,
        "min_chi2_nu": 4.0,
        "min_eta": 3.0,
    },
    ThresholdPreset.MODERATE: {
        "min_sigma": 3.5,
        "min_chi2_nu": 2.5,
        "min_eta": 2.0,
    },
    ThresholdPreset.SENSITIVE: {
        "min_sigma": 2.5,
        "min_chi2_nu": 1.5,
        "min_eta": 1.0,
    },
}


def get_threshold_preset(
    preset: ThresholdPreset | str | dict[str, float],
) -> dict[str, float]:
    """Get threshold preset values.

    Parameters
    ----------
    preset : ThresholdPreset, str, or dict
        Either a ThresholdPreset enum, preset name string, or custom dict

    Returns
    -------
        dict
        Dictionary of threshold values

    Raises
    ------
        ValueError
        If preset name is invalid

        Example
    -------
        >>> # Get conservative preset
        >>> thresholds = get_threshold_preset(ThresholdPreset.CONSERVATIVE)
        >>> print(f"Min sigma: {thresholds['min_sigma']}")

        >>> # Get by name
        >>> thresholds = get_threshold_preset("moderate")

        >>> # Use custom thresholds
        >>> custom = {'min_sigma': 4.5, 'min_chi2_nu': 3.5, 'min_eta': 2.5}
        >>> thresholds = get_threshold_preset(custom)
    """
    # If it's already a dict, return as-is
    if isinstance(preset, dict):
        return preset.copy()

    # Convert string to enum if needed
    if isinstance(preset, str):
        try:
            preset = ThresholdPreset(preset.lower())
        except ValueError:
            raise ValueError(
                f"Invalid preset name: {preset}. "
                f"Valid options: {[p.value for p in ThresholdPreset]}"
            )

    # Get preset thresholds
    if preset not in PRESET_THRESHOLDS:
        raise ValueError(f"Preset {preset} not found in PRESET_THRESHOLDS")

    return PRESET_THRESHOLDS[preset].copy()
