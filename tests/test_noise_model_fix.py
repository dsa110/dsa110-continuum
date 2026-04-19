"""Tests for corrected noise model parameters (96 antennas, 188 MHz bandwidth)."""
import numpy as np
import pytest
from dsa110_continuum.qa.noise_model import calculate_theoretical_rms


def test_default_antenna_count_is_96():
    """Default num_antennas must be 96 (not 117)."""
    import inspect
    sig = inspect.signature(calculate_theoretical_rms)
    assert sig.parameters["num_antennas"].default == 96, (
        f"Expected num_antennas default=96, got {sig.parameters['num_antennas'].default}"
    )


def test_theoretical_rms_96_antennas():
    """Radiometer equation with 96 antennas, 12.88s, 188 MHz → ~9–15 mJy/beam."""
    rms = calculate_theoretical_rms(
        ms_path=None,
        bandwidth_hz=188e6,
        integration_time_s=12.88,
        num_antennas=96,
        sefd_per_element_jy=5800.0,
        efficiency=0.7,
    )
    # With 96 antennas the theoretical noise is ~9-14 mJy/beam depending on t_int
    # N-only formula gives ~12.15 mJy/beam; allow ±40% for parameter variation
    assert 7.0 < rms < 17.0, (
        f"Unexpected theoretical RMS: {rms:.3f} mJy/beam "
        f"(expected ~12 mJy/beam with N=96, BW=188 MHz, t=12.88s, eta=0.7)"
    )


def test_rms_higher_with_fewer_antennas():
    """Fewer antennas → higher predicted noise (monotonic)."""
    rms_96 = calculate_theoretical_rms(
        ms_path=None, bandwidth_hz=188e6, integration_time_s=12.88,
        num_antennas=96, sefd_per_element_jy=5800.0, efficiency=0.7,
    )
    rms_117 = calculate_theoretical_rms(
        ms_path=None, bandwidth_hz=188e6, integration_time_s=12.88,
        num_antennas=117, sefd_per_element_jy=5800.0, efficiency=0.7,
    )
    assert rms_96 > rms_117, "96-antenna noise should be higher than 117-antenna"
    expected_ratio = np.sqrt(117 / 96)  # ~1.104 for N-only formula
    actual_ratio = rms_96 / rms_117
    assert abs(actual_ratio - expected_ratio) < 0.05, (
        f"Scaling ratio {actual_ratio:.4f} deviates from expected {expected_ratio:.4f} "
        f"(N-only formula: rms ∝ 1/sqrt(N))"
    )
