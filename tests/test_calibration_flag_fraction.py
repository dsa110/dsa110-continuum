"""Tests for receptor-aware FLAG fraction QA helper."""

from __future__ import annotations

import numpy as np
import pytest

from dsa110_continuum.calibration.calibration import (
    _flag_fraction_excluding_dead_receptors,
)


N_ANTENNAS = 117
N_SPWS = 16
N_ROWS = N_ANTENNAS * N_SPWS  # 1872
N_RECEPTORS = 2
N_CHANNELS = 48


def _antenna_ids() -> np.ndarray:
    return np.repeat(np.arange(N_ANTENNAS), N_SPWS)


def _empty_flags() -> np.ndarray:
    return np.zeros((N_ROWS, N_RECEPTORS, N_CHANNELS), dtype=bool)


class TestFlagFractionExcludingDeadReceptors:
    def test_realistic_bandpass_shape_no_dead_receptors(self):
        flags = _empty_flags()
        antenna_ids = _antenna_ids()

        result = _flag_fraction_excluding_dead_receptors(flags, antenna_ids)

        assert result["effective_flag_fraction"] == 0.0
        assert result["dead_receptor_count"] == 0
        assert result["dead_antenna_count"] == 0
        assert result["working_receptor_count"] == N_ANTENNAS * N_RECEPTORS

    def test_one_dead_receptor_excluded(self):
        flags = _empty_flags()
        antenna_ids = _antenna_ids()
        rows_for_ant_3 = np.where(antenna_ids == 3)[0]
        flags[rows_for_ant_3, 0, :] = True

        result = _flag_fraction_excluding_dead_receptors(flags, antenna_ids)

        assert result["dead_receptor_count"] == 1
        assert result["dead_antenna_count"] == 1
        assert result["effective_flag_fraction"] == pytest.approx(0.0)
        assert result["working_receptor_count"] == N_ANTENNAS * N_RECEPTORS - 1

    def test_both_receptors_dead_one_antenna(self):
        flags = _empty_flags()
        antenna_ids = _antenna_ids()
        rows_for_ant_5 = np.where(antenna_ids == 5)[0]
        flags[rows_for_ant_5, :, :] = True

        result = _flag_fraction_excluding_dead_receptors(flags, antenna_ids)

        assert result["dead_receptor_count"] == 2
        assert result["dead_antenna_count"] == 1
        assert result["working_receptor_count"] == N_ANTENNAS * N_RECEPTORS - 2
        assert result["effective_flag_fraction"] == pytest.approx(0.0)

    def test_does_not_misread_axis_0_as_antennas(self):
        flags = _empty_flags()
        antenna_ids = _antenna_ids()
        rows_one_per_antenna = np.arange(N_ANTENNAS) * N_SPWS
        flagged_rows = rows_one_per_antenna[:116]
        flags[flagged_rows, :, :] = True

        result = _flag_fraction_excluding_dead_receptors(flags, antenna_ids)

        assert result["dead_antenna_count"] == 0
        assert result["dead_receptor_count"] == 0

    def test_partial_flagging_in_one_receptor(self):
        flags = _empty_flags()
        antenna_ids = _antenna_ids()
        rows_for_ant_7 = np.where(antenna_ids == 7)[0]
        flags[rows_for_ant_7, 0, : N_CHANNELS // 2] = True

        result = _flag_fraction_excluding_dead_receptors(flags, antenna_ids)

        assert result["dead_receptor_count"] == 0
        assert result["dead_antenna_count"] == 0
        assert result["working_receptor_count"] == N_ANTENNAS * N_RECEPTORS

        flagged_per_partial_receptor = N_SPWS * (N_CHANNELS // 2)
        total_per_receptor = N_SPWS * N_CHANNELS
        expected_fraction = flagged_per_partial_receptor / (
            (N_ANTENNAS * N_RECEPTORS) * total_per_receptor
        )
        assert result["effective_flag_fraction"] == pytest.approx(expected_fraction)
        assert result["working_flagged"] == flagged_per_partial_receptor
        assert result["working_total"] == N_ANTENNAS * N_RECEPTORS * total_per_receptor
