"""Tests for ``detect_and_flag_dead_antennas`` pre-calibration wiring.

These tests mock the casa_tables adapter so they run without a real
Measurement Set, and assert both the detection result shape and that
``flag_antenna`` is invoked with the correct antenna selector when
``dry_run=False``.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pytest

from dsa110_continuum.calibration.flagging import detect_and_flag_dead_antennas


def _synthetic_baseline_arrays(
    n_antennas: int,
    target_antenna: int,
    target_flag_fraction: float,
    n_chan: int = 4,
    n_pol: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build cross-correlation ANTENNA1/ANTENNA2/FLAG arrays.

    All ``i<j`` baselines are present. Within the rows where
    ``target_antenna`` participates, exactly ``target_flag_fraction`` of the
    individual (row, chan, pol) cells are flagged (cell-level, so any fraction
    is reachable). Everything outside those rows is unflagged.
    """
    pairs = [(i, j) for i in range(n_antennas) for j in range(i + 1, n_antennas)]
    ant1 = np.array([p[0] for p in pairs], dtype=int)
    ant2 = np.array([p[1] for p in pairs], dtype=int)
    flags = np.zeros((len(pairs), n_chan, n_pol), dtype=bool)

    target_rows = np.where((ant1 == target_antenna) | (ant2 == target_antenna))[0]
    target_block = np.zeros((len(target_rows), n_chan, n_pol), dtype=bool)
    n_to_flag = int(round(target_block.size * target_flag_fraction))
    target_block.reshape(-1)[:n_to_flag] = True
    flags[target_rows] = target_block
    return ant1, ant2, flags


class _MockTable:
    """Context-manager double for ``casa_tables.table``."""

    def __init__(self, ant1: np.ndarray, ant2: np.ndarray, flags: np.ndarray) -> None:
        self._ant1 = ant1
        self._ant2 = ant2
        self._flags = flags

    def __enter__(self) -> "_MockTable":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def nrows(self) -> int:
        return len(self._ant1)

    def getcol(self, name: str) -> np.ndarray:
        if name == "ANTENNA1":
            return self._ant1
        if name == "ANTENNA2":
            return self._ant2
        if name == "FLAG":
            return self._flags
        raise KeyError(name)


def _patch_casa_tables(ant1: np.ndarray, ant2: np.ndarray, flags: np.ndarray):
    """Patch the ``table`` class on the casa_tables adapter to return a mock."""
    def _factory(*_args, **_kwargs):
        return _MockTable(ant1, ant2, flags)

    return patch("dsa110_continuum.adapters.casa_tables.table", new=_factory)


class TestDetectAndFlagDeadAntennas:
    def test_dry_run_detects_one_dead_antenna_at_97pct(self, tmp_path):
        ms_path = str(tmp_path / "fake.ms")
        ant1, ant2, flags = _synthetic_baseline_arrays(
            n_antennas=10, target_antenna=3, target_flag_fraction=0.97
        )

        with _patch_casa_tables(ant1, ant2, flags):
            result = detect_and_flag_dead_antennas(
                ms_path, threshold=0.95, dry_run=True
            )

        assert result["n_dead"] == 1, result
        assert result["dead_antennas"] == [3]
        assert result["action_taken"] is False
        # Dry run does not modify the MS, so before == after.
        assert result["total_flagged_after"] == result["total_flagged_before"]
        # Per-antenna stats are reported for every participating antenna.
        assert set(result["antenna_stats"].keys()) == set(range(10))
        # Sidecar JSON report is written next to the MS.
        report_path = tmp_path / "fake_antenna_health.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["n_dead"] == 1
        assert report["dry_run"] is True

    def test_non_dry_run_invokes_flag_antenna_with_correct_selector(self, tmp_path):
        ms_path = str(tmp_path / "fake.ms")
        ant1, ant2, flags = _synthetic_baseline_arrays(
            n_antennas=10, target_antenna=5, target_flag_fraction=0.97
        )

        with _patch_casa_tables(ant1, ant2, flags), patch(
            "dsa110_continuum.calibration.flagging.flag_antenna"
        ) as mock_flag_antenna:
            result = detect_and_flag_dead_antennas(
                ms_path, threshold=0.95, dry_run=False
            )

        mock_flag_antenna.assert_called_once()
        args, _kwargs = mock_flag_antenna.call_args
        assert args[0] == ms_path
        assert args[1] == "5"
        assert result["n_dead"] == 1
        assert result["dead_antennas"] == [5]
        assert result["action_taken"] is True

    def test_no_dead_antennas_does_not_call_flag_antenna(self, tmp_path):
        ms_path = str(tmp_path / "fake.ms")
        # All antennas at most 30% flagged → none above 95% threshold.
        ant1, ant2, flags = _synthetic_baseline_arrays(
            n_antennas=10, target_antenna=2, target_flag_fraction=0.30
        )

        with _patch_casa_tables(ant1, ant2, flags), patch(
            "dsa110_continuum.calibration.flagging.flag_antenna"
        ) as mock_flag_antenna:
            result = detect_and_flag_dead_antennas(
                ms_path, threshold=0.95, dry_run=False
            )

        mock_flag_antenna.assert_not_called()
        assert result["n_dead"] == 0
        assert result["dead_antennas"] == []
        assert result["action_taken"] is False

    def test_threshold_boundary_just_below_does_not_flag(self, tmp_path):
        """An antenna at 90% flagged is partial, not dead, at threshold=0.95."""
        ms_path = str(tmp_path / "fake.ms")
        ant1, ant2, flags = _synthetic_baseline_arrays(
            n_antennas=10, target_antenna=4, target_flag_fraction=0.90
        )

        with _patch_casa_tables(ant1, ant2, flags), patch(
            "dsa110_continuum.calibration.flagging.flag_antenna"
        ) as mock_flag_antenna:
            result = detect_and_flag_dead_antennas(
                ms_path, threshold=0.95, dry_run=False
            )

        mock_flag_antenna.assert_not_called()
        assert result["n_dead"] == 0
        assert 4 in result["partial_antennas"]
        # Antenna 4 should be reported with frac in [0.5, 0.95).
        assert 0.5 <= result["antenna_stats"][4] < 0.95
