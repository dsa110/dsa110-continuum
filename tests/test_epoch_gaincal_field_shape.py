"""Regression tests for the FIELD-shape fix in epoch_gaincal._read_ms_phase_center.

Pre-fix bug: `_read_ms_phase_center` did `phase_dir[:, 0, 1]` on the result of
`getcol("PHASE_DIR")`, assuming rows-first shape `(nfields, 1, 2)`. When CASA's
table backend returned the column-major shape `(nfields, 2, 1)`, the index `1`
on axis 2 (size 1) raised ``IndexError: index 1 is out of bounds for axis 2
with size 1``. That cascaded into the orchestrator's epoch-gaincal fallback
path on real DSA-110 MS files freshly converted by the CASA adapter.

Post-fix: `_read_ms_phase_center` calls
`dsa110_continuum.calibration.runner._extract_field_ra_dec`, which is shape-
tolerant (handles both rows-first and column-major + the 2-D fallback).
"""

from __future__ import annotations

import numpy as np


def _make_fake_table(phase_dir: np.ndarray):
    """Return a fake casa_tables.table replacement that returns *phase_dir*.

    Mirrors the mock pattern in tests/test_skymodel_phase_dir.py — cloud-safe,
    no real CASA needed.
    """

    class FakeTable:
        def __init__(self, path, *_, **__):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def colnames(self):
            return ["PHASE_DIR"]

        def getcol(self, name):
            assert name == "PHASE_DIR"
            return phase_dir

    return FakeTable


def test_read_ms_phase_center_rows_first_shape(monkeypatch):
    """`(nfields, 1, 2)` — the historical rows-first shape, must keep working."""
    from dsa110_continuum.adapters import casa_tables
    from dsa110_continuum.calibration import epoch_gaincal

    phase_dir = np.array(
        [
            [[np.radians(180.0), np.radians(22.0)]],
            [[np.radians(181.0), np.radians(22.5)]],
        ]
    )  # shape (2, 1, 2)
    monkeypatch.setattr(casa_tables, "table", _make_fake_table(phase_dir))

    ra_deg, dec_deg = epoch_gaincal._read_ms_phase_center("/fake/ms")

    expected_ra = np.degrees(np.angle(np.mean(np.exp(1j * np.radians([180.0, 181.0]))))) % 360
    np.testing.assert_allclose(ra_deg, expected_ra)
    np.testing.assert_allclose(dec_deg, np.degrees(np.median([np.radians(22.0), np.radians(22.5)])))


def test_read_ms_phase_center_casa_column_major_shape(monkeypatch):
    """`(nfields, 2, 1)` — the CASA column-major shape that broke pre-fix.

    This is the canonical regression test for the smoke-test failure on
    2026-01-25 (run_2026-04-29T18_44_03Z.log line 256:
    `Epoch gaincal: FAILED (index 1 is out of bounds for axis 2 with size 1)`).
    """
    from dsa110_continuum.adapters import casa_tables
    from dsa110_continuum.calibration import epoch_gaincal

    phase_dir = np.array(
        [
            [[np.radians(180.0)], [np.radians(22.0)]],
            [[np.radians(181.0)], [np.radians(22.5)]],
        ]
    )  # shape (2, 2, 1)
    monkeypatch.setattr(casa_tables, "table", _make_fake_table(phase_dir))

    # Pre-fix this raised IndexError. Post-fix it returns plausible values.
    ra_deg, dec_deg = epoch_gaincal._read_ms_phase_center("/fake/ms")

    expected_ra = np.degrees(np.angle(np.mean(np.exp(1j * np.radians([180.0, 181.0]))))) % 360
    np.testing.assert_allclose(ra_deg, expected_ra)
    np.testing.assert_allclose(dec_deg, np.degrees(np.median([np.radians(22.0), np.radians(22.5)])))


def test_read_ms_phase_center_2d_fallback_shape(monkeypatch):
    """`(nfields, 2)` — the helper's third supported shape."""
    from dsa110_continuum.adapters import casa_tables
    from dsa110_continuum.calibration import epoch_gaincal

    phase_dir = np.array(
        [
            [np.radians(180.0), np.radians(22.0)],
            [np.radians(181.0), np.radians(22.5)],
        ]
    )  # shape (2, 2)
    monkeypatch.setattr(casa_tables, "table", _make_fake_table(phase_dir))

    ra_deg, dec_deg = epoch_gaincal._read_ms_phase_center("/fake/ms")

    expected_ra = np.degrees(np.angle(np.mean(np.exp(1j * np.radians([180.0, 181.0]))))) % 360
    np.testing.assert_allclose(ra_deg, expected_ra)
    np.testing.assert_allclose(dec_deg, np.degrees(np.median([np.radians(22.0), np.radians(22.5)])))


def test_read_ms_phase_center_unsupported_shape_raises(monkeypatch):
    """Helper's defensive ValueError surfaces cleanly for unrecognized shapes."""
    import pytest
    from dsa110_continuum.adapters import casa_tables
    from dsa110_continuum.calibration import epoch_gaincal

    phase_dir = np.zeros((2, 3, 4))  # not a recognized PHASE_DIR shape
    monkeypatch.setattr(casa_tables, "table", _make_fake_table(phase_dir))

    with pytest.raises(ValueError, match="Unsupported FIELD direction column shape"):
        epoch_gaincal._read_ms_phase_center("/fake/ms")
