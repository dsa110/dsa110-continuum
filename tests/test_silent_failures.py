"""Tests for CLAUDE.md critical silent failures.

These three bugs produce no runtime error but yield wrong science output.
Each test creates a minimal mock MS (FIELD/OBSERVATION tables only) and
verifies the fix functions produce the correct metadata state.

Requires: casacore (available in casa6 env).
"""
import sys
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Skip entire module if casatools is not installed (e.g., CI without CASA env)
# The adapter module itself is always importable; we need the backing library.
# ---------------------------------------------------------------------------
pytest.importorskip("casatools", reason="casatools (modular CASA 6) required for MS table tests")
import dsa110_continuum.adapters.casa_tables as ct  # noqa: E402


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_ms(tmp_path):
    """Create a minimal Measurement Set with FIELD and OBSERVATION subtables.

    The MS has 3 fields with PHASE_DIR at (10°, 20°), (11°, 21°), (12°, 22°)
    and REFERENCE_DIR at a deliberately different position (0°, 0°).
    OBSERVATION::TELESCOPE_NAME is set to 'OVRO_MMA' (post-merge_spws state).
    """
    ms_path = str(tmp_path / "test.ms")

    # Create main table (empty, just needs to exist)
    main_tb = ct.default_ms(ms_path)
    main_tb.close()

    # Populate FIELD table with 3 fields
    field_path = f"{ms_path}/FIELD"
    with ct.table(field_path, readonly=False) as tb:
        nfields = 3
        # PHASE_DIR: shape (nfields, 1, 2) in radians
        phase_dir = np.zeros((nfields, 1, 2), dtype=np.float64)
        ref_dir = np.zeros((nfields, 1, 2), dtype=np.float64)
        for i in range(nfields):
            phase_dir[i, 0, 0] = np.radians(10.0 + i)   # RA
            phase_dir[i, 0, 1] = np.radians(20.0 + i)   # Dec
            # REFERENCE_DIR deliberately differs (simulates pre-sync state)
            ref_dir[i, 0, 0] = 0.0
            ref_dir[i, 0, 1] = 0.0

        # Ensure we have 3 rows
        while tb.nrows() < nfields:
            tb.addrows(1)
        tb.putcol("PHASE_DIR", phase_dir)
        tb.putcol("REFERENCE_DIR", ref_dir)

    # Populate OBSERVATION table
    obs_path = f"{ms_path}/OBSERVATION"
    with ct.table(obs_path, readonly=False) as tb:
        if tb.nrows() < 1:
            tb.addrows(1)
        tb.putcol("TELESCOPE_NAME", ["OVRO_MMA"])

    return ms_path


# ── Test 1: PHASE_DIR updated after chgcentre ────────────────────────────────

class TestUpdatePhaseDirToTarget:
    """CLAUDE.md critical failure #1: FIELD::PHASE_DIR not updated by chgcentre."""

    def test_phase_dir_set_to_target(self, tmp_ms):
        """After update_phase_dir_to_target, all fields should point at the target."""
        from dsa110_continuum.calibration.runner import update_phase_dir_to_target

        target_ra, target_dec = 180.0, 45.0
        update_phase_dir_to_target(tmp_ms, target_ra, target_dec)

        with ct.table(f"{tmp_ms}::FIELD") as tb:
            phase_dir = tb.getcol("PHASE_DIR")

        expected_ra = np.radians(target_ra)
        expected_dec = np.radians(target_dec)

        for i in range(len(phase_dir)):
            np.testing.assert_allclose(phase_dir[i, 0, 0], expected_ra, atol=1e-10,
                                       err_msg=f"Field {i} RA not updated")
            np.testing.assert_allclose(phase_dir[i, 0, 1], expected_dec, atol=1e-10,
                                       err_msg=f"Field {i} Dec not updated")

    def test_all_fields_uniform_after_update(self, tmp_ms):
        """All fields should have identical PHASE_DIR after update (rephased to single target)."""
        from dsa110_continuum.calibration.runner import update_phase_dir_to_target

        update_phase_dir_to_target(tmp_ms, 90.0, -30.0)

        with ct.table(f"{tmp_ms}::FIELD") as tb:
            phase_dir = tb.getcol("PHASE_DIR")

        # All rows should be identical
        for i in range(1, len(phase_dir)):
            np.testing.assert_array_equal(phase_dir[0], phase_dir[i],
                                          err_msg=f"Field {i} differs from field 0")

    def test_reference_dir_unchanged(self, tmp_ms):
        """update_phase_dir_to_target should NOT touch REFERENCE_DIR."""
        from dsa110_continuum.calibration.runner import update_phase_dir_to_target

        with ct.table(f"{tmp_ms}::FIELD") as tb:
            ref_before = tb.getcol("REFERENCE_DIR").copy()

        update_phase_dir_to_target(tmp_ms, 180.0, 45.0)

        with ct.table(f"{tmp_ms}::FIELD") as tb:
            ref_after = tb.getcol("REFERENCE_DIR")

        np.testing.assert_array_equal(ref_before, ref_after,
                                      err_msg="REFERENCE_DIR was modified unexpectedly")


# ── Test 2: REFERENCE_DIR synced with PHASE_DIR ──────────────────────────────

class TestSyncReferenceDirWithPhaseDir:
    """CLAUDE.md critical failure #2: REFERENCE_DIR must match PHASE_DIR for ft()."""

    def test_reference_dir_matches_phase_dir_after_sync(self, tmp_ms):
        """After sync, REFERENCE_DIR should exactly equal PHASE_DIR."""
        from dsa110_continuum.calibration.runner import sync_reference_dir_with_phase_dir

        sync_reference_dir_with_phase_dir(tmp_ms)

        with ct.table(f"{tmp_ms}::FIELD") as tb:
            phase_dir = tb.getcol("PHASE_DIR")
            ref_dir = tb.getcol("REFERENCE_DIR")

        np.testing.assert_array_equal(phase_dir, ref_dir,
                                      err_msg="REFERENCE_DIR != PHASE_DIR after sync")

    def test_sync_after_phase_update_propagates(self, tmp_ms):
        """Full workflow: update PHASE_DIR then sync → REFERENCE_DIR should match new target."""
        from dsa110_continuum.calibration.runner import (
            update_phase_dir_to_target,
            sync_reference_dir_with_phase_dir,
        )

        target_ra, target_dec = 270.0, -10.0
        update_phase_dir_to_target(tmp_ms, target_ra, target_dec)
        sync_reference_dir_with_phase_dir(tmp_ms)

        with ct.table(f"{tmp_ms}::FIELD") as tb:
            ref_dir = tb.getcol("REFERENCE_DIR")

        expected_ra = np.radians(target_ra)
        expected_dec = np.radians(target_dec)

        for i in range(len(ref_dir)):
            np.testing.assert_allclose(ref_dir[i, 0, 0], expected_ra, atol=1e-10)
            np.testing.assert_allclose(ref_dir[i, 0, 1], expected_dec, atol=1e-10)

    def test_desync_detectable_before_fix(self, tmp_ms):
        """Before sync, REFERENCE_DIR should NOT match PHASE_DIR (confirming the bug scenario)."""
        with ct.table(f"{tmp_ms}::FIELD") as tb:
            phase_dir = tb.getcol("PHASE_DIR")
            ref_dir = tb.getcol("REFERENCE_DIR")

        # The fixture deliberately sets them differently
        assert not np.allclose(phase_dir, ref_dir), \
            "Fixture should create a desync'd state to test the fix"


# ── Test 3: TELESCOPE_NAME must be DSA_110 for EveryBeam ─────────────────────

class TestSetMsTelescopeName:
    """CLAUDE.md critical failure #3: TELESCOPE_NAME must be DSA_110 before WSClean."""

    def test_telescope_name_set_to_dsa110(self, tmp_ms):
        """set_ms_telescope_name should overwrite OVRO_MMA with DSA_110."""
        from dsa110_continuum.conversion.helpers_telescope import set_ms_telescope_name

        # Precondition: fixture sets it to OVRO_MMA (post-merge_spws state)
        with ct.table(f"{tmp_ms}/OBSERVATION") as tb:
            assert tb.getcol("TELESCOPE_NAME")[0] == "OVRO_MMA"

        set_ms_telescope_name(tmp_ms, name="DSA_110")

        with ct.table(f"{tmp_ms}/OBSERVATION") as tb:
            assert tb.getcol("TELESCOPE_NAME")[0] == "DSA_110"

    def test_idempotent_when_already_correct(self, tmp_ms):
        """Calling set_ms_telescope_name when already DSA_110 should be a no-op."""
        from dsa110_continuum.conversion.helpers_telescope import set_ms_telescope_name

        set_ms_telescope_name(tmp_ms, name="DSA_110")
        set_ms_telescope_name(tmp_ms, name="DSA_110")  # second call

        with ct.table(f"{tmp_ms}/OBSERVATION") as tb:
            names = tb.getcol("TELESCOPE_NAME")
            assert len(names) == 1
            assert names[0] == "DSA_110"

    def test_toggle_roundtrip(self, tmp_ms):
        """Simulate merge_spws → WSClean cycle: OVRO_MMA → DSA_110 → OVRO_MMA → DSA_110."""
        from dsa110_continuum.conversion.helpers_telescope import set_ms_telescope_name

        # Start: OVRO_MMA (from fixture)
        set_ms_telescope_name(tmp_ms, name="DSA_110")  # pre-WSClean
        set_ms_telescope_name(tmp_ms, name="OVRO_MMA")  # simulate merge_spws
        set_ms_telescope_name(tmp_ms, name="DSA_110")  # pre-WSClean again

        with ct.table(f"{tmp_ms}/OBSERVATION") as tb:
            assert tb.getcol("TELESCOPE_NAME")[0] == "DSA_110"
