"""Tests for dsa110_continuum.calibration.ensure — automated calibration table generation."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest
from astropy.time import Time
from dsa110_continuum.calibration.ensure import (
    CalibrationError,
    CalibrationResult,
    _build_provenance,
    _enrich_result_provenance,
    _find_nearest_real_tables,
    _parse_dec_deg,
    _parse_ra_deg,
    _validate_strip_compatibility,
    ensure_bandpass,
    find_cal_tables,
    load_provenance_sidecar,
    provenance_sidecar_path,
    resolve_cal_table_paths,
    select_bandpass_calibrator,
    validate_table_strip_compatibility,
    write_provenance_sidecar,
)

# ── Coordinate parsing ──────────────────────────────────────────────────


class TestCoordinateParsing:
    def test_parse_ra_deg(self):
        # 3C286: 13h31m08.288s -> ~202.78 deg
        ra = _parse_ra_deg("13h31m08.288s")
        assert 202.5 < ra < 203.0

    def test_parse_dec_deg(self):
        # 3C286: +30d30m32.96s -> ~30.51 deg
        dec = _parse_dec_deg("+30d30m32.96s")
        assert 30.4 < dec < 30.6


# ── find_cal_tables ──────────────────────────────────────────────────────


class TestFindCalTables:
    def test_returns_none_when_no_tables(self, tmp_path):
        result = find_cal_tables("2026-03-16", str(tmp_path))
        assert result is None

    def test_returns_none_for_symlinks(self, tmp_path):
        """Symlinks should NOT count as 'real' tables."""
        real_bp = tmp_path / "real.b"
        real_bp.mkdir()
        link_bp = tmp_path / "2026-03-16T22:26:05_0~23.b"
        link_bp.symlink_to(real_bp)

        real_g = tmp_path / "real.g"
        real_g.mkdir()
        link_g = tmp_path / "2026-03-16T22:26:05_0~23.g"
        link_g.symlink_to(real_g)

        result = find_cal_tables("2026-03-16", str(tmp_path))
        assert result is None

    def test_returns_result_for_real_tables(self, tmp_path):
        """Real files (not symlinks) should be found."""
        bp = tmp_path / "2026-03-16T04:15:00_0~23.b"
        bp.mkdir()
        g = tmp_path / "2026-03-16T04:15:00_0~23.g"
        g.mkdir()

        result = find_cal_tables("2026-03-16", str(tmp_path))
        assert result is not None
        assert result.source == "existing"
        assert result.cal_date == "2026-03-16"
        assert result.bp_table == str(bp)
        assert result.g_table == str(g)

    def test_finds_tables_with_any_transit_time(self, tmp_path):
        """Should find tables regardless of the transit timestamp."""
        bp = tmp_path / "2026-03-16T13:31:08_0~23.b"
        bp.mkdir()
        g = tmp_path / "2026-03-16T13:31:08_0~23.g"
        g.mkdir()

        result = find_cal_tables("2026-03-16", str(tmp_path))
        assert result is not None
        assert "T13:31:08" in result.bp_table


# ── resolve_cal_table_paths ──────────────────────────────────────────────


class TestResolveCalTablePaths:
    def test_falls_back_to_legacy_when_no_glob_match(self, tmp_path):
        """When no tables exist, falls back to the T22:26:05 convention."""
        bp, g = resolve_cal_table_paths(str(tmp_path), "2026-01-25")
        assert bp == os.path.join(str(tmp_path), "2026-01-25T22:26:05_0~23.b")
        assert g == os.path.join(str(tmp_path), "2026-01-25T22:26:05_0~23.g")

    def test_finds_existing_tables(self, tmp_path):
        """When tables exist, picks the first match."""
        bp = tmp_path / "2026-03-16T04:15:00_0~23.b"
        bp.mkdir()
        g = tmp_path / "2026-03-16T04:15:00_0~23.g"
        g.mkdir()

        bp_path, g_path = resolve_cal_table_paths(str(tmp_path), "2026-03-16")
        assert bp_path == str(bp)
        assert g_path == str(g)

    def test_backward_compat_existing_t222605(self, tmp_path):
        """Existing T22:26:05 tables are found by glob."""
        bp = tmp_path / "2026-01-25T22:26:05_0~23.b"
        bp.mkdir()
        g = tmp_path / "2026-01-25T22:26:05_0~23.g"
        g.mkdir()

        bp_path, g_path = resolve_cal_table_paths(str(tmp_path), "2026-01-25")
        assert bp_path == str(bp)
        assert g_path == str(g)


# ── select_bandpass_calibrator ───────────────────────────────────────────


class TestSelectBandpassCalibrator:
    @staticmethod
    def _all_have_data(db_path, ra_deg, dec_deg, **kwargs):
        """Mock: every calibrator has data on 2026-01-25."""
        return [{"group_id": "2026-01-25T00:00:00", "ra_deg": ra_deg,
                 "dec_deg": dec_deg, "transit_time_iso": "2026-01-25T00:00:00",
                 "delta_minutes": 0.5}]

    def test_brightest_when_no_dec(self):
        """Without obs_dec_deg, picks the brightest calibrator (3C147 at 22.45 Jy)."""
        with patch(
            "dsa110_continuum.calibration.transit.find_transits_for_source"
        ) as mock_find:
            mock_find.side_effect = self._all_have_data

            with tempfile.NamedTemporaryFile(suffix=".sqlite3") as f:
                name, ra, dec, transit = select_bandpass_calibrator(
                    "2026-01-25", db_path=f.name,
                )
                # 3C147 (22.45 Jy) or 3C295 (22.0 Jy) are the brightest
                assert name in ("3C147", "3C295")

    def test_dec_aware_prefers_closest(self):
        """With obs_dec_deg=16 deg, should prefer 3C138 (Dec +16.6 deg) over brighter sources."""
        with patch(
            "dsa110_continuum.calibration.transit.find_transits_for_source"
        ) as mock_find:
            mock_find.side_effect = self._all_have_data

            with tempfile.NamedTemporaryFile(suffix=".sqlite3") as f:
                name, ra, dec, transit = select_bandpass_calibrator(
                    "2026-01-25", db_path=f.name, obs_dec_deg=16.0,
                )
                # 3C138 is at Dec +16.6 deg -- closest to 16 deg
                assert name == "3C138"

    def test_dec_filter_excludes_far_calibrators(self):
        """Tight Dec tolerance should exclude distant calibrators."""
        with patch(
            "dsa110_continuum.calibration.transit.find_transits_for_source"
        ) as mock_find:
            mock_find.side_effect = self._all_have_data

            with tempfile.NamedTemporaryFile(suffix=".sqlite3") as f:
                name, ra, dec, transit = select_bandpass_calibrator(
                    "2026-01-25", db_path=f.name,
                    obs_dec_deg=50.0, dec_tolerance_deg=5.0,
                )
                # Only 3C147 (+49.9 deg), 3C295 (+52.2 deg), 3C196 (+48.2 deg) within 5 deg
                assert name in ("3C147", "3C295", "3C196")
                assert abs(dec - 50.0) < 5.0

    def test_raises_when_no_data(self):
        """Should raise CalibrationError when no calibrators have data."""
        with patch(
            "dsa110_continuum.calibration.transit.find_transits_for_source"
        ) as mock_find:
            mock_find.return_value = []

            with tempfile.NamedTemporaryFile(suffix=".sqlite3") as f:
                with pytest.raises(CalibrationError, match="No primary flux calibrator"):
                    select_bandpass_calibrator("2026-01-25", db_path=f.name)


# ── Provenance sidecar I/O ──────────────────────────────────────────────


class TestProvenanceSidecar:
    def _make_provenance(self, bp_table="/fake/bp.b", **overrides):
        defaults = dict(
            selection_mode="dec_aware",
            obs_dec_deg_used=16.1,
            selection_dec_tolerance_deg=10.0,
            calibrator_name="3C138",
            calibrator_ra_deg=80.29,
            calibrator_dec_deg=16.64,
            calibrator_flux_jy=8.36,
            calibrator_dec_offset_deg=0.54,
            transit_time_iso="2026-01-25T05:21:10",
            source="generated",
            cal_date="2026-01-25",
            bp_table=bp_table,
            g_table=bp_table.replace(".b", ".g"),
        )
        defaults.update(overrides)
        return _build_provenance(**defaults)

    def test_sidecar_path(self):
        assert provenance_sidecar_path("/foo/bar.b") == "/foo/bar.b.cal_provenance.json"

    def test_write_and_load(self, tmp_path):
        bp = str(tmp_path / "2026-01-25T05:21:10_0~23.b")
        os.mkdir(bp)
        prov = self._make_provenance(bp_table=bp)
        write_provenance_sidecar(bp, prov)

        loaded = load_provenance_sidecar(bp)
        assert loaded is not None
        assert loaded["calibrator_name"] == "3C138"
        assert loaded["selection_mode"] == "dec_aware"
        assert loaded["obs_dec_deg_used"] == 16.1

    def test_load_returns_none_when_missing(self, tmp_path):
        assert load_provenance_sidecar(str(tmp_path / "nonexistent.b")) is None

    def test_load_follows_symlink(self, tmp_path):
        """For borrowed (symlinked) tables, follows to the real table's sidecar."""
        # Real table with sidecar
        real_bp = tmp_path / "2026-01-25T05:21:10_0~23.b"
        real_bp.mkdir()
        prov = self._make_provenance(bp_table=str(real_bp))
        write_provenance_sidecar(str(real_bp), prov)

        # Symlink
        link_bp = tmp_path / "2026-01-26T05:21:10_0~23.b"
        link_bp.symlink_to(real_bp)

        loaded = load_provenance_sidecar(str(link_bp))
        assert loaded is not None
        assert loaded["calibrator_name"] == "3C138"

    def test_generated_result_has_provenance(self, tmp_path):
        """A CalibrationResult with source='generated' should carry provenance."""
        prov = self._make_provenance()
        result = CalibrationResult(
            bp_table="/fake/bp.b",
            g_table="/fake/bp.g",
            cal_date="2026-01-25",
            calibrator_name="3C138",
            source="generated",
            provenance=prov,
        )
        assert result.provenance["selection_mode"] == "dec_aware"
        assert result.provenance["calibrator_dec_deg"] == pytest.approx(16.64, abs=0.01)

    def test_dec_aware_provenance_values(self, tmp_path):
        """Verify the provenance dict carries correct Dec-aware selection values."""
        prov = self._make_provenance(
            obs_dec_deg_used=16.1,
            calibrator_dec_deg=16.64,
            calibrator_dec_offset_deg=0.54,
        )
        assert prov["obs_dec_deg_used"] == 16.1
        assert prov["calibrator_dec_offset_deg"] == 0.54
        assert prov["selection_mode"] == "dec_aware"


# ── Strip compatibility validation ───────────────────────────────────────


class TestStripCompatibility:
    def _make_tables_with_provenance(
        self, tmp_path, date, obs_dec_used=16.1, cal_dec=16.64, source="existing",
    ):
        """Build tables and sidecar; *obs_dec_used* controls strip provenance."""
        bp = tmp_path / f"{date}T05:21:10_0~23.b"
        bp.mkdir()
        g = tmp_path / f"{date}T05:21:10_0~23.g"
        g.mkdir()

        prov = _build_provenance(
            selection_mode="dec_aware",
            obs_dec_deg_used=obs_dec_used,
            selection_dec_tolerance_deg=10.0,
            calibrator_name="3C138",
            calibrator_ra_deg=80.29,
            calibrator_dec_deg=cal_dec,
            calibrator_flux_jy=8.36,
            calibrator_dec_offset_deg=abs(cal_dec - obs_dec_used),
            transit_time_iso=f"{date}T05:21:10",
            source="generated",
            cal_date=date,
            bp_table=str(bp),
            g_table=str(g),
        )
        write_provenance_sidecar(str(bp), prov)

        return CalibrationResult(
            bp_table=str(bp),
            g_table=str(g),
            cal_date=date,
            calibrator_name="3C138",
            source=source,
        )

    def test_compatible_existing_accepted(self, tmp_path):
        """Same-date existing tables with compatible provenance pass validation."""
        result = self._make_tables_with_provenance(tmp_path, "2026-01-25", obs_dec_used=16.1)
        validated = _validate_strip_compatibility(result, obs_dec_deg=16.1)
        assert validated.bp_table == result.bp_table

    def test_incompatible_existing_rejected(self, tmp_path):
        """Same-date existing tables at wrong strip Dec are rejected."""
        # Tables were generated for strip Dec 50.0, but we're observing at 16.1
        result = self._make_tables_with_provenance(tmp_path, "2026-01-25", obs_dec_used=50.0)
        with pytest.raises(CalibrationError, match="Existing cal tables"):
            _validate_strip_compatibility(result, obs_dec_deg=16.1)

    def test_borrowed_with_compatible_provenance(self, tmp_path):
        """Borrowed tables with compatible provenance pass validation."""
        real_result = self._make_tables_with_provenance(
            tmp_path, "2026-01-25", obs_dec_used=16.1, source="existing",
        )

        link_bp = tmp_path / "2026-01-26T05:21:10_0~23.b"
        link_bp.symlink_to(real_result.bp_table)
        link_g = tmp_path / "2026-01-26T05:21:10_0~23.g"
        link_g.symlink_to(real_result.g_table)

        borrowed = CalibrationResult(
            bp_table=str(link_bp),
            g_table=str(link_g),
            cal_date="2026-01-25",
            calibrator_name="3C138",
            source="borrowed",
        )
        validated = _validate_strip_compatibility(borrowed, obs_dec_deg=16.1)
        assert validated.bp_table == str(link_bp)

    def test_borrowed_incompatible_rejected(self, tmp_path):
        """Borrowed tables with incompatible strip Dec are rejected."""
        # Tables were generated for strip Dec 50.0
        real_result = self._make_tables_with_provenance(
            tmp_path, "2026-01-25", obs_dec_used=50.0, source="existing",
        )

        link_bp = tmp_path / "2026-01-26T05:21:10_0~23.b"
        link_bp.symlink_to(real_result.bp_table)
        link_g = tmp_path / "2026-01-26T05:21:10_0~23.g"
        link_g.symlink_to(real_result.g_table)

        borrowed = CalibrationResult(
            bp_table=str(link_bp),
            g_table=str(link_g),
            cal_date="2026-01-25",
            calibrator_name="unknown",
            source="borrowed",
        )
        with pytest.raises(CalibrationError, match="Borrowed cal tables"):
            _validate_strip_compatibility(borrowed, obs_dec_deg=16.1)

    def test_borrowed_missing_provenance_rejected(self, tmp_path):
        """Borrowed tables with no provenance sidecar are rejected."""
        bp = tmp_path / "2026-01-25T05:21:10_0~23.b"
        bp.mkdir()
        g = tmp_path / "2026-01-25T05:21:10_0~23.g"
        g.mkdir()
        link_bp = tmp_path / "2026-01-26T05:21:10_0~23.b"
        link_bp.symlink_to(bp)
        link_g = tmp_path / "2026-01-26T05:21:10_0~23.g"
        link_g.symlink_to(g)

        borrowed = CalibrationResult(
            bp_table=str(link_bp),
            g_table=str(link_g),
            cal_date="2026-01-25",
            calibrator_name="unknown",
            source="borrowed",
        )
        with pytest.raises(CalibrationError, match="no provenance sidecar"):
            _validate_strip_compatibility(borrowed, obs_dec_deg=16.1)

    def test_existing_missing_provenance_warned(self, tmp_path):
        """Same-date existing tables with no provenance get a warning, not rejection."""
        bp = tmp_path / "2026-01-25T05:21:10_0~23.b"
        bp.mkdir()
        g = tmp_path / "2026-01-25T05:21:10_0~23.g"
        g.mkdir()

        result = CalibrationResult(
            bp_table=str(bp),
            g_table=str(g),
            cal_date="2026-01-25",
            calibrator_name="unknown",
            source="existing",
        )
        # Should NOT raise, just warn
        validated = _validate_strip_compatibility(result, obs_dec_deg=16.1)
        assert validated.bp_table == str(bp)

    def test_no_obs_dec_skips_validation(self, tmp_path):
        """When obs_dec_deg is None, validation is skipped entirely."""
        result = CalibrationResult(
            bp_table="/fake/bp.b",
            g_table="/fake/bp.g",
            cal_date="2026-01-25",
            calibrator_name="unknown",
            source="borrowed",
        )
        # Should not raise even for borrowed with no provenance
        validated = _validate_strip_compatibility(result, obs_dec_deg=None)
        assert validated.bp_table == result.bp_table


# ── _find_nearest_real_tables ────────────────────────────────────────────


class TestFindNearestRealTables:
    def test_finds_nearby_date(self, tmp_path):
        """Should find tables from an adjacent date and create symlinks."""
        # Create real tables for 2026-01-25
        bp = tmp_path / "2026-01-25T22:26:05_0~23.b"
        bp.mkdir()
        g = tmp_path / "2026-01-25T22:26:05_0~23.g"
        g.mkdir()

        result = _find_nearest_real_tables("2026-01-26", str(tmp_path), max_borrow_days=5)
        assert result is not None
        assert result.source == "borrowed"
        assert result.cal_date == "2026-01-25"

        # Verify symlinks were created
        link_bp = tmp_path / "2026-01-26T22:26:05_0~23.b"
        assert link_bp.is_symlink()

    def test_returns_none_when_nothing_nearby(self, tmp_path):
        result = _find_nearest_real_tables("2026-01-25", str(tmp_path), max_borrow_days=5)
        assert result is None


# ── ensure_bandpass ──────────────────────────────────────────────────────


class TestEnsureBandpass:
    def test_returns_existing_tables(self, tmp_path):
        """If real tables exist, returns them immediately (no generation)."""
        bp = tmp_path / "2026-01-25T22:26:05_0~23.b"
        bp.mkdir()
        g = tmp_path / "2026-01-25T22:26:05_0~23.g"
        g.mkdir()

        result = ensure_bandpass(
            "2026-01-25", ms_dir=str(tmp_path),
            input_dir="/nonexistent", db_path="/nonexistent",
        )
        assert result.source == "existing"
        assert result.bp_table == str(bp)

    def test_force_skips_existing(self, tmp_path):
        """With force=True, should not return existing tables."""
        bp = tmp_path / "2026-01-25T22:26:05_0~23.b"
        bp.mkdir()
        g = tmp_path / "2026-01-25T22:26:05_0~23.g"
        g.mkdir()

        # Mock select_bandpass_calibrator to raise (no data for generation)
        with patch(
            "dsa110_continuum.calibration.ensure.select_bandpass_calibrator",
            side_effect=CalibrationError("no data"),
        ):
            # Should fall through to borrowing, then fail (no nearby dates)
            with pytest.raises(CalibrationError):
                ensure_bandpass(
                    "2026-01-26", ms_dir=str(tmp_path),
                    input_dir="/nonexistent", db_path="/nonexistent",
                    max_borrow_days=0, force=True,
                )

    def test_borrows_when_generation_fails(self, tmp_path):
        """If generation fails, should borrow from a nearby date."""
        # Real tables for 2026-01-25 with provenance
        bp = tmp_path / "2026-01-25T22:26:05_0~23.b"
        bp.mkdir()
        g = tmp_path / "2026-01-25T22:26:05_0~23.g"
        g.mkdir()
        prov = _build_provenance(
            selection_mode="dec_aware",
            obs_dec_deg_used=16.1,
            selection_dec_tolerance_deg=10.0,
            calibrator_name="3C138",
            calibrator_ra_deg=80.29,
            calibrator_dec_deg=16.64,
            calibrator_flux_jy=8.36,
            calibrator_dec_offset_deg=0.54,
            transit_time_iso="2026-01-25T05:21:10",
            source="generated",
            cal_date="2026-01-25",
            bp_table=str(bp),
            g_table=str(g),
        )
        write_provenance_sidecar(str(bp), prov)

        with patch(
            "dsa110_continuum.calibration.ensure.select_bandpass_calibrator",
            side_effect=CalibrationError("no data"),
        ):
            result = ensure_bandpass(
                "2026-01-26", ms_dir=str(tmp_path),
                input_dir="/nonexistent", db_path="/nonexistent",
                max_borrow_days=5, obs_dec_deg=16.1,
            )
            assert result.source == "borrowed"
            assert result.cal_date == "2026-01-25"

    def test_raises_when_all_fail(self, tmp_path):
        """If existing, generation, and borrowing all fail, raise."""
        with patch(
            "dsa110_continuum.calibration.ensure.select_bandpass_calibrator",
            side_effect=CalibrationError("no data"),
        ):
            with pytest.raises(CalibrationError, match="No calibration tables"):
                ensure_bandpass(
                    "2026-01-26", ms_dir=str(tmp_path),
                    input_dir="/nonexistent", db_path="/nonexistent",
                    max_borrow_days=0,
                )

    @patch("dsa110_continuum.calibration.ensure.generate_bandpass_tables")
    @patch("dsa110_continuum.calibration.ensure.select_bandpass_calibrator")
    def test_generates_when_no_existing(self, mock_select, mock_generate, tmp_path):
        """If no existing tables, should select calibrator and generate."""
        mock_select.return_value = (
            "3C147", 85.65, 49.85, Time("2026-01-25T05:42:36", scale="utc"),
        )
        mock_generate.return_value = CalibrationResult(
            bp_table=str(tmp_path / "2026-01-25T05:42:36_0~23.b"),
            g_table=str(tmp_path / "2026-01-25T05:42:36_0~23.g"),
            cal_date="2026-01-25",
            calibrator_name="3C147",
            source="generated",
            provenance={"selection_mode": "dec_aware", "calibrator_name": "3C147"},
        )

        result = ensure_bandpass(
            "2026-01-25", ms_dir=str(tmp_path),
            input_dir="/nonexistent", db_path="/nonexistent",
        )
        assert result.source == "generated"
        assert result.calibrator_name == "3C147"
        assert result.provenance["selection_mode"] == "dec_aware"
        mock_select.assert_called_once()
        mock_generate.assert_called_once()

    def test_borrowed_without_provenance_rejected(self, tmp_path):
        """Borrowing tables that have no provenance sidecar should fail."""
        # Real tables without provenance sidecar
        bp = tmp_path / "2026-01-25T22:26:05_0~23.b"
        bp.mkdir()
        g = tmp_path / "2026-01-25T22:26:05_0~23.g"
        g.mkdir()

        with patch(
            "dsa110_continuum.calibration.ensure.select_bandpass_calibrator",
            side_effect=CalibrationError("no data"),
        ):
            with pytest.raises(CalibrationError, match="no provenance sidecar"):
                ensure_bandpass(
                    "2026-01-26", ms_dir=str(tmp_path),
                    input_dir="/nonexistent", db_path="/nonexistent",
                    max_borrow_days=5, obs_dec_deg=16.1,
                )


# ── Validator uses strip Dec (obs_dec_deg_used), not calibrator Dec ──────


class TestValidatorUsesStripDec:
    """Confirm that validation keys off obs_dec_deg_used, NOT calibrator_dec_deg."""

    def test_calibrator_dec_far_but_strip_dec_compatible(self, tmp_path):
        """Calibrator Dec=50° would fail old logic, but obs_dec_deg_used=16.1 matches."""
        bp = tmp_path / "2026-01-25T05:21:10_0~23.b"
        bp.mkdir()
        g = tmp_path / "2026-01-25T05:21:10_0~23.g"
        g.mkdir()

        prov = _build_provenance(
            selection_mode="dec_aware",
            obs_dec_deg_used=16.1,
            selection_dec_tolerance_deg=10.0,
            calibrator_name="3C147",
            calibrator_ra_deg=85.65,
            calibrator_dec_deg=49.85,  # far from obs_dec 16.1 — would fail old logic
            calibrator_flux_jy=22.45,
            calibrator_dec_offset_deg=33.75,
            transit_time_iso="2026-01-25T05:42:36",
            source="generated",
            cal_date="2026-01-25",
            bp_table=str(bp),
            g_table=str(g),
        )
        write_provenance_sidecar(str(bp), prov)

        result = CalibrationResult(
            bp_table=str(bp), g_table=str(g),
            cal_date="2026-01-25", calibrator_name="3C147", source="existing",
        )
        # Current obs is at 16.1°, stored strip Dec is 16.1° → should PASS
        validated = _validate_strip_compatibility(result, obs_dec_deg=16.1)
        assert validated.bp_table == result.bp_table

    def test_calibrator_dec_close_but_strip_dec_incompatible(self, tmp_path):
        """Calibrator Dec=16.6° close to obs_dec, but strip Dec=50° → reject."""
        bp = tmp_path / "2026-01-25T05:21:10_0~23.b"
        bp.mkdir()
        g = tmp_path / "2026-01-25T05:21:10_0~23.g"
        g.mkdir()

        prov = _build_provenance(
            selection_mode="dec_aware",
            obs_dec_deg_used=50.0,  # table was generated for strip Dec 50°
            selection_dec_tolerance_deg=10.0,
            calibrator_name="3C138",
            calibrator_ra_deg=80.29,
            calibrator_dec_deg=16.64,  # calibrator Dec close to current obs
            calibrator_flux_jy=8.36,
            calibrator_dec_offset_deg=33.36,
            transit_time_iso="2026-01-25T05:21:10",
            source="generated",
            cal_date="2026-01-25",
            bp_table=str(bp),
            g_table=str(g),
        )
        write_provenance_sidecar(str(bp), prov)

        result = CalibrationResult(
            bp_table=str(bp), g_table=str(g),
            cal_date="2026-01-25", calibrator_name="3C138", source="existing",
        )
        # Current obs is at 16.1°, but stored strip Dec is 50.0° → should REJECT
        with pytest.raises(CalibrationError, match="strip Dec 50.0"):
            _validate_strip_compatibility(result, obs_dec_deg=16.1)


# ── Borrowed tables with sidecar but missing obs_dec_deg_used ────────────


class TestBorrowedPartialProvenance:
    def test_borrowed_sidecar_missing_obs_dec_used_rejected(self, tmp_path):
        """Borrowed table with sidecar that lacks obs_dec_deg_used → reject."""
        import json

        real_bp = tmp_path / "2026-01-25T05:21:10_0~23.b"
        real_bp.mkdir()
        real_g = tmp_path / "2026-01-25T05:21:10_0~23.g"
        real_g.mkdir()

        # Write a sidecar WITHOUT obs_dec_deg_used (simulating old/partial provenance)
        partial_prov = {
            "selection_mode": "brightest",
            "calibrator_name": "3C147",
            "calibrator_dec_deg": 49.85,
            "source": "generated",
        }
        sidecar = str(real_bp) + ".cal_provenance.json"
        with open(sidecar, "w") as f:
            json.dump(partial_prov, f)

        link_bp = tmp_path / "2026-03-16T05:21:10_0~23.b"
        link_bp.symlink_to(real_bp)
        link_g = tmp_path / "2026-03-16T05:21:10_0~23.g"
        link_g.symlink_to(real_g)

        borrowed = CalibrationResult(
            bp_table=str(link_bp), g_table=str(link_g),
            cal_date="2026-01-25", calibrator_name="unknown", source="borrowed",
        )
        with pytest.raises(CalibrationError, match="missing obs_dec_deg_used"):
            _validate_strip_compatibility(borrowed, obs_dec_deg=16.1)

    def test_existing_sidecar_missing_obs_dec_used_warned(self, tmp_path):
        """Existing table with sidecar that lacks obs_dec_deg_used → warn, allow."""
        import json

        bp = tmp_path / "2026-01-25T05:21:10_0~23.b"
        bp.mkdir()
        g = tmp_path / "2026-01-25T05:21:10_0~23.g"
        g.mkdir()

        partial_prov = {
            "selection_mode": "brightest",
            "calibrator_name": "3C147",
            "calibrator_dec_deg": 49.85,
            "source": "generated",
        }
        sidecar = str(bp) + ".cal_provenance.json"
        with open(sidecar, "w") as f:
            json.dump(partial_prov, f)

        result = CalibrationResult(
            bp_table=str(bp), g_table=str(g),
            cal_date="2026-01-25", calibrator_name="unknown", source="existing",
        )
        # Should NOT raise — backward-compatible warn+allow
        validated = _validate_strip_compatibility(result, obs_dec_deg=16.1)
        assert validated.bp_table == str(bp)


# ── validate_table_strip_compatibility (public, for fallback path) ───────


class TestValidateTableStripCompatibility:
    def test_rejects_incompatible_symlinked_tables(self, tmp_path):
        """Symlinked (borrowed) tables at wrong strip Dec → CalibrationError."""
        real_bp = tmp_path / "2026-01-25T05:21:10_0~23.b"
        real_bp.mkdir()

        prov = _build_provenance(
            selection_mode="dec_aware",
            obs_dec_deg_used=50.0,
            selection_dec_tolerance_deg=10.0,
            calibrator_name="3C147",
            calibrator_ra_deg=85.65,
            calibrator_dec_deg=49.85,
            calibrator_flux_jy=22.45,
            calibrator_dec_offset_deg=0.85,
            transit_time_iso="2026-01-25T05:42:36",
            source="generated",
            cal_date="2026-01-25",
            bp_table=str(real_bp),
            g_table=str(real_bp).replace(".b", ".g"),
        )
        write_provenance_sidecar(str(real_bp), prov)

        link_bp = tmp_path / "2026-03-16T05:21:10_0~23.b"
        link_bp.symlink_to(real_bp)

        with pytest.raises(CalibrationError, match="strip Dec 50.0"):
            validate_table_strip_compatibility(str(link_bp), obs_dec_deg=16.1)

    def test_accepts_compatible_real_tables(self, tmp_path):
        """Real (existing) tables at correct strip Dec → no error."""
        bp = tmp_path / "2026-01-25T05:21:10_0~23.b"
        bp.mkdir()

        prov = _build_provenance(
            selection_mode="dec_aware",
            obs_dec_deg_used=16.1,
            selection_dec_tolerance_deg=10.0,
            calibrator_name="3C138",
            calibrator_ra_deg=80.29,
            calibrator_dec_deg=16.64,
            calibrator_flux_jy=8.36,
            calibrator_dec_offset_deg=0.54,
            transit_time_iso="2026-01-25T05:21:10",
            source="generated",
            cal_date="2026-01-25",
            bp_table=str(bp),
            g_table=str(bp).replace(".b", ".g"),
        )
        write_provenance_sidecar(str(bp), prov)

        # Should not raise
        validate_table_strip_compatibility(str(bp), obs_dec_deg=16.1)

    def test_skips_when_no_obs_dec(self, tmp_path):
        """When obs_dec_deg is None, validation is skipped."""
        bp = tmp_path / "2026-01-25T05:21:10_0~23.b"
        bp.mkdir()
        # No sidecar — would fail for borrowed, but obs_dec=None skips
        validate_table_strip_compatibility(str(bp), obs_dec_deg=None)


# ── Runtime source fidelity in provenance ────────────────────────────────


class TestRuntimeSourceFidelity:
    def _setup_real_tables_with_provenance(self, tmp_path, date="2026-01-25"):
        """Create real tables with 'generated' sidecar provenance."""
        bp = tmp_path / f"{date}T05:21:10_0~23.b"
        bp.mkdir()
        g = tmp_path / f"{date}T05:21:10_0~23.g"
        g.mkdir()
        prov = _build_provenance(
            selection_mode="dec_aware",
            obs_dec_deg_used=16.1,
            selection_dec_tolerance_deg=10.0,
            calibrator_name="3C138",
            calibrator_ra_deg=80.29,
            calibrator_dec_deg=16.64,
            calibrator_flux_jy=8.36,
            calibrator_dec_offset_deg=0.54,
            transit_time_iso=f"{date}T05:21:10",
            source="generated",
            cal_date=date,
            bp_table=str(bp),
            g_table=str(g),
        )
        write_provenance_sidecar(str(bp), prov)
        return str(bp), str(g)

    def test_existing_tables_have_source_existing(self, tmp_path):
        """ensure_bandpass on same-date existing tables → source='existing' in provenance."""
        self._setup_real_tables_with_provenance(tmp_path)

        result = ensure_bandpass(
            "2026-01-25", ms_dir=str(tmp_path),
            input_dir="/nonexistent", db_path="/nonexistent",
            obs_dec_deg=16.1,
        )
        assert result.source == "existing"
        assert result.provenance["source"] == "existing"  # NOT "generated"
        assert result.provenance["calibrator_name"] == "3C138"

    def test_borrowed_tables_have_source_borrowed(self, tmp_path):
        """ensure_bandpass borrowing from nearby date → source='borrowed' in provenance."""
        self._setup_real_tables_with_provenance(tmp_path)

        with patch(
            "dsa110_continuum.calibration.ensure.select_bandpass_calibrator",
            side_effect=CalibrationError("no data"),
        ):
            result = ensure_bandpass(
                "2026-01-26", ms_dir=str(tmp_path),
                input_dir="/nonexistent", db_path="/nonexistent",
                max_borrow_days=5, obs_dec_deg=16.1,
            )
        assert result.source == "borrowed"
        assert result.provenance["source"] == "borrowed"  # NOT "generated"
        assert result.provenance["calibrator_name"] == "3C138"

    def test_enrich_overlays_runtime_fields(self, tmp_path):
        """_enrich_result_provenance overlays source, bp_table, g_table, cal_date."""
        bp, g = self._setup_real_tables_with_provenance(tmp_path)

        raw = CalibrationResult(
            bp_table=bp, g_table=g,
            cal_date="2026-01-25", calibrator_name="unknown", source="existing",
        )
        enriched = _enrich_result_provenance(raw)
        assert enriched.provenance["source"] == "existing"
        assert enriched.provenance["bp_table"] == bp
        assert enriched.provenance["calibrator_name"] == "3C138"  # from sidecar

    def test_enrich_noop_for_generated(self, tmp_path):
        """_enrich_result_provenance is a no-op when provenance is already populated."""
        existing_prov = {"source": "generated", "calibrator_name": "3C147"}
        result = CalibrationResult(
            bp_table="/fake/bp.b", g_table="/fake/bp.g",
            cal_date="2026-01-25", calibrator_name="3C147",
            source="generated", provenance=existing_prov,
        )
        enriched = _enrich_result_provenance(result)
        assert enriched is result  # Same object, not modified
