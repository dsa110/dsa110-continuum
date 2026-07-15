"""Regression tests for transit-time-scoped group selection (issue #72).

In drift scan the same RA transits every sidereal day, so purely positional
group selection can pick a group from a different date than the requested
transit. These tests pin that ``generate_from_transit`` honors the requested
``transit_time`` when the index contains same-position candidates on
multiple dates.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from astropy.time import Time
from dsa110_continuum.conversion.calibrator_ms_generator import (
    CalibratorInfo,
    CalibratorMSGenerator,
)

D1_GROUP = "2026-01-25T22:26:05"
D2_GROUP = "2026-01-26T22:22:09"  # same RA strip, next sidereal day
TRANSIT_D1 = Time("2026-01-25T22:30:00")

CAL = CalibratorInfo(name="3C454.3", ra_deg=343.49, dec_deg=16.15, flux_jy=12.5)

LEGACY_SELECTOR = "dsa110_continuum.database.hdf5_index.select_hdf5_groups_by_position"


@pytest.fixture
def generator(tmp_path):
    catalog = tmp_path / "vla_calibrators.sqlite3"
    catalog.touch()
    return CalibratorMSGenerator(
        input_dir=tmp_path,
        output_dir=tmp_path / "ms",
        db_path=tmp_path / "pipeline.sqlite3",
        vla_catalog_path=catalog,
    )


class TestSelectGroupsByPosition:
    def test_transit_time_filters_other_dates(self, generator):
        with patch(LEGACY_SELECTOR, return_value=[D2_GROUP, D1_GROUP]):
            groups = generator.select_groups_by_position(
                source_ra_deg=CAL.ra_deg,
                source_dec_deg=CAL.dec_deg,
                transit_time=TRANSIT_D1,
            )
        assert groups == [D1_GROUP]

    def test_no_transit_time_keeps_positional_order(self, generator):
        with patch(LEGACY_SELECTOR, return_value=[D2_GROUP, D1_GROUP]):
            groups = generator.select_groups_by_position(
                source_ra_deg=CAL.ra_deg,
                source_dec_deg=CAL.dec_deg,
            )
        assert groups == [D2_GROUP, D1_GROUP]

    def test_n_groups_cap_applied_after_time_scoping(self, generator):
        """A wrong-date closest candidate must not crowd out the requested date.

        The legacy selector caps to n_groups before this wrapper can
        time-scope, so with n_groups=1 a positionally closer D2 group would
        be the only candidate returned. The wrapper must fetch uncapped,
        filter by transit_time, then re-apply the caller's n_groups.
        """
        with patch(LEGACY_SELECTOR, return_value=[D2_GROUP, D1_GROUP]) as legacy:
            groups = generator.select_groups_by_position(
                source_ra_deg=CAL.ra_deg,
                source_dec_deg=CAL.dec_deg,
                n_groups=1,
                transit_time=TRANSIT_D1,
            )
        assert legacy.call_args.kwargs["n_groups"] > 1000
        assert groups == [D1_GROUP]

    def test_no_transit_time_passes_caller_n_groups(self, generator):
        with patch(LEGACY_SELECTOR, return_value=[D2_GROUP]) as legacy:
            generator.select_groups_by_position(
                source_ra_deg=CAL.ra_deg,
                source_dec_deg=CAL.dec_deg,
                n_groups=1,
            )
        assert legacy.call_args.kwargs["n_groups"] == 1

    def test_raises_when_no_group_near_transit(self, generator):
        with patch(LEGACY_SELECTOR, return_value=[D2_GROUP]):
            with pytest.raises(ValueError, match="transit"):
                generator.select_groups_by_position(
                    source_ra_deg=CAL.ra_deg,
                    source_dec_deg=CAL.dec_deg,
                    transit_time=TRANSIT_D1,
                )


class TestGenerateFromTransit:
    def test_selects_group_from_requested_date(self, generator):
        """D2 is positionally closer (listed first) but D1 holds the transit."""
        with (
            patch.object(generator, "get_calibrator", return_value=CAL),
            patch(LEGACY_SELECTOR, return_value=[D2_GROUP, D1_GROUP]),
            patch.object(generator, "convert_groups", return_value=[Path("d1.ms")]) as convert,
        ):
            result = generator.generate_from_transit(
                calibrator_name=CAL.name,
                transit_time=TRANSIT_D1,
                verify=False,
            )
        assert result.success
        assert convert.call_args.args[0] == [D1_GROUP]

    def test_fails_loudly_when_requested_date_missing(self, generator):
        with (
            patch.object(generator, "get_calibrator", return_value=CAL),
            patch(LEGACY_SELECTOR, return_value=[D2_GROUP]),
            patch.object(generator, "convert_groups") as convert,
        ):
            result = generator.generate_from_transit(
                calibrator_name=CAL.name,
                transit_time=TRANSIT_D1,
                verify=False,
            )
        assert not result.success
        assert convert.call_count == 0
        assert "transit" in (result.error_message or "").lower()


class TestGenerateMultiple:
    def test_passes_transit_time_to_selection(self, generator):
        with (
            patch.object(generator, "get_calibrator", return_value=CAL),
            patch(LEGACY_SELECTOR, return_value=[D2_GROUP, D1_GROUP]),
            patch.object(generator, "convert_groups", return_value=[Path("d1.ms")]) as convert,
        ):
            result = generator.generate_multiple(
                calibrator_name=CAL.name,
                transit_time=TRANSIT_D1,
                verify=False,
            )
        assert result.success
        assert convert.call_args.args[0] == [D1_GROUP]


class TestConvertGroupsAcceptsTimestampStrings:
    """select_groups_by_position returns timestamp strings; convert_groups must
    treat a string group as the group timestamp, not a file list (regression:
    it indexed the string, so Time("2", format="isot") raised ValueError and
    auto-cal table generation always failed)."""

    def test_string_group_uses_full_timestamp_window(self, generator):
        import astropy.units as u

        def fake_convert(**kwargs):
            Path(kwargs["output_dir"], f"{kwargs['start_time']}.ms").mkdir(parents=True)

        with patch(
            "dsa110_continuum.conversion.convert_subband_groups_to_ms",
            side_effect=fake_convert,
        ) as conv:
            ms_paths = generator.convert_groups([D1_GROUP])

        kwargs = conv.call_args.kwargs
        assert kwargs["start_time"] == D1_GROUP
        assert kwargs["end_time"] == (Time(D1_GROUP, format="isot") + 2 * u.minute).isot
        assert ms_paths == [generator.output_dir / f"{D1_GROUP}.ms"]


class TestSiderealDayScoping:
    """Correctness criterion (analytic): in drift scan the same RA transits
    once per mean sidereal day (86164.0905 s = 1436.07 min). A positionally
    matched group exactly one sidereal day from the requested transit belongs
    to a different date and MUST be rejected; a group within the beam-crossing
    time (~15 min at Dec 16° for the 3.5° beam) MUST be kept. This requires
    beam-crossing < tolerance < sidereal_day / 2 (= 718.03 min)."""

    def _group_at(self, offset_minutes):
        import astropy.units as u

        return (TRANSIT_D1 + offset_minutes * u.minute).isot

    def test_default_tolerance_in_analytic_safe_range(self):
        import inspect

        from dsa110_continuum.conversion.calibrator_ms_generator import SIDEREAL_DAY_MINUTES

        sig = inspect.signature(CalibratorMSGenerator.select_groups_by_position)
        default_tol = sig.parameters["max_transit_offset_minutes"].default
        beam_crossing_min = 3.5 / 15.0 * 60.0  # FWHM / drift rate at Dec 0
        assert beam_crossing_min < default_tol < SIDEREAL_DAY_MINUTES / 2

    def test_rejects_group_exactly_one_sidereal_day_away(self, generator):
        adjacent_transit = self._group_at(86164.0905 / 60.0)
        with patch(LEGACY_SELECTOR, return_value=[adjacent_transit]):
            with pytest.raises(ValueError, match="different transit date"):
                generator.select_groups_by_position(
                    source_ra_deg=CAL.ra_deg,
                    source_dec_deg=CAL.dec_deg,
                    transit_time=TRANSIT_D1,
                )

    def test_keeps_in_beam_group_drops_adjacent_day(self, generator):
        in_beam = self._group_at(10.0)
        adjacent = self._group_at(86164.0905 / 60.0)
        with patch(LEGACY_SELECTOR, return_value=[adjacent, in_beam]):
            groups = generator.select_groups_by_position(
                source_ra_deg=CAL.ra_deg,
                source_dec_deg=CAL.dec_deg,
                transit_time=TRANSIT_D1,
            )
        assert groups == [in_beam]

    def test_tolerance_boundary_semantics(self, generator):
        """Pins the default 360 min tolerance to ±1 min without asserting
        exact float equality at the boundary."""
        just_inside = self._group_at(359.0)
        just_outside = self._group_at(361.0)
        with patch(LEGACY_SELECTOR, return_value=[just_outside, just_inside]):
            groups = generator.select_groups_by_position(
                source_ra_deg=CAL.ra_deg,
                source_dec_deg=CAL.dec_deg,
                transit_time=TRANSIT_D1,
            )
        assert groups == [just_inside]

    def test_unsafe_tolerance_raises(self, generator):
        """Fail-loud guard: a tolerance ≥ half a sidereal day would silently
        re-admit wrong-date groups (the issue #72 failure mode)."""
        with patch(LEGACY_SELECTOR, return_value=[D1_GROUP]):
            with pytest.raises(ValueError, match="sidereal"):
                generator.select_groups_by_position(
                    source_ra_deg=CAL.ra_deg,
                    source_dec_deg=CAL.dec_deg,
                    transit_time=TRANSIT_D1,
                    max_transit_offset_minutes=800.0,
                )

    def test_nonpositive_tolerance_raises(self, generator):
        with patch(LEGACY_SELECTOR, return_value=[D1_GROUP]):
            with pytest.raises(ValueError, match="sidereal"):
                generator.select_groups_by_position(
                    source_ra_deg=CAL.ra_deg,
                    source_dec_deg=CAL.dec_deg,
                    transit_time=TRANSIT_D1,
                    max_transit_offset_minutes=0.0,
                )

    def test_tolerance_unchecked_without_transit_time(self, generator):
        """Without a transit_time there is no scoping, so no guard applies."""
        with patch(LEGACY_SELECTOR, return_value=[D1_GROUP]):
            groups = generator.select_groups_by_position(
                source_ra_deg=CAL.ra_deg,
                source_dec_deg=CAL.dec_deg,
                max_transit_offset_minutes=800.0,
            )
        assert groups == [D1_GROUP]


class TestConvertGroupsBehavior:
    """Behavioral invariants of convert_groups: group-type dispatch,
    skip_existing short-circuit, and per-group failure isolation (one bad
    group must not abort the remaining conversions)."""

    CONVERTER = "dsa110_continuum.conversion.convert_subband_groups_to_ms"

    def test_subband_group_object_timestamp_from_sb_stem(self, generator):
        class FakeSubbandGroup:
            files = [f"/data/incoming/{D1_GROUP}_sb00.hdf5"]

        with patch(self.CONVERTER) as conv:
            generator.convert_groups([FakeSubbandGroup()])
        assert conv.call_args.kwargs["start_time"] == D1_GROUP

    def test_empty_file_list_skipped(self, generator):
        with patch(self.CONVERTER) as conv:
            ms_paths = generator.convert_groups([[]])
        assert conv.call_count == 0
        assert ms_paths == []

    def test_skip_existing_returns_without_converting(self, generator):
        existing = generator.output_dir / f"{D1_GROUP}.ms"
        existing.mkdir(parents=True)
        with patch(self.CONVERTER) as conv:
            ms_paths = generator.convert_groups([D1_GROUP])
        assert conv.call_count == 0
        assert ms_paths == [existing]

    def test_failed_group_does_not_abort_remaining(self, generator):
        calls = []

        def fake_convert(**kwargs):
            calls.append(kwargs["start_time"])
            if kwargs["start_time"] == D1_GROUP:
                raise RuntimeError("conversion blew up")
            Path(kwargs["output_dir"], f"{kwargs['start_time']}.ms").mkdir(parents=True)

        with patch(self.CONVERTER, side_effect=fake_convert):
            ms_paths = generator.convert_groups([D1_GROUP, D2_GROUP])

        assert calls == [D1_GROUP, D2_GROUP]
        assert ms_paths == [generator.output_dir / f"{D2_GROUP}.ms"]


class TestFailLoudPaths:
    def test_find_last_transit_raises_when_no_transits(self, generator):
        with patch.object(generator, "find_transits", return_value=[]):
            with pytest.raises(RuntimeError, match="No transits"):
                generator.find_last_transit(CAL)

    def test_verify_calibrator_absent_returns_false(self, generator):
        with patch(
            "dsa110_continuum.calibration.selection.select_bandpass_from_catalog",
            side_effect=RuntimeError("no calibrator in MS"),
        ):
            present, peak = generator.verify_calibrator_in_ms(Path("x.ms"), CAL)
        assert present is False
        assert peak is None
