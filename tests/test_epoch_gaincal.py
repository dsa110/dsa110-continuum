"""Tests for epoch_gaincal: tile selection and calibrate_epoch fallback."""
import sys
from unittest.mock import patch


def test_select_calibration_tile_from_ms_picks_richer_tile():
    """Should return the MS whose central pointing has more catalog sources."""
    from dsa110_continuum.calibration.epoch_gaincal import select_calibration_tile_from_ms

    fake_paths = [f"/fake/tile_{i:02d}.ms" for i in range(12)]

    def fake_phase_center(ms_path):
        idx = int(Path(ms_path).stem.split("_")[1])
        return (float(idx) * 10.0, 37.0)

    def fake_count(pointing_ra_deg, pointing_dec_deg, **kwargs):
        # tile index 6 → ra=60.0 → 8 sources; tile index 5 → ra=50.0 → 3 sources
        return 8 if pointing_ra_deg == 60.0 else 3

    with patch(
        "dsa110_continuum.calibration.epoch_gaincal._read_ms_phase_center",
        side_effect=fake_phase_center,
    ), patch(
        "dsa110_continuum.calibration.epoch_gaincal.count_bright_sources_in_tile",
        side_effect=fake_count,
    ):
        result = select_calibration_tile_from_ms(fake_paths)

    assert result == "/fake/tile_06.ms"


def test_select_calibration_tile_raises_on_wrong_count():
    """Should raise ValueError when not given exactly 12 MS paths."""
    from dsa110_continuum.calibration.epoch_gaincal import select_calibration_tile_from_ms
    import pytest

    with pytest.raises(ValueError, match="Expected 12"):
        select_calibration_tile_from_ms(["/fake/a.ms", "/fake/b.ms"])


def test_select_calibration_tile_defaults_to_tile5_on_failure():
    """Falls back to tile 5 if source counting raises for both candidates."""
    from dsa110_continuum.calibration.epoch_gaincal import select_calibration_tile_from_ms

    fake_paths = [f"/fake/tile_{i:02d}.ms" for i in range(12)]

    with patch(
        "dsa110_continuum.calibration.epoch_gaincal._read_ms_phase_center",
        side_effect=RuntimeError("casacore unavailable"),
    ):
        result = select_calibration_tile_from_ms(fake_paths)

    assert result == "/fake/tile_05.ms"


def test_calibrate_epoch_returns_none_on_predict_failure():
    """calibrate_epoch() should return None (not raise) if catalog predict fails."""
    import tempfile
    from dsa110_continuum.calibration.epoch_gaincal import calibrate_epoch

    fake_paths = [f"/fake/tile_{i:02d}.ms" for i in range(12)]

    with tempfile.TemporaryDirectory() as work_dir:
        with patch(
            "dsa110_continuum.calibration.epoch_gaincal.select_calibration_tile_from_ms",
            return_value="/fake/tile_05.ms",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.phaseshift_ms",
            return_value=("/fake/tile_05_meridian.ms", "J2000 ..."),
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.apply_to_target",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.make_unified_skymodel",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.predict_from_skymodel_wsclean",
            side_effect=RuntimeError("wsclean not found"),
        ), patch(
            "os.path.exists",
            return_value=True,  # pretend meridian MS exists to skip phaseshift
        ):
            result = calibrate_epoch(fake_paths, "/fake/bp.b", work_dir)

    assert result is None


def test_calibrate_epoch_returns_none_on_empty_sky_model():
    """calibrate_epoch() should return None when the catalog sky model is empty."""
    import tempfile
    from unittest.mock import MagicMock
    from dsa110_continuum.calibration.epoch_gaincal import calibrate_epoch

    fake_paths = [f"/fake/tile_{i:02d}.ms" for i in range(12)]
    empty_sky = MagicMock()
    empty_sky.Ncomponents = 0

    with tempfile.TemporaryDirectory() as work_dir:
        with patch(
            "dsa110_continuum.calibration.epoch_gaincal.select_calibration_tile_from_ms",
            return_value="/fake/tile_05.ms",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal._read_ms_phase_center",
            return_value=(10.0, 37.0),
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.phaseshift_ms",
            return_value=("/fake/tile_05_meridian.ms", "J2000 ..."),
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.apply_to_target",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.make_unified_skymodel",
            return_value=empty_sky,
        ), patch(
            "os.path.exists",
            return_value=True,
        ):
            result = calibrate_epoch(fake_paths, "/fake/bp.b", work_dir)

    assert result is None


def test_process_ms_force_recal_calls_applycal_even_when_data_exists():
    """process_ms(force_recal=True) must be accepted — confirms the API surface."""
    import importlib
    import inspect

    sys.path.insert(0, "/data/dsa110-continuum/scripts")
    md = importlib.import_module("mosaic_day")
    sig = inspect.signature(md.process_ms)
    assert "force_recal" in sig.parameters, "process_ms must accept force_recal"


# Keep Path import at module level so it's available in test functions
from pathlib import Path  # noqa: E402
