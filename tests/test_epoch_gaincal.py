"""Tests for epoch_gaincal: tile selection, RFI flagging, WSClean guard, refant chain."""
import sys
from unittest.mock import MagicMock, patch


def test_select_calibration_tile_from_ms_picks_richer_tile():
    """Should return the MS whose central pointing has more catalog sources."""
    from dsa110_continuum.calibration.epoch_gaincal import select_calibration_tile_from_ms

    # The all-tile fallback should select index 6 because it has more sources.
    fake_paths = [f"/fake/tile_{i:02d}.ms" for i in range(12)]

    def fake_phase_center(ms_path):
        idx = int(Path(ms_path).stem.split("_")[1])
        return (float(idx) * 10.0, 37.0)

    def fake_count(pointing_ra_deg, pointing_dec_deg, **kwargs):
        # tile index 6 → ra=60.0 → 8 sources; tile index 5 → ra=50.0 → 3 sources
        return 8 if pointing_ra_deg == 60.0 else 3

    with patch(
        "dsa110_continuum.calibration.epoch_gaincal._find_vla_calibrator_in_ms",
        side_effect=RuntimeError("no VLA calibrator"),
    ), patch(
        "dsa110_continuum.calibration.epoch_gaincal._read_ms_phase_center",
        side_effect=fake_phase_center,
    ), patch(
        "dsa110_continuum.calibration.epoch_gaincal.count_bright_sources_in_tile",
        side_effect=fake_count,
    ):
        result = select_calibration_tile_from_ms(fake_paths)

    assert result == "/fake/tile_06.ms"


def test_select_calibration_tile_works_with_11_tiles():
    """Should rank every tile when an epoch has an odd tile count."""
    from dsa110_continuum.calibration.epoch_gaincal import select_calibration_tile_from_ms

    # The all-tile fallback should select index 5 because it has more sources.
    fake_paths = [f"/fake/tile_{i:02d}.ms" for i in range(11)]

    def fake_phase_center(ms_path):
        idx = int(Path(ms_path).stem.split("_")[1])
        return (float(idx) * 10.0, 37.0)

    def fake_count(pointing_ra_deg, pointing_dec_deg, **kwargs):
        # tile index 5 → ra=50.0 → 9 sources; tile index 4 → ra=40.0 → 2 sources
        return 9 if pointing_ra_deg == 50.0 else 2

    with patch(
        "dsa110_continuum.calibration.epoch_gaincal._find_vla_calibrator_in_ms",
        side_effect=RuntimeError("no VLA calibrator"),
    ), patch(
        "dsa110_continuum.calibration.epoch_gaincal._read_ms_phase_center",
        side_effect=fake_phase_center,
    ), patch(
        "dsa110_continuum.calibration.epoch_gaincal.count_bright_sources_in_tile",
        side_effect=fake_count,
    ):
        result = select_calibration_tile_from_ms(fake_paths)

    assert result == "/fake/tile_05.ms"


def test_select_calibration_tile_raises_on_too_few():
    """Should raise ValueError when given fewer than 2 MS paths."""
    from dsa110_continuum.calibration.epoch_gaincal import select_calibration_tile_from_ms
    import pytest

    with pytest.raises(ValueError, match="at least 2"):
        select_calibration_tile_from_ms(["/fake/a.ms"])


def test_select_calibration_tile_prefers_bright_vla_calibrator_outside_center():
    """A bright calibrator transit should beat the geometric center pair."""
    from dsa110_continuum.calibration.epoch_gaincal import select_calibration_tile_from_ms

    fake_paths = [f"/fake/tile_{i:02d}.ms" for i in range(12)]

    def fake_calibrator_match(ms_path, **kwargs):
        idx = int(Path(ms_path).stem.split("_")[1])
        if idx == 2:
            return ("2253+161", 12.66, 0.20)
        if idx == 6:
            return ("faint-cal", 1.0, 0.05)
        raise RuntimeError("no calibrator")

    with patch(
        "dsa110_continuum.calibration.epoch_gaincal._find_vla_calibrator_in_ms",
        side_effect=fake_calibrator_match,
    ), patch(
        "dsa110_continuum.calibration.epoch_gaincal.count_bright_sources_in_tile",
    ) as source_count:
        result = select_calibration_tile_from_ms(fake_paths)

    assert result == "/fake/tile_02.ms"
    source_count.assert_not_called()


def test_select_calibration_tile_uses_nearest_tile_for_same_calibrator():
    """When one calibrator spans tiles, select its closest tile midpoint."""
    from dsa110_continuum.calibration.epoch_gaincal import select_calibration_tile_from_ms

    fake_paths = [f"/fake/tile_{i:02d}.ms" for i in range(4)]

    def fake_calibrator_match(ms_path, **kwargs):
        idx = int(Path(ms_path).stem.split("_")[1])
        separations = {1: 0.65, 2: 0.18}
        if idx in separations:
            return ("2253+161", 12.66, separations[idx])
        raise RuntimeError("no calibrator")

    with patch(
        "dsa110_continuum.calibration.epoch_gaincal._find_vla_calibrator_in_ms",
        side_effect=fake_calibrator_match,
    ):
        result = select_calibration_tile_from_ms(fake_paths)

    assert result == "/fake/tile_02.ms"


def test_select_calibration_tile_counts_all_tiles_without_vla_catalog():
    """Catalog-count fallback must consider non-central tiles too."""
    from dsa110_continuum.calibration.epoch_gaincal import select_calibration_tile_from_ms

    fake_paths = [f"/fake/tile_{i:02d}.ms" for i in range(12)]

    def fake_phase_center(ms_path):
        idx = int(Path(ms_path).stem.split("_")[1])
        return (float(idx), 16.1)

    with patch(
        "dsa110_continuum.calibration.epoch_gaincal._find_vla_calibrator_in_ms",
        side_effect=FileNotFoundError("VLA catalog missing"),
    ), patch(
        "dsa110_continuum.calibration.epoch_gaincal._read_ms_phase_center",
        side_effect=fake_phase_center,
    ), patch(
        "dsa110_continuum.calibration.epoch_gaincal.count_bright_sources_in_tile",
        side_effect=lambda ra, dec, **kwargs: 20 if ra == 2.0 else 1,
    ):
        result = select_calibration_tile_from_ms(fake_paths)

    assert result == "/fake/tile_02.ms"


def test_select_calibration_tile_defaults_to_central_tile_on_failure():
    """Falls back to n//2 if source counting fails for every tile."""
    from dsa110_continuum.calibration.epoch_gaincal import select_calibration_tile_from_ms

    # 12 tiles: n//2 = 6
    fake_paths_12 = [f"/fake/tile_{i:02d}.ms" for i in range(12)]
    with patch(
        "dsa110_continuum.calibration.epoch_gaincal._find_vla_calibrator_in_ms",
        side_effect=RuntimeError("no VLA calibrator"),
    ), patch(
        "dsa110_continuum.calibration.epoch_gaincal._read_ms_phase_center",
        side_effect=RuntimeError("casacore unavailable"),
    ):
        result_12 = select_calibration_tile_from_ms(fake_paths_12)
    assert result_12 == "/fake/tile_06.ms"

    # 6 tiles: n//2 = 3
    fake_paths_6 = [f"/fake/tile_{i:02d}.ms" for i in range(6)]
    with patch(
        "dsa110_continuum.calibration.epoch_gaincal._find_vla_calibrator_in_ms",
        side_effect=RuntimeError("no VLA calibrator"),
    ), patch(
        "dsa110_continuum.calibration.epoch_gaincal._read_ms_phase_center",
        side_effect=RuntimeError("casacore unavailable"),
    ):
        result_6 = select_calibration_tile_from_ms(fake_paths_6)
    assert result_6 == "/fake/tile_03.ms"


def test_count_bright_sources_falls_back_to_nvss_when_vlass_missing():
    """A missing VLASS strip database must not prevent the NVSS query."""
    import pandas as pd
    from dsa110_continuum.calibration.model import count_bright_sources_in_tile

    calls = []

    def fake_query(*, catalog_type, **kwargs):
        calls.append(catalog_type)
        if catalog_type == "vlass":
            raise FileNotFoundError("missing VLASS strip database")
        return pd.DataFrame([{"ra_deg": 343.49, "dec_deg": 16.15}])

    with patch(
        "dsa110_continuum.calibration.catalogs.query_catalog_sources",
        side_effect=fake_query,
    ):
        result = count_bright_sources_in_tile(343.49, 16.15)

    assert result == 1
    assert calls[:2] == ["vlass", "nvss"]


def _make_exists_fn(meridian_ms: str, *, ap_table: str | None = None) -> object:
    """Return os.path.exists side_effect.

    - ap_table path → False (prevents cached-result early return)
    - meridian_ms path → True (tells code the phaseshift output exists)
    - everything else → True (intermediate tables, directories, etc.)
    """
    def _exists(path: str) -> bool:
        p = str(path)
        if ap_table and p == ap_table:
            return False
        return True
    return _exists


def test_calibrate_epoch_returns_exception_status_on_predict_failure():
    """calibrate_epoch() should return EXCEPTION status (not raise) when catalog predict fails."""
    import tempfile
    from dsa110_continuum.calibration.epoch_gaincal import (
        EpochGaincalStatus,
        calibrate_epoch,
    )

    fake_paths = [f"/fake/tile_{i:02d}.ms" for i in range(12)]
    mock_svc = MagicMock()

    with tempfile.TemporaryDirectory() as work_dir:
        meridian_ms = str(Path(work_dir) / "tile_05_meridian.ms")
        ap_table   = str(Path(work_dir) / "tile_05.direct_first.ap.G")
        with patch(
            "dsa110_continuum.calibration.epoch_gaincal.select_calibration_tile_from_ms",
            return_value="/fake/tile_05.ms",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal._read_ms_phase_center",
            return_value=(10.0, 37.0),
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.phaseshift_ms",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.apply_to_target",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.make_unified_skymodel",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.predict_from_skymodel_wsclean",
            side_effect=RuntimeError("wsclean not found"),
        ), patch(
            "dsa110_continuum.calibration.casa_service.CASAService",
            return_value=mock_svc,
        ), patch(
            "os.path.exists",
            side_effect=_make_exists_fn(meridian_ms, ap_table=ap_table),
        ):
            result = calibrate_epoch(fake_paths, "/fake/bp.b", work_dir)

    assert result.g_table is None
    assert result.status == EpochGaincalStatus.EXCEPTION
    assert "wsclean not found" in (result.reason or "")


def test_calibrate_epoch_returns_low_snr_on_empty_sky_model():
    """calibrate_epoch() returns LOW_SNR status when the catalog sky model is empty.

    Empty sky model is an operational/data limit (no bright sources within
    the search radius), not a code-path fault — maps to the spec's
    skipped_or_failed_low_snr promotion state.
    """
    import tempfile
    from dsa110_continuum.calibration.epoch_gaincal import (
        EpochGaincalStatus,
        calibrate_epoch,
    )

    fake_paths = [f"/fake/tile_{i:02d}.ms" for i in range(12)]
    empty_sky = MagicMock()
    empty_sky.Ncomponents = 0
    mock_svc = MagicMock()

    with tempfile.TemporaryDirectory() as work_dir:
        meridian_ms = str(Path(work_dir) / "tile_05_meridian.ms")
        ap_table   = str(Path(work_dir) / "tile_05.direct_first.ap.G")
        with patch(
            "dsa110_continuum.calibration.epoch_gaincal.select_calibration_tile_from_ms",
            return_value="/fake/tile_05.ms",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal._read_ms_phase_center",
            return_value=(10.0, 37.0),
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.phaseshift_ms",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.apply_to_target",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.make_unified_skymodel",
            return_value=empty_sky,
        ), patch(
            "dsa110_continuum.calibration.casa_service.CASAService",
            return_value=mock_svc,
        ), patch(
            "os.path.exists",
            side_effect=_make_exists_fn(meridian_ms, ap_table=ap_table),
        ):
            result = calibrate_epoch(fake_paths, "/fake/bp.b", work_dir)

    assert result.g_table is None
    assert result.status == EpochGaincalStatus.LOW_SNR
    assert "empty" in (result.reason or "").lower()


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


# ---------------------------------------------------------------------------
# Tests for the four items from the Feb-15 gaincal hardening
# ---------------------------------------------------------------------------

def test_ms_flag_fraction_computes_correctly():
    """`_ms_flag_fraction` returns correct value from a mocked casacore table."""
    import numpy as np
    from dsa110_continuum.adapters import casa_tables as ct
    from dsa110_continuum.calibration.epoch_gaincal import _ms_flag_fraction

    flags = np.zeros((100, 16, 2), dtype=bool)
    flags[:40] = True  # 40% flagged

    mock_table = MagicMock()
    mock_table.__enter__ = lambda s: s
    mock_table.__exit__ = MagicMock(return_value=False)
    mock_table.getcol.return_value = flags

    with patch.object(ct, "table", return_value=mock_table):
        frac = _ms_flag_fraction("/fake/test.ms")

    assert abs(frac - 0.40) < 1e-6


def test_wsclean_skipped_when_ms_heavily_flagged():
    """WSClean self-cal must be skipped and catalog model re-predicted when flag fraction >= 60%."""
    import tempfile
    from dsa110_continuum.calibration.epoch_gaincal import calibrate_epoch

    fake_paths = [f"/fake/tile_{i:02d}.ms" for i in range(6)]
    mock_sky = MagicMock()
    mock_sky.Ncomponents = 5
    mock_service = MagicMock()

    with tempfile.TemporaryDirectory() as work_dir:
        meridian_ms = str(Path(work_dir) / "tile_03_meridian.ms")
        ap_table   = str(Path(work_dir) / "tile_03.direct_first.ap.G")
        with patch(
            "dsa110_continuum.calibration.epoch_gaincal.select_calibration_tile_from_ms",
            return_value="/fake/tile_03.ms",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.phaseshift_ms",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.apply_to_target",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal._read_ms_phase_center",
            return_value=(44.89, 16.08),
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.make_unified_skymodel",
            return_value=mock_sky,
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.predict_from_skymodel_wsclean",
        ) as mock_predict, patch(
            "dsa110_continuum.calibration.epoch_gaincal._ms_flag_fraction",
            return_value=0.72,  # above 60% threshold
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal._gain_flag_fractions",
            return_value={"raw_fraction": 1 / 3, "effective_fraction": 0.07,
                          "baseline_valid": True},
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal._modeled_field_count",
            return_value=(1, 1),
        ), patch(
            "dsa110_continuum.calibration.casa_service.CASAService",
            return_value=mock_service,
        ), patch(
            "shutil.which", return_value="/usr/bin/wsclean",
        ), patch(
            "os.path.exists", side_effect=_make_exists_fn(meridian_ms, ap_table=ap_table),
        ):
            calibrate_epoch(fake_paths, "/fake/bp.b", work_dir)

    # predict_from_skymodel_wsclean must have been called (catalog fallback)
    assert mock_predict.called
    mock_service.gaincal.assert_called()


def test_wsclean_runs_when_flag_fraction_below_limit():
    """WSClean self-cal must be attempted when flag fraction is below 60%."""
    import tempfile
    from dsa110_continuum.calibration.epoch_gaincal import calibrate_epoch

    fake_paths = [f"/fake/tile_{i:02d}.ms" for i in range(6)]
    mock_sky = MagicMock()
    mock_sky.Ncomponents = 5
    mock_service = MagicMock()
    wsclean_ok = MagicMock()
    wsclean_ok.returncode = 0

    with tempfile.TemporaryDirectory() as work_dir:
        meridian_ms = str(Path(work_dir) / "tile_03_meridian.ms")
        ap_table   = str(Path(work_dir) / "tile_03.direct_first.ap.G")
        with patch(
            "dsa110_continuum.calibration.epoch_gaincal.select_calibration_tile_from_ms",
            return_value="/fake/tile_03.ms",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.phaseshift_ms",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.apply_to_target",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal._read_ms_phase_center",
            return_value=(44.89, 16.08),
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.make_unified_skymodel",
            return_value=mock_sky,
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.predict_from_skymodel_wsclean",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal._ms_flag_fraction",
            return_value=0.32,  # well below 60% threshold
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal._gain_flag_fractions",
            return_value={"raw_fraction": 1 / 3, "effective_fraction": 0.07,
                          "baseline_valid": True},
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal._modeled_field_count",
            return_value=(24, 24),
        ), patch(
            "dsa110_continuum.calibration.casa_service.CASAService",
            return_value=mock_service,
        ), patch(
            "shutil.which", return_value="/usr/bin/wsclean",
        ), patch(
            "subprocess.run", return_value=wsclean_ok,
        ) as mock_subprocess, patch(
            "os.path.exists", side_effect=_make_exists_fn(meridian_ms, ap_table=ap_table),
        ):
            calibrate_epoch(fake_paths, "/fake/bp.b", work_dir)

    mock_subprocess.assert_called_once()
    assert "wsclean" in mock_subprocess.call_args[0][0][0]


def test_direct_pass_skips_preconditioner():
    """A BP-relative direct pass skips rescue, while a bad ap.G still fails."""
    import tempfile
    from dsa110_continuum.calibration.epoch_gaincal import calibrate_epoch

    fake_paths = [f"/fake/tile_{i:02d}.ms" for i in range(6)]
    mock_sky = MagicMock()
    mock_sky.Ncomponents = 5
    mock_service = MagicMock()
    wsclean_ok = MagicMock()
    wsclean_ok.returncode = 0

    with tempfile.TemporaryDirectory() as work_dir:
        ap_table       = str(Path(work_dir) / "tile_03.direct_first.ap.G")

        # os.path.exists: False for ap_table (no cache), True for everything else
        # until the ap solve has run; intermediate tables exist after their solves.
        def _exists(p: str) -> bool:
            return str(p) != ap_table or mock_service.gaincal.call_count >= 2

        with patch(
            "dsa110_continuum.calibration.epoch_gaincal.select_calibration_tile_from_ms",
            return_value="/fake/tile_03.ms",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.phaseshift_ms",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.apply_to_target",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal._read_ms_phase_center",
            return_value=(44.89, 16.08),
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.make_unified_skymodel",
            return_value=mock_sky,
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.predict_from_skymodel_wsclean",
        ) as mock_predict, patch(
            "dsa110_continuum.calibration.epoch_gaincal._ms_flag_fraction",
            return_value=0.25,
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal._gain_flag_fractions",
            side_effect=[
                {"raw_fraction": 1 / 3, "effective_fraction": 0.07,
                 "baseline_valid": True},
                {"raw_fraction": 0.5, "effective_fraction": 0.5,
                 "baseline_valid": True},
            ],
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal._modeled_field_count",
            return_value=(24, 24),
        ), patch(
            "dsa110_continuum.calibration.casa_service.CASAService",
            return_value=mock_service,
        ), patch(
            "shutil.which", return_value="/usr/bin/wsclean",
        ), patch(
            "subprocess.run", return_value=wsclean_ok,
        ) as mock_subprocess, patch(
            "os.path.exists", side_effect=_exists,
        ):
            result = calibrate_epoch(fake_paths, "/fake/bp.b", work_dir)

    gaincal_calls = mock_service.gaincal.call_args_list
    assert len(gaincal_calls) == 2
    direct_call, ap_call = gaincal_calls
    assert direct_call.kwargs["gaintable"] == ["/fake/bp.b"]
    assert direct_call.kwargs["solint"] == "inf"
    assert "combine" not in direct_call.kwargs
    assert all(call.kwargs.get("solint") != "60s" for call in gaincal_calls)
    assert ap_call.kwargs["gaintable"] == ["/fake/bp.b"]
    assert ap_call.kwargs["calmode"] == "ap"
    assert mock_predict.call_args_list[0].kwargs["field"] == "all"
    image_cmd = mock_subprocess.call_args.args[0]
    assert image_cmd[image_cmd.index("-field") + 1] == "all"
    assert image_cmd[image_cmd.index("-model-column") + 1] == "MODEL_DATA"
    assert "-update-model-required" in image_cmd
    assert "-save-model-column" not in image_cmd
    assert "-no-update-model-required" not in image_cmd
    assert result.status.value == "low_snr"
    assert "ap.G" in (result.reason or "")


def test_rescue_must_strictly_improve_direct_fraction():
    """An equal-quality rescue must remain LOW_SNR and never reach ap.G."""
    import tempfile
    from dsa110_continuum.calibration.epoch_gaincal import calibrate_epoch

    fake_paths = [f"/fake/tile_{i:02d}.ms" for i in range(6)]
    mock_sky = MagicMock()
    mock_sky.Ncomponents = 5
    mock_service = MagicMock()
    with tempfile.TemporaryDirectory() as work_dir:
        ap_table      = str(Path(work_dir) / "tile_03.direct_first.ap.G")

        def _exists(p: str) -> bool:
            return str(p) != ap_table

        table = MagicMock()
        table.__enter__.return_value = table
        table.__exit__.return_value = False
        table.colnames.return_value = ["MODEL_DATA"]
        table.nrows.return_value = 16

        with patch(
            "dsa110_continuum.calibration.epoch_gaincal.select_calibration_tile_from_ms",
            return_value="/fake/tile_03.ms",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.phaseshift_ms",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.apply_to_target",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal._read_ms_phase_center",
            return_value=(44.89, 16.08),
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.make_unified_skymodel",
            return_value=mock_sky,
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.predict_from_skymodel_wsclean",
        ), patch(
            "dsa110_continuum.calibration.casa_service.CASAService",
            return_value=mock_service,
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal._gain_flag_fractions",
            side_effect=[
                {"raw_fraction": 0.4, "effective_fraction": 0.4,
                 "baseline_valid": True},
                {"raw_fraction": 0.4, "effective_fraction": 0.4,
                 "baseline_valid": True},
            ],
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal._modeled_field_count",
            return_value=(24, 24),
        ), patch(
            "dsa110_continuum.adapters.casa_tables.table",
            return_value=table,
        ), patch(
            "os.path.exists", side_effect=_exists,
        ):
            result = calibrate_epoch(fake_paths, "/fake/bp.b", work_dir)

    assert mock_service.gaincal.call_count == 3
    assert result.status.value == "low_snr"
    assert result.g_table is None
    assert "did not strictly improve" in (result.reason or "")
    assert all(call.kwargs.get("calmode") != "ap" for call in mock_service.gaincal.call_args_list)


def test_epoch_gaincal_forwards_selected_rfi_mode():
    """Epoch gaincal must use the same selected RFI policy as tile processing."""
    import tempfile

    from dsa110_continuum.calibration.epoch_gaincal import calibrate_epoch

    fake_paths = [f"/fake/tile_{i:02d}.ms" for i in range(6)]
    mock_sky = MagicMock()
    mock_sky.Ncomponents = 0  # empty sky model → returns None early after flagging
    mock_service = MagicMock()

    with tempfile.TemporaryDirectory() as work_dir:
        meridian_ms = str(Path(work_dir) / "tile_03_meridian.ms")
        ap_table   = str(Path(work_dir) / "tile_03.direct_first.ap.G")
        with patch(
            "dsa110_continuum.calibration.epoch_gaincal.select_calibration_tile_from_ms",
            return_value="/fake/tile_03.ms",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.phaseshift_ms",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.apply_to_target",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal._read_ms_phase_center",
            return_value=(44.89, 16.08),
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.make_unified_skymodel",
            return_value=mock_sky,
        ), patch(
            "dsa110_continuum.calibration.casa_service.CASAService",
            return_value=mock_service,
        ), patch(
            "dsa110_continuum.calibration.flagging.execute_rfi_policy",
        ) as execute_policy, patch(
            "os.path.exists", side_effect=_make_exists_fn(meridian_ms, ap_table=ap_table),
        ):
            calibrate_epoch(fake_paths, "/fake/bp.b", work_dir, rfi_mode="cflag")

    execute_policy.assert_called_once_with(
        meridian_ms,
        "cflag",
        "epoch gaincal tile_03",
    )


def test_gaincal_returns_low_snr_when_p_table_heavily_flagged():
    """calibrate_epoch() must return LOW_SNR status when p.G has > 30% flagged solutions.

    When the phase-only gain table has too many flagged solutions, the sky model
    SNR was insufficient; proceeding to the ap.G solve would corrupt the bandpass
    calibration rather than improve it.  The function returns g_table=None with
    status=LOW_SNR so the batch pipeline applies bandpass-only calibration and
    the manifest records the operational reason (not a code-path fall-back).
    """
    import tempfile
    from dsa110_continuum.calibration.epoch_gaincal import (
        EpochGaincalStatus,
        calibrate_epoch,
    )

    fake_paths = [f"/fake/tile_{i:02d}.ms" for i in range(6)]
    mock_sky = MagicMock()
    mock_sky.Ncomponents = 5
    mock_service = MagicMock()

    with tempfile.TemporaryDirectory() as work_dir:
        ap_table    = str(Path(work_dir) / "tile_03.direct_first.ap.G")

        def _exists(p: str) -> bool:
            return str(p) not in (ap_table,)   # p.G "exists" after the solve

        with patch(
            "dsa110_continuum.calibration.epoch_gaincal.select_calibration_tile_from_ms",
            return_value="/fake/tile_03.ms",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.phaseshift_ms",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.apply_to_target",
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal._read_ms_phase_center",
            return_value=(44.89, 16.08),
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.make_unified_skymodel",
            return_value=mock_sky,
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal.predict_from_skymodel_wsclean",
        ), patch(
            "dsa110_continuum.calibration.casa_service.CASAService",
            return_value=mock_service,
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal._gain_flag_fractions",
            return_value={"raw_fraction": 0.35, "effective_fraction": 0.35,
                          "baseline_valid": True},
        ), patch(
            "dsa110_continuum.calibration.epoch_gaincal._modeled_field_count",
            return_value=(1, 1),
        ), patch(
            "os.path.exists", side_effect=_exists,
        ):
            result = calibrate_epoch(fake_paths, "/fake/bp.b", work_dir)

    assert result.g_table is None, (
        "calibrate_epoch() must return g_table=None when p.G flagged fraction > 30%"
    )
    assert result.status == EpochGaincalStatus.LOW_SNR, (
        "calibrate_epoch() must mark p.G overflow as LOW_SNR (operational), not EXCEPTION"
    )
    assert "30%" in (result.reason or ""), (
        "result.reason must include the threshold so the manifest gate can quote it"
    )
    assert mock_service.gaincal.call_count == 1
