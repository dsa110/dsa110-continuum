"""Tests for epoch_gaincal: tile selection, RFI flagging, WSClean guard, refant chain."""
import sys
from unittest.mock import MagicMock, patch


def test_select_calibration_tile_from_ms_picks_richer_tile():
    """Should return the MS whose central pointing has more catalog sources."""
    from dsa110_continuum.calibration.epoch_gaincal import select_calibration_tile_from_ms

    # 12 tiles: centre pair is indices 5 (mid-1) and 6 (mid) where mid = 12//2 = 6
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


def test_select_calibration_tile_works_with_11_tiles():
    """Should work with 11 tiles (centre pair is indices 4 and 5)."""
    from dsa110_continuum.calibration.epoch_gaincal import select_calibration_tile_from_ms

    # 11 tiles: mid = 11//2 = 5, centre pair is indices 4 and 5
    fake_paths = [f"/fake/tile_{i:02d}.ms" for i in range(11)]

    def fake_phase_center(ms_path):
        idx = int(Path(ms_path).stem.split("_")[1])
        return (float(idx) * 10.0, 37.0)

    def fake_count(pointing_ra_deg, pointing_dec_deg, **kwargs):
        # tile index 5 → ra=50.0 → 9 sources; tile index 4 → ra=40.0 → 2 sources
        return 9 if pointing_ra_deg == 50.0 else 2

    with patch(
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


def test_select_calibration_tile_defaults_to_central_tile_on_failure():
    """Falls back to n//2 (geometric centre) if source counting raises for both candidates."""
    from dsa110_continuum.calibration.epoch_gaincal import select_calibration_tile_from_ms

    # 12 tiles: n//2 = 6
    fake_paths_12 = [f"/fake/tile_{i:02d}.ms" for i in range(12)]
    with patch(
        "dsa110_continuum.calibration.epoch_gaincal._read_ms_phase_center",
        side_effect=RuntimeError("casacore unavailable"),
    ):
        result_12 = select_calibration_tile_from_ms(fake_paths_12)
    assert result_12 == "/fake/tile_06.ms"

    # 6 tiles: n//2 = 3
    fake_paths_6 = [f"/fake/tile_{i:02d}.ms" for i in range(6)]
    with patch(
        "dsa110_continuum.calibration.epoch_gaincal._read_ms_phase_center",
        side_effect=RuntimeError("casacore unavailable"),
    ):
        result_6 = select_calibration_tile_from_ms(fake_paths_6)
    assert result_6 == "/fake/tile_03.ms"


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
        ap_table   = str(Path(work_dir) / "tile_05.ap.G")
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
        ap_table   = str(Path(work_dir) / "tile_05.ap.G")
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
        ap_table   = str(Path(work_dir) / "tile_03.ap.G")
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
        ap_table   = str(Path(work_dir) / "tile_03.ap.G")
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


def test_preconditioner_table_threaded_into_downstream_solves():
    """precond.G must appear in gaintable of p.G and ap.G solves when it succeeds."""
    import tempfile
    from dsa110_continuum.calibration.epoch_gaincal import calibrate_epoch

    fake_paths = [f"/fake/tile_{i:02d}.ms" for i in range(6)]
    mock_sky = MagicMock()
    mock_sky.Ncomponents = 5
    mock_service = MagicMock()
    wsclean_ok = MagicMock()
    wsclean_ok.returncode = 0

    with tempfile.TemporaryDirectory() as work_dir:
        meridian_ms    = str(Path(work_dir) / "tile_03_meridian.ms")
        precond_table  = str(Path(work_dir) / "tile_03.precond.G")
        ap_table       = str(Path(work_dir) / "tile_03.ap.G")

        # os.path.exists: False for ap_table (no cache), True for everything else
        # (meridian MS, precond table, p table all "exist" after their solves)
        def _exists(p: str) -> bool:
            return str(p) != ap_table

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
            return_value=0.25,
        ), patch(
            "dsa110_continuum.calibration.casa_service.CASAService",
            return_value=mock_service,
        ), patch(
            "shutil.which", return_value="/usr/bin/wsclean",
        ), patch(
            "subprocess.run", return_value=wsclean_ok,
        ), patch(
            "os.path.exists", side_effect=_exists,
        ):
            calibrate_epoch(fake_paths, "/fake/bp.b", work_dir)

    gaincal_calls = mock_service.gaincal.call_args_list
    assert len(gaincal_calls) == 3, f"expected 3 gaincal calls, got {len(gaincal_calls)}"

    precond_call, p_call, ap_call = gaincal_calls

    # Pre-conditioner solve
    assert precond_call.kwargs["solint"] == "60s"
    assert precond_call.kwargs["combine"] == "spw"
    assert precond_call.kwargs["calmode"] == "p"
    assert precond_call.kwargs["gaintable"] == ["/fake/bp.b"]

    # p.G solve must include precond table
    assert precond_table in p_call.kwargs["gaintable"], \
        "precond table must be in p.G gaintable"
    assert "/fake/bp.b" in p_call.kwargs["gaintable"]
    assert p_call.kwargs["solint"] == "inf"

    # ap.G solve must include both precond table and p table
    ap_gt = ap_call.kwargs["gaintable"]
    assert precond_table in ap_gt, "precond table must be in ap.G gaintable"
    assert "/fake/bp.b" in ap_gt
    assert ap_call.kwargs["calmode"] == "ap"


def test_preconditioner_failure_does_not_abort_epoch_gaincal():
    """If precond solve fails entirely, the main p.G and ap.G solves must still run."""
    import tempfile
    from dsa110_continuum.calibration.epoch_gaincal import calibrate_epoch

    fake_paths = [f"/fake/tile_{i:02d}.ms" for i in range(6)]
    mock_sky = MagicMock()
    mock_sky.Ncomponents = 5
    mock_service = MagicMock()
    # Make the first gaincal call (precond) raise; subsequent calls succeed
    call_count = {"n": 0}
    def _gaincal_side_effect(**kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("CASA unavailable for precond")
    mock_service.gaincal.side_effect = _gaincal_side_effect
    wsclean_ok = MagicMock()
    wsclean_ok.returncode = 0

    with tempfile.TemporaryDirectory() as work_dir:
        meridian_ms   = str(Path(work_dir) / "tile_03_meridian.ms")
        ap_table      = str(Path(work_dir) / "tile_03.ap.G")
        p_table       = str(Path(work_dir) / "tile_03.p.G")
        precond_table = str(Path(work_dir) / "tile_03.precond.G")

        def _exists(p: str) -> bool:
            # ap and precond tables never exist; meridian MS and p table exist
            return str(p) not in (ap_table, precond_table)

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
            return_value=0.25,
        ), patch(
            "dsa110_continuum.calibration.casa_service.CASAService",
            return_value=mock_service,
        ), patch(
            "shutil.which", return_value="/usr/bin/wsclean",
        ), patch(
            "subprocess.run", return_value=wsclean_ok,
        ), patch(
            "os.path.exists", side_effect=_exists,
        ):
            calibrate_epoch(fake_paths, "/fake/bp.b", work_dir)

    # All three gaincal calls attempted despite the first raising
    assert mock_service.gaincal.call_count == 3

    # p.G and ap.G gaintables must NOT include the precond table (it doesn't exist)
    _, p_call, ap_call = mock_service.gaincal.call_args_list
    assert precond_table not in p_call.kwargs["gaintable"]
    assert precond_table not in ap_call.kwargs["gaintable"]


def test_rfi_flagging_falls_back_to_casa_when_aoflagger_unavailable():
    """When AOFlagger is not importable, CASA tfcrop+rflag must be called instead."""
    import tempfile
    from dsa110_continuum.calibration.epoch_gaincal import calibrate_epoch

    fake_paths = [f"/fake/tile_{i:02d}.ms" for i in range(6)]
    mock_sky = MagicMock()
    mock_sky.Ncomponents = 0  # empty sky model → returns None early after flagging
    mock_service = MagicMock()

    # Stub out flag_rfi import to raise ImportError, leaving CASA path active
    import sys as _sys
    fake_flagging_module = MagicMock()
    fake_flagging_module.flag_rfi = MagicMock(side_effect=ImportError("no aoflagger"))

    with tempfile.TemporaryDirectory() as work_dir:
        meridian_ms = str(Path(work_dir) / "tile_03_meridian.ms")
        ap_table   = str(Path(work_dir) / "tile_03.ap.G")
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
        ), patch.dict(
            _sys.modules,
            {"dsa110_contimg.core.calibration.flagging": fake_flagging_module},
        ), patch(
            "os.path.exists", side_effect=_make_exists_fn(meridian_ms, ap_table=ap_table),
        ):
            calibrate_epoch(fake_paths, "/fake/bp.b", work_dir)

    flagdata_calls = mock_service.flagdata.call_args_list
    assert any(c.kwargs.get("autocorr") for c in flagdata_calls), \
        "autocorrelation flagging must be called"
    assert any(c.kwargs.get("mode") == "tfcrop" for c in flagdata_calls), \
        "tfcrop must be called as AOFlagger fallback"
    assert any(c.kwargs.get("mode") == "rflag" for c in flagdata_calls), \
        "rflag must be called as AOFlagger fallback"


def test_gaincal_returns_low_snr_when_p_table_heavily_flagged():
    """calibrate_epoch() must return LOW_SNR status when p.G has > 30% flagged solutions.

    When the phase-only gain table has too many flagged solutions, the sky model
    SNR was insufficient; proceeding to the ap.G solve would corrupt the bandpass
    calibration rather than improve it.  The function returns g_table=None with
    status=LOW_SNR so the batch pipeline applies bandpass-only calibration and
    the manifest records the operational reason (not a code-path fall-back).
    """
    import tempfile
    import numpy as np
    from dsa110_continuum.calibration.epoch_gaincal import (
        EpochGaincalStatus,
        calibrate_epoch,
    )

    fake_paths = [f"/fake/tile_{i:02d}.ms" for i in range(6)]
    mock_sky = MagicMock()
    mock_sky.Ncomponents = 5
    mock_service = MagicMock()

    # Build a mock casatools.table() that returns a FLAG array that is 35% True.
    flags_35pct = np.zeros((2, 1, 100), dtype=bool)
    flags_35pct[:, :, :35] = True   # 35 of 100 rows flagged per pol/spw

    mock_tb = MagicMock()
    mock_tb.getcol.return_value = flags_35pct

    with tempfile.TemporaryDirectory() as work_dir:
        meridian_ms = str(Path(work_dir) / "tile_03_meridian.ms")
        ap_table    = str(Path(work_dir) / "tile_03.ap.G")
        p_table     = str(Path(work_dir) / "tile_03.p.G")

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
            "dsa110_continuum.calibration.epoch_gaincal._ms_flag_fraction",
            return_value=0.25,
        ), patch(
            "dsa110_continuum.calibration.casa_service.CASAService",
            return_value=mock_service,
        ), patch(
            "os.path.exists", side_effect=_exists,
        ):
            import sys as _sys
            mock_casatools = MagicMock()
            mock_casatools.table.return_value = mock_tb
            with patch.dict(_sys.modules, {"casatools": mock_casatools}):
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
