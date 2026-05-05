"""Tests for ``run_pre_calibration_flagging`` — the pre-cal orchestration helper.

These tests assert the helper coordinates pre-cal flagging primitives
(autocorr flag, AOFlagger RFI, dead-antenna detection, optional single-pol
detection) in the right order with the right defaults. The primitives are
tested separately in:
- ``test_flagging.py`` (low-level CASA wrappers)
- ``test_detect_dead_antennas.py`` (full-antenna dead detection)
- ``test_detect_bad_polarizations.py`` (single-pol detection)

These tests verify the orchestration shape, not primitive correctness.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from dsa110_continuum.calibration.flagging import run_pre_calibration_flagging


def _empty_dead_result() -> dict:
    return {
        "n_dead": 0,
        "dead_antennas": [],
        "total_flagged_before": 0.0,
        "total_flagged_after": 0.0,
    }


def test_default_invocation_runs_autocorr_aoflagger_and_dead_detection(tmp_path):
    """Default flow: autocorr flag + AOFlagger RFI + dead-antenna detection.

    Bad-pol detection is OFF by default (cautious rollout — must be opt-in).
    """
    ms_file = str(tmp_path / "fake.ms")
    mock_service = MagicMock()
    mock_service_class = MagicMock(return_value=mock_service)
    mock_dead_result = _empty_dead_result()

    with patch(
        "dsa110_continuum.calibration.flagging.CASAService", mock_service_class
    ), patch(
        "dsa110_continuum.calibration.flagging.flag_rfi"
    ) as mock_flag_rfi, patch(
        "dsa110_continuum.calibration.flagging.detect_and_flag_dead_antennas",
        return_value=mock_dead_result,
    ) as mock_dead, patch(
        "dsa110_continuum.calibration.flagging.detect_and_flag_bad_polarizations"
    ) as mock_bad_pol:
        result = run_pre_calibration_flagging(ms_file)

    # autocorr flagdata call (one of potentially many flagdata calls — find by kwarg).
    autocorr_calls = [
        c for c in mock_service.flagdata.call_args_list
        if c.kwargs.get("autocorr") is True
    ]
    assert len(autocorr_calls) == 1, (
        f"expected exactly 1 autocorr flagdata call, got {mock_service.flagdata.call_args_list}"
    )

    mock_flag_rfi.assert_called_once_with(ms_file, backend="aoflagger")
    mock_dead.assert_called_once_with(ms_file, threshold=0.95, dry_run=False)
    # Cautious-rollout default: bad-pol detection must be off.
    mock_bad_pol.assert_not_called()

    assert result["dead_result"] == mock_dead_result
    assert result["bad_pol_result"] is None


def test_do_flagging_false_skips_autocorr_and_aoflagger(tmp_path):
    """``do_flagging=False`` ⇒ CASAService not instantiated, flag_rfi not called.

    Dead-antenna detection still runs unconditionally — it's pre-cal hygiene
    that protects against `getcell::TIME` errors in CASA solvers regardless
    of whether RFI flagging happened upstream.
    """
    ms_file = str(tmp_path / "fake.ms")
    mock_service_class = MagicMock()
    mock_dead_result = _empty_dead_result()

    with patch(
        "dsa110_continuum.calibration.flagging.CASAService", mock_service_class
    ), patch(
        "dsa110_continuum.calibration.flagging.flag_rfi"
    ) as mock_flag_rfi, patch(
        "dsa110_continuum.calibration.flagging.detect_and_flag_dead_antennas",
        return_value=mock_dead_result,
    ) as mock_dead:
        result = run_pre_calibration_flagging(ms_file, do_flagging=False)

    mock_service_class.assert_not_called()
    mock_flag_rfi.assert_not_called()
    mock_dead.assert_called_once_with(ms_file, threshold=0.95, dry_run=False)
    assert result["dead_result"] == mock_dead_result


def test_aoflagger_failure_falls_back_to_tfcrop_then_rflag(tmp_path):
    """If ``flag_rfi`` raises, the helper must fall back to CASA tfcrop, then rflag,
    in that order — preserving the inline pre-cal sequence's failure semantics.
    """
    ms_file = str(tmp_path / "fake.ms")
    mock_service = MagicMock()
    mock_service_class = MagicMock(return_value=mock_service)

    with patch(
        "dsa110_continuum.calibration.flagging.CASAService", mock_service_class
    ), patch(
        "dsa110_continuum.calibration.flagging.flag_rfi",
        side_effect=RuntimeError("AOFlagger blew up"),
    ) as mock_flag_rfi, patch(
        "dsa110_continuum.calibration.flagging.detect_and_flag_dead_antennas",
        return_value=_empty_dead_result(),
    ):
        run_pre_calibration_flagging(ms_file)

    mock_flag_rfi.assert_called_once_with(ms_file, backend="aoflagger")

    # All flagdata calls in order — autocorr first, then tfcrop, then rflag.
    modes = [c.kwargs.get("mode") for c in mock_service.flagdata.call_args_list]
    autocorr_flags = [c.kwargs.get("autocorr") for c in mock_service.flagdata.call_args_list]
    # Assert autocorr was the first flagdata call (mode is None there; autocorr=True).
    assert autocorr_flags[0] is True, modes
    # Assert tfcrop and rflag follow, in that order.
    fallback_modes = [m for m in modes if m in ("tfcrop", "rflag")]
    assert fallback_modes == ["tfcrop", "rflag"], modes


def test_dead_antenna_detection_failure_is_swallowed(tmp_path):
    """If ``detect_and_flag_dead_antennas`` raises, the helper must NOT propagate
    the exception. Returns successfully with ``dead_result=None``. Mirrors the
    ``# noqa: BLE001 - science-safe: never abort the pipeline here`` invariant.
    """
    ms_file = str(tmp_path / "fake.ms")
    mock_service_class = MagicMock(return_value=MagicMock())

    with patch(
        "dsa110_continuum.calibration.flagging.CASAService", mock_service_class
    ), patch(
        "dsa110_continuum.calibration.flagging.flag_rfi"
    ), patch(
        "dsa110_continuum.calibration.flagging.detect_and_flag_dead_antennas",
        side_effect=RuntimeError("synthetic dead-ant failure"),
    ):
        result = run_pre_calibration_flagging(ms_file)

    assert result["dead_result"] is None
    assert result["bad_pol_result"] is None


# --------------------------------------------------------------------------------------
# Phase B step 2: opt-in single-polarization detection wiring.
#
# Production safety: detection is OFF by default. Callers must explicitly pass
# ``enable_bad_pol_detection=True`` to activate it. ``bad_pol_phase_table`` and
# ``bad_pol_dry_run`` propagate to ``detect_and_flag_bad_polarizations``.
# Failures in the detection do NOT abort the pipeline.


def test_enable_bad_pol_detection_invokes_detector_and_populates_result(tmp_path):
    """``enable_bad_pol_detection=True`` ⇒ ``detect_and_flag_bad_polarizations``
    is called once with the propagated phase_table + dry_run, and its return
    value populates ``result["bad_pol_result"]``.
    """
    ms_file = str(tmp_path / "fake.ms")
    mock_service_class = MagicMock(return_value=MagicMock())
    mock_dead_result = _empty_dead_result()
    mock_bad_pol_result = {
        "bad_polarizations": [(3, 0, "XX")],
        "n_antennas_affected": 1,
        "action_taken": False,
        "detection_method": "phase_table",
        "total_flagged_before": 0.01,
        "total_flagged_after": 0.012,
    }

    with patch(
        "dsa110_continuum.calibration.flagging.CASAService", mock_service_class
    ), patch(
        "dsa110_continuum.calibration.flagging.flag_rfi"
    ), patch(
        "dsa110_continuum.calibration.flagging.detect_and_flag_dead_antennas",
        return_value=mock_dead_result,
    ), patch(
        "dsa110_continuum.calibration.flagging.detect_and_flag_bad_polarizations",
        return_value=mock_bad_pol_result,
    ) as mock_bad_pol:
        result = run_pre_calibration_flagging(
            ms_file,
            enable_bad_pol_detection=True,
            bad_pol_phase_table="/tmp/some.gcal",
            bad_pol_dry_run=True,
        )

    mock_bad_pol.assert_called_once()
    _args, kwargs = mock_bad_pol.call_args
    # Helper must propagate the dry_run + phase_table kwargs verbatim.
    assert kwargs.get("dry_run") is True
    assert kwargs.get("phase_table") == "/tmp/some.gcal"
    # ms path should appear positionally or as kwarg
    if _args:
        assert _args[0] == ms_file
    else:
        assert kwargs.get("ms_path") == ms_file

    assert result["bad_pol_result"] == mock_bad_pol_result


def test_bad_pol_detection_failure_is_swallowed(tmp_path):
    """If ``detect_and_flag_bad_polarizations`` raises, the helper must NOT
    propagate the exception — pipeline continues, ``bad_pol_result`` stays
    ``None``. Mirrors the dead-ant invariant.
    """
    ms_file = str(tmp_path / "fake.ms")
    mock_service_class = MagicMock(return_value=MagicMock())

    with patch(
        "dsa110_continuum.calibration.flagging.CASAService", mock_service_class
    ), patch(
        "dsa110_continuum.calibration.flagging.flag_rfi"
    ), patch(
        "dsa110_continuum.calibration.flagging.detect_and_flag_dead_antennas",
        return_value=_empty_dead_result(),
    ), patch(
        "dsa110_continuum.calibration.flagging.detect_and_flag_bad_polarizations",
        side_effect=RuntimeError("synthetic bad-pol failure"),
    ):
        result = run_pre_calibration_flagging(ms_file, enable_bad_pol_detection=True)

    assert result["bad_pol_result"] is None
