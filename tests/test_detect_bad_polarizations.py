"""Tests for ``detect_and_flag_bad_polarizations`` — pre-cal single-pol detection.

Mock-based unit tests that characterize the function's behavior on synthetic
visibility data via the ``casa_tables`` adapter, mirroring the structure of
``test_detect_dead_antennas.py``. A separate real-MS integration smoke test
exercises the wiring end-to-end (see ``test_bad_pol_wiring_smoke.py``, planned).

Coverage targets (per the TDD plan):
- Clean MS (both pols equally coherent) → no bad polarizations detected.
- One antenna with one decoherent polarization → detected and reported.
- ``dry_run=True`` → no ``flagdata`` call; ``dry_run=False`` → ``flagdata``
  called with correct ``(antenna, correlation)`` selectors.
- Phase-table primary path vs. MS-coherence fallback (separate test).
- SNR-ratio threshold boundary on the phase-table path (separate test).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from dsa110_continuum.calibration.flagging import detect_and_flag_bad_polarizations


def _baseline_pairs(n_ant: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ANTENNA1, ANTENNA2 arrays for all i<j cross-correlation baselines."""
    pairs = [(i, j) for i in range(n_ant) for j in range(i + 1, n_ant)]
    return (
        np.array([p[0] for p in pairs], dtype=int),
        np.array([p[1] for p in pairs], dtype=int),
    )


def _coherent_visibilities(
    n_rows: int, n_chan: int = 4, n_pol: int = 2
) -> np.ndarray:
    """Phase-aligned unit-amplitude visibilities — both pols perfectly coherent."""
    return np.ones((n_rows, n_chan, n_pol), dtype=complex)


class _MockMSTable:
    """Context-manager double for ``casa_tables.table`` over an MS row table."""

    def __init__(
        self,
        ant1: np.ndarray,
        ant2: np.ndarray,
        data: np.ndarray,
        flags: np.ndarray,
    ) -> None:
        self._cols = {"ANTENNA1": ant1, "ANTENNA2": ant2, "DATA": data, "FLAG": flags}

    def __enter__(self) -> "_MockMSTable":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def nrows(self) -> int:
        return int(self._cols["ANTENNA1"].shape[0])

    def getcol(self, name: str) -> np.ndarray:
        if name not in self._cols:
            raise KeyError(name)
        return self._cols[name]


def _patch_ms_table(
    ant1: np.ndarray, ant2: np.ndarray, data: np.ndarray, flags: np.ndarray
):
    """Patch the ``table`` factory on the casa_tables adapter."""
    def _factory(*_args, **_kwargs):
        return _MockMSTable(ant1, ant2, data, flags)

    return patch("dsa110_continuum.adapters.casa_tables.table", new=_factory)


def test_clean_ms_returns_no_bad_polarizations(tmp_path):
    """Both pols equally coherent → empty bad_polarizations list, no false positives.

    This is the highest-stakes assertion for production rollout: a healthy MS
    must not produce spurious flag actions.
    """
    ms_path = str(tmp_path / "fake.ms")
    ant1, ant2 = _baseline_pairs(n_ant=8)
    data = _coherent_visibilities(n_rows=len(ant1))
    flags = np.zeros(data.shape, dtype=bool)

    with _patch_ms_table(ant1, ant2, data, flags), patch(
        "dsa110_continuum.calibration.flagging.flag_summary",
        return_value={"total_fraction_flagged": 0.0},
    ):
        result = detect_and_flag_bad_polarizations(
            ms_path, dry_run=True, phase_table=None
        )

    assert result["bad_polarizations"] == []
    assert result["n_antennas_affected"] == 0
    assert result["action_taken"] is False
    assert result["detection_method"] == "ms_coherence"


def test_one_antenna_with_decoherent_xx_pol_is_detected(tmp_path):
    """Antenna with random-phase XX (low coherence) and aligned YY → flagged on XX only.

    Construction: all baselines start phase-aligned. Then for every baseline involving
    the chosen ``bad_ant``, we replace its XX (pol 0) data with random-phase unit-amp
    complex values. Per-antenna coherence:
      - bad_ant XX: ~0 (random phase across ~28 samples → vector_avg → 0)
      - bad_ant YY: ~1 (all aligned)
      - other antennas: only 1 of their 7 baselines has the random XX, so their XX
        coherence drops to ~0.85 (still well above 0.5 threshold).

    With ``coherence_ratio_threshold=2.0`` (the hard-coded fallback threshold), only
    bad_ant's XX should be flagged.
    """
    ms_path = str(tmp_path / "fake.ms")
    n_ant = 8
    bad_ant = 3
    n_chan = 4

    ant1, ant2 = _baseline_pairs(n_ant=n_ant)
    n_rows = len(ant1)
    data = _coherent_visibilities(n_rows=n_rows, n_chan=n_chan)

    rng = np.random.default_rng(seed=42)
    bad_rows_mask = (ant1 == bad_ant) | (ant2 == bad_ant)
    random_phases = rng.uniform(-np.pi, np.pi, size=(int(bad_rows_mask.sum()), n_chan))
    data[bad_rows_mask, :, 0] = np.exp(1j * random_phases)

    flags = np.zeros(data.shape, dtype=bool)

    with _patch_ms_table(ant1, ant2, data, flags), patch(
        "dsa110_continuum.calibration.flagging.flag_summary",
        return_value={"total_fraction_flagged": 0.0},
    ):
        result = detect_and_flag_bad_polarizations(
            ms_path, dry_run=True, phase_table=None
        )

    bad_pols = result["bad_polarizations"]
    assert len(bad_pols) == 1, f"expected exactly 1 bad pol, got {bad_pols}"
    ant_id, pol_idx, pol_name = bad_pols[0]
    assert ant_id == bad_ant
    assert pol_idx == 0
    assert pol_name == "XX"
    assert result["n_antennas_affected"] == 1
    assert result["action_taken"] is False
    assert result["detection_method"] == "ms_coherence"


def _bad_xx_dataset(n_ant: int = 8, bad_ant: int = 3, n_chan: int = 4, seed: int = 42):
    """Build the canonical 'one decoherent XX pol on bad_ant' fixture used by T2/T3."""
    ant1, ant2 = _baseline_pairs(n_ant=n_ant)
    n_rows = len(ant1)
    data = _coherent_visibilities(n_rows=n_rows, n_chan=n_chan)
    rng = np.random.default_rng(seed=seed)
    bad_rows_mask = (ant1 == bad_ant) | (ant2 == bad_ant)
    random_phases = rng.uniform(-np.pi, np.pi, size=(int(bad_rows_mask.sum()), n_chan))
    data[bad_rows_mask, :, 0] = np.exp(1j * random_phases)
    flags = np.zeros(data.shape, dtype=bool)
    return ant1, ant2, data, flags, bad_ant


def test_non_dry_run_calls_flagdata_with_per_pol_selectors(tmp_path):
    """``dry_run=False`` with a detected bad pol → ``CASAService().flagdata`` called
    once with ``mode="manual"``, ``antenna=str(bad_ant)``, ``correlation="XX"``,
    ``action="apply"``. ``result["action_taken"]`` should be ``True``.

    Asserts per-polarization granularity: only the affected pol selector is passed,
    not the whole antenna.
    """
    ms_path = str(tmp_path / "fake.ms")
    ant1, ant2, data, flags, bad_ant = _bad_xx_dataset()

    mock_service = MagicMock()
    mock_service_class = MagicMock(return_value=mock_service)

    with _patch_ms_table(ant1, ant2, data, flags), patch(
        "dsa110_continuum.calibration.flagging.flag_summary",
        return_value={"total_fraction_flagged": 0.0},
    ), patch(
        "dsa110_continuum.calibration.flagging.CASAService", mock_service_class
    ):
        result = detect_and_flag_bad_polarizations(
            ms_path, dry_run=False, phase_table=None
        )

    mock_service_class.assert_called_once()
    mock_service.flagdata.assert_called_once()
    _args, kwargs = mock_service.flagdata.call_args
    assert kwargs["vis"] == ms_path
    assert kwargs["mode"] == "manual"
    assert kwargs["antenna"] == str(bad_ant)
    assert kwargs["correlation"] == "XX"
    assert kwargs["action"] == "apply"
    assert result["action_taken"] is True


def test_dry_run_does_not_instantiate_casa_service(tmp_path):
    """``dry_run=True`` with a detected bad pol → ``CASAService`` is never
    instantiated (no flagdata can fire)."""
    ms_path = str(tmp_path / "fake.ms")
    ant1, ant2, data, flags, _ = _bad_xx_dataset()

    mock_service_class = MagicMock()

    with _patch_ms_table(ant1, ant2, data, flags), patch(
        "dsa110_continuum.calibration.flagging.flag_summary",
        return_value={"total_fraction_flagged": 0.0},
    ), patch(
        "dsa110_continuum.calibration.flagging.CASAService", mock_service_class
    ):
        result = detect_and_flag_bad_polarizations(
            ms_path, dry_run=True, phase_table=None
        )

    mock_service_class.assert_not_called()
    assert result["action_taken"] is False
    # The detection itself still ran; it found the bad pol but didn't flag it.
    assert result["n_antennas_affected"] == 1


def test_one_antenna_with_decoherent_yy_pol_is_detected(tmp_path):
    """Mirror of the XX test on the YY axis — guards against off-by-one in the
    coherence-ratio comparison logic (different code branch in the function).
    """
    ms_path = str(tmp_path / "fake.ms")
    n_ant = 8
    bad_ant = 5
    n_chan = 4

    ant1, ant2 = _baseline_pairs(n_ant=n_ant)
    n_rows = len(ant1)
    data = _coherent_visibilities(n_rows=n_rows, n_chan=n_chan)

    rng = np.random.default_rng(seed=7)
    bad_rows_mask = (ant1 == bad_ant) | (ant2 == bad_ant)
    random_phases = rng.uniform(-np.pi, np.pi, size=(int(bad_rows_mask.sum()), n_chan))
    data[bad_rows_mask, :, 1] = np.exp(1j * random_phases)  # pol 1 = YY

    flags = np.zeros(data.shape, dtype=bool)

    with _patch_ms_table(ant1, ant2, data, flags), patch(
        "dsa110_continuum.calibration.flagging.flag_summary",
        return_value={"total_fraction_flagged": 0.0},
    ):
        result = detect_and_flag_bad_polarizations(
            ms_path, dry_run=True, phase_table=None
        )

    bad_pols = result["bad_polarizations"]
    assert len(bad_pols) == 1, f"expected 1 bad YY pol, got {bad_pols}"
    ant_id, pol_idx, pol_name = bad_pols[0]
    assert ant_id == bad_ant
    assert pol_idx == 1
    assert pol_name == "YY"


# --------------------------------------------------------------------------------------
# Phase-table primary path tests
# --------------------------------------------------------------------------------------
#
# These exercise the *recommended* production detection path (cf. function docstring:
# "Primary (if phase_table provided)"). The phase_table branch reads SNR + FLAG columns
# from a CASA calibration table via ``casatools.table`` (not the casa_tables adapter
# used elsewhere), so the mocking strategy differs from the MS-coherence path.


class _MockCaltable:
    """Stand-in for ``casatools.table()`` — opens a caltable and serves columns."""

    def __init__(self, snr: np.ndarray, flags: np.ndarray, antenna1: np.ndarray) -> None:
        self._cols = {"ANTENNA1": antenna1, "SNR": snr, "FLAG": flags}

    def open(self, _path: str) -> None:
        return None

    def close(self) -> None:
        return None

    def getcol(self, name: str) -> np.ndarray:
        if name not in self._cols:
            raise KeyError(name)
        return self._cols[name]


def _patch_caltable(snr: np.ndarray, flags: np.ndarray, antenna1: np.ndarray):
    """Patch the ``casatools.table`` class so the phase-table branch reads our data."""
    def _factory():
        return _MockCaltable(snr, flags, antenna1)

    return patch("casatools.table", new=_factory)


def test_phase_table_clean_caltable_returns_no_bad_polarizations(tmp_path):
    """Clean phase table (both pols comparable SNR, zero flags) → empty result.

    Establishes the phase-table primary path is wired correctly and produces no
    false positives — the matched-clean test for the highest-quality detection
    signal. This is the path runner.py should call once we wire it up.
    """
    n_ant = 8
    antenna1 = np.arange(n_ant, dtype=int)
    # SNR shape: (npol, nchan, nrow). Both pols have comparable SNR, all unflagged.
    snr = np.full((2, 1, n_ant), 15.0, dtype=float)
    flags = np.zeros((2, 1, n_ant), dtype=bool)

    # Phase-table branch requires the file to exist on disk before opening.
    phase_table = tmp_path / "phase.gcal"
    phase_table.touch()
    ms_path = str(tmp_path / "fake.ms")

    with _patch_caltable(snr, flags, antenna1), patch(
        "dsa110_continuum.calibration.flagging.flag_summary",
        return_value={"total_fraction_flagged": 0.0},
    ):
        result = detect_and_flag_bad_polarizations(
            ms_path, dry_run=True, phase_table=str(phase_table)
        )

    assert result["bad_polarizations"] == []
    assert result["n_antennas_affected"] == 0
    assert result["action_taken"] is False
    assert result["detection_method"] == "phase_table"


def test_phase_table_detects_low_snr_yy_via_ratio_threshold(tmp_path):
    """Phase-table secondary check: ``snr0/snr1 > snr_ratio_threshold (5×)`` AND
    ``snr1 < min_good_snr (10.0)`` → YY flagged on that antenna only.

    Concretely: ``snr0=12.0, snr1=1.5`` ⇒ ratio 8.0, well above 5×. snr1 below
    min_good_snr ⇒ flagged. Other antennas have matched ``snr0 = snr1 = 15.0`` so
    their ratio is 1.0 ⇒ not flagged.
    """
    n_ant = 8
    bad_ant_idx = 4
    antenna1 = np.arange(n_ant, dtype=int)
    snr = np.full((2, 1, n_ant), 15.0, dtype=float)
    flags = np.zeros((2, 1, n_ant), dtype=bool)
    snr[0, 0, bad_ant_idx] = 12.0  # XX healthy
    snr[1, 0, bad_ant_idx] = 1.5  # YY low

    phase_table = tmp_path / "phase.gcal"
    phase_table.touch()
    ms_path = str(tmp_path / "fake.ms")

    with _patch_caltable(snr, flags, antenna1), patch(
        "dsa110_continuum.calibration.flagging.flag_summary",
        return_value={"total_fraction_flagged": 0.0},
    ):
        result = detect_and_flag_bad_polarizations(
            ms_path, dry_run=True, phase_table=str(phase_table),
            snr_ratio_threshold=5.0, min_good_snr=10.0,
        )

    bad_pols = result["bad_polarizations"]
    assert len(bad_pols) == 1, f"expected 1 YY-bad antenna, got {bad_pols}"
    ant_id, pol_idx, pol_name = bad_pols[0]
    assert ant_id == bad_ant_idx
    assert pol_idx == 1
    assert pol_name == "YY"
    assert result["detection_method"] == "phase_table"


def test_phase_table_threshold_boundary_just_below_does_not_flag(tmp_path):
    """SNR ratio ``4.44× < 5.0×`` ⇒ NOT flagged, even though snr1 is below
    min_good_snr. Locks in the boundary behavior; mirrors the existing
    boundary test in ``test_detect_dead_antennas.py``.

    Concretely: ``snr0=12.0, snr1=2.7`` ⇒ ratio 4.44.
    """
    n_ant = 8
    boundary_ant_idx = 4
    antenna1 = np.arange(n_ant, dtype=int)
    snr = np.full((2, 1, n_ant), 15.0, dtype=float)
    flags = np.zeros((2, 1, n_ant), dtype=bool)
    snr[0, 0, boundary_ant_idx] = 12.0
    snr[1, 0, boundary_ant_idx] = 2.7  # ratio = 12/2.7 = 4.44 < 5.0

    phase_table = tmp_path / "phase.gcal"
    phase_table.touch()
    ms_path = str(tmp_path / "fake.ms")

    with _patch_caltable(snr, flags, antenna1), patch(
        "dsa110_continuum.calibration.flagging.flag_summary",
        return_value={"total_fraction_flagged": 0.0},
    ):
        result = detect_and_flag_bad_polarizations(
            ms_path, dry_run=True, phase_table=str(phase_table),
            snr_ratio_threshold=5.0, min_good_snr=10.0,
        )

    assert result["bad_polarizations"] == [], (
        f"4.44× ratio should be below threshold but flagged: "
        f"{result['bad_polarizations']}"
    )
    assert result["n_antennas_affected"] == 0
    assert result["detection_method"] == "phase_table"
