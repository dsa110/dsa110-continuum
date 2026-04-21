"""Write a pyuvdata UVData object to a Measurement Set using casatools.

This replaces ``UVData.write_ms()`` which requires ``python-casacore``
(incompatible with ``casatools`` in the same process due to duplicate
casacore C++ shared libraries).

Only the columns required by the DSA-110 simulation pipeline are written:
DATA, FLAG, UVW, TIME, ANTENNA1, ANTENNA2, plus the ANTENNA, FIELD,
SPECTRAL_WINDOW, DATA_DESCRIPTION, POLARIZATION, and OBSERVATION subtables.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from dsa110_continuum.adapters.casa_tables import table as _Table

__all__ = ["uvdata_to_ms"]

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pyuvdata

try:
    import casatools as _casatools
except ModuleNotFoundError:  # pragma: no cover - depends on CASA runtime
    _casatools = None


_AIPS_TO_CASA_CORR = {
    -5: 9,   # XX
    -7: 10,  # XY
    -8: 11,  # YX
    -6: 12,  # YY
    -1: 5,   # RR
    -3: 6,   # RL
    -4: 7,   # LR
    -2: 8,   # LL
}

_CASA_CORR_TO_PRODUCT = {
    5: (0, 0),   # RR
    6: (0, 1),   # RL
    7: (1, 0),   # LR
    8: (1, 1),   # LL
    9: (0, 0),   # XX
    10: (0, 1),  # XY
    11: (1, 0),  # YX
    12: (1, 1),  # YY
}


def _require_casatools():
    if _casatools is None:
        raise RuntimeError(
            "casatools is not installed in this environment. "
            "Install modular CASA 6 to use uvdata_to_ms()."
        )
    return _casatools


def _normalize_freqs(freq_array: np.ndarray, expected_nfreqs: int) -> np.ndarray:
    freqs = np.asarray(freq_array, dtype=np.float64).reshape(-1)
    if freqs.size != int(expected_nfreqs):
        raise ValueError(
            f"freq_array contains {freqs.size} values; expected {expected_nfreqs}"
        )
    return freqs


def _normalize_vis_cube(
    arr: np.ndarray,
    *,
    n_rows: int,
    n_chan: int,
    n_pol: int,
    name: str,
) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 4:
        # Legacy/non-future-array-shapes path: (Nblts, Nspws, Nfreqs, Npols)
        if a.shape[1] != 1:
            raise ValueError(
                f"{name} has {a.shape[1]} spectral windows; uvdata_to_ms currently "
                "supports only single-SPW writes"
            )
        a = a[:, 0, :, :]
    if a.ndim != 3:
        raise ValueError(f"{name} must be 3-D or 4-D, got shape {a.shape}")
    expected = (n_rows, n_chan, n_pol)
    if a.shape != expected:
        raise ValueError(f"{name} has shape {a.shape}, expected {expected}")
    return a


def uvdata_to_ms(uv: pyuvdata.UVData, ms_path: str | Path) -> Path:
    """Write a UVData object to a CASA Measurement Set.

    Parameters
    ----------
    uv
        Populated ``pyuvdata.UVData`` object.
    ms_path
        Output path for the MS directory.

    Returns
    -------
    Path
        The written MS path.
    """
    ms_path = Path(ms_path)
    if ms_path.exists():
        import shutil
        shutil.rmtree(ms_path)

    casatools = _require_casatools()
    sim = casatools.simulator()
    sim.open(str(ms_path))
    sim.close()

    tel = uv.telescope
    n_rows = int(uv.Nblts)
    n_pol = int(uv.Npols)
    freqs = _normalize_freqs(uv.freq_array, int(uv.Nfreqs))
    n_chan = int(freqs.size)
    data_cube = _normalize_vis_cube(
        uv.data_array, n_rows=n_rows, n_chan=n_chan, n_pol=n_pol, name="data_array"
    ).astype(np.complex64)
    flag_cube = None
    if uv.flag_array is not None:
        flag_cube = _normalize_vis_cube(
            uv.flag_array, n_rows=n_rows, n_chan=n_chan, n_pol=n_pol, name="flag_array"
        ).astype(bool)
    # casacore expects DATA/FLAG as (row, chan, pol), but casatools putcol with
    # the compatibility wrapper effectively interprets in (row, pol, chan) order.
    data_for_putcol = np.swapaxes(data_cube, 1, 2)
    flag_for_putcol = np.swapaxes(flag_cube, 1, 2) if flag_cube is not None else None

    # ── ANTENNA subtable ───────────────────────────────────────────────
    with _Table(str(ms_path / "ANTENNA"), readonly=False) as t:
        n_ant = tel.Nants
        ant_numbers = getattr(tel, "antenna_numbers", None)
        if ant_numbers is None:
            ant_numbers = np.arange(n_ant, dtype=np.int32)
        else:
            ant_numbers = np.asarray(ant_numbers[:n_ant], dtype=np.int32)
        while t.nrows() < n_ant:
            t.addrows(1)
        if t.nrows() > n_ant:
            t.removerows(list(range(n_ant, t.nrows())))

        names = list(tel.antenna_names) if tel.antenna_names is not None else [f"ANT{i}" for i in range(n_ant)]
        t.putcol("NAME", names[:n_ant])
        t.putcol("STATION", names[:n_ant])

        positions = tel.antenna_positions  # (Nants, 3) in ECEF metres
        if positions is not None and positions.shape[0] >= n_ant:
            t.putcol("POSITION", positions[:n_ant])  # wrapper handles transpose
        diameters = tel.antenna_diameters
        if diameters is not None and len(diameters) >= n_ant:
            t.putcol("DISH_DIAMETER", np.asarray(diameters[:n_ant], dtype=np.float64))
        else:
            t.putcol("DISH_DIAMETER", np.full(n_ant, 4.65))
        t.putcol("FLAG_ROW", np.zeros(n_ant, dtype=bool))
        t.putcol("TYPE", ["GROUND-BASED"] * n_ant)
        t.putcol("MOUNT", ["ALT-AZ"] * n_ant)

    ant_row_by_number = {int(num): idx for idx, num in enumerate(ant_numbers.tolist())}
    ant1 = np.asarray(uv.ant_1_array, dtype=np.int64)
    ant2 = np.asarray(uv.ant_2_array, dtype=np.int64)
    try:
        ant1_rows = np.array([ant_row_by_number[int(a)] for a in ant1], dtype=np.int32)
        ant2_rows = np.array([ant_row_by_number[int(a)] for a in ant2], dtype=np.int32)
    except KeyError as exc:
        raise ValueError(f"Antenna number {int(exc.args[0])} missing from ANTENNA table map") from exc

    # ── FIELD subtable ─────────────────────────────────────────────────
    with _Table(str(ms_path / "FIELD"), readonly=False) as t:
        while t.nrows() < 1:
            t.addrows(1)
        pc_id = int(uv.phase_center_id_array[0])
        pc_cat = uv.phase_center_catalog[pc_id]
        phase_ra = float(pc_cat["cat_lon"])
        phase_dec = float(pc_cat["cat_lat"])
        phase_dir = np.array([[[phase_ra, phase_dec]]])  # (1, 1, 2) rows-first
        t.putcol("PHASE_DIR", phase_dir)
        t.putcol("DELAY_DIR", phase_dir)
        t.putcol("REFERENCE_DIR", phase_dir)
        t.putcol("NAME", ["FIELD_0"])

    # ── SPECTRAL_WINDOW subtable ───────────────────────────────────────
    with _Table(str(ms_path / "SPECTRAL_WINDOW"), readonly=False) as t:
        while t.nrows() < 1:
            t.addrows(1)
        chan_width = float(np.median(np.diff(freqs))) if n_chan > 1 else 1e6
        t.putcell("NUM_CHAN", 0, n_chan)
        t.putcell("CHAN_FREQ", 0, freqs.astype(np.float64))
        t.putcell("CHAN_WIDTH", 0, np.full(n_chan, chan_width, dtype=np.float64))
        t.putcell("EFFECTIVE_BW", 0, np.full(n_chan, abs(chan_width), dtype=np.float64))
        t.putcell("RESOLUTION", 0, np.full(n_chan, abs(chan_width), dtype=np.float64))
        t.putcell("REF_FREQUENCY", 0, float(freqs[n_chan // 2]))
        t.putcell("TOTAL_BANDWIDTH", 0, float(abs(chan_width) * n_chan))
        t.putcell("MEAS_FREQ_REF", 0, 5)  # TOPO

    # ── POLARIZATION subtable ──────────────────────────────────────────
    with _Table(str(ms_path / "POLARIZATION"), readonly=False) as t:
        while t.nrows() < 1:
            t.addrows(1)
        try:
            corr_types = np.array(
                [_AIPS_TO_CASA_CORR[int(p)] for p in uv.polarization_array],
                dtype=np.int32,
            )
        except KeyError as exc:
            raise ValueError(f"Unsupported polarization code: {int(exc.args[0])}") from exc
        corr_product = np.array(
            [_CASA_CORR_TO_PRODUCT[int(c)] for c in corr_types], dtype=np.int32
        ).T
        t.putcell("NUM_CORR", 0, n_pol)
        t.putcell("CORR_TYPE", 0, corr_types)
        t.putcell("CORR_PRODUCT", 0, corr_product)

    # ── DATA_DESCRIPTION subtable ──────────────────────────────────────
    with _Table(str(ms_path / "DATA_DESCRIPTION"), readonly=False) as t:
        while t.nrows() < 1:
            t.addrows(1)
        t.putcell("SPECTRAL_WINDOW_ID", 0, 0)
        t.putcell("POLARIZATION_ID", 0, 0)

    # ── OBSERVATION subtable ───────────────────────────────────────────
    with _Table(str(ms_path / "OBSERVATION"), readonly=False) as t:
        while t.nrows() < 1:
            t.addrows(1)
        t.putcol("TELESCOPE_NAME", ["DSA_110"])

    # ── Main table ─────────────────────────────────────────────────────
    with _Table(str(ms_path), readonly=False) as t:
        while t.nrows() < n_rows:
            t.addrows(min(n_rows - t.nrows(), 10000))

        # TIME — pyuvdata stores as JD, CASA needs MJD seconds
        time_jd = uv.time_array
        time_mjd_s = (time_jd - 2400000.5) * 86400.0
        t.putcol("TIME", time_mjd_s)
        t.putcol("TIME_CENTROID", time_mjd_s)

        # Interval / exposure
        if uv.integration_time is None:
            int_time = np.full(n_rows, 12.885, dtype=np.float64)
        else:
            int_time = np.asarray(uv.integration_time, dtype=np.float64)
            if int_time.ndim == 0:
                int_time = np.full(n_rows, float(int_time), dtype=np.float64)
            else:
                int_time = np.ravel(int_time)
                if int_time.size != n_rows:
                    raise ValueError(
                        f"integration_time has {int_time.size} values, expected {n_rows}"
                    )
        t.putcol("INTERVAL", int_time)
        t.putcol("EXPOSURE", int_time)

        # Map pyuvdata antenna numbers to ANTENNA table row indices.
        t.putcol("ANTENNA1", ant1_rows)
        t.putcol("ANTENNA2", ant2_rows)

        # Preserve prior pyuvdata.write_ms() sign convention used by simulation pipeline.
        t.putcol("UVW", -np.asarray(uv.uvw_array, dtype=np.float64))

        # DATA — rows-first convention: (Nblts, Nfreqs, Npols)
        t.putcol("DATA", data_for_putcol)

        # FLAG
        if flag_for_putcol is not None:
            t.putcol("FLAG", flag_for_putcol)
        else:
            t.putcol("FLAG", np.zeros((n_rows, n_pol, n_chan), dtype=bool))

        # Other required columns
        t.putcol("FLAG_ROW", np.zeros(n_rows, dtype=bool))
        t.putcol("DATA_DESC_ID", np.zeros(n_rows, dtype=np.int32))
        t.putcol("FIELD_ID", np.zeros(n_rows, dtype=np.int32))
        t.putcol("SCAN_NUMBER", np.ones(n_rows, dtype=np.int32))

        # SIGMA and WEIGHT — rows-first: (Nblts, Npols)
        t.putcol("SIGMA", np.ones((n_rows, n_pol), dtype=np.float32))
        t.putcol("WEIGHT", np.ones((n_rows, n_pol), dtype=np.float32))

    _log.info("Wrote MS with %d rows, %d channels, %d pols to %s",
              n_rows, n_chan, n_pol, ms_path)
    return ms_path
