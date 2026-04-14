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

import casatools
import numpy as np

from dsa110_continuum.adapters.casa_tables import table as _Table

__all__ = ["uvdata_to_ms"]

_log = logging.getLogger(__name__)


def uvdata_to_ms(uv: "pyuvdata.UVData", ms_path: str | Path) -> Path:
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

    sim = casatools.simulator()
    sim.open(str(ms_path))
    sim.close()

    tel = uv.telescope

    # ── ANTENNA subtable ───────────────────────────────────────────────
    with _Table(str(ms_path / "ANTENNA"), readonly=False) as t:
        n_ant = tel.Nants
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
        freqs = uv.freq_array  # (Nfreqs,)
        n_chan = len(freqs)
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
        n_pol = uv.Npols
        pol_map = {-5: 9, -6: 10, -7: 11, -8: 12, 1: 5, 2: 6, 3: 7, 4: 8}
        corr_types = np.array([pol_map.get(int(p), 9) for p in uv.polarization_array], dtype=np.int32)
        t.putcell("NUM_CORR", 0, n_pol)
        t.putcell("CORR_TYPE", 0, corr_types)
        t.putcell("CORR_PRODUCT", 0, np.array([[0, 1, 0, 1], [0, 0, 1, 1]], dtype=np.int32)[:, :n_pol])

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
        n_rows = uv.Nblts
        n_chan = len(uv.freq_array)
        n_pol = uv.Npols
        while t.nrows() < n_rows:
            t.addrows(min(n_rows - t.nrows(), 10000))

        # TIME — pyuvdata stores as JD, CASA needs MJD seconds
        time_jd = uv.time_array
        time_mjd_s = (time_jd - 2400000.5) * 86400.0
        t.putcol("TIME", time_mjd_s)
        t.putcol("TIME_CENTROID", time_mjd_s)

        # Interval / exposure
        int_time = uv.integration_time if uv.integration_time is not None else np.full(n_rows, 12.885)
        t.putcol("INTERVAL", int_time)
        t.putcol("EXPOSURE", int_time)

        # Antenna indices (pyuvdata is 0-based)
        t.putcol("ANTENNA1", uv.ant_1_array.astype(np.int32))
        t.putcol("ANTENNA2", uv.ant_2_array.astype(np.int32))

        # UVW — rows-first convention: (Nblts, 3)
        t.putcol("UVW", uv.uvw_array)

        # DATA — rows-first convention: (Nblts, Nfreqs, Npols)
        t.putcol("DATA", uv.data_array.astype(np.complex64))

        # FLAG
        if uv.flag_array is not None:
            t.putcol("FLAG", uv.flag_array)
        else:
            t.putcol("FLAG", np.zeros((n_rows, n_chan, n_pol), dtype=bool))

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
