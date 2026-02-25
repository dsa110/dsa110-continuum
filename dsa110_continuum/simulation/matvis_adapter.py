"""Adapter for matvis visibility simulator (OOM speedup over pyuvsim).

matvis uses primary beam interpolation and vectorized outer products to achieve
order-of-magnitude faster simulation than direct RIME summation. Use with
--simulator matvis when runtime is critical and unpolarized Stokes I is acceptable.

References
----------
- matvis: Kittiwisit et al., arXiv:2312.09763
- pip install matvis
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
from astropy.time import Time
from pyuvdata import UVData

from dsa110_contimg.core.simulation.pyuvsim_adapter import create_dsa110_beam, sources_to_skymodel

logger = logging.getLogger(__name__)


def check_matvis_available() -> bool:
    """Return True if matvis is installed and importable."""
    try:
        from matvis import simulate_vis  # noqa: F401
        return True
    except ImportError:
        return False


def _uvdata_to_matvis_inputs(uvdata: UVData):
    """Build matvis.simulate_vis() inputs from UVData.

    Returns
    -------
    ants : dict[int, np.ndarray]
        Antenna index -> (x,y,z) in meters relative to array center.
    freqs : np.ndarray
        Frequency channels in Hz.
    times : astropy.time.Time
        Unique observation times.
    antpairs : list[tuple[int, int]]
        Baseline pairs in blt order (first Nbls).
    telescope_loc : EarthLocation
    """
    # Antenna positions relative to array center (matvis convention)
    ant_pos = np.asarray(uvdata.antenna_positions, dtype=float)
    center = np.mean(ant_pos, axis=0)
    ant_pos_relative = ant_pos - center
    ants = {int(uvdata.antenna_numbers[i]): ant_pos_relative[i] for i in range(uvdata.Nants_telescope)}

    # Unique times (JD) -> astropy Time
    time_array = np.asarray(uvdata.time_array).reshape(-1)
    unique_times_jd = np.unique(time_array)
    times = Time(unique_times_jd, format="jd", scale="utc")

    # Frequencies in Hz (support both 1D and 2D freq_array)
    freq_array = np.asarray(uvdata.freq_array).reshape(-1)
    freqs = np.array(freq_array, dtype=float)

    # Baseline pairs in same order as UVData blt (time, baseline): first Nbls
    nbls = uvdata.Nbls
    ant_1 = uvdata.ant_1_array[:nbls]
    ant_2 = uvdata.ant_2_array[:nbls]
    # matvis expects numpy array for antpairs to support slicing
    antpairs = np.vstack((ant_1, ant_2)).T.astype(int)

    # Telescope location
    telescope_loc = uvdata.telescope.location
    if telescope_loc is None:
        raise ValueError("UVData has no telescope.location; required for matvis.")

    return ants, freqs, times, antpairs, telescope_loc


def simulate_visibilities_matvis(
    uvdata: UVData,
    sources: Sequence[Any],
    beam_type: str = "airy",
    quiet: bool = True,
    sky_model: Any = None,
    use_gpu: bool = False,
) -> UVData:
    """Simulate visibilities using matvis (matrix-based, OOM faster than pyuvsim).

    Parameters
    ----------
    uvdata : UVData
        UVData with antenna_positions, time_array, freq_array set. data_array
        will be overwritten with simulated visibilities.
    sources : sequence of SyntheticSource
        Point sources; ignored if sky_model is provided.
    beam_type : str
        "airy" or "gaussian" (passed to create_dsa110_beam).
    quiet : bool
        Suppress matvis logging.
    sky_model : SkyModel, optional
        Pre-computed SkyModel; if provided, sources is ignored.
    use_gpu : bool
        Use matvis GPU backend if available.

    Returns
    -------
    UVData
        uvdata with data_array filled. Polarization: matvis returns unpolarized;
        XX and YY are set to the same value (Stokes I approximation).
    """
    from matvis import simulate_vis

    if sky_model is None:
        sky_model = sources_to_skymodel(sources)
    if sky_model.Ncomponents == 0:
        nblts = uvdata.Nblts
        nfreqs = uvdata.Nfreqs
        npols = uvdata.Npols
        uvdata.data_array = np.zeros((nblts, nfreqs, npols), dtype=np.complex128)
        return uvdata

    # Build flux matrix (Nsrcs, Nfreqs) in Jy from SkyModel
    freqs_1d = np.asarray(uvdata.freq_array).reshape(-1)
    n_freqs = len(freqs_1d)
    
    # Stokes I: (4, Nfreqs, Ncomponents) or (4, 1, Ncomponents)
    stokes = np.asarray(sky_model.stokes)
    if stokes.shape[1] == 1:
        stokes_i = np.asarray(sky_model.stokes[0, 0, :])
    else:
        stokes_i = np.asarray(sky_model.stokes[0, :, :]).mean(axis=0)  # mean over freq

    fluxes = np.zeros((sky_model.Ncomponents, n_freqs), dtype=np.float64)

    if sky_model.spectral_type == "flat":
        # Constant flux across frequency
        fluxes = np.tile(stokes_i[:, np.newaxis], (1, n_freqs))
    else:
        # Power-law spectrum
        ref_freqs = np.atleast_1d(sky_model.reference_frequency.to_value("Hz"))
        if ref_freqs.size == 1:
            ref_freqs = np.full(sky_model.Ncomponents, ref_freqs[0])
        spectral_index = (
            np.atleast_1d(getattr(sky_model, "spectral_index", np.zeros(sky_model.Ncomponents)))
            if hasattr(sky_model, "spectral_index")
            else np.zeros(sky_model.Ncomponents)
        )
        if np.isscalar(spectral_index):
            spectral_index = np.full(sky_model.Ncomponents, spectral_index)
        
        for s in range(sky_model.Ncomponents):
            fluxes[s, :] = float(stokes_i[s]) * (freqs_1d / ref_freqs[s]) ** spectral_index[s]

    # RA/Dec in radians (matvis: ra [0, 2pi], dec [-pi/2, pi/2])
    ra_rad = np.asarray(sky_model.ra.to_value("rad")).reshape(-1)
    dec_rad = np.asarray(sky_model.dec.to_value("rad")).reshape(-1)

    ants, freqs, times, antpairs, telescope_loc = _uvdata_to_matvis_inputs(uvdata)
    beam = create_dsa110_beam(beam_type=beam_type)
    beams = [beam]

    if not quiet:
        logger.info(
            "Running matvis: %d sources, %d baselines, %d times, %d freqs",
            sky_model.Ncomponents,
            len(antpairs),
            len(times),
            len(freqs),
        )

    vis = simulate_vis(
        ants=ants,
        fluxes=fluxes,
        ra=ra_rad,
        dec=dec_rad,
        freqs=freqs,
        times=times,
        beams=beams,
        telescope_loc=telescope_loc,
        polarized=False,
        precision=2,
        use_feed="x",
        use_gpu=use_gpu,
        antpairs=antpairs,
    )
    # vis shape: (NFREQS, NTIMES, NBLS)
    nbls = len(antpairs)
    ntimes = len(times)
    nfreqs = len(freqs)
    npols = uvdata.Npols
    nblts = ntimes * nbls

    # Map to UVData (blt_order = time, baseline): blt_idx = time_idx * nbls + bl_idx
    data = np.zeros((nblts, nfreqs, npols), dtype=np.complex128)
    for ti in range(ntimes):
        for bl in range(nbls):
            blt_idx = ti * nbls + bl
            data[blt_idx, :, :] = vis[:, ti, bl, np.newaxis]  # same value for all pols
    uvdata.data_array = data
    if not quiet:
        logger.info("matvis simulation complete")
    return uvdata
