#!/usr/bin/env python
# pylint: disable=no-member  # astropy.units uses dynamic attributes (deg, arcsec, m, etc.)
# pylint: disable=consider-using-sys-exit  # No explicit sys.exit calls - warning appears to be false positive
"""Generate synthetic DSA-110 UVH5 subband files for end-to-end testing."""

import argparse
import concurrent.futures
import functools
import json
import logging
import os
import random
import sys
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import astropy.units as u  # pylint: disable=no-member
import numpy as np
import yaml
from astropy.coordinates import EarthLocation
from astropy.time import Time
import astropy.constants as const
from pyuvdata import UVData

from dsa110_contimg.core.simulation.source_selection import (
    CatalogRegion,
    SourceSelector,
    SyntheticSource,
    summarize_sources,
)
from dsa110_contimg.common.utils.antpos_local import get_itrf
from dsa110_contimg.common.utils.constants import DSA110_ALT, DSA110_LAT, DSA110_LON
from dsa110_contimg.common.utils.paths import get_repo_root

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = get_repo_root()
if (REPO_ROOT / "backend" / "src" / "dsa110_contimg").exists():
    REPO_ROOT = REPO_ROOT / "backend"
CONFIG_DIR = PACKAGE_ROOT / "config"
PYUVSIM_DIR = PACKAGE_ROOT / "pyuvsim"
SECONDS_PER_DAY = 86400.0


@dataclass
class TelescopeConfig:
    layout_csv: Path
    polarizations: list[int]
    num_subbands: int
    channels_per_subband: int
    channel_width_hz: float
    freq_min_hz: float
    freq_max_hz: float
    reference_frequency_hz: float
    integration_time_sec: float
    total_duration_sec: float
    site_location: EarthLocation
    phase_ra: u.Quantity
    phase_dec: u.Quantity
    extra_keywords: dict[str, str]
    freq_template: np.ndarray = field(default_factory=lambda: np.array([]))
    freq_order: str = "desc"


def load_reference_layout(path: Path, columns: list[str] | None = None) -> dict[str, Any]:
    """Load reference layout from Parquet format.

    Parameters
    ----------
    path : Path
        Path to Parquet reference layout file
    columns : list[str], optional
        Specific columns to load for selective reading

    Returns
    -------
    dict
        Reference layout dictionary
    """
    from dsa110_contimg.core.simulation.config.parquet_io import load_reference_layout_parquet
    return load_reference_layout_parquet(path, columns=columns)


def load_telescope_config(config_path: Path, layout_meta: dict, freq_order: str) -> TelescopeConfig:
    from dsa110_contimg.common.utils.yaml_loader import load_yaml_with_env

    raw = load_yaml_with_env(config_path, expand_vars=True)

    site = raw["site"]
    layout = raw["layout"]
    spectral = raw["spectral"]
    temporal = raw["temporal"]
    phase_center = raw["phase_center"]

    location = EarthLocation.from_geodetic(
        lon=site["longitude_deg"] * u.deg,
        lat=site["latitude_deg"] * u.deg,
        height=site["altitude_m"] * u.m,
    )

    polarizations = [int(pol) for pol in layout["polarization_array"]]
    extra_keywords = layout_meta.get("extra_keywords", {})

    # Calculate frequency limits from spectral configuration
    # CRITICAL: Do NOT use layout_meta["freq_array_hz"] - it contains wrong channel count
    # Real DSA-110 has 48 channels per subband × 16 subbands = 768 total channels
    # Using telescope.yaml values only ensures correct structure
    freq_min = spectral.get("freq_min_hz")
    freq_max = spectral.get("freq_max_hz")
    freq_template = np.array([])  # Empty - generate from config only
    if freq_min is None or freq_max is None:
        raise ValueError("Unable to derive frequency bounds from configuration or layout metadata")

    norm_freq_order = freq_order.lower()
    if norm_freq_order not in {"asc", "desc"}:
        raise ValueError(f"Unsupported frequency order '{freq_order}'")
    if freq_template.size > 0 and norm_freq_order == "asc" and freq_template[0] > freq_template[-1]:
        freq_template = freq_template[::-1]
    if (
        freq_template.size > 0
        and norm_freq_order == "desc"
        and freq_template[0] < freq_template[-1]
    ):
        freq_template = freq_template[::-1]

    return TelescopeConfig(
        layout_csv=config_path.parent / layout["csv"],
        polarizations=polarizations,
        num_subbands=int(spectral["num_subbands"]),
        channels_per_subband=int(spectral["channels_per_subband"]),
        channel_width_hz=float(spectral["channel_width_hz"]),
        freq_min_hz=float(freq_min),
        freq_max_hz=float(freq_max),
        reference_frequency_hz=float(spectral["reference_frequency_hz"]),
        integration_time_sec=float(temporal["integration_time_sec"]),
        total_duration_sec=float(temporal["total_duration_sec"]),
        site_location=location,
        phase_ra=float(phase_center["ra_deg"]) * u.deg,
        phase_dec=float(phase_center["dec_deg"]) * u.deg,
        extra_keywords=extra_keywords,
        freq_template=freq_template,
        freq_order=norm_freq_order,
    )


def build_time_arrays(config: TelescopeConfig, nbls: int, ntimes: int, start_time: Time):
    """Build time arrays for synthetic data generation.

    Parameters
    ----------
    config :
        Telescope configuration
    nbls : int
        Number of baselines
    ntimes : int
        Number of time integrations
    start_time :
        Observation start time

    Returns
    -------
    tuple
        Tuple of (unique_times_jd, time_array, integration_time)

    Notes
    -----
    Times are returned in Julian Date (JD) format as required by pyuvdata.
    LST is not computed here - use UVData.set_lsts_from_time_array()
    after setting up the telescope to ensure self-consistency.
    """
    dt_days = config.integration_time_sec / SECONDS_PER_DAY

    # Use JD (not MJD) as required by pyuvdata's time_array
    unique_times = start_time.jd + dt_days * np.arange(ntimes)
    time_array = np.repeat(unique_times, nbls)
    integration_time = np.full(time_array.shape, config.integration_time_sec, dtype=float)

    return unique_times, time_array, integration_time


def build_uvw(
    config: TelescopeConfig,
    unique_times_jd: np.ndarray,
    ant1_array: np.ndarray,
    ant2_array: np.ndarray,
    nants_telescope: int,
) -> np.ndarray:
    """Build UVW array for synthetic data generation.

    Parameters
    ----------
    config :
        Telescope configuration
    unique_times_jd :
        Array of unique times in Julian Date (JD)
    ant1_array :
        Antenna 1 array for baselines
    ant2_array :
        Antenna 2 array for baselines
    nants_telescope :
        Number of antennas

    Returns
    -------
        UVW array with shape (nbls * ntimes, 3)

    """
    from dsa110_contimg.core.simulation.fast_uvw import fast_build_uvw_grid

    nbls = len(ant1_array)

    # Convert JD to MJD for fast_build_uvw_grid
    # Convert JD to MJD using astropy Time (handling array input)
    # unique_times_jd is a float array, so we can use it directly
    unique_times_mjd = Time(unique_times_jd, format="jd").mjd

    # Get antenna ITRF offsets
    ant_df = get_itrf(latlon_center=(DSA110_LAT * u.rad, DSA110_LON * u.rad, DSA110_ALT * u.m))
    ant_offsets = {}
    missing = []
    for ant in range(nants_telescope):
        station = ant + 1
        if station in ant_df.index:
            row = ant_df.loc[station]
            ant_offsets[ant] = np.array([row["dx_m"], row["dy_m"], row["dz_m"]], dtype=float)
        else:
            missing.append(station)
    if missing:
        raise ValueError(f"Missing antenna offsets for stations: {missing}")

    # Compute baseline vectors (ant2 - ant1)
    blen = np.zeros((nbls, 3))
    for idx, (a1, a2) in enumerate(zip(ant1_array, ant2_array)):
        blen[idx] = ant_offsets[int(a2)] - ant_offsets[int(a1)]

    # Use fast astropy-based UVW computation (100x faster than CASA)
    uvw = fast_build_uvw_grid(
        baseline_vectors_itrf=blen,
        times_mjd=unique_times_mjd,
        phase_center_ra_deg=config.phase_ra.to_value(u.deg),
        phase_center_dec_deg=config.phase_dec.to_value(u.deg),
    )
    return uvw




def build_uvdata_from_scratch(
    config: TelescopeConfig,
    nants: int = 110,
    ntimes: int = 30,
    start_time: Time = None,
    *,
    allocate_data_arrays: bool = True,
    include_uvw: bool = True,
) -> UVData:
    """Build a minimal UVData object from scratch without requiring a template.

    This function creates a UVData object with realistic DSA-110 structure
    using only configuration files and antenna position data.

    Parameters
    ----------
    config :
        Telescope configuration
    nants :
        Number of antennas (default: 110 for DSA-110)
    ntimes :
        Number of time integrations (default: 30 for ~5 minutes)
    start_time :
        Observation start time (default: current time)

    Returns
    -------
        UVData object with basic structure populated

    """
    if start_time is None:
        start_time = Time.now()

    # Get antenna positions
    ant_df = get_itrf(latlon_center=(DSA110_LAT * u.rad, DSA110_LON * u.rad, DSA110_ALT * u.m))

    # Select antennas (use first nants stations)
    available_stations = sorted(ant_df.index)[:nants]
    ant_offsets = {}
    for station in available_stations:
        ant_idx = station - 1  # Convert station number to antenna index
        row = ant_df.loc[station]
        ant_offsets[ant_idx] = np.array([row["dx_m"], row["dy_m"], row["dz_m"]], dtype=float)

    # Create baseline pairs
    ant_indices = sorted(ant_offsets.keys())
    baselines = [(i, j) for i in ant_indices for j in ant_indices if i < j]
    nbls = len(baselines)

    # Build antenna arrays
    ant1_list = [b[0] for b in baselines]
    ant2_list = [b[1] for b in baselines]

    # Build time arrays (LST computed later via set_lsts_from_time_array)
    unique_times, time_array, integration_time = build_time_arrays(config, nbls, ntimes, start_time)

    uvw_array: np.ndarray | None
    if include_uvw:
        # Build UVW array
        uvw_array = build_uvw(
            config,
            unique_times,
            np.array(ant1_list),
            np.array(ant2_list),
            nants,
        )
    else:
        # Placeholder to avoid allocating/copying a huge (Nblts, 3) array when
        # the caller will overwrite uv.uvw_array (e.g., synthetic dataset generator).
        uvw_array = None

    # Create UVData object
    uv = UVData()

    # Calculate dimensions - in pyuvdata 3.x these are computed properties
    # so we need to calculate them before creating arrays
    nblts = nbls * ntimes
    nspws = 1
    nfreqs = config.channels_per_subband * config.num_subbands  # Full frequency range (all 16 SBs)
    npols = len(config.polarizations)

    # Set basic dimensions that can still be set
    uv.Nspws = nspws

    # Set antenna arrays (these define Nants_data and Nants_telescope)
    # Use tile() for Time-major ordering (Baseline varies fast, Time varies slow)
    # This matches build_time_arrays() which uses repeat() for time (Time varies slow)
    uv.ant_1_array = np.tile(ant1_list, ntimes)
    uv.ant_2_array = np.tile(ant2_list, ntimes)
    uv.antenna_numbers = np.array(ant_indices, dtype=int)
    uv.antenna_names = [str(i) for i in ant_indices]
    uv.antenna_positions = np.array(
        [ant_offsets[idx] for idx in ant_indices],
        dtype=float,
    )
    uv.antenna_diameters = np.full(len(ant_indices), 4.65, dtype=float)
    uv.Nants_data = len(ant_indices)

    # Set time arrays (lst_array computed after telescope setup)
    uv.time_array = time_array
    uv.integration_time = integration_time

    # Set frequency array (full band: all 16 subbands)
    # Generate frequencies spanning all subbands
    frequencies = []
    for sb_idx in range(config.num_subbands):
        sb_center_hz = config.freq_min_hz + (sb_idx + 0.5) * (
            config.channels_per_subband * config.channel_width_hz
        )
        sb_start_hz = sb_center_hz - (config.channels_per_subband / 2.0) * config.channel_width_hz
        sb_freqs = sb_start_hz + config.channel_width_hz * np.arange(config.channels_per_subband)
        frequencies.extend(sb_freqs)

    uv.freq_array = np.array(frequencies, dtype=float).reshape(1, -1)  # Shape: (1, nfreqs)
    uv.channel_width = np.full(nfreqs, config.channel_width_hz, dtype=float)
    uv.spw_array = np.array([0], dtype=int)  # Single spectral window
    uv.flex_spw_id_array = np.zeros(nfreqs, dtype=int)  # All channels in spw 0

    # Set phase center with pyuvdata 3.x phase_center_catalog
    # Note: info_source must be included for compatibility with pyuvdata's += operator
    phase_center_id = 0
    uv.phase_center_catalog = {
        phase_center_id: {
            "cat_name": "synthetic_calibrator",
            "cat_type": "sidereal",
            "cat_lon": config.phase_ra.to_value(u.rad),
            "cat_lat": config.phase_dec.to_value(u.rad),
            "cat_frame": "icrs",
            "cat_epoch": 2000.0,
            "info_source": None,  # Required for combining subbands
        }
    }
    uv.phase_center_id_array = np.full(nblts, phase_center_id, dtype=int)
    uv._Nphase.value = 1

    # Legacy phase center attributes (pyuvdata 2.x / UVH5 write compatibility)
    uv.phase_center_ra = config.phase_ra.to_value(u.rad)
    uv.phase_center_dec = config.phase_dec.to_value(u.rad)
    uv.phase_center_frame = "icrs"
    uv.phase_center_epoch = 2000.0

    # Apparent coordinates (same as catalog for ICRS at J2000)
    uv.phase_center_app_ra = np.full(nblts, config.phase_ra.to_value(u.rad), dtype=float)
    uv.phase_center_app_dec = np.full(nblts, config.phase_dec.to_value(u.rad), dtype=float)
    uv.phase_center_frame_pa = np.zeros(nblts, dtype=float)  # Position angle

    # Set baseline array
    uv.baseline_array = uv.antnums_to_baseline(uv.ant_1_array, uv.ant_2_array)

    # Set UVW (optional; can be overwritten later)
    if uvw_array is not None:
        uv.uvw_array = uvw_array

    # Set polarization
    uv.polarization_array = np.array(config.polarizations, dtype=int)

    # Set telescope metadata using the Telescope object (pyuvdata 3.x API)
    from pyuvdata import Telescope

    tel = Telescope()
    tel.name = "DSA-110"
    tel.instrument = "DSA-110"  # Required for pyuvdata 3.x UVH5 write
    tel.location = config.site_location
    tel.Nants = len(uv.antenna_numbers)
    tel.antenna_numbers = np.array(uv.antenna_numbers, dtype=int)
    tel.antenna_names = list(uv.antenna_names)
    if hasattr(uv, "antenna_positions") and uv.antenna_positions is not None:
        tel.antenna_positions = np.array(uv.antenna_positions, dtype=float)
    else:
        tel.antenna_positions = np.zeros((tel.Nants, 3), dtype=float)
    if hasattr(uv, "antenna_diameters") and uv.antenna_diameters is not None:
        tel.antenna_diameters = np.array(uv.antenna_diameters, dtype=float)
    uv.telescope = tel

    # Compute LST from time_array using pyuvdata's method for self-consistency
    # Filter ERFA "dubious year" warnings - these are prediction accuracy warnings
    # for dates beyond IERS bulletins, which is fine for synthetic test data
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="ERFA function.*dubious year")
        uv.set_lsts_from_time_array()

    if allocate_data_arrays:
        # Set data arrays using calculated dimensions
        uv.data_array = np.zeros((nblts, nfreqs, npols), dtype=np.complex64)
        uv.flag_array = np.zeros((nblts, nfreqs, npols), dtype=bool)
        uv.nsample_array = np.ones((nblts, nfreqs, npols), dtype=np.float32)

    # Set computed dimension properties explicitly for pyuvdata 3.x
    # These are computed properties backed by UVParameter objects
    uv._Nbls.value = nbls
    uv._Nblts.value = nblts
    uv._Nfreqs.value = nfreqs
    uv._Npols.value = npols
    uv._Ntimes.value = ntimes

    # Set units and metadata
    uv.vis_units = "Jy"
    uv.history = "Synthetic UVData created from scratch (template-free mode)"
    uv.object_name = "synthetic_calibrator"
    uv.extra_keywords = config.extra_keywords.copy()
    uv.extra_keywords["synthetic"] = True
    uv.extra_keywords["template_free"] = True

    # Set blt_order so UVH5 files record it; avoids "blt_order could not be identified"
    # warning when subbands are read and combined (time_array is time-major: repeat(times, nbls))
    uv.blt_order = ("time", "baseline")

    return uv


def write_subband_uvh5(
    subband_index: int,
    uv_template: UVData,
    config: TelescopeConfig,
    start_time: Time,
    times_jd: np.ndarray,
    integration_time: np.ndarray,
    uvw_array: np.ndarray,
    amplitude_jy: float,
    output_dir: Path,
    source_model: str = "point",
    source_size_arcsec: float | None = None,
    source_pa_deg: float = 0.0,
    add_noise: bool = False,
    system_temperature_k: float = 50.0,
    add_cal_errors: bool = False,
    gain_std: float = 0.1,
    phase_std_deg: float = 10.0,
    rng: np.random.Generator | None = None,
    sources: Sequence[SyntheticSource] | None = None,
    pyuvsim_beam_type: str = "airy",
    sky_model: Any | None = None,
    beam_list: Any | None = None,
    source_summary: dict | None = None,
    use_mpi: bool = False,
    mpi_rank: int = 0,
    simulator: str = "matvis",
    use_gpu: bool = False,
) -> Path:
    """Write a single subband UVH5 file.

    Parameters
    ----------
    subband_index : int
        Subband index (0-based)
    uv_template : UVData
        Template UVData object (or created from scratch)
    config : TelescopeConfig
        Telescope configuration
    start_time : Time
        Observation start time
    times_jd : np.ndarray
        Time array in Julian Date (JD) as required by pyuvdata
    integration_time : np.ndarray
        Integration time array in seconds
    uvw_array : np.ndarray
        UVW array
    amplitude_jy : float
        Source flux density in Jy
    output_dir : Path
        Output directory
    source_model : str, optional
        Source model type (default is "point")
    source_size_arcsec : float or None, optional
        Source size in arcseconds (default is None)
    source_pa_deg : float, optional
        Source position angle in degrees (default is 0.0)
    add_noise : bool, optional
        Whether to add noise (default is False)
    system_temperature_k : float, optional
        System temperature in Kelvin (default is 50.0)
    add_cal_errors : bool, optional
        Whether to add calibration errors (default is False)
    gain_std : float, optional
        Gain standard deviation (default is 0.1)
    phase_std_deg : float, optional
        Phase standard deviation in degrees (default is 10.0)
    rng : np.random.Generator or None, optional
        Random number generator (default is None)
    sources : sequence of SyntheticSource or None, optional
        Catalog-based sources that override single point-source model (default is None)
    pyuvsim_beam_type : str, optional
        Beam model type for pyuvsim: "airy" or "gaussian" (default is "airy")
    sky_model : SkyModel, optional
        Pre-computed SkyModel object.
    beam_list : BeamList, optional
        Pre-computed BeamList object.
    source_summary : dict or None, optional
        Pre-computed result of summarize_sources(sources). When provided, used for
        extra_keywords instead of calling summarize_sources again (avoids N_subband calls).
    use_mpi : bool, optional
        If True, pyuvsim will use MPI (all ranks must call this function collectively).
    mpi_rank : int, optional
        MPI rank (0 for root). When use_mpi is True, only rank 0 performs add_noise,
        add_cal_errors, and write_uvh5; other ranks return after simulate_visibilities.
    simulator : str, optional
        "matvis" (default) or "pyuvsim". matvis gives OOM speedup; unpolarized Stokes I.
    use_gpu : bool, optional
        Use GPU backend when simulator is "matvis" (requires matvis[gpu]).

    Returns
    -------
        Path
        Path to created UVH5 file

    Notes
    -----
        LST is computed internally via set_lsts_from_time_array() for self-consistency with the telescope location.
        
        Visibility calculation uses the pyuvsim library (Lanman et al., 2019)
        which provides validated, high-precision simulations. pyuvsim is required
        for all synthetic data generation.
    """
    uv = uv_template.copy()
    uv.history += f"\nSynthetic point-source dataset generated (subband {subband_index:02d})."

    # Mark as synthetic in extra_keywords
    uv.extra_keywords["synthetic"] = True
    uv.extra_keywords["synthetic_flux_jy"] = float(amplitude_jy)
    uv.extra_keywords["synthetic_source_model"] = source_model
    # Store source position for catalog generation
    uv.extra_keywords["synthetic_source_ra_deg"] = float(config.phase_ra.to_value(u.deg))
    uv.extra_keywords["synthetic_source_dec_deg"] = float(config.phase_dec.to_value(u.deg))
    if source_size_arcsec is not None:
        uv.extra_keywords["synthetic_source_size_arcsec"] = float(source_size_arcsec)
    if add_noise:
        uv.extra_keywords["synthetic_has_noise"] = True
        uv.extra_keywords["synthetic_system_temp_k"] = float(system_temperature_k)
    if add_cal_errors:
        uv.extra_keywords["synthetic_has_cal_errors"] = True
        uv.extra_keywords["synthetic_gain_std"] = float(gain_std)
        uv.extra_keywords["synthetic_phase_std_deg"] = float(phase_std_deg)
    if sources:
        if source_summary is not None:
            uv.extra_keywords["synthetic_source_count"] = source_summary["count"]
            uv.extra_keywords["synthetic_source_summary"] = json.dumps(source_summary)
        else:
            summary = summarize_sources(list(sources))
            uv.extra_keywords["synthetic_source_count"] = summary["count"]
            uv.extra_keywords["synthetic_source_summary"] = json.dumps(summary)

    delta_f = abs(config.channel_width_hz)
    nchan = config.channels_per_subband
    subband_width = nchan * delta_f  # Width of one subband in Hz
    total_bandwidth = config.num_subbands * subband_width  # Total correlator output bandwidth

    # Calculate frequency array for this subband
    # DSA-110 correlator outputs 16 subbands centered on reference frequency.
    # The freq_min/freq_max in config may represent the science band, but
    # the actual correlator output spans: ref_freq ± total_bandwidth/2
    #
    # For synthetic data, we center on reference_frequency:
    #   - sb00 = highest frequencies
    #   - sb15 = lowest frequencies
    center_freq = config.reference_frequency_hz
    band_top = center_freq + total_bandwidth / 2

    if config.freq_order == "desc":
        # Start at top of band, work down
        sb_start_freq = band_top - subband_index * subband_width
        # Channels descend within subband
        freqs = sb_start_freq - delta_f * np.arange(nchan)
        channel_width_signed = -delta_f
    else:
        # Start at bottom of band, work up
        band_bottom = center_freq - total_bandwidth / 2
        sb_start_freq = band_bottom + subband_index * subband_width
        # Channels ascend within subband
        freqs = sb_start_freq + delta_f * np.arange(nchan)
        channel_width_signed = delta_f

    # Force future_array_shapes=True for pyuvdata 3.x / pyuvsim compatibility
    # This requires freq_array to be 1D (Nfreqs,)
    if hasattr(uv, "future_array_shapes"):
        uv.future_array_shapes = True
    
    uv.freq_array = freqs
    uv.channel_width = np.full(nchan, channel_width_signed, dtype=float)
    uv._Nfreqs.value = nchan
    uv._Nspws.value = 1
    uv.spw_array = np.array([0], dtype=int)
    uv.flex_spw_id_array = np.zeros(nchan, dtype=int)
    uv.spw_array = np.array([0], dtype=int)
    uv.flex_spw_id_array = np.zeros(nchan, dtype=int)

    # Preserve template LSTs when the provided times match the template.
    # Computing LST for a full baseline-time array (Nblts) can be extremely slow
    # for large synthetic datasets (e.g., 60 minutes, 110 antennas).
    template_lst = getattr(uv, "lst_array", None)
    template_times = np.asarray(getattr(uv, "time_array", []))

    def _times_match_template() -> bool:
        if template_lst is None:
            return False
        if template_times.shape != np.asarray(times_jd).shape:
            return False
        if template_times.size == 0:
            return False
        if float(template_times[0]) != float(times_jd[0]) or float(template_times[-1]) != float(
            times_jd[-1]
        ):
            return False
        # Cheap spot-check to avoid an O(N) full compare.
        step = max(1, template_times.size // 10)
        # Use a small tolerance (1e-8 days ~= 0.86ms) to handle potential float precision loss
        # if the template was read from disk.
        return bool(
            np.allclose(template_times[::step], np.asarray(times_jd)[::step], rtol=0.0, atol=1e-8)
        )

    reuse_template_lst = _times_match_template()

    uv.time_array = times_jd
    uv.integration_time = integration_time
    uv.uvw_array = uvw_array

    if reuse_template_lst:
        uv.lst_array = template_lst
    else:
        # Let pyuvdata compute LST from time_array for self-consistency
        uv.set_lsts_from_time_array()

    # Calculate dimensions from arrays (pyuvdata 3.x compatibility)
    # Computed properties like Nblts, Nfreqs, Npols may be None
    nblts = len(times_jd)
    nspws = 1
    nfreqs = nchan
    npols = len(uv.polarization_array)
    nants = len(uv.antenna_numbers)

    # Calculate u, v coordinates in wavelengths for extended sources
    u_lambda = None
    v_lambda = None
    if source_model != "point" and source_size_arcsec is not None:
        # Extract u, v from uvw_array (w is third column)
        # uvw_array shape: (Nblts, 3) where columns are [u, v, w] in meters
        # Convert to wavelengths using mean frequency
        mean_freq_hz = np.mean(uv.freq_array)
        wavelength_m = const.c.value / mean_freq_hz
        u_lambda = uvw_array[:, 0] / wavelength_m
        v_lambda = uvw_array[:, 1] / wavelength_m

    freq_1d = np.asarray(uv.freq_array).reshape(-1)

    if simulator == "matvis":
        # matvis: matrix-based, OOM faster; unpolarized Stokes I (XX/YY set equal)
        from dsa110_contimg.core.simulation.matvis_adapter import (
            check_matvis_available,
            simulate_visibilities_matvis,
        )

        if not check_matvis_available():
            raise RuntimeError(
                "simulator=matvis requires matvis. Install with: pip install matvis"
            ) from None
        _sky = sky_model
        if _sky is None and not sources and amplitude_jy > 0:
            from dsa110_contimg.core.simulation.pyuvsim_adapter import sources_to_skymodel
            from dsa110_contimg.core.simulation.source_selection import SyntheticSource
            _sky = sources_to_skymodel([
                SyntheticSource(
                    source_id="synthetic_point_source",
                    ra_deg=config.phase_ra.to_value(u.deg),
                    dec_deg=config.phase_dec.to_value(u.deg),
                    flux_ref_jy=amplitude_jy,
                    reference_freq_hz=config.reference_frequency_hz,
                    spectral_index=0.0,
                )
            ])
        try:
            uv = simulate_visibilities_matvis(
                uv,
                list(sources) if sources else [],
                beam_type=pyuvsim_beam_type,
                quiet=True,
                sky_model=_sky,
                use_gpu=use_gpu,
            )
            if hasattr(uv_template, "phase_center_catalog"):
                uv.phase_center_catalog = uv_template.phase_center_catalog.copy()
        except Exception as e:
            raise RuntimeError(f"matvis simulation failed: {e}") from e
    elif sources:
        # Use pyuvsim for high-precision visibility simulation
        from dsa110_contimg.core.simulation.pyuvsim_adapter import (
            check_pyuvsim_available,
            simulate_visibilities,
        )

        if not check_pyuvsim_available():
            raise RuntimeError(
                "pyuvsim is required for synthetic data generation. "
                "Install with: pip install pyuvsim[sim]"
            ) from None

        # Pyuvsim analytic beams (Airy/Gaussian) compute the full Jones matrix (4 pols)
        # We must expand the UVData to 4 pols for simulation, then downselect to match DSA-110 2-pol format
        original_pols = uv.polarization_array.copy()
        
        # Standard pyuvdata pol integers: XX=-5, YY=-6, XY=-7, YX=-8
        # Ensure we have all 4 for the simulation
        four_pols = np.array([-5, -6, -7, -8], dtype=int)
        uv.polarization_array = four_pols
        uv._Npols.value = 4
        
        # Resize arrays for 4 polarizations
        uv.data_array = np.zeros((nblts, nfreqs, 4), dtype=uv.data_array.dtype)
        uv.flag_array = np.zeros((nblts, nfreqs, 4), dtype=uv.flag_array.dtype)
        uv.nsample_array = np.ones((nblts, nfreqs, 4), dtype=uv.nsample_array.dtype)

        try:
            uv = simulate_visibilities(
                uv, sources, beam_type=pyuvsim_beam_type, quiet=True, use_mpi=use_mpi
            )
            
            # Downselect back to original polarizations (XX, YY)
            # Use pyuvdata's select method if available, or manual slicing if simpler/faster
            # Since we know exactly what we want (first 2 pols if they are -5, -6), manual slicing is safer
            # assuming the order is preserved. But select is safer for correctness.
             
            # Verify original pols are a subset of simulated pols
            if not np.all(np.isin(original_pols, uv.polarization_array)):
                 raise RuntimeError(f"Simulated polarizations {uv.polarization_array} do not contain originals {original_pols}")

            # Keep only the requested polarizations
            # Note: pyuvdata.select is in-place
            uv.select(polarizations=original_pols)

            # Restore phase center catalog from template to ensure consistency across subbands
            # pyuvsim might add extra keys or reformat the catalog, causing concatenation issues
            # We assume the phase center used in simulation matches the template's phase center.
            # This ensures that all subbands have identical catalog metadata.
            if hasattr(uv_template, "phase_center_catalog"):
                uv.phase_center_catalog = uv_template.phase_center_catalog.copy()
           
        except ImportError as e:
            raise RuntimeError(
                "pyuvsim or its dependencies are not installed. "
                "Install with: pip install pyuvsim[sim]"
            ) from e
        except ValueError as e:
            raise RuntimeError(
                f"pyuvsim configuration error (check UVData dimensions and phase center): {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"pyuvsim simulation failed: {e}") from e
    else:
        # pyuvsim is required for all visibility simulation
        # Use pre-computed sky_model from main when provided; otherwise create point source once per subband
        if amplitude_jy > 0:
            from dsa110_contimg.core.simulation.pyuvsim_adapter import (
                check_pyuvsim_available,
                simulate_visibilities,
            )

            if not check_pyuvsim_available():
                raise RuntimeError(
                    "pyuvsim is required for synthetic data generation. "
                    "Install with: pip install pyuvsim[sim]"
                )

            # Expand to 4 pols for pyuvsim (same as multi-source path)
            original_pols = uv.polarization_array.copy()
            four_pols = np.array([-5, -6, -7, -8], dtype=int)
            uv.polarization_array = four_pols
            uv._Npols.value = 4
            uv.data_array = np.zeros((nblts, nfreqs, 4), dtype=uv.data_array.dtype)
            uv.flag_array = np.zeros((nblts, nfreqs, 4), dtype=uv.flag_array.dtype)
            uv.nsample_array = np.ones((nblts, nfreqs, 4), dtype=uv.nsample_array.dtype)

            if sky_model is not None:
                # Pre-computed in main; avoid creating SyntheticSource 16 times
                sources_for_sim: list = []
            else:
                from dsa110_contimg.core.simulation.source_selection import SyntheticSource

                sources_for_sim = [
                    SyntheticSource(
                        source_id="synthetic_point_source",
                        ra_deg=config.phase_ra.to_value(u.deg),
                        dec_deg=config.phase_dec.to_value(u.deg),
                        flux_ref_jy=amplitude_jy,
                        reference_freq_hz=config.reference_frequency_hz,
                        spectral_index=0.0,
                    )
                ]

            try:
                uv = simulate_visibilities(
                    uv,
                    sources_for_sim,
                    beam_type=pyuvsim_beam_type,
                    quiet=True,
                    sky_model=sky_model,
                    beam_list=beam_list,
                    use_mpi=use_mpi,
                )
                
                # Downselect back to original polarizations
                if not np.all(np.isin(original_pols, uv.polarization_array)):
                    raise RuntimeError(
                        f"Simulated polarizations {uv.polarization_array} "
                        f"do not contain originals {original_pols}"
                    )
                uv.select(polarizations=original_pols)
                
                # Restore phase center catalog
                if hasattr(uv_template, "phase_center_catalog"):
                    uv.phase_center_catalog = uv_template.phase_center_catalog.copy()
            except ImportError as e:
                raise RuntimeError(
                    "pyuvsim or its dependencies are not installed. "
                    "Install with: pip install pyuvsim[sim]"
                ) from e
            except ValueError as e:
                raise RuntimeError(
                    f"pyuvsim configuration error (check UVData dimensions and phase center): {e}"
                ) from e
            except Exception as e:
                raise RuntimeError(f"pyuvsim simulation failed: {e}") from e
        else:
            # No sources and no amplitude - return zeros
            uv.data_array = np.zeros((nblts, nfreqs, npols), dtype=np.complex128)
            uv.flag_array = np.zeros((nblts, nfreqs, npols), dtype=bool)
            uv.nsample_array = np.ones((nblts, nfreqs, npols), dtype=np.float32)

    # MPI: only rank 0 does noise, cal errors, and write; other ranks return after sim
    if use_mpi and mpi_rank != 0:
        anchor_str = start_time.strftime("%Y-%m-%dT%H:%M:%S")
        return output_dir / f"{anchor_str}_sb{subband_index:02d}.hdf5"

    # Add thermal noise if requested
    if add_noise:
        from dsa110_contimg.core.simulation.visibility_models import add_thermal_noise

        # Get integration time and channel width
        int_time = config.integration_time_sec
        chan_width = abs(config.channel_width_hz)

        # Get mean frequency for noise calculation (use center of frequency array)
        mean_freq_hz = (
            np.mean(uv.freq_array)
            if hasattr(uv, "freq_array") and uv.freq_array.size > 0
            else config.reference_frequency_hz
        )

        uv.data_array = add_thermal_noise(
            uv.data_array,
            int_time,
            chan_width,
            system_temperature_k=system_temperature_k,
            frequency_hz=mean_freq_hz,
            rng=rng,
        )

    # Add calibration errors if requested
    if add_cal_errors:
        from dsa110_contimg.core.simulation.visibility_models import (
            add_calibration_errors,
            apply_calibration_errors_to_visibilities,
        )

        _, complex_gains, _ = add_calibration_errors(
            uv.data_array,
            nants,  # Use calculated nants instead of uv.Nants_telescope
            gain_std=gain_std,
            phase_std_deg=phase_std_deg,
            rng=rng,
        )

        uv.data_array = apply_calibration_errors_to_visibilities(
            uv.data_array,
            uv.ant_1_array,
            uv.ant_2_array,
            complex_gains,
        )

    # Ensure flag and nsample arrays match data_array dimensions
    # (Already set in pyuvsim path; ensure consistency)
    if not hasattr(uv, 'flag_array') or uv.flag_array.shape != uv.data_array.shape:
        uv.flag_array = np.zeros_like(uv.data_array, dtype=bool)
    if not hasattr(uv, 'nsample_array') or uv.nsample_array.shape != uv.data_array.shape:
        uv.nsample_array = np.ones_like(uv.data_array, dtype=np.float32)

    uv.extra_keywords.update(config.extra_keywords)
    uv.extra_keywords["phase_center_dec"] = config.phase_dec.to_value(u.rad)
    uv.extra_keywords["ha_phase_center"] = 0.0
    uv.extra_keywords["phase_center_epoch"] = "HADEC"
    uv.extra_keywords["synthetic"] = True

    uv.phase_center_ra = config.phase_ra.to_value(u.rad)
    uv.phase_center_dec = config.phase_dec.to_value(u.rad)
    uv.phase_center_frame = "icrs"
    uv.phase_center_epoch = 2000.0

    # Ensure blt_order is set before write so combined subbands do not emit warnings
    uv.blt_order = ("time", "baseline")

    anchor_str = start_time.strftime("%Y-%m-%dT%H:%M:%S")
    filename = f"{anchor_str}_sb{subband_index:02d}.hdf5"
    output_path = output_dir / filename
    output_dir.mkdir(parents=True, exist_ok=True)
    uv.write_uvh5(output_path, run_check=False, clobber=True)

    return output_path


def _validate_outputs_parallel(
    paths: list[Path], max_workers: int
) -> tuple[bool, list[tuple[Path, bool, list[str]]]]:
    """Run validate_uvh5_file over paths in parallel. Returns (all_valid, [(path, is_valid, errors), ...])."""
    from dsa110_contimg.core.simulation.validate_synthetic import validate_uvh5_file

    if not paths:
        return True, []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(validate_uvh5_file, paths))
    all_valid = all(r[0] for r in results)
    return all_valid, list(zip(paths, [r[0] for r in results], [r[1] for r in results]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic DSA-110 UVH5 generator")
    parser.add_argument(
        "--nants",
        type=int,
        default=117,
        help="Number of antennas (only used in template-free mode, default: 117 for full array)",
    )
    parser.add_argument(
        "--ntimes",
        type=int,
        default=24,
        help="Number of time integrations (only used in template-free mode, default: 24)",
    )
    parser.add_argument(
        "--layout-meta",
        type=Path,
        default=CONFIG_DIR / "reference_layout.parquet",
        help="Parquet metadata produced by analyse_reference_uvh5.py",
    )
    parser.add_argument(
        "--telescope-config",
        type=Path,
        default=PYUVSIM_DIR / "telescope.yaml",
        help="Telescope configuration YAML",
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default="2025-01-01T00:00:00",
        help="Observation start time (UTC, ISO format)",
    )
    parser.add_argument(
        "--duration-minutes",
        type=float,
        default=5.0,
        help="Approximate observation duration in minutes",
    )
    parser.add_argument(
        "--flux-jy",
        type=float,
        default=25.0,
        help="Total Stokes I flux density of the calibrator",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("simulation/output"),
        help="Directory where UVH5 files will be written",
    )
    parser.add_argument(
        "--subbands",
        type=int,
        default=16,
        help="Number of subbands to synthesise",
    )
    parser.add_argument(
        "--freq-order",
        choices=["asc", "desc"],
        default="desc",
        help="Per-subband frequency ordering (default: desc)",
    )
    parser.add_argument(
        "--shuffle-subbands",
        action="store_true",
        help="Emit subband files in a shuffled order to exercise ingestion ordering",
    )
    parser.add_argument(
        "--source-model",
        choices=["point", "gaussian", "disk"],
        default="point",
        help="Source model type (default: point)",
    )
    parser.add_argument(
        "--source-size-arcsec",
        type=float,
        default=None,
        help="Source size in arcseconds (FWHM for Gaussian, radius for disk)",
    )
    parser.add_argument(
        "--source-pa-deg",
        type=float,
        default=0.0,
        help="Position angle in degrees for Gaussian sources (default: 0)",
    )
    parser.add_argument(
        "--add-noise",
        action="store_true",
        help="Add realistic thermal noise to visibilities",
    )
    parser.add_argument(
        "--system-temp-k",
        type=float,
        default=50.0,
        help="System temperature in Kelvin for noise calculation (default: 50K)",
    )
    parser.add_argument(
        "--add-cal-errors",
        action="store_true",
        help="Add realistic calibration errors (gain and phase)",
    )
    parser.add_argument(
        "--gain-std",
        type=float,
        default=0.1,
        help="Standard deviation of gain errors (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--phase-std-deg",
        type=float,
        default=10.0,
        help="Standard deviation of phase errors in degrees (default: 10 deg)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (for noise and cal errors)",
    )
    parser.add_argument(
        "--create-catalog",
        action="store_true",
        help="Create synthetic catalog database matching source positions",
    )
    parser.add_argument(
        "--catalog-type",
        type=str,
        choices=["nvss", "first", "rax", "vlass", "unicat"],
        default="unicat",
        help="Catalog type for synthetic catalog (default: unicat)",
    )
    parser.add_argument(
        "--catalog-output",
        type=Path,
        default=None,
        help="Output path for synthetic catalog database (auto-generated if not specified)",
    )
    parser.add_argument(
        "--source-catalog-type",
        type=str,
        choices=["nvss", "first", "rax", "vlass"],
        default=None,
        help="Use real catalog sources instead of a single synthetic point source",
    )
    parser.add_argument(
        "--source-catalog-path",
        type=Path,
        default=None,
        help="Explicit path to catalog (overrides env/auto detection)",
    )
    parser.add_argument(
        "--source-region-ra",
        type=float,
        default=None,
        help="RA center (deg) for catalog query; defaults to telescope phase center",
    )
    parser.add_argument(
        "--source-region-dec",
        type=float,
        default=None,
        help="Dec center (deg) for catalog query; defaults to telescope phase center",
    )
    parser.add_argument(
        "--source-region-radius-deg",
        type=float,
        default=1.0,
        help="Search radius in degrees when querying catalog sources",
    )
    parser.add_argument(
        "--min-source-flux-mjy",
        type=float,
        default=None,
        help="Minimum catalog flux density (mJy) to include in simulation",
    )
    parser.add_argument(
        "--max-source-count",
        type=int,
        default=64,
        help="Maximum number of catalog sources to include",
    )
    # Note: --use-pyuvsim flag removed - pyuvsim is now always required
    parser.add_argument(
        "--pyuvsim-beam-type",
        choices=["airy", "gaussian"],
        default="airy",
        help="Beam model type for pyuvsim simulation (default: airy). Use gaussian for ~2-5x speedup when beam accuracy is not critical.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max parallel workers for subband generation (default: min(16, cpu_count)). Set to 1 for sequential; use 16 on many-core machines for full parallelism.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable pyuvsim line_profiler (MPI: profiles one rank only). Call profiling.set_profiler() before run.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation of generated UVH5 files.",
    )
    parser.add_argument(
        "--show-env",
        action="store_true",
        help="Print NumPy/BLAS config and threading env vars then exit.",
    )
    parser.add_argument(
        "--simulator",
        choices=["pyuvsim", "matvis"],
        default="matvis",
        help="Visibility simulator: matvis (default, OOM faster, unpolarized) or pyuvsim (high precision).",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU when --simulator matvis (requires matvis[gpu]).",
    )
    return parser.parse_args()


def generate_subband_task(
    subband_index: int,
    uv_template: UVData,
    config: TelescopeConfig,
    start_time: Time,
    time_array: np.ndarray,
    integration_time: np.ndarray,
    uvw_array: np.ndarray,
    flux_jy: float,
    output_dir: Path,
    source_model: str,
    source_size_arcsec: float | None,
    source_pa_deg: float,
    add_noise: bool,
    system_temp_k: float,
    add_cal_errors: bool,
    gain_std: float,
    phase_std_deg: float,
    seed: int | None,
    sources: list[SyntheticSource] | None,
    pyuvsim_beam_type: str,
    sky_model: Any | None,
    beam_list: Any | None,
    source_summary: dict | None = None,
    simulator: str = "matvis",
    use_gpu: bool = False,
) -> Path:
    """Worker function for parallel subband generation."""
    # Re-initialize RNG inside process to ensure independence if seed is None,
    # or deterministic behavior if seed is provided (but varied by subband).
    # If we use the SAME seed for all subbands, noise will be identical, which is bad.
    # So we combine base seed with subband index.
    
    if seed is not None:
        subband_seed = seed + subband_index
        rng = np.random.default_rng(subband_seed)
    else:
        rng = np.random.default_rng()

    return write_subband_uvh5(
        subband_index=subband_index,
        uv_template=uv_template,
        config=config,
        start_time=start_time,
        times_jd=time_array,
        integration_time=integration_time,
        uvw_array=uvw_array,
        amplitude_jy=flux_jy,
        output_dir=output_dir,
        source_model=source_model,
        source_size_arcsec=source_size_arcsec,
        source_pa_deg=source_pa_deg,
        add_noise=add_noise,
        system_temperature_k=system_temp_k,
        add_cal_errors=add_cal_errors,
        gain_std=gain_std,
        phase_std_deg=phase_std_deg,
        rng=rng,
        sources=sources,
        pyuvsim_beam_type=pyuvsim_beam_type,
        sky_model=sky_model,
        beam_list=beam_list,
        source_summary=source_summary,
        simulator=simulator,
        use_gpu=use_gpu,
    )


def main() -> None:
    args = parse_args()

    if args.show_env:
        try:
            np.show_config()
        except Exception:
            print("numpy.show_config() not available")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            val = os.environ.get(name, "not set")
            print(f"{name}={val}")
        sys.exit(0)

    # Detect MPI (pyuvsim uses MPI when run with mpirun; all ranks must participate)
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        mpi_size = comm.Get_size()
        mpi_rank = comm.Get_rank()
        use_mpi_mode = mpi_size > 1
    except ImportError:
        use_mpi_mode = False
        mpi_rank = 0
        mpi_size = 1

    if not args.layout_meta.exists():
        raise FileNotFoundError(f"Layout metadata not found at {args.layout_meta}")
    if not args.telescope_config.exists():
        raise FileNotFoundError(f"Telescope configuration not found at {args.telescope_config}")

    layout_meta = load_reference_layout(args.layout_meta)
    config = load_telescope_config(args.telescope_config, layout_meta, args.freq_order)

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    if args.profile:
        try:
            from pyuvsim import profiling
            profiling.set_profiler()
            logger.info("pyuvsim line_profiler enabled (MPI: one rank only)")
        except ImportError:
            logger.warning("--profile requested but pyuvsim.profiling not available")

    if args.subbands != config.num_subbands:
        logger.warning(
            f"Requested {args.subbands} subbands,"
            f" but configuration expects {config.num_subbands}."
        )

    start_time = Time(args.start_time, format="isot", scale="utc")

    selected_sources: list[SyntheticSource] = []
    source_summary_for_tasks: dict | None = None
    if args.source_catalog_type:
        region_ra = (
            args.source_region_ra
            if args.source_region_ra is not None
            else float(config.phase_ra.to_value(u.deg))
        )
        region_dec = (
            args.source_region_dec
            if args.source_region_dec is not None
            else float(config.phase_dec.to_value(u.deg))
        )
        region = CatalogRegion(
            ra_deg=region_ra,
            dec_deg=region_dec,
            radius_deg=float(args.source_region_radius_deg),
        )
        selector = SourceSelector(
            region,
            args.source_catalog_type,
            catalog_path=args.source_catalog_path,
        )
        selected_sources = selector.select_sources(
            min_flux_mjy=args.min_source_flux_mjy,
            max_sources=args.max_source_count,
        )
        if not selected_sources:
            raise RuntimeError(
                "Catalog selection returned zero sources. "
                "Adjust --min-source-flux-mjy or radius and try again."
            )
        summary = summarize_sources(selected_sources)
        source_summary_for_tasks = summary
        logger.info(
            f"Using {summary['count']} catalog sources "
            f"(total flux {summary.get('total_flux_jy', 0):.2f} Jy)"
        )

    # Template-free mode (only supported mode)
    logger.info("Using template-free generation mode...")
    uv_template = build_uvdata_from_scratch(
        config, nants=args.nants, ntimes=args.ntimes, start_time=start_time
    )
    nbls = uv_template.Nbls
    ntimes = uv_template.Ntimes
    unique_times, time_array, integration_time = build_time_arrays(
        config, nbls, ntimes, start_time
    )
    uvw_array = build_uvw(
        config,
        unique_times,
        uv_template.ant_1_array[:nbls],
        uv_template.ant_2_array[:nbls],
        uv_template.Nants_telescope,
    )

    # Set up random number generator for reproducibility
    if args.seed is not None:
        rng = np.random.default_rng(args.seed)
    else:
        rng = np.random.default_rng()

    # Pre-compute SkyModel and BeamList for efficiency
    from dsa110_contimg.core.simulation.pyuvsim_adapter import (
        sources_to_skymodel,
        create_dsa110_beam,
    )
    from pyuvsim.telescope import BeamList

    sky_model = None
    if selected_sources:
        sky_model = sources_to_skymodel(selected_sources)
    elif args.flux_jy > 0:
        # Create single point source for SkyModel
        from dsa110_contimg.core.simulation.source_selection import SyntheticSource
        point_source = SyntheticSource(
            source_id="synthetic_point_source",
            ra_deg=config.phase_ra.to_value(u.deg),
            dec_deg=config.phase_dec.to_value(u.deg),
            flux_ref_jy=args.flux_jy,
            reference_freq_hz=config.reference_frequency_hz,
            spectral_index=0.0,
        )
        sky_model = sources_to_skymodel([point_source])
    
    # Create BeamList
    beam = create_dsa110_beam(beam_type=args.pyuvsim_beam_type)
    beam_list = BeamList([beam])

    outputs = []
    total_subbands = min(args.subbands, config.num_subbands)
    subband_indices = list(range(total_subbands))
    if args.shuffle_subbands:
        random.shuffle(subband_indices)

    t0 = time.perf_counter()

    if use_mpi_mode:
        # MPI path: all ranks run subbands sequentially; pyuvsim parallelizes within each subband
        if mpi_rank == 0:
            logger.info(
                "Using MPI with %d ranks (mpirun -n %d). Subbands run sequentially; pyuvsim parallelizes within each subband.",
                mpi_size,
                mpi_size,
            )
        for subband_index in subband_indices:
            subband_rng = (
                (np.random.default_rng(args.seed + subband_index) if args.seed is not None else np.random.default_rng())
                if mpi_rank == 0
                else None
            )
            path = write_subband_uvh5(
                subband_index=subband_index,
                uv_template=uv_template,
                config=config,
                start_time=start_time,
                times_jd=time_array,
                integration_time=integration_time,
                uvw_array=uvw_array,
                amplitude_jy=args.flux_jy,
                output_dir=args.output,
                source_model=args.source_model,
                source_size_arcsec=args.source_size_arcsec,
                source_pa_deg=args.source_pa_deg,
                add_noise=args.add_noise,
                system_temperature_k=args.system_temp_k,
                add_cal_errors=args.add_cal_errors,
                gain_std=args.gain_std,
                phase_std_deg=args.phase_std_deg,
                rng=subband_rng,
                sources=selected_sources if selected_sources else None,
                pyuvsim_beam_type=args.pyuvsim_beam_type,
                sky_model=sky_model,
                beam_list=beam_list,
                source_summary=source_summary_for_tasks,
                use_mpi=True,
                mpi_rank=mpi_rank,
                simulator=args.simulator,
                use_gpu=args.use_gpu,
            )
            if mpi_rank == 0:
                outputs.append(path)
    else:
        # ProcessPool path: subbands run in parallel across workers
        max_workers = args.max_workers if args.max_workers is not None else min(16, os.cpu_count() or 1)
        logger.info(
            "Starting parallel simulation for %d subbands with max_workers=%d",
            len(subband_indices),
            max_workers,
        )
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Create a partial function with all constant arguments
            task_func = functools.partial(
                generate_subband_task,
                uv_template=uv_template,
                config=config,
                start_time=start_time,
                time_array=time_array,
                integration_time=integration_time,
                uvw_array=uvw_array,
                flux_jy=args.flux_jy,
                output_dir=args.output,
                source_model=args.source_model,
                source_size_arcsec=args.source_size_arcsec,
                source_pa_deg=args.source_pa_deg,
                add_noise=args.add_noise,
                system_temp_k=args.system_temp_k,
                add_cal_errors=args.add_cal_errors,
                gain_std=args.gain_std,
                phase_std_deg=args.phase_std_deg,
                seed=args.seed,
                sources=selected_sources if selected_sources else None,
                pyuvsim_beam_type=args.pyuvsim_beam_type,
                sky_model=sky_model,
                beam_list=beam_list,
                source_summary=source_summary_for_tasks,
                simulator=args.simulator,
                use_gpu=args.use_gpu,
            )

            # Map task to subband indices
            try:
                results = executor.map(task_func, subband_indices)
                outputs = list(results)
            except Exception as e:
                logger.error(f"Parallel simulation failed: {e}")
                raise

    elapsed_s = time.perf_counter() - t0
    logger.info(
        "Parallel simulation complete: %d subbands in %.1f s (%.1f min)",
        len(outputs),
        elapsed_s,
        elapsed_s / 60.0,
    )
    logger.info("Generated synthetic subbands:")
    for path in outputs:
        logger.info(f"  {path}")

    # Create synthetic catalog if requested
    if args.create_catalog:
        from dsa110_contimg.core.simulation.synthetic_catalog import (
            create_synthetic_catalog_from_uvh5,
        )

        # Use first output file to extract source positions
        uvh5_path = outputs[0]

        # Determine catalog output path
        if args.catalog_output is None:
            # Auto-generate path based on declination strip
            dec_strip = round(float(uv_template.phase_center_dec_degrees), 1)
            catalog_name = f"{args.catalog_type}_dec{dec_strip:+.1f}.sqlite3"
            catalog_output = args.output.parent / "catalogs" / catalog_name
        else:
            catalog_output = args.catalog_output

        logger.info(f"Creating synthetic {args.catalog_type.upper()} catalog...")
        catalog_path = create_synthetic_catalog_from_uvh5(
            uvh5_path=uvh5_path,
            catalog_output_path=catalog_output,
            catalog_type=args.catalog_type,
            add_noise=True,  # Add realistic catalog errors
            rng=rng,
        )
        logger.info(f"  Created: {catalog_path}")
        logger.info("\nTo use in pipeline testing, set environment variable:")
        logger.info(f"  export {args.catalog_type.upper()}_CATALOG={catalog_path}")

    # Print summary of features used
    features = []
    features.append(f"pyuvsim visibility simulation (beam={args.pyuvsim_beam_type})")
    if args.source_model != "point":
        features.append(f"Extended source ({args.source_model}, {args.source_size_arcsec} arcsec)")
    if args.add_noise:
        features.append(f"Thermal noise (T_sys={args.system_temp_k}K)")
    if args.add_cal_errors:
        features.append(
            f"Calibration errors (gain_std={args.gain_std}, phase_std={args.phase_std_deg}deg)"
        )

    if features:
        logger.info("Features enabled:")
        for feature in features:
            logger.info(f"  - {feature}")
    else:
        logger.info("Using basic point source model (no noise, no cal errors)")

    # Validate generated files if validation module is available (unless --no-validate)
    if not args.no_validate:
        try:
            if not outputs:
                logger.info("No outputs to validate.")
            else:
                logger.info("Validating generated files...")
                max_workers = min(16, len(outputs), os.cpu_count() or 1)
                all_valid, validated = _validate_outputs_parallel(outputs, max_workers)
                for path, is_valid, errors in validated:
                    if is_valid:
                        logger.info(f"  ✅ {path.name}: Valid")
                    else:
                        logger.error(f"  ❌ {path.name}: Invalid - {errors}")
                if all_valid:
                    logger.info("✅ All generated files validated successfully")
                else:
                    logger.warning("⚠️ Some generated files failed validation")
        except ImportError:
            logger.warning("⚠️ Validation module not available, skipping validation")


if __name__ == "__main__":
    main()
