"""Adapter layer for integrating pyuvsim with the DSA-110 simulation framework.

This module bridges the custom DSA-110 simulation toolkit with the well-tested
pyuvsim library for accurate visibility simulation. It provides:

1. Conversion from `SyntheticSource` to pyradiosky `SkyModel`
2. DSA-110-specific beam model setup
3. High-level simulation interface using pyuvsim's engine

Why pyuvsim?
------------
pyuvsim is a comprehensive, peer-reviewed simulation package for radio
interferometers (Lanman et al., 2019, JOSS). It provides:

- High-precision visibility calculation validated against analytic solutions
- Full polarization support with proper Jones matrix handling
- Beam interpolation and multiple beam model support
- MPI parallelization for large simulations

By using pyuvsim, we replace ~500 lines of custom visibility code with a
battle-tested implementation, reducing bugs and improving maintainability.

Example
-------
>>> from dsa110_contimg.core.simulation.pyuvsim_adapter import simulate_visibilities
>>> from dsa110_contimg.core.simulation.source_selection import SyntheticSource
>>> 
>>> sources = [SyntheticSource(
...     source_id="TEST", ra_deg=180.0, dec_deg=54.6,
...     flux_ref_jy=1.0, reference_freq_hz=1.4e9, spectral_index=-0.7
... )]
>>> uvdata = ... # Your UVData object
>>> result = simulate_visibilities(uvdata, sources)
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from astropy.coordinates import Latitude, Longitude, SkyCoord
from astropy.time import Time
from pyradiosky import SkyModel
from pyuvdata import AiryBeam, GaussianBeam, UVData

if TYPE_CHECKING:
    from dsa110_contimg.core.simulation.source_selection import SyntheticSource

logger = logging.getLogger(__name__)

# DSA-110 beam parameters
DSA110_DISH_DIAMETER_M = 4.65  # meters
DSA110_REFERENCE_FREQ_HZ = 1.498e9  # Hz (center of band)


def sources_to_skymodel(
    sources: Sequence[SyntheticSource],
    reference_freq_hz: float = DSA110_REFERENCE_FREQ_HZ,
) -> SkyModel:
    """Convert a list of SyntheticSource objects to a pyradiosky SkyModel.

    Parameters
    ----------
    sources : sequence of SyntheticSource
        List of sources from the DSA-110 catalog query or synthetic generation.
    reference_freq_hz : float, optional
        Reference frequency for spectral modeling (default: DSA-110 band center).

    Returns
    -------
    SkyModel
        pyradiosky SkyModel object suitable for pyuvsim simulation.

    Notes
    -----
    This function handles two spectral types:

    1. "spectral_index" - Sources with a power-law spectrum: S(ν) = S₀(ν/ν₀)^α
    2. "flat" - Sources with constant flux across frequency (α=0)

    All sources are modeled as unpolarized (Stokes I only).
    """
    if not sources:
        # Return empty SkyModel
        return SkyModel(
            ra=Longitude(np.array([]) * u.deg),
            dec=Latitude(np.array([]) * u.deg),
            stokes=np.array([]).reshape(4, 1, 0) * u.Jy,
            spectral_type="flat",
            name=np.array([], dtype=str),
            frame="icrs",
        )

    # Extract source properties
    n_sources = len(sources)
    ra_deg = np.array([s.ra_deg for s in sources])
    dec_deg = np.array([s.dec_deg for s in sources])
    flux_jy = np.array([s.flux_ref_jy for s in sources])
    spectral_indices = np.array([
        s.spectral_index if s.spectral_index is not None else 0.0
        for s in sources
    ])
    ref_freqs = np.array([
        s.reference_freq_hz if s.reference_freq_hz else reference_freq_hz
        for s in sources
    ])
    names = np.array([
        s.source_id if s.source_id else f"source_{i}"
        for i, s in enumerate(sources)
    ])

    # Build Stokes array (4 Stokes params × 1 freq × N sources)
    # For unpolarized sources: I = flux, Q = U = V = 0
    stokes = np.zeros((4, 1, n_sources)) * u.Jy
    stokes[0, 0, :] = flux_jy * u.Jy

    # Check if all sources have the same spectral behavior
    has_spectral_index = np.any(spectral_indices != 0.0)

    if has_spectral_index:
        # Use spectral_index type for power-law spectra
        sky = SkyModel(
            ra=Longitude(ra_deg * u.deg),
            dec=Latitude(dec_deg * u.deg),
            stokes=stokes,
            spectral_type="spectral_index",
            spectral_index=spectral_indices,
            reference_frequency=ref_freqs * u.Hz,
            name=names,
            frame="icrs",
        )
    else:
        # Use flat spectrum for constant-flux sources
        sky = SkyModel(
            ra=Longitude(ra_deg * u.deg),
            dec=Latitude(dec_deg * u.deg),
            stokes=stokes,
            spectral_type="flat",
            name=names,
            frame="icrs",
        )

    logger.info(
        "Created SkyModel with %d sources (spectral_type=%s)",
        n_sources,
        sky.spectral_type,
    )
    return sky


def create_dsa110_beam(
    beam_type: str = "airy",
    diameter_m: float = DSA110_DISH_DIAMETER_M,
) -> AiryBeam | GaussianBeam:
    """Create a beam model appropriate for DSA-110 antennas.

    Parameters
    ----------
    beam_type : str, optional
        Type of beam model: "airy" (default) or "gaussian".
    diameter_m : float, optional
        Dish diameter in meters (default: 4.65m for DSA-110).

    Returns
    -------
    AiryBeam or GaussianBeam
        pyuvdata analytic beam object.

    Notes
    -----
    - Airy beam is more physically accurate for a circular aperture
    - Gaussian beam is faster but less accurate at large angles
    - Both are E-field beams suitable for pyuvsim
    """
    if beam_type.lower() == "airy":
        beam = AiryBeam(diameter=diameter_m, feed_array=np.array(['x', 'y']))
        logger.debug("Created Airy beam with diameter %.2f m", diameter_m)
    elif beam_type.lower() == "gaussian":
        # FWHM ≈ 1.22 λ / D (in radians) for diffraction limit
        # At 1.5 GHz: λ ≈ 0.2m, FWHM ≈ 3 degrees
        beam = GaussianBeam(diameter=diameter_m, feed_array=np.array(['x', 'y']))
        logger.debug("Created Gaussian beam with diameter %.2f m", diameter_m)
    else:
        raise ValueError(f"Unknown beam type: {beam_type}. Use 'airy' or 'gaussian'.")

    return beam


def simulate_visibilities(
    uvdata: UVData,
    sources: Sequence[SyntheticSource],
    beam_type: str = "airy",
    use_mpi: bool = False,
    quiet: bool = True,
    sky_model: SkyModel | None = None,
    beam_list: object | None = None,
) -> UVData:
    """Simulate visibilities using pyuvsim.

    This is the main entry point for pyuvsim-based simulation. It takes a
    configured UVData object (with antenna positions, times, frequencies)
    and a list of sources, and returns a UVData object with simulated
    visibilities.

    Parameters
    ----------
    uvdata : UVData
        Input UVData object defining the observation parameters.
        Must have antenna positions, time_array, freq_array set.
        The data_array will be overwritten with simulated visibilities.
    sources : sequence of SyntheticSource
        List of sources to simulate. Ignored if sky_model is provided.
    beam_type : str, optional
        Beam model type: "airy" (default) or "gaussian". Ignored if beam_list is provided.
    use_mpi : bool, optional
        Whether to use MPI parallelization (requires mpi4py).
        Default is False for single-process operation.
    quiet : bool, optional
        Suppress progress output (default: True).
    sky_model : SkyModel, optional
        Pre-computed SkyModel object. If provided, `sources` is ignored.
    beam_list : BeamList, optional
        Pre-computed BeamList object. If provided, `beam_type` is ignored.

    Returns
    -------
    UVData
        UVData object with simulated visibilities in data_array.

    Raises
    ------
    ImportError
        If pyuvsim or mpi4py is not installed when use_mpi=True.
    ValueError
        If uvdata is not properly configured.

    Notes
    -----
    This function wraps pyuvsim's run_uvdata_uvsim() which provides:
    - Exact phase calculation for each source/baseline/time/frequency
    - Proper primary beam attenuation
    - Full polarization handling

    The simulation does NOT add thermal noise. Use `add_thermal_noise()`
    from visibility_models.py after simulation if noise is needed.

    Example
    -------
    >>> uvdata = build_uvdata_from_scratch(config, nants=117, ntimes=24)
    >>> sources = selector.select_sources(min_flux_mjy=100, max_sources=50)
    >>> uvdata_sim = simulate_visibilities(uvdata, sources)
    """
    # Validate input
    if uvdata.data_array is None:
        # Initialize data array if not present
        nblts = uvdata.Nblts
        nfreqs = uvdata.Nfreqs
        npols = uvdata.Npols
        uvdata.data_array = np.zeros((nblts, nfreqs, npols), dtype=np.complex128)

    # Convert sources to SkyModel if not provided
    if sky_model is None:
        sky_model = sources_to_skymodel(sources)

    if sky_model.Ncomponents == 0:
        logger.warning("No sources provided, returning zeros")
        uvdata.data_array[:] = 0.0
        return uvdata

    # Import pyuvsim components
    try:
        from pyuvsim import run_uvdata_uvsim
        from pyuvsim.telescope import BeamList
    except ImportError as e:
        raise ImportError(
            "pyuvsim is required for visibility simulation. "
            "Install with: pip install pyuvsim[sim]"
        ) from e

    # Create BeamList if not provided
    if beam_list is None:
        # Create beam
        beam = create_dsa110_beam(beam_type=beam_type)
        # Create BeamList (all antennas use the same beam)
        beam_list = BeamList([beam])

    # beam_dict maps antenna names/numbers to beam indices
    # None means all antennas use beam index 0
    beam_dict = None

    # Run simulation
    logger.info(
        "Running pyuvsim simulation: %d sources, %d baselines, %d times, %d freqs",
        sky_model.Ncomponents,
        uvdata.Nbls,
        uvdata.Ntimes,
        uvdata.Nfreqs,
    )

    try:
        # pyuvsim.run_uvdata_uvsim modifies input_uv in place and returns it
        run_kw = dict(
            input_uv=uvdata,
            beam_list=beam_list,
            beam_dict=beam_dict,
            catalog=sky_model,
            quiet=quiet,
        )
        if use_mpi:
            try:
                from mpi4py import MPI
                run_kw["mpi_comm"] = MPI.COMM_WORLD
            except ImportError:
                raise ImportError("use_mpi=True requires mpi4py. Install with: pip install mpi4py") from None
        try:
            result = run_uvdata_uvsim(**run_kw)
        except TypeError:
            run_kw.pop("mpi_comm", None)
            result = run_uvdata_uvsim(**run_kw)
    except Exception as e:
        logger.error("pyuvsim simulation failed: %s", e)
        raise

    logger.info("pyuvsim simulation complete")
    return result if result is not None else uvdata


def check_pyuvsim_available() -> bool:
    """Check if pyuvsim simulation is available.

    Returns
    -------
    bool
        True if pyuvsim with MPI support is available.
    """
    try:
        from pyuvsim import run_uvdata_uvsim  # noqa: F401
        from pyuvsim.telescope import BeamList  # noqa: F401
        return True
    except ImportError:
        return False


def check_mpi_available() -> bool:
    """Check if MPI is available for parallel simulation.

    Returns
    -------
    bool
        True if mpi4py is installed and functional.
    """
    try:
        import mpi4py  # noqa: F401
        return True
    except ImportError:
        return False
