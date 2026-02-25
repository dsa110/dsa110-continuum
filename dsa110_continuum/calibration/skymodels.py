# pylint: disable=no-member  # astropy.units uses dynamic attributes (deg, Jy, etc.)
"""
Skymodel helpers: create sky models and predict visibilities.

This module uses pyradiosky for sky model construction and WSClean for visibility
prediction, providing better sky model management, support for multiple catalog
formats, and advanced spectral modeling capabilities.

**Usage (Multi-source models):**
  from dsa110_contimg.core.calibration.skymodels import predict_from_skymodel_wsclean, make_unified_skymodel
  sky = make_unified_skymodel(ra_deg, dec_deg, radius_deg=1.0, min_mjy=2.0)
  predict_from_skymodel_wsclean('/path/to/obs.ms', sky, field='0')

**Usage (Single point source):**
  from dsa110_contimg.core.calibration.skymodels import make_point_skymodel, predict_from_skymodel_wsclean
  sky = make_point_skymodel('0834+555', ra_deg, dec_deg, flux_jy=2.3)
  predict_from_skymodel_wsclean('/path/to/obs.ms', sky, field='0')
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def make_point_skymodel(
    name: str,
    ra_deg: float,
    dec_deg: float,
    *,
    flux_jy: float,
    freq_ghz: float | str = 1.4,
) -> Any:  # pyradiosky.SkyModel - imported conditionally
    """Create a pyradiosky SkyModel for a single point source.

    Parameters
    ----------
    name : str
        Source name
    ra_deg : float
        RA in degrees
    dec_deg : float
        Dec in degrees
    flux_jy : float
        Flux in Jy
    freq_ghz : float
        Reference frequency in GHz, default is 1.4

    Returns
    -------
        pyradiosky.SkyModel
        SkyModel object representing the point source
    """
    try:
        from pyradiosky import SkyModel  # noqa: F401
    except ImportError:
        raise ImportError(
            "pyradiosky is required for make_point_skymodel(). Install with: pip install pyradiosky"
        )

    import astropy.units as u
    import numpy as np
    from astropy.coordinates import SkyCoord

    # Get reference frequency
    if isinstance(freq_ghz, (int, float)):
        ref_freq = freq_ghz * u.GHz
    else:
        ref_freq = 1.4 * u.GHz  # Default

    # Create SkyCoord
    skycoord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")

    # Create stokes array: (4, Nfreqs, Ncomponents)
    stokes = np.zeros((4, 1, 1)) * u.Jy
    stokes[0, 0, 0] = flux_jy * u.Jy  # I stokes

    # Create SkyModel
    sky = SkyModel(
        name=[name],
        skycoord=skycoord,
        stokes=stokes,
        spectral_type="flat",
        component_type="point",
        freq_array=np.array([ref_freq.to("Hz").value]) * u.Hz,
    )

    return sky


def make_nvss_skymodel(
    center_ra_deg: float,
    center_dec_deg: float,
    radius_deg: float,
    *,
    min_mjy: float = 10.0,
    freq_ghz: float | str = 1.4,
    catalog: str = "nvss",
) -> Any:  # pyradiosky.SkyModel - imported conditionally
    """Create a pyradiosky SkyModel from NVSS sources in a sky region.

        Selects NVSS sources with flux >= min_mjy within radius_deg of (RA,Dec)
        and returns a pyradiosky SkyModel object.

    Parameters
    ----------
    center_ra_deg : float
        Center RA in degrees.
    center_dec_deg : float
        Center Dec in degrees.
    radius_deg : float
        Search radius in degrees.
    min_mjy : float, optional
        Minimum flux in mJy. Default is 10.0.
    freq_ghz : float, optional
        Reference frequency in GHz. Default is 1.4.

    Returns
    -------
        Any
        pyradiosky SkyModel object.
    """
    try:
        from pyradiosky import SkyModel  # noqa: F401
    except ImportError:
        raise ImportError(
            "pyradiosky is required for make_nvss_skymodel(). Install with: pip install pyradiosky"
        )

    import astropy.units as u
    import numpy as np
    from astropy.coordinates import SkyCoord

    # Use SQLite-first query function (falls back to CSV if needed)
    from dsa110_contimg.core.catalog.query import query_sources  # type: ignore

    df = query_sources(
        catalog_type=catalog,
        ra_center=center_ra_deg,
        dec_center=center_dec_deg,
        radius_deg=float(radius_deg),
        min_flux_mjy=float(min_mjy),
    )
    # Rename columns to match expected format
    df = df.rename(columns={"ra_deg": "ra", "dec_deg": "dec", "flux_mjy": "flux_20_cm"})
    flux_mjy = np.asarray(df["flux_20_cm"].to_numpy(), float)

    if len(df) == 0:
        # Return empty SkyModel
        return SkyModel(
            name=[],
            skycoord=SkyCoord([], [], unit=u.deg, frame="icrs"),
            stokes=np.zeros((4, 1, 0)) * u.Jy,
            spectral_type="flat",
            component_type="point",
        )

    # Extract sources (already filtered by query_sources)
    ras = df["ra"].to_numpy()
    decs = df["dec"].to_numpy()
    fluxes = flux_mjy / 1000.0  # Convert to Jy

    # Create SkyCoord
    ra = ras * u.deg
    dec = decs * u.deg
    skycoord = SkyCoord(ra=ra, dec=dec, frame="icrs")

    # Create stokes array: (4, Nfreqs, Ncomponents)
    # For flat spectrum, we use a single frequency
    n_components = len(ras)
    stokes = np.zeros((4, 1, n_components)) * u.Jy
    stokes[0, 0, :] = fluxes * u.Jy  # I stokes

    # Get reference frequency
    if isinstance(freq_ghz, (int, float)):
        ref_freq = freq_ghz * u.GHz
    else:
        ref_freq = 1.4 * u.GHz  # Default

    # Create SkyModel
    sky = SkyModel(
        name=[f"nvss_{i}" for i in range(n_components)],
        skycoord=skycoord,
        stokes=stokes,
        spectral_type="flat",
        component_type="point",
        freq_array=np.array([ref_freq.to("Hz").value]) * u.Hz,
    )

    return sky


def make_unified_skymodel(
    center_ra_deg: float,
    center_dec_deg: float,
    radius_deg: float,
    *,
    min_mjy: float = 2.0,
    freq_ghz: float | str = 1.4,
    match_radius_arcsec: float = 5.0,
) -> Any:
    """Create a unified SkyModel by merging FIRST, RACS, and NVSS catalogs.

    Priority: FIRST > RACS > NVSS.
        Sources are cross-matched, and lower-priority counterparts are removed
        if they fall within match_radius_arcsec.

    Parameters
    ----------
    center_ra_deg : float
        Center RA in degrees.
    center_dec_deg : float
        Center Dec in degrees.
    radius_deg : float
        Search radius in degrees.
    min_mjy : float, optional
        Minimum flux in mJy. Default is 2.0.
    freq_ghz : float or str, optional
        Reference frequency in GHz. Default is 1.4.
    match_radius_arcsec : float, optional
        Cross-match radius in arcseconds. Default is 5.0.

    Returns
    -------
        None
    """
    try:
        from pyradiosky import SkyModel  # noqa: F401
    except ImportError:
        raise ImportError(
            "pyradiosky is required for make_unified_skymodel(). "
            "Install with: pip install pyradiosky"
        )

    import astropy.units as u
    import numpy as np
    import pandas as pd
    from astropy.coordinates import SkyCoord

    from dsa110_contimg.core.catalog.query import query_sources

    # Helper to standardize DataFrame
    def fetch_catalog(ctype: str) -> pd.DataFrame:
        try:
            df = query_sources(
                catalog_type=ctype,
                ra_center=center_ra_deg,
                dec_center=center_dec_deg,
                radius_deg=float(radius_deg),
                min_flux_mjy=float(min_mjy),
            )
            # Rename for consistency if needed (query_sources returns ra_deg, dec_deg, flux_mjy)
            return df
        except (ValueError, KeyError, OSError):
            return pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_mjy"])

    # 1. Fetch all catalogs
    df_first = fetch_catalog("first")
    df_racs = fetch_catalog("racs")
    df_nvss = fetch_catalog("nvss")

    # Add source origin label
    if not df_first.empty:
        df_first["origin"] = "FIRST"
    if not df_racs.empty:
        df_racs["origin"] = "RACS"
    if not df_nvss.empty:
        df_nvss["origin"] = "NVSS"

    # 2. Start with FIRST (Highest Priority)
    unified_df = df_first.copy()

    # 3. Merge RACS (Medium Priority)
    if not df_racs.empty:
        if unified_df.empty:
            unified_df = df_racs.copy()
        else:
            # Match RACS to current Unified (FIRST)
            c_unified = SkyCoord(
                ra=unified_df["ra_deg"].values * u.deg,
                dec=unified_df["dec_deg"].values * u.deg,
                frame="icrs",
            )
            c_racs = SkyCoord(
                ra=df_racs["ra_deg"].values * u.deg,
                dec=df_racs["dec_deg"].values * u.deg,
                frame="icrs",
            )

            # Find matches
            idx, d2d, _ = c_racs.match_to_catalog_sky(c_unified)

            # Keep RACS sources that are NOT matched within radius
            is_unmatched = d2d > (match_radius_arcsec * u.arcsec)
            unique_racs = df_racs[is_unmatched]

            unified_df = pd.concat([unified_df, unique_racs], ignore_index=True)

    # 4. Merge NVSS (Lowest Priority)
    if not df_nvss.empty:
        if unified_df.empty:
            unified_df = df_nvss.copy()
        else:
            # Match NVSS to current Unified (FIRST + RACS)
            c_unified = SkyCoord(
                ra=unified_df["ra_deg"].values * u.deg,
                dec=unified_df["dec_deg"].values * u.deg,
                frame="icrs",
            )
            c_nvss = SkyCoord(
                ra=df_nvss["ra_deg"].values * u.deg,
                dec=df_nvss["dec_deg"].values * u.deg,
                frame="icrs",
            )

            # Find matches
            idx, d2d, _ = c_nvss.match_to_catalog_sky(c_unified)

            # Keep NVSS sources that are NOT matched within radius
            is_unmatched = d2d > (match_radius_arcsec * u.arcsec)
            unique_nvss = df_nvss[is_unmatched]

            unified_df = pd.concat([unified_df, unique_nvss], ignore_index=True)

    if unified_df.empty:
        # Return an empty SkyModel with run_check=False to avoid validation errors
        # on empty arrays (pyradiosky doesn't handle zero-component models well)
        return SkyModel(
            name=[],
            skycoord=SkyCoord([], [], unit=u.deg, frame="icrs"),
            stokes=np.zeros((4, 1, 0)) * u.Jy,
            spectral_type="flat",
            component_type="point",
            run_check=False,
        )

    # 5. Create Final SkyModel
    ras = unified_df["ra_deg"].to_numpy()
    decs = unified_df["dec_deg"].to_numpy()
    fluxes = unified_df["flux_mjy"].to_numpy() / 1000.0  # Jy
    origins = (
        unified_df["origin"].to_numpy() if "origin" in unified_df.columns else ["UNK"] * len(ras)
    )

    n_components = len(ras)

    # Create SkyCoord
    skycoord = SkyCoord(ra=ras * u.deg, dec=decs * u.deg, frame="icrs")

    # Create stokes array
    stokes = np.zeros((4, 1, n_components)) * u.Jy
    stokes[0, 0, :] = fluxes * u.Jy

    # Get reference frequency
    if isinstance(freq_ghz, (int, float)):
        ref_freq = freq_ghz * u.GHz
    else:
        ref_freq = 1.4 * u.GHz

    # Create unique names
    names = [f"{origins[i]}_J{ras[i]:.4f}{decs[i]:+.4f}" for i in range(n_components)]

    sky = SkyModel(
        name=names,
        skycoord=skycoord,
        stokes=stokes,
        spectral_type="flat",
        component_type="point",
        freq_array=np.array([ref_freq.to("Hz").value]) * u.Hz,
    )

    return sky


def write_wsclean_source_list(
    sky: Any,  # pyradiosky.SkyModel
    out_path: str,
    freq_ghz: float | str = 1.4,
) -> str:
    """Write pyradiosky SkyModel to WSClean text format.

    Format: Name, Type, Ra, Dec, I, Q, U, V, SpectralIndex, LogarithmicSI, ReferenceFrequency, MajorAxis, MinorAxis, Orientation

    Parameters
    ----------
    sky : Any
        pyradiosky SkyModel object.
    out_path : str
        Output file path.
    freq_ghz : float or str, optional
        Reference frequency. Default is 1.4.

    Returns
    -------
        None
    """
    try:
        # Ensure we start with a fresh file
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass

        with open(out_path, "w") as f:
            # Write Header (WSClean 3.6 requires 'format = ...')
            f.write(
                "format = Name, Type, Ra, Dec, I, Q, U, V, SpectralIndex, LogarithmicSI, ReferenceFrequency, MajorAxis, MinorAxis, Orientation\n"
            )

            # Get ref freq
            if sky.freq_array is not None and len(sky.freq_array) > 0:
                ref_freq_hz = sky.freq_array[0].to("Hz").value
            else:
                ref_freq_hz = float(freq_ghz) * 1e9

            for i in range(sky.Ncomponents):
                name = sky.name[i]
                ra = sky.skycoord[i].ra
                dec = sky.skycoord[i].dec
                flux_jy = sky.stokes[0, 0, i].to("Jy").value

                # Format RA/Dec as hms/dms (WSClean 3.6 requirement)
                # Re-do formatting to be safe and match WSClean strictness
                ra_hours = ra.hour
                ra_h = int(ra_hours)
                ra_m = int((ra_hours - ra_h) * 60)
                ra_s = ((ra_hours - ra_h) * 60 - ra_m) * 60
                ra_fmt = f"{ra_h:02d}h{ra_m:02d}m{ra_s:06.3f}s"

                dec_deg = dec.deg
                dec_sign = "+" if dec_deg >= 0 else "-"
                dec_abs = abs(dec_deg)
                dec_d = int(dec_abs)
                dec_m = int((dec_abs - dec_d) * 60)
                dec_s = ((dec_abs - dec_d) * 60 - dec_m) * 60
                dec_fmt = f"{dec_sign}{dec_d:02d}d{dec_m:02d}m{dec_s:06.3f}s"

                # Spectral Index
                si = "[]"
                if sky.spectral_type == "spectral_index" and hasattr(sky, "spectral_index"):
                    if sky.spectral_index is not None:
                        si = f"[{float(sky.spectral_index[i])}]"
                else:
                    # Default to -0.7 for radio sources if not specified
                    si = "[-0.7]"

                # Check for extended source shape (WSClean uses arcsec for axes, deg for PA)
                major = 0.0
                minor = 0.0
                pa = 0.0
                source_type = "POINT"

                # PyRadioSky typically stores these in SkyModel.major_axis etc as Quantities
                if hasattr(sky, "major_axis") and sky.major_axis is not None:
                    # Access the i-th element
                    maj_val = sky.major_axis[i]
                    if maj_val is not None and maj_val.value > 0:
                        major = maj_val.to("arcsec").value
                        source_type = "GAUSSIAN"

                if hasattr(sky, "minor_axis") and sky.minor_axis is not None:
                    min_val = sky.minor_axis[i]
                    if min_val is not None and min_val.value > 0:
                        minor = min_val.to("arcsec").value

                if hasattr(sky, "position_angle") and sky.position_angle is not None:
                    pa_val = sky.position_angle[i]
                    if pa_val is not None:
                        pa = pa_val.to("deg").value

                # Line
                # Name, Type, Ra, Dec, I, Q, U, V, SpectralIndex, LogarithmicSI, ReferenceFrequency, MajorAxis, MinorAxis, Orientation
                line = f"{name},{source_type},{ra_fmt},{dec_fmt},{flux_jy},0,0,0,{si},[false],{ref_freq_hz},{major},{minor},{pa}\n"
                f.write(line)

    except Exception as e:
        raise RuntimeError(f"Failed to write WSClean source list: {e}")

    return out_path


def make_nvss_wsclean_list(
    center_ra_deg: float,
    center_dec_deg: float,
    radius_deg: float,
    *,
    min_mjy: float = 10.0,
    freq_ghz: float | str = 1.4,
    out_path: str,
) -> str:
    """Build a WSClean source list from NVSS sources.

    Parameters
    ----------
    center_ra_deg : float
        Center RA.
    center_dec_deg : float
        Center Dec.
    radius_deg : float
        Radius.
    min_mjy : float, optional
        Minimum flux. Default is 10.0.
    freq_ghz : float or str, optional
        Reference frequency. Default is 1.4.
    out_path : str
        Output file path.

    Returns
    -------
        None
    """
    sky = make_nvss_skymodel(
        center_ra_deg,
        center_dec_deg,
        radius_deg,
        min_mjy=min_mjy,
        freq_ghz=freq_ghz,
    )

    return write_wsclean_source_list(sky, out_path, freq_ghz=freq_ghz)


def make_unified_wsclean_list(
    center_ra_deg: float,
    center_dec_deg: float,
    radius_deg: float,
    *,
    min_mjy: float = 2.0,
    freq_ghz: float | str = 1.4,
    out_path: str,
) -> str:
    """Build a WSClean source list from the unified catalog (FIRST+RACS+NVSS).

    Parameters
    ----------
    center_ra_deg : float
        Center RA.
    center_dec_deg : float
        Center Dec.
    radius_deg : float
        Radius.
    min_mjy : float, optional
        Minimum flux. Default is 2.0.
    freq_ghz : float or str, optional
        Reference frequency. Default is 1.4.
    out_path : str
        Output file path.

    Returns
    -------
        None
    """
    sky = make_unified_skymodel(
        center_ra_deg,
        center_dec_deg,
        radius_deg,
        min_mjy=min_mjy,
        freq_ghz=freq_ghz,
    )

    return write_wsclean_source_list(sky, out_path, freq_ghz=freq_ghz)


def predict_from_skymodel_wsclean(
    ms_path: str,
    sky_model: Any,  # pyradiosky.SkyModel
    *,
    field: str | None = None,
    wsclean_path: str | None = None,
    temp_dir: str | None = None,
    imsize: int = 1024,
    cell_arcsec: float = 6.0,
    cleanup: bool = True,
) -> None:
    """Populate MODEL_DATA from multi-source SkyModel using WSClean -draw-model + -predict.

    This replaces CASA ft() for multi-source models by:
    1. Converting SkyModel to WSClean text format
    2. Rendering model image with wsclean -draw-model
    3. Predicting visibilities with wsclean -predict

    **Advantages over ft():**
    - Faster (multi-threaded, GPU-accelerated)
    - More reliable (avoids ft() phase center bugs)
    - Handles extended sources (Gaussian, not just points)
    - No component list intermediate files needed

    **Workflow:**
    1. SkyModel → WSClean text format
    2. wsclean -draw-model (creates model image)
    3. wsclean -predict (populates MODEL_DATA)

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set (will be modified)
    sky_model : SkyModel
        pyradiosky SkyModel with multiple sources
    field : str or None
        Field selection string. If None or empty, defaults to all fields in the MS
        (e.g., "0~23" for a 24-field MS). Used to determine which field's phase
        center to use for the WSClean model image. For a range like "0~23", the
        midpoint field's phase center is used.
    wsclean_path : str, optional
        Path to wsclean executable. If None, uses Docker or searches PATH
    temp_dir : str, optional
        Temporary directory for intermediate files. If None, uses system temp
    imsize : int
        Image size for model rendering in pixels (default: 1024)
    cell_arcsec : float
        Cell size in arcseconds (default: 6.0)
    cleanup : bool
        If True, remove temporary files after prediction (default: True)

    Raises
    ------
    RuntimeError
        If WSClean is not available or prediction fails
    ValueError
        If SkyModel is empty or MS cannot be read

    Examples
    --------
    >>> from pyradiosky import SkyModel
    >>> from dsa110_contimg.core.calibration.skymodels import predict_from_skymodel_wsclean, make_unified_skymodel
    >>> sky = make_unified_skymodel(ra_deg, dec_deg, radius_deg=1.0)
    >>> predict_from_skymodel_wsclean("obs.ms", sky, field="0~23")
    """
    import casacore.tables as casatables

    if sky_model.Ncomponents == 0:
        raise ValueError("SkyModel is empty - cannot predict visibilities")

    # Ensure MODEL_DATA column exists
    try:
        import casacore.tables as ct

        with ct.table(ms_path, readonly=True) as t:
            cols = set(t.colnames())
            if "MODEL_DATA" not in cols or "CORRECTED_DATA" not in cols:
                # Need to create columns - use clearcal with protection
                try:
                    from dsa110_contimg.core.calibration.casa_service import (
                        CalibrationProtectionError,
                        CASAService,
                    )

                    service = CASAService()
                    service.clearcal(vis=ms_path, addmodel=True)
                except CalibrationProtectionError as e:
                    # Calibration exists - cannot use clearcal safely
                    raise RuntimeError(
                        f"Cannot create MODEL_DATA column in {ms_path}: "
                        f"clearcal would destroy applied calibration."
                    ) from e
                except ImportError:
                    pass

    except RuntimeError:
        raise  # Re-raise RuntimeError from CalibrationProtectionError
    except Exception:
        pass  # Non-fatal if columns already exist

    # Clear existing MODEL_DATA to avoid conflicts
    try:
        t = casatables.table(ms_path, readonly=False)
        if "MODEL_DATA" in t.colnames() and t.nrows() > 0:
            if "DATA" in t.colnames():
                data_sample = t.getcell("DATA", 0)
                data_shape = getattr(data_sample, "shape", None)
                data_dtype = getattr(data_sample, "dtype", None)
                if data_shape and data_dtype:
                    zeros = np.zeros((t.nrows(),) + data_shape, dtype=data_dtype)
                    t.putcol("MODEL_DATA", zeros)
        t.close()
    except Exception as e:
        logger.warning(f"Failed to clear MODEL_DATA before WSClean predict: {e}")

    # Get frequency and bandwidth from MS
    freq_ghz = 1.4
    bandwidth_hz = 250e6
    try:
        with casatables.table(f"{ms_path}::SPECTRAL_WINDOW", readonly=True) as spw_tb:
            chan_freq = spw_tb.getcol("CHAN_FREQ")  # Shape: (nspw, nchan)
            if len(chan_freq) > 0 and len(chan_freq[0]) > 0:
                all_freqs = chan_freq.flatten()
                freq_ghz = float(np.nanmean(all_freqs)) / 1e9
                if len(all_freqs) > 1:
                    bandwidth_hz = float(
                        np.max(all_freqs) - np.min(all_freqs) + abs(all_freqs[1] - all_freqs[0])
                    )
                else:
                    try:
                        bandwidth_hz = float(spw_tb.getcol("TOTAL_BANDWIDTH")[0])
                    except (KeyError, IndexError):
                        bandwidth_hz = 250e6
    except Exception as e:
        logger.warning(f"Could not read frequency from MS, using defaults: {e}")

    # Get phase center from MS, using the field parameter to select which field
    ra0_deg = None
    dec0_deg = None
    try:
        with casatables.table(f"{ms_path}::FIELD", readonly=True) as field_tb:
            phase_dir = field_tb.getcol("PHASE_DIR")  # Shape: (nfields, 1, 2)
            nfields = len(phase_dir)
            if nfields == 0:
                raise ValueError(
                    "PHASE_DIR array is empty in FIELD table. "
                    "MS must have at least one field with a valid phase center."
                )

            # Parse field parameter to determine which field's phase center to use
            field_idx = 0  # fallback
            field_str = str(field).strip() if field is not None else ""
            if not field_str:
                if nfields > 1:
                    field_str = f"0~{nfields - 1}"
                else:
                    field_str = "0"
            if "~" in field_str:
                # Range like "0~23" — use the midpoint field
                parts = field_str.split("~")
                try:
                    start_idx = int(parts[0])
                    end_idx = int(parts[1])
                    field_idx = (start_idx + end_idx) // 2
                except (ValueError, IndexError):
                    field_idx = 0
            elif field_str.isdigit():
                field_idx = int(field_str)

            if field_idx >= nfields:
                logger.warning(
                    f"Requested field index {field_idx} exceeds MS field count ({nfields}). "
                    f"Falling back to field 0."
                )
                field_idx = 0

            ra0_rad = phase_dir[field_idx, 0, 0]
            dec0_rad = phase_dir[field_idx, 0, 1]
            ra0_deg = np.degrees(ra0_rad)
            dec0_deg = np.degrees(dec0_rad)

            # Error if fields have divergent phase centers (WSClean uses a single image)
            if nfields > 1:
                all_ra_deg = np.degrees(phase_dir[:, 0, 0])
                all_dec_deg = np.degrees(phase_dir[:, 0, 1])
                ra_spread = np.ptp(all_ra_deg)
                dec_spread = np.ptp(all_dec_deg)
                if ra_spread > 0.01 or dec_spread > 0.01:
                    raise ValueError(
                        f"MS has {nfields} fields with divergent phase centers "
                        f"(RA spread={ra_spread:.4f}°, Dec spread={dec_spread:.4f}°). "
                        f"WSClean -predict uses a single phase center (field {field_idx}), "
                        f"which will produce inaccurate MODEL_DATA for other fields. "
                        f"Phaseshift the MS to a common phase center before calling this function, "
                        f"or use _calculate_manual_model_data() for per-field phase centers."
                    )

    except ValueError:
        # Re-raise ValueError (empty phase_dir) as-is
        raise
    except Exception as e:
        logger.warning(f"Could not read phase center from MS: {e}")
        raise ValueError("Cannot determine phase center from MS") from e

    # Validate phase center was successfully extracted
    if ra0_deg is None or dec0_deg is None:
        raise ValueError(
            f"Failed to extract phase center from MS. "
            f"PHASE_DIR may be empty or invalid. ra0_deg={ra0_deg}, dec0_deg={dec0_deg}"
        )

    # Convert phase center to WSClean format (h:m:s and d:m:s)
    ra_hours = ra0_deg / 15.0
    ra_h = int(ra_hours)
    ra_m = int((ra_hours - ra_h) * 60)
    ra_s = ((ra_hours - ra_h) * 60 - ra_m) * 60
    ra_str = f"{ra_h}h{ra_m}m{ra_s:.3f}s"

    dec_sign = "+" if dec0_deg >= 0 else "-"
    dec_abs = abs(dec0_deg)
    dec_d = int(dec_abs)
    dec_m = int((dec_abs - dec_d) * 60)
    dec_s = ((dec_abs - dec_d) * 60 - dec_m) * 60
    dec_str = f"{dec_sign}{dec_d}d{dec_m}m{dec_s:.3f}s"

    # Create temporary directory for intermediate files
    if temp_dir is None:
        from dsa110_contimg.common.utils.temp_manager import get_temp_subdir

        temp_dir_obj = get_temp_subdir("calibration_wsclean")
        temp_dir = str(temp_dir_obj)
    else:
        temp_dir_obj = Path(temp_dir)
        temp_dir_obj.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Convert SkyModel to WSClean text format
        txt_path = str(temp_dir_obj / "model_sources.txt")
        write_wsclean_source_list(sky_model, txt_path, freq_ghz=freq_ghz)
        logger.info(f"Created WSClean source list with {sky_model.Ncomponents} sources: {txt_path}")

        # Step 2: Determine WSClean executable
        wsclean_exec = wsclean_path
        use_docker = False

        if not wsclean_exec:
            wsclean_exec = shutil.which("wsclean")

        if not wsclean_exec:
            if shutil.which("docker"):
                use_docker = True
                logger.info("Using Docker-based wsclean for model prediction")
            else:
                raise RuntimeError(
                    "wsclean executable not found and Docker is not available. "
                    "Install wsclean or Docker to use WSClean prediction."
                )

        # Step 3: Build model prefix
        model_prefix = str(temp_dir_obj / "model")

        if use_docker:
            # Build Docker command with volume mounts
            from dsa110_contimg.common.utils.gpu_utils import build_docker_command, get_gpu_config

            gpu_config = get_gpu_config()
            docker_user_flags = None
            if os.getenv("WSCLEAN_DOCKER_USER", "").lower() == "host":
                docker_user_flags = ["--user", f"{os.getuid()}:{os.getgid()}"]

            # Determine volume mounts needed for MS and temp directory
            volumes = {}
            ms_dir = os.path.dirname(os.path.abspath(ms_path))
            temp_dir_parent = str(temp_dir_obj.parent)

            # Mount MS directory if not already covered by default mounts
            default_mounts = ["/stage", "/dev/shm/dsa110-contimg", "/dev/shm"]
            if not any(ms_dir.startswith(mount) for mount in default_mounts):
                volumes[ms_dir] = ms_dir

            # Mount temp directory parent if not already covered
            if not any(temp_dir_parent.startswith(mount) for mount in default_mounts):
                volumes[temp_dir_parent] = temp_dir_parent

            docker_base = build_docker_command(
                image="dsa110-contimg:gpu",
                command=["wsclean"],
                gpu_config=gpu_config,
                volumes=volumes if volumes else None,  # None uses default mounts
                env_vars={
                    "NVIDIA_DISABLE_REQUIRE": "1",
                    "MAMBA_ROOT_PREFIX": "/dev/shm/micromamba",
                    "HOME": "/dev/shm/dsa110-contimg",
                },
                extra_flags=docker_user_flags,
            )

            # Step 3a: Render model image with -draw-model
            cmd_draw = docker_base.copy()
            cmd_draw.extend(
                [
                    "-draw-model",
                    txt_path,
                    "-name",
                    model_prefix,
                    "-draw-frequencies",
                    f"{freq_ghz * 1e9}",
                    f"{bandwidth_hz}",
                    "-draw-spectral-terms",
                    "2",
                    "-size",
                    str(imsize),
                    str(imsize),
                    "-scale",
                    f"{cell_arcsec}arcsec",
                    "-draw-centre",
                    ra_str,
                    dec_str,
                ]
            )
            logger.info("Running wsclean -draw-model via Docker...")
            subprocess.run(cmd_draw, check=True, timeout=300, capture_output=True)

            # Step 3b: Rename output file for prediction
            # WSClean -draw-model creates prefix-term-0.fits
            # WSClean -predict expects prefix-model.fits
            term_file = f"{model_prefix}-term-0.fits"
            model_file = f"{model_prefix}-model.fits"
            if os.path.exists(term_file):
                shutil.move(term_file, model_file)
                logger.debug("Renamed %s -> %s", term_file, model_file)
            else:
                # Check if model_file already exists (maybe from a previous run)
                if not os.path.exists(model_file):
                    raise RuntimeError(
                        f"wsclean -draw-model failed to produce expected output file: {term_file}. "
                        f"The -draw-model step may have failed. Check wsclean logs for errors."
                    )
                else:
                    logger.debug("Model file already exists: %s (skipping rename)", model_file)

            # Step 3c: Predict visibilities with -predict
            cmd_predict = docker_base.copy()
            cmd_predict.extend(
                [
                    "-predict",
                    "-reorder",  # Required for multi-SPW MS
                    "-name",
                    model_prefix,
                    ms_path,
                ]
            )
            logger.info("Running wsclean -predict via Docker...")
            subprocess.run(cmd_predict, check=True, timeout=600, capture_output=True)

        else:
            # Native WSClean execution
            # Step 3a: Render model image
            cmd_draw = [
                wsclean_exec,
                "-draw-model",
                txt_path,
                "-name",
                model_prefix,
                "-draw-frequencies",
                f"{freq_ghz * 1e9}",
                f"{bandwidth_hz}",
                "-draw-spectral-terms",
                "2",
                "-size",
                str(imsize),
                str(imsize),
                "-scale",
                f"{cell_arcsec}arcsec",
                "-draw-centre",
                ra_str,
                dec_str,
            ]
            logger.info("Running wsclean -draw-model...")
            subprocess.run(cmd_draw, check=True, timeout=300, capture_output=True)

            # Step 3b: Rename output file
            term_file = f"{model_prefix}-term-0.fits"
            model_file = f"{model_prefix}-model.fits"
            if os.path.exists(term_file):
                shutil.move(term_file, model_file)
                logger.debug("Renamed %s -> %s", term_file, model_file)
            else:
                # Check if model_file already exists (maybe from a previous run)
                if not os.path.exists(model_file):
                    raise RuntimeError(
                        f"wsclean -draw-model failed to produce expected output file: {term_file}. "
                        f"The -draw-model step may have failed. Check wsclean logs for errors."
                    )
                else:
                    logger.debug("Model file already exists: %s (skipping rename)", model_file)

            # Step 3c: Predict visibilities
            cmd_predict = [
                wsclean_exec,
                "-predict",
                "-reorder",  # Required for multi-SPW MS
                "-name",
                model_prefix,
                ms_path,
            ]
            logger.info("Running wsclean -predict...")
            subprocess.run(cmd_predict, check=True, timeout=600, capture_output=True)

        logger.info(":check: MODEL_DATA populated from SkyModel using WSClean")

    except subprocess.CalledProcessError as e:
        error_msg = f"WSClean prediction failed: {e}"
        if e.stdout:
            error_msg += f"\nstdout: {e.stdout.decode('utf-8', errors='ignore')[:500]}"
        if e.stderr:
            error_msg += f"\nstderr: {e.stderr.decode('utf-8', errors='ignore')[:500]}"
        raise RuntimeError(error_msg) from e
    except Exception as e:
        raise RuntimeError(f"Failed to predict from SkyModel using WSClean: {e}") from e
    finally:
        # Cleanup temporary files
        if cleanup and temp_dir_obj.exists():
            try:
                shutil.rmtree(temp_dir_obj)
                logger.debug(f"Cleaned up temporary directory: {temp_dir_obj}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory {temp_dir_obj}: {e}")
