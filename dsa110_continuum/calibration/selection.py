# pylint: disable=no-member  # astropy.units uses dynamic attributes (deg, etc.)
from __future__ import annotations

# Ensure CASAPATH is set before importing CASA modules
from dsa110_contimg.common.utils.casa_init import ensure_casa_path

ensure_casa_path()

import astropy.units as u
import casacore.tables as casatables
import numpy as np
import pandas as pd
from astropy.coordinates import Angle

table = casatables.table  # noqa: N816

from dsa110_contimg.core.calibration import BeamConfig, primary_beam_response

from .catalogs import (
    load_vla_catalog,
    read_vla_calibrator_catalog,
)


def _read_field_dirs(ms_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read FIELD::PHASE_DIR and return arrays of RA/Dec in radians per field.

    Handles column shapes (n,1,2), (n,2), or (2,) per row.

    Parameters
    ----------
    """
    with table(f"{ms_path}::FIELD") as tf:
        pd = tf.getcol("PHASE_DIR")
        n = tf.nrows()
        ra = np.zeros(n, dtype=float)
        dec = np.zeros(n, dtype=float)
        for i in range(n):
            arr = np.asarray(pd[i])
            if arr.ndim == 3 and arr.shape[-1] == 2:  # (1,1,2)
                ra[i] = float(arr[0, 0, 0])
                dec[i] = float(arr[0, 0, 1])
            elif arr.ndim == 2 and arr.shape[-1] == 2:  # (1,2)
                ra[i] = float(arr[0, 0])
                dec[i] = float(arr[0, 1])
            elif arr.ndim == 1 and arr.shape[0] == 2:  # (2,)
                ra[i] = float(arr[0])
                dec[i] = float(arr[1])
            else:
                # Fallback
                ra[i] = float(arr.ravel()[-2])
                dec[i] = float(arr.ravel()[-1])
    return ra, dec


def select_bandpass_fields(
    ms_path: str,
    cal_ra_deg: float,
    cal_dec_deg: float,
    cal_flux_jy: float,
    *,
    window: int = 3,
    min_pb: float | None = None,
    freq_GHz: float = 1.4,
    use_beam_weighting: bool = False,
) -> tuple[str, list[int], np.ndarray]:
    """Pick optimal FIELD indices for bandpass solving.

    By default, returns all fields (0~23) for maximum SNR since bandpass
    calibration benefits from all available data and the beam response
    cancels out in the calibration solve.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    cal_ra_deg, cal_dec_deg :
        Calibrator RA/Dec in degrees
    cal_flux_jy :
        Calibrator flux in Jy at observation frequency
    window :
        Number of fields around peak to include (only used if use_beam_weighting=True)
    min_pb :
        Minimum primary beam response threshold (only used if use_beam_weighting=True)
    freq_GHz :
        Observation frequency in GHz
    use_beam_weighting :
        If True, weight fields by primary beam response (Airy model).
        If False (default), return all fields for maximum SNR.
        Returns (field_sel_str, indices, weighted_flux_per_field)
    """
    ra_f, dec_f = _read_field_dirs(ms_path)
    n = ra_f.size
    if n == 0:
        return "", [], np.array([])

    src_ra = float(Angle(cal_ra_deg, unit=u.deg).rad)
    src_dec = float(Angle(cal_dec_deg, unit=u.deg).rad)

    # If only one field exists, return it directly (no range needed)
    if n == 1:
        return "0", [0], np.ones(1) * float(cal_flux_jy)

    # Default: Use all fields for maximum SNR in bandpass calibration
    # Beam response cancels out in calibration since we solve for per-antenna gains
    if not use_beam_weighting:
        sel_str = f"0~{n - 1}"
        indices = list(range(n))
        # Return uniform flux for all fields (no beam weighting)
        wflux = np.ones(n, dtype=float) * float(cal_flux_jy)
        return sel_str, indices, wflux

    # Optional: Primary-beam weighted field selection
    # Uses Airy disk model to weight fields by beam response
    wflux = np.zeros(n, dtype=float)
    resp = np.zeros(n, dtype=float)
    for i in range(n):
        config_i = BeamConfig(
            frequency_ghz=freq_GHz,
            antenna_ra=src_ra,
            antenna_dec=src_dec,
            ms_path=ms_path,
            field_id=i,
        )
        r = primary_beam_response(
            src_ra=ra_f[i],
            src_dec=dec_f[i],
            config=config_i,
        )
        resp[i] = r
        wflux[i] = r * float(cal_flux_jy)

    # Pick best center and window
    idx = int(np.nanargmax(wflux))
    if min_pb is not None and np.isfinite(min_pb):
        thr = float(min_pb) * max(resp[idx], 1e-12)
        # Expand contiguously around peak while resp >= thr
        start = idx
        end = idx
        while start - 1 >= 0 and resp[start - 1] >= thr:
            start -= 1
        while end + 1 < n and resp[end + 1] >= thr:
            end += 1
    else:
        half = max(1, int(window)) // 2
        start = max(0, idx - half)
        end = min(n - 1, idx + half)

    sel_str = f"{start}~{end}" if start != end else f"{start}"
    indices = list(range(start, end + 1))
    return sel_str, indices, wflux


def select_bandpass_from_catalog(
    ms_path: str,
    catalog_path: str | None = None,
    *,
    search_radius_deg: float = 1.0,
    freq_GHz: float = 1.4,
    window: int = 3,
    min_pb: float | None = None,
    calibrator_name: str | None = None,
    use_beam_weighting: bool = False,
) -> tuple[str, list[int], np.ndarray, tuple[str, float, float, float], int]:
    """Select bandpass fields by scanning a VLA calibrator catalog.

        By default, returns all fields (0~23) when a calibrator is found, since
        bandpass calibration benefits from maximum SNR and the beam response
        cancels out in the calibration solve.

        Automatically prefers SQLite catalog if available, falls back to CSV.
        If catalog_path is None, uses automatic resolution (prefers SQLite).

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set
    catalog_path : Optional[str]
        Path to calibrator catalog (auto-resolved if None), default is None
    search_radius_deg : float
        Maximum angular separation for calibrator match (used to verify calibrator is present), default is 1.0
    freq_GHz : float
        Observation frequency in GHz, default is 1.4
    window : int
        Number of fields around peak to include (only used if use_beam_weighting=True), default is 3
    min_pb : float or None
        Minimum primary beam response threshold (only used if use_beam_weighting=True), default is None
    calibrator_name : Optional[str]
        If specified, only match this calibrator (e.g., "0834+555"), default is None
    use_beam_weighting : bool
        If True, weight fields by primary beam response (Airy model).
        If False (default), return all fields for maximum SNR.
        Returns (field_sel_str, indices, weighted_flux_per_field, calibrator_info, peak_field_idx), default is False

    """
    # Use load_vla_catalog for SQLite support (preferred method per memory)
    if catalog_path is None:
        df = load_vla_catalog()  # Auto-resolves to SQLite if available
    else:
        # Check if it's SQLite or CSV
        if str(catalog_path).endswith(".sqlite3"):
            df = load_vla_catalog(catalog_path)
        else:
            # CSV format - use old function
            df = read_vla_calibrator_catalog(catalog_path)

    if df.empty:
        raise RuntimeError("Catalog contains no entries")

    ra_field, dec_field = _read_field_dirs(ms_path)
    if ra_field.size == 0:
        raise RuntimeError("MS has no FIELD rows")

    # Filter out time-dependent phase center fields (meridian_icrs_t*)
    # These are created during conversion but aren't separate observational fields
    # NOTE: When ALL fields are meridian phase centers (drift-scan), we must check ALL fields
    # because the calibrator could be in any field during the drift-scan, not just field 0.
    # When only some fields are meridian phase centers, we keep non-meridian fields plus field 0.
    with table(f"{ms_path}::FIELD") as tf:
        field_names = tf.getcol("NAME")

    # Check if ALL fields are meridian phase centers
    all_meridian = all(
        isinstance(name, str) and name.startswith("meridian_icrs_t") for name in field_names
    )

    if all_meridian:
        # In drift-scan mode: check ALL fields to find which one contains the calibrator
        valid_field_indices = np.arange(len(field_names))
    else:
        # Mixed mode: keep only fields that don't match meridian phase center pattern, EXCEPT field 0
        # Field 0 is kept as a fallback even if it's a meridian phase center
        valid_field_mask = np.array(
            [
                (i == 0) or not (isinstance(name, str) and name.startswith("meridian_icrs_t"))
                for i, name in enumerate(field_names)
            ]
        )
        valid_field_indices = np.where(valid_field_mask)[0]

    if len(valid_field_indices) == 0:
        raise RuntimeError(
            "No valid calibrator fields found (all fields are time-dependent phase centers)"
        )

    # Filter field coordinates to only valid fields
    ra_field = ra_field[valid_field_indices]
    dec_field = dec_field[valid_field_indices]

    field_coords = Angle(ra_field, unit=u.rad), Angle(dec_field, unit=u.rad)
    field_ra = field_coords[0].rad
    field_dec = field_coords[1].rad

    best: tuple[float, int, str, float, float, float] | None = None
    best_wflux: np.ndarray | None = None

    for name, row in df.iterrows():
        # If calibrator_name is specified, only match that exact calibrator
        if calibrator_name is not None:
            # Normalize names for comparison (handle J2000 prefix variations)
            target_name = calibrator_name.upper().replace("J", "").replace(" ", "")
            catalog_name = str(name).upper().replace("J", "").replace(" ", "")
            if target_name not in catalog_name and catalog_name not in target_name:
                continue  # Skip this calibrator, it's not the one we want

        try:
            # Handle both SQLite (ra_deg, dec_deg) and CSV (ra, dec) formats
            ra_deg = float(row.get("ra_deg", row.get("ra", row.get("RA", np.nan))))
            dec_deg = float(row.get("dec_deg", row.get("dec", row.get("DEC", np.nan))))
            # Flux: try flux_jy first (SQLite), then flux_20_cm (CSV), then flux
            flux_jy = None
            if "flux_jy" in row.index and pd.notna(row.get("flux_jy")):
                flux_jy = float(row["flux_jy"])
            elif "flux_20_cm" in row.index and pd.notna(row.get("flux_20_cm")):
                flux_mJy = float(row["flux_20_cm"])
                flux_jy = flux_mJy / 1000.0
            elif "flux" in row.index and pd.notna(row.get("flux")):
                # Assume flux is in mJy if > 1, otherwise Jy
                flux_val = float(row["flux"])
                flux_jy = flux_val / 1000.0 if flux_val > 1.0 else flux_val
            else:
                # Default flux if not available (use 1 Jy as fallback)
                flux_jy = 1.0
        except (KeyError, TypeError, ValueError):
            continue
        if (
            not np.isfinite(ra_deg)
            or not np.isfinite(dec_deg)
            or flux_jy is None
            or not np.isfinite(flux_jy)
        ):
            continue
        src_ra = Angle(ra_deg, unit=u.deg).rad
        src_dec = Angle(dec_deg, unit=u.deg).rad

        # Compute angular separation to each field and filter by search radius
        sep = np.rad2deg(
            np.arccos(
                np.clip(
                    np.sin(field_dec) * np.sin(src_dec)
                    + np.cos(field_dec) * np.cos(src_dec) * np.cos(field_ra - src_ra),
                    -1.0,
                    1.0,
                )
            )
        )
        if np.nanmin(sep) > float(search_radius_deg):
            continue

        # Find field closest to calibrator (minimum angular separation)
        peak_idx = int(np.nanargmin(sep))

        # Calculate flux weighting per field
        if use_beam_weighting:
            # Use Airy disk model to weight fields by beam response
            resp = np.array(
                [
                    primary_beam_response(
                        src_ra=ra,
                        src_dec=dec,
                        config=BeamConfig(
                            frequency_ghz=freq_GHz,
                            antenna_ra=src_ra,
                            antenna_dec=src_dec,
                            ms_path=ms_path,
                            field_id=int(valid_field_indices[i]),
                        ),
                    )
                    for i, (ra, dec) in enumerate(zip(field_ra, field_dec))
                ]
            )
            wflux = resp * flux_jy
            peak_val = float(wflux[peak_idx])
        else:
            # No beam weighting: uniform flux for all fields
            wflux = np.ones(len(field_ra)) * flux_jy
            peak_val = float(flux_jy)

        if best is None:
            best = (peak_val, peak_idx, name, ra_deg, dec_deg, flux_jy)
            best_wflux = wflux
        elif peak_val > best[0]:
            best = (peak_val, peak_idx, name, ra_deg, dec_deg, flux_jy)
            best_wflux = wflux

    if best is None or best_wflux is None:
        if calibrator_name is not None:
            raise RuntimeError(
                f"Specified calibrator '{calibrator_name}' not found within "
                f"{search_radius_deg}° search radius of any field in the MS"
            )
        raise RuntimeError("No calibrator candidates found within search radius")

    _, peak_idx_filtered, name, ra_deg, dec_deg, flux_jy = best
    wflux = best_wflux
    nfields = len(wflux)

    # Map filtered index back to original MS field index
    peak_idx_original = int(valid_field_indices[peak_idx_filtered])

    # If only one valid field exists, return it directly (no range needed)
    if nfields == 1:
        cal_info = (name, ra_deg, dec_deg, flux_jy)
        return (
            str(peak_idx_original),
            [peak_idx_original],
            wflux,
            cal_info,
            peak_idx_original,
        )

    # Default: Use all fields for maximum SNR in bandpass calibration
    # Beam response cancels out since we solve for per-antenna gains
    if not use_beam_weighting:
        # Return all valid fields
        start_original = int(valid_field_indices[0])
        end_original = int(valid_field_indices[-1])
        sel_str = (
            f"{start_original}~{end_original}"
            if start_original != end_original
            else f"{start_original}"
        )
        indices = [int(idx) for idx in valid_field_indices]
        cal_info = (name, ra_deg, dec_deg, flux_jy)
        return sel_str, indices, wflux, cal_info, peak_idx_original

    # Optional: Beam-weighted field selection
    if min_pb is not None and np.isfinite(min_pb):
        resp_peak = max(wflux[peak_idx_filtered] / max(flux_jy, 1e-12), 0.0)
        thr = float(min_pb) * max(resp_peak, 1e-12)
        start_filtered = peak_idx_filtered
        end_filtered = peak_idx_filtered
        while start_filtered - 1 >= 0 and (wflux[start_filtered - 1] / max(flux_jy, 1e-12)) >= thr:
            start_filtered -= 1
        while (
            end_filtered + 1 < len(wflux) and (wflux[end_filtered + 1] / max(flux_jy, 1e-12)) >= thr
        ):
            end_filtered += 1
    else:
        half = max(1, int(window)) // 2
        start_filtered = max(0, peak_idx_filtered - half)
        end_filtered = min(len(wflux) - 1, peak_idx_filtered + half)

    # Map filtered indices back to original MS field indices
    start_original = int(valid_field_indices[start_filtered])
    end_original = int(valid_field_indices[end_filtered])

    sel_str = (
        f"{start_original}~{end_original}"
        if start_original != end_original
        else f"{start_original}"
    )
    indices = [int(valid_field_indices[i]) for i in range(start_filtered, end_filtered + 1)]
    cal_info = (name, ra_deg, dec_deg, flux_jy)
    return sel_str, indices, wflux, cal_info, peak_idx_original
