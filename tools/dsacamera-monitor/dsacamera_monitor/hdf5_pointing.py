"""Read phase-center metadata from DSA-110 UVH5/HDF5 without loading visibilities."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

# OVRO / DSA-110 (WGS84), same as dsa110_continuum simulation.harness
_OVRO_LON_DEG = -118.2825
_OVRO_LAT_DEG = 37.2339
_OVRO_ALT_M = 1222.0
DEC_ROUND_DIGITS = 3

try:
    from astropy import units as u
    from astropy.coordinates import EarthLocation
    from astropy.time import Time
    from astropy.utils import iers

    # Keep metadata scans independent of network/IERS download state.
    iers.conf.auto_download = False
    _OVRO_LOCATION = EarthLocation.from_geodetic(
        lon=_OVRO_LON_DEG * u.deg, lat=_OVRO_LAT_DEG * u.deg, height=_OVRO_ALT_M * u.m
    )
except Exception:  # pragma: no cover - astropy import/runtime failure handled in scan path
    u = None
    Time = None
    _OVRO_LOCATION = None


def read_phase_center_dec_deg(path: Path) -> float | None:
    """Return phase-center declination in degrees, or None if missing/unreadable."""
    meta = read_pointing_metadata(path)
    return meta["dec_deg"] if meta["dec_status"] == "ok" else None


def _ha_to_deg(ha_val: float) -> float | None:
    import numpy as np

    if abs(ha_val) <= 2 * math.pi + 1e-6:
        return float(np.degrees(ha_val))
    if abs(ha_val) <= 24.0:
        return float(ha_val * 15.0)
    if abs(ha_val) <= 360.0:
        return float(ha_val)
    return None


def read_time_median_jd(path: Path) -> float | None:
    """Return median JD from ``Header/time_array`` if present."""
    try:
        import h5py
        import numpy as np

        with h5py.File(path, "r") as h:
            if "Header/time_array" not in h:
                return None
            ta = h["Header/time_array"][()]
            if ta is None or len(ta) == 0:
                return None
            return float(np.median(ta))
    except Exception:
        return None


def _extract_float(h5_obj: Any, key: str) -> float | None:
    import numpy as np

    if key not in h5_obj:
        return None
    return float(np.asarray(h5_obj[key][()]).reshape(-1)[0])


def read_pointing_metadata(path: Path) -> dict[str, Any]:
    """Read Dec/RA/mid-time from one HDF5 open.

    Returns status-bearing metadata so callers can distinguish missing keys from read failures.
    """
    meta: dict[str, Any] = {
        "filename": path.name,
        "t_mid_utc": None,
        "ra_deg": None,
        "dec_deg": None,
        "dec_status": "missing",  # one of: ok, missing, read_failed
        "pointing_status": "missing",  # one of: ok, missing, read_failed
        "error": None,
    }
    try:
        import h5py
        import numpy as np
    except (ImportError, ModuleNotFoundError) as exc:
        meta["dec_status"] = "read_failed"
        meta["pointing_status"] = "read_failed"
        meta["error"] = f"{type(exc).__name__}: {exc}"
        return meta

    try:
        with h5py.File(path, "r") as h:
            try:
                dec_rad = _extract_float(h, "Header/extra_keywords/phase_center_dec")
                if dec_rad is None:
                    dec_rad = _extract_float(h, "Header/phase_center_app_dec")
                if dec_rad is None:
                    dec_rad = _extract_float(h, "Header/phase_center_dec")
                if dec_rad is not None:
                    meta["dec_deg"] = float(np.degrees(dec_rad))
                    meta["dec_status"] = "ok"
            except (KeyError, ValueError, TypeError) as exc:
                meta["dec_status"] = "read_failed"
                meta["error"] = f"{type(exc).__name__}: {exc}"

            try:
                # Direct RA keys first.
                ra_rad = _extract_float(h, "Header/extra_keywords/phase_center_ra")
                if ra_rad is None:
                    ra_rad = _extract_float(h, "Header/phase_center_app_ra")
                if ra_rad is None:
                    ra_rad = _extract_float(h, "Header/phase_center_ra")
                if ra_rad is not None:
                    meta["ra_deg"] = float(np.degrees(ra_rad))

                time_array = h["Header/time_array"][()] if "Header/time_array" in h else None
                if time_array is not None and len(time_array) > 0 and Time is not None:
                    mid_jd = float(np.median(time_array))
                    obs_time = Time(mid_jd, format="jd", scale="utc")
                    meta["t_mid_utc"] = obs_time.isot + "Z"

                # HA fallback if direct RA is missing.
                if (
                    meta["ra_deg"] is None
                    and "Header/extra_keywords/ha_phase_center" in h
                    and time_array is not None
                    and len(time_array) > 0
                ):
                    ha_val = float(h["Header/extra_keywords/ha_phase_center"][()])
                    ha_deg = _ha_to_deg(ha_val)
                    if ha_deg is not None and Time is not None and _OVRO_LOCATION is not None:
                        mid_jd = float(np.median(time_array))
                        obs_time = Time(mid_jd, format="jd", scale="utc")
                        lst = obs_time.sidereal_time("mean", longitude=_OVRO_LOCATION.lon).to(u.deg).value
                        meta["ra_deg"] = (lst - ha_deg) % 360.0

                if meta["ra_deg"] is not None or meta["t_mid_utc"] is not None:
                    meta["pointing_status"] = "ok"
                elif "Header/time_array" in h or "Header/extra_keywords/ha_phase_center" in h:
                    meta["pointing_status"] = "read_failed"
                    if meta["error"] is None:
                        meta["error"] = "Could not derive RA/time from available header keys"
            except (KeyError, ValueError, TypeError, RuntimeError) as exc:
                meta["pointing_status"] = "read_failed"
                if meta["error"] is None:
                    meta["error"] = f"{type(exc).__name__}: {exc}"
    except OSError as exc:
        meta["dec_status"] = "read_failed"
        meta["pointing_status"] = "read_failed"
        meta["error"] = f"{type(exc).__name__}: {exc}"

    return meta


def read_pointing_ra_dec_deg(path: Path) -> tuple[float | None, float | None]:
    """RA/Dec in degrees from headers; RA may be derived from HA + LST at median time."""
    meta = read_pointing_metadata(path)
    return meta["ra_deg"], meta["dec_deg"]


def read_pointing_row(path: Path) -> dict[str, Any]:
    """One manifest row: filename, ISO UTC at median JD, ra_deg, dec_deg (nullable)."""
    meta = read_pointing_metadata(path)
    return {
        "filename": path.name,
        "t_mid_utc": meta["t_mid_utc"],
        "ra_deg": meta["ra_deg"],
        "dec_deg": meta["dec_deg"],
    }
