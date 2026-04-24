"""Read phase-center metadata from DSA-110 UVH5/HDF5 without loading visibilities.

Key paths match dsa110_continuum.pointing.transit_selection.get_pointing_from_hdf5
and dsa110_continuum.pointing.utils.read_uvh5_dec_fast.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

# OVRO / DSA-110 (WGS84), same as dsa110_continuum simulation.harness
_OVRO_LON_DEG = -118.2825

_DEC_ROUND = 3


def read_phase_center_dec_deg(path: Path) -> float | None:
    """Return phase-center declination in degrees, or None if missing/unreadable."""
    try:
        import h5py
        import numpy as np

        with h5py.File(path, "r") as h:
            dec_rad = None
            if "Header/extra_keywords/phase_center_dec" in h:
                dec_rad = h["Header/extra_keywords/phase_center_dec"][()]
            elif "Header/phase_center_app_dec" in h:
                dec_rad = h["Header/phase_center_app_dec"][()]
            elif "Header/phase_center_dec" in h:
                dec_rad = h["Header/phase_center_dec"][()]
            if dec_rad is None:
                return None
            v = float(np.asarray(dec_rad).reshape(-1)[0])
            return float(np.degrees(v))
    except Exception:
        return None


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


def read_pointing_ra_dec_deg(path: Path) -> tuple[float | None, float | None]:
    """RA/Dec in degrees from headers; RA may be derived from HA + LST at median time."""
    try:
        import h5py
        import numpy as np
        from astropy import units as u
        from astropy.coordinates import EarthLocation
        from astropy.time import Time

        with h5py.File(path, "r") as h:
            ra_deg: float | None = None
            dec_deg: float | None = None

            if "Header/extra_keywords/phase_center_dec" in h:
                v = h["Header/extra_keywords/phase_center_dec"][()]
                dec_deg = float(np.degrees(float(np.asarray(v).reshape(-1)[0])))
            elif "Header/phase_center_app_dec" in h:
                v = h["Header/phase_center_app_dec"][()]
                dec_deg = float(np.degrees(float(np.asarray(v).reshape(-1)[0])))

            if "Header/extra_keywords/phase_center_ra" in h:
                v = h["Header/extra_keywords/phase_center_ra"][()]
                ra_deg = float(np.degrees(float(np.asarray(v).reshape(-1)[0])))
            elif "Header/phase_center_app_ra" in h:
                v = h["Header/phase_center_app_ra"][()]
                ra_deg = float(np.degrees(float(np.asarray(v).reshape(-1)[0])))

            if ra_deg is None and "Header/extra_keywords/ha_phase_center" in h and "Header/time_array" in h:
                ha_val = float(h["Header/extra_keywords/ha_phase_center"][()])
                ha_deg = _ha_to_deg(ha_val)
                time_array = h["Header/time_array"][()]
                if ha_deg is not None and time_array is not None and len(time_array) > 0:
                    mid_jd = float(np.median(time_array))
                    obs_time = Time(mid_jd, format="jd", scale="utc")
                    loc = EarthLocation.from_geodetic(
                        lon=_OVRO_LON_DEG * u.deg, lat=37.2339 * u.deg, height=1222.0 * u.m
                    )
                    lst = obs_time.sidereal_time("mean", longitude=loc.lon).to(u.deg).value
                    ra_deg = (lst - ha_deg) % 360.0

            return ra_deg, dec_deg
    except Exception:
        return None, None


def read_pointing_row(path: Path) -> dict[str, Any]:
    """One manifest row: filename, ISO UTC at median JD, ra_deg, dec_deg (nullable)."""
    from astropy.time import Time

    ra, dec = read_pointing_ra_dec_deg(path)
    t_mid: str | None = None
    mid_jd = read_time_median_jd(path)
    if mid_jd is not None:
        t = Time(mid_jd, format="jd", scale="utc")
        t_mid = t.isot + "Z"  # UTC, ISO-8601-like

    return {
        "filename": path.name,
        "t_mid_utc": t_mid,
        "ra_deg": ra,
        "dec_deg": dec,
    }
