# pylint: disable=no-member  # astropy.units uses dynamic attributes (deg, hourangle, etc.)

import astropy.units as u
from astropy.coordinates import Angle, EarthLocation
from astropy.time import Time

# Use DSA-110 coordinates from constants.py (single source of truth)
from dsa110_contimg.common.utils.constants import DSA110_LOCATION

DSA110_LON_DEG = DSA110_LOCATION.lon.to(u.deg).value
DSA110_LAT_DEG = DSA110_LOCATION.lat.to(u.deg).value
DSA110_ALT_M = DSA110_LOCATION.height.to(u.m).value

SIDEREAL_RATE = 1.002737909350795  # sidereal days per solar day


def next_transit_time(
    ra_deg: float,
    start_time_mjd: float,
    location: EarthLocation = DSA110_LOCATION,
    max_iter: int = 4,
) -> Time:
    """Compute next transit (HA=0) after start_time_mjd for a source with RA=ra_deg."""
    ra_hours = Angle(ra_deg, u.deg).to(u.hourangle).value
    t = Time(start_time_mjd, format="mjd", scale="utc", location=location)
    for _ in range(max_iter):
        lst = t.sidereal_time("apparent").hour
        delta_lst = (ra_hours - lst + 12) % 24 - 12  # wrap to [-12, +12]
        delta_utc_days = (delta_lst / 24.0) / SIDEREAL_RATE
        t = t + delta_utc_days * u.day
    if t < Time(start_time_mjd, format="mjd", scale="utc"):
        t = t + (1.0 / SIDEREAL_RATE) * u.day
    return t


# previous_transits removed - use find_transits_for_source() from transit.py instead


def cal_in_datetime(
    dt_start_iso: str,
    transit_time: Time,
    duration: u.Quantity = 5 * u.min,
    filelength: u.Quantity = 15 * u.min,
) -> bool:
    """Return True if a file starting at dt_start_iso overlaps the desired window around transit.

    A file of length `filelength` starting at `dt_start_iso` overlaps a window of +/- duration around `transit_time`.
    """
    mjd0 = Time(dt_start_iso, scale="utc").mjd
    mjd1 = (Time(dt_start_iso, scale="utc") + filelength).mjd
    window0 = (transit_time - duration).mjd
    window1 = (transit_time + duration).mjd
    return (mjd0 <= window1) and (mjd1 >= window0)
