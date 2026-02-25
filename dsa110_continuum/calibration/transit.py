"""
Transit time calculations for DSA-110 calibrators.

Provides utilities for computing meridian transit times of sources
at the DSA-110 site. Used for scheduling observations and finding
optimal calibrator observation windows.
"""

from __future__ import annotations

import warnings
from datetime import UTC
from functools import lru_cache

import astropy.units as u
from astropy.coordinates import Angle, EarthLocation
from astropy.time import Time
from astropy.utils import iers
from astropy.utils.exceptions import AstropyWarning
from astropy.utils.iers import IERSDegradedAccuracyWarning

from dsa110_contimg.common.utils.constants import DSA110_LOCATION

# Sidereal day in solar days
SIDEREAL_RATE = 1.002737909350795  # sidereal days per solar day


@lru_cache(maxsize=1)
def _configure_astropy_environment() -> None:
    iers.conf.auto_download = False
    iers.conf.iers_degraded_accuracy = "ignore"
    try:
        iers.conf.auto_max_age = None
    except Exception:
        pass
    warnings.filterwarnings(
        "ignore",
        message=".*IERS.*accuracy is degraded.*",
        category=IERSDegradedAccuracyWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="Tried to get polar motions.*",
        category=AstropyWarning,
    )
    warnings.filterwarnings(
        "ignore",
        category=AstropyWarning,
        module=r"astropy\.coordinates\.builtin_frames\.utils",
    )


def next_transit_time(
    ra_deg: float,
    start_time_mjd: float,
    location: EarthLocation = DSA110_LOCATION,
    max_iter: int = 4,
) -> Time:
    """Compute the next meridian transit (HA=0) after a given time.

    Uses iterative refinement to find when a source at the given RA
    crosses the local meridian.

    Parameters
    ----------
    ra_deg :
        Right ascension of the source in degrees
    start_time_mjd :
        Start time in MJD format
    location :
        Observatory location (default: DSA-110 site)
    max_iter :
        Number of iterations for convergence

    Returns
    -------
        astropy Time object for the next transit

    """
    _configure_astropy_environment()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=AstropyWarning)
        warnings.filterwarnings("ignore", category=IERSDegradedAccuracyWarning)
        ra_hours = Angle(ra_deg, u.deg).to(u.hourangle).value
        t = Time(start_time_mjd, format="mjd", scale="utc", location=location)

        for _ in range(max_iter):
            lst = t.sidereal_time("apparent").hour
            delta_lst = (ra_hours - lst + 12) % 24 - 12
            delta_utc_days = (delta_lst / 24.0) / SIDEREAL_RATE
            t = t + delta_utc_days * u.day

        if t < Time(start_time_mjd, format="mjd", scale="utc"):
            t = t + (1.0 / SIDEREAL_RATE) * u.day

        return t


def upcoming_transits(
    ra_deg: float,
    *,
    start_time: Time | None = None,
    n: int = 3,
    location: EarthLocation = DSA110_LOCATION,
) -> list[Time]:
    """Return the next n meridian transits (UTC) for a source.

    Parameters
    ----------
    ra_deg :
        Right ascension of the source in degrees
    start_time :
        Reference time
    n :
        Number of upcoming transits to return
    location :
        Observatory location

    Returns
    -------
    type
        List of astropy Time objects for upcoming transits

    """
    _configure_astropy_environment()
    t0 = start_time or Time.now()
    tnext = next_transit_time(ra_deg, t0.mjd, location=location)
    sidereal_day = (1.0 / SIDEREAL_RATE) * u.day

    out: list[Time] = []
    cur = tnext

    for _ in range(max(0, n)):
        out.append(cur)
        cur = cur + sidereal_day

    return out


def transit_times(
    ra_deg: float,
    start_time: Time,
    end_time: Time,
    location: EarthLocation = DSA110_LOCATION,
) -> list[Time]:
    """Calculate all transit times within a specified time window.

        This function returns the exact times when a source at the given RA
        crosses the meridian (LST = RA) between start_time and end_time.

        Transits occur once per sidereal day (~23h 56m 4s), so for a multi-day
        window, multiple transits will be returned.

    Parameters
    ----------
    ra_deg : float
        Right Ascension in degrees (0-360).
    start_time : Time
        Start of the time window (astropy Time).
    end_time : Time
        End of the time window (astropy Time).
    location : EarthLocation
        Observatory location (default: DSA-110 site).
        Default is DSA110_LOCATION.

    Returns
    -------
        list of Time
        List of astropy Time objects for all transits within the window,
        sorted chronologically. Empty list if no transits occur in window.

    Raises
    ------
        ValueError
        If end_time is before start_time.

    Examples
    --------
        >>> from astropy.time import Time
        >>> from dsa110_contimg.core.calibration.transit import transit_times
        >>> # Find all transits of 0834+555 (RA=128.47°) in January 2025
        >>> start = Time("2025-01-01T00:00:00")
        >>> end = Time("2025-01-31T23:59:59")
        >>> transits = transit_times(128.47, start, end)
        >>> print(f"Found {len(transits)} transits")
        Found 31 transits
    """
    _configure_astropy_environment()
    if end_time < start_time:
        raise ValueError(f"end_time ({end_time.iso}) must be after start_time ({start_time.iso})")

    sidereal_day = (1.0 / SIDEREAL_RATE) * u.day
    out: list[Time] = []

    # Find the first transit at or after start_time
    first_transit = next_transit_time(ra_deg, start_time.mjd, location=location)

    # If the first transit found is already past end_time, no transits in window
    if first_transit > end_time:
        return out

    # Collect all transits within the window
    current = first_transit
    while current <= end_time:
        out.append(current)
        current = current + sidereal_day

    return out


def observation_contains_transit(
    obs_start_time: Time,
    transit_time: Time,
    observation_duration: u.Quantity = 309 * u.s,
) -> bool:
    """Check if an observation's time window CONTAINS the peak transit time.

        Critical
    --------
        For DSA-110 drift-scan observations, the transit must occur
        WITHIN the observation window, not before or after it.

    Parameters
    ----------
    obs_start_time : Time
        Observation start time.
    transit_time : Time
        Peak transit time (must be within observation).
    observation_duration : u.Quantity, optional
        Length of observation. Default is 309 seconds (~5 min for DSA-110).

    Returns
    -------
        bool
        True if transit_time falls within [obs_start, obs_start + duration].

    Raises
    ------
        ValueError
        If transit occurs before or after the observation window.

    Examples
    --------
        Peak transit: 2025-10-02T12:20:25

        Valid (transit is INSIDE observation):
        - Obs start: 12:15:25, end: 12:20:34  (transit at end)
        - Obs start: 12:17:25, end: 12:22:34  (transit in middle)
        - Obs start: 12:20:00, end: 12:25:09  (transit at start)

        Invalid (transit is OUTSIDE observation):
        - Obs start: 12:14:25, end: 12:19:34  (transit AFTER obs ends)
        - Obs start: 12:21:25, end: 12:26:34  (transit BEFORE obs starts)
    """
    obs_end_time = obs_start_time + observation_duration

    # Check if transit is within observation window
    transit_in_window = obs_start_time <= transit_time <= obs_end_time

    if not transit_in_window:
        # Calculate how far off we are
        if transit_time < obs_start_time:
            offset = (obs_start_time - transit_time).to(u.min)
            raise ValueError(
                f"Transit time {transit_time.iso} occurs {offset:.1f} BEFORE "
                f"observation starts at {obs_start_time.iso}. "
                f"You need an EARLIER observation that starts before the transit."
            )
        else:  # transit_time > obs_end_time
            offset = (transit_time - obs_end_time).to(u.min)
            raise ValueError(
                f"Transit time {transit_time.iso} occurs {offset:.1f} AFTER "
                f"observation ends at {obs_end_time.iso}. "
                f"You need a LATER observation that includes the transit time. "
                f"(Observation: {obs_start_time.iso} to {obs_end_time.iso})"
            )

    return True


def find_transits_for_source(
    db_path: str,
    ra_deg: float,
    dec_deg: float,
    ra_tolerance_deg: float = 2.0,
    dec_tolerance_deg: float = 2.0,
    observation_duration: u.Quantity = 309 * u.s,
) -> list[dict]:
    """Find all observations that contain transits of a source.

        This is the recommended function for finding source transits in the database.
        It queries observations near the source coordinates, computes transit times,
        and returns only those observations where the transit peak falls within
        the observation window.

    Parameters
    ----------
    db_path : str
        Path to pipeline.sqlite3 database.
    ra_deg : float
        Source right ascension in degrees.
    dec_deg : float
        Source declination in degrees.
    ra_tolerance_deg : float
        RA search tolerance. Default is 2°.
    dec_tolerance_deg : float
        Dec search tolerance. Default is 2°.
    observation_duration : u.Quantity
        Observation length. Default is 309s for DSA-110.

    Returns
    -------
        list of dict
        List of dicts with keys:
        - group_id: Observation group ID (normalized timestamp, ISO format)
        - ra_deg: Observation RA
        - dec_deg: Observation Dec
        - transit_time_iso: Transit peak time (ISO format)
        - delta_minutes: Time from obs start to transit peak

    Examples
    --------
        >>> from dsa110_contimg.core.calibration.transit import find_transits_for_source
        >>> results = find_transits_for_source(
        ...     db_path=os.environ.get('PIPELINE_DB', '/data/dsa110-contimg/state/db/pipeline.sqlite3'),
        ...     ra_deg=128.6083,  # 0834+555
        ...     dec_deg=55.5000
        ... )
        >>> print(f'Found {len(results)} observations with transits')
    """
    import sqlite3

    conn = sqlite3.connect(db_path, timeout=30)
    cursor = conn.cursor()

    # Query complete observations (16 subbands) near the source coordinates
    cursor.execute(
        """
        SELECT group_id, ra_deg, dec_deg, COUNT(*) as file_count
        FROM hdf5_files
        WHERE ra_deg IS NOT NULL
          AND dec_deg IS NOT NULL
          AND ra_deg BETWEEN ? AND ?
          AND dec_deg BETWEEN ? AND ?
        GROUP BY group_id, ra_deg, dec_deg
        HAVING file_count = 16
        ORDER BY group_id
    """,
        (
            ra_deg - ra_tolerance_deg,
            ra_deg + ra_tolerance_deg,
            dec_deg - dec_tolerance_deg,
            dec_deg + dec_tolerance_deg,
        ),
    )

    candidate_obs = cursor.fetchall()
    conn.close()

    # For each complete observation, check if it contains the transit
    matches = []
    for group_id, obs_ra, obs_dec, _file_count in candidate_obs:
        obs_start = Time(group_id, scale="utc")

        # Calculate the transit time for this source
        transit_time = next_transit_time(ra_deg, obs_start.mjd)

        # Check if transit falls within this observation window
        try:
            if observation_contains_transit(obs_start, transit_time, observation_duration):
                delta = (transit_time - obs_start).to(u.min).value
                matches.append(
                    {
                        "group_id": group_id,
                        "ra_deg": obs_ra,
                        "dec_deg": obs_dec,
                        "transit_time_iso": transit_time.iso,
                        "delta_minutes": delta,
                    }
                )
        except ValueError:
            continue

    return matches


def find_observations_containing_transit(
    observations: list[tuple[str, float, float]],
    transit_time: Time,
    observation_duration: u.Quantity = 309 * u.s,
) -> list[tuple[str, float, float]]:
    """Find all observations whose time window CONTAINS the peak transit.

    This function filters a pre-existing list of observations. For most use cases,
    prefer find_transits_for_source() which queries the database directly.

    Parameters
    ----------
    observations : List[Tuple[str, float, float]]
        List of (obs_id, start_mjd, end_mjd) tuples
    transit_time : Time
        Peak transit time
    observation_duration : u.Quantity
        Observation length (default: 309s for DSA-110)

    Returns
    -------
    list
        List of observations that contain the transit, sorted by how close
        the transit is to the observation midpoint (best first)

    Examples
    --------
    >>> # Peak transit at 12:20:25
    >>> obs = [
    ...     ("obs1", mjd(12:14:25), mjd(12:19:34)),  # Ends before peak
    ...     ("obs2", mjd(12:17:25), mjd(12:22:34)),  # Contains peak
    ...     ("obs3", mjd(12:21:25), mjd(12:26:34)),  # Starts after peak
    ... ]
    >>> result = find_observations_containing_transit(obs, transit_time)
    >>> # Returns only obs2
    """
    valid_obs = []

    for obs_id, start_mjd, end_mjd in observations:
        obs_start = Time(start_mjd, format="mjd")

        # Check if this observation contains the transit
        try:
            if observation_contains_transit(obs_start, transit_time, observation_duration):
                # Compute distance from transit to observation midpoint
                mid_mjd = 0.5 * (start_mjd + end_mjd)
                delta_min = abs((Time(mid_mjd, format="mjd") - transit_time).to(u.min).value)
                valid_obs.append((obs_id, start_mjd, end_mjd, delta_min))
        except ValueError:
            # observation_contains_transit raises ValueError if transit not in window
            continue

    # Sort by delta_min (closest to midpoint first)
    valid_obs.sort(key=lambda x: x[3])

    # Return without delta_min for backward compatibility
    return [(obs_id, start_mjd, end_mjd) for obs_id, start_mjd, end_mjd, _ in valid_obs]


def transit_time_for_local_time(
    ra_deg: float,
    local_hour: int,
    local_minute: int = 0,
    date_str: str | None = None,
    location: EarthLocation = DSA110_LOCATION,
) -> Time | None:
    """Find when a source with given RA transits at a specific local time.

    This is useful for finding calibrators that transit during specific
    observing windows (e.g., "which calibrators transit around 3 AM local?").

    Parameters
    ----------
    ra_deg :
        Right ascension of source in degrees
    local_hour :
        Target local hour (0-23)
    local_minute :
        Target local minute (0-59)
    date_str :
        Date in YYYY-MM-DD format (default: today)
    location :
        Observatory location

    Returns
    -------
        Transit time if one occurs near the target time, or None

    """
    from datetime import datetime

    if date_str:
        base_date = datetime.strptime(date_str, "%Y-%m-%d")
    else:
        base_date = datetime.now(UTC)

    # Create target time (assume Pacific time, -8 hours from UTC in winter)
    # DSA-110 is in California
    target_utc_hour = (local_hour + 8) % 24  # Approximate UTC offset

    target_time = Time(
        f"{base_date.year}-{base_date.month:02d}-{base_date.day:02d}T{target_utc_hour:02d}:{local_minute:02d}:00",
        scale="utc",
    )

    # Find the nearest transit
    transit = next_transit_time(ra_deg, target_time.mjd - 0.5, location=location)

    # Check if it's within a few hours of target
    delta_hours = abs((transit - target_time).to(u.hour).value)
    if delta_hours < 12:
        return transit

    return None
