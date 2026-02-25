"""Utilities for selecting calibrator transits from the pipeline database."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time

from dsa110_contimg.common.utils.constants import DSA110_LOCATION
from dsa110_contimg.core.pointing.monitor import predict_calibrator_transit_by_coords

logger = logging.getLogger(__name__)

# Default tolerances and windows (minutes/degrees)
POINTING_TOLERANCE_DEG = 2.0
TRANSIT_TIME_TOLERANCE_MIN = 30.0
DEFAULT_TRANSIT_WINDOW_MIN = 40.0
DEFAULT_LOOKBACK_DAYS = 60


@dataclass
class TransitObservation:
    """A complete subband group observation near a calibrator transit."""

    group_id: str
    timestamp_iso: str
    file_count: int
    altitude_deg: float
    azimuth_deg: float
    file_paths: list[str]
    pointing_ra_deg: Optional[float] = None
    pointing_dec_deg: Optional[float] = None
    pointing_sep_deg: Optional[float] = None
    pointing_validated: bool = False

    @property
    def is_complete(self) -> bool:
        return self.file_count == 16

    @property
    def is_near_meridian(self) -> bool:
        """Check if calibrator was near meridian (az ~ 0 or 360 for northern sources)."""
        return self.azimuth_deg < 30 or self.azimuth_deg > 330


def _parse_iso(timestamp: str) -> datetime:
    if timestamp.endswith("Z"):
        timestamp = timestamp[:-1]
    return datetime.fromisoformat(timestamp)


def get_pointing_from_hdf5(hdf5_path: str | Path) -> tuple[Optional[float], Optional[float]]:
    """Extract the pointing RA/Dec from an HDF5 file."""
    import h5py
    import numpy as np

    path = Path(hdf5_path)
    if not path.exists():
        return None, None

    try:
        with h5py.File(path, "r") as h:
            ra_deg = None
            dec_deg = None

            # Standard locations for phase center Dec
            if "Header/extra_keywords/phase_center_dec" in h:
                dec_rad = h["Header/extra_keywords/phase_center_dec"][()]
                dec_deg = float(np.degrees(dec_rad))
            if "Header/phase_center_app_dec" in h:
                dec_rad = h["Header/phase_center_app_dec"][()]
                dec_deg = float(np.degrees(dec_rad))

            # Read RA directly if available
            if "Header/extra_keywords/phase_center_ra" in h:
                ra_rad = h["Header/extra_keywords/phase_center_ra"][()]
                ra_deg = float(np.degrees(ra_rad))
            elif "Header/phase_center_app_ra" in h:
                ra_rad = h["Header/phase_center_app_ra"][()]
                ra_deg = float(np.degrees(ra_rad))

            # Fallback: derive RA from LST - HA if stored
            if ra_deg is None and "Header/extra_keywords/ha_phase_center" in h and "Header/time_array" in h:
                ha_val = float(h["Header/extra_keywords/ha_phase_center"][()])
                if abs(ha_val) <= 2 * np.pi + 1e-6:
                    ha_deg = float(np.degrees(ha_val))
                elif abs(ha_val) <= 24.0:
                    ha_deg = float(ha_val * 15.0)
                elif abs(ha_val) <= 360.0:
                    ha_deg = ha_val
                else:
                    ha_deg = None

                time_array = h["Header/time_array"][()]
                if ha_deg is not None and time_array is not None and len(time_array) > 0:
                    mid_jd = float(np.median(time_array))
                    obs_time = Time(mid_jd, format="jd", scale="utc")
                    lst = obs_time.sidereal_time("mean", longitude=DSA110_LOCATION.lon).to(u.deg).value
                    ra_deg = (lst - ha_deg) % 360.0

            return ra_deg, dec_deg
    except Exception as e:
        logger.debug("Could not read pointing from %s: %s", path, e)
    return None, None


def validate_pointing_matches_calibrator(
    file_paths: list[str],
    calibrator_ra_deg: float,
    calibrator_dec_deg: float,
    tolerance_deg: float = POINTING_TOLERANCE_DEG,
) -> tuple[bool, Optional[float], Optional[float], Optional[float]]:
    """Validate that an observation's pointing matches the expected calibrator."""
    import os

    existing_files = [p for p in file_paths if os.path.exists(p)]
    if not existing_files:
        logger.warning(
            "None of %d HDF5 files exist on disk - data may have been moved/deleted",
            len(file_paths),
        )
        return False, None, None, None

    calibrator_coord = SkyCoord(ra=calibrator_ra_deg * u.deg, dec=calibrator_dec_deg * u.deg)
    for path in existing_files[:3]:
        pointing_ra, pointing_dec = get_pointing_from_hdf5(path)
        if pointing_dec is None:
            continue

        if pointing_ra is not None:
            pointing_coord = SkyCoord(ra=pointing_ra * u.deg, dec=pointing_dec * u.deg)
            separation = pointing_coord.separation(calibrator_coord).deg
            is_valid = separation <= tolerance_deg
            if not is_valid:
                logger.warning(
                    "Pointing mismatch: RA=%.2f°, Dec=%.2f° vs expected RA=%.2f°, Dec=%.2f° "
                    "(sep=%.2f°, tolerance=%.2f°)",
                    pointing_ra,
                    pointing_dec,
                    calibrator_ra_deg,
                    calibrator_dec_deg,
                    separation,
                    tolerance_deg,
                )
            return is_valid, pointing_dec, pointing_ra, separation

        # Fallback: Dec-only validation if RA missing
        offset = abs(pointing_dec - calibrator_dec_deg)
        is_valid = offset <= tolerance_deg
        if not is_valid:
            logger.warning(
                "Pointing mismatch (Dec-only): Dec=%.2f° vs expected %.2f° "
                "(offset=%.2f°, tolerance=%.2f°). RA unavailable in HDF5.",
                pointing_dec,
                calibrator_dec_deg,
                offset,
                tolerance_deg,
            )
        else:
            logger.warning(
                "Pointing validation used Dec-only (RA unavailable in HDF5). "
                "Dec=%.2f°, expected=%.2f°.",
                pointing_dec,
                calibrator_dec_deg,
            )
        return is_valid, pointing_dec, None, None

    logger.warning("Could not extract pointing from any of %d existing files", len(existing_files))
    return False, None, None, None


def _merge_split_groups(
    cur: sqlite3.Cursor,
    start_time: datetime,
    end_time: datetime,
    tolerance_seconds: int = 5,
) -> list[dict]:
    """Find and merge split subband groups that should be one observation."""
    start_iso = start_time.isoformat(timespec="seconds")
    end_iso = end_time.isoformat(timespec="seconds")

    cur.execute(
        """
        SELECT path, group_id, subband_code, timestamp_iso
        FROM hdf5_files
        WHERE timestamp_iso BETWEEN ? AND ?
        ORDER BY timestamp_iso, subband_code
        """,
        (start_iso, end_iso),
    )
    rows = cur.fetchall()

    if not rows:
        return []

    from collections import defaultdict

    groups_by_id = defaultdict(list)
    group_timestamps: dict[str, str] = {}

    for path, group_id, subband_code, timestamp_iso in rows:
        groups_by_id[group_id].append((path, subband_code))
        if group_id not in group_timestamps:
            group_timestamps[group_id] = timestamp_iso

    sorted_group_ids = sorted(groups_by_id.keys(), key=lambda g: group_timestamps[g])

    merged_groups = []
    current_merge = None

    for group_id in sorted_group_ids:
        ts_iso = group_timestamps[group_id]
        files = groups_by_id[group_id]

        if current_merge is None:
            current_merge = {
                "group_id": group_id,
                "timestamp_iso": ts_iso,
                "file_paths": [f[0] for f in files],
                "subbands": set(f[1] for f in files),
                "all_group_ids": [group_id],
                "last_timestamp": _parse_iso(ts_iso),
            }
        else:
            this_time = _parse_iso(ts_iso)
            time_diff = abs((this_time - current_merge["last_timestamp"]).total_seconds())

            if time_diff <= tolerance_seconds:
                current_merge["file_paths"].extend([f[0] for f in files])
                current_merge["subbands"].update(f[1] for f in files)
                current_merge["all_group_ids"].append(group_id)
                current_merge["last_timestamp"] = max(current_merge["last_timestamp"], this_time)
            else:
                current_merge["file_count"] = len(current_merge["file_paths"])
                del current_merge["last_timestamp"]
                del current_merge["subbands"]
                merged_groups.append(current_merge)

                current_merge = {
                    "group_id": group_id,
                    "timestamp_iso": ts_iso,
                    "file_paths": [f[0] for f in files],
                    "subbands": set(f[1] for f in files),
                    "all_group_ids": [group_id],
                    "last_timestamp": this_time,
                }

    if current_merge is not None:
        current_merge["file_count"] = len(current_merge["file_paths"])
        del current_merge["last_timestamp"]
        del current_merge["subbands"]
        merged_groups.append(current_merge)

    return merged_groups


def find_transit_observations(
    db_path: Path,
    calibrator_name: str,
    calibrator_ra_deg: float,
    calibrator_dec_deg: float,
    num_transits: int = 5,
    min_subbands: int = 16,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    transit_window_min: float = DEFAULT_TRANSIT_WINDOW_MIN,
    transit_tolerance_min: float = TRANSIT_TIME_TOLERANCE_MIN,
) -> list[TransitObservation]:
    """Find the peak transit observation for each day with calibrator data."""
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")

    start_date = (datetime.utcnow().date() - timedelta(days=lookback_days - 1)).isoformat()

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT DISTINCT date(timestamp_iso) as obs_date
            FROM hdf5_files
            WHERE date(timestamp_iso) >= ?
            ORDER BY obs_date DESC
            """,
            (start_date,),
        )
        dates = [r[0] for r in cur.fetchall()]

        results: list[TransitObservation] = []
        cal_coord = SkyCoord(
            ra=calibrator_ra_deg * u.deg, dec=calibrator_dec_deg * u.deg, frame="icrs"
        )

        for obs_date in dates:
            pred = None
            try:
                pred = predict_calibrator_transit_by_coords(
                    ra_deg=calibrator_ra_deg,
                    dec_deg=calibrator_dec_deg,
                    from_time=datetime.fromisoformat(f"{obs_date}T00:00:00"),
                    name=calibrator_name,
                )
            except Exception as e:
                logger.warning("Could not predict transit for %s: %s", obs_date, e)

            if pred is None:
                continue

            window_half = transit_window_min / 2.0
            window_start = pred.transit_utc - timedelta(minutes=window_half)
            window_end = pred.transit_utc + timedelta(minutes=window_half)

            merged_groups = _merge_split_groups(cur, window_start, window_end)
            if not merged_groups:
                continue

            best_for_day = None
            best_altitude = -90.0

            for mg in merged_groups:
                if mg["file_count"] < min_subbands:
                    continue

                try:
                    obs_time = Time(mg["timestamp_iso"], format="isot")
                    time_diff = abs(obs_time.unix - Time(pred.transit_utc).unix)
                    if time_diff > transit_tolerance_min * 60:
                        continue

                    altaz = cal_coord.transform_to(
                        AltAz(obstime=obs_time, location=DSA110_LOCATION)
                    )
                    altitude = altaz.alt.deg

                    if altitude > best_altitude:
                        best_altitude = altitude
                        best_for_day = TransitObservation(
                            group_id=mg["group_id"],
                            timestamp_iso=mg["timestamp_iso"],
                            file_count=mg["file_count"],
                            altitude_deg=altitude,
                            azimuth_deg=altaz.az.deg,
                            file_paths=mg["file_paths"],
                        )

                except Exception as e:
                    logger.warning("Error processing group %s: %s", mg.get("group_id"), e)
                    continue

            if best_for_day is not None:
                is_valid, actual_dec, actual_ra, separation = validate_pointing_matches_calibrator(
                    best_for_day.file_paths,
                    calibrator_ra_deg=calibrator_ra_deg,
                    calibrator_dec_deg=calibrator_dec_deg,
                )
                best_for_day.pointing_ra_deg = actual_ra
                best_for_day.pointing_dec_deg = actual_dec
                best_for_day.pointing_sep_deg = separation
                best_for_day.pointing_validated = is_valid

                if is_valid:
                    results.append(best_for_day)
                else:
                    dec_str = f"{actual_dec:.2f}" if actual_dec is not None else "unknown"
                    ra_str = f"{actual_ra:.2f}" if actual_ra is not None else "unknown"
                    sep_str = f"{separation:.2f}°" if separation is not None else "n/a"
                    logger.info(
                        "Skipping %s %s: pointing RA=%s°, Dec=%s° (sep=%s) does not match %s "
                        "(expected RA=%.2f°, Dec=%.2f°)",
                        obs_date,
                        best_for_day.group_id,
                        ra_str,
                        dec_str,
                        sep_str,
                        calibrator_name,
                        calibrator_ra_deg,
                        calibrator_dec_deg,
                    )

            if len(results) >= num_transits:
                break

        results.sort(key=lambda x: -x.altitude_deg)
        return results[:num_transits]
    finally:
        conn.close()


__all__ = [
    "TransitObservation",
    "get_pointing_from_hdf5",
    "validate_pointing_matches_calibrator",
    "find_transit_observations",
]
