"""
Precomputation Module - Proactively prepares resources when telescope pointing changes.

This module provides utilities for:
1. Detecting declination changes from incoming HDF5 metadata
2. Precomputing bandpass calibrator selection for new declinations
3. Triggering background catalog strip database builds
4. Caching transit predictions for upcoming calibrators

The goal is to reduce pipeline latency by doing expensive work before it's needed.

Architecture:
- PointingTracker: Monitors Dec changes and triggers precomputation
- CalibrationPreloader: Precomputes calibrator selection and transit times
- CatalogPreloader: Background builds catalog strip databases

Usage:
    from dsa110_contimg.core.calibration.precompute import PointingTracker, get_pointing_tracker

    tracker = PointingTracker()

    # Call when new HDF5 file arrives
    change = tracker.check_pointing_change(hdf5_path)
    if change:
        print(f"Pointing changed to Dec={change.new_dec_deg:.2f}°")
        print(f"Precomputed calibrator: {change.precomputed_calibrator}")
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

# Import shared pointing utilities
from dsa110_contimg.common.utils.env_utils import get_env_path
from dsa110_contimg.core.pointing.utils import (
    detect_dec_change,
    read_uvh5_dec_fast,
)

logger = logging.getLogger(__name__)

# Configuration
# Note: This threshold is higher than DEFAULT_TOLERANCE (0.1°) because it's used
# to trigger expensive precomputation, not just detect pointing changes.
DEFAULT_DEC_CHANGE_THRESHOLD = 1.0  # degrees - threshold for triggering precomputation
DEFAULT_CATALOG_TYPES = ["nvss", "first", "vlass"]  # Catalogs to preload
DEFAULT_TRANSIT_HORIZON_HOURS = 48.0  # How far ahead to precompute transits
DEFAULT_MAX_DEC_SEPARATION = 1.5  # degrees - matches DSA-110 primary beam (~3° diameter)
_CONTIMG_BASE = str(get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg"))
VLA_CALIBRATOR_DB = Path(_CONTIMG_BASE) / "state" / "catalogs" / "vla_calibrators.sqlite3"
MIN_FLUX_JY_20CM = 1.0  # Minimum 20cm flux for VLA catalog fallback search


@dataclass
class PointingChange:
    """Represents a detected pointing change."""

    old_dec_deg: float | None
    new_dec_deg: float
    timestamp: datetime
    source_file: str

    # Precomputed resources
    precomputed_calibrator: str | None = None
    calibrator_transit_utc: datetime | None = None
    calibrator_dec_deg: float | None = None
    catalog_build_started: bool = False
    catalog_types_queued: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON/logging."""
        return {
            "old_dec_deg": self.old_dec_deg,
            "new_dec_deg": self.new_dec_deg,
            "timestamp": self.timestamp.isoformat() + "Z",
            "source_file": self.source_file,
            "precomputed_calibrator": self.precomputed_calibrator,
            "calibrator_transit_utc": (
                self.calibrator_transit_utc.isoformat() + "Z"
                if self.calibrator_transit_utc
                else None
            ),
            "calibrator_dec_deg": self.calibrator_dec_deg,
            "catalog_build_started": self.catalog_build_started,
            "catalog_types_queued": self.catalog_types_queued,
        }


@dataclass
class CalibratorPrediction:
    """Precomputed calibrator transit prediction."""

    name: str
    ra_deg: float
    dec_deg: float
    transit_utc: datetime
    time_to_transit_sec: float
    dec_separation_deg: float  # How close to telescope pointing
    expected_flux_jy: float | None = None
    priority_score: float = 0.0  # Higher = better choice
    source: str = "registered"  # "registered" or "vla_catalog"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ra_deg": self.ra_deg,
            "dec_deg": self.dec_deg,
            "transit_utc": self.transit_utc.isoformat() + "Z",
            "time_to_transit_sec": round(self.time_to_transit_sec, 1),
            "dec_separation_deg": round(self.dec_separation_deg, 3),
            "expected_flux_jy": self.expected_flux_jy,
            "source": self.source,
            "priority_score": round(self.priority_score, 2),
        }


# Note: read_uvh5_dec_fast is now imported from dsa110_contimg.core.pointing.utils


def read_uvh5_metadata_fast(path: Path) -> dict:
    """Read basic metadata from UVH5 file (fast path).

    Parameters
    ----------
    path : Path
        Path to UVH5 file

    """
    result: dict[str, float | str | None] = {
        "dec_deg": None,
        "ra_deg": None,
        "mid_time_mjd": None,
        "file_path": str(path),
    }

    try:
        import h5py

        with h5py.File(path, "r") as f:
            # Declination
            if "Header/extra_keywords/phase_center_dec" in f:
                dec_rad = f["Header/extra_keywords/phase_center_dec"][()]
                result["dec_deg"] = float(np.degrees(dec_rad))

            # RA (if stored)
            if "Header/extra_keywords/phase_center_ra" in f:
                ra_rad = f["Header/extra_keywords/phase_center_ra"][()]
                result["ra_deg"] = float(np.degrees(ra_rad))

            # Time (JD array)
            if "Header/time_array" in f:
                times = f["Header/time_array"][:]
                mid_jd = (times.min() + times.max()) / 2
                result["mid_time_mjd"] = mid_jd - 2400000.5
    except Exception as e:
        logger.debug(f"Error reading metadata from {path}: {e}")

    return result


class PointingTracker:
    """Tracks telescope pointing and detects declination changes.

    When a Dec change is detected, triggers precomputation of:
    - Best calibrator for the new Dec
    - Transit times for upcoming calibrators
    - Catalog strip database builds (background)

    """

    def __init__(
        self,
        dec_change_threshold: float = DEFAULT_DEC_CHANGE_THRESHOLD,
        catalog_types: list[str] | None = None,
        auto_precompute: bool = True,
        max_workers: int = 2,
    ):
        """Initialize the pointing tracker.

        Parameters
        ----------
        dec_change_threshold : float
            Minimum Dec change to trigger precomputation (degrees)
        catalog_types : list of str
            Catalog types to preload on Dec change
        auto_precompute : bool
            If True, automatically trigger precomputation
        max_workers : int
            Thread pool size for background tasks
        """
        self.dec_change_threshold = dec_change_threshold
        self.catalog_types = catalog_types or DEFAULT_CATALOG_TYPES
        self.auto_precompute = auto_precompute

        self._current_dec: float | None = None
        self._last_change: PointingChange | None = None
        self._change_history: list[PointingChange] = []
        self._lock = threading.Lock()

        # Background worker for catalog builds
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._pending_builds: dict[str, Any] = {}  # catalog_type -> Future

        # Transit prediction cache
        self._transit_cache: dict[float, list[CalibratorPrediction]] = {}
        self._cache_expiry: dict[float, datetime] = {}

    @property
    def current_dec(self) -> float | None:
        """Current telescope declination."""
        return self._current_dec

    @property
    def last_change(self) -> PointingChange | None:
        """Most recent pointing change."""
        return self._last_change

    def check_pointing_change(
        self,
        hdf5_path: Path,
        force: bool = False,
    ) -> PointingChange | None:
        """Check if an HDF5 file indicates a pointing change.

        Parameters
        ----------
        hdf5_path : Path
            Path to HDF5 file to check
        force : bool, optional
            If True, process even if Dec hasn't changed significantly (default is False)

        """
        new_dec = read_uvh5_dec_fast(hdf5_path)
        if new_dec is None:
            logger.debug(f"Could not read Dec from {hdf5_path}")
            return None

        with self._lock:
            old_dec = self._current_dec

            # Check if change is significant using shared detection logic
            if not force and not detect_dec_change(new_dec, old_dec, self.dec_change_threshold):
                return None

            # Create change record
            change = PointingChange(
                old_dec_deg=old_dec,
                new_dec_deg=new_dec,
                timestamp=datetime.utcnow(),
                source_file=str(hdf5_path),
            )

            # Update state
            self._current_dec = new_dec
            self._last_change = change
            self._change_history.append(change)

            # Limit history size
            if len(self._change_history) > 100:
                self._change_history = self._change_history[-100:]

        logger.info(
            f"Pointing change detected: Dec {old_dec}° -> {new_dec}° "
            f"(delta={abs(new_dec - (old_dec or 0)):.3f}°)"
        )

        # Trigger precomputation if enabled
        if self.auto_precompute:
            self._trigger_precomputation(change)

        return change

    def _trigger_precomputation(self, change: PointingChange) -> None:
        """Trigger precomputation tasks for a pointing change.

        Parameters
        ----------
        change : PointingChange
            Pointing change event to process

        """
        # 1. Precompute best calibrator
        try:
            calibrator = self._precompute_calibrator(change.new_dec_deg)
            if calibrator:
                change.precomputed_calibrator = calibrator.name
                change.calibrator_transit_utc = calibrator.transit_utc
                change.calibrator_dec_deg = calibrator.dec_deg
                logger.info(
                    f"Precomputed calibrator: {calibrator.name} "
                    f"(Dec separation={calibrator.dec_separation_deg:.2f}°, "
                    f"transit at {calibrator.transit_utc.isoformat()})"
                )
        except Exception as e:
            logger.error(f"Failed to precompute calibrator: {e}")

        # 2. Queue catalog strip builds
        try:
            queued = self._queue_catalog_builds(change.new_dec_deg)
            change.catalog_build_started = len(queued) > 0
            change.catalog_types_queued = queued
            if queued:
                logger.info(f"Queued catalog builds: {queued}")
        except Exception as e:
            logger.error(f"Failed to queue catalog builds: {e}")

    def _precompute_calibrator(
        self,
        dec_deg: float,
        max_dec_separation: float = DEFAULT_MAX_DEC_SEPARATION,
        min_flux_jy: float = MIN_FLUX_JY_20CM,
    ) -> CalibratorPrediction | None:
        """Find best calibrator for the given declination.

        Uses the canonical ``rank_calibrators_near_position()`` for
        unified scoring (flux × quality × position × proximity), then
        computes transit times for each candidate to select the best
        combination of quality and accessibility.

        Parameters
        ----------
        dec_deg : float
            Target declination
        max_dec_separation : float, optional
            Maximum Dec difference from target in degrees (default is DEFAULT_MAX_DEC_SEPARATION)
        min_flux_jy : float, optional
            Minimum flux at 20 cm in Jy (default is MIN_FLUX_JY_20CM)

        """
        from dsa110_contimg.core.catalog.calibrator_integration import (
            rank_calibrators_near_position,
        )
        from dsa110_contimg.core.pointing.monitor import predict_calibrator_transit_by_coords

        now = datetime.utcnow()

        # Use canonical ranking function (Dec-only mode: ra_deg=None)
        try:
            ranked = rank_calibrators_near_position(
                ra_deg=None,
                dec_deg=dec_deg,
                search_radius_deg=max_dec_separation,
                num_sources=10,  # Get multiple candidates for transit selection
                catalog_path=str(VLA_CALIBRATOR_DB),
                min_flux_jy=min_flux_jy,
            )
        except Exception as e:
            logger.error(f"Error querying VLA calibrator catalog: {e}")
            return None

        if not ranked:
            logger.warning(
                f"No calibrators found within {max_dec_separation}° of Dec={dec_deg}° "
                f"with flux >= {min_flux_jy} Jy"
            )
            return None

        # Compute transit times and build CalibratorPrediction objects
        candidates: list[CalibratorPrediction] = []
        for cal in ranked:
            try:
                pred = predict_calibrator_transit_by_coords(
                    ra_deg=cal["ra_deg"],
                    dec_deg=cal["dec_deg"],
                    from_time=now,
                    name=cal["name"],
                )
            except Exception as e:
                logger.debug(f"Could not compute transit for {cal['name']}: {e}")
                continue

            if pred is None:
                continue

            candidates.append(
                CalibratorPrediction(
                    name=cal["name"],
                    ra_deg=cal["ra_deg"],
                    dec_deg=cal["dec_deg"],
                    transit_utc=pred.transit_utc,
                    time_to_transit_sec=pred.time_to_transit_sec,
                    dec_separation_deg=cal["separation_deg"],
                    expected_flux_jy=cal["flux_jy"],
                    priority_score=cal["composite_score"],
                    source="vla_catalog",
                )
            )

        if not candidates:
            logger.warning(
                f"No calibrators with computable transits near Dec={dec_deg}°"
            )
            return None

        # Already sorted by composite_score (from rank_calibrators_near_position)
        # Cache all predictions for this Dec
        self._transit_cache[dec_deg] = candidates
        self._cache_expiry[dec_deg] = now + timedelta(hours=1)

        return candidates[0]

    def _queue_catalog_builds(self, dec_deg: float) -> list[str]:
        """Queue background catalog strip database builds.

        Parameters
        ----------
        dec_deg : float
            Declination for catalog strips

        """
        from dsa110_contimg.core.catalog.builders import (
            CATALOG_COVERAGE_LIMITS,
            check_catalog_database_exists,
        )

        queued = []

        for catalog_type in self.catalog_types:
            # Check if already exists
            exists, _ = check_catalog_database_exists(catalog_type, dec_deg)
            if exists:
                logger.debug(f"Catalog {catalog_type} already exists for Dec={dec_deg:.1f}°")
                continue

            # Check coverage
            limits = CATALOG_COVERAGE_LIMITS.get(catalog_type, {})
            if not (limits.get("dec_min", -90) <= dec_deg <= limits.get("dec_max", 90)):
                logger.debug(f"Dec={dec_deg}° outside {catalog_type} coverage")
                continue

            # Check if already building
            key = f"{catalog_type}_{dec_deg:.1f}"
            if key in self._pending_builds:
                future = self._pending_builds[key]
                if not future.done():
                    logger.debug(f"Build already in progress for {key}")
                    continue

            # Submit build task
            future = self._executor.submit(
                self._build_catalog_strip,
                catalog_type,
                dec_deg,
            )
            self._pending_builds[key] = future
            queued.append(catalog_type)

        return queued

    def _build_catalog_strip(self, catalog_type: str, dec_deg: float) -> Path | None:
        """Build a catalog strip database (runs in background thread).

        Parameters
        ----------
        catalog_type : str
            Catalog type ("nvss", "first", etc.)
        dec_deg : float
            Center declination

        """
        try:
            from dsa110_contimg.core.catalog.builders import (
                build_first_strip_db,
                build_nvss_strip_db,
                build_vlass_strip_db,
            )

            logger.info(f"Building {catalog_type} catalog strip for Dec={dec_deg:.1f}°...")
            start = time.time()

            dec_range = (dec_deg - 6.0, dec_deg + 6.0)

            if catalog_type == "nvss":
                db_path = build_nvss_strip_db(dec_center=dec_deg, dec_range=dec_range)
            elif catalog_type == "first":
                db_path = build_first_strip_db(dec_center=dec_deg, dec_range=dec_range)
            elif catalog_type == "vlass":
                db_path = build_vlass_strip_db(dec_center=dec_deg, dec_range=dec_range)
            else:
                logger.warning(f"Unknown catalog type: {catalog_type}")
                return None

            elapsed = time.time() - start
            logger.info(f"Built {catalog_type} catalog at {db_path} in {elapsed:.1f}s")
            return db_path

        except Exception as e:
            logger.error(f"Failed to build {catalog_type} catalog for Dec={dec_deg:.1f}°: {e}")
            return None

    def get_cached_transits(
        self,
        dec_deg: float | None = None,
    ) -> list[CalibratorPrediction]:
        """Get cached transit predictions.

        Parameters
        ----------
        dec_deg :
            Target Dec (default: current pointing)
        dec_deg : Optional[float] :
            (Default value = None)
        dec_deg : Optional[float] :
            (Default value = None)
        dec_deg : Optional[float] :
            (Default value = None)
        dec_deg: Optional[float] :
             (Default value = None)

        """
        if dec_deg is None:
            dec_deg = self._current_dec
        if dec_deg is None:
            return []

        # Check cache expiry
        now = datetime.utcnow()
        if dec_deg in self._cache_expiry:
            if now > self._cache_expiry[dec_deg]:
                # Cache expired, recompute
                del self._transit_cache[dec_deg]
                del self._cache_expiry[dec_deg]

        # Return cached or compute
        if dec_deg in self._transit_cache:
            return self._transit_cache[dec_deg]

        # Compute and cache
        self._precompute_calibrator(dec_deg)
        return self._transit_cache.get(dec_deg, [])

    def get_best_calibrator(self, dec_deg: float | None = None) -> CalibratorPrediction | None:
        """Get the best calibrator for the given (or current) declination.

        Parameters
        ----------
        dec_deg :
            Target Dec (default: current pointing)
        dec_deg : Optional[float] :
            (Default value = None)
        dec_deg : Optional[float] :
            (Default value = None)
        dec_deg : Optional[float] :
            (Default value = None)
        dec_deg: Optional[float] :
             (Default value = None)

        """
        transits = self.get_cached_transits(dec_deg)
        return transits[0] if transits else None

    def get_status(self) -> dict:
        """Get current tracker status for monitoring/API."""
        pending = {k: not v.done() for k, v in self._pending_builds.items()}

        return {
            "current_dec_deg": self._current_dec,
            "last_change": self._last_change.to_dict() if self._last_change else None,
            "change_count": len(self._change_history),
            "cached_transit_decs": list(self._transit_cache.keys()),
            "pending_catalog_builds": pending,
            "auto_precompute_enabled": self.auto_precompute,
        }

    def shutdown(self):
        """Shutdown background workers."""
        self._executor.shutdown(wait=False)


# Global tracker instance
_tracker: PointingTracker | None = None


def get_pointing_tracker() -> PointingTracker:
    """Get or create the global pointing tracker."""
    global _tracker
    if _tracker is None:
        _tracker = PointingTracker()
    return _tracker


async def precompute_all_transits(
    hours_ahead: float = DEFAULT_TRANSIT_HORIZON_HOURS,
    min_flux_jy: float = 5.0,
    max_sources: int = 20,
) -> dict[str, list[CalibratorPrediction]]:
    """Precompute all calibrator transits for the next N hours.

    Queries the VLA calibrator catalog for the brightest sources and
    computes their transit times over the specified horizon.  This can
    be called at startup or periodically to warm the cache.

    Parameters
    ----------
    hours_ahead : float
        How many hours ahead to compute (default 48).
    min_flux_jy : float
        Minimum L-band flux in Jy to include (default 5.0).
    max_sources : int
        Maximum number of bright calibrators to include (default 20).

    Returns
    -------
    dict
        Mapping of calibrator name to list of CalibratorPrediction.
    """
    import sqlite3

    from dsa110_contimg.core.pointing.monitor import predict_calibrator_transit_by_coords

    now = datetime.utcnow()
    end_time = now + timedelta(hours=hours_ahead)

    # Query VLA catalog for the brightest calibrators
    catalog_path = str(VLA_CALIBRATOR_DB)
    try:
        conn = sqlite3.connect(catalog_path)
        rows = conn.execute(
            """
            SELECT c.name, c.ra_deg, c.dec_deg, MAX(f.flux_jy) AS flux_jy
            FROM calibrators c
            JOIN fluxes f ON f.name = c.name
            WHERE f.band = '20cm'
              AND f.flux_jy >= ?
            GROUP BY c.name
            ORDER BY flux_jy DESC
            LIMIT ?
            """,
            (min_flux_jy, max_sources),
        ).fetchall()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to query VLA catalog for cache warming: {e}")
        return {}

    all_predictions: dict[str, list[CalibratorPrediction]] = {}

    for name, ra_deg, dec_deg, flux_jy in rows:
        predictions: list[CalibratorPrediction] = []
        search_time = now

        while search_time < end_time:
            pred = predict_calibrator_transit_by_coords(
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                from_time=search_time,
                name=name,
            )
            if pred is None or pred.transit_utc > end_time:
                break

            predictions.append(
                CalibratorPrediction(
                    name=name,
                    ra_deg=ra_deg,
                    dec_deg=dec_deg,
                    transit_utc=pred.transit_utc,
                    time_to_transit_sec=pred.time_to_transit_sec,
                    dec_separation_deg=0.0,  # N/A for general precompute
                    expected_flux_jy=flux_jy,
                    priority_score=0.0,
                )
            )

            # Move past this transit
            search_time = pred.transit_utc + timedelta(minutes=10)

        all_predictions[name] = predictions

    logger.info(
        f"Precomputed {sum(len(v) for v in all_predictions.values())} transits "
        f"for {len(all_predictions)} calibrators over next {hours_ahead:.0f} hours"
    )

    return all_predictions


def ensure_catalogs_for_dec(
    dec_deg: float,
    catalog_types: list[str] | None = None,
    wait: bool = True,
    timeout_sec: float = 300.0,
) -> dict[str, Path | None]:
    """Ensure catalog databases exist for the given declination.

    Parameters
    ----------
    dec_deg : float
        Target declination
    catalog_types : list of str, optional
        Catalog types to ensure (default is None, which implies nvss, first, vlass)
    wait : bool, optional
        If True, wait for builds to complete (default is True)
    timeout_sec : float, optional
        Maximum wait time in seconds (default is 300.0)

    """
    from dsa110_contimg.core.catalog.builders import check_catalog_database_exists

    catalog_types = catalog_types or DEFAULT_CATALOG_TYPES
    results: dict[str, Path | None] = {}

    tracker = get_pointing_tracker()

    for catalog_type in catalog_types:
        exists, db_path = check_catalog_database_exists(catalog_type, dec_deg)

        if exists:
            results[catalog_type] = db_path
            continue

        # Need to build
        key = f"{catalog_type}_{dec_deg:.1f}"
        if key in tracker._pending_builds:
            future = tracker._pending_builds[key]
        else:
            # Start new build
            future = tracker._executor.submit(
                tracker._build_catalog_strip,
                catalog_type,
                dec_deg,
            )
            tracker._pending_builds[key] = future

        if wait:
            try:
                results[catalog_type] = future.result(timeout=timeout_sec)
            except Exception as e:
                logger.error(f"Catalog build failed for {catalog_type}: {e}")
                results[catalog_type] = None
        else:
            results[catalog_type] = None  # Build in progress

    return results
