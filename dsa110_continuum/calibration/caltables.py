"""
Calibration table discovery utilities.

Provides functions for discovering calibration tables associated with
Measurement Sets, bidirectional time-based search, and staleness monitoring.
"""

from __future__ import annotations

import glob
import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CalibrationCandidate:
    """A calibration table set candidate."""

    set_name: str
    mjd: float
    time_diff_hours: float
    tables: dict[str, str | None]
    source_ms_path: str | None = None
    status: str = "active"
    staleness_level: str = "fresh"  # fresh, aging, stale, critical


@dataclass
class CalibrationHealth:
    """Health status for calibration tables."""

    status: str  # healthy, warning, critical
    message: str
    nearest_cal_hours: float | None = None
    active_sets: int = 0
    stale_sets: int = 0
    missing_types: list[str] = None

    def __post_init__(self):
        if self.missing_types is None:
            self.missing_types = []


# Staleness thresholds (hours)
STALENESS_THRESHOLDS = {
    "fresh": 6.0,  # < 6 hours
    "aging": 12.0,  # 6-12 hours
    "stale": 24.0,  # 12-24 hours
    "critical": 48.0,  # > 24 hours is critical
}


def discover_caltables(ms_path: str) -> dict[str, str | None]:
    """Discover calibration tables associated with an MS.

    Parameters
    ----------
    ms_path :
        Path to the Measurement Set

    Returns
    -------
        Dictionary with keys 'K', 'B', 'G' mapping to table paths (or None if not found)
        Note: Keys are uppercase to match CASA convention used by jobs.py

    """
    if not os.path.exists(ms_path):
        return {"K": None, "B": None, "G": None}

    # Get MS directory and base name
    ms_dir = os.path.dirname(ms_path)
    ms_base = os.path.basename(ms_path).replace(".ms", "")

    # Search patterns for cal tables - support both old (*kcal) and new (.k) naming
    # Delay tables: *.kcal, *.k, *.2k
    k_patterns = [
        os.path.join(ms_dir, f"{ms_base}*kcal"),
        os.path.join(ms_dir, f"{ms_base}*.k"),
        os.path.join(ms_dir, f"{ms_base}*.2k"),
    ]
    # Bandpass tables: *.bpcal, *.b, *.prebp
    bp_patterns = [
        os.path.join(ms_dir, f"{ms_base}*bpcal"),
        os.path.join(ms_dir, f"{ms_base}*.b"),
    ]
    # Gain tables: *.gpcal, *.gacal, *.g, *.2g
    g_patterns = [
        os.path.join(ms_dir, f"{ms_base}*gpcal"),
        os.path.join(ms_dir, f"{ms_base}*gacal"),
        os.path.join(ms_dir, f"{ms_base}*.g"),
        os.path.join(ms_dir, f"{ms_base}*.2g"),
    ]

    # Find tables matching any pattern, sort by mtime
    def find_latest(patterns: list[str]) -> str | None:
        all_tables = []
        for pattern in patterns:
            all_tables.extend(glob.glob(pattern))
        if not all_tables:
            return None
        # Sort by modification time, newest first
        return sorted(all_tables, key=os.path.getmtime, reverse=True)[0]

    return {
        "K": find_latest(k_patterns),
        "B": find_latest(bp_patterns),
        "G": find_latest(g_patterns),
    }


def _classify_staleness(hours: float) -> str:
    """Classify calibration staleness based on age in hours.

    Parameters
    ----------
    """
    if hours <= STALENESS_THRESHOLDS["fresh"]:
        return "fresh"
    elif hours <= STALENESS_THRESHOLDS["aging"]:
        return "aging"
    elif hours <= STALENESS_THRESHOLDS["stale"]:
        return "stale"
    else:
        return "critical"


def find_nearest_calibration(
    target_mjd: float,
    calibration_dir: Path | None = None,
    *,
    search_window_hours: float = 24.0,
    registry_db: str | None = None,
    require_types: list[str] | None = None,
) -> CalibrationCandidate | None:
    """Find calibration nearest to target MJD with bidirectional search.

    Searches both the database registry and filesystem for calibration
    tables within the specified time window. Returns the closest valid
    calibration set.

    Parameters
    ----------
    target_mjd :
        MJD of target observation
    calibration_dir :
        Root directory containing calibration tables (filesystem search)
    search_window_hours :
        Maximum time difference (default: ±24h)
    registry_db :
        Path to calibration registry database (default: pipeline.sqlite3)
    require_types :
        Required table types (default: ['bp', 'g'])

    Returns
    -------
        CalibrationCandidate with calibration info or None if not found

    """
    if require_types is None:
        require_types = ["bp", "g"]

    search_window_mjd = search_window_hours / 24.0
    candidates: list[CalibrationCandidate] = []

    # Strategy 1: Search database registry
    if registry_db is None:
        registry_db = os.environ.get(
            "PIPELINE_DB",
            os.environ.get("PIPELINE_DB", "/data/dsa110-contimg/state/db/pipeline.sqlite3"),
        )

    if Path(registry_db).exists():
        candidates.extend(
            _search_registry(target_mjd, registry_db, search_window_mjd, require_types)
        )

    # Strategy 2: Search filesystem (if directory provided)
    if calibration_dir and Path(calibration_dir).exists():
        candidates.extend(
            _search_filesystem(target_mjd, Path(calibration_dir), search_window_mjd, require_types)
        )

    if not candidates:
        logger.warning(
            "No calibrations within ±%.1fh of MJD %.5f",
            search_window_hours,
            target_mjd,
        )
        return None

    # Sort by time difference (closest first)
    candidates.sort(key=lambda x: x.time_diff_hours)
    best = candidates[0]

    # Classify staleness
    best.staleness_level = _classify_staleness(best.time_diff_hours)

    logger.info(
        "Selected calibration %s from MJD %.4f (Δt = %.1f hours, staleness: %s)",
        best.set_name,
        best.mjd,
        best.time_diff_hours,
        best.staleness_level,
    )

    # Alert if stale
    if best.staleness_level in ("stale", "critical"):
        logger.warning(
            "Using %s calibration from %.1f hours ago",
            best.staleness_level,
            best.time_diff_hours,
        )

    return best


def _search_registry(
    target_mjd: float,
    registry_db: str,
    search_window_mjd: float,
    require_types: list[str],
    *,
    validate_per_type: bool = True,
) -> list[CalibrationCandidate]:
    """Search database registry for calibration sets.

    When validate_per_type=True (default), validates each table type against
    its own validity window (BP: ±24h, G: ±1h). Sets with any stale table
    type are excluded from results.

    Parameters
    ----------
    target_mjd : float
        Target observation MJD
    registry_db : str
        Path to calibration registry database
    search_window_mjd : float
        Search window in MJD (days)
    require_types : List[str]
        Required table types (e.g., ['bp', 'g'])
    validate_per_type : bool
        If True, validate each type against its own validity window

    Returns
    -------
    List[CalibrationCandidate]
        List of valid calibration candidates
    """
    from dsa110_contimg.core.calibration.hardening import get_validity_hours_for_type

    candidates = []

    try:
        conn = sqlite3.connect(registry_db, timeout=10.0)
        conn.row_factory = sqlite3.Row

        # Find all active calibration sets within the time window
        # Use the midpoint of validity window as the calibration time
        rows = conn.execute(
            """
            SELECT set_name, path, table_type, source_ms_path, created_at,
                   valid_start_mjd, valid_end_mjd, status
            FROM caltables
            WHERE status = 'active'
              AND (valid_start_mjd IS NULL OR valid_start_mjd <= ?)
              AND (valid_end_mjd IS NULL OR valid_end_mjd >= ?)
            ORDER BY set_name, table_type
            """,
            (target_mjd + search_window_mjd, target_mjd - search_window_mjd),
        ).fetchall()

        conn.close()

        # Group by set_name, keeping per-table validity info
        sets: dict[str, dict[str, Any]] = {}
        for row in rows:
            set_name = row["set_name"]
            if set_name not in sets:
                sets[set_name] = {
                    "tables": {},
                    "table_validity": {},  # table_type -> (mid_mjd, valid_start, valid_end)
                    "source_ms_path": row["source_ms_path"],
                    "created_at": row["created_at"],
                    "valid_start": row["valid_start_mjd"],
                    "valid_end": row["valid_end_mjd"],
                    "status": row["status"],
                }

            # Map table_type to our standard keys
            ttype_raw = row["table_type"]
            ttype = ttype_raw.lower()
            if "bandpass" in ttype or ttype == "bp" or ttype == "ba":
                key = "bp"
            elif "delay" in ttype or ttype == "k":
                key = "k"
            elif "gain" in ttype or ttype in ("g", "ga", "gp", "2g"):
                key = "g"
            else:
                key = ttype

            sets[set_name]["tables"][key] = row["path"]
            # Store validity info for this specific table
            valid_start = row["valid_start_mjd"]
            valid_end = row["valid_end_mjd"]
            if valid_start and valid_end:
                mid_mjd = (valid_start + valid_end) / 2.0
            else:
                mid_mjd = None
            sets[set_name]["table_validity"][ttype_raw.upper()] = (mid_mjd, valid_start, valid_end)

        # Create candidates
        for set_name, info in sets.items():
            # Check if all required types are present
            if not all(info["tables"].get(t) for t in require_types):
                continue

            # Calculate calibration MJD (midpoint of validity or created_at)
            if info["valid_start"] and info["valid_end"]:
                cal_mjd = (info["valid_start"] + info["valid_end"]) / 2
            elif info["created_at"]:
                # Convert Unix timestamp to MJD
                try:
                    from astropy.time import Time

                    cal_mjd = Time(info["created_at"], format="unix").mjd
                except ImportError:
                    cal_mjd = info["created_at"] / 86400.0 + 40587.0  # Approx
            else:
                continue

            time_diff = abs(cal_mjd - target_mjd)
            if time_diff > search_window_mjd:
                continue

            # Per-type validity check
            if validate_per_type:
                all_types_valid = True
                stale_types = []

                for req_type in require_types:
                    # Find the actual table type in this set that matches
                    type_upper = req_type.upper()
                    # Check if we have validity info for this type or a related type
                    validity_info = None
                    for stored_type, vinfo in info["table_validity"].items():
                        stored_lower = stored_type.lower()
                        if req_type == "bp" and stored_lower in ("bp", "ba", "bandpass"):
                            validity_info = vinfo
                            break
                        elif req_type == "g" and stored_lower in ("g", "ga", "gp", "2g", "gain"):
                            validity_info = vinfo
                            break
                        elif req_type == "k" and stored_lower in ("k", "delay"):
                            validity_info = vinfo
                            break
                        elif stored_lower == req_type:
                            validity_info = vinfo
                            break

                    if validity_info is None:
                        # No validity info, skip type check
                        continue

                    mid_mjd, valid_start, valid_end = validity_info
                    if mid_mjd is None:
                        continue

                    # Get type-specific validity window
                    type_validity_hours = get_validity_hours_for_type(type_upper)
                    offset_hours = abs(target_mjd - mid_mjd) * 24.0

                    if offset_hours > type_validity_hours:
                        all_types_valid = False
                        stale_types.append(
                            f"{type_upper} (stale by {offset_hours - type_validity_hours:.1f}h)"
                        )

                if not all_types_valid:
                    logger.debug(
                        "Excluding set '%s': stale types: %s", set_name, ", ".join(stale_types)
                    )
                    continue

            candidates.append(
                CalibrationCandidate(
                    set_name=set_name,
                    mjd=cal_mjd,
                    time_diff_hours=time_diff * 24.0,
                    tables=info["tables"],
                    source_ms_path=info["source_ms_path"],
                    status=info["status"],
                )
            )

    except sqlite3.Error as exc:
        logger.warning("Registry search failed: %s", exc)

    return candidates


def _search_filesystem(
    target_mjd: float,
    calibration_dir: Path,
    search_window_mjd: float,
    require_types: list[str],
) -> list[CalibrationCandidate]:
    """Search filesystem for calibration tables.

    Parameters
    ----------
    """
    candidates = []

    try:
        from astropy.time import Time

        for ms_path in calibration_dir.glob("*.ms"):
            try:
                # Use modification time as proxy for observation time
                mtime = os.path.getmtime(ms_path)
                cal_mjd = Time(mtime, format="unix").mjd

                # Check time window
                time_diff = abs(cal_mjd - target_mjd)
                if time_diff > search_window_mjd:
                    continue

                # Discover associated calibration tables
                tables = discover_caltables(str(ms_path))

                # Check if required types are present
                if not all(tables.get(t) for t in require_types):
                    logger.debug("Incomplete calibration for %s", ms_path)
                    continue

                candidates.append(
                    CalibrationCandidate(
                        set_name=ms_path.stem,
                        mjd=cal_mjd,
                        time_diff_hours=time_diff * 24.0,
                        tables=tables,
                        source_ms_path=str(ms_path),
                    )
                )

            except (OSError, ValueError) as exc:
                logger.debug("Error processing %s: %s", ms_path, exc)
                continue

    except ImportError:
        logger.warning("astropy not available for filesystem search")

    return candidates


def get_applylist_for_mjd(
    target_mjd: float,
    calibration_dir: Path | None = None,
    **kwargs: Any,
) -> list[str]:
    """Get ordered list of calibration tables for target MJD.

    Parameters
    ----------
    target_mjd :
        MJD of target observation
    calibration_dir :
        Root calibration directory
    **kwargs :
        Additional arguments passed to find_nearest_calibration

    Returns
    -------
    List of calibration table paths to apply (ordered
        k, bp, g)

    """
    result = find_nearest_calibration(target_mjd, calibration_dir, **kwargs)

    if result is None:
        return []

    # Return ordered list (delay, bandpass, gain)
    tables = []
    for key in ["k", "bp", "g"]:
        if result.tables.get(key):
            tables.append(result.tables[key])

    return tables


def check_calibration_staleness(
    target_mjd: float,
    registry_db: str | None = None,
    *,
    warning_threshold_hours: float = 12.0,
    critical_threshold_hours: float = 24.0,
) -> CalibrationHealth:
    """Check calibration health for a given observation time.

    Evaluates the freshness of available calibrations and returns
    health status with recommendations.

    Parameters
    ----------
    target_mjd :
        MJD to check calibration health for
    registry_db :
        Path to calibration registry
    warning_threshold_hours :
        Hours until warning (default: 12)
    critical_threshold_hours :
        Hours until critical (default: 24)

    Returns
    -------
        CalibrationHealth with status and diagnostics

    """
    if registry_db is None:
        registry_db = os.environ.get(
            "PIPELINE_DB",
            os.environ.get("PIPELINE_DB", "/data/dsa110-contimg/state/db/pipeline.sqlite3"),
        )

    # Find nearest calibration with wide search
    nearest = find_nearest_calibration(
        target_mjd,
        registry_db=registry_db,
        search_window_hours=critical_threshold_hours * 2,
    )

    if nearest is None:
        return CalibrationHealth(
            status="critical",
            message="No calibrations found within search window",
            active_sets=0,
            stale_sets=0,
        )

    # Count active and stale sets
    active_sets = 0
    stale_sets = 0

    try:
        conn = sqlite3.connect(registry_db, timeout=10.0)
        conn.row_factory = sqlite3.Row

        # Count sets by staleness
        rows = conn.execute(
            """
            SELECT COUNT(DISTINCT set_name) as count
            FROM caltables
            WHERE status = 'active'
            """
        ).fetchone()
        active_sets = rows["count"] if rows else 0

        # Count stale sets (validity ended more than threshold ago)
        stale_threshold = target_mjd - (critical_threshold_hours / 24.0)
        rows = conn.execute(
            """
            SELECT COUNT(DISTINCT set_name) as count
            FROM caltables
            WHERE status = 'active'
              AND valid_end_mjd IS NOT NULL
              AND valid_end_mjd < ?
            """,
            (stale_threshold,),
        ).fetchone()
        stale_sets = rows["count"] if rows else 0

        conn.close()

    except sqlite3.Error as exc:
        logger.warning("Failed to query registry: %s", exc)

    # Determine health status
    if nearest.time_diff_hours > critical_threshold_hours:
        status = "critical"
        message = f"Nearest calibration is {nearest.time_diff_hours:.1f} hours old"
    elif nearest.time_diff_hours > warning_threshold_hours:
        status = "warning"
        message = f"Nearest calibration is {nearest.time_diff_hours:.1f} hours old"
    else:
        status = "healthy"
        message = f"Fresh calibration available ({nearest.time_diff_hours:.1f} hours old)"

    # Check for missing table types
    missing = []
    for ttype in ["bp", "g"]:
        if not nearest.tables.get(ttype):
            missing.append(ttype)

    if missing:
        status = "warning" if status == "healthy" else status
        message += f"; missing table types: {', '.join(missing)}"

    return CalibrationHealth(
        status=status,
        message=message,
        nearest_cal_hours=nearest.time_diff_hours,
        active_sets=active_sets,
        stale_sets=stale_sets,
        missing_types=missing,
    )


def get_calibration_timeline(
    start_mjd: float,
    end_mjd: float,
    registry_db: str | None = None,
) -> list[dict[str, Any]]:
    """Get timeline of calibration sets within a time range.

    Useful for visualizing calibration coverage over time.

    Parameters
    ----------
    start_mjd :
        Start of time range
    end_mjd :
        End of time range
    registry_db :
        Path to registry database

    Returns
    -------
        List of calibration sets with validity windows

    """
    if registry_db is None:
        registry_db = os.environ.get(
            "PIPELINE_DB",
            os.environ.get("PIPELINE_DB", "/data/dsa110-contimg/state/db/pipeline.sqlite3"),
        )

    if not Path(registry_db).exists():
        return []

    timeline = []

    try:
        from astropy.time import Time

        conn = sqlite3.connect(registry_db, timeout=10.0)
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            """
            SELECT DISTINCT set_name,
                   MIN(valid_start_mjd) as start_mjd,
                   MAX(valid_end_mjd) as end_mjd,
                   GROUP_CONCAT(DISTINCT table_type) as types,
                   COUNT(*) as table_count
            FROM caltables
            WHERE status = 'active'
              AND (
                (valid_start_mjd IS NULL AND valid_end_mjd IS NULL)
                OR (valid_start_mjd <= ? AND valid_end_mjd >= ?)
                OR (valid_start_mjd BETWEEN ? AND ?)
                OR (valid_end_mjd BETWEEN ? AND ?)
              )
            GROUP BY set_name
            ORDER BY start_mjd
            """,
            (end_mjd, start_mjd, start_mjd, end_mjd, start_mjd, end_mjd),
        ).fetchall()

        conn.close()

        for row in rows:
            entry = {
                "set_name": row["set_name"],
                "start_mjd": row["start_mjd"],
                "end_mjd": row["end_mjd"],
                "start_iso": (
                    Time(row["start_mjd"], format="mjd").isot if row["start_mjd"] else None
                ),
                "end_iso": (Time(row["end_mjd"], format="mjd").isot if row["end_mjd"] else None),
                "table_types": row["types"].split(",") if row["types"] else [],
                "table_count": row["table_count"],
            }
            timeline.append(entry)

    except (sqlite3.Error, ImportError) as exc:
        logger.warning("Failed to get calibration timeline: %s", exc)

    return timeline
