"""Calibration hardening logic (Issues #1, #2, #3, #6)."""

from __future__ import annotations

import json
import logging
import sqlite3
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from dsa110_contimg.infrastructure.database.unified import ensure_db, retire_caltable_set

logger = logging.getLogger(__name__)

# =============================================================================
# Issue #1: Bidirectional Calibration Validity Windows
# =============================================================================

# Type-specific calibration validity windows (half-windows, i.e., ± from midpoint)
# Bandpass (BP, BA) and delay (K) are stable over longer periods
BP_VALIDITY_HOURS = 24.0  # ±24h = 48h total window for bandpass
K_VALIDITY_HOURS = 24.0  # ±24h = 48h total window for delays (align with BP)
# Gains (G, GA, GP, 2G) track atmospheric/instrumental drifts, need tighter windows
G_VALIDITY_HOURS = 1.0  # ±1h = 2h total window for gains

# Derived constants in days
BP_VALIDITY_DAYS = BP_VALIDITY_HOURS / 24.0
K_VALIDITY_DAYS = K_VALIDITY_HOURS / 24.0
G_VALIDITY_DAYS = G_VALIDITY_HOURS / 24.0

# Legacy default (kept for backward compatibility, but prefer type-specific)
DEFAULT_CAL_VALIDITY_HOURS = 12.0
DEFAULT_CAL_VALIDITY_DAYS = DEFAULT_CAL_VALIDITY_HOURS / 24.0

# Mapping from table type to validity hours
_TABLE_TYPE_VALIDITY_MAP: dict[str, float] = {
    # Bandpass tables (stable over ~days)
    "BP": BP_VALIDITY_HOURS,
    "BA": BP_VALIDITY_HOURS,
    "BANDPASS": BP_VALIDITY_HOURS,
    # Delay tables (stable over ~days)
    "K": K_VALIDITY_HOURS,
    "DELAY": K_VALIDITY_HOURS,
    # Gain tables (track atmospheric/instrumental drifts)
    "G": G_VALIDITY_HOURS,
    "GA": G_VALIDITY_HOURS,
    "GP": G_VALIDITY_HOURS,
    "2G": G_VALIDITY_HOURS,
    "GAIN": G_VALIDITY_HOURS,
    # Fluxscale (align with BP since derived from BP calibrator)
    "FLUX": BP_VALIDITY_HOURS,
}


def get_validity_hours_for_type(table_type: str) -> float:
    """Get the validity window (in hours) for a given calibration table type.

    Parameters
    ----------
    table_type : str
        Calibration table type (e.g., 'BP', 'G', 'K', 'GA', 'GP', '2G', 'FLUX')

    Returns
    -------
        float
        Validity half-window in hours. For unknown types, returns BP_VALIDITY_HOURS
        as a conservative default (24h).

    Examples
    --------
        >>> get_validity_hours_for_type("BP")
        24.0
        >>> get_validity_hours_for_type("G")
        1.0
        >>> get_validity_hours_for_type("unknown")  # Conservative default
        24.0
    """
    normalized = table_type.upper().strip()
    validity = _TABLE_TYPE_VALIDITY_MAP.get(normalized)
    if validity is not None:
        return validity
    # Conservative default for unknown types
    logger.warning(
        "Unknown table type '%s', using BP validity (%.1fh) as conservative default",
        table_type,
        BP_VALIDITY_HOURS,
    )
    return BP_VALIDITY_HOURS


def get_min_validity_for_types(table_types: list[str]) -> float:
    """Get the minimum (most restrictive) validity window across table types.

        Used when querying for multiple table types to ensure all are valid.

    Parameters
    ----------
    table_types : List[str]
        List of table types (e.g., ['BP', 'G'])

    Returns
    -------
        float
        Minimum validity half-window in hours across all types.

    Examples
    --------
        >>> get_min_validity_for_types(["BP", "G"])
        1.0  # G is most restrictive
    """
    if not table_types:
        return DEFAULT_CAL_VALIDITY_HOURS
    return min(get_validity_hours_for_type(t) for t in table_types)


@dataclass
class CalibrationSelection:
    """Result of calibration selection with quality metadata."""

    set_name: str
    paths: list[str]
    mid_mjd: float
    selection_method: str  # 'exact', 'nearest_before', 'nearest_after', 'interpolated'
    time_offset_hours: float  # How far from target MJD
    quality_score: float  # Combined quality metric
    warnings: list[str] = field(default_factory=list)
    # Type-specific staleness info (table_type -> hours stale, negative = still valid)
    staleness_by_type: dict[str, float] = field(default_factory=dict)


@dataclass
class TableTypeValidity:
    """Validity status for a specific table type within a calibration set."""

    table_type: str
    path: str
    mid_mjd: float
    valid_start_mjd: float
    valid_end_mjd: float
    offset_hours: float  # Distance from target
    validity_hours: float  # Type-specific validity window
    is_valid: bool  # Whether table is within its validity window
    staleness_hours: float  # Positive = stale by this many hours, negative = still valid


def _validate_set_by_type(
    conn: sqlite3.Connection,
    set_name: str,
    target_mjd: float,
    required_types: list[str],
) -> tuple[bool, list[TableTypeValidity], list[str]]:
    """Validate a calibration set by checking each table type's validity.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection
    set_name : str
        Calibration set name
    target_mjd : float
        Target observation MJD
    required_types : List[str]
        Required table types (e.g., ['BP', 'G'])
    """
    rows = conn.execute(
        """
        SELECT path, table_type, valid_start_mjd, valid_end_mjd
        FROM caltables
        WHERE set_name = ? AND status = 'active'
        ORDER BY order_index
        """,
        (set_name,),
    ).fetchall()

    if not rows:
        return False, [], [f"No tables found in set '{set_name}'"]

    # Group by type
    type_to_rows: dict[str, list[tuple]] = {}
    for row in rows:
        ttype = row[1].upper()
        if ttype not in type_to_rows:
            type_to_rows[ttype] = []
        type_to_rows[ttype].append(row)

    # Check each required type
    validity_list = []
    warnings = []
    all_valid = True

    for req_type in required_types:
        req_upper = req_type.upper()
        matching_rows = type_to_rows.get(req_upper, [])

        if not matching_rows:
            warnings.append(f"Required table type '{req_type}' not found in set")
            all_valid = False
            continue

        # Use first matching row for this type
        row = matching_rows[0]
        path, ttype, valid_start, valid_end = row

        # Get type-specific validity window
        type_validity_hours = get_validity_hours_for_type(ttype)

        # Calculate midpoint and offset
        if valid_start is not None and valid_end is not None:
            mid_mjd = (valid_start + valid_end) / 2.0
        else:
            mid_mjd = target_mjd  # Fallback

        offset_hours = abs(target_mjd - mid_mjd) * 24.0

        # Check if within this type's validity window
        is_valid = offset_hours <= type_validity_hours
        staleness_hours = offset_hours - type_validity_hours  # Positive = stale

        validity_list.append(
            TableTypeValidity(
                table_type=ttype,
                path=path,
                mid_mjd=mid_mjd,
                valid_start_mjd=valid_start or (mid_mjd - type_validity_hours / 24.0),
                valid_end_mjd=valid_end or (mid_mjd + type_validity_hours / 24.0),
                offset_hours=offset_hours,
                validity_hours=type_validity_hours,
                is_valid=is_valid,
                staleness_hours=staleness_hours,
            )
        )

        if not is_valid:
            all_valid = False
            warnings.append(
                f"{ttype} table stale by {staleness_hours:.1f}h "
                f"(offset {offset_hours:.1f}h > validity {type_validity_hours:.1f}h)"
            )

    return all_valid, validity_list, warnings


def get_active_applylist_bidirectional(
    db_path: Path,
    target_mjd: float,
    *,
    set_name: str | None = None,
    table_types: list[str] | None = None,
    validity_hours: float | None = None,
    prefer_nearest: bool = True,
    require_all_types_valid: bool = True,
) -> CalibrationSelection:
    """Return calibration tables with bidirectional validity windows.

        This fixes Issue #1: Forward-only validity windows that leave pre-calibrator
        observations without valid calibration.

        When table_types is specified, validates each type against its own validity
        window (BP: ±24h, G: ±1h). If require_all_types_valid=True (default), rejects
        sets where any requested type is stale.

        Strategy:
        1. First try exact match (target within validity window)
        2. If no exact match and prefer_nearest=True, find nearest calibration
        (before or after) within the most restrictive validity window
        3. Validate all requested table types are within their individual windows
        4. Return selection with quality metadata

    Parameters
    ----------
    db_path : Path
        Path to calibration registry database
    target_mjd : float
        Target observation MJD
    set_name : Optional[str], optional
        Optional specific set name (bypasses search), by default None
    table_types : Optional[List[str]], optional
        Required table types (e.g., ['BP', 'G']). When specified, validates
        each type against its own validity window, by default None
    validity_hours : Optional[float], optional
        Override search window. If None and table_types specified, uses the
        minimum validity across requested types, by default None
    prefer_nearest : bool, optional
        If True, find nearest calibration even outside strict window, by default True
    require_all_types_valid : bool, optional
        If True (default), reject sets where any requested type is stale.
        If False, return set with warnings about stale types, by default True
    """
    conn = ensure_db(db_path)

    # Determine search validity window
    if validity_hours is not None:
        search_validity = validity_hours
    elif table_types:
        search_validity = get_min_validity_for_types(table_types)
    else:
        search_validity = DEFAULT_CAL_VALIDITY_HOURS

    validity_days = search_validity / 24.0

    # Direct set lookup
    if set_name:
        rows = conn.execute(
            """
            SELECT path, valid_start_mjd, valid_end_mjd, quality_metrics
            FROM caltables
            WHERE set_name = ? AND status = 'active'
            ORDER BY order_index ASC
            """,
            (set_name,),
        ).fetchall()

        if rows:
            # Calculate time offset from set's validity window
            mid_mjds = [(r[1] + r[2]) / 2 if r[1] and r[2] else target_mjd for r in rows]
            avg_mid = statistics.mean(mid_mjds) if mid_mjds else target_mjd
            offset_hours = abs(target_mjd - avg_mid) * 24.0

            # Validate by type if requested
            staleness_by_type: dict[str, float] = {}
            warnings: list[str] = []

            if table_types:
                all_valid, validity_list, type_warnings = _validate_set_by_type(
                    conn, set_name, target_mjd, table_types
                )
                warnings.extend(type_warnings)
                staleness_by_type = {v.table_type: v.staleness_hours for v in validity_list}

                if require_all_types_valid and not all_valid:
                    stale_types = [v for v in validity_list if not v.is_valid]
                    stale_msg = ", ".join(
                        f"{v.table_type} (stale by {v.staleness_hours:.1f}h)" for v in stale_types
                    )
                    raise ValueError(f"Set '{set_name}' has stale tables: {stale_msg}")

            return CalibrationSelection(
                set_name=set_name,
                paths=[r[0] for r in rows],
                mid_mjd=avg_mid,
                selection_method="exact",
                time_offset_hours=offset_hours,
                quality_score=_calculate_quality_score(rows, offset_hours),
                warnings=warnings,
                staleness_by_type=staleness_by_type,
            )
        raise ValueError(f"No active calibration tables for set '{set_name}'")

    # BIDIRECTIONAL SEARCH: Look for calibrations ±validity_hours from target
    # This is the key fix for Issue #1

    # Build type filter clause if needed
    type_filter = ""
    if table_types:
        type_list = ", ".join(f"'{t.upper()}'" for t in table_types)
        type_filter = f"AND table_type IN ({type_list})"

    # 1. Try exact match (target within validity window)
    exact_sets = conn.execute(
        f"""
        SELECT DISTINCT set_name,
               (valid_start_mjd + valid_end_mjd) / 2.0 AS mid_mjd,
               MAX(created_at) AS newest
        FROM caltables
        WHERE status = 'active'
          AND (valid_start_mjd IS NULL OR valid_start_mjd <= ?)
          AND (valid_end_mjd IS NULL OR valid_end_mjd >= ?)
          {type_filter}
        GROUP BY set_name
        ORDER BY newest DESC
        """,
        (target_mjd, target_mjd),
    ).fetchall()

    # Validate each candidate set
    for set_row in exact_sets:
        chosen_set = set_row[0]
        mid_mjd = set_row[1] or target_mjd
        offset_hours = abs(target_mjd - mid_mjd) * 24.0

        staleness_by_type: dict[str, float] = {}
        warnings: list[str] = []

        if table_types:
            all_valid, validity_list, type_warnings = _validate_set_by_type(
                conn, chosen_set, target_mjd, table_types
            )
            warnings.extend(type_warnings)
            staleness_by_type = {v.table_type: v.staleness_hours for v in validity_list}

            if require_all_types_valid and not all_valid:
                # Log and try next candidate
                stale_types = [v for v in validity_list if not v.is_valid]
                logger.debug(
                    "Skipping set '%s': stale types %s",
                    chosen_set,
                    [(v.table_type, f"{v.staleness_hours:.1f}h") for v in stale_types],
                )
                continue

        paths = _get_set_paths(conn, chosen_set)
        return CalibrationSelection(
            set_name=chosen_set,
            paths=paths,
            mid_mjd=mid_mjd,
            selection_method="exact",
            time_offset_hours=offset_hours,
            quality_score=1.0 - (offset_hours / (search_validity * 2)),
            warnings=warnings,
            staleness_by_type=staleness_by_type,
        )

    # 2. BIDIRECTIONAL: Find nearest calibration (before OR after)
    if prefer_nearest:
        # Search window: target ± validity_hours
        search_min = target_mjd - validity_days
        search_max = target_mjd + validity_days

        # Find all sets with validity windows overlapping search range
        nearby_sets = conn.execute(
            f"""
            SELECT DISTINCT set_name,
                   (valid_start_mjd + COALESCE(valid_end_mjd, valid_start_mjd)) / 2.0 AS mid_mjd,
                   ABS((valid_start_mjd + COALESCE(valid_end_mjd, valid_start_mjd)) / 2.0 - ?) AS distance,
                   MAX(created_at) AS newest
            FROM caltables
            WHERE status = 'active'
              AND valid_start_mjd IS NOT NULL
              AND (
                  -- Set's validity window overlaps our search window
                  (valid_start_mjd <= ? AND (valid_end_mjd IS NULL OR valid_end_mjd >= ?))
                  OR
                  -- Set's midpoint is within search range
                  ((valid_start_mjd + COALESCE(valid_end_mjd, valid_start_mjd)) / 2.0 BETWEEN ? AND ?)
              )
              {type_filter}
            GROUP BY set_name
            ORDER BY distance ASC, newest DESC
            LIMIT 10
            """,
            (target_mjd, search_max, search_min, search_min, search_max),
        ).fetchall()

        # Try each candidate, validating by type
        for set_row in nearby_sets:
            chosen_set = set_row[0]
            mid_mjd = set_row[1]
            offset_hours = set_row[2] * 24.0

            staleness_by_type: dict[str, float] = {}
            warnings: list[str] = []

            if table_types:
                all_valid, validity_list, type_warnings = _validate_set_by_type(
                    conn, chosen_set, target_mjd, table_types
                )
                warnings.extend(type_warnings)
                staleness_by_type = {v.table_type: v.staleness_hours for v in validity_list}

                if require_all_types_valid and not all_valid:
                    stale_types = [v for v in validity_list if not v.is_valid]
                    logger.debug(
                        "Skipping set '%s': stale types %s",
                        chosen_set,
                        [(v.table_type, f"{v.staleness_hours:.1f}h") for v in stale_types],
                    )
                    continue

            # Determine selection method
            if mid_mjd < target_mjd:
                method = "nearest_before"
            else:
                method = "nearest_after"

            paths = _get_set_paths(conn, chosen_set)

            if offset_hours > search_validity / 2:
                warnings.append(
                    f"Calibration is {offset_hours:.1f}h from target "
                    f"(recommended max: {search_validity / 2:.1f}h)"
                )

            return CalibrationSelection(
                set_name=chosen_set,
                paths=paths,
                mid_mjd=mid_mjd,
                selection_method=method,
                time_offset_hours=offset_hours,
                quality_score=max(0.0, 1.0 - (offset_hours / search_validity)),
                warnings=warnings,
                staleness_by_type=staleness_by_type,
            )

    # No valid set found
    if table_types:
        raise ValueError(
            f"No calibration found within ±{search_validity:.1f}h of MJD {target_mjd:.6f} "
            f"with valid tables for types: {table_types}"
        )
    raise ValueError(f"No calibration found within ±{search_validity:.1f}h of MJD {target_mjd:.6f}")


def get_calibration_for_science(
    db_path: Path,
    target_mjd: float,
    *,
    require_bandpass: bool = True,
    require_gains: bool = True,
    prefer_nearest: bool = True,
) -> CalibrationSelection:
    """High-level wrapper to get calibration for science imaging.

        Requests both BP and G tables by default, enforces all-or-nothing validity
        (both must be valid within their respective windows: BP ±24h, G ±1h).

    Parameters
    ----------
    db_path : Path
        Path to calibration registry database
    target_mjd : float
        Target observation MJD
    require_bandpass : bool, optional
        If True, require valid BP table (within ±24h), by default True
    require_gains : bool, optional
        If True, require valid G table (within ±1h), by default True
    prefer_nearest : bool, optional
        If True, find nearest calibration even outside strict window, by default True

    Returns
    -------
        CalibrationSelection
        Selection with all required tables valid

    Raises
    ------
        ValueError
        If any required table type is stale, with detailed message about
        which type failed and by how much

    Examples
    --------
        >>> selection = get_calibration_for_science(db_path, 60617.5)
        >>> print(selection.staleness_by_type)
        {'BP': -12.5, 'G': -0.3}  # Negative = still valid by this many hours
    """
    table_types = []
    if require_bandpass:
        table_types.append("BP")
    if require_gains:
        table_types.append("G")

    if not table_types:
        raise ValueError("At least one of require_bandpass or require_gains must be True")

    try:
        return get_active_applylist_bidirectional(
            db_path,
            target_mjd,
            table_types=table_types,
            require_all_types_valid=True,
            prefer_nearest=prefer_nearest,
        )
    except ValueError as e:
        # Re-raise with more context
        logger.error("Failed to find valid calibration for science at MJD %.6f: %s", target_mjd, e)
        raise


# =============================================================================
# Issue #2: Calibration Interpolation Between Sets
# =============================================================================


@dataclass
class InterpolatedCalibration:
    """Result of interpolated calibration selection.

    When a target observation falls between two calibration observations,
    this provides both sets with appropriate temporal weights for combination.

    The weight_before value indicates how much weight to give the "before"
    calibration set (weight_after = 1.0 - weight_before). A weight of 0.5
    means the target is equidistant from both calibrations.

    """

    # Before calibration (earlier in time)
    set_before: str | None
    paths_before: list[str]
    mid_mjd_before: float | None

    # After calibration (later in time)
    set_after: str | None
    paths_after: list[str]
    mid_mjd_after: float | None

    # Interpolation weight for "before" set (0.0-1.0)
    # weight_after = 1.0 - weight_before
    weight_before: float

    # Metadata
    target_mjd: float
    selection_method: str  # 'single', 'interpolated', 'extrapolated'
    hours_from_before: float | None = None
    hours_from_after: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def weight_after(self) -> float:
        """Weight for the 'after' calibration set."""
        return 1.0 - self.weight_before

    @property
    def is_interpolated(self) -> bool:
        """True if this is a true interpolation (both before and after sets)."""
        return self.set_before is not None and self.set_after is not None

    @property
    def effective_paths(self) -> list[str]:
        """Get the primary paths for this selection.

        If interpolation is available, returns paths_before (the primary).
        The caller should handle weight-based combination separately.
        For single-set selection, returns the available paths.

        """
        if self.paths_before:
            return self.paths_before
        return self.paths_after


def get_interpolated_calibration(
    db_path: Path,
    target_mjd: float,
    *,
    validity_hours: float = DEFAULT_CAL_VALIDITY_HOURS,
    min_interpolation_gap_hours: float = 1.0,
) -> InterpolatedCalibration:
    """Find calibration sets for interpolation across calibrator observations.

        This fixes Issue #2: The pipeline only interpolates within a single
        calibration table's time axis, not between different calibration
        observations (e.g., yesterday's 3pm calibrator and today's 3pm calibrator).

        Strategy:
        1. Find the nearest calibration set BEFORE target_mjd
        2. Find the nearest calibration set AFTER target_mjd
        3. Calculate interpolation weights based on temporal distance
        4. Return both sets with weights for the caller to combine

        If only one set is found (extrapolation case), returns that set
        with weight=1.0.

    Parameters
    ----------
    db_path : Path
        Path to calibration registry database
    target_mjd : float
        Target observation MJD
    validity_hours : float, optional
        Maximum time offset for calibration validity, by default DEFAULT_CAL_VALIDITY_HOURS
    min_interpolation_gap_hours : float, optional
        Minimum gap between sets to use interpolation
        (if sets are very close, just use the nearest one), by default 1.0
    """
    conn = ensure_db(db_path)
    validity_days = validity_hours / 24.0
    search_min = target_mjd - validity_days
    search_max = target_mjd + validity_days

    # Find calibration set BEFORE target
    before_rows = conn.execute(
        """
        SELECT DISTINCT set_name,
               (valid_start_mjd + COALESCE(valid_end_mjd, valid_start_mjd)) / 2.0 AS mid_mjd,
               MAX(created_at) AS newest
        FROM caltables
        WHERE status = 'active'
          AND valid_start_mjd IS NOT NULL
          AND ((valid_start_mjd + COALESCE(valid_end_mjd, valid_start_mjd)) / 2.0) <= ?
          AND ((valid_start_mjd + COALESCE(valid_end_mjd, valid_start_mjd)) / 2.0) >= ?
        GROUP BY set_name
        ORDER BY mid_mjd DESC, newest DESC
        LIMIT 1
        """,
        (target_mjd, search_min),
    ).fetchall()

    # Find calibration set AFTER target
    after_rows = conn.execute(
        """
        SELECT DISTINCT set_name,
               (valid_start_mjd + COALESCE(valid_end_mjd, valid_start_mjd)) / 2.0 AS mid_mjd,
               MAX(created_at) AS newest
        FROM caltables
        WHERE status = 'active'
          AND valid_start_mjd IS NOT NULL
          AND ((valid_start_mjd + COALESCE(valid_end_mjd, valid_start_mjd)) / 2.0) > ?
          AND ((valid_start_mjd + COALESCE(valid_end_mjd, valid_start_mjd)) / 2.0) <= ?
        GROUP BY set_name
        ORDER BY mid_mjd ASC, newest DESC
        LIMIT 1
        """,
        (target_mjd, search_max),
    ).fetchall()

    set_before: str | None = None
    mid_mjd_before: float | None = None
    paths_before: list[str] = []

    set_after: str | None = None
    mid_mjd_after: float | None = None
    paths_after: list[str] = []

    if before_rows:
        set_before = before_rows[0][0]
        mid_mjd_before = before_rows[0][1]
        paths_before = _get_set_paths(conn, set_before)

    if after_rows:
        set_after = after_rows[0][0]
        mid_mjd_after = after_rows[0][1]
        paths_after = _get_set_paths(conn, set_after)

    # Calculate time offsets
    hours_from_before: float | None = None
    hours_from_after: float | None = None

    if mid_mjd_before is not None:
        hours_from_before = (target_mjd - mid_mjd_before) * 24.0
    if mid_mjd_after is not None:
        hours_from_after = (mid_mjd_after - target_mjd) * 24.0

    # Determine selection method and calculate weights
    warnings: list[str] = []
    weight_before: float = 1.0  # Default
    selection_method: str = "single"

    if set_before and set_after:
        # Both found - calculate interpolation weights
        total_gap_hours = (mid_mjd_after - mid_mjd_before) * 24.0  # type: ignore

        if total_gap_hours < min_interpolation_gap_hours:
            # Gap too small - just use nearest
            if hours_from_before <= hours_from_after:  # type: ignore
                weight_before = 1.0
                selection_method = "single"
            else:
                weight_before = 0.0
                selection_method = "single"
        else:
            # True interpolation
            # Weight inversely proportional to distance
            # If target is at mid_mjd_before, weight_before=1.0
            # If target is at mid_mjd_after, weight_before=0.0
            weight_before = hours_from_after / total_gap_hours  # type: ignore
            selection_method = "interpolated"

            # Warn if gap is large
            if total_gap_hours > 24.0:
                warnings.append(
                    f"Large gap between calibrations: {total_gap_hours:.1f}h. "
                    f"Interpolation quality may be degraded."
                )

    elif set_before:
        # Only before found - extrapolation forward
        weight_before = 1.0
        selection_method = "extrapolated"
        if hours_from_before is not None and hours_from_before > validity_hours / 2:
            warnings.append(
                f"Extrapolating {hours_from_before:.1f}h forward from calibration. "
                f"Consider observing a calibrator."
            )

    elif set_after:
        # Only after found - extrapolation backward
        weight_before = 0.0
        selection_method = "extrapolated"
        if hours_from_after is not None and hours_from_after > validity_hours / 2:
            warnings.append(
                f"Extrapolating {hours_from_after:.1f}h backward from calibration. "
                f"Consider observing a calibrator."
            )

    else:
        # No calibration found
        raise ValueError(
            f"No calibration found within ±{validity_hours:.1f}h of MJD {target_mjd:.6f}"
        )

    return InterpolatedCalibration(
        set_before=set_before,
        paths_before=paths_before,
        mid_mjd_before=mid_mjd_before,
        set_after=set_after,
        paths_after=paths_after,
        mid_mjd_after=mid_mjd_after,
        weight_before=weight_before,
        target_mjd=target_mjd,
        selection_method=selection_method,
        hours_from_before=hours_from_before,
        hours_from_after=hours_from_after,
        warnings=warnings,
    )


def _get_set_paths(conn: sqlite3.Connection, set_name: str) -> list[str]:
    """Get ordered paths for a calibration set.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection
    set_name : str
        Calibration set name
    """
    rows = conn.execute(
        "SELECT path FROM caltables WHERE set_name = ? AND status = 'active' ORDER BY order_index",
        (set_name,),
    ).fetchall()
    return [r[0] for r in rows]


def _calculate_quality_score(rows: list[tuple], offset_hours: float) -> float:
    """Calculate combined quality score from table metrics and time offset.

    Parameters
    ----------
    rows : List[tuple]
        Table metric rows
    offset_hours : float
        Time offset in hours
    """
    base_score = 1.0 - min(offset_hours / 24.0, 0.5)

    # Add quality metrics if available
    quality_scores = []
    for row in rows:
        metrics_json = row[3] if len(row) > 3 else None
        if metrics_json:
            try:
                metrics = json.loads(metrics_json)
                if "snr_median" in metrics:
                    # SNR > 50 is excellent, < 10 is poor
                    snr_score = min(metrics["snr_median"] / 50.0, 1.0)
                    quality_scores.append(snr_score)
                if "flagged_fraction" in metrics:
                    # Lower flagged fraction is better
                    flag_score = 1.0 - metrics["flagged_fraction"]
                    quality_scores.append(flag_score)
            except (json.JSONDecodeError, TypeError):
                pass

    if quality_scores:
        avg_quality = statistics.mean(quality_scores)
        return (base_score + avg_quality) / 2.0

    return base_score


# =============================================================================
# Issue #3: Calibrator Redundancy
# =============================================================================


@dataclass
class CalibratorCandidate:
    """Candidate calibrator for redundancy."""

    source_name: str
    ra_deg: float
    dec_deg: float
    flux_jy: float
    separation_deg: float
    is_primary: bool


def find_backup_calibrators(
    target_ra: float,
    target_dec: float,
    primary_calibrator: str,
    radius_deg: float = 10.0,
    min_flux_jy: float = 5.0,
) -> list[CalibratorCandidate]:
    """Find backup calibrators near a target field.

        This fixes Issue #3: Single point of failure for calibrators.

    Parameters
    ----------
    target_ra : float
        Target RA in degrees
    target_dec : float
        Target Dec in degrees
    primary_calibrator : str
        Name of primary calibrator (to mark/exclude)
    radius_deg : float, optional
        Search radius in degrees, by default 10.0
    min_flux_jy : float, optional
        Minimum flux in Jy, by default 5.0
    """
    # Identify redundant calibrators from known list (simplified)
    # In production, this would query a calibrator database
    known_calibrators = [
        {"name": "3C48", "ra": 24.4, "dec": 33.1, "flux": 16.0},
        {"name": "3C147", "ra": 85.6, "dec": 49.8, "flux": 22.0},
        {"name": "3C196", "ra": 124.1, "dec": 48.2, "flux": 14.0},
        {"name": "3C286", "ra": 202.8, "dec": 30.5, "flux": 15.0},
        {"name": "3C295", "ra": 212.8, "dec": 52.2, "flux": 21.0},
        {"name": "3C380", "ra": 277.4, "dec": 48.7, "flux": 13.0},
        {"name": "3C454.3", "ra": 343.5, "dec": 16.1, "flux": 10.0},
    ]

    candidates = []

    # Simple separation calculation (ignoring spherical trig complexity for now)
    import math

    for cal in known_calibrators:
        # Approximate separation
        d_ra = (cal["ra"] - target_ra) * math.cos(math.radians(target_dec))
        d_dec = cal["dec"] - target_dec
        sep = math.sqrt(d_ra**2 + d_dec**2)

        if sep <= radius_deg and cal["flux"] >= min_flux_jy:
            candidates.append(
                CalibratorCandidate(
                    source_name=cal["name"],
                    ra_deg=cal["ra"],
                    dec_deg=cal["dec"],
                    flux_jy=cal["flux"],
                    separation_deg=sep,
                    is_primary=(cal["name"] == primary_calibrator),
                )
            )

    # Sort by proximity
    candidates.sort(key=lambda x: x.separation_deg)
    return candidates


# =============================================================================
# Issue #6: Overlapping Calibration Handling
# =============================================================================


def check_calibration_overlap(
    db_path: Path,
    new_set_name: str,
    valid_start_mjd: float,
    valid_end_mjd: float | None,
    cal_field: str | None = None,
    refant: str | None = None,
) -> list[dict[str, Any]]:
    """Check for overlapping calibration sets before registration.

        This fixes Issue #6: Weak overlapping calibration handling.

        Returns list of conflicting sets with metadata.

    Parameters
    ----------
    db_path : Path
        Path to calibration database
    new_set_name : str
        Name of the new calibration set
    valid_start_mjd : float
        Start of validity window (MJD)
    valid_end_mjd : Optional[float], optional
        End of validity window (MJD), by default None
    cal_field : Optional[str], optional
        Calibration field name, by default None
    refant : Optional[str], optional
        Reference antenna, by default None
    """
    conn = ensure_db(db_path)

    # Find overlapping sets
    if valid_end_mjd is None:
        # Open-ended: conflicts with anything after start
        overlaps = conn.execute(
            """
            SELECT DISTINCT set_name, cal_field, refant,
                   MIN(valid_start_mjd) as set_start,
                   MAX(valid_end_mjd) as set_end
            FROM caltables
            WHERE status = 'active'
              AND set_name != ?
              AND (valid_end_mjd IS NULL OR valid_end_mjd >= ?)
            GROUP BY set_name
            """,
            (new_set_name, valid_start_mjd),
        ).fetchall()
    else:
        overlaps = conn.execute(
            """
            SELECT DISTINCT set_name, cal_field, refant,
                   MIN(valid_start_mjd) as set_start,
                   MAX(valid_end_mjd) as set_end
            FROM caltables
            WHERE status = 'active'
              AND set_name != ?
              AND (
                  (valid_start_mjd <= ? AND (valid_end_mjd IS NULL OR valid_end_mjd >= ?))
                  OR
                  (valid_start_mjd >= ? AND valid_start_mjd <= ?)
              )
            GROUP BY set_name
            """,
            (new_set_name, valid_end_mjd, valid_start_mjd, valid_start_mjd, valid_end_mjd),
        ).fetchall()

    conflicts = []
    for row in overlaps:
        conflict = {
            "set_name": row[0],
            "cal_field": row[1],
            "refant": row[2],
            "valid_start_mjd": row[3],
            "valid_end_mjd": row[4],
            "issues": [],
        }

        # Check for incompatibilities
        if refant and row[2] and refant != row[2]:
            conflict["issues"].append(f"Different reference antenna: {refant} vs {row[2]}")
        if cal_field and row[1] and cal_field != row[1]:
            conflict["issues"].append(f"Different calibrator field: {cal_field} vs {row[1]}")

        conflicts.append(conflict)

    return conflicts


def resolve_calibration_overlap(
    db_path: Path,
    new_set_name: str,
    valid_start_mjd: float,
    valid_end_mjd: float | None,
    *,
    strategy: str = "trim",  # 'trim', 'retire', 'error'
) -> None:
    """Resolve overlapping calibrations.

        Strategies:
        - trim
        Adjust validity windows of existing sets to not overlap
        - retire
        Retire overlapping sets entirely
        - error
        Raise error if overlap exists

    Parameters
    ----------
    db_path : Path
        Path to calibration database
    new_set_name : str
        Name of the new calibration set
    valid_start_mjd : float
        Start of validity window (MJD)
    valid_end_mjd : Optional[float]
        End of validity window (MJD)
    strategy : str, optional
        Resolution strategy ('trim', 'retire', 'error'), by default "trim"
    """
    conflicts = check_calibration_overlap(db_path, new_set_name, valid_start_mjd, valid_end_mjd)

    if not conflicts:
        return

    if strategy == "error":
        conflict_names = [c["set_name"] for c in conflicts]
        raise ValueError(
            f"Calibration overlap detected with sets: {conflict_names}. "
            f"Use strategy='trim' or 'retire' to resolve."
        )

    conn = ensure_db(db_path)

    for conflict in conflicts:
        if strategy == "retire":
            retire_caltable_set(
                db_path,
                conflict["set_name"],
                reason=f"Superseded by {new_set_name}",
            )
            logger.info(f"Retired overlapping set: {conflict['set_name']}")

        elif strategy == "trim":
            # Trim existing set's validity to end before new set starts
            conn.execute(
                """
                UPDATE caltables
                SET valid_end_mjd = ?
                WHERE set_name = ?
                  AND status = 'active'
                  AND (valid_end_mjd IS NULL OR valid_end_mjd > ?)
                """,
                (valid_start_mjd, conflict["set_name"], valid_start_mjd),
            )
            conn.commit()
            logger.info(
                f"Trimmed validity of {conflict['set_name']} to end at MJD {valid_start_mjd}"
            )
