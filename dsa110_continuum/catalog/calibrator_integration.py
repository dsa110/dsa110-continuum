"""Calibrator detection and ranking for DSA-110 observations.

Matches pipeline observations to the VLA calibrator catalog using proper
spherical coordinate separation (astropy SkyCoord), then ranks candidates
by a transparent composite score based on flux, quality, astrometric
accuracy, and proximity to the pointing.
"""

import math

from astropy.coordinates import SkyCoord
import astropy.units as u

from dsa110_contimg.core.calibration.catalogs import (
    load_vla_catalog_from_sqlite,
    resolve_vla_catalog_path,
)
from dsa110_contimg.infrastructure.database.unified import get_db


# =============================================================================
# Calibrator Scoring Utilities
# =============================================================================

# VLA epoch code pattern: 3-letter month + 2-digit year (e.g., "Aug01", "Feb97")
_VLA_EPOCH_MONTHS = {
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
}


def is_real_alias(alt_name: str | None) -> bool:
    """Filter out bogus VLA catalog alt_name values.

    The VLA calibrator catalog stores heterogeneous data in the alt_name
    column: real source aliases (3C454.3, NGC315, BLLAC), VLA flux
    measurement epoch codes (Aug01, Feb97), and declination strings
    (-06d29'35.888"). Only real aliases should be displayed.
    """
    if not alt_name:
        return False
    # Declination strings contain degree-minute-second notation
    if '"' in alt_name or "d" in alt_name and "'" in alt_name:
        return False
    # VLA epoch codes: 3-letter month + 2-digit year
    if len(alt_name) == 5 and alt_name[:3] in _VLA_EPOCH_MONTHS and alt_name[3:].isdigit():
        return False
    # Known non-alias codes
    if alt_name in {"AW574", "CJ2", "GVAR", "JVAS", "USNO", "VERA", "VSOP"}:
        return False
    return True


def compute_quality_score(quality_codes: str | None) -> tuple[float, str]:
    """Compute calibrator quality score relevant to DSA-110.

    VLA quality codes are 4 characters, one per array config [A, B, C, D]:
      P = Primary calibrator (excellent)
      S = Secondary calibrator (good)
      X = Use with caution (variable, extended, or weak)
      W = Weak source

    DSA-110 has ~1.2 km baselines at 1.4 GHz, giving ~36" resolution.
    This is between VLA C-config (~14") and D-config (~46"), so the
    C and D quality codes are most relevant. A/B codes (long baselines,
    ~1-5") matter less for DSA-110.

    Returns
    -------
    tuple[float, str]
        (score, summary) where score is 0.0-1.0 and summary is human-readable.
    """
    if not quality_codes or len(quality_codes) < 4:
        return 0.5, "? (no data)"

    code_scores = {"P": 1.0, "S": 0.75, "X": 0.25, "W": 0.1}
    # Positions: [0]=A-config, [1]=B-config, [2]=C-config, [3]=D-config
    # Weights: C/D dominant for DSA-110's resolution
    weights = [0.05, 0.15, 0.40, 0.40]  # A, B, C, D

    score = sum(
        weight * code_scores.get(quality_codes[idx], 0.5)
        for idx, weight in enumerate(weights)
    )

    # Human-readable summary from the C/D codes that matter
    cd_codes = quality_codes[2:4]
    if cd_codes in ("PP", "PS", "SP"):
        summary = f"{quality_codes} (excellent for DSA-110)"
    elif "P" in cd_codes or cd_codes == "SS":
        summary = f"{quality_codes} (good)"
    elif "X" not in cd_codes and "W" not in cd_codes:
        summary = f"{quality_codes} (ok)"
    elif cd_codes.count("X") == 1:
        summary = f"{quality_codes} (marginal — extended?)"
    else:
        summary = f"{quality_codes} (poor at DSA-110 resolution)"

    return score, summary


def rank_calibrators_near_position(
    ra_deg: float | None,
    dec_deg: float,
    search_radius_deg: float = 2.0,
    num_sources: int = 5,
    catalog_path: str | None = None,
    min_flux_jy: float = 0.0,
) -> list[dict]:
    """Find and rank VLA calibrators near a sky position.

    Uses proper spherical coordinate matching (astropy SkyCoord) and
    composite scoring incorporating flux, VLA quality codes, astrometric
    position quality, and angular proximity.

    This is the canonical calibrator ranking function for DSA-110.  All
    pre-MS calibrator lookup systems should delegate to this function
    rather than implementing their own matching/scoring.

    Parameters
    ----------
    ra_deg : float or None
        Right ascension of the target position in degrees.  If None,
        uses Dec-only filtering (all calibrators within
        ``search_radius_deg`` of ``dec_deg`` regardless of RA).  The
        separation score is set to 1.0 in this mode since angular
        proximity is not meaningful without RA.
    dec_deg : float
        Declination of the target position in degrees.
    search_radius_deg : float
        Maximum angular separation for calibrator match (default: 2.0).
        When ``ra_deg`` is None, this is the maximum Dec difference.
    num_sources : int
        Maximum number of calibrator candidates to return (default: 5).
    catalog_path : str or None
        Path to VLA calibrator SQLite database.  If None, uses automatic
        resolution via ``resolve_vla_catalog_path()``.
    min_flux_jy : float
        Minimum flux at 20 cm to include (default: 0.0, no filter).

    Returns
    -------
    list[dict]
        Ranked calibrator dicts (best first), each with keys:
        name, ra_deg, dec_deg, flux_jy, alt_name, position_code,
        quality_codes, quality_summary, quality_score, position_score,
        separation_deg, separation_score, composite_score, rank.
    """
    # Load VLA catalog
    if catalog_path is None:
        cat_path = str(resolve_vla_catalog_path())
    else:
        cat_path = catalog_path
    catalog = load_vla_catalog_from_sqlite(cat_path)

    have_ra = ra_deg is not None
    if have_ra:
        target_coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")

    scored: list[dict] = []
    for cal_name, cal_data in catalog.iterrows():
        cal_ra = cal_data["ra_deg"]
        cal_dec = cal_data["dec_deg"]

        if have_ra:
            # Proper spherical coordinate separation
            cal_coord = SkyCoord(ra=cal_ra * u.deg, dec=cal_dec * u.deg, frame="icrs")
            separation = target_coord.separation(cal_coord).deg
        else:
            # Dec-only mode: use absolute Dec difference
            separation = abs(float(cal_dec) - dec_deg)

        if separation > search_radius_deg:
            continue

        flux_jy = float(cal_data.get("flux_jy", 0) or 0)
        if flux_jy < min_flux_jy:
            continue

        position_code = cal_data.get("position_code") or "T"

        # Quality score: weighted for DSA-110 resolution (~36")
        quality_score, quality_summary = compute_quality_score(
            cal_data.get("quality_codes")
        )

        # Position score: astrometric quality from VLA catalog
        pos_scores = {"A": 1.0, "B": 0.9, "C": 0.7, "T": 0.5}
        position_score = pos_scores.get(position_code, 0.5)

        # Flux score: log-compressed, always non-negative
        flux_score = math.log10(max(flux_jy, 0.001) + 1.0)

        # Separation score
        if have_ra:
            # Linear decay from 1.0 at pointing to 0.0 at edge
            separation_score = max(0.0, 1.0 - (separation / search_radius_deg))
        else:
            # Without RA, separation is Dec-only — still useful as proximity
            separation_score = max(0.0, 1.0 - (separation / search_radius_deg))

        # Composite score (multiplicative)
        # - flux × quality × position are hard factors
        # - separation is a soft modifier (50% floor)
        composite = (
            flux_score
            * quality_score
            * position_score
            * (0.5 + 0.5 * separation_score)
        )

        # Filter bogus aliases
        alt_name_raw = cal_data.get("alt_name")
        clean_alt = alt_name_raw if is_real_alias(alt_name_raw) else None

        scored.append({
            "name": cal_name,
            "ra_deg": round(float(cal_ra), 4),
            "dec_deg": round(float(cal_dec), 4),
            "flux_jy": round(flux_jy, 2),
            "alt_name": clean_alt,
            "position_code": position_code,
            "quality_codes": cal_data.get("quality_codes"),
            "quality_summary": quality_summary,
            "quality_score": round(quality_score, 3),
            "position_score": round(position_score, 2),
            "separation_deg": round(separation, 4),
            "separation_score": round(separation_score, 4),
            "composite_score": round(composite, 4),
        })

    # Sort by composite score descending, assign ranks
    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    for rank, cal in enumerate(scored[:num_sources], 1):
        cal["rank"] = rank

    return scored[:num_sources]


def detect_calibrator_from_observations(
    start_time: str,
    end_time: str,
    ra_tolerance_deg: float = 2.0,
    dec_tolerance_deg: float = 2.0,
    num_sources: int = 5,
) -> dict:
    """Detect which calibrator was observed during a time range.

    Queries the pipeline database for HDF5 observation groups in the
    given time window, then delegates to ``rank_calibrators_near_position``
    for spherical matching and composite scoring.  Adds an ``obs_fraction``
    modifier that rewards calibrators seen consistently across groups.

    Parameters
    ----------
    start_time : str
        ISO8601 start time
    end_time : str
        ISO8601 end time
    ra_tolerance_deg : float
        RA matching tolerance in degrees (default: 2.0)
    dec_tolerance_deg : float
        Dec matching tolerance in degrees (default: 2.0)
    num_sources : int
        Maximum number of calibrator candidates to return (default: 5)

    Returns
    -------
    dict
        Results with keys:
        - calibrators: list of dict with calibrator matches (ranked)
        - observations: list of dict with observation info
        - best_match: str or None (calibrator name)
        - best_match_alt: str or None (common alias like 3C454.3)
        - dec_deg: float (average pointing Dec)
        - file_count: int (total HDF5 files)
        - group_count: int (observation groups)
        - errors: list of str
    """
    result = {
        "calibrators": [],
        "observations": [],
        "best_match": None,
        "best_match_alt": None,
        "dec_deg": None,
        "file_count": 0,
        "group_count": 0,
        "errors": [],
    }

    try:
        # Get database connection
        pipeline_db = get_db()

        # Query observations in time range with pointing information
        query = """
            SELECT
                group_id,
                timestamp_iso,
                ra_deg,
                dec_deg,
                COUNT(*) as file_count
            FROM hdf5_files
            WHERE timestamp_iso >= ?
                AND timestamp_iso <= ?
                AND ra_deg IS NOT NULL
                AND dec_deg IS NOT NULL
            GROUP BY group_id
            ORDER BY timestamp_iso
        """

        cursor = pipeline_db.conn.execute(query, (start_time, end_time))
        rows = cursor.fetchall()

        if not rows:
            result["errors"].append(
                f"No observations found with pointing information "
                f"for {start_time} to {end_time}"
            )
            result["errors"].append(
                "The pipeline database has NULL ra_deg/dec_deg for this time range. "
                "Re-index with updated metadata extraction or backfill with "
                "scripts/ops/maintenance/backfill_hdf5_radec.py."
            )
            return result

        # Store observation info and compute aggregate pointing
        total_files = 0
        dec_values = []
        ra_values = []
        for row in rows:
            result["observations"].append(
                {
                    "group_id": row[0],
                    "timestamp": row[1],
                    "ra_deg": row[2],
                    "dec_deg": row[3],
                    "file_count": row[4],
                }
            )
            total_files += row[4]
            ra_values.append(row[2])
            dec_values.append(row[3])

        avg_ra = sum(ra_values) / len(ra_values)
        avg_dec = sum(dec_values) / len(dec_values)
        result["dec_deg"] = round(avg_dec, 4)
        result["file_count"] = total_files
        result["group_count"] = len(rows)

        # Resolve catalog path once
        try:
            catalog_path = str(resolve_vla_catalog_path())
        except FileNotFoundError as catalog_err:
            result["errors"].append(
                f"VLA calibrator catalog not found: {catalog_err}. "
                "Cannot match observations to calibrators."
            )
            return result

        search_radius = max(ra_tolerance_deg, dec_tolerance_deg)

        # Use the canonical ranking function per observation group,
        # then aggregate observation counts across groups.
        calibrator_votes: dict[str, dict] = {}
        for obs in result["observations"]:
            group_cals = rank_calibrators_near_position(
                ra_deg=obs["ra_deg"],
                dec_deg=obs["dec_deg"],
                search_radius_deg=search_radius,
                num_sources=100,  # get all matches, we'll trim later
                catalog_path=catalog_path,
            )
            for cal in group_cals:
                name = cal["name"]
                if name not in calibrator_votes:
                    calibrator_votes[name] = {**cal, "observation_count": 0}
                entry = calibrator_votes[name]
                entry["observation_count"] += 1
                # Keep the closest separation seen
                if cal["separation_deg"] < entry["separation_deg"]:
                    for k in cal:
                        if k != "observation_count":
                            entry[k] = cal[k]

        # Re-score with obs_fraction modifier
        group_count = result["group_count"]
        scored = []
        for cal in calibrator_votes.values():
            obs_fraction = cal["observation_count"] / group_count
            # Apply obs_fraction as soft modifier (50% floor) on top of
            # the base composite score from rank_calibrators_near_position
            adjusted = cal["composite_score"] * (0.5 + 0.5 * obs_fraction)
            cal["composite_score"] = round(adjusted, 4)
            scored.append(cal)

        # Sort by adjusted composite score, assign ranks
        scored.sort(key=lambda x: x["composite_score"], reverse=True)
        for rank, cal in enumerate(scored[:num_sources], 1):
            cal["rank"] = rank

        result["calibrators"] = scored[:num_sources]

        if result["calibrators"]:
            best = result["calibrators"][0]
            result["best_match"] = best["name"]
            result["best_match_alt"] = best.get("alt_name")

    except Exception as exc:
        result["errors"].append(f"Calibrator detection failed: {exc}")

    return result

