"""Automated calibration table generation for DSA-110.

Ensures bandpass (.b) and gain (.g) calibration tables exist for a given date.
If they don't exist, selects the best primary flux calibrator with available
HDF5 data, converts to a Measurement Set, and runs the full calibration sequence.

The typical usage is::

    from dsa110_continuum.calibration.ensure import ensure_bandpass
    result = ensure_bandpass("2026-03-16", obs_dec_deg=16.1)
    print(result.bp_table, result.g_table)
    print(result.provenance)  # selection metadata dict
"""

from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astropy.coordinates import Angle
from astropy.time import Time

logger = logging.getLogger(__name__)

# Default paths — derived from central config so env vars override them
try:
    from dsa110_continuum.config import PathConfig as _PathConfig
    _default_paths = _PathConfig()
    DEFAULT_MS_DIR = str(_default_paths.ms_dir)
    DEFAULT_INPUT_DIR = str(_default_paths.incoming_dir)
    DEFAULT_DB_PATH = str(_default_paths.pipeline_db)
except ImportError:
    # Fallback for standalone use
    DEFAULT_MS_DIR = "/stage/dsa110-contimg/ms"
    DEFAULT_INPUT_DIR = "/data/incoming"
    DEFAULT_DB_PATH = "/data/dsa110-contimg/state/db/pipeline.sqlite3"

# Strip compatibility tolerance — reject reused tables whose calibrator Dec
# differs from the current observation strip by more than this.
STRIP_COMPAT_TOLERANCE_DEG = 5.0

# Bright-source fallback: minimum flux and candidate cap for VLA catalog query.
BRIGHT_FALLBACK_MIN_FLUX_JY = 5.0
BRIGHT_FALLBACK_MAX_CANDIDATES = 25
try:
    DEFAULT_VLA_CAL_DB = str(_default_paths.vla_cal_db)
except NameError:
    DEFAULT_VLA_CAL_DB = "/data/dsa110-contimg/state/catalogs/vla_calibrators.sqlite3"


@dataclass(frozen=True)
class CalibrationResult:
    """Result of calibration table lookup or generation."""

    bp_table: str
    g_table: str
    cal_date: str
    calibrator_name: str
    source: str  # "generated" | "existing" | "borrowed"
    provenance: dict[str, Any] = field(default_factory=dict)


class CalibrationError(RuntimeError):
    """Raised when calibration tables cannot be obtained."""


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def _parse_ra_deg(ra_j2000: str) -> float:
    """Parse RA in sexagesimal (e.g. '13h31m08.288s') to degrees."""
    return Angle(ra_j2000).deg


def _parse_dec_deg(dec_j2000: str) -> float:
    """Parse Dec in sexagesimal (e.g. '+30d30m32.96s') to degrees."""
    return Angle(dec_j2000).deg


# ---------------------------------------------------------------------------
# Provenance sidecar I/O
# ---------------------------------------------------------------------------


def provenance_sidecar_path(bp_table: str) -> str:
    """Return the path of the JSON provenance sidecar for a BP table.

    Convention: ``{bp_table}.cal_provenance.json``  (sits adjacent to the
    ``.b`` table directory/file).
    """
    return bp_table + ".cal_provenance.json"


def write_provenance_sidecar(bp_table: str, provenance: dict[str, Any]) -> str:
    """Write calibration-selection provenance to a JSON sidecar file.

    Returns the written path.
    """
    path = provenance_sidecar_path(bp_table)
    with open(path, "w") as f:
        json.dump(provenance, f, indent=2, default=str)
    logger.info("Cal provenance written: %s", path)
    return path


def load_provenance_sidecar(bp_table: str) -> dict[str, Any] | None:
    """Load provenance from the sidecar adjacent to *bp_table*.

    Returns ``None`` if the sidecar does not exist.  If the BP table is a
    symlink (borrowed), follows the symlink to the real table's sidecar.
    """
    # First try sidecar for the given path
    path = provenance_sidecar_path(bp_table)
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)

    # For borrowed (symlinked) tables, follow to the real table's sidecar
    if os.path.islink(bp_table):
        real_bp = os.path.realpath(bp_table)
        real_path = provenance_sidecar_path(real_bp)
        if os.path.isfile(real_path):
            with open(real_path) as f:
                return json.load(f)

    return None


def _build_provenance(
    *,
    selection_mode: str,
    obs_dec_deg_used: float | None,
    selection_dec_tolerance_deg: float,
    calibrator_name: str,
    calibrator_ra_deg: float,
    calibrator_dec_deg: float,
    calibrator_flux_jy: float,
    calibrator_dec_offset_deg: float | None,
    transit_time_iso: str,
    source: str,
    cal_date: str,
    bp_table: str,
    g_table: str,
    selection_pool: str = "primary",
    flux_anchor: str = "perley_butler_primary",
) -> dict[str, Any]:
    """Construct the provenance dict recorded in the sidecar and manifest."""
    return {
        "selection_mode": selection_mode,
        "selection_pool": selection_pool,
        "flux_anchor": flux_anchor,
        "obs_dec_deg_used": obs_dec_deg_used,
        "selection_dec_tolerance_deg": selection_dec_tolerance_deg,
        "calibrator_name": calibrator_name,
        "calibrator_ra_deg": round(calibrator_ra_deg, 5),
        "calibrator_dec_deg": round(calibrator_dec_deg, 5),
        "calibrator_flux_jy": calibrator_flux_jy,
        "calibrator_dec_offset_deg": (
            round(calibrator_dec_offset_deg, 2) if calibrator_dec_offset_deg is not None else None
        ),
        "transit_time_iso": transit_time_iso,
        "source": source,
        "cal_date": cal_date,
        "bp_table": bp_table,
        "g_table": g_table,
    }


# ---------------------------------------------------------------------------
# Strip compatibility validation
# ---------------------------------------------------------------------------


def _validate_strip_compatibility(
    result: CalibrationResult,
    obs_dec_deg: float | None,
    tolerance_deg: float = STRIP_COMPAT_TOLERANCE_DEG,
) -> CalibrationResult:
    """Validate that reused tables are compatible with the observation strip.

    The key provenance field is ``obs_dec_deg_used`` — the science strip Dec
    that was active when the tables were originally generated.  This is distinct
    from ``calibrator_dec_deg`` (the calibrator's sky position), which can be
    within tolerance of one strip yet incompatible with another.

    Policy:
    - **borrowed** tables with missing provenance → reject (CalibrationError)
    - **borrowed** tables with provenance but missing ``obs_dec_deg_used`` → reject
    - **borrowed** tables with provenance at incompatible strip Dec → reject
    - **existing** same-date tables with missing provenance → warn, allow
    - **existing** same-date tables with provenance but missing ``obs_dec_deg_used`` → warn, allow
    - **existing** same-date tables with provenance at incompatible strip Dec → reject

    Returns the *result* unchanged if validation passes.
    """
    if obs_dec_deg is None:
        # Cannot validate without knowing the observation Dec
        return result

    prov = load_provenance_sidecar(result.bp_table)

    if result.source == "borrowed":
        if prov is None:
            raise CalibrationError(
                f"Borrowed cal tables {result.bp_table} have no provenance sidecar — "
                "cannot verify strip compatibility. Generate fresh tables or add provenance."
            )
        stored_dec = prov.get("obs_dec_deg_used")
        if stored_dec is None:
            raise CalibrationError(
                f"Borrowed cal tables {result.bp_table} have provenance sidecar "
                "but missing obs_dec_deg_used — cannot verify strip compatibility."
            )
        if abs(stored_dec - obs_dec_deg) > tolerance_deg:
            raise CalibrationError(
                f"Borrowed cal tables from {result.cal_date} were generated for "
                f"strip Dec {stored_dec:.1f}° but current observation is at "
                f"Dec {obs_dec_deg:.1f}° (offset {abs(stored_dec - obs_dec_deg):.1f}° "
                f"> tolerance {tolerance_deg}°)."
            )

    elif result.source == "existing":
        if prov is None:
            logger.warning(
                "Existing cal tables %s have no provenance sidecar — "
                "cannot verify strip compatibility. Accepting for backward compatibility.",
                result.bp_table,
            )
            return result
        stored_dec = prov.get("obs_dec_deg_used")
        if stored_dec is None:
            logger.warning(
                "Existing cal tables %s have provenance but missing obs_dec_deg_used — "
                "cannot verify strip compatibility. Accepting for backward compatibility.",
                result.bp_table,
            )
            return result
        if abs(stored_dec - obs_dec_deg) > tolerance_deg:
            raise CalibrationError(
                f"Existing cal tables for {result.cal_date} were generated for "
                f"strip Dec {stored_dec:.1f}° but current observation is at "
                f"Dec {obs_dec_deg:.1f}° (offset {abs(stored_dec - obs_dec_deg):.1f}° "
                f"> tolerance {tolerance_deg}°)."
            )

    return result


def _enrich_result_provenance(result: CalibrationResult) -> CalibrationResult:
    """Merge sidecar provenance into *result*, overlaying runtime source/paths.

    For ``generated`` results the provenance is already populated at creation
    time, so this is a no-op.  For ``existing`` and ``borrowed`` results the
    sidecar (if present) carries the *original* generation metadata — we merge
    it but override ``source``, ``bp_table``, ``g_table``, and ``cal_date``
    with the runtime values so the manifest reflects the actual table provenance
    chain.
    """
    if result.provenance:
        return result  # Already populated (generated path)

    prov = load_provenance_sidecar(result.bp_table)
    if prov is None:
        return result  # No sidecar — nothing to merge

    merged = dict(prov)
    merged["source"] = result.source  # runtime source overrides original
    merged["bp_table"] = result.bp_table
    merged["g_table"] = result.g_table
    merged["cal_date"] = result.cal_date

    return CalibrationResult(
        bp_table=result.bp_table,
        g_table=result.g_table,
        cal_date=result.cal_date,
        calibrator_name=prov.get("calibrator_name", result.calibrator_name),
        source=result.source,
        provenance=merged,
    )


def validate_table_strip_compatibility(
    bp_table: str,
    obs_dec_deg: float | None,
    tolerance_deg: float = STRIP_COMPAT_TOLERANCE_DEG,
) -> None:
    """Validate strip compatibility for manually resolved calibration tables.

    This is the public entry point for callers outside ``ensure_bandpass()``
    (e.g. the fallback path in ``batch_pipeline.py``).  Determines whether
    the tables are symlinks (borrowed) or real files (existing), then applies
    the same validation policy as the internal ``_validate_strip_compatibility``.

    Raises
    ------
    CalibrationError
        If the tables are incompatible with the observation strip.
    """
    source = "borrowed" if os.path.islink(bp_table) else "existing"
    result = CalibrationResult(
        bp_table=bp_table,
        g_table=bp_table,  # placeholder — validation only reads bp sidecar
        cal_date="unknown",
        calibrator_name="unknown",
        source=source,
    )
    _validate_strip_compatibility(result, obs_dec_deg, tolerance_deg)


# ---------------------------------------------------------------------------
# Table discovery
# ---------------------------------------------------------------------------


def find_cal_tables(date: str, ms_dir: str = DEFAULT_MS_DIR) -> CalibrationResult | None:
    """Check for existing real (non-symlink) calibration tables for a date.

    Globs for ``{ms_dir}/{date}T*_0~23.b`` and ``.g`` to handle any transit
    time, not just the hardcoded T22:26:05.

    Parameters
    ----------
    date : str
        Date in YYYY-MM-DD format.
    ms_dir : str
        Measurement Set directory.

    Returns
    -------
    CalibrationResult or None
        Result if real tables are found, None otherwise.
    """
    bp_matches = sorted(glob.glob(os.path.join(ms_dir, f"{date}T*_0~23.b")))
    g_matches = sorted(glob.glob(os.path.join(ms_dir, f"{date}T*_0~23.g")))

    # Only accept real files, not symlinks
    real_bp = [p for p in bp_matches if not os.path.islink(p)]
    real_g = [p for p in g_matches if not os.path.islink(p)]

    if real_bp and real_g:
        logger.info("Found existing real cal tables: %s, %s", real_bp[0], real_g[0])
        return CalibrationResult(
            bp_table=real_bp[0],
            g_table=real_g[0],
            cal_date=date,
            calibrator_name="unknown",
            source="existing",
        )
    return None


def resolve_cal_table_paths(
    ms_dir: str, cal_date: str
) -> tuple[str, str]:
    """Find the bandpass and gain table paths for a cal date.

    Globs for ``{ms_dir}/{cal_date}T*_0~23.{b,g}`` to handle any transit
    timestamp.  Falls back to the legacy ``T22:26:05`` convention if no
    glob matches are found.

    Parameters
    ----------
    ms_dir : str
        Measurement Set directory.
    cal_date : str
        Calibration date (YYYY-MM-DD).

    Returns
    -------
    tuple of (bp_table_path, g_table_path)
    """
    bp_matches = sorted(glob.glob(os.path.join(ms_dir, f"{cal_date}T*_0~23.b")))
    g_matches = sorted(glob.glob(os.path.join(ms_dir, f"{cal_date}T*_0~23.g")))

    bp = bp_matches[0] if bp_matches else os.path.join(ms_dir, f"{cal_date}T22:26:05_0~23.b")
    g = g_matches[0] if g_matches else os.path.join(ms_dir, f"{cal_date}T22:26:05_0~23.g")
    return bp, g


# ---------------------------------------------------------------------------
# Calibrator selection
# ---------------------------------------------------------------------------


def _find_candidates_with_data(
    calibrator_list: list[tuple[str, float, float, float]],
    date: str,
    db_path: str,
    obs_dec_deg: float | None,
    dec_tolerance_deg: float,
) -> list[tuple[str, float, float, Time, float]]:
    """Evaluate which calibrators have date-matched HDF5 transit data.

    Parameters
    ----------
    calibrator_list :
        Each entry is ``(name, ra_deg, dec_deg, flux_jy)``.
    date :
        Observation date (YYYY-MM-DD).
    db_path :
        Pipeline SQLite database path.
    obs_dec_deg :
        Observation strip Dec (for filtering).
    dec_tolerance_deg :
        Maximum Dec offset from *obs_dec_deg*.

    Returns
    -------
    list of (name, ra_deg, dec_deg, transit_time, flux_jy) tuples.
    """
    from dsa110_continuum.calibration.transit import find_transits_for_source, next_transit_time

    day_start = Time(f"{date}T00:00:00", scale="utc")
    candidates: list[tuple[str, float, float, Time, float]] = []

    for cal_name, ra_deg, dec_deg, flux in calibrator_list:
        if obs_dec_deg is not None and abs(dec_deg - obs_dec_deg) > dec_tolerance_deg:
            logger.debug(
                "Skipping %s: Dec %.1f too far from obs Dec %.1f (tol %.1f)",
                cal_name, dec_deg, obs_dec_deg, dec_tolerance_deg,
            )
            continue

        transit = next_transit_time(ra_deg, day_start.mjd)
        if transit.iso[:10] != date:
            logger.debug("Skipping %s: transit %s not on %s", cal_name, transit.iso, date)
            continue

        if not os.path.isfile(db_path):
            logger.warning("Pipeline DB not found at %s; assuming data available", db_path)
            candidates.append((cal_name, ra_deg, dec_deg, transit, flux))
            continue

        matches = find_transits_for_source(
            db_path=db_path, ra_deg=ra_deg, dec_deg=dec_deg,
            ra_tolerance_deg=2.0, dec_tolerance_deg=2.0,
        )
        date_matches = [m for m in matches if m["group_id"].startswith(date)]
        if date_matches:
            logger.info(
                "Candidate %s: Dec %.1f, transit %s, %d HDF5 groups, %.1f Jy",
                cal_name, dec_deg, transit.iso, len(date_matches), flux,
            )
            candidates.append((cal_name, ra_deg, dec_deg, transit, flux))
        else:
            logger.debug("Skipping %s: no HDF5 data near transit on %s", cal_name, date)

    return candidates


def _rank_candidates(
    candidates: list[tuple[str, float, float, Time, float]],
    obs_dec_deg: float | None,
) -> list[tuple[str, float, float, Time, float]]:
    """Sort candidates by Dec proximity (primary) then flux (tiebreaker)."""
    if obs_dec_deg is not None:
        candidates.sort(key=lambda c: (abs(c[2] - obs_dec_deg), -c[4]))
    else:
        candidates.sort(key=lambda c: c[4], reverse=True)
    return candidates


def _get_bright_fallback_list(
    obs_dec_deg: float | None,
    dec_tolerance_deg: float,
    min_flux_jy: float = BRIGHT_FALLBACK_MIN_FLUX_JY,
    max_candidates: int = BRIGHT_FALLBACK_MAX_CANDIDATES,
    vla_cal_db: str = DEFAULT_VLA_CAL_DB,
) -> list[tuple[str, float, float, float]]:
    """Query VLA calibrator catalog for bright non-primary fallback candidates.

    Returns list of ``(name, ra_deg, dec_deg, flux_jy)`` tuples.
    """
    from pathlib import Path as _Path

    from dsa110_continuum.calibration.fluxscale import PRIMARY_FLUX_CALIBRATORS
    # Import at call site so unittest.mock.patch on the module attribute is in effect
    from dsa110_continuum.catalog import build_vla_calibrators as _bvc

    db = _Path(vla_cal_db)

    if obs_dec_deg is not None:
        search_dec = obs_dec_deg
        search_sep = dec_tolerance_deg
    else:
        search_dec = 0.0
        search_sep = 90.0  # full-sky search when no strip Dec specified

    try:
        rows = _bvc.query_calibrators_by_dec(
            dec_deg=search_dec,
            max_separation=search_sep,
            min_flux_jy=min_flux_jy,
            band="20cm",
            db_path=db,
            use_beam_weighting=False,
        )
    except (FileNotFoundError, OSError) as e:
        logger.warning("VLA calibrator DB unavailable (%s); bright fallback unavailable", e)
        return []

    primary_names = set(PRIMARY_FLUX_CALIBRATORS.keys())
    primary_alt_names: set[str] = set()
    for info in PRIMARY_FLUX_CALIBRATORS.values():
        primary_alt_names.update(info.get("alt_names", []))

    result: list[tuple[str, float, float, float]] = []
    for row in rows:
        name = row["name"]
        if name in primary_names or name in primary_alt_names:
            continue
        result.append((name, row["ra_deg"], row["dec_deg"], row["flux_jy"]))
        if len(result) >= max_candidates:
            break

    if obs_dec_deg is not None:
        logger.info(
            "Bright fallback pool: %d candidates from VLA catalog (min %.1f Jy, Dec %.1f ± %.1f)",
            len(result), min_flux_jy, search_dec, search_sep,
        )
    else:
        logger.info(
            "Bright fallback pool: %d candidates from VLA catalog "
            "(min %.1f Jy, full-sky Dec search)",
            len(result), min_flux_jy,
        )
    return result


def select_bandpass_calibrator(
    date: str,
    input_dir: str = DEFAULT_INPUT_DIR,
    db_path: str = DEFAULT_DB_PATH,
    obs_dec_deg: float | None = None,
    dec_tolerance_deg: float = 10.0,
    bright_fallback_min_flux_jy: float = BRIGHT_FALLBACK_MIN_FLUX_JY,
) -> tuple[str, float, float, Time]:
    """Auto-select the best calibrator with data on a given date.

    Two-phase selection:

    **Phase A — primary flux calibrators** (Perley-Butler 2017 models):
    preferred because they provide model-anchored absolute flux accuracy.

    **Phase B — bright VLA catalog sources** (fallback): used only when no
    primary calibrator has date-matched transit data.  These provide a
    correct bandpass shape but catalog-anchored (not model-anchored) flux.

    Parameters
    ----------
    date : str
        Observation date (YYYY-MM-DD).
    input_dir : str
        Directory containing HDF5 subband files.
    db_path : str
        Path to the pipeline SQLite database.
    obs_dec_deg : float or None
        Observed declination.  Calibrators are ranked by Dec proximity.
    dec_tolerance_deg : float
        Maximum Dec offset from *obs_dec_deg*.
    bright_fallback_min_flux_jy : float
        Minimum 20cm flux for VLA catalog fallback candidates.

    Returns
    -------
    tuple of (name, ra_deg, dec_deg, transit_time)

    Raises
    ------
    CalibrationError
        If no usable calibrator (primary or fallback) has data on the date.
    """
    from dsa110_continuum.calibration.fluxscale import PRIMARY_FLUX_CALIBRATORS

    # ── Phase A: primary calibrators ──────────────────────────────────────
    primary_list: list[tuple[str, float, float, float]] = []
    for cal_name, info in PRIMARY_FLUX_CALIBRATORS.items():
        primary_list.append((
            cal_name,
            _parse_ra_deg(info["ra_j2000"]),
            _parse_dec_deg(info["dec_j2000"]),
            info["flux_1400mhz_jy"],
        ))

    primary_candidates = _find_candidates_with_data(
        primary_list, date, db_path, obs_dec_deg, dec_tolerance_deg,
    )
    if primary_candidates:
        ranked = _rank_candidates(primary_candidates, obs_dec_deg)
        best = ranked[0]
        logger.info(
            "Selected PRIMARY calibrator: %s (Dec %.1f, %.1f Jy, transit %s)",
            best[0], best[2], best[4], best[3].iso,
        )
        return best[0], best[1], best[2], best[3]

    logger.warning(
        "No primary flux calibrator has HDF5 data on %s — "
        "searching bright VLA catalog sources as fallback",
        date,
    )

    # ── Phase B: bright VLA catalog fallback ──────────────────────────────
    fallback_list = _get_bright_fallback_list(
        obs_dec_deg, dec_tolerance_deg,
        min_flux_jy=bright_fallback_min_flux_jy,
    )

    if fallback_list:
        fallback_candidates = _find_candidates_with_data(
            fallback_list, date, db_path, obs_dec_deg, dec_tolerance_deg,
        )
        if fallback_candidates:
            ranked = _rank_candidates(fallback_candidates, obs_dec_deg)
            best = ranked[0]
            logger.info(
                "Selected BRIGHT FALLBACK calibrator: %s (Dec %.1f, %.1f Jy, transit %s)",
                best[0], best[2], best[4], best[3].iso,
            )
            return best[0], best[1], best[2], best[3]

    dec_msg = f" near Dec {obs_dec_deg:.1f} deg" if obs_dec_deg is not None else ""
    raise CalibrationError(
        f"No usable calibrator{dec_msg} has HDF5 data on {date}. "
        f"Primary pool: 0 candidates with date data. "
        f"Bright fallback (>= {bright_fallback_min_flux_jy} Jy): "
        f"0 candidates with date data. "
        f"Ensure the date is indexed: dsa110 index add --start {date} --end {date}"
    )


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------


def _lookup_calibrator_coords(
    calibrator_name: str,
) -> tuple[float, float, float, str, str]:
    """Look up RA, Dec, flux, selection_pool, and flux_anchor for a calibrator.

    Checks PRIMARY_FLUX_CALIBRATORS first; falls back to VLA catalog DB.
    Returns (ra_deg, dec_deg, flux_jy, selection_pool, flux_anchor).
    """
    from dsa110_continuum.calibration.fluxscale import PRIMARY_FLUX_CALIBRATORS

    info = PRIMARY_FLUX_CALIBRATORS.get(calibrator_name)
    if info is not None:
        return (
            _parse_ra_deg(info["ra_j2000"]),
            _parse_dec_deg(info["dec_j2000"]),
            info["flux_1400mhz_jy"],
            "primary",
            "perley_butler_primary",
        )

    # Not a primary — query VLA catalog
    from pathlib import Path as _Path

    db = _Path(DEFAULT_VLA_CAL_DB)
    if db.exists():
        import sqlite3

        with sqlite3.connect(str(db)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT c.ra_deg, c.dec_deg, f.flux_jy
                FROM calibrators c
                JOIN fluxes f ON c.name = f.name
                WHERE c.name = ? AND f.band = '20cm'
                LIMIT 1
                """,
                (calibrator_name,),
            ).fetchone()
            if row is not None:
                return (
                    row["ra_deg"], row["dec_deg"], row["flux_jy"],
                    "bright_fallback", "vla_catalog",
                )

    logger.warning(
        "Calibrator %s not found in primary list or VLA catalog — "
        "provenance will have zeroed coordinates",
        calibrator_name,
    )
    return 0.0, 0.0, 0.0, "unknown", "unknown"


def generate_bandpass_tables(
    date: str,
    calibrator_name: str,
    transit_time: Time,
    ms_dir: str = DEFAULT_MS_DIR,
    input_dir: str = DEFAULT_INPUT_DIR,
    db_path: str = DEFAULT_DB_PATH,
    refant: str = "103",
    obs_dec_deg: float | None = None,
    selection_dec_tolerance_deg: float = 10.0,
) -> CalibrationResult:
    """Generate bandpass and gain tables from a calibrator transit.

    Steps:
    1. Convert HDF5 data near the transit to a calibrator MS
    2. Run the full calibration sequence (phaseshift, model, bandpass, gains)
    3. Write provenance sidecar (distinguishes primary vs fallback pool)

    Parameters
    ----------
    date : str
        Observation date (YYYY-MM-DD).
    calibrator_name : str
        Calibrator name (e.g. "3C454.3", "3C286", "J0834+555").
    transit_time : Time
        Meridian transit time of the calibrator.
    ms_dir : str
        Measurement Set directory.
    input_dir : str
        HDF5 input directory.
    db_path : str
        Pipeline database path.
    refant : str
        Reference antenna.
    obs_dec_deg : float or None
        Observation Dec used for selection (recorded in provenance).
    selection_dec_tolerance_deg : float
        Dec tolerance used during selection (recorded in provenance).

    Returns
    -------
    CalibrationResult

    Raises
    ------
    CalibrationError
        If conversion or calibration fails.
    """
    from dsa110_continuum.calibration.runner import run_calibrator
    from dsa110_continuum.conversion.calibrator_ms_generator import CalibratorMSGenerator

    logger.info(
        "Generating bandpass tables: date=%s, calibrator=%s, transit=%s",
        date, calibrator_name, transit_time.iso,
    )

    # Look up calibrator coords/flux and determine pool
    cal_ra, cal_dec, cal_flux, sel_pool, flux_anchor = _lookup_calibrator_coords(
        calibrator_name,
    )

    # Step 1: Convert HDF5 -> calibrator MS
    generator = CalibratorMSGenerator(
        input_dir=Path(input_dir),
        output_dir=Path(ms_dir),
        db_path=Path(db_path),
    )

    result = generator.generate_from_transit(
        calibrator_name=calibrator_name,
        transit_time=transit_time,
        window_minutes=30,
        verify=True,
    )

    if not result.success or result.ms_path is None:
        raise CalibrationError(
            f"Failed to generate calibrator MS for {calibrator_name} "
            f"on {date}: {result.error_message}"
        )

    ms_path = str(result.ms_path)
    logger.info("Calibrator MS generated: %s", ms_path)

    # Step 2: Run full calibration sequence
    ms_basename = os.path.splitext(os.path.basename(ms_path))[0]
    table_prefix = os.path.join(ms_dir, f"{ms_basename}_0~23")

    tables = run_calibrator(
        ms_path=ms_path,
        cal_field="0~23",
        refant=refant,
        do_flagging=True,
        do_k=False,
        table_prefix=table_prefix,
        calibrator_name=calibrator_name,
        do_phaseshift=True,
    )

    # Find the .b and .g among the returned tables
    bp_table = None
    g_table = None
    for t in tables:
        if t.endswith(".b"):
            bp_table = t
        elif t.endswith(".g"):
            g_table = t

    if bp_table is None:
        raise CalibrationError(
            f"Calibration completed but no .b table found. Tables: {tables}"
        )
    if g_table is None:
        raise CalibrationError(
            f"Calibration completed but no .g table found. Tables: {tables}"
        )

    logger.info("Bandpass table: %s", bp_table)
    logger.info("Gain table: %s", g_table)

    # Step 3: Write provenance sidecar
    dec_offset = abs(cal_dec - obs_dec_deg) if obs_dec_deg is not None else None
    prov = _build_provenance(
        selection_mode="dec_aware" if obs_dec_deg is not None else "brightest",
        obs_dec_deg_used=obs_dec_deg,
        selection_dec_tolerance_deg=selection_dec_tolerance_deg,
        calibrator_name=calibrator_name,
        calibrator_ra_deg=cal_ra,
        calibrator_dec_deg=cal_dec,
        calibrator_flux_jy=cal_flux,
        calibrator_dec_offset_deg=dec_offset,
        transit_time_iso=transit_time.iso,
        source="generated",
        cal_date=date,
        bp_table=bp_table,
        g_table=g_table,
        selection_pool=sel_pool,
        flux_anchor=flux_anchor,
    )
    write_provenance_sidecar(bp_table, prov)

    return CalibrationResult(
        bp_table=bp_table,
        g_table=g_table,
        cal_date=date,
        calibrator_name=calibrator_name,
        source="generated",
        provenance=prov,
    )


# ---------------------------------------------------------------------------
# Borrowing from nearby dates
# ---------------------------------------------------------------------------


def _find_nearest_real_tables(
    date: str, ms_dir: str, max_borrow_days: int
) -> CalibrationResult | None:
    """Find the nearest date with real (non-symlink) .b tables and borrow them.

    Searches up to *max_borrow_days* in both directions from *date*.

    Returns
    -------
    CalibrationResult or None
    """
    from datetime import datetime, timedelta

    base = datetime.strptime(date, "%Y-%m-%d")

    for offset in range(1, max_borrow_days + 1):
        for direction in (-1, 1):
            check_date = (base + timedelta(days=direction * offset)).strftime("%Y-%m-%d")
            result = find_cal_tables(check_date, ms_dir)
            if result is not None:
                logger.info(
                    "Borrowing cal tables from %s (offset %+d days)",
                    check_date, direction * offset,
                )
                # Create symlinks for the target date
                bp_link = result.bp_table.replace(check_date, date)
                g_link = result.g_table.replace(check_date, date)

                if not os.path.exists(bp_link):
                    os.symlink(result.bp_table, bp_link)
                    logger.info("Created symlink: %s -> %s", bp_link, result.bp_table)
                if not os.path.exists(g_link):
                    os.symlink(result.g_table, g_link)
                    logger.info("Created symlink: %s -> %s", g_link, result.g_table)

                return CalibrationResult(
                    bp_table=bp_link,
                    g_table=g_link,
                    cal_date=check_date,
                    calibrator_name=result.calibrator_name,
                    source="borrowed",
                )
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def ensure_bandpass(
    date: str,
    ms_dir: str = DEFAULT_MS_DIR,
    input_dir: str = DEFAULT_INPUT_DIR,
    db_path: str = DEFAULT_DB_PATH,
    refant: str = "103",
    max_borrow_days: int = 30,
    force: bool = False,
    obs_dec_deg: float | None = None,
) -> CalibrationResult:
    """Ensure bandpass and gain calibration tables exist for a date.

    Strategy:
    1. Check for existing real tables (skip if *force* is True)
       - Validate strip compatibility via provenance sidecar
    2. Auto-select the best primary flux calibrator with data
       (preferring calibrators near *obs_dec_deg* for beam-model fidelity)
    3. Generate tables from calibrator transit, write provenance
    4. If no calibrator data available, borrow from nearest date
       - Validate strip compatibility via provenance sidecar
    5. Raise CalibrationError if all strategies fail

    Parameters
    ----------
    date : str
        Observation date (YYYY-MM-DD).
    ms_dir : str
        Measurement Set directory.
    input_dir : str
        HDF5 input directory.
    db_path : str
        Pipeline database path.
    refant : str
        Reference antenna.
    max_borrow_days : int
        Maximum number of days to search for borrowable tables.
    force : bool
        If True, regenerate tables even if they already exist.
    obs_dec_deg : float or None
        Observed declination in degrees.  When provided, the calibrator
        selection prefers sources closest to this Dec, and reused tables
        are validated for strip compatibility.

    Returns
    -------
    CalibrationResult

    Raises
    ------
    CalibrationError
        If no calibration tables can be obtained, or if reused tables
        are incompatible with the current observation strip.
    """
    logger.info(
        "ensure_bandpass: date=%s, ms_dir=%s, obs_dec=%.1f deg, force=%s",
        date, ms_dir, obs_dec_deg if obs_dec_deg is not None else float("nan"), force,
    )

    # 1. Check for existing real tables
    if not force:
        existing = find_cal_tables(date, ms_dir)
        if existing is not None:
            existing = _validate_strip_compatibility(existing, obs_dec_deg)
            existing = _enrich_result_provenance(existing)
            return existing

    # 2. Try to generate from calibrator transit
    try:
        cal_name, ra_deg, dec_deg, transit = select_bandpass_calibrator(
            date, input_dir=input_dir, db_path=db_path, obs_dec_deg=obs_dec_deg,
        )
        return generate_bandpass_tables(
            date=date,
            calibrator_name=cal_name,
            transit_time=transit,
            ms_dir=ms_dir,
            input_dir=input_dir,
            db_path=db_path,
            refant=refant,
            obs_dec_deg=obs_dec_deg,
        )
    except CalibrationError as exc:
        logger.warning("Cannot generate tables for %s: %s", date, exc)
    except Exception as exc:
        logger.error("Unexpected error generating tables for %s: %s", date, exc)

    # 3. Borrow from nearest date with real tables
    borrowed = _find_nearest_real_tables(date, ms_dir, max_borrow_days)
    if borrowed is not None:
        borrowed = _validate_strip_compatibility(borrowed, obs_dec_deg)
        borrowed = _enrich_result_provenance(borrowed)
        return borrowed

    raise CalibrationError(
        f"No calibration tables available for {date} and no nearby date "
        f"has real tables within {max_borrow_days} days."
    )
