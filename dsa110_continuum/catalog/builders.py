"""
Build per-declination strip SQLite databases from source catalogs.

These databases are optimized for fast spatial queries during long-term
drift scan operations at fixed declinations.

Ported from dsa110_contimg.core.catalog.builders.
Only strip-from-full functions are included; download functions are omitted.
"""

from __future__ import annotations

import fcntl
import logging
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Hardcoded base paths (replaces get_env_path)
_CATALOG_DIR = Path("/data/dsa110-contimg/state/catalogs")

# Catalog coverage limits (declination ranges)
CATALOG_COVERAGE_LIMITS = {
    "nvss": {"dec_min": -40.0, "dec_max": 90.0},
    "first": {"dec_min": -40.0, "dec_max": 90.0},
    "rax": {"dec_min": -90.0, "dec_max": 49.9},
    "vlass": {"dec_min": -40.0, "dec_max": 90.0},
    "atnf": {"dec_min": -90.0, "dec_max": 90.0},
}

# Full database paths
NVSS_FULL_DB_PATH = _CATALOG_DIR / "nvss_full.sqlite3"
FIRST_FULL_DB_PATH = _CATALOG_DIR / "first_full.sqlite3"
VLASS_FULL_DB_PATH = _CATALOG_DIR / "vlass_full.sqlite3"
RAX_FULL_DB_PATH = _CATALOG_DIR / "rax_full.sqlite3"
ATNF_FULL_DB_PATH = _CATALOG_DIR / "atnf_full.sqlite3"


def get_nvss_full_db_path() -> Path:
    return NVSS_FULL_DB_PATH


def get_first_full_db_path() -> Path:
    return FIRST_FULL_DB_PATH


def get_vlass_full_db_path() -> Path:
    return VLASS_FULL_DB_PATH


def get_rax_full_db_path() -> Path:
    return RAX_FULL_DB_PATH


def get_atnf_full_db_path() -> Path:
    return ATNF_FULL_DB_PATH


def _acquire_db_lock(
    lock_path: Path, timeout_sec: float = 300.0, max_retries: int = 10
) -> int | None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(lock_path, "w")
    start_time = time.time()
    retry_count = 0

    while retry_count < max_retries:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock_file.fileno()
        except BlockingIOError:
            elapsed = time.time() - start_time
            if elapsed > timeout_sec:
                logger.warning(f"Timeout waiting for database lock {lock_path}")
                lock_file.close()
                return None
            wait_time = min(2.0**retry_count, 10.0)
            time.sleep(wait_time)
            retry_count += 1
        except Exception as e:
            logger.error(f"Error acquiring database lock {lock_path}: {e}")
            lock_file.close()
            return None

    lock_file.close()
    return None


def _release_db_lock(lock_fd: int | None, lock_path: Path):
    if lock_fd is not None:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except Exception as e:
            logger.warning(f"Error releasing database lock {lock_path}: {e}")
    try:
        if lock_path.exists():
            lock_path.unlink()
    except Exception as e:
        logger.warning(f"Error removing lock file {lock_path}: {e}")


# --------------------------------------------------------------------------
# Strip-from-full builders
# --------------------------------------------------------------------------

def build_nvss_strip_from_full(
    dec_center: float,
    dec_range: tuple[float, float],
    output_path: Path | None = None,
    min_flux_mjy: float | None = None,
    full_db_path: Path | None = None,
) -> Path:
    """Build NVSS declination strip database from the full NVSS database."""
    dec_min, dec_max = dec_range

    if full_db_path is None:
        full_db_path = get_nvss_full_db_path()

    if not full_db_path.exists():
        raise FileNotFoundError(f"Full NVSS database not found: {full_db_path}")

    if output_path is None:
        dec_rounded = round(dec_center, 1)
        output_path = _CATALOG_DIR / f"nvss_dec{dec_rounded:+.1f}.sqlite3"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.info(f"Dec strip database already exists: {output_path}")
        return output_path

    lock_path = output_path.with_suffix(".lock")
    lock_fd = _acquire_db_lock(lock_path)

    if lock_fd is None:
        if output_path.exists():
            return output_path
        raise RuntimeError(f"Could not acquire lock for {output_path}")

    try:
        if output_path.exists():
            return output_path

        logger.info(f"Building NVSS dec strip: {dec_min:.2f}° to {dec_max:.2f}°")

        with sqlite3.connect(str(full_db_path)) as src_conn:
            query = "SELECT ra_deg, dec_deg, flux_mjy FROM sources WHERE dec_deg >= ? AND dec_deg <= ?"
            params: list = [dec_min, dec_max]
            if min_flux_mjy is not None:
                query += " AND flux_mjy >= ?"
                params.append(min_flux_mjy)
            rows = src_conn.execute(query, params).fetchall()

        logger.info(f"Found {len(rows)} sources in dec range")

        with sqlite3.connect(str(output_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE sources (
                    source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ra_deg REAL NOT NULL,
                    dec_deg REAL NOT NULL,
                    flux_mjy REAL,
                    UNIQUE(ra_deg, dec_deg)
                )
            """)
            conn.execute("CREATE INDEX idx_radec ON sources(ra_deg, dec_deg)")
            conn.execute("CREATE INDEX idx_dec ON sources(dec_deg)")
            conn.execute("CREATE INDEX idx_flux ON sources(flux_mjy)")
            conn.executemany(
                "INSERT OR IGNORE INTO sources(ra_deg, dec_deg, flux_mjy) VALUES(?, ?, ?)", rows
            )
            conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
            conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", [
                ("dec_center", str(dec_center)),
                ("dec_min", str(dec_min)),
                ("dec_max", str(dec_max)),
                ("build_time_iso", datetime.now(UTC).isoformat()),
                ("n_sources", str(len(rows))),
                ("source", "nvss_full.sqlite3"),
            ])
            conn.commit()

        logger.info(f"Created dec strip database: {output_path} ({len(rows)} sources)")
        return output_path

    finally:
        _release_db_lock(lock_fd, lock_path)


def build_first_strip_from_full(
    dec_center: float,
    dec_range: tuple[float, float],
    output_path: Path | None = None,
    min_flux_mjy: float | None = None,
    full_db_path: Path | None = None,
) -> Path:
    """Build FIRST declination strip database from the full FIRST database."""
    dec_min, dec_max = dec_range

    if full_db_path is None:
        full_db_path = get_first_full_db_path()

    if not full_db_path.exists():
        raise FileNotFoundError(f"Full FIRST database not found: {full_db_path}")

    if output_path is None:
        dec_rounded = round(dec_center, 1)
        output_path = _CATALOG_DIR / f"first_dec{dec_rounded:+.1f}.sqlite3"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.info(f"Dec strip database already exists: {output_path}")
        return output_path

    lock_path = output_path.with_suffix(".lock")
    lock_fd = _acquire_db_lock(lock_path)

    if lock_fd is None:
        if output_path.exists():
            return output_path
        raise RuntimeError(f"Could not acquire lock for {output_path}")

    try:
        if output_path.exists():
            return output_path

        logger.info(f"Building FIRST dec strip: {dec_min:.2f}° to {dec_max:.2f}°")

        with sqlite3.connect(str(full_db_path)) as src_conn:
            query = "SELECT ra_deg, dec_deg, flux_mjy, maj_arcsec, min_arcsec FROM sources WHERE dec_deg >= ? AND dec_deg <= ?"
            params = [dec_min, dec_max]
            if min_flux_mjy is not None:
                query += " AND flux_mjy >= ?"
                params.append(min_flux_mjy)
            rows = src_conn.execute(query, params).fetchall()

        logger.info(f"Found {len(rows)} sources in dec range")

        with sqlite3.connect(str(output_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE sources (
                    source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ra_deg REAL NOT NULL,
                    dec_deg REAL NOT NULL,
                    flux_mjy REAL,
                    maj_arcsec REAL,
                    min_arcsec REAL,
                    UNIQUE(ra_deg, dec_deg)
                )
            """)
            conn.execute("CREATE INDEX idx_radec ON sources(ra_deg, dec_deg)")
            conn.execute("CREATE INDEX idx_dec ON sources(dec_deg)")
            conn.execute("CREATE INDEX idx_flux ON sources(flux_mjy)")
            conn.executemany(
                "INSERT OR IGNORE INTO sources(ra_deg, dec_deg, flux_mjy, maj_arcsec, min_arcsec) VALUES(?, ?, ?, ?, ?)",
                rows,
            )
            conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
            conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", [
                ("dec_center", str(dec_center)),
                ("dec_min", str(dec_min)),
                ("dec_max", str(dec_max)),
                ("build_time_iso", datetime.now(UTC).isoformat()),
                ("n_sources", str(len(rows))),
                ("source", "first_full.sqlite3"),
            ])
            conn.commit()

        logger.info(f"Created dec strip database: {output_path} ({len(rows)} sources)")
        return output_path

    finally:
        _release_db_lock(lock_fd, lock_path)


def build_vlass_strip_from_full(
    dec_center: float,
    dec_range: tuple[float, float],
    output_path: Path | None = None,
    min_flux_mjy: float | None = None,
    full_db_path: Path | None = None,
) -> Path:
    """Build VLASS dec strip database from the full VLASS database."""
    dec_min, dec_max = dec_range

    if full_db_path is None:
        full_db_path = get_vlass_full_db_path()

    if not full_db_path.exists():
        raise FileNotFoundError(f"Full VLASS database not found: {full_db_path}")

    if output_path is None:
        dec_rounded = round(dec_center, 1)
        output_path = _CATALOG_DIR / f"vlass_dec{dec_rounded:+.1f}.sqlite3"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.info(f"Dec strip database already exists: {output_path}")
        return output_path

    lock_path = output_path.with_suffix(".lock")
    lock_fd = _acquire_db_lock(lock_path)

    if lock_fd is None:
        if output_path.exists():
            return output_path
        raise RuntimeError(f"Could not acquire lock for {output_path}")

    try:
        if output_path.exists():
            return output_path

        logger.info(f"Building VLASS dec strip: {dec_min:.2f}° to {dec_max:.2f}°")

        with sqlite3.connect(str(full_db_path)) as src_conn:
            query = "SELECT ra_deg, dec_deg, flux_mjy FROM sources WHERE dec_deg >= ? AND dec_deg <= ?"
            params = [dec_min, dec_max]
            if min_flux_mjy is not None:
                query += " AND flux_mjy >= ?"
                params.append(min_flux_mjy)
            rows = src_conn.execute(query, params).fetchall()

        logger.info(f"Found {len(rows)} sources in dec range")

        with sqlite3.connect(str(output_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE sources (
                    source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ra_deg REAL NOT NULL,
                    dec_deg REAL NOT NULL,
                    flux_mjy REAL,
                    UNIQUE(ra_deg, dec_deg)
                )
            """)
            conn.execute("CREATE INDEX idx_radec ON sources(ra_deg, dec_deg)")
            conn.execute("CREATE INDEX idx_dec ON sources(dec_deg)")
            conn.execute("CREATE INDEX idx_flux ON sources(flux_mjy)")
            conn.executemany(
                "INSERT OR IGNORE INTO sources(ra_deg, dec_deg, flux_mjy) VALUES(?, ?, ?)", rows
            )
            conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
            conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", [
                ("dec_center", str(dec_center)),
                ("dec_min", str(dec_min)),
                ("dec_max", str(dec_max)),
                ("build_time_iso", datetime.now(UTC).isoformat()),
                ("n_sources", str(len(rows))),
                ("source", "vlass_full.sqlite3"),
            ])
            conn.commit()

        logger.info(f"Created dec strip database: {output_path} ({len(rows)} sources)")
        return output_path

    finally:
        _release_db_lock(lock_fd, lock_path)


def build_rax_strip_from_full(
    dec_center: float,
    dec_range: tuple[float, float],
    output_path: Path | None = None,
    min_flux_mjy: float | None = None,
    full_db_path: Path | None = None,
) -> Path:
    """Build RAX dec strip database from the full RAX database."""
    dec_min, dec_max = dec_range

    if full_db_path is None:
        full_db_path = get_rax_full_db_path()

    if not full_db_path.exists():
        raise FileNotFoundError(f"Full RAX database not found: {full_db_path}")

    if output_path is None:
        dec_rounded = round(dec_center, 1)
        output_path = _CATALOG_DIR / f"rax_dec{dec_rounded:+.1f}.sqlite3"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.info(f"Dec strip database already exists: {output_path}")
        return output_path

    lock_path = output_path.with_suffix(".lock")
    lock_fd = _acquire_db_lock(lock_path)

    if lock_fd is None:
        if output_path.exists():
            return output_path
        raise RuntimeError(f"Could not acquire lock for {output_path}")

    try:
        if output_path.exists():
            return output_path

        logger.info(f"Building RAX dec strip: {dec_min:.2f}° to {dec_max:.2f}°")

        with sqlite3.connect(str(full_db_path)) as src_conn:
            query = "SELECT ra_deg, dec_deg, flux_mjy FROM sources WHERE dec_deg >= ? AND dec_deg <= ?"
            params = [dec_min, dec_max]
            if min_flux_mjy is not None:
                query += " AND flux_mjy >= ?"
                params.append(min_flux_mjy)
            rows = src_conn.execute(query, params).fetchall()

        logger.info(f"Found {len(rows)} sources in dec range")

        with sqlite3.connect(str(output_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE sources (
                    source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ra_deg REAL NOT NULL,
                    dec_deg REAL NOT NULL,
                    flux_mjy REAL,
                    UNIQUE(ra_deg, dec_deg)
                )
            """)
            conn.execute("CREATE INDEX idx_radec ON sources(ra_deg, dec_deg)")
            conn.execute("CREATE INDEX idx_dec ON sources(dec_deg)")
            conn.execute("CREATE INDEX idx_flux ON sources(flux_mjy)")
            conn.executemany(
                "INSERT OR IGNORE INTO sources(ra_deg, dec_deg, flux_mjy) VALUES(?, ?, ?)", rows
            )
            conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
            conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", [
                ("dec_center", str(dec_center)),
                ("dec_min", str(dec_min)),
                ("dec_max", str(dec_max)),
                ("build_time_iso", datetime.now(UTC).isoformat()),
                ("n_sources", str(len(rows))),
                ("source", "rax_full.sqlite3"),
            ])
            conn.commit()

        logger.info(f"Created dec strip database: {output_path} ({len(rows)} sources)")
        return output_path

    finally:
        _release_db_lock(lock_fd, lock_path)


# --------------------------------------------------------------------------
# Catalog existence checks
# --------------------------------------------------------------------------

def check_catalog_database_exists(
    catalog_type: str,
    dec_deg: float,
    tolerance_deg: float = 1.0,
) -> tuple[bool, Path | None]:
    """Check if a catalog database (strip or full) exists for the given declination."""
    from dsa110_continuum.catalog.query import resolve_catalog_path

    try:
        db_path = resolve_catalog_path(catalog_type, dec_strip=dec_deg)
        if db_path.exists():
            return True, db_path
    except FileNotFoundError:
        pass

    return False, None


def check_missing_catalog_databases(
    dec_deg: float,
    logger_instance: logging.Logger | None = None,
    auto_build: bool = False,
    dec_range_deg: float = 6.0,
) -> dict[str, bool]:
    """Check which catalog databases are available for a given declination.

    Parameters
    ----------
    dec_deg : float
        Declination in degrees
    logger_instance : Optional[logging.Logger]
        Logger to use (defaults to module logger)
    auto_build : bool
        If True, automatically build missing strip databases from full databases
    dec_range_deg : float
        Half-width of declination strip to build if auto_build=True
    """
    if logger_instance is None:
        logger_instance = logger

    results = {}

    for catalog_type, limits in CATALOG_COVERAGE_LIMITS.items():
        dec_min = limits.get("dec_min", -90.0)
        dec_max = limits.get("dec_max", 90.0)

        within_coverage = dec_min <= dec_deg <= dec_max

        if within_coverage:
            exists, db_path = check_catalog_database_exists(catalog_type, dec_deg)
            results[catalog_type] = exists

            if not exists:
                logger_instance.warning(
                    f"{catalog_type.upper()} catalog not found for dec={dec_deg:.2f}°"
                )

                if auto_build:
                    dec_range = (dec_deg - dec_range_deg, dec_deg + dec_range_deg)
                    try:
                        if catalog_type == "nvss":
                            db_path = build_nvss_strip_from_full(dec_deg, dec_range)
                        elif catalog_type == "first":
                            db_path = build_first_strip_from_full(dec_deg, dec_range)
                        elif catalog_type == "rax":
                            db_path = build_rax_strip_from_full(dec_deg, dec_range)
                        elif catalog_type == "vlass":
                            db_path = build_vlass_strip_from_full(dec_deg, dec_range)
                        else:
                            logger_instance.warning(
                                f"Auto-build not supported for {catalog_type}"
                            )
                            continue

                        results[catalog_type] = True
                        logger_instance.info(
                            f"Built {catalog_type.upper()} database: {db_path}"
                        )
                    except Exception as e:
                        logger_instance.error(f"Failed to build {catalog_type.upper()}: {e}")
                        results[catalog_type] = False
        else:
            results[catalog_type] = False

    return results
