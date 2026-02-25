"""
Build per-declination strip SQLite databases from source catalogs.

These databases are optimized for fast spatial queries during long-term
drift scan operations at fixed declinations.
"""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dsa110_contimg.common.utils import get_env_path

logger = logging.getLogger(__name__)


def _hash_file(path: Path) -> str | None:
    try:
        if not path.exists():
            return None
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _write_raw_parquet(df_raw: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_parquet(output_path, index=False)
    return output_path

# Catalog coverage limits (declination ranges)
CATALOG_COVERAGE_LIMITS = {
    "nvss": {"dec_min": -40.0, "dec_max": 90.0},
    "first": {"dec_min": -40.0, "dec_max": 90.0},
    "rax": {"dec_min": -90.0, "dec_max": 49.9},
    "vlass": {"dec_min": -40.0, "dec_max": 90.0},  # VLA Sky Survey
    "atnf": {"dec_min": -90.0, "dec_max": 90.0},  # All-sky pulsar catalog
}

# Default cache directory for catalog files
DEFAULT_CACHE_DIR = str(get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg")) + "/.cache/catalogs"
RAW_ROWS_DIR = (
    get_env_path("CONTIMG_STATE_DIR", default="/data/dsa110-contimg/state")
    / "catalogs/raw_rows"
)


def _acquire_db_lock(
    lock_path: Path, timeout_sec: float = 300.0, max_retries: int = 10
) -> int | None:
    """Acquire an exclusive lock on a database build operation.

    Parameters
    ----------
    lock_path : str
        Path to lock file
    timeout_sec : float, optional
        Maximum time to wait for lock (default is 300.0 seconds)
    max_retries : int, optional
        Maximum number of retry attempts (default is 10)
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    lock_file = open(lock_path, "w")
    start_time = time.time()
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Try to acquire exclusive lock (non-blocking)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Success!
            return lock_file.fileno()
        except BlockingIOError:
            # Lock is held by another process
            elapsed = time.time() - start_time
            if elapsed > timeout_sec:
                logger.warning(
                    f"Timeout waiting for database lock {lock_path} "
                    f"(waited {elapsed:.1f}s, timeout={timeout_sec}s)"
                )
                lock_file.close()
                return None

            # Wait before retrying (exponential backoff)
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
    """Release a database lock.

    Parameters
    ----------
    lock_fd :
        File descriptor from _acquire_db_lock()
    lock_path :
        Path to lock file
    lock_fd: Optional[int] :

    """
    if lock_fd is not None:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except Exception as e:
            logger.warning(f"Error releasing database lock {lock_path}: {e}")

    # Remove lock file if it exists
    try:
        if lock_path.exists():
            lock_path.unlink()
    except Exception as e:
        logger.warning(f"Error removing lock file {lock_path}: {e}")


# --------------------------------------------------------------------------
# Full catalog database builders (one-time operations)
# --------------------------------------------------------------------------

# Default path for full NVSS database
NVSS_FULL_DB_PATH = (
    get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg")
    / "state/catalogs/nvss_full.sqlite3"
)


def get_nvss_full_db_path() -> Path:
    """Get the path to the full NVSS database."""
    return NVSS_FULL_DB_PATH


def nvss_full_db_exists() -> bool:
    """Check if the full NVSS database exists."""
    db_path = get_nvss_full_db_path()
    if not db_path.exists():
        return False

    # Verify it has data
    try:
        with sqlite3.connect(str(db_path)) as conn:
            count = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
            return count > 0
    except Exception:
        return False


def build_nvss_full_db(
    output_path: Path | None = None,
    force_rebuild: bool = False,
) -> Path:
    """Build a full NVSS SQLite database from the raw HEASARC file.

        This creates a comprehensive database with all ~1.77M NVSS sources,
        indexed for fast spatial queries. Dec strip databases can then be
        built efficiently from this database instead of re-parsing the raw file.

    Parameters
    ----------
    output_path : Optional[Path], optional
        Output database path (default is None)
    force_rebuild : bool, optional
        If True, rebuild even if database exists (default is False)
    """
    if output_path is None:
        output_path = get_nvss_full_db_path()

    output_path = Path(output_path)

    # Check if already exists
    if output_path.exists() and not force_rebuild:
        logger.info(f"Full NVSS database already exists: {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load from raw HEASARC file
    from dsa110_contimg.core.calibration.catalogs import read_nvss_catalog

    logger.info("Loading NVSS catalog from raw HEASARC file...")
    df_full = read_nvss_catalog(cache_dir=DEFAULT_CACHE_DIR)
    logger.info(f"Loaded {len(df_full)} NVSS sources")

    # Acquire lock
    lock_path = output_path.with_suffix(".lock")
    lock_fd = _acquire_db_lock(lock_path, timeout_sec=600.0)

    if lock_fd is None:
        if output_path.exists():
            logger.info(f"Database {output_path} was created by another process")
            return output_path
        raise RuntimeError(f"Could not acquire lock for {output_path}")

    try:
        # Double-check after lock
        if output_path.exists() and not force_rebuild:
            logger.info(f"Database {output_path} created while waiting for lock")
            return output_path

        # Remove existing if force rebuild
        if output_path.exists() and force_rebuild:
            output_path.unlink()

        logger.info(f"Creating full NVSS database: {output_path}")

        # Prepare DataFrame for SQL insertion
        # Map HEASARC columns to DB schema columns
        column_map = {
            "ra": "ra_deg", 
            "ra_deg": "ra_deg",
            "dec": "dec_deg", 
            "dec_deg": "dec_deg",
            "flux_20_cm": "flux_mjy",
            "flux_mjy": "flux_mjy",
            "flux_20_cm_error": "flux_err_mjy",
            "major_axis": "major_axis",
            "minor_axis": "minor_axis",
            "position_angle": "position_angle"
        }
        
        # Select and rename columns
        available_cols = [c for c in column_map.keys() if c in df_full.columns]
        # De-duplicate target columns (prioritize earlier matches in column_map keys if needed, 
        # but here we just need to ensure we map correctly)
        
        # Easier way: standardise the dataframe
        df_insert = df_full.copy()
        
        # Standardize column names
        rename_dict = {}
        for src, dst in column_map.items():
            if src in df_insert.columns and dst not in rename_dict.values():
                rename_dict[src] = dst
        
        df_insert = df_insert.rename(columns=rename_dict)
        
        # Ensure all required columns exist
        required_cols = [
            "ra_deg",
            "dec_deg",
            "flux_mjy",
            "flux_err_mjy",
            "major_axis",
            "minor_axis",
            "position_angle",
        ]
        for col in required_cols:
            if col not in df_insert.columns:
                df_insert[col] = None
                
        # Numeric coercion
        for col in required_cols:
            df_insert[col] = pd.to_numeric(df_insert[col], errors="coerce")
            
        # Drop rows with invalid coordinates
        df_insert = df_insert.dropna(subset=["ra_deg", "dec_deg"])

        # Keep only schema columns
        df_insert = df_insert[required_cols]

        # Align raw rows to inserted rows and assign catalog_row_id
        valid_index = df_insert.index
        df_raw = df_full.loc[valid_index].reset_index(drop=True)
        df_insert = df_insert.reset_index(drop=True)
        df_insert["catalog_row_id"] = np.arange(1, len(df_insert) + 1, dtype=int)
        df_raw["catalog_row_id"] = df_insert["catalog_row_id"]

        with sqlite3.connect(str(output_path)) as conn:
            # Enable WAL mode
            conn.execute("PRAGMA journal_mode=WAL")

            # Create table with correct schema (to ensure types and PK)
            # source_id will be auto-generated
            conn.execute("""
                CREATE TABLE sources (
                    source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ra_deg REAL NOT NULL,
                    dec_deg REAL NOT NULL,
                    flux_mjy REAL,
                    flux_err_mjy REAL,
                    major_axis REAL,
                    minor_axis REAL,
                    position_angle REAL,
                    catalog_row_id INTEGER NOT NULL UNIQUE
                )
            """)

            # Bulk insert using pandas
            logger.info("Bulk inserting sources...")
            df_insert.to_sql(
                "sources", 
                conn, 
                if_exists="append", 
                index=False,
                chunksize=10000,
                method=None # default is faster than 'multi' for simple inserts sometimes, let's verify. 
                # actually 'multi' is faster for many rows but sqlite limit is 999 vars. 
                # standard to_sql usually iterates batches. 
                # let's try standard first (None) which uses executemany.
            )
            
            # Create indexes AFTER insert for speed
            logger.info("Creating indexes...")
            conn.execute("CREATE INDEX idx_dec ON sources(dec_deg)")
            conn.execute("CREATE INDEX idx_radec ON sources(ra_deg, dec_deg)")
            conn.execute("CREATE INDEX idx_flux ON sources(flux_mjy)")
            conn.execute("CREATE INDEX idx_catalog_row_id ON sources(catalog_row_id)")

            # Raw row storage (parquet)
            logger.info("Storing raw NVSS rows (parquet)...")
            raw_rows_path = _write_raw_parquet(df_raw, RAW_ROWS_DIR / "nvss_full.parquet")
            raw_rows_hash = _hash_file(raw_rows_path)

            # Create metadata table
            conn.execute("""
                CREATE TABLE meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            build_time = datetime.now(UTC).isoformat()
            source_hash = _hash_file(Path(DEFAULT_CACHE_DIR) / "heasarc_nvss.tdat")
            meta_data = [
                ("build_time_iso", build_time),
                ("source_hash", source_hash or ""),
                ("raw_rows_format", "parquet"),
                ("raw_rows_path", str(raw_rows_path)),
                ("raw_rows_hash", raw_rows_hash or ""),
                ("n_sources", str(len(df_insert))),
                ("source", "HEASARC NVSS catalog")
            ]
            conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta_data)
            
            conn.commit()

        logger.info(f"Created full NVSS database with {len(df_insert)} sources")
        return output_path

    finally:
        _release_db_lock(lock_fd, lock_path)


def build_nvss_strip_from_full(
    dec_center: float,
    dec_range: tuple[float, float],
    output_path: Path | None = None,
    min_flux_mjy: float | None = None,
    full_db_path: Path | None = None,
) -> Path:
    """Build NVSS declination strip database from the full NVSS database.

        This is faster than parsing the raw HEASARC file because it uses
        indexed SQLite queries.

    Parameters
    ----------
    dec_center : float
        Center declination in degrees
    dec_range : tuple of float
        Tuple of (dec_min, dec_max) in degrees
    output_path : Optional[Path], optional
        Output SQLite database path (auto-generated if None)
    min_flux_mjy : Optional[float], optional
        Minimum flux threshold in mJy (None means no threshold)
    full_db_path : Optional[Path], optional
        Path to full NVSS database (default is None)
    """
    dec_min, dec_max = dec_range

    # Resolve full database path
    if full_db_path is None:
        full_db_path = get_nvss_full_db_path()

    if not full_db_path.exists():
        raise FileNotFoundError(
            f"Full NVSS database not found: {full_db_path}. Run build_nvss_full_db() first."
        )

    # Resolve output path
    if output_path is None:
        dec_rounded = round(dec_center, 1)
        db_name = f"nvss_dec{dec_rounded:+.1f}.sqlite3"
        output_path = (
            get_env_path("CONTIMG_STATE_DIR", default="/data/dsa110-contimg/state")
            / "catalogs"
            / db_name
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if already exists
    if output_path.exists():
        logger.info(f"Dec strip database already exists: {output_path}")
        return output_path

    # Acquire lock
    lock_path = output_path.with_suffix(".lock")
    lock_fd = _acquire_db_lock(lock_path, timeout_sec=300.0)

    if lock_fd is None:
        if output_path.exists():
            return output_path
        raise RuntimeError(f"Could not acquire lock for {output_path}")

    try:
        if output_path.exists():
            return output_path

        logger.info(f"Building NVSS dec strip from full database: {dec_min:.2f}° to {dec_max:.2f}°")

        # Query sources from full database
        with sqlite3.connect(str(full_db_path)) as src_conn:
            query = """
                SELECT ra_deg, dec_deg, flux_mjy
                FROM sources
                WHERE dec_deg >= ? AND dec_deg <= ?
            """
            params = [dec_min, dec_max]

            if min_flux_mjy is not None:
                query += " AND flux_mjy >= ?"
                params.append(min_flux_mjy)

            cursor = src_conn.execute(query, params)
            rows = cursor.fetchall()

        logger.info(f"Found {len(rows)} sources in dec range")

        # Create output database
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
                "INSERT OR IGNORE INTO sources(ra_deg, dec_deg, flux_mjy) VALUES(?, ?, ?)",
                rows,
            )

            # Metadata
            conn.execute("""
                CREATE TABLE meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            build_time = datetime.now(UTC).isoformat()
            meta = [
                ("dec_center", str(dec_center)),
                ("dec_min", str(dec_min)),
                ("dec_max", str(dec_max)),
                ("build_time_iso", build_time),
                ("n_sources", str(len(rows))),
                ("source", "nvss_full.sqlite3"),
            ]
            conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta)

            conn.commit()

        logger.info(f"Created dec strip database: {output_path} ({len(rows)} sources)")
        return output_path

    finally:
        _release_db_lock(lock_fd, lock_path)


def regenerate_nvss_strip_db(
    dec_center: float,
    dec_range: tuple[float, float] | None = None,
    catalog_dir: Path | None = None,
    force: bool = True,
) -> Path:
    """Regenerate an NVSS declination strip database.

        This function removes any existing (potentially corrupted) strip database
        and rebuilds it from the full NVSS database.

    Parameters
    ----------
    dec_center : float
        Center declination in degrees (used for filename)
    dec_range : Optional[tuple of float], optional
        Tuple of (dec_min, dec_max) in degrees. If None, uses dec_center ± 6.0 degrees (default is None)
    catalog_dir : Optional[Path], optional
        Directory containing catalog databases. If None, uses standardized catalog path (default is None)
    force : bool, optional
        If True, remove existing database even if it appears valid.
        If False, only regenerate if database is corrupted (default is True)

    Returns
    -------
        Path
        Path to the regenerated declination strip database

    Examples
    --------
        >>> # Regenerate the Dec +54.6 strip database
        >>> regenerate_nvss_strip_db(54.6)
        Path('/data/dsa110-contimg/state/catalogs/nvss_dec+54.6.sqlite3')

        >>> # Regenerate with custom range
        >>> regenerate_nvss_strip_db(55.0, dec_range=(52.0, 58.0))
    """
    # Resolve catalog directory
    if catalog_dir is None:
        catalog_dir = (
            get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg") / "state/catalogs"
        )
        # catalog_dir = Path("/data/dsa110-contimg/state/catalogs")
    catalog_dir = Path(catalog_dir)

    # Resolve dec range
    if dec_range is None:
        dec_range = (dec_center - 6.0, dec_center + 6.0)

    # Build filename
    dec_rounded = round(dec_center, 1)
    db_name = f"nvss_dec{dec_rounded:+.1f}.sqlite3"
    db_path = catalog_dir / db_name

    # Check if database needs regeneration
    needs_regen = force
    if not force and db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            conn.execute("SELECT 1 FROM sources LIMIT 1").fetchone()
            conn.close()
            logger.info(f"Database {db_name} appears valid, skipping regeneration")
            return db_path
        except sqlite3.DatabaseError as e:
            logger.warning(f"Database {db_name} is corrupted: {e}")
            needs_regen = True

    if needs_regen and db_path.exists():
        logger.info(f"Removing corrupted database: {db_path}")
        try:
            db_path.unlink()
        except OSError as e:
            logger.error(f"Failed to remove {db_path}: {e}")
            raise RuntimeError(f"Cannot remove corrupted database: {db_path}") from e

        # Also remove any lock file
        lock_path = db_path.with_suffix(".lock")
        if lock_path.exists():
            try:
                lock_path.unlink()
            except OSError:
                pass

    # Rebuild from full database
    logger.info(f"Regenerating {db_name} from full NVSS database...")
    try:
        new_path = build_nvss_strip_from_full(
            dec_center=dec_center,
            dec_range=dec_range,
            output_path=db_path,
        )
        logger.info(f"Successfully regenerated: {new_path}")
        return new_path
    except Exception as e:
        logger.error(f"Failed to regenerate {db_name}: {e}")
        raise RuntimeError(f"Failed to regenerate {db_name}: {e}") from e


def check_and_regenerate_nvss_strips(
    catalog_dir: Path | None = None,
    dec_centers: list[float] | None = None,
) -> dict[str, str]:
    """Check all NVSS strip databases and regenerate corrupted ones.

    Parameters
    ----------
    catalog_dir : Optional[Path], optional
        Directory containing catalog databases (default is None)
    dec_centers : Optional[list of float], optional
        List of declination centers to check. If None, scans catalog_dir for existing nvss_dec*.sqlite3 files (default is None)

    Returns
    -------
        dict
        Dictionary mapping database filenames to status strings

    Examples
    --------
        >>> # Check and fix all existing strips
        >>> results = check_and_regenerate_nvss_strips()
        >>> print(results)
        {'nvss_dec+54.6.sqlite3': 'regenerated', 'nvss_dec+52.0.sqlite3': 'regenerated'}
    """
    if catalog_dir is None:
        catalog_dir = (
            get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg") / "state/catalogs"
        )
        # catalog_dir = Path("/data/dsa110-contimg/state/catalogs")
    catalog_dir = Path(catalog_dir)

    results = {}

    # Find databases to check
    if dec_centers is None:
        # Scan for existing strip databases
        strip_files = list(catalog_dir.glob("nvss_dec*.sqlite3"))
        strip_files = [f for f in strip_files if f.name != "nvss_full.sqlite3"]
    else:
        strip_files = []
        for dec in dec_centers:
            dec_rounded = round(dec, 1)
            db_name = f"nvss_dec{dec_rounded:+.1f}.sqlite3"
            strip_files.append(catalog_dir / db_name)

    for db_path in strip_files:
        db_name = db_path.name

        # Check if database is valid
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path))
                conn.execute("SELECT COUNT(*) FROM sources").fetchone()
                conn.close()
                results[db_name] = "ok"
                continue
            except sqlite3.DatabaseError:
                logger.warning(f"{db_name} is corrupted, will regenerate")

        # Extract dec_center from filename
        try:
            dec_str = db_name.replace("nvss_dec", "").replace(".sqlite3", "")
            dec_center = float(dec_str)
        except ValueError:
            logger.error(f"Cannot parse declination from {db_name}")
            results[db_name] = "failed"
            continue

        # Regenerate
        try:
            regenerate_nvss_strip_db(dec_center, catalog_dir=catalog_dir, force=True)
            results[db_name] = "regenerated"
        except Exception as e:
            logger.error(f"Failed to regenerate {db_name}: {e}")
            results[db_name] = "failed"

    return results


# --------------------------------------------------------------------------
# FIRST catalog full database builders
# --------------------------------------------------------------------------

# Default path for full FIRST database
FIRST_FULL_DB_PATH = (
    get_env_path("CONTIMG_STATE_DIR", default="/data/dsa110-contimg/state")
    / "catalogs/first_full.sqlite3"
)


def get_first_full_db_path() -> Path:
    """Get the path to the full FIRST database."""
    return FIRST_FULL_DB_PATH


def first_full_db_exists() -> bool:
    """Check if the full FIRST database exists."""
    db_path = get_first_full_db_path()
    if not db_path.exists():
        return False

    try:
        with sqlite3.connect(str(db_path)) as conn:
            count = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
            return count > 0
    except Exception:
        return False


def build_first_full_db(
    output_path: Path | None = None,
    force_rebuild: bool = False,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Path:
    """Build a full FIRST SQLite database from Vizier/cached data.

        Creates a comprehensive database with all FIRST sources,
        indexed for fast spatial queries.

    Parameters
    ----------
    output_path : Optional[Path], optional
        Output database path (default is None)
    force_rebuild : bool, optional
        If True, rebuild even if database exists (default is False)
    cache_dir : str
        Directory for cached catalog files (default is DEFAULT_CACHE_DIR)
    """
    from dsa110_contimg.core.calibration.catalogs import read_first_catalog
    from dsa110_contimg.core.catalog.build_master import _normalize_columns

    if output_path is None:
        output_path = get_first_full_db_path()

    output_path = Path(output_path)

    if output_path.exists() and not force_rebuild:
        logger.info(f"Full FIRST database already exists: {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading FIRST catalog...")
    df_full = read_first_catalog(cache_dir=cache_dir)
    logger.info(f"Loaded {len(df_full)} FIRST sources")

    # Normalize columns
    FIRST_CANDIDATES = {
        "ra": ["ra", "ra_deg", "raj2000"],
        "dec": ["dec", "dec_deg", "dej2000"],
        "flux": [
            "peak_flux",
            "peak_mjy_per_beam",
            "flux_peak",
            "flux",
            "total_flux",
            "fpeak",
            "fint",
            "flux_mjy",
        ],
        "maj": ["deconv_maj", "maj", "fwhm_maj", "deconvolved_major", "maj_deconv"],
        "min": ["deconv_min", "min", "fwhm_min", "deconvolved_minor", "min_deconv"],
    }
    col_map = _normalize_columns(df_full, FIRST_CANDIDATES)
    ra_col = col_map.get("ra", "ra")
    dec_col = col_map.get("dec", "dec")
    flux_col = col_map.get("flux", None)
    maj_col = col_map.get("maj", None)
    min_col = col_map.get("min", None)

    # Normalize RA/Dec columns to numeric degrees when provided as sexagesimal strings
    ra_series = df_full[ra_col]
    dec_series = df_full[dec_col]
    ra_numeric = pd.to_numeric(ra_series, errors="coerce")
    dec_numeric = pd.to_numeric(dec_series, errors="coerce")
    ra_valid = float(ra_numeric.notna().mean()) if len(ra_numeric) else 0.0
    dec_valid = float(dec_numeric.notna().mean()) if len(dec_numeric) else 0.0

    if ra_valid > 0.9 and dec_valid > 0.9:
        df_full["_ra_deg"] = ra_numeric
        df_full["_dec_deg"] = dec_numeric
        ra_col = "_ra_deg"
        dec_col = "_dec_deg"
    else:
        try:
            import astropy.units as u
            from astropy.coordinates import SkyCoord

            coords = SkyCoord(
                ra=ra_series.astype(str).values,
                dec=dec_series.astype(str).values,
                unit=(u.hourangle, u.deg),
                frame="icrs",
            )
            df_full["_ra_deg"] = coords.ra.deg
            df_full["_dec_deg"] = coords.dec.deg
            ra_col = "_ra_deg"
            dec_col = "_dec_deg"
        except Exception as exc:
            logger.warning("Failed to parse FIRST RA/Dec as sexagesimal: %s", exc)

    # Acquire lock
    lock_path = output_path.with_suffix(".lock")
    lock_fd = _acquire_db_lock(lock_path, timeout_sec=600.0)

    if lock_fd is None:
        if output_path.exists():
            return output_path
        raise RuntimeError(f"Could not acquire lock for {output_path}")

    try:
        if output_path.exists() and force_rebuild:
            output_path.unlink()

        logger.info(f"Creating full FIRST database: {output_path}")

        # Prepare DataFrame
        df_insert = df_full.copy()
        
        # Rename columns to schema names
        rename_map = {
            ra_col: "ra_deg",
            dec_col: "dec_deg"
        }
        if flux_col:
            rename_map[flux_col] = "flux_mjy"
        if maj_col:
            rename_map[maj_col] = "maj_arcsec"
        if min_col:
            rename_map[min_col] = "min_arcsec"
            
        df_insert = df_insert.rename(columns=rename_map)
        
        # Ensure schema columns exist
        required_cols = ["ra_deg", "dec_deg", "flux_mjy", "maj_arcsec", "min_arcsec"]
        for col in required_cols:
            if col not in df_insert.columns:
                df_insert[col] = None
                
        # Numeric coercion
        for col in required_cols:
            df_insert[col] = pd.to_numeric(df_insert[col], errors="coerce")
            
        # Drop rows with invalid coordinates
        df_insert = df_insert.dropna(subset=["ra_deg", "dec_deg"])

        # Keep only schema columns
        df_insert = df_insert[required_cols]

        # Align raw rows to inserted rows and assign catalog_row_id
        valid_index = df_insert.index
        df_raw = df_full.loc[valid_index].reset_index(drop=True)
        df_insert = df_insert.reset_index(drop=True)
        df_insert["catalog_row_id"] = np.arange(1, len(df_insert) + 1, dtype=int)
        df_raw["catalog_row_id"] = df_insert["catalog_row_id"]

        source_path: Path | None = None
        cache_path = Path(cache_dir) / "first_catalog"
        for ext in [".csv", ".fits", ".fits.gz", ".csv.gz"]:
            candidate = cache_path.with_suffix(ext)
            if candidate.exists():
                source_path = candidate
                break
        if source_path is None:
            vizier_cache = Path(cache_dir) / "first_catalog_from_vizier.csv"
            if vizier_cache.exists():
                source_path = vizier_cache

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
                    catalog_row_id INTEGER NOT NULL UNIQUE
                )
            """)

            # Bulk insert
            logger.info("Bulk inserting FIRST sources...")
            df_insert.to_sql(
                "sources", 
                conn, 
                if_exists="append", 
                index=False,
                chunksize=10000
            )

            # Indexes
            logger.info("Creating indexes...")
            conn.execute("CREATE INDEX idx_dec ON sources(dec_deg)")
            conn.execute("CREATE INDEX idx_radec ON sources(ra_deg, dec_deg)")
            conn.execute("CREATE INDEX idx_flux ON sources(flux_mjy)")
            conn.execute("CREATE INDEX idx_catalog_row_id ON sources(catalog_row_id)")

            # Raw row storage (parquet)
            logger.info("Storing raw FIRST rows (parquet)...")
            raw_rows_path = _write_raw_parquet(df_raw, RAW_ROWS_DIR / "first_full.parquet")
            raw_rows_hash = _hash_file(raw_rows_path)

            conn.execute("""
                CREATE TABLE meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            
            build_time = datetime.now(UTC).isoformat()
            source_hash = _hash_file(source_path) if source_path is not None else None
            meta_data = [
                ("build_time_iso", build_time),
                ("source_hash", source_hash or ""),
                ("raw_rows_format", "parquet"),
                ("raw_rows_path", str(raw_rows_path)),
                ("raw_rows_hash", raw_rows_hash or ""),
                ("n_sources", str(len(df_insert))),
                ("source", "FIRST catalog")
            ]
            conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta_data)
            
            conn.commit()
            
        logger.info(f"Created full FIRST database with {len(df_insert)} sources")
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
    """Build FIRST declination strip database from the full FIRST database.

    Parameters
    ----------
    dec_center : float
        Center declination in degrees
    dec_range : tuple of float
        Tuple of (dec_min, dec_max) in degrees
    output_path : Optional[Path], optional
        Output SQLite database path (auto-generated if None)
    min_flux_mjy : Optional[float], optional
        Minimum flux threshold in mJy
    full_db_path : Optional[Path], optional
        Path to full FIRST database (default is None)
    """
    dec_min, dec_max = dec_range

    if full_db_path is None:
        full_db_path = get_first_full_db_path()

    if not full_db_path.exists():
        raise FileNotFoundError(
            f"Full FIRST database not found: {full_db_path}. Run build_first_full_db() first."
        )

    if output_path is None:
        dec_rounded = round(dec_center, 1)
        db_name = f"first_dec{dec_rounded:+.1f}.sqlite3"
        output_path = (
            get_env_path("CONTIMG_STATE_DIR", default="/data/dsa110-contimg/state")
            / "catalogs"
            / db_name
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.info(f"Dec strip database already exists: {output_path}")
        return output_path

    lock_path = output_path.with_suffix(".lock")
    lock_fd = _acquire_db_lock(lock_path, timeout_sec=300.0)

    if lock_fd is None:
        if output_path.exists():
            return output_path
        raise RuntimeError(f"Could not acquire lock for {output_path}")

    try:
        if output_path.exists():
            return output_path

        logger.info(
            f"Building FIRST dec strip from full database: {dec_min:.2f}° to {dec_max:.2f}°"
        )

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

            conn.execute("""
                CREATE TABLE meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            build_time = datetime.now(UTC).isoformat()
            meta = [
                ("dec_center", str(dec_center)),
                ("dec_min", str(dec_min)),
                ("dec_max", str(dec_max)),
                ("build_time_iso", build_time),
                ("n_sources", str(len(rows))),
                ("source", "first_full.sqlite3"),
            ]
            conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta)
            conn.commit()

        logger.info(f"Created dec strip database: {output_path} ({len(rows)} sources)")
        return output_path

    finally:
        _release_db_lock(lock_fd, lock_path)


# --------------------------------------------------------------------------
# VLASS catalog full database builders
# --------------------------------------------------------------------------

VLASS_FULL_DB_PATH = (
    get_env_path("CONTIMG_STATE_DIR", default="/data/dsa110-contimg/state")
    / "catalogs/vlass_full.sqlite3"
)


def get_vlass_full_db_path() -> Path:
    """Get the path to the full VLASS database."""
    return VLASS_FULL_DB_PATH


def vlass_full_db_exists() -> bool:
    """Check if the full VLASS database exists."""
    db_path = get_vlass_full_db_path()
    if not db_path.exists():
        return False
    try:
        with sqlite3.connect(str(db_path)) as conn:
            count = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
            return count > 0
    except Exception:
        return False


def build_vlass_full_db(
    output_path: Path | None = None,
    force_rebuild: bool = False,
    cache_dir: str = DEFAULT_CACHE_DIR,
    vlass_catalog_path: str | None = None,
) -> Path:
    """Build a full VLASS SQLite database from Vizier/cached data.

    Parameters
    ----------
    output_path : Optional[Path], optional
        Output database path (default is None)
    force_rebuild : bool, optional
        If True, rebuild even if database exists (default is False)
    cache_dir : str
        Directory for cached catalog files (default is DEFAULT_CACHE_DIR)
    vlass_catalog_path : Optional[str], optional
        Explicit path to VLASS catalog file (default is None)
    """
    from dsa110_contimg.core.calibration.catalogs import read_vlass_catalog
    from dsa110_contimg.core.catalog.build_master import _normalize_columns

    if output_path is None:
        output_path = get_vlass_full_db_path()

    output_path = Path(output_path)

    if output_path.exists() and not force_rebuild:
        logger.info(f"Full VLASS database already exists: {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load VLASS catalog (auto-downloads from Vizier if needed)
    logger.info("Loading VLASS catalog...")
    df_full = read_vlass_catalog(cache_dir=cache_dir, vlass_catalog_path=vlass_catalog_path)

    logger.info(f"Loaded {len(df_full)} VLASS sources")

    VLASS_CANDIDATES = {
        "ra": ["ra", "ra_deg", "raj2000"],
        "dec": ["dec", "dec_deg", "dej2000"],
        "flux": [
            "flux_mjy",
            "fpeak",
            "ftot",
            "peak_flux",
            "peak_mjy_per_beam",
            "flux_peak",
            "flux",
            "total_flux",
        ],
    }
    col_map = _normalize_columns(df_full, VLASS_CANDIDATES)
    ra_col = col_map.get("ra", "ra")
    dec_col = col_map.get("dec", "dec")
    flux_col = col_map.get("flux", None)

    lock_path = output_path.with_suffix(".lock")
    lock_fd = _acquire_db_lock(lock_path, timeout_sec=600.0)

    if lock_fd is None:
        if output_path.exists():
            return output_path
        raise RuntimeError(f"Could not acquire lock for {output_path}")

    try:
        if output_path.exists() and not force_rebuild:
            return output_path

        if output_path.exists() and force_rebuild:
            output_path.unlink()

        logger.info(f"Creating full VLASS database: {output_path}")

        # Prepare DataFrame
        df_insert = df_full.copy()
        
        # Rename columns to schema names
        rename_map = {
            ra_col: "ra_deg",
            dec_col: "dec_deg"
        }
        if flux_col:
            rename_map[flux_col] = "flux_mjy"
            
        df_insert = df_insert.rename(columns=rename_map)
        
        # Ensure schema columns exist
        required_cols = ["ra_deg", "dec_deg", "flux_mjy"]
        for col in required_cols:
            if col not in df_insert.columns:
                df_insert[col] = None
        
        # Numeric coercion
        for col in required_cols:
            df_insert[col] = pd.to_numeric(df_insert[col], errors="coerce")
            
        # Drop rows with invalid coordinates
        df_insert = df_insert.dropna(subset=["ra_deg", "dec_deg"])

        # Keep only schema columns
        df_insert = df_insert[required_cols]

        # Align raw rows to inserted rows and assign catalog_row_id
        valid_index = df_insert.index
        df_raw = df_full.loc[valid_index].reset_index(drop=True)
        df_insert = df_insert.reset_index(drop=True)
        df_insert["catalog_row_id"] = np.arange(1, len(df_insert) + 1, dtype=int)
        df_raw["catalog_row_id"] = df_insert["catalog_row_id"]

        source_path: Path | None = None
        if vlass_catalog_path:
            source_path = Path(vlass_catalog_path)
        else:
            for filename in [
                "vlass_catalog_from_vizier.csv",
                "vlass_catalog.csv",
                "vlass_catalog.fits",
                "vlass_catalog.fits.gz",
            ]:
                candidate = Path(cache_dir) / filename
                if candidate.exists():
                    source_path = candidate
                    break

        with sqlite3.connect(str(output_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")

            conn.execute("""
                CREATE TABLE sources (
                    source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ra_deg REAL NOT NULL,
                    dec_deg REAL NOT NULL,
                    flux_mjy REAL,
                    catalog_row_id INTEGER NOT NULL UNIQUE
                )
            """)

            # Bulk insert
            logger.info("Bulk inserting VLASS sources...")
            df_insert.to_sql(
                "sources", 
                conn, 
                if_exists="append", 
                index=False,
                chunksize=10000
            )

            # Indexes
            logger.info("Creating indexes...")
            conn.execute("CREATE INDEX idx_dec ON sources(dec_deg)")
            conn.execute("CREATE INDEX idx_radec ON sources(ra_deg, dec_deg)")
            conn.execute("CREATE INDEX idx_flux ON sources(flux_mjy)")
            conn.execute("CREATE INDEX idx_catalog_row_id ON sources(catalog_row_id)")

            # Raw row storage (parquet)
            logger.info("Storing raw VLASS rows (parquet)...")
            raw_rows_path = _write_raw_parquet(df_raw, RAW_ROWS_DIR / "vlass_full.parquet")
            raw_rows_hash = _hash_file(raw_rows_path)

            conn.execute("""
                CREATE TABLE meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            build_time = datetime.now(UTC).isoformat()
            source_hash = _hash_file(source_path) if source_path is not None else None
            meta_data = [
                ("build_time_iso", build_time),
                ("source_hash", source_hash or ""),
                ("raw_rows_format", "parquet"),
                ("raw_rows_path", str(raw_rows_path)),
                ("raw_rows_hash", raw_rows_hash or ""),
                ("n_sources", str(len(df_insert))),
                ("source", "VLASS catalog (cached)" if vlass_catalog_path is None else "VLASS catalog")
            ]
            conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta_data)
            
            conn.commit()

        logger.info(f"Created full VLASS database with {len(df_insert)} sources")
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
        raise FileNotFoundError(
            f"Full VLASS database not found: {full_db_path}. Run build_vlass_full_db() first."
        )

    if output_path is None:
        dec_rounded = round(dec_center, 1)
        db_name = f"vlass_dec{dec_rounded:+.1f}.sqlite3"
        output_path = (
            get_env_path("CONTIMG_STATE_DIR", default="/data/dsa110-contimg/state")
            / "catalogs"
            / db_name
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.info(f"Dec strip database already exists: {output_path}")
        return output_path

    lock_path = output_path.with_suffix(".lock")
    lock_fd = _acquire_db_lock(lock_path, timeout_sec=300.0)

    if lock_fd is None:
        if output_path.exists():
            return output_path
        raise RuntimeError(f"Could not acquire lock for {output_path}")

    try:
        if output_path.exists():
            return output_path

        logger.info(
            f"Building VLASS dec strip from full database: {dec_min:.2f}° to {dec_max:.2f}°"
        )

        with sqlite3.connect(str(full_db_path)) as src_conn:
            query = (
                "SELECT ra_deg, dec_deg, flux_mjy FROM sources WHERE dec_deg >= ? AND dec_deg <= ?"
            )
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
                "INSERT OR IGNORE INTO sources(ra_deg, dec_deg, flux_mjy) VALUES(?, ?, ?)",
                rows,
            )

            conn.execute("""
                CREATE TABLE meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            build_time = datetime.now(UTC).isoformat()
            meta = [
                ("dec_center", str(dec_center)),
                ("dec_min", str(dec_min)),
                ("dec_max", str(dec_max)),
                ("build_time_iso", build_time),
                ("n_sources", str(len(rows))),
                ("source", "vlass_full.sqlite3"),
            ]
            conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta)
            conn.commit()

        logger.info(f"Created dec strip database: {output_path} ({len(rows)} sources)")
        return output_path

    finally:
        _release_db_lock(lock_fd, lock_path)


# --------------------------------------------------------------------------
# RAX catalog full database builders
# --------------------------------------------------------------------------

RAX_FULL_DB_PATH = (
    get_env_path("CONTIMG_STATE_DIR", default="/data/dsa110-contimg/state")
    / "catalogs/rax_full.sqlite3"
)


def get_rax_full_db_path() -> Path:
    """Get the path to the full RAX database."""
    return RAX_FULL_DB_PATH


def rax_full_db_exists() -> bool:
    """Check if the full RAX database exists."""
    db_path = get_rax_full_db_path()
    if not db_path.exists():
        return False
    try:
        with sqlite3.connect(str(db_path)) as conn:
            count = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
            return count > 0
    except Exception:
        return False


def build_rax_full_db(
    output_path: Path | None = None,
    force_rebuild: bool = False,
    cache_dir: str = DEFAULT_CACHE_DIR,
    rax_catalog_path: str | None = None,
) -> Path:
    """Build a full RAX SQLite database from cached data.

    Parameters
    ----------
    output_path : Optional[Path], optional
        Output database path (default is None)
    force_rebuild : bool, optional
        If True, rebuild even if database exists (default is False)
    cache_dir : str
        Directory for cached catalog files (default is DEFAULT_CACHE_DIR)
    rax_catalog_path : Optional[str], optional
        Explicit path to RAX catalog file (default is None)
    """
    from dsa110_contimg.core.calibration.catalogs import read_rax_catalog
    from dsa110_contimg.core.catalog.build_master import _normalize_columns

    if output_path is None:
        output_path = get_rax_full_db_path()

    output_path = Path(output_path)

    if output_path.exists() and not force_rebuild:
        logger.info(f"Full RAX database already exists: {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading RAX catalog...")
    df_full = read_rax_catalog(cache_dir=cache_dir, rax_catalog_path=rax_catalog_path)
    logger.info(f"Loaded {len(df_full)} RAX sources")

    RAX_CANDIDATES = {
        "ra": ["ra", "ra_deg", "raj2000", "ra_hms"],
        "dec": ["dec", "dec_deg", "dej2000", "dec_dms"],
        "flux": ["flux", "flux_mjy", "flux_jy", "peak_flux", "fpeak", "s1.4"],
    }
    col_map = _normalize_columns(df_full, RAX_CANDIDATES)
    ra_col = col_map.get("ra", "ra")
    dec_col = col_map.get("dec", "dec")
    flux_col = col_map.get("flux", None)

    lock_path = output_path.with_suffix(".lock")
    lock_fd = _acquire_db_lock(lock_path, timeout_sec=600.0)

    if lock_fd is None:
        if output_path.exists():
            return output_path
        raise RuntimeError(f"Could not acquire lock for {output_path}")

    try:
        if output_path.exists() and not force_rebuild:
            return output_path

        if output_path.exists() and force_rebuild:
            output_path.unlink()

        logger.info(f"Creating full RAX database: {output_path}")

        # Prepare DataFrame
        df_insert = df_full.copy()

        rename_map = {
            ra_col: "ra_deg",
            dec_col: "dec_deg",
        }
        if flux_col:
            rename_map[flux_col] = "flux_mjy"

        df_insert = df_insert.rename(columns=rename_map)

        required_cols = ["ra_deg", "dec_deg", "flux_mjy"]
        for col in required_cols:
            if col not in df_insert.columns:
                df_insert[col] = None

        for col in required_cols:
            df_insert[col] = pd.to_numeric(df_insert[col], errors="coerce")

        df_insert = df_insert.dropna(subset=["ra_deg", "dec_deg"])
        df_insert = df_insert[required_cols]

        valid_index = df_insert.index
        df_raw = df_full.loc[valid_index].reset_index(drop=True)
        df_insert = df_insert.reset_index(drop=True)
        df_insert["catalog_row_id"] = np.arange(1, len(df_insert) + 1, dtype=int)
        df_raw["catalog_row_id"] = df_insert["catalog_row_id"]

        source_path: Path | None = None
        if rax_catalog_path:
            source_path = Path(rax_catalog_path)
        else:
            cache_path = Path(cache_dir) / "rax_catalog"
            for ext in [".fits", ".csv", ".fits.gz", ".csv.gz"]:
                candidate = cache_path.with_suffix(ext)
                if candidate.exists():
                    source_path = candidate
                    break

        with sqlite3.connect(str(output_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")

            conn.execute("""
                CREATE TABLE sources (
                    source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ra_deg REAL NOT NULL,
                    dec_deg REAL NOT NULL,
                    flux_mjy REAL,
                    catalog_row_id INTEGER NOT NULL UNIQUE
                )
            """)

            logger.info("Bulk inserting RAX sources...")
            df_insert.to_sql(
                "sources",
                conn,
                if_exists="append",
                index=False,
                chunksize=10000,
            )

            conn.execute("CREATE INDEX idx_dec ON sources(dec_deg)")
            conn.execute("CREATE INDEX idx_radec ON sources(ra_deg, dec_deg)")
            conn.execute("CREATE INDEX idx_flux ON sources(flux_mjy)")
            conn.execute("CREATE INDEX idx_catalog_row_id ON sources(catalog_row_id)")

            logger.info("Storing raw RAX rows (parquet)...")
            raw_rows_path = _write_raw_parquet(df_raw, RAW_ROWS_DIR / "rax_full.parquet")
            raw_rows_hash = _hash_file(raw_rows_path)

            conn.execute("""
                CREATE TABLE meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            build_time = datetime.now(UTC).isoformat()
            source_hash = _hash_file(source_path) if source_path is not None else None
            meta_data = [
                ("build_time_iso", build_time),
                ("source_hash", source_hash or ""),
                ("raw_rows_format", "parquet"),
                ("raw_rows_path", str(raw_rows_path)),
                ("raw_rows_hash", raw_rows_hash or ""),
                ("n_sources", str(len(df_insert))),
                ("source", "RAX catalog (cached)" if rax_catalog_path is None else "RAX catalog"),
            ]
            conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta_data)
            conn.commit()

        logger.info(f"Created full RAX database with {len(df_insert)} sources")
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
        raise FileNotFoundError(
            f"Full RAX database not found: {full_db_path}. Run build_rax_full_db() first."
        )

    if output_path is None:
        dec_rounded = round(dec_center, 1)
        db_name = f"rax_dec{dec_rounded:+.1f}.sqlite3"
        output_path = (
            get_env_path("CONTIMG_STATE_DIR", default="/data/dsa110-contimg/state")
            / "catalogs"
            / db_name
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.info(f"Dec strip database already exists: {output_path}")
        return output_path

    lock_path = output_path.with_suffix(".lock")
    lock_fd = _acquire_db_lock(lock_path, timeout_sec=300.0)

    if lock_fd is None:
        if output_path.exists():
            return output_path
        raise RuntimeError(f"Could not acquire lock for {output_path}")

    try:
        if output_path.exists():
            return output_path

        logger.info(f"Building RAX dec strip from full database: {dec_min:.2f}° to {dec_max:.2f}°")

        with sqlite3.connect(str(full_db_path)) as src_conn:
            query = (
                "SELECT ra_deg, dec_deg, flux_mjy FROM sources WHERE dec_deg >= ? AND dec_deg <= ?"
            )
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
                "INSERT OR IGNORE INTO sources(ra_deg, dec_deg, flux_mjy) VALUES(?, ?, ?)",
                rows,
            )

            conn.execute("""
                CREATE TABLE meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            build_time = datetime.now(UTC).isoformat()
            meta = [
                ("dec_center", str(dec_center)),
                ("dec_min", str(dec_min)),
                ("dec_max", str(dec_max)),
                ("build_time_iso", build_time),
                ("n_sources", str(len(rows))),
                ("source", "rax_full.sqlite3"),
            ]
            conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta)
            conn.commit()

        logger.info(f"Created dec strip database: {output_path} ({len(rows)} sources)")
        return output_path

    finally:
        _release_db_lock(lock_fd, lock_path)


# --------------------------------------------------------------------------
# ATNF pulsar catalog full database builders
# --------------------------------------------------------------------------

ATNF_FULL_DB_PATH = (
    get_env_path("CONTIMG_STATE_DIR", default="/data/dsa110-contimg/state")
    / "catalogs/atnf_full.sqlite3"
)


def get_atnf_full_db_path() -> Path:
    """Get the path to the full ATNF database."""
    return ATNF_FULL_DB_PATH


def atnf_full_db_exists() -> bool:
    """Check if the full ATNF database exists."""
    db_path = get_atnf_full_db_path()
    if not db_path.exists():
        return False
    try:
        with sqlite3.connect(str(db_path)) as conn:
            count = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
            return count > 0
    except Exception:
        return False


def build_atnf_full_db(
    output_path: Path | None = None,
    force_rebuild: bool = False,
) -> Path:
    """Build a full ATNF pulsar SQLite database from psrqpy.

    Parameters
    ----------
    output_path : Optional[Path], optional
        Output database path (default is None)
    force_rebuild : bool, optional
        If True, rebuild even if database exists (default is False)
    """
    from dsa110_contimg.core.catalog.build_atnf_pulsars import (
        _download_atnf_catalog,
        _process_atnf_data,
    )

    if output_path is None:
        output_path = get_atnf_full_db_path()

    output_path = Path(output_path)

    if output_path.exists() and not force_rebuild:
        logger.info(f"Full ATNF database already exists: {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading ATNF pulsar catalog...")
    df_raw = _download_atnf_catalog()
    df_processed = _process_atnf_data(df_raw, min_flux_mjy=None)
    logger.info(f"Loaded {len(df_processed)} ATNF pulsars")

    lock_path = output_path.with_suffix(".lock")
    lock_fd = _acquire_db_lock(lock_path, timeout_sec=600.0)

    if lock_fd is None:
        if output_path.exists():
            return output_path
        raise RuntimeError(f"Could not acquire lock for {output_path}")

    try:
        if output_path.exists() and not force_rebuild:
            return output_path

        if output_path.exists() and force_rebuild:
            output_path.unlink()

        logger.info(f"Creating full ATNF database: {output_path}")

        with sqlite3.connect(str(output_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")

            conn.execute("""
                CREATE TABLE sources (
                    source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ra_deg REAL NOT NULL,
                    dec_deg REAL NOT NULL,
                    flux_mjy REAL,
                    name TEXT,
                    period_s REAL,
                    dm REAL
                )
            """)

            conn.execute("CREATE INDEX idx_dec ON sources(dec_deg)")
            conn.execute("CREATE INDEX idx_radec ON sources(ra_deg, dec_deg)")
            conn.execute("CREATE INDEX idx_flux ON sources(flux_mjy)")

            insert_data = []
            for _, row in df_processed.iterrows():
                ra = float(row["ra_deg"])
                dec = float(row["dec_deg"])

                if not (np.isfinite(ra) and np.isfinite(dec)):
                    continue

                flux = (
                    float(row["flux_1400mhz_mjy"])
                    if pd.notna(row.get("flux_1400mhz_mjy"))
                    else None
                )
                name = str(row.get("pulsar_name", "")) if pd.notna(row.get("pulsar_name")) else None
                period = float(row.get("period_s")) if pd.notna(row.get("period_s")) else None
                dm = float(row.get("dm_pc_cm3")) if pd.notna(row.get("dm_pc_cm3")) else None

                insert_data.append((ra, dec, flux, name, period, dm))

            conn.executemany(
                "INSERT INTO sources (ra_deg, dec_deg, flux_mjy, name, period_s, dm) VALUES (?, ?, ?, ?, ?, ?)",
                insert_data,
            )

            conn.execute("""
                CREATE TABLE meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            build_time = datetime.now(UTC).isoformat()
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?)", ("build_time_iso", build_time)
            )
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?)", ("n_sources", str(len(insert_data)))
            )
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?)",
                ("source", "ATNF Pulsar Catalogue (psrqpy)"),
            )
            conn.commit()

        logger.info(f"Created full ATNF database with {len(insert_data)} sources")
        return output_path

    finally:
        _release_db_lock(lock_fd, lock_path)


def build_atnf_strip_from_full(
    dec_center: float,
    dec_range: tuple[float, float],
    output_path: Path | None = None,
    min_flux_mjy: float | None = None,
    full_db_path: Path | None = None,
) -> Path:
    """Build ATNF dec strip database from the full ATNF database."""
    dec_min, dec_max = dec_range

    if full_db_path is None:
        full_db_path = get_atnf_full_db_path()

    if not full_db_path.exists():
        raise FileNotFoundError(
            f"Full ATNF database not found: {full_db_path}. Run build_atnf_full_db() first."
        )

    if output_path is None:
        dec_rounded = round(dec_center, 1)
        db_name = f"atnf_dec{dec_rounded:+.1f}.sqlite3"
        output_path = (
            get_env_path("CONTIMG_STATE_DIR", default="/data/dsa110-contimg/state")
            / "catalogs"
            / db_name
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.info(f"Dec strip database already exists: {output_path}")
        return output_path

    lock_path = output_path.with_suffix(".lock")
    lock_fd = _acquire_db_lock(lock_path, timeout_sec=300.0)

    if lock_fd is None:
        if output_path.exists():
            return output_path
        raise RuntimeError(f"Could not acquire lock for {output_path}")

    try:
        if output_path.exists():
            return output_path

        logger.info(f"Building ATNF dec strip from full database: {dec_min:.2f}° to {dec_max:.2f}°")

        with sqlite3.connect(str(full_db_path)) as src_conn:
            query = (
                "SELECT ra_deg, dec_deg, flux_mjy FROM sources WHERE dec_deg >= ? AND dec_deg <= ?"
            )
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
                "INSERT OR IGNORE INTO sources(ra_deg, dec_deg, flux_mjy) VALUES(?, ?, ?)",
                rows,
            )

            conn.execute("""
                CREATE TABLE meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            build_time = datetime.now(UTC).isoformat()
            meta = [
                ("dec_center", str(dec_center)),
                ("dec_min", str(dec_min)),
                ("dec_max", str(dec_max)),
                ("build_time_iso", build_time),
                ("n_sources", str(len(rows))),
                ("source", "atnf_full.sqlite3"),
            ]
            conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta)
            conn.commit()

        logger.info(f"Created dec strip database: {output_path} ({len(rows)} sources)")
        return output_path

    finally:
        _release_db_lock(lock_fd, lock_path)


def check_catalog_database_exists(
    catalog_type: str,
    dec_deg: float,
    tolerance_deg: float = 1.0,
) -> tuple[bool, Path | None]:
    """Check if a catalog database exists for the given declination.

    Parameters
    ----------
    catalog_type : str
        One of "nvss", "first", "rax"
    dec_deg : float
        Declination in degrees
    tolerance_deg : float, optional
        Tolerance for matching declination (default is 1.0)
    """
    from dsa110_contimg.core.catalog.query import resolve_catalog_path

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
    """Check which catalog databases are missing when they should exist.

    Parameters
    ----------
    dec_deg : float
        Declination in degrees
    logger_instance : Optional[logging.Logger], optional
        Optional logger instance (uses module logger if None) (default is None)
    auto_build : bool, optional
        If True, automatically build missing databases (default is False)
    dec_range_deg : float, optional
        Declination range (±degrees) for building databases (default is 6.0)
    """
    if logger_instance is None:
        logger_instance = logger

    results = {}
    built_databases = []

    for catalog_type, limits in CATALOG_COVERAGE_LIMITS.items():
        dec_min = limits.get("dec_min", -90.0)
        dec_max = limits.get("dec_max", 90.0)

        # Check if declination is within coverage
        within_coverage = dec_deg >= dec_min and dec_deg <= dec_max

        if within_coverage:
            exists, db_path = check_catalog_database_exists(catalog_type, dec_deg)
            results[catalog_type] = exists

            if not exists:
                logger_instance.warning(
                    f":warning:  {catalog_type.upper()} catalog database is missing for declination {dec_deg:.2f}°, "
                    f"but should exist (within coverage limits: {dec_min:.1f}° to {dec_max:.1f}°)."
                )

                if auto_build:
                    try:
                        logger_instance.info(
                            f":hammer: Auto-building {catalog_type.upper()} catalog database for declination {dec_deg:.2f}°..."
                        )
                        dec_range = (dec_deg - dec_range_deg, dec_deg + dec_range_deg)

                        if catalog_type == "nvss":
                            db_path = build_nvss_strip_db(
                                dec_center=dec_deg,
                                dec_range=dec_range,
                            )
                        elif catalog_type == "first":
                            db_path = build_first_strip_db(
                                dec_center=dec_deg,
                                dec_range=dec_range,
                            )
                        elif catalog_type == "rax":
                            db_path = build_rax_strip_db(
                                dec_center=dec_deg,
                                dec_range=dec_range,
                            )
                        elif catalog_type == "vlass":
                            db_path = build_vlass_strip_db(
                                dec_center=dec_deg,
                                dec_range=dec_range,
                            )
                        elif catalog_type == "atnf":
                            # ATNF is all-sky, but we build per-declination
                            # strip databases for efficiency
                            db_path = build_atnf_strip_db(
                                dec_center=dec_deg,
                                dec_range=dec_range,
                            )
                        else:
                            logger_instance.warning(
                                f"Unknown catalog type for auto-build: {catalog_type}"
                            )
                            continue

                        built_databases.append((catalog_type, db_path))
                        results[catalog_type] = True
                        logger_instance.info(
                            f":check: Successfully built {catalog_type.upper()} database: {db_path}"
                        )
                    except Exception as e:
                        logger_instance.error(
                            f":cross: Failed to auto-build {catalog_type.upper()} database: {e}",
                            exc_info=True,
                        )
                        results[catalog_type] = False
                else:
                    logger_instance.warning(
                        "   Database should be built by CatalogSetupStage or use auto_build=True."
                    )
        else:
            # Outside coverage, so database is not expected
            results[catalog_type] = False

    if auto_build and built_databases:
        logger_instance.info(
            f":check: Auto-built {len(built_databases)} catalog database(s): "
            f"{', '.join([f'{cat.upper()}' for cat, _ in built_databases])}"
        )

    return results


def auto_build_missing_catalog_databases(
    dec_deg: float,
    dec_range_deg: float = 6.0,
    logger_instance: logging.Logger | None = None,
) -> dict[str, Path]:
    """Automatically build missing catalog databases for a given declination.

    Parameters
    ----------
    dec_deg : float
        Declination in degrees
    dec_range_deg : float, optional
        Declination range (±degrees) for building databases (default is 6.0)
    logger_instance : Optional[logging.Logger], optional
        Optional logger instance (uses module logger if None) (default is None)
    """
    if logger_instance is None:
        logger_instance = logger

    # Use check_missing_catalog_databases with auto_build=True
    check_missing_catalog_databases(
        dec_deg=dec_deg,
        logger_instance=logger_instance,
        auto_build=True,
        dec_range_deg=dec_range_deg,
    )

    # Return paths of databases that now exist
    built_paths = {}
    for catalog_type in CATALOG_COVERAGE_LIMITS.keys():
        exists, db_path = check_catalog_database_exists(catalog_type, dec_deg)
        if exists and db_path:
            built_paths[catalog_type] = db_path

    return built_paths


def build_nvss_strip_db(
    dec_center: float,
    dec_range: tuple[float, float],
    output_path: str | os.PathLike[str] | None = None,
    nvss_csv_path: str | None = None,
    min_flux_mjy: float | None = None,
    prefer_full_db: bool = True,
) -> Path:
    """Build SQLite database for NVSS sources in a declination strip.

        If a full NVSS database (nvss_full.sqlite3) exists and prefer_full_db=True,
        the strip will be built from that database (faster). Otherwise, falls back
        to parsing the raw HEASARC text file.

    Parameters
    ----------
    dec_center : float
        Center declination in degrees
    dec_range : Tuple[float, float]
        Tuple of (dec_min, dec_max) in degrees
    output_path : Optional[str or os.PathLike[str]], optional
        Output SQLite database path (auto-generated if None)
    nvss_csv_path : Optional[str], optional
        Path to full NVSS CSV catalog (downloaded if None)
    min_flux_mjy : Optional[float], optional
        Minimum flux threshold in mJy (None means no threshold)
    prefer_full_db : bool, optional
        If True, use nvss_full.sqlite3 if available (default is True)
    """
    dec_min, dec_max = dec_range

    # Resolve output path - use absolute path to state/catalogs
    if output_path is None:
        dec_rounded = round(dec_center, 1)
        db_name = f"nvss_dec{dec_rounded:+.1f}.sqlite3"
        output_path = (
            get_env_path("CONTIMG_STATE_DIR", default="/data/dsa110-contimg/state")
            / "catalogs"
            / db_name
        )

    output_path = Path(output_path)

    # Check if already exists
    if output_path.exists():
        logger.info(f"NVSS dec strip database already exists: {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try to use full database if available and preferred
    if prefer_full_db and nvss_full_db_exists():
        logger.info("Using full NVSS database for faster strip extraction")
        return build_nvss_strip_from_full(
            dec_center=dec_center,
            dec_range=dec_range,
            output_path=output_path,
            min_flux_mjy=min_flux_mjy,
        )

    # Fall back to raw HEASARC file
    logger.info("Building from raw HEASARC file (full DB not available)")

    # Load NVSS catalog
    if nvss_csv_path is None:
        from dsa110_contimg.core.calibration.catalogs import read_nvss_catalog

        df_full = read_nvss_catalog()
    else:
        from dsa110_contimg.core.calibration.catalogs import read_nvss_catalog

        # If CSV path provided, we'd need to read it differently
        # For now, use the cached read function
        df_full = read_nvss_catalog()

    # Check coverage limits
    coverage_limits = CATALOG_COVERAGE_LIMITS.get("nvss", {})
    if dec_center < coverage_limits.get("dec_min", -90.0):
        logger.warning(
            f":warning:  Declination {dec_center:.2f}° is outside NVSS coverage "
            f"(southern limit: {coverage_limits.get('dec_min', -40.0)}°). "
            f"Database may be empty or have very few sources."
        )
    if dec_center > coverage_limits.get("dec_max", 90.0):
        logger.warning(
            f":warning:  Declination {dec_center:.2f}° is outside NVSS coverage "
            f"(northern limit: {coverage_limits.get('dec_max', 90.0)}°). "
            f"Database may be empty or have very few sources."
        )

    # Filter to declination strip
    dec_col = "dec" if "dec" in df_full.columns else "dec_deg"
    df_strip = df_full[(df_full[dec_col] >= dec_min) & (df_full[dec_col] <= dec_max)].copy()

    print(f"Filtered NVSS catalog: {len(df_full)} :arrow_right: {len(df_strip)} sources")
    print(f"Declination range: {dec_min:.6f} to {dec_max:.6f} degrees")

    # Warn if result is empty
    if len(df_strip) == 0:
        logger.warning(
            f":warning:  No NVSS sources found in declination range [{dec_min:.2f}°, {dec_max:.2f}°]. "
            f"This may indicate declination {dec_center:.2f}° is outside NVSS coverage limits "
            f"(southern limit: {coverage_limits.get('dec_min', -40.0)}°)."
        )

    # Apply flux threshold if specified
    if min_flux_mjy is not None:
        flux_col = "flux_20_cm" if "flux_20_cm" in df_strip.columns else "flux_mjy"
        if flux_col in df_strip.columns:
            flux_val = pd.to_numeric(df_strip[flux_col], errors="coerce")
            df_strip = df_strip[flux_val >= min_flux_mjy].copy()
            print(f"After flux cut ({min_flux_mjy} mJy): {len(df_strip)} sources")

    # Standardize column names
    ra_col = "ra" if "ra" in df_strip.columns else "ra_deg"
    dec_col = "dec" if "dec" in df_strip.columns else "dec_deg"
    flux_col = "flux_20_cm" if "flux_20_cm" in df_strip.columns else "flux_mjy"

    # Ensure flux is in mJy
    df_strip["ra_deg"] = pd.to_numeric(df_strip[ra_col], errors="coerce")
    df_strip["dec_deg"] = pd.to_numeric(df_strip[dec_col], errors="coerce")

    if flux_col in df_strip.columns:
        df_strip["flux_mjy"] = pd.to_numeric(df_strip[flux_col], errors="coerce")
    else:
        df_strip["flux_mjy"] = None

    # Check if database already exists (another process may have created it)
    if output_path.exists():
        logger.info(f"Database {output_path} already exists, skipping build")
        return output_path

    # Acquire lock for database creation
    lock_path = output_path.with_suffix(".lock")
    lock_fd = _acquire_db_lock(lock_path, timeout_sec=300.0)

    if lock_fd is None:
        # Could not acquire lock - check if database was created by another process
        if output_path.exists():
            logger.info(f"Database {output_path} was created by another process")
            return output_path
        else:
            raise RuntimeError(
                f"Could not acquire lock for {output_path} and database does not exist"
            )

    try:
        # Double-check database doesn't exist (another process may have created it while we waited)
        if output_path.exists():
            logger.info(
                f"Database {output_path} was created by another process while waiting for lock"
            )
            return output_path

        # Create SQLite database
        print(f"Creating SQLite database: {output_path}")

        # Enable WAL mode for concurrent reads
        with sqlite3.connect(str(output_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")

        with sqlite3.connect(str(output_path)) as conn:
            # Create sources table
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS sources (
                source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ra_deg REAL NOT NULL,
                dec_deg REAL NOT NULL,
                flux_mjy REAL,
                UNIQUE(ra_deg, dec_deg)
            )
        """
            )

        # Create spatial index
        conn.execute("CREATE INDEX IF NOT EXISTS idx_radec ON sources(ra_deg, dec_deg)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dec ON sources(dec_deg)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_flux ON sources(flux_mjy)")

        # Clear existing data
        conn.execute("DELETE FROM sources")

        # Insert sources
        insert_data = []
        for _, row in df_strip.iterrows():
            ra = float(row["ra_deg"])
            dec = float(row["dec_deg"])
            flux = float(row["flux_mjy"]) if pd.notna(row.get("flux_mjy")) else None

            if np.isfinite(ra) and np.isfinite(dec):
                insert_data.append((ra, dec, flux))

        conn.executemany(
            "INSERT OR IGNORE INTO sources(ra_deg, dec_deg, flux_mjy) VALUES(?, ?, ?)",
            insert_data,
        )

        # Create metadata table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """
        )

        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('dec_center', ?)",
            (str(dec_center),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('dec_min', ?)",
            (str(dec_min),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('dec_max', ?)",
            (str(dec_max),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('build_time_iso', ?)",
            (datetime.now(UTC).isoformat(),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('n_sources', ?)",
            (str(len(insert_data)),),
        )
        if min_flux_mjy is not None:
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('min_flux_mjy', ?)",
                (str(min_flux_mjy),),
            )

        # Store coverage status
        coverage_limits = CATALOG_COVERAGE_LIMITS.get("nvss", {})
        within_coverage = dec_center >= coverage_limits.get(
            "dec_min", -90.0
        ) and dec_center <= coverage_limits.get("dec_max", 90.0)
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('within_coverage', ?)",
            ("true" if within_coverage else "false",),
        )

        conn.commit()

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f":check: Database created: {output_path}")
        print(f"  Size: {file_size_mb:.2f} MB")
        print(f"  Sources: {len(insert_data)}")

        return output_path
    finally:
        # Always release the lock
        _release_db_lock(lock_fd, lock_path)


def build_first_strip_db(
    dec_center: float,
    dec_range: tuple[float, float],
    first_catalog_path: str | None = None,
    output_path: str | os.PathLike[str] | None = None,
    min_flux_mjy: float | None = None,
    cache_dir: str = ".cache/catalogs",
    prefer_full_db: bool = True,
) -> Path:
    """Build SQLite database for FIRST sources in a declination strip.

        If a full FIRST database (first_full.sqlite3) exists and prefer_full_db=True,
        the strip will be built from that database (faster). Otherwise, falls back
        to downloading/parsing the raw catalog.

    Parameters
    ----------
    dec_center : float
        Center declination in degrees
    dec_range : Tuple[float, float]
        Tuple of (dec_min, dec_max) in degrees
    first_catalog_path : Optional[str], optional
        Optional path to FIRST catalog (CSV/FITS). If None, attempts to auto-download/cache like NVSS (default is None)
    output_path : Optional[str or os.PathLike[str]], optional
        Output SQLite database path (auto-generated if None)
    min_flux_mjy : Optional[float], optional
        Minimum flux threshold in mJy (None means no threshold)
    cache_dir : str, optional
        Directory for caching catalog files (if auto-downloading) (default is ".cache/catalogs")
    prefer_full_db : bool, optional
        If True, use first_full.sqlite3 if available (default is True)
    """
    from dsa110_contimg.core.calibration.catalogs import read_first_catalog
    from dsa110_contimg.core.catalog.build_master import _normalize_columns

    dec_min, dec_max = dec_range

    # Resolve output path
    if output_path is None:
        dec_rounded = round(dec_center, 1)
        db_name = f"first_dec{dec_rounded:+.1f}.sqlite3"
        output_path = (
            get_env_path("CONTIMG_STATE_DIR", default="/data/dsa110-contimg/state")
            / "catalogs"
            / db_name
        )

    output_path = Path(output_path)

    # Check if already exists
    if output_path.exists():
        logger.info(f"FIRST dec strip database already exists: {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try to use full database if available and preferred
    if prefer_full_db and first_full_db_exists():
        logger.info("Using full FIRST database for faster strip extraction")
        return build_first_strip_from_full(
            dec_center=dec_center,
            dec_range=dec_range,
            output_path=output_path,
            min_flux_mjy=min_flux_mjy,
        )

    # Fall back to raw catalog
    logger.info("Building from raw FIRST catalog (full DB not available)")

    # Check coverage limits
    coverage_limits = CATALOG_COVERAGE_LIMITS.get("first", {})
    if dec_center < coverage_limits.get("dec_min", -90.0):
        logger.warning(
            f":warning:  Declination {dec_center:.2f}° is outside FIRST coverage "
            f"(southern limit: {coverage_limits.get('dec_min', -40.0)}°). "
            f"Database may be empty or have very few sources."
        )
    if dec_center > coverage_limits.get("dec_max", 90.0):
        logger.warning(
            f":warning:  Declination {dec_center:.2f}° is outside FIRST coverage "
            f"(northern limit: {coverage_limits.get('dec_max', 90.0)}°). "
            f"Database may be empty or have very few sources."
        )

    # Load FIRST catalog (auto-downloads if needed, similar to NVSS)
    df_full = read_first_catalog(cache_dir=cache_dir, first_catalog_path=first_catalog_path)

    # Normalize column names (similar to build_master.py)
    FIRST_CANDIDATES = {
        "ra": ["ra", "ra_deg", "raj2000"],
        "dec": ["dec", "dec_deg", "dej2000"],
        "flux": [
            "peak_flux",
            "peak_mjy_per_beam",
            "flux_peak",
            "flux",
            "total_flux",
            "fpeak",
        ],
        "maj": ["deconv_maj", "maj", "fwhm_maj", "deconvolved_major", "maj_deconv"],
        "min": ["deconv_min", "min", "fwhm_min", "deconvolved_minor", "min_deconv"],
    }

    col_map = _normalize_columns(df_full, FIRST_CANDIDATES)

    # Standardize column names
    ra_col = col_map.get("ra", "ra")
    dec_col = col_map.get("dec", "dec")
    flux_col = col_map.get("flux", None)
    maj_col = col_map.get("maj", None)
    min_col = col_map.get("min", None)

    # Filter to declination strip
    df_strip = df_full[
        (pd.to_numeric(df_full[dec_col], errors="coerce") >= dec_min)
        & (pd.to_numeric(df_full[dec_col], errors="coerce") <= dec_max)
    ].copy()

    print(f"Filtered FIRST catalog: {len(df_full)} :arrow_right: {len(df_strip)} sources")
    print(f"Declination range: {dec_min:.6f} to {dec_max:.6f} degrees")

    # Warn if result is empty
    if len(df_strip) == 0:
        coverage_limits = CATALOG_COVERAGE_LIMITS.get("first", {})
        logger.warning(
            f":warning:  No FIRST sources found in declination range [{dec_min:.2f}°, {dec_max:.2f}°]. "
            f"This may indicate declination {dec_center:.2f}° is outside FIRST coverage limits "
            f"(southern limit: {coverage_limits.get('dec_min', -40.0)}°)."
        )

    # Apply flux threshold if specified
    if min_flux_mjy is not None and flux_col:
        flux_val = pd.to_numeric(df_strip[flux_col], errors="coerce")
        # Convert to mJy if needed (assume > 1000 means Jy, otherwise mJy)
        if len(flux_val) > 0 and flux_val.max() > 1000:
            flux_val = flux_val * 1000.0  # Convert Jy to mJy
        df_strip = df_strip[flux_val >= min_flux_mjy].copy()
        print(f"After flux cut ({min_flux_mjy} mJy): {len(df_strip)} sources")

    # Check if database already exists (another process may have created it)
    if output_path.exists():
        logger.info(f"Database {output_path} already exists, skipping build")
        return output_path

    # Acquire lock for database creation
    lock_path = output_path.with_suffix(".lock")
    lock_fd = _acquire_db_lock(lock_path, timeout_sec=300.0)

    if lock_fd is None:
        # Could not acquire lock - check if database was created by another process
        if output_path.exists():
            logger.info(f"Database {output_path} was created by another process")
            return output_path
        else:
            raise RuntimeError(
                f"Could not acquire lock for {output_path} and database does not exist"
            )

    try:
        # Double-check database doesn't exist (another process may have created it while we waited)
        if output_path.exists():
            logger.info(
                f"Database {output_path} was created by another process while waiting for lock"
            )
            return output_path

        # Create SQLite database
        print(f"Creating SQLite database: {output_path}")

        # Enable WAL mode for concurrent reads
        with sqlite3.connect(str(output_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")

        with sqlite3.connect(str(output_path)) as conn:
            # Create sources table with FIRST-specific columns
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS sources (
                source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ra_deg REAL NOT NULL,
                dec_deg REAL NOT NULL,
                flux_mjy REAL,
                maj_arcsec REAL,
                min_arcsec REAL,
                UNIQUE(ra_deg, dec_deg)
            )
        """
            )

        # Create spatial index
        conn.execute("CREATE INDEX IF NOT EXISTS idx_radec ON sources(ra_deg, dec_deg)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dec ON sources(dec_deg)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_flux ON sources(flux_mjy)")

        # Clear existing data
        conn.execute("DELETE FROM sources")

        # Insert sources
        insert_data = []
        for _, row in df_strip.iterrows():
            ra = pd.to_numeric(row[ra_col], errors="coerce")
            dec = pd.to_numeric(row[dec_col], errors="coerce")

            if not (np.isfinite(ra) and np.isfinite(dec)):
                continue

            # Handle flux
            flux = None
            if flux_col and flux_col in row.index:
                flux_val = pd.to_numeric(row[flux_col], errors="coerce")
                if np.isfinite(flux_val):
                    # Convert to mJy if needed (assume > 1000 means Jy, otherwise mJy)
                    flux_val_float = float(flux_val)
                    if flux_val_float > 1000:
                        flux = flux_val_float * 1000.0  # Convert Jy to mJy
                    else:
                        flux = flux_val_float

            # Handle size
            maj = None
            if maj_col and maj_col in row.index:
                maj_val = pd.to_numeric(row[maj_col], errors="coerce")
                if np.isfinite(maj_val):
                    maj = float(maj_val)

            min_val = None
            if min_col and min_col in row.index:
                min_val_num = pd.to_numeric(row[min_col], errors="coerce")
                if np.isfinite(min_val_num):
                    min_val = float(min_val_num)

            insert_data.append((ra, dec, flux, maj, min_val))

        conn.executemany(
            "INSERT OR IGNORE INTO sources(ra_deg, dec_deg, flux_mjy, maj_arcsec, min_arcsec) VALUES(?, ?, ?, ?, ?)",
            insert_data,
        )

        # Create metadata table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """
        )

        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('dec_center', ?)",
            (str(dec_center),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('dec_min', ?)",
            (str(dec_min),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('dec_max', ?)",
            (str(dec_max),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('build_time_iso', ?)",
            (datetime.now(UTC).isoformat(),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('n_sources', ?)",
            (str(len(insert_data)),),
        )

        # Store coverage status
        coverage_limits = CATALOG_COVERAGE_LIMITS.get("first", {})
        within_coverage = dec_center >= coverage_limits.get(
            "dec_min", -90.0
        ) and dec_center <= coverage_limits.get("dec_max", 90.0)
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('within_coverage', ?)",
            ("true" if within_coverage else "false",),
        )

        source_file_str = (
            str(first_catalog_path) if first_catalog_path else "auto-downloaded/cached"
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('source_file', ?)",
            (source_file_str,),
        )
        if min_flux_mjy is not None:
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('min_flux_mjy', ?)",
                (str(min_flux_mjy),),
            )

        conn.commit()

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f":check: Database created: {output_path}")
        print(f"  Size: {file_size_mb:.2f} MB")
        print(f"  Sources: {len(insert_data)}")

        return output_path
    finally:
        # Always release the lock
        _release_db_lock(lock_fd, lock_path)


def build_rax_strip_db(
    dec_center: float,
    dec_range: tuple[float, float],
    rax_catalog_path: str | None = None,
    output_path: str | os.PathLike[str] | None = None,
    min_flux_mjy: float | None = None,
    cache_dir: str = ".cache/catalogs",
    prefer_full_db: bool = True,
) -> Path:
    """Build SQLite database for RAX sources in a declination strip.

        If a full RAX database (rax_full.sqlite3) exists and prefer_full_db=True,
        the strip will be built from that database (faster). Otherwise, falls back
        to the cached catalog file.

    Parameters
    ----------
    dec_center : float
        Center declination in degrees
    dec_range : tuple of float
        Tuple of (dec_min, dec_max) in degrees
    rax_catalog_path : Optional[str], optional
        Path to RAX catalog (CSV/FITS). If None, attempts to find cached catalog.
        Default is None.
    output_path : Optional[str | os.PathLike[str]], optional
        Output SQLite database path (auto-generated if None). Default is None.
    min_flux_mjy : Optional[float], optional
        Minimum flux threshold in mJy (None = no threshold). Default is None.
    cache_dir : str, optional
        Directory for caching catalog files. Default is ".cache/catalogs".
    prefer_full_db : bool, optional
        If True, use rax_full.sqlite3 if available. Default is True.

    Returns
    -------
        None
    """
    from dsa110_contimg.core.calibration.catalogs import read_rax_catalog
    from dsa110_contimg.core.catalog.build_master import _normalize_columns

    dec_min, dec_max = dec_range

    # Resolve output path
    if output_path is None:
        dec_rounded = round(dec_center, 1)
        db_name = f"rax_dec{dec_rounded:+.1f}.sqlite3"
        output_path = (
            get_env_path("CONTIMG_STATE_DIR", default="/data/dsa110-contimg/state")
            / "catalogs"
            / db_name
        )

    output_path = Path(output_path)

    # Check if already exists
    if output_path.exists():
        logger.info(f"RAX dec strip database already exists: {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try to use full database if available and preferred
    if prefer_full_db and rax_full_db_exists():
        logger.info("Using full RAX database for faster strip extraction")
        return build_rax_strip_from_full(
            dec_center=dec_center,
            dec_range=dec_range,
            output_path=output_path,
            min_flux_mjy=min_flux_mjy,
        )

    # Fall back to raw catalog
    logger.info("Building from raw RAX catalog (full DB not available)")

    # Check coverage limits
    coverage_limits = CATALOG_COVERAGE_LIMITS.get("rax", {})
    if dec_center < coverage_limits.get("dec_min", -90.0):
        logger.warning(
            f":warning:  Declination {dec_center:.2f}° is outside RACS coverage "
            f"(southern limit: {coverage_limits.get('dec_min', -90.0)}°). "
            f"Database may be empty or have very few sources."
        )
    if dec_center > coverage_limits.get("dec_max", 90.0):
        logger.warning(
            f":warning:  Declination {dec_center:.2f}° is outside RACS coverage "
            f"(northern limit: {coverage_limits.get('dec_max', 49.9)}°). "
            f"Database may be empty or have very few sources."
        )

    # Load RAX catalog (uses cached or provided path)
    df_full = read_rax_catalog(cache_dir=cache_dir, rax_catalog_path=rax_catalog_path)

    # Normalize column names (RAX typically similar to NVSS structure)
    RAX_CANDIDATES = {
        "ra": ["ra", "ra_deg", "raj2000", "ra_hms"],
        "dec": ["dec", "dec_deg", "dej2000", "dec_dms"],
        "flux": ["flux", "flux_mjy", "flux_jy", "peak_flux", "fpeak", "s1.4"],
    }

    col_map = _normalize_columns(df_full, RAX_CANDIDATES)

    # Standardize column names
    ra_col = col_map.get("ra", "ra")
    dec_col = col_map.get("dec", "dec")
    flux_col = col_map.get("flux", None)

    # Filter to declination strip
    df_strip = df_full[
        (pd.to_numeric(df_full[dec_col], errors="coerce") >= dec_min)
        & (pd.to_numeric(df_full[dec_col], errors="coerce") <= dec_max)
    ].copy()

    print(f"Filtered RAX catalog: {len(df_full)} :arrow_right: {len(df_strip)} sources")
    print(f"Declination range: {dec_min:.6f} to {dec_max:.6f} degrees")

    # Warn if result is empty
    if len(df_strip) == 0:
        coverage_limits = CATALOG_COVERAGE_LIMITS.get("rax", {})
        logger.warning(
            f":warning:  No RACS sources found in declination range [{dec_min:.2f}°, {dec_max:.2f}°]. "
            f"This may indicate declination {dec_center:.2f}° is outside RACS coverage limits "
            f"(northern limit: {coverage_limits.get('dec_max', 49.9)}°)."
        )

    # Apply flux threshold if specified
    if min_flux_mjy is not None and flux_col:
        flux_val = pd.to_numeric(df_strip[flux_col], errors="coerce")
        # Convert to mJy if needed (assume > 1000 means Jy, otherwise mJy)
        if len(flux_val) > 0 and flux_val.max() > 1000:
            flux_val = flux_val * 1000.0  # Convert Jy to mJy
        df_strip = df_strip[flux_val >= min_flux_mjy].copy()
        print(f"After flux cut ({min_flux_mjy} mJy): {len(df_strip)} sources")

    # Check if database already exists (another process may have created it)
    if output_path.exists():
        logger.info(f"Database {output_path} already exists, skipping build")
        return output_path

    # Acquire lock for database creation
    lock_path = output_path.with_suffix(".lock")
    lock_fd = _acquire_db_lock(lock_path, timeout_sec=300.0)

    if lock_fd is None:
        # Could not acquire lock - check if database was created by another process
        if output_path.exists():
            logger.info(f"Database {output_path} was created by another process")
            return output_path
        else:
            raise RuntimeError(
                f"Could not acquire lock for {output_path} and database does not exist"
            )

    try:
        # Double-check database doesn't exist (another process may have created it while we waited)
        if output_path.exists():
            logger.info(
                f"Database {output_path} was created by another process while waiting for lock"
            )
            return output_path

        # Create SQLite database
        print(f"Creating SQLite database: {output_path}")

        # Enable WAL mode for concurrent reads
        with sqlite3.connect(str(output_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")

        with sqlite3.connect(str(output_path)) as conn:
            # Create sources table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sources (
                    source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ra_deg REAL NOT NULL,
                    dec_deg REAL NOT NULL,
                flux_mjy REAL,
                UNIQUE(ra_deg, dec_deg)
            )
        """
            )

        # Create spatial index
        conn.execute("CREATE INDEX IF NOT EXISTS idx_radec ON sources(ra_deg, dec_deg)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dec ON sources(dec_deg)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_flux ON sources(flux_mjy)")

        # Clear existing data
        conn.execute("DELETE FROM sources")

        # Insert sources
        insert_data = []
        for _, row in df_strip.iterrows():
            ra = pd.to_numeric(row[ra_col], errors="coerce")
            dec = pd.to_numeric(row[dec_col], errors="coerce")

            if not (np.isfinite(ra) and np.isfinite(dec)):
                continue

            # Handle flux
            flux = None
            if flux_col and flux_col in row.index:
                flux_val = pd.to_numeric(row[flux_col], errors="coerce")
                if np.isfinite(flux_val):
                    # Convert to mJy if needed (assume > 1000 means Jy, otherwise mJy)
                    flux_val_float = float(flux_val)
                    if flux_val_float > 1000:
                        flux = flux_val_float * 1000.0  # Convert Jy to mJy
                    else:
                        flux = flux_val_float

            insert_data.append((ra, dec, flux))

        conn.executemany(
            "INSERT OR IGNORE INTO sources(ra_deg, dec_deg, flux_mjy) VALUES(?, ?, ?)",
            insert_data,
        )

        # Create metadata table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """
        )

        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('dec_center', ?)",
            (str(dec_center),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('dec_min', ?)",
            (str(dec_min),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('dec_max', ?)",
            (str(dec_max),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('build_time_iso', ?)",
            (datetime.now(UTC).isoformat(),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('n_sources', ?)",
            (str(len(insert_data)),),
        )

        # Store coverage status
        coverage_limits = CATALOG_COVERAGE_LIMITS.get("rax", {})
        within_coverage = dec_center >= coverage_limits.get(
            "dec_min", -90.0
        ) and dec_center <= coverage_limits.get("dec_max", 90.0)
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('within_coverage', ?)",
            ("true" if within_coverage else "false",),
        )

        source_file_str = str(rax_catalog_path) if rax_catalog_path else "auto-downloaded/cached"
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('source_file', ?)",
            (source_file_str,),
        )
        if min_flux_mjy is not None:
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('min_flux_mjy', ?)",
                (str(min_flux_mjy),),
            )

        conn.commit()

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f":check: Database created: {output_path}")
        print(f"  Size: {file_size_mb:.2f} MB")
        print(f"  Sources: {len(insert_data)}")

        return output_path
    finally:
        # Always release the lock
        _release_db_lock(lock_fd, lock_path)


def build_vlass_strip_db(
    dec_center: float,
    dec_range: tuple[float, float],
    vlass_catalog_path: str | None = None,
    output_path: str | os.PathLike[str] | None = None,
    min_flux_mjy: float | None = None,
    cache_dir: str = ".cache/catalogs",
    prefer_full_db: bool = True,
) -> Path:
    """Build SQLite database for VLASS sources in a declination strip.

        If a full VLASS database (vlass_full.sqlite3) exists and prefer_full_db=True,
        the strip will be built from that database (faster). Otherwise, falls back
        to the cached catalog file.

    Parameters
    ----------
    dec_center : float
        Center declination in degrees
    dec_range : tuple of float
        Tuple of (dec_min, dec_max) in degrees
    vlass_catalog_path : Optional[str], optional
        Path to VLASS catalog (CSV/FITS). If None, attempts to find cached catalog.
        Default is None.
    output_path : Optional[str | os.PathLike[str]], optional
        Output SQLite database path (auto-generated if None). Default is None.
    min_flux_mjy : Optional[float], optional
        Minimum flux threshold in mJy (None = no threshold). Default is None.
    cache_dir : str, optional
        Directory for caching catalog files. Default is ".cache/catalogs".
    prefer_full_db : bool, optional
        If True, use vlass_full.sqlite3 if available. Default is True.

    Returns
    -------
        None
    """
    from dsa110_contimg.core.catalog.build_master import _normalize_columns, _read_table

    dec_min, dec_max = dec_range

    # Resolve output path
    if output_path is None:
        dec_rounded = round(dec_center, 1)
        db_name = f"vlass_dec{dec_rounded:+.1f}.sqlite3"
        output_path = (
            get_env_path("CONTIMG_STATE_DIR", default="/data/dsa110-contimg/state")
            / "catalogs"
            / db_name
        )

    output_path = Path(output_path)

    # Check if already exists
    if output_path.exists():
        logger.info(f"VLASS dec strip database already exists: {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try to use full database if available and preferred
    if prefer_full_db and vlass_full_db_exists():
        logger.info("Using full VLASS database for faster strip extraction")
        return build_vlass_strip_from_full(
            dec_center=dec_center,
            dec_range=dec_range,
            output_path=output_path,
            min_flux_mjy=min_flux_mjy,
        )

    # Fall back to raw catalog
    logger.info("Building from raw VLASS catalog (full DB not available)")

    # Load VLASS catalog
    if vlass_catalog_path:
        df_full = _read_table(vlass_catalog_path)
    else:
        # Try to find cached VLASS catalog
        cache_path = Path(cache_dir) / "vlass_catalog"
        for ext in [".csv", ".fits", ".fits.gz", ".csv.gz"]:
            candidate = cache_path.with_suffix(ext)
            if candidate.exists():
                df_full = _read_table(str(candidate))
                break
        else:
            raise FileNotFoundError(
                f"VLASS catalog not found. Provide vlass_catalog_path or place "
                f"catalog in {cache_dir}/vlass_catalog.csv or .fits"
            )

    # Normalize column names for VLASS
    VLASS_CANDIDATES = {
        "ra": ["ra", "ra_deg", "raj2000"],
        "dec": ["dec", "dec_deg", "dej2000"],
        "flux": ["peak_flux", "peak_mjy_per_beam", "flux_peak", "flux", "total_flux"],
    }

    col_map = _normalize_columns(df_full, VLASS_CANDIDATES)

    # Standardize column names
    ra_col = col_map.get("ra", "ra")
    dec_col = col_map.get("dec", "dec")
    flux_col = col_map.get("flux", None)

    # Filter to declination strip
    df_strip = df_full[
        (pd.to_numeric(df_full[dec_col], errors="coerce") >= dec_min)
        & (pd.to_numeric(df_full[dec_col], errors="coerce") <= dec_max)
    ].copy()

    print(f"Filtered VLASS catalog: {len(df_full)} :arrow_right: {len(df_strip)} sources")
    print(f"Declination range: {dec_min:.6f} to {dec_max:.6f} degrees")

    # Apply flux threshold if specified
    if min_flux_mjy is not None and flux_col:
        flux_val = pd.to_numeric(df_strip[flux_col], errors="coerce")
        # Convert to mJy if needed (assume > 1000 means Jy, otherwise mJy)
        if len(flux_val) > 0 and flux_val.max() > 1000:
            flux_val = flux_val * 1000.0  # Convert Jy to mJy
        df_strip = df_strip[flux_val >= min_flux_mjy].copy()
        print(f"After flux cut ({min_flux_mjy} mJy): {len(df_strip)} sources")

    # Create SQLite database
    print(f"Creating SQLite database: {output_path}")

    with sqlite3.connect(str(output_path)) as conn:
        # Create sources table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sources (
                source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ra_deg REAL NOT NULL,
                dec_deg REAL NOT NULL,
                flux_mjy REAL,
                UNIQUE(ra_deg, dec_deg)
            )
        """
        )

        # Create spatial index
        conn.execute("CREATE INDEX IF NOT EXISTS idx_radec ON sources(ra_deg, dec_deg)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dec ON sources(dec_deg)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_flux ON sources(flux_mjy)")

        # Clear existing data
        conn.execute("DELETE FROM sources")

        # Insert sources
        insert_data = []
        for _, row in df_strip.iterrows():
            ra = pd.to_numeric(row[ra_col], errors="coerce")
            dec = pd.to_numeric(row[dec_col], errors="coerce")

            if not (np.isfinite(ra) and np.isfinite(dec)):
                continue

            # Handle flux
            flux = None
            if flux_col and flux_col in row.index:
                flux_val = pd.to_numeric(row[flux_col], errors="coerce")
                if np.isfinite(flux_val):
                    # Convert to mJy if needed (assume > 1000 means Jy, otherwise mJy)
                    flux_val_float = float(flux_val)
                    if flux_val_float > 1000:
                        flux = flux_val_float * 1000.0  # Convert Jy to mJy
                    else:
                        flux = flux_val_float

            if np.isfinite(ra) and np.isfinite(dec):
                insert_data.append((ra, dec, flux))

        conn.executemany(
            "INSERT OR IGNORE INTO sources(ra_deg, dec_deg, flux_mjy) VALUES(?, ?, ?)",
            insert_data,
        )

        # Create metadata table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """
        )

        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('dec_center', ?)",
            (str(dec_center),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('dec_min', ?)",
            (str(dec_min),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('dec_max', ?)",
            (str(dec_max),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('build_time_iso', ?)",
            (datetime.now(UTC).isoformat(),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('n_sources', ?)",
            (str(len(insert_data)),),
        )
        source_file_str = (
            str(vlass_catalog_path) if vlass_catalog_path else "auto-downloaded/cached"
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('source_file', ?)",
            (source_file_str,),
        )
        if min_flux_mjy is not None:
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('min_flux_mjy', ?)",
                (str(min_flux_mjy),),
            )

        conn.commit()

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f":check: Database created: {output_path}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Sources: {len(insert_data)}")

    return output_path


def build_atnf_strip_db(
    dec_center: float,
    dec_range: tuple[float, float],
    output_path: str | os.PathLike[str] | None = None,
    min_flux_mjy: float | None = None,
    cache_dir: str = ".cache/catalogs",
    prefer_full_db: bool = True,
) -> Path:
    """Build SQLite database for ATNF pulsars in a declination strip.

        If a full ATNF database (atnf_full.sqlite3) exists and prefer_full_db=True,
        the strip will be built from that database (faster). Otherwise, falls back
        to downloading from psrqpy.

    Parameters
    ----------
    dec_center : float
        Center declination in degrees
    dec_range : tuple of float
        Tuple of (dec_min, dec_max) in degrees
    output_path : Optional[str | os.PathLike[str]], optional
        Output SQLite database path (auto-generated if None). Default is None.
    min_flux_mjy : Optional[float], optional
        Minimum flux at 1400 MHz in mJy (None = no threshold). Default is None.
    cache_dir : str, optional
        Directory for caching catalog files. Default is ".cache/catalogs".
    prefer_full_db : bool, optional
        If True, use atnf_full.sqlite3 if available. Default is True.

    Returns
    -------
        None
    """
    from dsa110_contimg.core.catalog.build_atnf_pulsars import (
        _download_atnf_catalog,
        _process_atnf_data,
    )

    dec_min, dec_max = dec_range

    # Resolve output path
    if output_path is None:
        dec_rounded = round(dec_center, 1)
        db_name = f"atnf_dec{dec_rounded:+.1f}.sqlite3"
        output_path = (
            get_env_path("CONTIMG_STATE_DIR", default="/data/dsa110-contimg/state")
            / "catalogs"
            / db_name
        )

    output_path = Path(output_path)

    # Check if already exists
    if output_path.exists():
        logger.info(f"ATNF dec strip database already exists: {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try to use full database if available and preferred
    if prefer_full_db and atnf_full_db_exists():
        logger.info("Using full ATNF database for faster strip extraction")
        return build_atnf_strip_from_full(
            dec_center=dec_center,
            dec_range=dec_range,
            output_path=output_path,
            min_flux_mjy=min_flux_mjy,
        )

    # Fall back to psrqpy download
    logger.info("Building from ATNF psrqpy download (full DB not available)")

    # Check coverage limits (ATNF is all-sky, but warn if outside typical range)
    coverage_limits = CATALOG_COVERAGE_LIMITS.get("atnf", {})
    if dec_center < coverage_limits.get("dec_min", -90.0):
        logger.warning(
            f":warning:  Declination {dec_center:.2f}° is outside typical ATNF coverage "
            f"(southern limit: {coverage_limits.get('dec_min', -90.0)}°). "
            f"Database may be empty or have very few sources."
        )
    if dec_center > coverage_limits.get("dec_max", 90.0):
        logger.warning(
            f":warning:  Declination {dec_center:.2f}° is outside typical ATNF coverage "
            f"(northern limit: {coverage_limits.get('dec_max', 90.0)}°). "
            f"Database may be empty or have very few sources."
        )

    # Acquire lock for database creation
    lock_path = output_path.with_suffix(".lock")
    lock_fd = _acquire_db_lock(lock_path, timeout_sec=300.0)

    if lock_fd is None:
        # Could not acquire lock - check if database was created by another process
        if output_path.exists():
            logger.info(f"Database {output_path} was created by another process")
            return output_path
        else:
            raise RuntimeError(
                f"Could not acquire lock for {output_path} and database does not exist"
            )

    try:
        # Double-check database doesn't exist (another process may have created it while we waited)
        if output_path.exists():
            logger.info(
                f"Database {output_path} was created by another process while waiting for lock"
            )
            return output_path

        # Download and process ATNF catalog
        print("Downloading ATNF Pulsar Catalogue...")
        df_raw = _download_atnf_catalog()
        df_processed = _process_atnf_data(df_raw, min_flux_mjy=None)  # Filter by flux later

        # Filter to declination strip
        df_strip = df_processed[
            (df_processed["dec_deg"] >= dec_min) & (df_processed["dec_deg"] <= dec_max)
        ].copy()

        print(f"Filtered ATNF catalog: {len(df_processed)} :arrow_right: {len(df_strip)} pulsars")
        print(f"Declination range: {dec_min:.6f} to {dec_max:.6f} degrees")

        # Warn if result is empty
        if len(df_strip) == 0:
            logger.warning(
                f":warning:  No ATNF pulsars found in declination range [{dec_min:.2f}°, {dec_max:.2f}°]."
            )

        # Apply flux threshold if specified (use 1400 MHz flux)
        if min_flux_mjy is not None:
            has_flux = df_strip["flux_1400mhz_mjy"].notna()
            bright_enough = df_strip["flux_1400mhz_mjy"] >= min_flux_mjy
            df_strip = df_strip[has_flux & bright_enough].copy()
            print(f"After flux cut ({min_flux_mjy} mJy at 1400 MHz): {len(df_strip)} pulsars")

        # Create SQLite database
        print(f"Creating SQLite database: {output_path}")

        # Enable WAL mode for concurrent reads
        with sqlite3.connect(str(output_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")

        with sqlite3.connect(str(output_path)) as conn:
            # Create sources table (same schema as other strip databases)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sources (
                    source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ra_deg REAL NOT NULL,
                    dec_deg REAL NOT NULL,
                    flux_mjy REAL,
                    UNIQUE(ra_deg, dec_deg)
                )
            """
            )

            # Create spatial index
            conn.execute("CREATE INDEX IF NOT EXISTS idx_radec ON sources(ra_deg, dec_deg)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dec ON sources(dec_deg)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_flux ON sources(flux_mjy)")

            # Clear existing data
            conn.execute("DELETE FROM sources")

            # Insert sources (use flux_1400mhz_mjy as flux_mjy)
            insert_data = []
            for _, row in df_strip.iterrows():
                ra = float(row["ra_deg"])
                dec = float(row["dec_deg"])
                flux = (
                    float(row["flux_1400mhz_mjy"])
                    if pd.notna(row.get("flux_1400mhz_mjy"))
                    else None
                )

                if np.isfinite(ra) and np.isfinite(dec):
                    insert_data.append((ra, dec, flux))

            conn.executemany(
                "INSERT OR IGNORE INTO sources(ra_deg, dec_deg, flux_mjy) VALUES(?, ?, ?)",
                insert_data,
            )

            # Create metadata table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """
            )

            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('dec_center', ?)",
                (str(dec_center),),
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('dec_min', ?)",
                (str(dec_min),),
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('dec_max', ?)",
                (str(dec_max),),
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('build_time_iso', ?)",
                (datetime.now(UTC).isoformat(),),
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('n_sources', ?)",
                (str(len(insert_data)),),
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('source_file', ?)",
                ("ATNF Pulsar Catalogue (psrqpy)",),
            )
            if min_flux_mjy is not None:
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES('min_flux_mjy', ?)",
                    (str(min_flux_mjy),),
                )

            # Store coverage status
            within_coverage = dec_center >= coverage_limits.get(
                "dec_min", -90.0
            ) and dec_center <= coverage_limits.get("dec_max", 90.0)
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('within_coverage', ?)",
                ("true" if within_coverage else "false",),
            )

            conn.commit()

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f":check: Database created: {output_path}")
        print(f"  Size: {file_size_mb:.2f} MB")
        print(f"  Sources: {len(insert_data)}")

        return output_path
    finally:
        # Always release the lock
        _release_db_lock(lock_fd, lock_path)
