"""
Generalized catalog querying interface for NVSS, FIRST, RAX, and other source catalogs.

Supports both SQLite databases (per-declination strips) and CSV fallback.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from dsa110_contimg.common.utils import get_env_path


def resolve_catalog_path(
    catalog_type: str,
    dec_strip: float | None = None,
    explicit_path: str | os.PathLike[str] | None = None,
    auto_build: bool = False,
) -> Path:
    """Resolve path to a catalog (SQLite or CSV) using standard precedence.

    Parameters
    ----------
    catalog_type : str
        One of "nvss", "first", "rax", "vlass", "master", "atnf"
    dec_strip : float or None
        Declination in degrees (for per-strip SQLite databases)
    explicit_path : str or None
        Override path (highest priority)
    auto_build : bool, optional
        If True, automatically build missing databases when within
        catalog coverage limits (default: False)

    Returns
    -------
        pathlib.Path
        Path object pointing to catalog file

    Raises
    ------
        FileNotFoundError
        If no catalog can be found (and auto_build=False or
        declination is outside catalog coverage)
    """
    # 1. Explicit path takes highest priority
    if explicit_path:
        path = Path(explicit_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Explicit catalog path does not exist: {explicit_path}")

    # 2. Check environment variable
    env_var = f"{catalog_type.upper()}_CATALOG"
    env_path = os.getenv(env_var)
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    # 3. Try per-declination SQLite database (if dec_strip provided)
    if dec_strip is not None:
        # Round to 0.1 degree precision for filename
        # Handle both scalar and array inputs
        if isinstance(dec_strip, np.ndarray):
            # Extract scalar from array (take first element if array)
            dec_strip = float(dec_strip.flat[0])
        dec_rounded = round(float(dec_strip), 1)
        # Map catalog type to database name
        db_name = f"{catalog_type}_dec{dec_rounded:+.1f}.sqlite3"

        # Try standard locations
        candidates = []
        try:
            current_file = Path(__file__).resolve()
            potential_root = current_file.parents[3]
            if (potential_root / "src" / "dsa110_contimg").exists():
                candidates.append(potential_root / "state" / "catalogs" / db_name)
        except (IndexError, OSError):
            pass

        for root_str in ["/data/dsa110-contimg", "/app"]:
            root_path = Path(root_str)
            if root_path.exists():
                candidates.append(root_path / "state" / "catalogs" / db_name)

        candidates.append(Path.cwd() / "state" / "catalogs" / db_name)
        candidates.append(
            get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg")
            / "state/catalogs"
            / db_name
        )

        for candidate in candidates:
            if candidate.exists():
                return candidate

        # If exact match not found, try to find nearest declination match (within 1.0 degree tolerance)
        catalog_dirs = []
        for root_str in ["/data/dsa110-contimg", "/app"]:
            root_path = Path(root_str)
            if root_path.exists():
                catalog_dirs.append(root_path / "state" / "catalogs")
        try:
            current_file = Path(__file__).resolve()
            potential_root = current_file.parents[3]
            if (potential_root / "src" / "dsa110_contimg").exists():
                catalog_dirs.append(potential_root / "state" / "catalogs")
        except (IndexError, OSError):
            pass
        catalog_dirs.append(Path.cwd() / "state" / "catalogs")
        catalog_dirs.append(
            get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg") / "state/catalogs"
        )

        best_match = None
        best_diff = float("inf")
        pattern = f"{catalog_type}_dec*.sqlite3"
        for catalog_dir in catalog_dirs:
            if not catalog_dir.exists():
                continue
            # Find all matching catalog files
            for catalog_file in catalog_dir.glob(pattern):
                try:
                    # Extract declination from filename: nvss_dec+54.6.sqlite3 -> 54.6
                    dec_str = catalog_file.stem.replace(f"{catalog_type}_dec", "").replace("+", "")
                    file_dec = float(dec_str)
                    diff = abs(file_dec - float(dec_strip))
                    if (
                        diff < best_diff and diff <= 6.0
                    ):  # Within 6 degree tolerance (matches strip width)
                        best_diff = diff
                        best_match = catalog_file
                except (ValueError, AttributeError):
                    continue

        if best_match is not None:
            return best_match

    # 4. Try standard catalog locations for all-sky catalogs (master, atnf)
    if catalog_type == "master":
        master_candidates = [
            get_env_path("CONTIMG_STATE_DIR", default="/data/dsa110-contimg/state")
            / "catalogs/master_sources.sqlite3",
            Path("state/catalogs/master_sources.sqlite3"),
        ]
        for candidate in master_candidates:
            if candidate.exists():
                return candidate

    if catalog_type == "atnf":
        atnf_candidates = [
            get_env_path("CONTIMG_STATE_DIR", default="/data/dsa110-contimg/state")
            / "catalogs/atnf_pulsars.sqlite3",
            Path("state/catalogs/atnf_pulsars.sqlite3"),
        ]
        # Also check for current directory variants
        try:
            current_file = Path(__file__).resolve()
            potential_root = current_file.parents[3]
            if (potential_root / "src" / "dsa110_contimg").exists():
                atnf_candidates.insert(
                    0, potential_root / "state" / "catalogs" / "atnf_pulsars.sqlite3"
                )
        except (IndexError, OSError):
            pass

        for candidate in atnf_candidates:
            if candidate.exists():
                return candidate

    # 5. Fallback: CSV (for NVSS, RAX, VLASS)
    if catalog_type in ["nvss", "rax", "vlass"]:
        # CSV fallback is handled in query_sources() function
        pass

    # 6. Auto-build if requested and within coverage
    if auto_build and dec_strip is not None:
        from dsa110_contimg.core.catalog.builders import (
            CATALOG_COVERAGE_LIMITS,
            build_atnf_strip_db,
            build_first_strip_db,
            build_nvss_strip_db,
            build_rax_strip_db,
            build_vlass_strip_db,
        )

        limits = CATALOG_COVERAGE_LIMITS.get(catalog_type, {})
        dec_min = limits.get("dec_min", -90.0)
        dec_max = limits.get("dec_max", 90.0)

        # Handle array inputs
        dec_val = (
            float(dec_strip.flat[0]) if isinstance(dec_strip, np.ndarray) else float(dec_strip)
        )

        if dec_min <= dec_val <= dec_max:
            # Within coverage - build the database
            dec_range = (dec_val - 6.0, dec_val + 6.0)

            try:
                if catalog_type == "nvss":
                    db_path = build_nvss_strip_db(dec_center=dec_val, dec_range=dec_range)
                elif catalog_type == "first":
                    db_path = build_first_strip_db(dec_center=dec_val, dec_range=dec_range)
                elif catalog_type == "rax":
                    db_path = build_rax_strip_db(dec_center=dec_val, dec_range=dec_range)
                elif catalog_type == "vlass":
                    db_path = build_vlass_strip_db(dec_center=dec_val, dec_range=dec_range)
                elif catalog_type == "atnf":
                    db_path = build_atnf_strip_db(dec_center=dec_val, dec_range=dec_range)
                else:
                    db_path = None

                if db_path and db_path.exists():
                    return db_path
            except Exception as e:
                # Log but don't crash - fall through to FileNotFoundError
                import logging

                logging.getLogger(__name__).warning(
                    f"Auto-build of {catalog_type} catalog for dec={dec_val:.1f}° failed: {e}"
                )

    raise FileNotFoundError(
        f"Catalog '{catalog_type}' not found. "
        f"Searched SQLite databases and standard locations. "
        f"Set {env_var} environment variable or provide explicit path."
    )


def query_sources(
    catalog_type: str = "nvss",
    ra_center: float = 0.0,
    dec_center: float = 0.0,
    radius_deg: float = 1.5,
    *,
    dec_strip: float | None = None,
    min_flux_mjy: float | None = None,
    max_sources: int | None = None,
    catalog_path: str | os.PathLike[str] | None = None,
    validate_coverage: bool = True,
    auto_build: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Query sources from catalog within a field of view.

    Parameters
    ----------
    catalog_type : str
        One of "nvss", "first", "rax", "vlass", "master", "atnf"
    ra_center : float
        Field center RA in degrees
    dec_center : float
        Field center Dec in degrees
    radius_deg : float
        Search radius in degrees
    dec_strip : float or None
        Declination strip (auto-detected from dec_center if None)
    min_flux_mjy : float
        Minimum flux in mJy (catalog-specific)
    max_sources : int
        Maximum number of sources to return
    catalog_path : str or None
        Explicit path to catalog (overrides auto-resolution)
    validate_coverage : bool
        If True, check if position is in catalog coverage
    auto_build : bool
        If True, automatically build missing databases when within
        catalog coverage limits (default: False)
        **kwargs
        Catalog-specific query parameters (e.g., min_period_s for ATNF)

    Returns
    -------
        pandas.DataFrame
        DataFrame with columns: ra_deg, dec_deg, flux_mjy, and catalog-specific fields
    """
    # Validate catalog coverage if requested
    if validate_coverage:
        try:
            import logging

            from dsa110_contimg.core.catalog.coverage import validate_catalog_choice

            logger = logging.getLogger(__name__)

            is_valid, warning = validate_catalog_choice(
                catalog_type=catalog_type, ra_deg=ra_center, dec_deg=dec_center
            )

            if not is_valid:
                logger.warning(f"Coverage validation: {warning}")
        except ImportError:
            pass  # coverage module not available

    # Auto-detect dec_strip from dec_center if not provided
    if dec_strip is None:
        # Ensure dec_center is a scalar (handle numpy arrays)
        if isinstance(dec_center, np.ndarray):
            dec_strip = float(dec_center.flat[0])
        else:
            dec_strip = float(dec_center)

    # Handle "racs" as an alias for "rax" (RACS catalog)
    if catalog_type == "racs":
        catalog_type = "rax"

    # Resolve catalog path
    try:
        catalog_file = resolve_catalog_path(
            catalog_type=catalog_type,
            dec_strip=dec_strip,
            explicit_path=catalog_path,
            auto_build=auto_build,
        )
    except FileNotFoundError:
        # Fallback to CSV for supported catalogs
        if catalog_type == "nvss":
            return _query_nvss_csv(
                ra_center=ra_center,
                dec_center=dec_center,
                radius_deg=radius_deg,
                min_flux_mjy=min_flux_mjy,
                max_sources=max_sources,
            )
        elif catalog_type == "rax":
            from dsa110_contimg.core.calibration.catalogs import query_rax_sources

            return query_rax_sources(
                ra_deg=ra_center,
                dec_deg=dec_center,
                radius_deg=radius_deg,
                min_flux_mjy=min_flux_mjy,
                max_sources=max_sources,
            )
        elif catalog_type == "vlass":
            from dsa110_contimg.core.calibration.catalogs import query_vlass_sources

            return query_vlass_sources(
                ra_deg=ra_center,
                dec_deg=dec_center,
                radius_deg=radius_deg,
                min_flux_mjy=min_flux_mjy,
                max_sources=max_sources,
            )
        raise

    # Load from SQLite
    if str(catalog_file).endswith(".sqlite3"):
        return _query_sqlite(
            catalog_type=catalog_type,
            catalog_path=str(catalog_file),
            ra_center=ra_center,
            dec_center=dec_center,
            radius_deg=radius_deg,
            min_flux_mjy=min_flux_mjy,
            max_sources=max_sources,
            **kwargs,
        )
    else:
        # CSV fallback
        return _query_csv(
            catalog_type=catalog_type,
            catalog_path=str(catalog_file),
            ra_center=ra_center,
            dec_center=dec_center,
            radius_deg=radius_deg,
            min_flux_mjy=min_flux_mjy,
            max_sources=max_sources,
            **kwargs,
        )


def _query_sqlite(
    catalog_type: str,
    catalog_path: str,
    ra_center: float,
    dec_center: float,
    radius_deg: float,
    min_flux_mjy: float | None = None,
    max_sources: int | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Query SQLite catalog database.

    If the database is corrupted, automatically falls back to the full catalog
    database if available.
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        conn = sqlite3.connect(catalog_path)
        conn.row_factory = sqlite3.Row
        # Test database integrity with a simple query
        conn.execute("SELECT 1").fetchone()
    except sqlite3.DatabaseError as e:
        logger.warning(f"Database error with {catalog_path}: {e}")
        conn = None

        # Try to fall back to full database
        full_db_path = Path(catalog_path).parent / f"{catalog_type}_full.sqlite3"
        if full_db_path.exists() and str(full_db_path) != catalog_path:
            logger.info(f"Falling back to full database: {full_db_path}")
            try:
                conn = sqlite3.connect(str(full_db_path))
                conn.row_factory = sqlite3.Row
                conn.execute("SELECT 1").fetchone()  # Test integrity
            except sqlite3.DatabaseError as e2:
                logger.error(f"Full database also corrupted: {e2}")
                conn = None

        if conn is None:
            raise RuntimeError(
                f"Catalog database corrupted: {catalog_path}. "
                f"Try rebuilding with: python -m dsa110_contimg.core.catalog.builders --rebuild {catalog_type}"
            ) from e

    try:
        # Approximate box search (faster than exact angular separation)
        dec_half = radius_deg
        ra_half = radius_deg / np.cos(np.radians(dec_center))

        # Build query based on catalog type
        if catalog_type == "nvss":
            flux_col = "flux_mjy"
            where_clauses = [
                "ra_deg BETWEEN ? AND ?",
                "dec_deg BETWEEN ? AND ?",
            ]
            if min_flux_mjy is not None:
                where_clauses.append(f"{flux_col} >= ?")

            query = f"""
            SELECT ra_deg, dec_deg, flux_mjy
            FROM sources
            WHERE {" AND ".join(where_clauses)}
            ORDER BY flux_mjy DESC
            """

            params = [
                ra_center - ra_half,
                ra_center + ra_half,
                dec_center - dec_half,
                dec_center + dec_half,
            ]
            if min_flux_mjy is not None:
                params.append(min_flux_mjy)

            if max_sources:
                query += f" LIMIT {max_sources}"

            rows = conn.execute(query, params).fetchall()

        elif catalog_type == "first":
            # FIRST catalog schema (includes major/minor axes)
            flux_col = "flux_mjy"
            where_clauses = [
                "ra_deg BETWEEN ? AND ?",
                "dec_deg BETWEEN ? AND ?",
            ]
            if min_flux_mjy is not None:
                where_clauses.append(f"{flux_col} >= ?")

            query = f"""
            SELECT ra_deg, dec_deg, flux_mjy, maj_arcsec, min_arcsec
            FROM sources
            WHERE {" AND ".join(where_clauses)}
            ORDER BY flux_mjy DESC
            """

            params = [
                ra_center - ra_half,
                ra_center + ra_half,
                dec_center - dec_half,
                dec_center + dec_half,
            ]
            if min_flux_mjy is not None:
                params.append(min_flux_mjy)

            if max_sources:
                query += f" LIMIT {max_sources}"

            rows = conn.execute(query, params).fetchall()

        elif catalog_type == "rax":
            # RAX catalog schema (similar to NVSS)
            flux_col = "flux_mjy"
            where_clauses = [
                "ra_deg BETWEEN ? AND ?",
                "dec_deg BETWEEN ? AND ?",
            ]
            if min_flux_mjy is not None:
                where_clauses.append(f"{flux_col} >= ?")

            query = f"""
            SELECT ra_deg, dec_deg, flux_mjy
            FROM sources
            WHERE {" AND ".join(where_clauses)}
            ORDER BY flux_mjy DESC
            """

            params = [
                ra_center - ra_half,
                ra_center + ra_half,
                dec_center - dec_half,
                dec_center + dec_half,
            ]
            if min_flux_mjy is not None:
                params.append(min_flux_mjy)

            if max_sources:
                query += f" LIMIT {max_sources}"

            rows = conn.execute(query, params).fetchall()

        elif catalog_type == "atnf":
            # Check which schema is used (full database has 'pulsars' table, strip databases have 'sources' table)
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND (name='pulsars' OR name='sources')"
            )
            tables = [row[0] for row in cursor.fetchall()]

            if "pulsars" in tables:
                # Full ATNF Pulsar Catalogue schema
                where_clauses = [
                    "ra_deg BETWEEN ? AND ?",
                    "dec_deg BETWEEN ? AND ?",
                ]

                # Add flux threshold (1400 MHz)
                if min_flux_mjy is not None:
                    where_clauses.append("flux_1400mhz_mjy >= ?")

                # Add period threshold (if provided via kwargs)
                min_period_s = kwargs.get("min_period_s")
                max_period_s = kwargs.get("max_period_s")
                if min_period_s is not None:
                    where_clauses.append("period_s >= ?")
                if max_period_s is not None:
                    where_clauses.append("period_s <= ?")

                # Add DM threshold (if provided via kwargs)
                min_dm = kwargs.get("min_dm_pc_cm3")
                max_dm = kwargs.get("max_dm_pc_cm3")
                if min_dm is not None:
                    where_clauses.append("dm_pc_cm3 >= ?")
                if max_dm is not None:
                    where_clauses.append("dm_pc_cm3 <= ?")

                # Note: SQLite does not support NULLS LAST clause.
                # DESC naturally places NULLs last for numeric values.
                query = f"""
                SELECT
                    pulsar_name, ra_deg, dec_deg,
                    period_s, period_dot, dm_pc_cm3,
                    flux_400mhz_mjy, flux_1400mhz_mjy, flux_2000mhz_mjy,
                    distance_kpc, pulsar_type, binary_type, association
                FROM pulsars
                WHERE {" AND ".join(where_clauses)}
                ORDER BY flux_1400mhz_mjy DESC
                """

                params = [
                    ra_center - ra_half,
                    ra_center + ra_half,
                    dec_center - dec_half,
                    dec_center + dec_half,
                ]
                if min_flux_mjy is not None:
                    params.append(min_flux_mjy)
                if min_period_s is not None:
                    params.append(min_period_s)
                if max_period_s is not None:
                    params.append(max_period_s)
                if min_dm is not None:
                    params.append(min_dm)
                if max_dm is not None:
                    params.append(max_dm)

                if max_sources:
                    query += f" LIMIT {max_sources}"

                rows = conn.execute(query, params).fetchall()
            elif "sources" in tables:
                # ATNF strip database schema (simpler, like NVSS/FIRST)
                where_clauses = [
                    "ra_deg BETWEEN ? AND ?",
                    "dec_deg BETWEEN ? AND ?",
                ]
                if min_flux_mjy is not None:
                    where_clauses.append("flux_mjy >= ?")

                query = f"""
                SELECT ra_deg, dec_deg, flux_mjy
                FROM sources
                WHERE {" AND ".join(where_clauses)}
                ORDER BY flux_mjy DESC
                """

                params = [
                    ra_center - ra_half,
                    ra_center + ra_half,
                    dec_center - dec_half,
                    dec_center + dec_half,
                ]
                if min_flux_mjy is not None:
                    params.append(min_flux_mjy)

                if max_sources:
                    query += f" LIMIT {max_sources}"

                rows = conn.execute(query, params).fetchall()
            else:
                raise ValueError(
                    f"ATNF database {catalog_path} does not contain 'pulsars' or 'sources' table"
                )

        elif catalog_type == "vlass":
            # VLASS catalog schema (similar to NVSS)
            flux_col = "flux_mjy"
            where_clauses = [
                "ra_deg BETWEEN ? AND ?",
                "dec_deg BETWEEN ? AND ?",
            ]
            if min_flux_mjy is not None:
                where_clauses.append(f"{flux_col} >= ?")

            query = f"""
            SELECT ra_deg, dec_deg, flux_mjy
            FROM sources
            WHERE {" AND ".join(where_clauses)}
            ORDER BY flux_mjy DESC
            """

            params = [
                ra_center - ra_half,
                ra_center + ra_half,
                dec_center - dec_half,
                dec_center + dec_half,
            ]
            if min_flux_mjy is not None:
                params.append(min_flux_mjy)

            if max_sources:
                query += f" LIMIT {max_sources}"

            rows = conn.execute(query, params).fetchall()

        elif catalog_type == "master":
            # Use master_sources schema
            where_clauses = [
                "ra_deg BETWEEN ? AND ?",
                "dec_deg BETWEEN ? AND ?",
            ]
            if min_flux_mjy is not None:
                where_clauses.append("flux_jy >= ?")

            query = f"""
                 SELECT ra_deg, dec_deg, flux_jy * 1000.0 as flux_mjy,
                     snr_nvss, s_nvss, s_vlass, s_first, s_rax,
                     alpha, resolved_flag, confusion_flag,
                     has_nvss, has_vlass, has_first, has_rax
            FROM sources
            WHERE {" AND ".join(where_clauses)}
                 ORDER BY flux_jy DESC
            """

            params = [
                ra_center - ra_half,
                ra_center + ra_half,
                dec_center - dec_half,
                dec_center + dec_half,
            ]
            if min_flux_mjy is not None:
                params.append(min_flux_mjy / 1000.0)  # Convert mJy to Jy

            if max_sources:
                query += f" LIMIT {max_sources}"

            rows = conn.execute(query, params).fetchall()

        else:
            raise ValueError(
                f"Unsupported catalog type for SQLite: {catalog_type}. "
                f"Supported types: nvss, first, rax, vlass, master, atnf"
            )

        # Convert to DataFrame
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([dict(row) for row in rows])

        # Exact angular separation filter
        if len(df) > 0:
            sc = SkyCoord(
                ra=df["ra_deg"].values * u.deg,  # pylint: disable=no-member
                dec=df["dec_deg"].values * u.deg,  # pylint: disable=no-member
                frame="icrs",
            )
            center = SkyCoord(
                ra_center * u.deg,
                dec_center * u.deg,
                frame="icrs",  # pylint: disable=no-member
            )  # pylint: disable=no-member
            sep = sc.separation(center).deg
            df = df[sep <= radius_deg].copy()

        return df

    finally:
        conn.close()


def _query_nvss_csv(
    ra_center: float,
    dec_center: float,
    radius_deg: float,
    min_flux_mjy: float | None = None,
    max_sources: int | None = None,
) -> pd.DataFrame:
    """Fallback: Query NVSS from CSV catalog."""
    from dsa110_contimg.core.calibration.catalogs import read_nvss_catalog

    df = read_nvss_catalog()
    sc = SkyCoord(
        ra=df["ra"].values * u.deg,  # pylint: disable=no-member
        dec=df["dec"].values * u.deg,  # pylint: disable=no-member
        frame="icrs",
    )
    center = SkyCoord(
        ra_center * u.deg,
        dec_center * u.deg,
        frame="icrs",  # pylint: disable=no-member
    )  # pylint: disable=no-member
    sep = sc.separation(center).deg

    keep = sep <= radius_deg
    if min_flux_mjy is not None:
        flux_mjy = pd.to_numeric(df["flux_20_cm"], errors="coerce")
        keep = keep & (flux_mjy >= min_flux_mjy)

    result = df[keep].copy()

    # Rename columns to standard format
    result = result.rename(
        columns={
            "ra": "ra_deg",
            "dec": "dec_deg",
            "flux_20_cm": "flux_mjy",
        }
    )

    # Sort by flux and limit
    if "flux_mjy" in result.columns:
        result = result.sort_values("flux_mjy", ascending=False)
    if max_sources:
        result = result.head(max_sources)

    return result


def _query_csv(
    catalog_type: str,
    catalog_path: str,
    ra_center: float,
    dec_center: float,
    radius_deg: float,
    min_flux_mjy: float | None = None,
    max_sources: int | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Query CSV catalog (fallback)."""
    # For now, only NVSS CSV is supported
    if catalog_type != "nvss":
        raise ValueError(f"CSV fallback not implemented for {catalog_type}")

    return _query_nvss_csv(
        ra_center=ra_center,
        dec_center=dec_center,
        radius_deg=radius_deg,
        min_flux_mjy=min_flux_mjy,
        max_sources=max_sources,
    )
