"""
Generalized catalog querying interface for NVSS, FIRST, RAX, and other source catalogs.

Supports SQLite databases (per-declination strips or full catalogs).

Ported from dsa110_contimg.core.catalog.query.
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

# Hardcoded catalog directory (replaces get_env_path)
_CATALOG_DIR = Path("/data/dsa110-contimg/state/catalogs")


def resolve_catalog_path(
    catalog_type: str,
    dec_strip: float | None = None,
    explicit_path: str | os.PathLike[str] | None = None,
    auto_build: bool = False,
) -> Path:
    """Resolve path to a catalog SQLite database using standard precedence.

    Precedence:
    1. Explicit path override
    2. Environment variable ``{CATALOG_TYPE}_CATALOG``
    3. Per-declination strip database (``{type}_dec{dec:+.1f}.sqlite3``)
    4. Full database (``{type}_full.sqlite3``)
    5. Auto-build strip from full (if ``auto_build=True``)

    Parameters
    ----------
    catalog_type : str
        One of "nvss", "first", "rax", "vlass", "master", "atnf"
    dec_strip : float or None
        Declination in degrees (used to find a strip database)
    explicit_path : str or None
        Override path (highest priority)
    auto_build : bool
        If True, build a strip database from the full database when no strip
        is found but the declination is within coverage limits.
    """
    # 1. Explicit path
    if explicit_path:
        path = Path(explicit_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Explicit catalog path does not exist: {explicit_path}")

    # 2. Environment variable
    env_var = f"{catalog_type.upper()}_CATALOG"
    env_path = os.getenv(env_var)
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    # 3. Per-declination strip database
    if dec_strip is not None:
        if isinstance(dec_strip, np.ndarray):
            dec_strip = float(dec_strip.flat[0])
        dec_rounded = round(float(dec_strip), 1)
        db_name = f"{catalog_type}_dec{dec_rounded:+.1f}.sqlite3"

        # Check standard location
        candidate = _CATALOG_DIR / db_name
        if candidate.exists():
            return candidate

        # Nearest-dec fuzzy match (within 6 deg)
        if _CATALOG_DIR.exists():
            best_match = None
            best_diff = float("inf")
            for catalog_file in _CATALOG_DIR.glob(f"{catalog_type}_dec*.sqlite3"):
                try:
                    dec_str = catalog_file.stem.replace(f"{catalog_type}_dec", "")
                    file_dec = float(dec_str)
                    diff = abs(file_dec - float(dec_strip))
                    if diff < best_diff and diff <= 6.0:
                        best_diff = diff
                        best_match = catalog_file
                except (ValueError, AttributeError):
                    continue
            if best_match is not None:
                return best_match

    # 4. Full database fallback
    if catalog_type in ("nvss", "first", "rax", "vlass", "atnf"):
        full_db = _CATALOG_DIR / f"{catalog_type}_full.sqlite3"
        if full_db.exists():
            return full_db

    if catalog_type == "master":
        master_db = _CATALOG_DIR / "master_sources.sqlite3"
        if master_db.exists():
            return master_db

    if catalog_type == "atnf":
        atnf_db = _CATALOG_DIR / "atnf_pulsars.sqlite3"
        if atnf_db.exists():
            return atnf_db

    # 5. Auto-build strip from full
    if auto_build and dec_strip is not None:
        from dsa110_continuum.catalog.builders import (
            CATALOG_COVERAGE_LIMITS,
            build_nvss_strip_from_full,
            build_first_strip_from_full,
            build_rax_strip_from_full,
            build_vlass_strip_from_full,
        )

        limits = CATALOG_COVERAGE_LIMITS.get(catalog_type, {})
        dec_min = limits.get("dec_min", -90.0)
        dec_max = limits.get("dec_max", 90.0)
        dec_val = float(dec_strip.flat[0]) if isinstance(dec_strip, np.ndarray) else float(dec_strip)

        if dec_min <= dec_val <= dec_max:
            dec_range = (dec_val - 6.0, dec_val + 6.0)
            try:
                if catalog_type == "nvss":
                    db_path = build_nvss_strip_from_full(dec_center=dec_val, dec_range=dec_range)
                elif catalog_type == "first":
                    db_path = build_first_strip_from_full(dec_center=dec_val, dec_range=dec_range)
                elif catalog_type == "rax":
                    db_path = build_rax_strip_from_full(dec_center=dec_val, dec_range=dec_range)
                elif catalog_type == "vlass":
                    db_path = build_vlass_strip_from_full(dec_center=dec_val, dec_range=dec_range)
                else:
                    db_path = None

                if db_path and db_path.exists():
                    return db_path
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Auto-build of {catalog_type} for dec={dec_val:.1f}° failed: {e}"
                )

    raise FileNotFoundError(
        f"Catalog '{catalog_type}' not found. "
        f"Searched strip databases and full database in {_CATALOG_DIR}. "
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
        Minimum flux in mJy
    max_sources : int
        Maximum number of sources to return
    catalog_path : str or None
        Explicit path to catalog (overrides auto-resolution)
    validate_coverage : bool
        If True, warn if position is outside catalog coverage
    auto_build : bool
        If True, automatically build missing strip databases

    Returns
    -------
        pandas.DataFrame
        DataFrame with columns: ra_deg, dec_deg, flux_mjy, and catalog-specific fields
    """
    if validate_coverage:
        try:
            import logging
            from dsa110_continuum.catalog.coverage import validate_catalog_choice
            _log = logging.getLogger(__name__)
            is_valid, warning = validate_catalog_choice(
                catalog_type=catalog_type, ra_deg=ra_center, dec_deg=dec_center
            )
            if not is_valid:
                _log.warning(f"Coverage validation: {warning}")
        except ImportError:
            pass

    if dec_strip is None:
        dec_strip = float(dec_center.flat[0]) if isinstance(dec_center, np.ndarray) else float(dec_center)

    # "racs" is an alias for "rax"
    if catalog_type == "racs":
        catalog_type = "rax"

    catalog_file = resolve_catalog_path(
        catalog_type=catalog_type,
        dec_strip=dec_strip,
        explicit_path=catalog_path,
        auto_build=auto_build,
    )

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


def cone_search(
    catalog_type: str,
    ra_center: float,
    dec_center: float,
    radius_deg: float,
    min_flux_mjy: float | None = None,
    max_sources: int | None = None,
    auto_build: bool = False,
) -> pd.DataFrame:
    """Convenience wrapper: query catalog sources within a cone.

    Parameters
    ----------
    catalog_type : str
        One of "nvss", "first", "rax", "racs", "vlass", "master", "atnf"
    ra_center : float
        Cone center RA in degrees
    dec_center : float
        Cone center Dec in degrees
    radius_deg : float
        Search radius in degrees
    min_flux_mjy : float or None
        Minimum flux threshold in mJy
    max_sources : int or None
        Maximum number of sources to return
    auto_build : bool
        If True, build a strip database from the full database if none exists

    Returns
    -------
        pandas.DataFrame
        DataFrame with columns: ra_deg, dec_deg, flux_mjy (and catalog-specific extras)
    """
    return query_sources(
        catalog_type=catalog_type,
        ra_center=ra_center,
        dec_center=dec_center,
        radius_deg=radius_deg,
        min_flux_mjy=min_flux_mjy,
        max_sources=max_sources,
        auto_build=auto_build,
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
    """Query SQLite catalog database with box pre-filter then exact angular separation."""
    import logging
    _log = logging.getLogger(__name__)

    try:
        conn = sqlite3.connect(catalog_path)
        conn.row_factory = sqlite3.Row
        conn.execute("SELECT 1").fetchone()
    except sqlite3.DatabaseError as e:
        _log.warning(f"Database error with {catalog_path}: {e}")
        conn = None
        # Try full database fallback
        full_db_path = Path(catalog_path).parent / f"{catalog_type}_full.sqlite3"
        if full_db_path.exists() and str(full_db_path) != catalog_path:
            _log.info(f"Falling back to full database: {full_db_path}")
            try:
                conn = sqlite3.connect(str(full_db_path))
                conn.row_factory = sqlite3.Row
                conn.execute("SELECT 1").fetchone()
            except sqlite3.DatabaseError as e2:
                _log.error(f"Full database also corrupted: {e2}")
                conn = None
        if conn is None:
            raise RuntimeError(f"Catalog database corrupted: {catalog_path}") from e

    try:
        dec_half = radius_deg
        ra_half = radius_deg / np.cos(np.radians(dec_center))

        if catalog_type == "nvss":
            where_clauses = ["ra_deg BETWEEN ? AND ?", "dec_deg BETWEEN ? AND ?"]
            params: list = [ra_center - ra_half, ra_center + ra_half, dec_center - dec_half, dec_center + dec_half]
            if min_flux_mjy is not None:
                where_clauses.append("flux_mjy >= ?")
                params.append(min_flux_mjy)
            query = f"SELECT ra_deg, dec_deg, flux_mjy FROM sources WHERE {' AND '.join(where_clauses)} ORDER BY flux_mjy DESC"
            if max_sources:
                query += f" LIMIT {max_sources}"
            rows = conn.execute(query, params).fetchall()

        elif catalog_type == "first":
            where_clauses = ["ra_deg BETWEEN ? AND ?", "dec_deg BETWEEN ? AND ?"]
            params = [ra_center - ra_half, ra_center + ra_half, dec_center - dec_half, dec_center + dec_half]
            if min_flux_mjy is not None:
                where_clauses.append("flux_mjy >= ?")
                params.append(min_flux_mjy)
            query = f"SELECT ra_deg, dec_deg, flux_mjy, maj_arcsec, min_arcsec FROM sources WHERE {' AND '.join(where_clauses)} ORDER BY flux_mjy DESC"
            if max_sources:
                query += f" LIMIT {max_sources}"
            rows = conn.execute(query, params).fetchall()

        elif catalog_type in ("rax", "vlass"):
            where_clauses = ["ra_deg BETWEEN ? AND ?", "dec_deg BETWEEN ? AND ?"]
            params = [ra_center - ra_half, ra_center + ra_half, dec_center - dec_half, dec_center + dec_half]
            if min_flux_mjy is not None:
                where_clauses.append("flux_mjy >= ?")
                params.append(min_flux_mjy)
            query = f"SELECT ra_deg, dec_deg, flux_mjy FROM sources WHERE {' AND '.join(where_clauses)} ORDER BY flux_mjy DESC"
            if max_sources:
                query += f" LIMIT {max_sources}"
            rows = conn.execute(query, params).fetchall()

        elif catalog_type == "atnf":
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND (name='pulsars' OR name='sources')"
            )
            tables = [row[0] for row in cursor.fetchall()]

            if "pulsars" in tables:
                where_clauses = ["ra_deg BETWEEN ? AND ?", "dec_deg BETWEEN ? AND ?"]
                params = [ra_center - ra_half, ra_center + ra_half, dec_center - dec_half, dec_center + dec_half]
                if min_flux_mjy is not None:
                    where_clauses.append("flux_1400mhz_mjy >= ?")
                    params.append(min_flux_mjy)
                min_period_s = kwargs.get("min_period_s")
                max_period_s = kwargs.get("max_period_s")
                min_dm = kwargs.get("min_dm_pc_cm3")
                max_dm = kwargs.get("max_dm_pc_cm3")
                if min_period_s is not None:
                    where_clauses.append("period_s >= ?")
                    params.append(min_period_s)
                if max_period_s is not None:
                    where_clauses.append("period_s <= ?")
                    params.append(max_period_s)
                if min_dm is not None:
                    where_clauses.append("dm_pc_cm3 >= ?")
                    params.append(min_dm)
                if max_dm is not None:
                    where_clauses.append("dm_pc_cm3 <= ?")
                    params.append(max_dm)
                query = f"""
                SELECT pulsar_name, ra_deg, dec_deg,
                    period_s, period_dot, dm_pc_cm3,
                    flux_400mhz_mjy, flux_1400mhz_mjy, flux_2000mhz_mjy,
                    distance_kpc, pulsar_type, binary_type, association
                FROM pulsars WHERE {' AND '.join(where_clauses)}
                ORDER BY flux_1400mhz_mjy DESC
                """
                if max_sources:
                    query += f" LIMIT {max_sources}"
                rows = conn.execute(query, params).fetchall()
            elif "sources" in tables:
                where_clauses = ["ra_deg BETWEEN ? AND ?", "dec_deg BETWEEN ? AND ?"]
                params = [ra_center - ra_half, ra_center + ra_half, dec_center - dec_half, dec_center + dec_half]
                if min_flux_mjy is not None:
                    where_clauses.append("flux_mjy >= ?")
                    params.append(min_flux_mjy)
                query = f"SELECT ra_deg, dec_deg, flux_mjy FROM sources WHERE {' AND '.join(where_clauses)} ORDER BY flux_mjy DESC"
                if max_sources:
                    query += f" LIMIT {max_sources}"
                rows = conn.execute(query, params).fetchall()
            else:
                raise ValueError(f"ATNF database does not contain 'pulsars' or 'sources' table")

        elif catalog_type == "master":
            where_clauses = ["ra_deg BETWEEN ? AND ?", "dec_deg BETWEEN ? AND ?"]
            params = [ra_center - ra_half, ra_center + ra_half, dec_center - dec_half, dec_center + dec_half]
            if min_flux_mjy is not None:
                where_clauses.append("flux_jy >= ?")
                params.append(min_flux_mjy / 1000.0)
            query = f"""
            SELECT ra_deg, dec_deg, flux_jy * 1000.0 as flux_mjy,
                snr_nvss, s_nvss, s_vlass, s_first, s_rax,
                alpha, resolved_flag, confusion_flag,
                has_nvss, has_vlass, has_first, has_rax
            FROM sources WHERE {' AND '.join(where_clauses)}
            ORDER BY flux_jy DESC
            """
            if max_sources:
                query += f" LIMIT {max_sources}"
            rows = conn.execute(query, params).fetchall()

        else:
            raise ValueError(
                f"Unsupported catalog type: {catalog_type}. "
                f"Supported: nvss, first, rax, vlass, master, atnf"
            )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([dict(row) for row in rows])

        # Exact angular separation filter
        if len(df) > 0:
            sc = SkyCoord(ra=df["ra_deg"].values * u.deg, dec=df["dec_deg"].values * u.deg, frame="icrs")
            center_sc = SkyCoord(ra_center * u.deg, dec_center * u.deg, frame="icrs")
            sep = sc.separation(center_sc).deg
            df = df[sep <= radius_deg].copy()

        return df

    finally:
        conn.close()
