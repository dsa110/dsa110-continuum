"""
Unified catalog registry and query interface.

Provides a single, consistent interface to query any supported survey catalog
(NVSS, FIRST, VLASS, ATNF, RAX, UNICAT) with standardized output format.

Examples
--------
>>> from dsa110_contimg.core.calibration.catalog_registry import query_catalog, CatalogName
>>> sources = query_catalog(
...     CatalogName.NVSS,
...     ra_deg=128.5,
...     dec_deg=55.0,
...     radius_deg=1.0,
...     min_flux_mjy=10.0,
... )
>>> print(sources.columns)
Index(['ra_deg', 'dec_deg', 'flux_mjy', 'catalog'])
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from dsa110_contimg.common.utils.paths import get_repo_root
from dsa110_contimg.common.utils import get_env_path

logger = logging.getLogger(__name__)


class CatalogName(str, Enum):
    """Available survey catalogs."""

    NVSS = "nvss"
    FIRST = "first"
    VLASS = "vlass"
    ATNF = "atnf"
    RAX = "rax"
    UNICAT = "unicat"  # Crossmatched master catalog (NVSS+FIRST+VLASS)


@dataclass
class CatalogConfig:
    """Configuration for a survey catalog."""

    name: CatalogName
    description: str
    db_pattern: str  # e.g., "nvss_dec{dec:+.1f}.sqlite3"
    full_db_name: str  # e.g., "nvss_full.sqlite3"
    flux_column: str
    ra_column: str = "ra_deg"
    dec_column: str = "dec_deg"
    default_min_flux_mjy: float = 1.0
    extra_columns: list[str] = field(default_factory=list)


# Registry of all supported catalogs
CATALOG_REGISTRY: dict[CatalogName, CatalogConfig] = {
    CatalogName.NVSS: CatalogConfig(
        name=CatalogName.NVSS,
        description='NRAO VLA Sky Survey (1.4 GHz, 45" beam)',
        db_pattern="nvss_dec{dec:+.1f}.sqlite3",
        full_db_name="nvss_full.sqlite3",
        flux_column="flux_mjy",
        default_min_flux_mjy=1.0,
        extra_columns=["flux_err_mjy", "major_axis", "minor_axis"],
    ),
    CatalogName.FIRST: CatalogConfig(
        name=CatalogName.FIRST,
        description='Faint Images of the Radio Sky at Twenty-cm (1.4 GHz, 5" beam)',
        db_pattern="first_dec{dec:+.1f}.sqlite3",
        full_db_name="first_full.sqlite3",
        flux_column="flux_mjy",
        default_min_flux_mjy=1.0,
        extra_columns=["maj_arcsec", "min_arcsec"],
    ),
    CatalogName.VLASS: CatalogConfig(
        name=CatalogName.VLASS,
        description='VLA Sky Survey (3 GHz, 2.5" beam)',
        db_pattern="vlass_dec{dec:+.1f}.sqlite3",
        full_db_name="vlass_full.sqlite3",
        flux_column="flux_mjy",
        default_min_flux_mjy=1.0,
    ),
    CatalogName.ATNF: CatalogConfig(
        name=CatalogName.ATNF,
        description="Australia Telescope National Facility Pulsar Catalogue",
        db_pattern="atnf_dec{dec:+.1f}.sqlite3",
        full_db_name="atnf_full.sqlite3",
        flux_column="flux_mjy",
        default_min_flux_mjy=0.1,
        extra_columns=["name", "period_s", "dm"],
    ),
    CatalogName.RAX: CatalogConfig(
        name=CatalogName.RAX,
        description="DSA-110 Radio Afterglow eXperiment sources",
        db_pattern="rax_dec{dec:+.1f}.sqlite3",
        full_db_name="rax_full.sqlite3",
        flux_column="flux_mjy",
        default_min_flux_mjy=1.0,
    ),
    CatalogName.UNICAT: CatalogConfig(
        name=CatalogName.UNICAT,
        description="Unified master catalog (union crossmatch of NVSS+VLASS+FIRST+RAX)",
        db_pattern="master_sources.sqlite3",  # Single file, not dec-specific
        full_db_name="master_sources.sqlite3",
        flux_column="flux_jy",  # Best-available flux in Jy (converted to mJy)
        default_min_flux_mjy=5.0,
        extra_columns=[
            "snr_nvss",
            "s_nvss",
            "s_vlass",
            "s_first",
            "s_rax",
            "alpha",
            "resolved_flag",
            "confusion_flag",
            "has_nvss",
            "has_vlass",
            "has_first",
            "has_rax",
        ],
    ),
}


def _resolve_catalog_path(
    catalog: CatalogName,
    dec_deg: float | None = None,
) -> Path | None:
    """Resolve the path to a catalog database.

    Tries declination-specific database first, then falls back to full catalog.

    Parameters
    ----------
    catalog :
        Which catalog to find
    dec_deg :
        Declination in degrees (for dec-specific databases)

    Returns
    -------
        Path to database file, or None if not found

    """
    config = CATALOG_REGISTRY[catalog]

    # Standard catalog locations
    catalog_dirs = [
        get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg") / "state/catalogs",
        Path("/app/state/catalogs"),
        Path.cwd() / "state" / "catalogs",
    ]

    # Also try relative to this file
    repo_root = get_repo_root(Path(__file__))
    catalog_dirs.insert(0, repo_root / "state" / "catalogs")

    # For UNICAT, use the single master file
    if catalog == CatalogName.UNICAT:
        for catalog_dir in catalog_dirs:
            path = catalog_dir / config.full_db_name
            if path.exists():
                return path
        return None

    # For other catalogs, try dec-specific first
    if dec_deg is not None:
        dec_rounded = round(float(dec_deg), 1)
        dec_db_name = config.db_pattern.format(dec=dec_rounded)

        for catalog_dir in catalog_dirs:
            path = catalog_dir / dec_db_name
            if path.exists():
                return path

        # Try to find nearest declination match (within 2 degrees)
        for catalog_dir in catalog_dirs:
            if not catalog_dir.exists():
                continue

            prefix = config.db_pattern.split("{")[0]  # e.g., "nvss_dec"
            for db_file in catalog_dir.glob(f"{prefix}*.sqlite3"):
                # Extract dec from filename
                try:
                    dec_str = db_file.stem.replace(prefix.rstrip("_dec"), "").replace("dec", "")
                    file_dec = float(dec_str.replace("p", ".").replace("m", "-"))
                    if abs(file_dec - dec_deg) < 2.0:
                        return db_file
                except (ValueError, IndexError):
                    continue

    # Fall back to full catalog
    for catalog_dir in catalog_dirs:
        path = catalog_dir / config.full_db_name
        if path.exists():
            return path

    return None


def query_catalog(
    catalog: CatalogName | str,
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    min_flux_mjy: float | None = None,
    max_sources: int | None = None,
    include_extra_columns: bool = False,
) -> pd.DataFrame:
    """Query a catalog for sources within a radius.

        This is the unified interface for all catalog queries. Returns a standardized
        DataFrame with columns: ra_deg, dec_deg, flux_mjy, catalog.

    Parameters
    ----------
    catalog : CatalogName or str
        Catalog to query (CatalogName enum or string like "nvss", "unicat").
    ra_deg : float
        Field center RA in degrees.
    dec_deg : float
        Field center Dec in degrees.
    radius_deg : float
        Search radius in degrees.
    min_flux_mjy : Optional[float]
        Minimum flux in mJy (uses catalog default if not specified). Default is None.
    max_sources : Optional[int]
        Maximum number of sources to return (sorted by flux, descending). Default is None.
    include_extra_columns : bool
        If True, include catalog-specific extra columns. Default is False.

    Returns
    -------
        DataFrame
        DataFrame with standardized columns ra_deg, dec_deg, flux_mjy, catalog.
        Plus any extra columns if include_extra_columns=True.

    Raises
    ------
        ValueError
        If catalog is unknown.
        FileNotFoundError
        If catalog database cannot be found.

    Examples
    --------
        >>> sources = query_catalog("nvss", 128.5, 55.0, 1.0, min_flux_mjy=10.0)
        >>> print(len(sources), "sources found")
    """
    # Normalize catalog name
    if isinstance(catalog, str):
        try:
            catalog = CatalogName(catalog.lower())
        except ValueError as e:
            valid = [c.value for c in CatalogName]
            raise ValueError(f"Unknown catalog '{catalog}'. Valid: {valid}") from e

    config = CATALOG_REGISTRY[catalog]

    # Resolve database path
    db_path = _resolve_catalog_path(catalog, dec_deg)
    if db_path is None:
        raise FileNotFoundError(
            f"Catalog database not found for {catalog.value}. "
            f"Expected: {config.db_pattern.format(dec=round(dec_deg, 1))} or {config.full_db_name}"
        )

    # Use catalog default if min_flux not specified
    if min_flux_mjy is None:
        min_flux_mjy = config.default_min_flux_mjy

    # Build query
    # Approximate box search (faster than exact angular separation)
    dec_half = radius_deg
    ra_half = radius_deg / max(0.1, np.cos(np.radians(dec_deg)))

    # Columns to select
    select_cols = [config.ra_column, config.dec_column, config.flux_column]
    if include_extra_columns and config.extra_columns:
        # Only include columns that exist (check lazily)
        select_cols.extend(config.extra_columns)

    # Handle UNICAT flux conversion (stored in Jy, we want mJy)
    if catalog == CatalogName.UNICAT:
        flux_condition = f"{config.flux_column} >= ?"
        flux_param = min_flux_mjy / 1000.0  # Convert mJy threshold to Jy
    else:
        flux_condition = f"{config.flux_column} >= ?"
        flux_param = min_flux_mjy

    query = f"""
    SELECT {", ".join(select_cols)}
    FROM sources
    WHERE {config.ra_column} BETWEEN ? AND ?
      AND {config.dec_column} BETWEEN ? AND ?
      AND {flux_condition}
    ORDER BY {config.flux_column} DESC
    """

    if max_sources:
        query += f" LIMIT {int(max_sources)}"

    # Execute query
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        params = (
            ra_deg - ra_half,
            ra_deg + ra_half,
            dec_deg - dec_half,
            dec_deg + dec_half,
            flux_param,
        )

        rows = conn.execute(query, params).fetchall()
        conn.close()
    except sqlite3.OperationalError as e:
        # Handle missing columns gracefully
        if "no such column" in str(e):
            # Retry with just core columns
            select_cols = [config.ra_column, config.dec_column, config.flux_column]
            query = f"""
            SELECT {", ".join(select_cols)}
            FROM sources
            WHERE {config.ra_column} BETWEEN ? AND ?
              AND {config.dec_column} BETWEEN ? AND ?
              AND {flux_condition}
            ORDER BY {config.flux_column} DESC
            """
            if max_sources:
                query += f" LIMIT {int(max_sources)}"

            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            conn.close()
        else:
            raise

    if not rows:
        return pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_mjy", "catalog"])

    # Convert to DataFrame
    df = pd.DataFrame([dict(row) for row in rows])

    # Standardize column names
    df = df.rename(
        columns={
            config.ra_column: "ra_deg",
            config.dec_column: "dec_deg",
            config.flux_column: "flux_mjy",
        }
    )

    # UNICAT stores flux in Jy, convert to mJy
    if catalog == CatalogName.UNICAT:
        df["flux_mjy"] = df["flux_mjy"] * 1000.0

    # Add catalog tag
    df["catalog"] = catalog.value

    # Exact angular separation filter (box search is approximate)
    dra = (df["ra_deg"] - ra_deg) * np.cos(np.radians(dec_deg))
    ddec = df["dec_deg"] - dec_deg
    sep = np.sqrt(dra**2 + ddec**2)
    df = df[sep <= radius_deg].copy()

    # Final limit after exact filter
    if max_sources and len(df) > max_sources:
        df = df.head(max_sources)

    logger.debug(
        "query_catalog(%s): found %d sources at (%.2f, %.2f) r=%.3f deg, flux >= %.1f mJy",
        catalog.value,
        len(df),
        ra_deg,
        dec_deg,
        radius_deg,
        min_flux_mjy,
    )

    return df


def query_multiple_catalogs(
    catalogs: list[CatalogName | str],
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    min_flux_mjy: float | None = None,
    max_sources_per_catalog: int | None = None,
    deduplicate_radius_arcsec: float = 10.0,
) -> pd.DataFrame:
    """Query multiple catalogs and optionally deduplicate.

    Parameters
    ----------
    catalogs :
        List of catalogs to query
    ra_deg :
        Field center RA in degrees
    dec_deg :
        Field center Dec in degrees
    radius_deg :
        Search radius in degrees
    min_flux_mjy :
        Minimum flux in mJy
    max_sources_per_catalog :
        Max sources per catalog before merge
    deduplicate_radius_arcsec :
        Matching radius for deduplication (0 to disable)
    catalogs: list[CatalogName | str] :

    Returns
    -------
        Combined DataFrame with sources from all catalogs

    """
    dfs = []

    for cat in catalogs:
        try:
            df = query_catalog(
                cat,
                ra_deg,
                dec_deg,
                radius_deg,
                min_flux_mjy=min_flux_mjy,
                max_sources=max_sources_per_catalog,
            )
            if len(df) > 0:
                dfs.append(df)
        except FileNotFoundError:
            logger.warning("Catalog %s not available, skipping", cat)
            continue

    if not dfs:
        return pd.DataFrame(columns=["ra_deg", "dec_deg", "flux_mjy", "catalog"])

    combined = pd.concat(dfs, ignore_index=True)

    # Deduplicate if requested
    if deduplicate_radius_arcsec > 0 and len(combined) > 1:
        combined = _deduplicate_sources(combined, deduplicate_radius_arcsec)

    return combined.sort_values("flux_mjy", ascending=False).reset_index(drop=True)


def _deduplicate_sources(df: pd.DataFrame, match_radius_arcsec: float) -> pd.DataFrame:
    """Remove duplicate sources, keeping highest flux match.

    Parameters
    ----------
    df: pd.DataFrame :

    """
    if len(df) <= 1:
        return df

    match_radius_deg = match_radius_arcsec / 3600.0

    # Sort by flux descending so we keep brightest
    df = df.sort_values("flux_mjy", ascending=False).reset_index(drop=True)

    keep_mask = np.ones(len(df), dtype=bool)

    for i in range(len(df)):
        if not keep_mask[i]:
            continue

        # Find matches to this source
        dra = (df["ra_deg"].values - df.iloc[i]["ra_deg"]) * np.cos(
            np.radians(df.iloc[i]["dec_deg"])
        )
        ddec = df["dec_deg"].values - df.iloc[i]["dec_deg"]
        sep = np.sqrt(dra**2 + ddec**2)

        # Mark duplicates (excluding self)
        matches = (sep < match_radius_deg) & (np.arange(len(df)) > i)
        keep_mask[matches] = False

    return df[keep_mask].copy()


def list_available_catalogs() -> list[dict]:
    """List all available catalogs with their configurations.

    Returns
    -------
    List of dicts with catalog info
        name, description, available, db_path

    """
    result = []
    for name, config in CATALOG_REGISTRY.items():
        db_path = _resolve_catalog_path(name)
        result.append(
            {
                "name": name.value,
                "description": config.description,
                "available": db_path is not None,
                "db_path": str(db_path) if db_path else None,
                "default_min_flux_mjy": config.default_min_flux_mjy,
            }
        )
    return result
