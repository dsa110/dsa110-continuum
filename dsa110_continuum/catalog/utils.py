"""Shared catalog loading helpers for ops pipeline scripts.

This module consolidates duplicate catalog loading functions from multiple
pipeline scripts.
"""

import os
import sqlite3
from typing import List, Optional, Tuple

import numpy as np
from dsa110_contimg.core.calibration.catalogs import (
    read_vla_parsed_catalog_csv,
    read_vla_parsed_catalog_with_flux,
)


def load_ra_dec_from_db(
    name: str, vla_db: Optional[str]
) -> Optional[Tuple[float, float]]:
    """Load RA/Dec from SQLite database.

    Args:
        name: Calibrator name
        vla_db: Path to SQLite database file (optional)

    Returns
    -------
        Tuple of (ra_deg, dec_deg) or None if not found

    """
    if not vla_db or not os.path.isfile(vla_db):
        return None
    try:
        with sqlite3.connect(vla_db) as conn:
            row = conn.execute(
                "SELECT ra_deg, dec_deg FROM calibrators WHERE name=?", (name,)
            ).fetchone()
            if row:
                return float(row[0]), float(row[1])
    except Exception:
        return None
    return None


def load_ra_dec(
    name: str, catalogs: List[str], vla_db: Optional[str] = None
) -> Tuple[float, float]:
    """Load RA/Dec for a calibrator from database or catalog files.

    Prefers SQLite database if available, otherwise searches catalog files.

    Args:
        name: Calibrator name
        catalogs: List of catalog file paths to search
        vla_db: Optional path to SQLite database file

    Returns
    -------
        Tuple of (ra_deg, dec_deg)

    Raises
    ------
        RuntimeError: If calibrator not found in any catalog or database

    """
    # Prefer SQLite DB if available
    db_val = load_ra_dec_from_db(name, vla_db)
    if db_val is not None:
        return db_val

    # Search catalog files
    for p in catalogs:
        try:
            df = read_vla_parsed_catalog_csv(p)
            if name in df.index:
                row = df.loc[name]
                try:
                    ra = float(row["ra_deg"].iloc[0])
                    dec = float(row["dec_deg"].iloc[0])
                except Exception:
                    ra = float(row["ra_deg"])
                    dec = float(row["dec_deg"])
                if np.isfinite(ra) and np.isfinite(dec):
                    return ra, dec
        except Exception:
            continue

    raise RuntimeError(
        f"Calibrator {name} not found in catalogs/DB: {catalogs} | {vla_db}"
    )


def load_flux_jy_from_db(
    name: str, vla_db: Optional[str], band: str = "20cm"
) -> Optional[float]:
    """Load flux density from SQLite database.

    Args:
        name: Calibrator name
        vla_db: Path to SQLite database file (optional)
        band: Frequency band (default: '20cm')

    Returns
    -------
        Flux density in Jy or None if not found

    """
    if not vla_db or not os.path.isfile(vla_db):
        return None
    try:
        with sqlite3.connect(vla_db) as conn:
            if band.lower() == "20cm":
                row = conn.execute(
                    "SELECT flux_jy FROM vla_20cm WHERE name=?", (name,)
                ).fetchone()
                if row and row[0] is not None:
                    return float(row[0])
    except Exception:
        return None
    return None


def load_flux_jy(
    name: str, catalogs: List[str], band: str = "20cm", vla_db: Optional[str] = None
) -> Optional[float]:
    """Load flux density for a calibrator from database or catalog files.

    Prefers SQLite database if available, otherwise searches catalog files.

    Args:
        name: Calibrator name
        catalogs: List of catalog file paths to search
        band: Frequency band (default: '20cm')
        vla_db: Optional path to SQLite database file

    Returns
    -------
        Flux density in Jy or None if not found

    """
    # Prefer SQLite DB if available
    fx = load_flux_jy_from_db(name, vla_db, band=band)
    if fx is not None and np.isfinite(fx):
        return fx

    # Search catalog files
    for p in catalogs:
        try:
            df = read_vla_parsed_catalog_with_flux(p, band=band)
            if name in df.index:
                row = df.loc[name]
                try:
                    fx = float(row["flux_jy"].iloc[0])
                except Exception:
                    fx = float(row["flux_jy"])
                if np.isfinite(fx):
                    return fx
        except Exception:
            continue

    return None
