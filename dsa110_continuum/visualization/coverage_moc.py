"""
MOC generation and visualization for DSA-110 and external catalogs.

Uses mocpy to generate Multi-Order Coverage (MOC) maps.
"""

import logging
import sqlite3
from pathlib import Path

import numpy as np
from astropy import units as u
from mocpy import MOC

logger = logging.getLogger(__name__)

# Catalog coverage definitions (matching coverage.py)
CATALOG_COVERAGE = {
    "nvss": {"dec_min": -40.0, "dec_max": 90.0},
    "first": {"dec_min": -10.0, "dec_max": 64.0},  # Approximate FIRST coverage
    "racs": {"dec_min": -90.0, "dec_max": 41.0},
    "vlass": {"dec_min": -40.0, "dec_max": 90.0},
    "sumss": {"dec_min": -90.0, "dec_max": -30.0},
    "atnf": {"dec_min": -90.0, "dec_max": 90.0},  # All sky
}


def get_db_path() -> Path:
    """Get path to pipeline database."""
    # Try environment variable or default location
    import os

    db_path = os.environ.get("PIPELINE_DB")
    if not db_path:
        # Fallback relative to this file
        current_dir = Path(__file__).parent
        # backend/src/dsa110_contimg/visualization/ -> backend/src/dsa110_contimg/database/pipeline.sqlite3
        db_path = current_dir.parent / "database" / "pipeline.sqlite3"

    return Path(db_path)


def generate_dsa110_moc(order: int = 10, db_path: Path | None = None) -> MOC:
    """Generate MOC for DSA-110 coverage based on pointing history.

    DSA-110 has a primary beam FWHM of approx 1.36 degrees.
    We approximate the coverage by creating a MOC from the pointing centers
    and expanding it by the beam radius.
    """
    if db_path is None:
        db_path = get_db_path()

    if not db_path.exists():
        logger.warning(f"Database not found at {db_path}. Returning empty MOC.")
        return MOC.new_empty(order)

    try:
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute("SELECT ra_deg, dec_deg FROM pointing_history")
            rows = cursor.fetchall()

        if not rows:
            logger.warning("No pointing history found. Returning empty MOC.")
            return MOC.new_empty(order)

        ra = np.array([r[0] for r in rows]) * u.deg
        dec = np.array([r[1] for r in rows]) * u.deg

        # Beam radius ~ 0.65 deg
        beam_radius = 0.65 * u.deg

        # For mocpy >= 0.12, max_depth is the parameter, not max_norder
        # Also need union_strategy for a single MOC
        moc = MOC.from_cones(
            lon=ra, lat=dec, radius=beam_radius, max_depth=order, union_strategy="small_cones"
        )

        return moc

    except Exception as e:
        logger.error(f"Failed to generate DSA-110 MOC: {e}")
        return MOC.new_empty(order)


def generate_catalog_moc(catalog_name: str, order: int = 9) -> MOC:
    """Generate approximate MOC for a given catalog based on Dec limits."""
    cat_info = CATALOG_COVERAGE.get(catalog_name.lower())
    if not cat_info:
        logger.warning(f"Unknown catalog: {catalog_name}")
        return MOC.new_empty(order)

    dec_min = cat_info["dec_min"] * u.deg
    dec_max = cat_info["dec_max"] * u.deg

    if dec_min <= -90 * u.deg and dec_max >= 90 * u.deg:
        # All sky
        return MOC.from_str("0/0-11")  # Base cells covering sky

    # Create a list of HEALPix indices that fall within the dec range
    import healpy as hp

    nside = 2**order

    # Get theta (co-latitude) range
    theta_min = (90 * u.deg - dec_max).to(u.rad).value
    theta_max = (90 * u.deg - dec_min).to(u.rad).value

    try:
        ipix = hp.query_strip(nside, theta_min, theta_max)
        moc = MOC.from_healpix_cells(ipix, order, order)
        return moc
    except Exception as e:
        logger.error(f"Failed to generate catalog MOC: {e}")
        return MOC.new_empty(order)


def export_moc_to_json(moc: MOC) -> dict:
    """Export MOC to a JSON-serializable format for frontend visualization."""
    return moc.serialize(format="json")


def get_coverage_data(catalog_list: list[str] | None = None) -> dict[str, tuple[dict, float]]:
    """Get MOC data for specified catalogs.

    Returns
    -------
    Dict[str, Tuple[Dict, float]]
        Map of catalog_name -> (MOC JSON data, sky_fraction)
    """
    if catalog_list is None:
        catalog_list = ["dsa110", "nvss", "vlass", "racs", "first", "atnf"]

    results = {}

    for cat in catalog_list:
        cat = cat.lower()
        if cat == "dsa110":
            moc = generate_dsa110_moc(order=9)  # Order 9 is ~7 arcmin res
        else:
            moc = generate_catalog_moc(cat, order=8)  # Lower res for background cats

        results[cat] = (export_moc_to_json(moc), moc.sky_fraction)

    return results


if __name__ == "__main__":
    # Test generation
    logging.basicConfig(level=logging.INFO)
    print("Generating DSA-110 MOC...")
    moc = generate_dsa110_moc()
    print(f"DSA-110 MOC coverage: {moc.sky_fraction:.4f}")

    print("Generating NVSS MOC...")
    moc_nvss = generate_catalog_moc("nvss")
    print(f"NVSS MOC coverage: {moc_nvss.sky_fraction:.4f}")
