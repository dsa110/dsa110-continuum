#!/usr/bin/env python3
"""Footprint-aware, catalog-driven source verification for DSA-110 mosaics.

Usage
-----
python scripts/verify_sources.py \\
    --fits /stage/products/2026-02-15_mosaic.fits \\
    [--master-db /data/dsa110-contimg/state/catalogs/master_sources.sqlite3] \\
    [--atnf-db   /data/dsa110-contimg/state/catalogs/atnf_full.sqlite3] \\
    [--out        products/2026-02-15_verify.csv] \\
    [--min-flux-jy 0.010] \\
    [--box-pix 5]

Prints a machine-parseable QA line to stdout:
    VERIFY PASS: median_ratio=0.923 n_sources=47
    VERIFY WARN: median_ratio=0.623 n_sources=12
    VERIFY FAIL: median_ratio=0.198 n_sources=5

PASS = median ratio ≥ 0.80
WARN = 0.70 ≤ median ratio < 0.80
FAIL = median ratio < 0.70  (or fewer than 3 S/N≥3 continuum detections)
"""
from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("verify_sources")

DEFAULT_MASTER_DB = Path("/data/dsa110-contimg/state/catalogs/master_sources.sqlite3")
DEFAULT_NVSS_DB   = Path("/data/dsa110-contimg/state/catalogs/nvss_full.sqlite3")
DEFAULT_ATNF_DB   = Path("/data/dsa110-contimg/state/catalogs/atnf_full.sqlite3")
DEFAULT_MIN_FLUX_JY  = 0.050   # 50 mJy — well above the ~10 mJy noise floor, avoids box confusion
DEFAULT_BOX_PIX      = 5
RATIO_PASS_THRESHOLD = 0.80
RATIO_WARN_THRESHOLD = 0.70
MIN_DETECTIONS_FOR_QA = 3


# ---------------------------------------------------------------------------
# Catalog queries
# ---------------------------------------------------------------------------

def _image_sky_bounds(wcs, ny: int, nx: int) -> tuple[float, float, float, float]:
    """Return (ra_min, ra_max, dec_min, dec_max) from the four image corners."""
    corners_pix = np.array([
        [0, 0], [nx - 1, 0], [0, ny - 1], [nx - 1, ny - 1],
        [nx // 2, 0], [nx // 2, ny - 1], [0, ny // 2], [nx - 1, ny // 2],
    ], dtype=float)
    sky = wcs.all_pix2world(corners_pix, 0)
    ra_vals  = sky[:, 0]
    dec_min  = float(sky[:, 1].min())
    dec_max  = float(sky[:, 1].max())
    # handle RA wrap: if span > 180°, assume cross-0h mosaic
    ra_span = float(ra_vals.max() - ra_vals.min())
    if ra_span > 180.0:
        ra_vals_wrapped = np.where(ra_vals > 180.0, ra_vals - 360.0, ra_vals)
        ra_min = float(ra_vals_wrapped.min())
        ra_max = float(ra_vals_wrapped.max())
    else:
        ra_min = float(ra_vals.min())
        ra_max = float(ra_vals.max())
    return ra_min, ra_max, dec_min, dec_max


def _query_master(
    db_path: Path,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    min_flux_jy: float,
) -> list[dict]:
    """Query master_sources.sqlite3 and return continuum sources."""
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    # Handle RA wrap-around (cross-meridian mosaics)
    if ra_min < ra_max:
        cur.execute(
            "SELECT source_id, ra_deg, dec_deg, flux_jy "
            "FROM sources "
            "WHERE dec_deg BETWEEN ? AND ? "
            "  AND ra_deg  BETWEEN ? AND ? "
            "  AND flux_jy >= ?",
            (dec_min, dec_max, ra_min, ra_max, min_flux_jy),
        )
    else:
        # RA range crosses 0h: [ra_min, 360) OR [0, ra_max]
        cur.execute(
            "SELECT source_id, ra_deg, dec_deg, flux_jy "
            "FROM sources "
            "WHERE dec_deg BETWEEN ? AND ? "
            "  AND (ra_deg >= ? OR ra_deg <= ?) "
            "  AND flux_jy >= ?",
            (dec_min, dec_max, ra_min + 360.0, ra_max, min_flux_jy),
        )
    rows = cur.fetchall()
    conn.close()
    log.info("master_sources query: %d candidates in sky box", len(rows))
    return [
        {"source_id": r[0], "ra_deg": r[1], "dec_deg": r[2],
         "flux_jy": r[3], "name": f"MS_{r[0]}", "source_type": "continuum",
         "catalog": "master"}
        for r in rows
    ]


def _query_nvss(
    db_path: Path,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    min_flux_jy: float,
) -> list[dict]:
    """Query nvss_full.sqlite3 and return sources above min_flux_jy.

    NVSS stores flux in mJy; we convert to Jy on output.
    """
    min_flux_mjy = min_flux_jy * 1000.0
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    if ra_min < ra_max:
        cur.execute(
            "SELECT source_id, ra_deg, dec_deg, flux_mjy "
            "FROM sources "
            "WHERE dec_deg BETWEEN ? AND ? "
            "  AND ra_deg  BETWEEN ? AND ? "
            "  AND flux_mjy >= ?",
            (dec_min, dec_max, ra_min, ra_max, min_flux_mjy),
        )
    else:
        cur.execute(
            "SELECT source_id, ra_deg, dec_deg, flux_mjy "
            "FROM sources "
            "WHERE dec_deg BETWEEN ? AND ? "
            "  AND (ra_deg >= ? OR ra_deg <= ?) "
            "  AND flux_mjy >= ?",
            (dec_min, dec_max, ra_min + 360.0, ra_max, min_flux_mjy),
        )
    rows = cur.fetchall()
    conn.close()
    log.info("nvss_full query: %d candidates in sky box", len(rows))
    return [
        {"source_id": r[0], "ra_deg": r[1], "dec_deg": r[2],
         "flux_jy": r[3] / 1000.0, "name": f"NVSS_{r[0]}",
         "source_type": "continuum", "catalog": "nvss"}
        for r in rows
    ]


def _query_atnf(
    db_path: Path,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
) -> list[dict]:
    """Query atnf_full.sqlite3 and return pulsars with known 1400 MHz flux."""
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    if ra_min < ra_max:
        cur.execute(
            "SELECT name, ra_deg, dec_deg, flux_mjy "
            "FROM sources "
            "WHERE dec_deg BETWEEN ? AND ? "
            "  AND ra_deg  BETWEEN ? AND ? "
            "  AND flux_mjy IS NOT NULL",
            (dec_min, dec_max, ra_min, ra_max),
        )
    else:
        cur.execute(
            "SELECT name, ra_deg, dec_deg, flux_mjy "
            "FROM sources "
            "WHERE dec_deg BETWEEN ? AND ? "
            "  AND (ra_deg >= ? OR ra_deg <= ?) "
            "  AND flux_mjy IS NOT NULL",
            (dec_min, dec_max, ra_min + 360.0, ra_max),
        )
    rows = cur.fetchall()
    conn.close()
    log.info("atnf query: %d pulsars in sky box", len(rows))
    return [
        {"source_id": None, "ra_deg": r[1], "dec_deg": r[2],
         "flux_jy": r[3] / 1000.0, "name": r[0],
         "source_type": "pulsar", "catalog": "atnf"}
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------

def verify(
    fits_path: Path,
    master_db: Path,
    nvss_db: Path,
    atnf_db: Path,
    out_csv: Path | None,
    min_flux_jy: float,
    box_pix: int,
) -> int:
    """Run verification; return 0 (PASS), 1 (WARN), 2 (FAIL)."""
    from dsa110_continuum.photometry.footprint import (
        load_mosaic,
        sources_in_footprint,
    )
    from dsa110_continuum.photometry.simple_peak import measure_peak_box

    data, wcs, rms, valid_mask = load_mosaic(fits_path)
    ny, nx = data.shape

    ra_min, ra_max, dec_min, dec_max = _image_sky_bounds(wcs, ny, nx)
    log.info(
        "Image bounds: RA %.2f – %.2f, Dec %.2f – %.2f",
        ra_min, ra_max, dec_min, dec_max,
    )

    # Collect catalog sources
    sources: list[dict] = []
    if master_db.exists():
        sources.extend(_query_master(master_db, ra_min, ra_max, dec_min, dec_max, min_flux_jy))
    else:
        log.warning("master_sources DB not found at %s — skipping", master_db)

    if nvss_db.exists():
        nvss_srcs = _query_nvss(nvss_db, ra_min, ra_max, dec_min, dec_max, min_flux_jy)
        # De-duplicate by position: skip NVSS sources already within 1 arcmin of a master source
        if nvss_srcs and sources:
            master_ra  = np.array([s["ra_deg"]  for s in sources])
            master_dec = np.array([s["dec_deg"] for s in sources])
            added = 0
            for ns in nvss_srcs:
                sep = np.sqrt(
                    ((ns["ra_deg"]  - master_ra) * np.cos(np.radians(ns["dec_deg"]))) ** 2
                    + (ns["dec_deg"] - master_dec) ** 2
                )
                if sep.min() > 1.0 / 60.0:   # > 1 arcmin from any master source
                    sources.append(ns)
                    added += 1
            log.info("nvss: %d new sources added (after dedup vs master)", added)
        else:
            sources.extend(nvss_srcs)
    else:
        log.warning("NVSS DB not found at %s — skipping NVSS sources", nvss_db)

    if atnf_db.exists():
        sources.extend(_query_atnf(atnf_db, ra_min, ra_max, dec_min, dec_max))
    else:
        log.warning("ATNF DB not found at %s — skipping pulsars", atnf_db)

    if not sources:
        print("VERIFY FAIL: median_ratio=nan n_sources=0", flush=True)
        return 2

    # Footprint filter
    ra_arr  = np.array([s["ra_deg"]  for s in sources])
    dec_arr = np.array([s["dec_deg"] for s in sources])
    in_fp   = sources_in_footprint(ra_arr, dec_arr, wcs, valid_mask)
    fp_sources = [s for s, ok in zip(sources, in_fp) if ok]
    log.info("%d / %d sources survive footprint filter", len(fp_sources), len(sources))

    # Measure peak flux for each source
    rows_out = []
    for s in fp_sources:
        flux, snr, xp, yp = measure_peak_box(
            data, wcs, s["ra_deg"], s["dec_deg"],
            box_pix=box_pix, rms=rms,
        )
        cat_flux = s["flux_jy"]
        if np.isfinite(flux) and cat_flux and cat_flux > 0:
            ratio = flux / cat_flux
        else:
            ratio = float("nan")
        is_upper_limit = not (np.isfinite(snr) and snr >= 3.0)
        rows_out.append({
            "source_name":      s["name"],
            "ra_deg":           f"{s['ra_deg']:.6f}",
            "dec_deg":          f"{s['dec_deg']:.6f}",
            "catalog_flux_jy":  f"{cat_flux:.5g}" if cat_flux else "nan",
            "dsa_peak_jyb":     f"{flux:.5g}" if np.isfinite(flux) else "nan",
            "snr":              f"{snr:.2f}"  if np.isfinite(snr)  else "nan",
            "ratio":            f"{ratio:.4f}" if np.isfinite(ratio) else "nan",
            "source_type":      s["source_type"],
            "is_upper_limit":   str(is_upper_limit),
            "catalog":          s["catalog"],
        })

    # Write CSV
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()) if rows_out else [])
            writer.writeheader()
            writer.writerows(rows_out)
        log.info("Wrote %d rows to %s", len(rows_out), out_csv)

    # QA gate: median ratio from S/N≥3 continuum detections
    continuum_ratios = [
        float(r["ratio"])
        for r in rows_out
        if r["source_type"] == "continuum"
        and r["is_upper_limit"] == "False"
        and r["ratio"] != "nan"
    ]
    n_det  = len(continuum_ratios)
    if n_det < MIN_DETECTIONS_FOR_QA:
        log.warning("Only %d S/N≥3 continuum detections — insufficient for QA", n_det)
        print(f"VERIFY FAIL: median_ratio=nan n_sources={n_det}", flush=True)
        return 2

    med_ratio = float(np.median(continuum_ratios))
    if med_ratio >= RATIO_PASS_THRESHOLD:
        status, code = "PASS", 0
    elif med_ratio >= RATIO_WARN_THRESHOLD:
        status, code = "WARN", 1
    else:
        status, code = "FAIL", 2

    print(
        f"VERIFY {status}: median_ratio={med_ratio:.3f} n_sources={n_det}",
        flush=True,
    )
    return code


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Footprint-aware catalog verification for DSA-110 mosaics."
    )
    p.add_argument("--fits",        required=True,  type=Path, help="Mosaic FITS path")
    p.add_argument("--master-db",   default=DEFAULT_MASTER_DB, type=Path,
                   help="Path to master_sources.sqlite3")
    p.add_argument("--nvss-db",     default=DEFAULT_NVSS_DB,   type=Path,
                   help="Path to nvss_full.sqlite3 (1.4 GHz reference; full-sky)")
    p.add_argument("--atnf-db",     default=DEFAULT_ATNF_DB,   type=Path,
                   help="Path to atnf_full.sqlite3")
    p.add_argument("--out",         default=None,   type=Path,
                   help="Output CSV path (optional)")
    p.add_argument("--min-flux-jy", default=DEFAULT_MIN_FLUX_JY, type=float,
                   help="Minimum catalog flux to include (Jy). Default 0.010")
    p.add_argument("--box-pix",     default=DEFAULT_BOX_PIX,     type=int,
                   help="Half-width of peak-search box in pixels. Default 5")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    exit_code = verify(
        fits_path  = args.fits,
        master_db  = args.master_db,
        nvss_db    = args.nvss_db,
        atnf_db    = args.atnf_db,
        out_csv    = args.out,
        min_flux_jy= args.min_flux_jy,
        box_pix    = args.box_pix,
    )
    sys.exit(exit_code)
