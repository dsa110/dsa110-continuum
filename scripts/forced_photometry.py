#!/opt/miniforge/envs/casa6/bin/python
"""
Forced photometry: cross-match Aegean catalog against NVSS,
measure flux at NVSS positions in the mosaic, output CSV.

Steps:
  1. Read mosaic FITS + Aegean source catalog
  2. Query NVSS (VII/272) via astroquery.vizier for the mosaic footprint
  3. Cross-match Aegean detections to NVSS sources (nearest within 60")
  4. For each matched NVSS source: measure forced flux at NVSS position
  5. Write CSV with source name, RA, Dec, NVSS flux, measured flux, ratio, S/N
"""
import csv
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, "/data/dsa110-continuum/src") if Path("/data/dsa110-continuum/src").exists() else None

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MOSAIC = "/stage/dsa110-contimg/images/mosaic_2026-01-25/full_mosaic.fits"
MOSAIC_RMS = "/stage/dsa110-contimg/images/mosaic_2026-01-25/full_mosaic_rms.fits"
CATALOG = "/stage/dsa110-contimg/images/mosaic_2026-01-25/sources.fits"
OUT_CSV = "/stage/dsa110-contimg/photometry/2026-01-25_forced_phot.csv"

# NVSS is at 1.4 GHz; DSA-110 is at 1.35 GHz center.
# Expected spectral index α ~ -0.7 → flux ratio = (1.35/1.4)^(-0.7) ≈ 1.03
# Well within factor-2 criterion — no correction applied here.
NVSS_MATCH_RADIUS_ARCSEC = 30.0   # tighter: ≤0.5 beam keeps only genuine matches
# Minimum NVSS flux for comparison: DSA σ≈4.5 mJy → 5σ≈22.5 mJy.
# Use 50 mJy to ensure clean bright-source comparison (peak ≈ integrated flux).
NVSS_MIN_FLUX_JY = 0.050
NVSS_CATALOG_DIR = Path("/data/dsa110-contimg/state/catalogs")

# Beam solid angle in pixels (for aperture photometry)
# Synthesized beam: ~60"x34", pixel=6" → beam area ~ (60*34*pi/(4*ln2)) / 36 px
BEAM_NPIX = int(np.pi * 60 * 34 / (4 * np.log(2) * 36))  # ~37 pixels


def load_mosaic() -> tuple[np.ndarray, WCS, fits.Header]:
    with fits.open(MOSAIC) as hdul:
        data = hdul[0].data.squeeze().astype(np.float64)
        hdr = hdul[0].header
        wcs = WCS(hdr).celestial
    return data, wcs, hdr


def load_rms_map() -> np.ndarray | None:
    """Load BANE RMS map if available, else return None (use global RMS)."""
    if os.path.exists(MOSAIC_RMS):
        with fits.open(MOSAIC_RMS) as hdul:
            return hdul[0].data.squeeze().astype(np.float64)
    return None


def get_mosaic_footprint(data: np.ndarray, wcs: WCS) -> tuple[SkyCoord, float]:
    """Return sky center and radius (deg) of the mosaic valid region."""
    ny, nx = data.shape
    valid = np.isfinite(data)
    if not valid.any():
        raise RuntimeError("Mosaic has no valid pixels")

    # Find bounding box of valid region
    rows = np.any(valid, axis=1)
    cols = np.any(valid, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    center_sky = wcs.pixel_to_world((cmin + cmax) / 2, (rmin + rmax) / 2)
    corner = wcs.pixel_to_world(cmin, rmin)
    radius_deg = center_sky.separation(corner).deg
    log.info("Mosaic footprint: center=(%.3f, %.3f) deg, radius=%.2f deg",
             center_sky.ra.deg, center_sky.dec.deg, radius_deg)
    return center_sky, radius_deg


def query_nvss(center: SkyCoord, radius_deg: float) -> "pd.DataFrame":
    """Query NVSS from local SQLite catalog for all sources within radius."""
    import pandas as pd
    from dsa110_continuum.catalog.query import cone_search
    log.info("Querying NVSS (local SQLite) within %.2f deg of (%.3f, %.3f) ...",
             radius_deg, center.ra.deg, center.dec.deg)
    df = cone_search(
        "nvss",
        ra_center=center.ra.deg,
        dec_center=center.dec.deg,
        radius_deg=radius_deg,
        min_flux_mjy=NVSS_MIN_FLUX_JY * 1000.0,  # convert Jy to mJy
    )
    if df is None or len(df) == 0:
        log.warning("No NVSS sources returned")
        return None
    log.info("NVSS query returned %d sources (>= %.0f mJy)", len(df), NVSS_MIN_FLUX_JY * 1000.0)
    return df


def read_aegean_catalog() -> SkyCoord | None:
    """Read Aegean FITS catalog. Returns SkyCoord or None if empty/missing."""
    from astropy.table import Table
    if not os.path.exists(CATALOG):
        log.info("No Aegean catalog at %s — will do blind forced photometry", CATALOG)
        return None, None
    t = Table.read(CATALOG)
    if len(t) == 0:
        log.warning("Aegean catalog is empty")
        return None, None
    cat_coords = SkyCoord(ra=t["ra_deg"] * u.deg, dec=t["dec_deg"] * u.deg)
    return cat_coords, t


def measure_forced_flux(
    data: np.ndarray,
    wcs: WCS,
    rms_map: np.ndarray | None,
    ra_deg: float,
    dec_deg: float,
    global_rms: float,
) -> tuple[float, float]:
    """
    Measure peak flux at (ra_deg, dec_deg) via nearest pixel lookup.
    Returns (flux_jy, noise_jy).
    """
    sky = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    x, y = wcs.world_to_pixel(sky)
    xi, yi = int(round(float(x))), int(round(float(y)))

    ny, nx = data.shape
    if not (0 <= xi < nx and 0 <= yi < ny):
        return np.nan, np.nan

    # Measure peak in a box (±5 pixels = ±30") centred on the NVSS position
    # Matches the 30" cross-match radius so displaced pairs still capture the peak
    r = 5
    box = data[
        max(0, yi - r): min(ny, yi + r + 1),
        max(0, xi - r): min(nx, xi + r + 1),
    ]
    valid = box[np.isfinite(box)]
    if len(valid) == 0:
        return np.nan, np.nan

    flux = float(np.nanmax(valid))  # peak flux in Jy/beam

    # Local RMS
    if rms_map is not None:
        rms_box = rms_map[
            max(0, yi - r): min(ny, yi + r + 1),
            max(0, xi - r): min(nx, xi + r + 1),
        ]
        rms_valid = rms_box[np.isfinite(rms_box)]
        noise = float(np.nanmedian(rms_valid)) if len(rms_valid) > 0 else global_rms
    else:
        noise = global_rms

    return flux, noise


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    # ── Load mosaic ────────────────────────────────────────────────────────────
    if not os.path.exists(MOSAIC):
        log.error("Mosaic not found: %s", MOSAIC)
        sys.exit(1)

    log.info("Loading mosaic ...")
    data, wcs, hdr = load_mosaic()
    rms_map = load_rms_map()

    # Global robust RMS
    finite_vals = data[np.isfinite(data)]
    global_rms = 1.4826 * float(np.median(np.abs(finite_vals - np.median(finite_vals))))
    log.info("Global MAD-RMS: %.4f Jy/beam", global_rms)

    # ── Mosaic footprint ───────────────────────────────────────────────────────
    center, radius_deg = get_mosaic_footprint(data, wcs)

    # ── Query NVSS ─────────────────────────────────────────────────────────────
    nvss_tbl = query_nvss(center, radius_deg)
    if nvss_tbl is None or len(nvss_tbl) == 0:
        log.error("No NVSS sources to cross-match")
        sys.exit(1)

    # nvss_tbl is now a pandas DataFrame with columns: ra_deg, dec_deg, flux_mjy
    # flux_mjy is already filtered by NVSS_MIN_FLUX_JY (cone_search applied it)
    nvss_coords = SkyCoord(
        ra=nvss_tbl["ra_deg"].values * u.deg,
        dec=nvss_tbl["dec_deg"].values * u.deg,
    )

    # ── Load Aegean catalog ────────────────────────────────────────────────────
    cat_coords, aegean_tbl = read_aegean_catalog()

    # ── Forced photometry at NVSS positions ────────────────────────────────────
    # If Aegean catalog exists, pre-filter to only NVSS sources with a nearby detection.
    # If not, do blind forced photometry at all NVSS positions.
    rows = []

    if cat_coords is not None:
        idx_nvss, sep, _ = nvss_coords.match_to_catalog_sky(cat_coords)
        matched_mask = sep.arcsec < NVSS_MATCH_RADIUS_ARCSEC
        log.info("NVSS sources: %d total, %d matched within %.0f\"",
                 len(nvss_tbl), matched_mask.sum(), NVSS_MATCH_RADIUS_ARCSEC)

        for i, (nvss_row, matched, sep_val) in enumerate(
            zip(nvss_tbl.itertuples(), matched_mask, sep.arcsec)
        ):
            if not matched:
                continue

            ra_deg = float(nvss_row.ra_deg)
            dec_deg = float(nvss_row.dec_deg)
            nvss_flux = float(nvss_row.flux_mjy) / 1000.0  # mJy → Jy

            if nvss_flux < NVSS_MIN_FLUX_JY:
                continue

            meas_flux, noise = measure_forced_flux(
                data, wcs, rms_map, ra_deg, dec_deg, global_rms
            )

            if not np.isfinite(meas_flux) or not np.isfinite(noise):
                continue

            snr = meas_flux / noise if noise > 0 else np.nan
            if not np.isfinite(snr) or snr < 3.0:
                continue

            ratio = meas_flux / nvss_flux if nvss_flux > 0 else np.nan
            source_name = f"NVSS J{ra_deg:.4f}{dec_deg:+.4f}"

            rows.append({
                "source_name": source_name,
                "ra_deg": round(ra_deg, 5),
                "dec_deg": round(dec_deg, 5),
                "nvss_flux_jy": round(nvss_flux, 5),
                "measured_flux_jy": round(meas_flux, 5),
                "flux_ratio": round(ratio, 4) if np.isfinite(ratio) else "",
                "snr": round(snr, 2) if np.isfinite(snr) else "",
                "match_sep_arcsec": round(float(sep_val), 2),
            })
    else:
        # No Aegean detections — do forced photometry at all NVSS positions in footprint
        log.warning("No Aegean catalog — doing blind forced photometry at all NVSS positions")
        for nvss_row in nvss_tbl.itertuples():
            ra_deg = float(nvss_row.ra_deg)
            dec_deg = float(nvss_row.dec_deg)
            nvss_flux = float(nvss_row.flux_mjy) / 1000.0  # mJy → Jy

            if nvss_flux < NVSS_MIN_FLUX_JY:
                continue

            meas_flux, noise = measure_forced_flux(
                data, wcs, rms_map, ra_deg, dec_deg, global_rms
            )
            if not np.isfinite(meas_flux) or not np.isfinite(noise):
                continue

            snr = meas_flux / noise if noise > 0 else np.nan
            if not np.isfinite(snr) or snr < 3.0:
                continue

            ratio = meas_flux / nvss_flux if nvss_flux > 0 else np.nan
            source_name = f"NVSS J{ra_deg:.4f}{dec_deg:+.4f}"

            rows.append({
                "source_name": source_name,
                "ra_deg": round(ra_deg, 5),
                "dec_deg": round(dec_deg, 5),
                "nvss_flux_jy": round(nvss_flux, 5),
                "measured_flux_jy": round(meas_flux, 5),
                "flux_ratio": round(ratio, 4) if np.isfinite(ratio) else "",
                "snr": round(snr, 2) if np.isfinite(snr) else "",
                "match_sep_arcsec": 0.0,
            })

    log.info("Writing %d forced photometry rows to %s", len(rows), OUT_CSV)

    if not rows:
        log.error("No rows to write — forced photometry failed")
        sys.exit(1)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source_name", "ra_deg", "dec_deg",
            "nvss_flux_jy", "measured_flux_jy",
            "flux_ratio", "snr", "match_sep_arcsec",
        ])
        writer.writeheader()
        writer.writerows(rows)

    # ── QA report ─────────────────────────────────────────────────────────────
    valid_ratios = [
        float(r["flux_ratio"]) for r in rows
        if r["flux_ratio"] != "" and np.isfinite(float(r["flux_ratio"]))
    ]
    log.info("\n=== Forced Photometry QA ===")
    log.info("Total rows: %d", len(rows))
    if valid_ratios:
        log.info("Flux ratio (DSA/NVSS): median=%.3f, std=%.3f",
                 np.median(valid_ratios), np.std(valid_ratios))
        log.info("Ratio range: %.3f – %.3f", min(valid_ratios), max(valid_ratios))
        outliers = sum(r < 0.5 or r > 2.0 for r in valid_ratios)
        log.info("Outliers (ratio outside 0.5–2.0): %d/%d",
                 outliers, len(valid_ratios))

        # Success criteria
        median_ratio = np.median(valid_ratios)
        criteria_passed = (
            len(rows) >= 10
            and 0.5 <= median_ratio <= 2.0
        )
    else:
        criteria_passed = len(rows) >= 10

    if criteria_passed:
        print(f"\nSUCCESS: {len(rows)} sources with median flux ratio {np.median(valid_ratios):.3f}")
        print(f"CSV: {OUT_CSV}")
    else:
        if len(rows) < 10:
            print(f"\nFAIL: Only {len(rows)} sources (need ≥ 10)")
        else:
            print(f"\nWARNING: Median flux ratio {np.median(valid_ratios):.3f} outside expected range")
        sys.exit(1)


if __name__ == "__main__":
    main()
