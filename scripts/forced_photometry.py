#!/opt/miniforge/envs/casa6/bin/python
"""
Forced photometry on DSA-110 mosaics using the unified master catalog.

Queries the master catalog (crossmatched NVSS + VLASS + FIRST + RACS) for
source positions, optionally filters out resolved/confused sources, then
measures peak flux at each position using weighted Condon convolution.

Usage:
    python scripts/forced_photometry.py --mosaic /path/to/mosaic.fits
    python scripts/forced_photometry.py --mosaic /path/to/mosaic.fits --catalog nvss
    python scripts/forced_photometry.py --mosaic /path/to/mosaic.fits --min-flux-mjy 20
"""
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, "/data/dsa110-continuum/src") if Path("/data/dsa110-continuum/src").exists() else None

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from dsa110_continuum.catalog.query import cone_search
from dsa110_continuum.photometry.forced import measure_many

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_MOSAIC = "/stage/dsa110-contimg/images/mosaic_2026-01-25/full_mosaic.fits"


def get_mosaic_footprint(
    data: np.ndarray, wcs: WCS
) -> tuple[float, float, float]:
    """Return (ra_center, dec_center, radius_deg) of mosaic valid region."""
    valid = np.isfinite(data)
    if not valid.any():
        raise RuntimeError("Mosaic has no valid pixels")

    rows = np.any(valid, axis=1)
    cols = np.any(valid, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    center_sky = wcs.pixel_to_world((cmin + cmax) / 2, (rmin + rmax) / 2)
    corner = wcs.pixel_to_world(cmin, rmin)
    radius_deg = center_sky.separation(corner).deg
    ra = float(center_sky.ra.deg)
    dec = float(center_sky.dec.deg)
    log.info(
        "Mosaic footprint: center=(%.3f, %.3f) deg, radius=%.2f deg",
        ra, dec, radius_deg,
    )
    return ra, dec, float(radius_deg)


def run_forced_photometry(
    mosaic_path: str,
    output_csv: str | None = None,
    catalog: str = "master",
    min_flux_mjy: float = 50.0,
    exclude_resolved: bool = True,
    exclude_confused: bool = True,
    snr_cut: float = 3.0,
) -> dict:
    """Run forced photometry on a mosaic and write results to CSV.

    Parameters
    ----------
    mosaic_path : str
        Path to the mosaic FITS file.
    output_csv : str, optional
        Output CSV path.  Defaults to ``{mosaic_stem}_forced_phot.csv``.
    catalog : str
        Catalog to query (``"master"``, ``"nvss"``, etc.).
    min_flux_mjy : float
        Minimum catalog flux in mJy.
    exclude_resolved, exclude_confused : bool
        Filter flags (master catalog only).
    snr_cut : float
        Minimum SNR for output rows.

    Returns
    -------
    dict
        ``{"n_sources": int, "median_ratio": float, "csv_path": str}``
    """
    if not Path(mosaic_path).exists():
        raise FileNotFoundError(f"Mosaic not found: {mosaic_path}")

    stem = mosaic_path.replace(".fits", "")
    out_csv = output_csv or f"{stem}_forced_phot.csv"
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    # ── Load mosaic ────────────────────────────────────────────────────────────
    log.info("Loading mosaic: %s", mosaic_path)
    with fits.open(mosaic_path) as hdul:
        data = hdul[0].data.squeeze().astype(np.float64)
        wcs = WCS(hdul[0].header).celestial

    # ── Query catalog ──────────────────────────────────────────────────────────
    ra_cen, dec_cen, radius = get_mosaic_footprint(data, wcs)
    log.info("Querying %s catalog (min_flux=%.0f mJy) ...", catalog, min_flux_mjy)

    df = cone_search(
        catalog,
        ra_center=ra_cen,
        dec_center=dec_cen,
        radius_deg=radius,
        min_flux_mjy=min_flux_mjy,
    )
    if df is None or len(df) == 0:
        raise RuntimeError("No catalog sources returned")
    log.info("Catalog returned %d sources", len(df))

    # Filter resolved and confused sources (master catalog only)
    if catalog == "master":
        n_before = len(df)
        if exclude_resolved and "resolved_flag" in df.columns:
            df = df[df["resolved_flag"] == 0]
        if exclude_confused and "confusion_flag" in df.columns:
            df = df[df["confusion_flag"] == 0]
        n_filtered = n_before - len(df)
        if n_filtered > 0:
            log.info(
                "Filtered %d resolved/confused sources → %d remaining",
                n_filtered, len(df),
            )

    if len(df) == 0:
        raise RuntimeError("No sources remaining after filtering")

    # ── Measure forced photometry ──────────────────────────────────────────────
    coords = list(zip(df["ra_deg"].values, df["dec_deg"].values))
    log.info("Measuring forced photometry at %d positions ...", len(coords))

    results = measure_many(mosaic_path, coords)

    # ── Build output rows ──────────────────────────────────────────────────────
    rows = []
    for i, res in enumerate(results):
        if not np.isfinite(res.peak_jyb) or not np.isfinite(res.peak_err_jyb):
            continue
        if res.peak_err_jyb <= 0:
            continue
        snr = res.peak_jyb / res.peak_err_jyb
        if snr < snr_cut:
            continue

        cat_flux_mjy = float(df.iloc[i]["flux_mjy"])
        cat_flux_jy = cat_flux_mjy / 1000.0
        ratio = res.peak_jyb / cat_flux_jy if cat_flux_jy > 0 else np.nan

        row = {
            "source_name": f"J{res.ra_deg:.4f}{res.dec_deg:+.4f}",
            "ra_deg": round(res.ra_deg, 5),
            "dec_deg": round(res.dec_deg, 5),
            "catalog_flux_jy": round(cat_flux_jy, 5),
            "measured_flux_jy": round(res.peak_jyb, 5),
            "flux_err_jy": round(res.peak_err_jyb, 5),
            "flux_ratio": round(ratio, 4) if np.isfinite(ratio) else "",
            "snr": round(snr, 2),
        }

        # Add master catalog metadata if available
        if catalog == "master":
            row_data = df.iloc[i]
            if "alpha" in df.columns and not np.isnan(row_data.get("alpha", np.nan)):
                row["spectral_index"] = round(float(row_data["alpha"]), 3)
            else:
                row["spectral_index"] = ""

        rows.append(row)

    log.info(
        "Photometry complete: %d/%d sources with SNR >= %.1f",
        len(rows), len(results), snr_cut,
    )

    if not rows:
        raise RuntimeError("No rows to write — forced photometry failed")

    # ── Write CSV ──────────────────────────────────────────────────────────────
    fieldnames = [
        "source_name", "ra_deg", "dec_deg",
        "catalog_flux_jy", "measured_flux_jy", "flux_err_jy",
        "flux_ratio", "snr",
    ]
    if catalog == "master":
        fieldnames.append("spectral_index")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── QA summary ─────────────────────────────────────────────────────────────
    valid_ratios = [
        float(r["flux_ratio"]) for r in rows
        if r["flux_ratio"] != "" and np.isfinite(float(r["flux_ratio"]))
    ]
    median_ratio = float(np.median(valid_ratios)) if valid_ratios else float("nan")

    log.info("\n=== Forced Photometry QA ===")
    log.info("Catalog: %s | Sources measured: %d", catalog, len(rows))
    if valid_ratios:
        log.info("Flux ratio (DSA/catalog): median=%.3f, std=%.3f", median_ratio, np.std(valid_ratios))
        log.info("Ratio range: %.3f – %.3f", min(valid_ratios), max(valid_ratios))
        outliers = sum(r < 0.5 or r > 2.0 for r in valid_ratios)
        log.info("Outliers (ratio outside 0.5–2.0): %d/%d", outliers, len(valid_ratios))

    return {"n_sources": len(rows), "median_ratio": median_ratio, "csv_path": out_csv}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Forced photometry on a DSA-110 mosaic."
    )
    parser.add_argument(
        "--mosaic", default=None, metavar="PATH",
        help="Path to mosaic FITS file (default: Jan 25 mosaic)",
    )
    parser.add_argument(
        "--catalog", default="master", choices=["master", "nvss", "first", "rax", "vlass"],
        help="Catalog to query for source positions (default: master)",
    )
    parser.add_argument(
        "--min-flux-mjy", type=float, default=50.0,
        help="Minimum catalog flux in mJy (default: 50)",
    )
    parser.add_argument(
        "--exclude-resolved", action="store_true", default=True,
        help="Exclude resolved sources from master catalog (default: True)",
    )
    parser.add_argument(
        "--no-exclude-resolved", action="store_false", dest="exclude_resolved",
        help="Include resolved sources",
    )
    parser.add_argument(
        "--exclude-confused", action="store_true", default=True,
        help="Exclude confused sources from master catalog (default: True)",
    )
    parser.add_argument(
        "--no-exclude-confused", action="store_false", dest="exclude_confused",
        help="Include confused sources",
    )
    parser.add_argument(
        "--snr-cut", type=float, default=3.0,
        help="Minimum SNR for output (default: 3.0)",
    )
    parser.add_argument(
        "--output", default=None, metavar="PATH",
        help="Output CSV path (default: derived from mosaic path)",
    )
    args = parser.parse_args()

    mosaic_path = args.mosaic or DEFAULT_MOSAIC
    try:
        result = run_forced_photometry(
            mosaic_path,
            output_csv=args.output,
            catalog=args.catalog,
            min_flux_mjy=args.min_flux_mjy,
            exclude_resolved=args.exclude_resolved,
            exclude_confused=args.exclude_confused,
            snr_cut=args.snr_cut,
        )
    except (FileNotFoundError, RuntimeError) as e:
        log.error("%s", e)
        sys.exit(1)

    med = result["median_ratio"]
    n = result["n_sources"]
    passed = n >= 10 and 0.5 <= med <= 2.0

    if passed:
        print(f"\nSUCCESS: {n} sources, median flux ratio {med:.3f}")
        print(f"CSV: {result['csv_path']}")
    else:
        if n < 10:
            print(f"\nFAIL: Only {n} sources (need >= 10)")
        else:
            print(f"\nWARNING: Median flux ratio {med:.3f} outside expected range")
        sys.exit(1)


if __name__ == "__main__":
    main()
