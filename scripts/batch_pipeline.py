#!/opt/miniforge/envs/casa6/bin/python
"""
Batch pipeline: calibrate → image → mosaic → forced photometry for a full day.

Usage:
    python scripts/batch_pipeline.py [--date DATE] [--keep-intermediates] [--skip-photometry]

Steps:
    1. Find all valid MS files for DATE (chronological order)
    2. For each MS: phaseshift → applycal → WSClean image (skip if tile FITS exists)
    3. Build mosaic from all tiles + QA check
    4. Run forced photometry against NVSS catalog on the mosaic
    5. Write CSV of results to products/mosaics/{DATE}/photometry_nvss.csv
    6. Print summary: tiles processed/skipped/failed, mosaic peak/RMS/DR, source count, median flux ratio
"""
import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

# ── Load scripts/.env before any other imports that might need env vars ──────
_ENV_FILE = Path(__file__).parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if _line.startswith("export "):
            _line = _line[len("export "):]
        if "=" in _line and not _line.startswith("#"):
            _key, _, _val = _line.partition("=")
            os.environ.setdefault(_key.strip(), _val.strip())

# ── Project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))  # so 'import mosaic_day' works

import numpy as np
from astropy.io import fits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_DATE = "2026-01-25"
MS_DIR = "/stage/dsa110-contimg/ms"
STAGE_IMAGE_BASE = "/stage/dsa110-contimg/images"
PRODUCTS_BASE = "/data/dsa110-continuum/products/mosaics"


def get_paths(date: str) -> dict:
    return {
        "ms_dir": MS_DIR,
        "image_dir": f"{STAGE_IMAGE_BASE}/mosaic_{date}",
        "mosaic_out": f"{STAGE_IMAGE_BASE}/mosaic_{date}/full_mosaic.fits",
        "products_dir": f"{PRODUCTS_BASE}/{date}",
        "photometry_csv": f"{PRODUCTS_BASE}/{date}/photometry_nvss.csv",
    }


# ── Tile processing (delegates to mosaic_day) ─────────────────────────────────

def run_tile_phase(ms_list: list[str], paths: dict, keep: bool, md) -> tuple[list[str], int, int, int]:
    """Process each MS through phaseshift→cal→image. Returns (tile_fits, n_done, n_skipped, n_failed)."""
    tile_images = []
    n_done = n_skipped = n_failed = 0

    for i, ms_path in enumerate(ms_list, 1):
        tag = Path(ms_path).stem
        log.info("[%d/%d] %s", i, len(ms_list), tag)
        t0 = time.time()

        result = md.process_ms(ms_path, keep_intermediates=keep)

        elapsed = time.time() - t0
        if result is None:
            log.error("  FAILED after %.0fs", elapsed)
            n_failed += 1
        elif elapsed < 2.0:
            # If it returned instantly the tile was already there (skipped)
            log.info("  Skipped (already processed) in %.1fs", elapsed)
            tile_images.append(result)
            n_skipped += 1
        else:
            log.info("  Done in %.0fs: %s", elapsed, Path(result).name)
            tile_images.append(result)
            n_done += 1

    return tile_images, n_done, n_skipped, n_failed


# ── Mosaic phase (delegates to mosaic_day) ────────────────────────────────────

def run_mosaic_phase(tile_images: list[str], paths: dict, md) -> str | None:
    """Build mosaic from tiles, run QA. Returns mosaic FITS path or None on failure."""
    mosaic_out = paths["mosaic_out"]
    os.makedirs(paths["image_dir"], exist_ok=True)

    if os.path.exists(mosaic_out):
        log.info("Mosaic already exists: %s", mosaic_out)
    else:
        if len(tile_images) < 2:
            log.error("Need at least 2 tile images to mosaic (have %d) — aborting", len(tile_images))
            return None
        log.info("=== Building mosaic from %d tiles ===", len(tile_images))
        out_wcs, ny, nx = md.build_common_wcs(tile_images)
        mosaic = md.coadd_tiles(tile_images, out_wcs, ny, nx)
        md.write_mosaic(mosaic, out_wcs, tile_images)

    log.info("=== Mosaic QA ===")
    md.check_mosaic_quality(mosaic_out)

    with fits.open(mosaic_out) as hdul:
        data = hdul[0].data.squeeze()
    finite = data[np.isfinite(data)]
    peak = float(np.nanmax(data))
    rms = float(1.4826 * np.nanmedian(np.abs(finite - np.nanmedian(finite))))
    dr = peak / rms if rms > 0 else float("nan")
    log.info("Mosaic peak: %.4f Jy/beam | RMS: %.4f Jy/beam | DR: %.0f", peak, rms, dr)

    return mosaic_out


# ── Forced photometry phase ───────────────────────────────────────────────────

def run_photometry_phase(mosaic_path: str, paths: dict) -> list[dict] | None:
    """Query NVSS sources within mosaic FoV and measure forced photometry.

    Returns list of result dicts (one per source) or None on failure.
    """
    try:
        from dsa110_continuum.photometry.helpers import query_sources_for_fits
        from dsa110_continuum.photometry.forced import measure_many
    except ImportError as e:
        log.error("Cannot import photometry modules: %s", e)
        return None

    log.info("=== Forced photometry against NVSS ===")

    # Query NVSS catalog for sources in the full mosaic FoV.
    # Use a generous radius so all sources across the drift strip are included.
    try:
        sources = query_sources_for_fits(
            fits_path=Path(mosaic_path),
            catalog="nvss",
            radius_deg=10.0,  # wide: mosaic covers ~hours of RA drift
            min_flux_mjy=10.0,  # only sources bright enough to detect reliably
        )
    except Exception as e:
        log.error("Catalog query failed: %s", e)
        return None

    if not sources:
        log.warning("No NVSS sources found in mosaic FoV")
        return []

    log.info("Found %d NVSS sources with flux > 10 mJy", len(sources))

    # Run forced photometry at each catalog position
    coords = [(s["ra"], s["dec"]) for s in sources]
    try:
        results = measure_many(mosaic_path, coords)
    except Exception as e:
        log.error("measure_many failed: %s", e)
        return None

    # Combine catalog and measurement data
    rows = []
    for src, meas in zip(sources, results):
        if meas is None:
            continue
        nvss_flux_jy = src.get("flux_mjy", 0.0) / 1000.0
        ratio = (meas.peak_jyb / nvss_flux_jy) if nvss_flux_jy > 0 else float("nan")
        rows.append({
            "ra_deg": round(src["ra"], 6),
            "dec_deg": round(src["dec"], 6),
            "nvss_flux_jy": round(nvss_flux_jy, 6),
            "dsa_peak_jyb": round(meas.peak_jyb, 6),
            "dsa_peak_err_jyb": round(meas.peak_err_jyb, 6),
            "dsa_nvss_ratio": round(ratio, 4),
        })

    return rows


def write_photometry_csv(rows: list[dict], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not rows:
        log.warning("No photometry rows to write")
        return
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info("Photometry results written: %s (%d sources)", csv_path, len(rows))


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(
    date: str,
    ms_total: int,
    n_done: int,
    n_skipped: int,
    n_failed: int,
    mosaic_path: str | None,
    phot_rows: list[dict] | None,
) -> None:
    print("\n" + "=" * 60)
    print(f"  DSA-110 Batch Pipeline Summary — {date}")
    print("=" * 60)
    print(f"  MS files found:    {ms_total}")
    print(f"  Tiles imaged:      {n_done}")
    print(f"  Tiles skipped:     {n_skipped}  (already done)")
    print(f"  Tiles failed:      {n_failed}")

    if mosaic_path and os.path.exists(mosaic_path):
        with fits.open(mosaic_path) as hdul:
            data = hdul[0].data.squeeze()
        finite = data[np.isfinite(data)]
        peak = float(np.nanmax(data))
        rms = float(1.4826 * np.nanmedian(np.abs(finite - np.nanmedian(finite))))
        dr = peak / rms if rms > 0 else float("nan")
        print(f"\n  Mosaic: {mosaic_path}")
        print(f"  Peak:   {peak:.4f} Jy/beam")
        print(f"  RMS:    {rms*1000:.2f} mJy/beam")
        print(f"  DR:     {dr:.0f}")

    if phot_rows is not None:
        n_src = len(phot_rows)
        if n_src > 0:
            ratios = [r["dsa_nvss_ratio"] for r in phot_rows if np.isfinite(r["dsa_nvss_ratio"])]
            median_ratio = float(np.median(ratios)) if ratios else float("nan")
            print(f"\n  NVSS sources measured: {n_src}")
            print(f"  Median DSA/NVSS ratio: {median_ratio:.3f}  (target: 0.8–1.2)")
            if median_ratio < 0.5:
                print("  WARNING: ratio < 0.5 — check primary beam correction")
            elif 0.8 <= median_ratio <= 1.2:
                print("  OK: flux scale looks good")
        else:
            print("\n  No NVSS sources measured (FoV too small or catalog miss)")
    print("=" * 60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full DSA-110 pipeline: cal→image→mosaic→photometry for a day of drift data."
    )
    parser.add_argument("--date", default=DEFAULT_DATE, help="Observation date (YYYY-MM-DD)")
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        default=False,
        help="Keep *_meridian.ms and skip moving mosaic to products/ (debug mode).",
    )
    parser.add_argument(
        "--skip-photometry",
        action="store_true",
        default=False,
        help="Skip forced photometry step.",
    )
    args = parser.parse_args()

    date = args.date
    keep = args.keep_intermediates
    paths = get_paths(date)

    log.info("=== DSA-110 Batch Pipeline — %s ===", date)
    log.info("Stage image dir: %s", paths["image_dir"])
    log.info("Products dir:    %s", paths["products_dir"])

    # Patch mosaic_day constants to use this date's paths before importing its functions
    # (mosaic_day uses module-level constants; we override IMAGE_DIR and MOSAIC_OUT via env)
    os.makedirs(paths["image_dir"], exist_ok=True)

    # ── Phase 1: Find valid MS files ──────────────────────────────────────────
    # Import after sys.path is set (scripts/ dir is on sys.path)
    import mosaic_day as _md  # noqa: E402

    # Override mosaic_day constants for this date's run
    _md.DATE = date
    _md.IMAGE_DIR = paths["image_dir"]
    _md.MOSAIC_OUT = paths["mosaic_out"]
    _md.PRODUCTS_DIR = paths["products_dir"]
    _md.BP_TABLE = f"{MS_DIR}/{date}T22:26:05_0~23.b"
    _md.G_TABLE = f"{MS_DIR}/{date}T22:26:05_0~23.g"

    ms_list = _md.find_valid_ms()
    if not ms_list:
        log.error("No valid MS files found for %s — aborting", date)
        sys.exit(1)
    log.info("Found %d valid MS files", len(ms_list))

    # ── Phase 2: Tile imaging ─────────────────────────────────────────────────
    log.info("=== Phase 1/3: Calibrate + Image tiles ===")
    tile_images, n_done, n_skipped, n_failed = run_tile_phase(ms_list, paths, keep, _md)
    log.info(
        "Tiles: %d imaged, %d skipped, %d failed (total valid: %d)",
        n_done, n_skipped, n_failed, len(tile_images),
    )

    # ── Phase 3: Mosaic ───────────────────────────────────────────────────────
    log.info("=== Phase 2/3: Mosaic ===")
    mosaic_path = run_mosaic_phase(tile_images, paths, _md)
    if mosaic_path is None:
        log.error("Mosaic failed — skipping photometry")
        print_summary(date, len(ms_list), n_done, n_skipped, n_failed, None, None)
        sys.exit(1)

    # ── Phase 4: Forced photometry ────────────────────────────────────────────
    phot_rows: list[dict] | None = None
    if not args.skip_photometry:
        log.info("=== Phase 3/3: Forced photometry ===")
        phot_rows = run_photometry_phase(mosaic_path, paths)
        if phot_rows:
            write_photometry_csv(phot_rows, paths["photometry_csv"])
    else:
        log.info("Skipping forced photometry (--skip-photometry)")

    # ── Move mosaic to products ───────────────────────────────────────────────
    if not keep:
        _md._move_mosaic_to_products()  # uses _md.IMAGE_DIR / _md.PRODUCTS_DIR patched above

    # ── Summary ───────────────────────────────────────────────────────────────
    # After move, mosaic lives in products dir
    final_mosaic = (
        paths["photometry_csv"].replace("photometry_nvss.csv", "full_mosaic.fits")
        if not keep else mosaic_path
    )
    if not os.path.exists(final_mosaic):
        final_mosaic = mosaic_path  # fallback if move didn't happen
    print_summary(date, len(ms_list), n_done, n_skipped, n_failed, final_mosaic, phot_rows)


if __name__ == "__main__":
    main()
