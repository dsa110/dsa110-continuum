#!/opt/miniforge/envs/casa6/bin/python
"""
Process a full day of DSA-110 drift observations and mosaic them.

Steps per MS:
  1. Phaseshift to median meridian (skip if *_meridian.ms already exists)
  2. Apply BP + G calibration (skip if CORRECTED_DATA ratio > 5)
  3. Image with WSClean

Then:
  4. Reproject all tile images onto a common WCS
  5. Coadd with noise (1/σ²) weighting
  6. Write final mosaic FITS
"""
import argparse
import glob
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from casacore.tables import table

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Pipeline modules ────────────────────────────────────────────────────────
from dsa110_continuum.calibration.applycal import apply_to_target
from dsa110_continuum.calibration.runner import phaseshift_ms
from dsa110_continuum.imaging.cli_imaging import image_ms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
DATE = "2026-01-25"

MS_DIR = "/stage/dsa110-contimg/ms"
IMAGE_DIR = f"/stage/dsa110-contimg/images/mosaic_{DATE}"
MOSAIC_OUT = f"{IMAGE_DIR}/full_mosaic.fits"
PRODUCTS_DIR = f"/data/dsa110-continuum/products/mosaics/{DATE}"

BP_TABLE = f"{MS_DIR}/{DATE}T22:26:05_0~23.b"
G_TABLE = f"{MS_DIR}/{DATE}T22:26:05_0~23.g"

# Imaging parameters — same as run_pipeline.py
IMSIZE = 2400
CELL_ARCSEC = 6.0
WEIGHTING = "briggs"
ROBUST = 0.5
NITER = 1000
THRESHOLD = "0.005Jy"

os.makedirs(IMAGE_DIR, exist_ok=True)

# Primary beam cutoff: blank tile pixels where the WSClean beam model is below
# this fraction of the peak response.  Values < PB_CUTOFF have high noise
# amplification in pb-corrected images and cause severe edge artefacts in the mosaic.
PB_CUTOFF = 0.2  # 20 % of peak response


# ── Helpers ──────────────────────────────────────────────────────────────────

def find_valid_ms() -> list[str]:
    """Return sorted list of valid (non-corrupt) raw MS paths for 2026-01-25."""
    candidates = sorted(glob.glob(f"{MS_DIR}/2026-01-25T*.ms"))
    candidates = [p for p in candidates if "meridian" not in p and "flagversion" not in p]
    valid = []
    for path in candidates:
        try:
            with table(path + "/FIELD", readonly=True, ack=False) as _:
                pass
            valid.append(path)
        except Exception:
            log.warning("Skipping corrupt MS (no FIELD table): %s", path)
    log.info("Found %d valid MS files", len(valid))
    return valid


def get_meridian_path(ms_path: str) -> str:
    return ms_path.replace(".ms", "_meridian.ms")


def needs_calibration(ms_path: str) -> bool:
    """Return True if CORRECTED_DATA doesn't exist or has ratio close to 1."""
    try:
        with table(ms_path, readonly=True, ack=False) as t:
            if "CORRECTED_DATA" not in t.colnames():
                return True
            raw = t.getcol("DATA", nrow=1000)
            corr = t.getcol("CORRECTED_DATA", nrow=1000)
            flag = t.getcol("FLAG", nrow=1000)
            good = ~flag
            if not good.any():
                return True
            ratio = np.mean(np.abs(corr[good])) / np.mean(np.abs(raw[good]))
            return ratio < 5.0
    except Exception:
        return True


def process_ms(ms_path: str, keep_intermediates: bool = False) -> str | None:
    """Phaseshift → applycal → image one MS. Returns path to pb-corrected FITS or None."""
    tag = Path(ms_path).stem  # e.g. 2026-01-25T21:17:33
    meridian_ms = get_meridian_path(ms_path)
    imagename = os.path.join(IMAGE_DIR, tag)
    # With pbcor=True, WSClean produces {imagename}-image-pb.fits (primary-beam corrected)
    pbcor_fits = imagename + "-image-pb.fits"
    image_fits = imagename + "-image.fits"

    # Skip if already fully processed (check pbcor output first)
    if os.path.exists(pbcor_fits):
        log.info("[%s] PB-corrected image already exists — skipping", tag)
        return pbcor_fits
    if os.path.exists(image_fits):
        log.info("[%s] Image already exists (no pbcor) — skipping", tag)
        return image_fits

    # ── Step 1: Phaseshift ────────────────────────────────────────────────
    if os.path.exists(meridian_ms):
        log.info("[%s] Meridian MS already exists", tag)
    else:
        log.info("[%s] Phaseshifting to median meridian ...", tag)
        try:
            phaseshift_ms(
                ms_path=ms_path,
                mode="median_meridian",
                output_ms=meridian_ms,
            )
        except Exception as e:
            log.error("[%s] Phaseshift failed: %s", tag, e)
            return None

    # ── Step 2: Calibration ────────────────────────────────────────────────
    if needs_calibration(meridian_ms):
        log.info("[%s] Applying calibration ...", tag)
        try:
            apply_to_target(
                ms_target=meridian_ms,
                field="",
                gaintables=[BP_TABLE, G_TABLE],
                interp=["linear", "linear"],
            )
        except Exception as e:
            log.error("[%s] Applycal failed: %s", tag, e)
            return None
    else:
        log.info("[%s] Calibration already applied", tag)

    # ── Step 3: Image ──────────────────────────────────────────────────────
    log.info("[%s] Imaging with WSClean ...", tag)
    try:
        image_ms(
            ms_path=meridian_ms,
            imagename=imagename,
            imsize=IMSIZE,
            cell_arcsec=CELL_ARCSEC,
            weighting=WEIGHTING,
            robust=ROBUST,
            niter=NITER,
            threshold=THRESHOLD,
            pbcor=True,
            gridder="wgridder",
            backend="wsclean",
            use_unicat_mask=False,
        )
    except Exception as e:
        log.error("[%s] Imaging failed: %s", tag, e)
        return None

    result_fits = None
    if os.path.exists(pbcor_fits):
        log.info("[%s] PB-corrected image done: %s", tag, pbcor_fits)
        result_fits = pbcor_fits
    elif os.path.exists(image_fits):
        log.warning("[%s] -image-pb.fits not found; falling back to plain image: %s", tag, image_fits)
        result_fits = image_fits
    else:
        log.error("[%s] WSClean finished but no image FITS found", tag)
        return None

    # ── Cleanup: delete meridian MS now that imaging succeeded ────────────────
    if not keep_intermediates and os.path.isdir(meridian_ms):
        try:
            shutil.rmtree(meridian_ms)
            log.info("[%s] Deleted intermediate meridian MS: %s", tag, meridian_ms)
        except Exception as e:
            log.warning("[%s] Could not delete meridian MS %s: %s", tag, meridian_ms, e)

    return result_fits


# ── Mosaicking ────────────────────────────────────────────────────────────────

def build_common_wcs(fits_paths: list[str], margin_deg: float = 0.5) -> tuple[WCS, int, int]:
    """Compute a common RA/Dec WCS that covers all input FITS images."""
    ra_min, ra_max, dec_min, dec_max = 360.0, 0.0, 90.0, -90.0

    for path in fits_paths:
        with fits.open(path) as hdul:
            hdr = hdul[0].header
            wcs = WCS(hdr).celestial
            ny, nx = hdul[0].data.squeeze().shape
            # Sample corners
            corners = wcs.pixel_to_world(
                [0, nx - 1, 0, nx - 1], [0, 0, ny - 1, ny - 1]
            )
            ras = [c.ra.deg for c in corners]
            decs = [c.dec.deg for c in corners]
            ra_min = min(ra_min, min(ras))
            ra_max = max(ra_max, max(ras))
            dec_min = min(dec_min, min(decs))
            dec_max = max(dec_max, max(decs))

    ra_min -= margin_deg
    ra_max += margin_deg
    dec_min -= margin_deg
    dec_max += margin_deg

    pixel_scale_deg = CELL_ARCSEC / 3600.0
    ra_center = (ra_min + ra_max) / 2.0
    dec_center = (dec_min + dec_max) / 2.0

    nx = int(np.ceil((ra_max - ra_min) / pixel_scale_deg))
    ny = int(np.ceil((dec_max - dec_min) / pixel_scale_deg))
    # Round to even for WSClean compatibility
    nx = nx + (nx % 2)
    ny = ny + (ny % 2)

    out_wcs = WCS(naxis=2)
    out_wcs.wcs.crpix = [nx / 2 + 0.5, ny / 2 + 0.5]
    out_wcs.wcs.cdelt = [-pixel_scale_deg, pixel_scale_deg]
    out_wcs.wcs.crval = [ra_center, dec_center]
    out_wcs.wcs.ctype = ["RA---SIN", "DEC--SIN"]

    log.info(
        "Common WCS: RA %.2f–%.2f deg, Dec %.2f–%.2f deg, %d×%d px",
        ra_min, ra_max, dec_min, dec_max, nx, ny,
    )
    return out_wcs, ny, nx


def coadd_tiles(fits_paths: list[str], out_wcs: WCS, ny: int, nx: int) -> np.ndarray:
    """Reproject each tile onto the common WCS and do noise-weighted coaddition."""
    from reproject import reproject_interp

    sum_image = np.zeros((ny, nx), dtype=np.float64)
    sum_weight = np.zeros((ny, nx), dtype=np.float64)
    out_header = out_wcs.to_header()
    out_header["NAXIS1"] = nx
    out_header["NAXIS2"] = ny

    for path in fits_paths:
        log.info("Reprojecting %s ...", Path(path).name)
        with fits.open(path) as hdul:
            data = hdul[0].data.squeeze().astype(np.float64)
            hdr = hdul[0].header

        # ── Primary beam cutoff ────────────────────────────────────────────
        # WSClean writes a companion *-pb.fits beam model alongside the
        # pb-corrected image.  Pixels where beam < PB_CUTOFF have been
        # noise-amplified by 1/beam and cause severe edge artefacts in the
        # mosaic; blank them before combining.
        # WSClean names the per-channel beam model "{tag}-beam-0.fits" (not -pb.fits)
        pb_path = path.replace("-image-pb.fits", "-beam-0.fits")
        if not os.path.exists(pb_path):
            # Fallback: plain image tile has no -image-pb suffix
            pb_path = path.replace("-image.fits", "-beam-0.fits")
        if PB_CUTOFF > 0 and os.path.exists(pb_path):
            with fits.open(pb_path) as pb_hdul:
                pb_data = pb_hdul[0].data.squeeze().astype(np.float64)
            # Normalise to peak in case WSClean stores absolute sensitivity
            pb_peak = np.nanmax(pb_data)
            if pb_peak > 0:
                pb_data /= pb_peak
            low_beam = (pb_data < PB_CUTOFF) | ~np.isfinite(pb_data)
            data[low_beam] = np.nan
            n_blanked = low_beam.sum()
            log.info("  PB cutoff (%.0f%%): blanked %d pixels", PB_CUTOFF * 100, n_blanked)
        elif PB_CUTOFF > 0:
            log.warning("  No pb.fits found for %s — skipping beam cutoff", Path(path).name)

        # Estimate per-tile noise from the central region (away from bright sources)
        cy, cx = data.shape[0] // 2, data.shape[1] // 2
        margin = 200  # pixels from center
        inner = data[
            max(0, cy - margin): cy + margin,
            max(0, cx - margin): cx + margin,
        ]
        noise = np.nanstd(inner[np.isfinite(inner)])
        if noise <= 0 or not np.isfinite(noise):
            noise = 1.0  # fallback unit weight
        weight = 1.0 / (noise ** 2)

        # Reproject onto common grid
        try:
            reprojected, footprint = reproject_interp(
                (data, WCS(hdr).celestial),
                out_header,
                shape_out=(ny, nx),
            )
        except Exception as e:
            log.warning("Reproject failed for %s: %s — skipping", Path(path).name, e)
            continue

        valid = footprint.astype(bool) & np.isfinite(reprojected)
        sum_image[valid] += weight * reprojected[valid]
        sum_weight[valid] += weight

    mosaic = np.where(sum_weight > 0, sum_image / sum_weight, np.nan)
    return mosaic


def write_mosaic(mosaic: np.ndarray, out_wcs: WCS, fits_paths: list[str]) -> str:
    """Write mosaic to FITS using a representative header from the first tile."""
    os.makedirs(os.path.dirname(MOSAIC_OUT), exist_ok=True)
    with fits.open(fits_paths[0]) as ref:
        ref_hdr = ref[0].header.copy()

    # Build minimal 2D header from common WCS
    new_hdr = fits.Header()
    new_hdr["SIMPLE"] = True
    new_hdr["BITPIX"] = -32
    new_hdr["NAXIS"] = 2
    new_hdr["NAXIS1"] = mosaic.shape[1]
    new_hdr["NAXIS2"] = mosaic.shape[0]
    # Copy beam parameters from reference
    for key in ("BMAJ", "BMIN", "BPA", "BUNIT", "RESTFRQ", "EQUINOX"):
        if key in ref_hdr:
            new_hdr[key] = ref_hdr[key]
    new_hdr.update(out_wcs.to_header())
    new_hdr["HISTORY"] = f"Mosaic of {len(fits_paths)} DSA-110 tiles (2026-01-25)"

    hdu = fits.PrimaryHDU(data=mosaic.astype(np.float32), header=new_hdr)
    hdu.writeto(MOSAIC_OUT, overwrite=True)
    log.info("Mosaic written: %s", MOSAIC_OUT)
    return MOSAIC_OUT


def check_mosaic_quality(mosaic_path: str) -> bool:
    """Basic QA: check noise consistency across strips of the mosaic."""
    with fits.open(mosaic_path) as hdul:
        data = hdul[0].data.squeeze()

    ny, nx = data.shape
    n_strips = 5
    strip_height = ny // n_strips
    noises = []
    for i in range(n_strips):
        strip = data[i * strip_height:(i + 1) * strip_height, :]
        valid = strip[np.isfinite(strip)]
        if len(valid) > 100:
            # Robust noise estimate via MAD
            mad = np.median(np.abs(valid - np.median(valid)))
            noises.append(1.4826 * mad)

    if not noises:
        log.warning("QA: no valid strips")
        return False

    log.info("QA noise per strip [Jy/beam]: %s", [f"{n:.4f}" for n in noises])
    max_ratio = max(noises) / min(noises)
    log.info("QA max/min noise ratio: %.2f (pass if < 3.0)", max_ratio)
    passed = max_ratio < 3.0
    if passed:
        log.info("QA PASSED: noise consistent across field")
    else:
        log.warning("QA FAILED: noise varies too much (ratio=%.2f)", max_ratio)
    return passed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Mosaic a full day of DSA-110 drift observations.")
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        default=False,
        help="Keep *_meridian.ms files and skip moving the mosaic to products/ (useful for debugging).",
    )
    args = parser.parse_args()
    keep = args.keep_intermediates

    ms_list = find_valid_ms()
    if not ms_list:
        log.error("No valid MS files found — aborting")
        sys.exit(1)

    # ── Phase 1: Process each MS ──────────────────────────────────────────────
    tile_images = []
    for ms_path in ms_list:
        result = process_ms(ms_path, keep_intermediates=keep)
        if result:
            tile_images.append(result)
        else:
            log.warning("Skipping failed MS: %s", ms_path)

    log.info("\n=== Processed %d/%d tiles successfully ===\n", len(tile_images), len(ms_list))

    if len(tile_images) < 2:
        log.error("Need at least 2 tile images to mosaic — aborting")
        sys.exit(1)

    # ── Phase 2: Build mosaic ─────────────────────────────────────────────────
    if os.path.exists(MOSAIC_OUT):
        log.info("Mosaic already exists: %s", MOSAIC_OUT)
    else:
        log.info("\n=== Building mosaic from %d tiles ===\n", len(tile_images))
        out_wcs, ny, nx = build_common_wcs(tile_images)
        mosaic = coadd_tiles(tile_images, out_wcs, ny, nx)
        write_mosaic(mosaic, out_wcs, tile_images)

    # ── Phase 3: QA ───────────────────────────────────────────────────────────
    log.info("\n=== Mosaic QA ===\n")
    passed = check_mosaic_quality(MOSAIC_OUT)

    # Report peak flux
    with fits.open(MOSAIC_OUT) as hdul:
        data = hdul[0].data.squeeze()
        peak = np.nanmax(data)
        rms = 1.4826 * np.nanmedian(np.abs(data[np.isfinite(data)] - np.nanmedian(data[np.isfinite(data)])))
        log.info("Mosaic peak: %.4f Jy/beam", peak)
        log.info("Mosaic RMS (MAD): %.4f Jy/beam", rms)
        log.info("Dynamic range: %.0f", peak / rms)

    if passed:
        print(f"\nSUCCESS: Mosaic at {MOSAIC_OUT}")
    else:
        print(f"\nWARNING: Mosaic QA failed — check noise consistency")

    # ── Phase 4: Move finished mosaic to products/ ────────────────────────────
    if not keep:
        _move_mosaic_to_products()

    return MOSAIC_OUT


def _move_mosaic_to_products() -> None:
    """Move the completed mosaic directory from stage to the products tree.

    Source:      IMAGE_DIR  (/stage/dsa110-contimg/images/mosaic_{DATE}/)
    Destination: PRODUCTS_DIR  (/data/dsa110-continuum/products/mosaics/{DATE}/)

    If PRODUCTS_DIR already exists (e.g. from a previous run), the move is
    skipped and a warning is logged rather than overwriting science products.
    """
    if not os.path.isfile(MOSAIC_OUT):
        log.warning("Move skipped: mosaic file not found at %s", MOSAIC_OUT)
        return

    if os.path.exists(PRODUCTS_DIR):
        log.warning(
            "Move skipped: destination already exists — remove it manually if you want to overwrite: %s",
            PRODUCTS_DIR,
        )
        return

    parent = Path(PRODUCTS_DIR).parent
    parent.mkdir(parents=True, exist_ok=True)

    log.info("Moving mosaic directory: %s → %s", IMAGE_DIR, PRODUCTS_DIR)
    try:
        shutil.move(IMAGE_DIR, PRODUCTS_DIR)
        log.info("Mosaic moved to products: %s", PRODUCTS_DIR)
        print(f"\nMosaic archived to: {PRODUCTS_DIR}/full_mosaic.fits")
    except Exception as e:
        log.error("Failed to move mosaic to products: %s", e)
        print(f"\nERROR: Could not move mosaic to {PRODUCTS_DIR}: {e}")


if __name__ == "__main__":
    main()
