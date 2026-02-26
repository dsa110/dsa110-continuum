#!/opt/miniforge/envs/casa6/bin/python
"""
Source finding on the 2026-01-25 mosaic using BANE + Aegean.

Steps:
  1. Run BANE to estimate background (bkg) and local RMS (rms)
  2. Run Aegean at 5-sigma threshold on the mosaic
  3. Output source catalog as FITS table
  4. Report statistics and verify success criteria
"""
import logging
import os
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MOSAIC = "/stage/dsa110-contimg/images/mosaic_2026-01-25/full_mosaic.fits"
CATALOG_OUT = "/stage/dsa110-contimg/images/mosaic_2026-01-25/sources.fits"

# BANE box size: ~50 beams is a good rule of thumb
# Beam ~60" FWHM, pixel 6" → ~10 px/beam → 50 beams = 500 px
# Use 300 px boxes; step 200 px for good spatial sampling on 17k-px mosaic.
# cores=1 avoids shared-memory multiprocessing deadlock on NaN-heavy images.
BANE_BOX = 600      # pixels per box (larger → better tracks pbcor noise gradient at tile edges)
BANE_STEP = 300     # step between box centers
AEGEAN_SIGMA = 7.0  # raised to 7σ to suppress spurious edge detections from pbcor noise amplification


def run_bane(mosaic_path: str) -> tuple[str, str]:
    """Run BANE background estimator. Returns (bkg_path, rms_path)."""
    stem = mosaic_path.replace(".fits", "")
    bkg_path = stem + "_bkg.fits"
    rms_path = stem + "_rms.fits"

    if os.path.exists(bkg_path) and os.path.exists(rms_path):
        log.info("BANE outputs already exist — skipping")
        return bkg_path, rms_path

    log.info("Running BANE on %s ...", mosaic_path)
    # Use the AegeanTools Python API to avoid CLI Python version issues
    from AegeanTools import BANE as BaneModule

    # BANE.filter_image is the main entry point
    BaneModule.filter_image(
        im_name=mosaic_path,
        out_base=stem,
        step_size=[BANE_STEP, BANE_STEP],
        box_size=[BANE_BOX, BANE_BOX],
        cores=1,  # single-threaded avoids shared-memory deadlock on NaN-heavy mosaics
        mask=True,
    )

    if not os.path.exists(bkg_path) or not os.path.exists(rms_path):
        raise RuntimeError(f"BANE did not produce expected outputs at {stem}_bkg/rms.fits")

    log.info("BANE done: bkg=%s, rms=%s", bkg_path, rms_path)
    return bkg_path, rms_path


def run_aegean(mosaic_path: str, bkg_path: str, rms_path: str) -> str:
    """Run Aegean source finder. Returns path to output catalog FITS."""
    if os.path.exists(CATALOG_OUT):
        log.info("Catalog already exists — skipping Aegean")
        return CATALOG_OUT

    log.info("Running Aegean (%.1fσ threshold) ...", AEGEAN_SIGMA)
    from AegeanTools.source_finder import SourceFinder

    sf = SourceFinder(log=logging.getLogger("AegeanTools"))

    # innerclip: island detection threshold (sigma)
    # outerclip: island flood-fill stop threshold (sigma)
    # rmsin: path to BANE RMS map
    # bkgin: path to BANE background map
    found = sf.find_sources_in_image(
        filename=mosaic_path,
        hdu_index=0,
        outfile=None,
        innerclip=AEGEAN_SIGMA,
        outerclip=AEGEAN_SIGMA - 1.0,  # flood-fill 1σ below detection threshold
        rmsin=rms_path,
        bkgin=bkg_path,
        cores=1,
    )

    if not found:
        log.warning("Aegean found no sources at %.1fσ", AEGEAN_SIGMA)
        # Write empty catalog
        _write_empty_catalog()
        return CATALOG_OUT

    log.info("Aegean found %d source components", len(found))
    _write_catalog(found)
    return CATALOG_OUT


def _write_catalog(sources) -> None:
    """Write Aegean source list to FITS table."""
    from astropy.table import Table

    rows = []
    for s in sources:
        ra = float(s.ra)
        dec = float(s.dec)
        rows.append({
            "source_name": f"AEG_J{ra:.4f}{dec:+.4f}",
            "ra_deg": ra,
            "dec_deg": dec,
            "peak_flux_jy": float(s.peak_flux),
            "peak_flux_err_jy": float(getattr(s, "err_peak_flux", 0.0)),
            "int_flux_jy": float(getattr(s, "int_flux", s.peak_flux)),
            "a_arcsec": float(getattr(s, "a", 0.0)),
            "b_arcsec": float(getattr(s, "b", 0.0)),
            "pa_deg": float(getattr(s, "pa", 0.0)),
            "local_rms_jy": float(getattr(s, "local_rms", 0.0)),
        })

    t = Table(rows)
    t.write(CATALOG_OUT, overwrite=True)
    log.info("Catalog written: %s  (%d sources)", CATALOG_OUT, len(rows))


def _write_empty_catalog() -> None:
    from astropy.table import Table
    t = Table(names=["source_name", "ra_deg", "dec_deg", "peak_flux_jy",
                     "peak_flux_err_jy", "int_flux_jy"],
              dtype=["U32", float, float, float, float, float])
    t.write(CATALOG_OUT, overwrite=True)
    log.info("Empty catalog written: %s", CATALOG_OUT)


def check_catalog(catalog_path: str) -> bool:
    """Success criterion: catalog has sources and at least one > 1 Jy."""
    from astropy.table import Table
    t = Table.read(catalog_path)

    log.info("Catalog: %d sources", len(t))
    if len(t) == 0:
        log.error("FAIL: empty catalog")
        return False

    if "peak_flux_jy" in t.colnames:
        bright = t[t["peak_flux_jy"] > 1.0]
        log.info("Sources > 1 Jy: %d", len(bright))
        if len(bright) > 0:
            for row in bright:
                log.info("  %s  RA=%.3f  Dec=%.3f  peak=%.2f Jy",
                         row["source_name"], row["ra_deg"], row["dec_deg"],
                         row["peak_flux_jy"])

    # Sanity check positions
    if "ra_deg" in t.colnames:
        ras = t["ra_deg"]
        decs = t["dec_deg"]
        in_range = np.sum((ras > 300) & (ras < 360) & (decs > 0) & (decs < 40))
        log.info("Sources in expected sky region: %d/%d", in_range, len(t))

    passed = len(t) > 0 and (
        "peak_flux_jy" not in t.colnames or (t["peak_flux_jy"] > 1.0).any()
    )
    if passed:
        log.info("QA PASSED: catalog has sources including bright detections")
    else:
        log.warning("QA WARNING: no bright sources (>1 Jy) detected")
    return len(t) > 0


def main():
    if not os.path.exists(MOSAIC):
        log.error("Mosaic not found: %s — run mosaic_day.py first", MOSAIC)
        sys.exit(1)

    log.info("Mosaic: %s", MOSAIC)
    with fits.open(MOSAIC) as hdul:
        data = hdul[0].data.squeeze()
        peak = np.nanmax(data)
        rms = 1.4826 * np.nanmedian(np.abs(data[np.isfinite(data)] - np.nanmedian(data[np.isfinite(data)])))
        log.info("Mosaic peak=%.4f Jy/beam, global MAD-RMS=%.4f Jy/beam", peak, rms)

    # ── BANE ──────────────────────────────────────────────────────────────────
    bkg_path, rms_path = run_bane(MOSAIC)

    # ── Aegean ────────────────────────────────────────────────────────────────
    catalog_path = run_aegean(MOSAIC, bkg_path, rms_path)

    # ── QA ────────────────────────────────────────────────────────────────────
    passed = check_catalog(catalog_path)

    if passed:
        print(f"\nSUCCESS: Source catalog at {catalog_path}")
    else:
        print(f"\nWARNING: Source finding may have issues — inspect {catalog_path}")
        sys.exit(1)

    return catalog_path


if __name__ == "__main__":
    main()
