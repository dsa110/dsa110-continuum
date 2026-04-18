#!/opt/miniforge/envs/casa6/bin/python
"""
Source finding on a DSA-110 mosaic using BANE + Aegean.

Steps:
  1. Run BANE to estimate background (bkg) and local RMS (rms)
  2. Run Aegean at --sigma threshold on the mosaic
  3. Write source catalog as FITS table
  4. Report statistics and verify success criteria

Usage:
  source_finding.py [--mosaic PATH] [--out PATH] [--sigma FLOAT] [--sim]
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

from dsa110_continuum.source_finding import run_source_finding

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Production default mosaic path (overridden by --mosaic or --sim)
_DEFAULT_MOSAIC = "/stage/dsa110-contimg/images/mosaic_2026-01-25/full_mosaic.fits"

# Sim-mode mosaic: pipeline_outputs/step6/step6_mosaic.fits relative to repo root
_SIM_MOSAIC = str(
    Path(__file__).resolve().parents[1]
    / "pipeline_outputs/step6/step6_mosaic.fits"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Source finding on a DSA-110 mosaic (BANE + Aegean).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mosaic", default=None, metavar="PATH",
        help="Path to mosaic FITS file.",
    )
    parser.add_argument(
        "--out", default=None, metavar="PATH",
        help="Output catalog path (default: {mosaic_stem}_sources.fits).",
    )
    parser.add_argument(
        "--sigma", type=float, default=7.0,
        help="Aegean detection threshold in σ.",
    )
    parser.add_argument(
        "--sim", action="store_true",
        help=(
            "Use the pipeline sim mosaic "
            "(pipeline_outputs/step6/step6_mosaic.fits). "
            "Runs full pipeline but exits 0 regardless of QA result."
        ),
    )
    args = parser.parse_args()

    # Resolve mosaic path
    if args.sim:
        mosaic_path = _SIM_MOSAIC
        log.info("[SIM MODE] Using simulated mosaic: %s", mosaic_path)
    else:
        mosaic_path = args.mosaic or _DEFAULT_MOSAIC

    if not Path(mosaic_path).exists():
        log.error("Mosaic not found: %s", mosaic_path)
        sys.exit(1)

    # Derive catalog output path from mosaic stem if not provided
    mosaic_p = Path(mosaic_path)
    out_path = args.out or str(mosaic_p.parent / (mosaic_p.stem + "_sources.fits"))

    # Quick-look: log mosaic peak and MAD-RMS before running BANE/Aegean
    with fits.open(mosaic_path) as hdul:
        data = hdul[0].data.squeeze()
        finite = data[np.isfinite(data)]
        peak = float(np.nanmax(data))
        med = float(np.median(finite))
        rms = float(1.4826 * np.median(np.abs(finite - med)))
        log.info(
            "Mosaic: peak=%.4f Jy/beam  MAD-RMS=%.4f Jy/beam  shape=%s",
            peak, rms, data.shape,
        )

    # Run full pipeline
    catalog_path = run_source_finding(
        mosaic_path,
        out_path,
        aegean_sigma=args.sigma,
    )

    print(f"\nCatalog written: {catalog_path}")

    if args.sim:
        log.info("[SIM MODE] Exiting 0 — QA not enforced on dirty-image mosaic")
        sys.exit(0)


if __name__ == "__main__":
    main()
