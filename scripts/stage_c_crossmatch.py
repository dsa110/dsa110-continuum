#!/opt/miniforge/envs/casa6/bin/python
"""
Stage C post-discovery cross-match for DSA-110 continuum pipeline.

Reads an Aegean FITS source catalog (Stage B output), cross-matches against the
master radio catalog (NVSS+VLASS+FIRST+RACS) and individual fallback catalogs,
and writes an annotated FITS table.

Usage:
  stage_c_crossmatch.py [--catalog PATH] [--out PATH]
                        [--ra RA_DEG] [--dec DEC_DEG]
                        [--radius RADIUS_DEG]
                        [--match-radius ARCSEC]
                        [--sim]
"""
import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Production default — Stage B output
_DEFAULT_CATALOG = "/stage/dsa110-contimg/images/mosaic_2026-01-25/full_mosaic_sources.fits"

# Sim-mode catalog: pipeline_outputs/step6/step6_mosaic_sources.fits
_SIM_CATALOG = str(
    Path(__file__).resolve().parents[1]
    / "pipeline_outputs/step6/step6_mosaic_sources.fits"
)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Stage C: cross-match Aegean detections against radio catalogs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--catalog", default=None, metavar="PATH",
        help=f"Stage B Aegean FITS catalog. Default (production): {_DEFAULT_CATALOG}",
    )
    parser.add_argument(
        "--out", default=None, metavar="PATH",
        help="Output path for annotated FITS (default: {catalog_stem}_crossmatched.fits).",
    )
    parser.add_argument("--ra",  type=float, default=None, metavar="DEG",
                        help="Field center RA (degrees). Derived from catalog centroid if omitted.")
    parser.add_argument("--dec", type=float, default=None, metavar="DEG",
                        help="Field center Dec (degrees). Derived from catalog centroid if omitted.")
    parser.add_argument("--radius", type=float, default=2.0, metavar="DEG",
                        help="Cone search radius (degrees).")
    parser.add_argument("--match-radius", type=float, default=10.0, metavar="ARCSEC",
                        help="Cross-match radius (arcsec).")
    parser.add_argument(
        "--sim", action="store_true",
        help=(
            "Use sim-mode Stage B catalog "
            "(pipeline_outputs/step6/step6_mosaic_sources.fits). "
            "Exits 0 even if catalog is missing."
        ),
    )
    args = parser.parse_args(argv)

    # Resolve catalog path
    if args.sim:
        # Allow env override for testing
        catalog_path = os.environ.get("DSA110_SIM_STAGE_B_CATALOG", _SIM_CATALOG)
        log.info("[SIM MODE] Using sim catalog: %s", catalog_path)
    else:
        catalog_path = args.catalog or _DEFAULT_CATALOG

    if not Path(catalog_path).exists():
        if args.sim:
            log.warning(
                "[SIM MODE] Stage B catalog not found: %s — "
                "run source_finding.py first. Exiting 0.",
                catalog_path,
            )
            sys.exit(0)
        else:
            log.error("Catalog not found: %s", catalog_path)
            sys.exit(1)

    from dsa110_continuum.catalog.stage_c import run_stage_c

    try:
        out_path = run_stage_c(
            catalog_path,
            args.out,
            ra_center=args.ra,
            dec_center=args.dec,
            radius_deg=args.radius,
            match_radius_arcsec=args.match_radius,
        )
        print(f"\nAnnotated catalog written: {out_path}")
    except ValueError as exc:
        log.error("Stage C failed: %s", exc)
        if args.sim:
            sys.exit(0)
        sys.exit(1)


if __name__ == "__main__":
    main()
