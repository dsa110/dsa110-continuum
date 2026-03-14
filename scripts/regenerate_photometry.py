#!/opt/miniforge/envs/casa6/bin/python
"""Regenerate all forced photometry CSVs with the unified master catalog schema.

Scans products/mosaics/*/ for *_mosaic.fits files and calls
run_forced_photometry() for each, overwriting old CSVs.

Usage:
    python scripts/regenerate_photometry.py [--products-dir ...]
"""
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from forced_photometry import run_forced_photometry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PRODUCTS_DIR = Path(os.environ.get("DSA110_PRODUCTS_BASE", "/data/dsa110-proc/products/mosaics"))


def main():
    parser = argparse.ArgumentParser(description="Regenerate forced photometry CSVs.")
    parser.add_argument("--products-dir", default=str(PRODUCTS_DIR))
    parser.add_argument("--min-flux-mjy", type=float, default=10.0)
    args = parser.parse_args()

    products = Path(args.products_dir)
    mosaic_paths = sorted(products.glob("*/*_mosaic.fits"))
    if not mosaic_paths:
        print(f"No mosaic FITS found under {products}/", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(mosaic_paths)} mosaics to process:")
    for p in mosaic_paths:
        print(f"  {p}")

    results = []
    for mosaic_path in mosaic_paths:
        csv_path = str(mosaic_path).replace("_mosaic.fits", "_forced_phot.csv")
        log.info("Processing: %s", mosaic_path.name)
        try:
            result = run_forced_photometry(
                str(mosaic_path),
                output_csv=csv_path,
                min_flux_mjy=args.min_flux_mjy,
            )
            results.append((mosaic_path.name, result))
            log.info(
                "  -> %d sources, median ratio %.3f",
                result["n_sources"], result["median_ratio"],
            )
        except Exception as e:
            log.error("  FAILED: %s", e)
            results.append((mosaic_path.name, None))

    print("\n=== Regeneration Summary ===")
    ok = sum(1 for _, r in results if r is not None)
    print(f"  {ok}/{len(results)} mosaics processed successfully")
    for name, r in results:
        status = f"{r['n_sources']} sources" if r else "FAILED"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
