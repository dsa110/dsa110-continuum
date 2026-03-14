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
STAGE_BASE = Path(os.environ.get("DSA110_STAGE_IMAGE_BASE", "/stage/dsa110-contimg/images"))


def find_mosaics() -> list[tuple[Path, Path]]:
    """Return (mosaic_fits, csv_output) pairs from both products and stage dirs."""
    pairs: list[tuple[Path, Path]] = []
    seen_names: set[str] = set()

    # Products dir first (authoritative)
    for fits_path in sorted(PRODUCTS_DIR.rglob("*_mosaic.fits")):
        csv_path = fits_path.with_name(fits_path.name.replace("_mosaic.fits", "_forced_phot.csv"))
        pairs.append((fits_path, csv_path))
        seen_names.add(fits_path.name)

    # Stage dir: mosaics not yet archived
    for mosaic_dir in sorted(STAGE_BASE.glob("mosaic_*")):
        date = mosaic_dir.name.replace("mosaic_", "")
        for fits_path in sorted(mosaic_dir.glob("*_mosaic.fits")):
            if fits_path.name in seen_names:
                continue
            out_dir = PRODUCTS_DIR / date
            csv_path = out_dir / fits_path.name.replace("_mosaic.fits", "_forced_phot.csv")
            pairs.append((fits_path, csv_path))

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Regenerate forced photometry CSVs.")
    parser.add_argument("--products-dir", default=str(PRODUCTS_DIR))
    parser.add_argument("--min-flux-mjy", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done")
    args = parser.parse_args()

    pairs = find_mosaics()
    mosaic_paths = [p[0] for p in pairs]
    if not mosaic_paths:
        print("No mosaic FITS found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pairs)} mosaics to process:")
    for fits_path, csv_path in pairs:
        print(f"  {fits_path} → {csv_path}")

    if args.dry_run:
        print("\nDry run — no files written.")
        return

    results = []
    for mosaic_path, csv_dest in pairs:
        csv_path = str(csv_dest)
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
