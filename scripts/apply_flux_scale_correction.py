#!/usr/bin/env python3
"""Apply Huber-regression flux scale correction to DSA-110 mosaics and cal tables.

Usage:
    /opt/miniforge/envs/casa6/bin/python scripts/apply_flux_scale_correction.py \
        --date 2026-01-25 --mosaics-dir /stage/dsa110-continuum/images/mosaic_2026-01-25

Per-epoch correction is derived from NVSS sources in the mosaic footprint using
Huber robust regression (``vast-crossref.md``).  The mosaic pixel values are
updated via ``S_corrected = (S_measured - offset) / gradient``.  New, corrected
bandpass/gain tables are also written and a provenance sidecar is emitted.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).parent.parent))

from dsa110_continuum.calibration.flux_scale_correction import (
    FluxScaleResult,
    apply_flux_scale_to_caltable,
    apply_flux_scale_to_mosaic,
    correction_factor,
    measure_flux_scale_from_mosaic,
)
from dsa110_continuum.photometry.epoch_qa import measure_epoch_qa

logger = logging.getLogger(__name__)

DEFAULT_NVSS_DB = "/data/dsa110-continuum/state/catalogs/nvss_full.sqlite3"
DEFAULT_STAGE = "/stage/dsa110-continuum/images"
DEFAULT_CAL_DIR = "/stage/dsa110-continuum/ms"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Observation date YYYY-MM-DD")
    parser.add_argument("--mosaics-dir", help="Directory containing *_mosaic.fits files")
    parser.add_argument("--nvss-db", default=DEFAULT_NVSS_DB, help="NVSS SQLite catalog")
    parser.add_argument("--cal-dir", default=DEFAULT_CAL_DIR, help="MS/cal table root")
    parser.add_argument(
        "--correct-cal-tables",
        action="store_true",
        help="Also generate corrected bandpass and gain tables",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip epochs whose mosaic already has FLUXCORR header",
    )
    parser.add_argument("--dry-run", action="store_true", help="Compute corrections only")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress info logging")
    return parser.parse_args()


def find_epoch_mosaics(mosaics_dir: Path) -> list[Path]:
    paths = sorted(mosaics_dir.glob("*_mosaic.fits"))
    # Exclude weight maps and diagnostic files
    return [p for p in paths if "weights" not in p.name and "qa" not in p.name]


def find_cal_tables(cal_dir: Path, date: str) -> tuple[Path | None, Path | None]:
    bp = list(cal_dir.glob(f"{date}T*_0~23.b"))
    g = list(cal_dir.glob(f"{date}T*_0~23.g"))
    return (bp[0] if bp else None, g[0] if g else None)


def write_correction_provenance(
    path: Path,
    epoch: str,
    result: FluxScaleResult,
    mosaic_path: str,
    qa_before: dict,
    qa_after: dict,
) -> None:
    provenance = {
        "epoch": epoch,
        "mosaic": mosaic_path,
        "gradient": result.gradient,
        "gradient_err": result.gradient_err,
        "offset": result.offset,
        "offset_err": result.offset_err,
        "n_fit": result.n_fit,
        "n_candidate": result.n_candidate,
        "median_ratio": result.median_flux_ratio,
        "multiplicative_correction": correction_factor(result),
        "qa_before": qa_before,
        "qa_after": qa_after,
    }
    with open(path, "w") as f:
        json.dump(provenance, f, indent=2, default=str)


def qa_summary(result) -> dict:
    return {
        "qa_result": result.qa_result,
        "median_ratio": result.median_ratio,
        "completeness_frac": result.completeness_frac,
        "mosaic_rms_mjy": result.mosaic_rms_mjy,
        "n_catalog": result.n_catalog,
        "n_recovered": result.n_recovered,
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    date = args.date
    mosaics_dir = Path(args.mosaics_dir or f"{DEFAULT_STAGE}/mosaic_{date}")
    if not mosaics_dir.is_dir():
        logger.error("Mosaics directory not found: %s", mosaics_dir)
        return 1

    mosaic_paths = find_epoch_mosaics(mosaics_dir)
    if not mosaic_paths:
        logger.error("No *_mosaic.fits files in %s", mosaics_dir)
        return 1

    out_dir = mosaics_dir / f"fluxscale_provenance_{date}"
    out_dir.mkdir(exist_ok=True)

    all_gradients: list[float] = []
    all_results: dict[str, FluxScaleResult] = {}

    for mosaic_path in mosaic_paths:
        epoch = mosaic_path.name.replace("_mosaic.fits", "")
        logger.info("--- %s ---", epoch)

        if args.skip_existing:
            with fits.open(mosaic_path) as hdul:
                if "FLUXCORR" in hdul[0].header:
                    logger.info("Skipping %s (already corrected)", epoch)
                    continue

        result = measure_flux_scale_from_mosaic(str(mosaic_path), args.nvss_db)
        logger.info("%s", result.message)
        all_results[epoch] = result
        if np.isfinite(result.gradient) and result.gradient != 1.0:
            all_gradients.append(result.gradient)

        if not result.passed:
            logger.warning("Flux scale fit did not pass for %s; skipping application", epoch)
            continue
        if result.gradient == 1.0 and result.offset == 0.0:
            logger.info("Identity correction for %s; nothing to apply", epoch)
            continue

        qa_before = qa_summary(measure_epoch_qa(str(mosaic_path)))
        logger.info(
            "QA before: ratio=%.3f (%s)",
            qa_before["median_ratio"],
            qa_before["qa_result"],
        )

        if args.dry_run:
            logger.info("Dry-run: would correct %s by %.4f", epoch, correction_factor(result))
            continue

        corrected = apply_flux_scale_to_mosaic(str(mosaic_path), result)
        logger.info("Corrected mosaic: %s", corrected)

        qa_after = qa_summary(measure_epoch_qa(corrected))
        logger.info(
            "QA after: ratio=%.3f (%s)",
            qa_after["median_ratio"],
            qa_after["qa_result"],
        )

        provenance_path = out_dir / f"{epoch}_fluxscale.json"
        write_correction_provenance(
            provenance_path, epoch, result, str(mosaic_path), qa_before, qa_after
        )

    if not all_results:
        logger.warning("No flux scale corrections computed")
        return 0

    if args.correct_cal_tables and all_gradients:
        median_gradient = float(np.median(all_gradients))
        cal_result = FluxScaleResult(
            gradient=median_gradient,
            offset=0.0,
            n_candidate=0,
            n_fit=0,
            n_outlier=0,
            passed=True,
            message=f"Median gradient across {len(all_gradients)} epochs = {median_gradient:.4f}",
        )
        logger.info("Cal table correction: gradient=%.4f scale=%.4f", median_gradient, correction_factor(cal_result))

        bp_path, g_path = find_cal_tables(Path(args.cal_dir), date)
        if bp_path and g_path:
            if not args.dry_run:
                bp_corr = apply_flux_scale_to_caltable(str(bp_path), cal_result)
                g_corr = apply_flux_scale_to_caltable(str(g_path), cal_result)
                logger.info("Corrected cal tables: %s, %s", bp_corr, g_corr)
                sidecar = {
                    "date": date,
                    "gradient": median_gradient,
                    "multiplicative_correction": correction_factor(cal_result),
                    "bp_original": str(bp_path),
                    "g_original": str(g_path),
                    "bp_corrected": bp_corr,
                    "g_corrected": g_corr,
                    "epochs": list(all_results.keys()),
                }
                with open(out_dir / "cal_tables_correction.json", "w") as f:
                    json.dump(sidecar, f, indent=2, default=str)
        else:
            logger.warning("Could not locate B/G tables for %s in %s", date, args.cal_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
