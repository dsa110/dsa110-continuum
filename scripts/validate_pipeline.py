#!/opt/miniforge/envs/casa6/bin/python
"""
Unified pipeline validation runner.

Runs all available QA checks against pipeline outputs for a given date and
produces a structured pass/warn/fail report. Designed to catch regressions
after code changes.

Usage:
    python scripts/validate_pipeline.py --date 2026-01-25
    python scripts/validate_pipeline.py --date 2026-01-25 --products-dir /data/dsa110-continuum/products
    python scripts/validate_pipeline.py --date 2026-01-25 --json-out /tmp/validation_report.json
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger("validate_pipeline")

# ── Thresholds ────────────────────────────────────────────────────────────────
# Mirror the thresholds used across the pipeline QA modules so they're
# centralized and auditable here.

FLUX_RATIO_PASS = 0.80       # median DSA/NVSS ratio for PASS (verify_sources.py)
FLUX_RATIO_WARN = 0.70       # below this is FAIL
MIN_DETECTIONS = 3            # minimum S/N>=3 detections for valid QA
MAX_RMS_MJYB = 500.0         # flag epochs with implausibly high noise
MIN_PEAK_JYB = 0.01          # flag epochs with no bright sources
MOSAIC_EXISTS = True          # require mosaic FITS to exist
PHOTOMETRY_EXISTS = True      # require forced_phot CSV to exist


# ── Individual checks ─────────────────────────────────────────────────────────

def check_mosaic_exists(mosaic_dir: Path, date: str) -> list[dict]:
    """Verify that at least one epoch mosaic FITS exists for the date."""
    results = []
    mosaics = sorted(mosaic_dir.glob(f"{date}T*_mosaic.fits"))

    results.append({
        "check": "mosaic_file_count",
        "value": len(mosaics),
        "status": "pass" if len(mosaics) > 0 else "fail",
        "detail": f"Found {len(mosaics)} mosaic FITS files",
    })

    return results


def check_photometry_csvs(mosaic_dir: Path, date: str) -> list[dict]:
    """Verify forced photometry CSVs exist and have plausible content."""
    results = []
    csvs = sorted(mosaic_dir.glob(f"{date}T*_forced_phot.csv"))

    results.append({
        "check": "photometry_csv_count",
        "value": len(csvs),
        "status": "pass" if len(csvs) > 0 else "warn",
        "detail": f"Found {len(csvs)} forced photometry CSVs",
    })

    for csv_path in csvs:
        epoch_label = csv_path.stem.replace("_forced_phot", "")
        try:
            import csv as csv_mod
            with open(csv_path) as f:
                reader = csv_mod.DictReader(f)
                rows = list(reader)

            n_sources = len(rows)
            if n_sources == 0:
                results.append({
                    "check": f"photometry_sources_{epoch_label}",
                    "value": 0,
                    "status": "fail",
                    "detail": "No sources in photometry CSV",
                })
                continue

            # Compute median DSA/NVSS ratio
            ratios = []
            for row in rows:
                try:
                    ratio = float(row.get("dsa_nvss_ratio", "nan"))
                    if np.isfinite(ratio) and ratio > 0:
                        ratios.append(ratio)
                except (ValueError, TypeError):
                    pass

            if len(ratios) >= MIN_DETECTIONS:
                median_ratio = float(np.median(ratios))
                if median_ratio >= FLUX_RATIO_PASS:
                    status = "pass"
                elif median_ratio >= FLUX_RATIO_WARN:
                    status = "warn"
                else:
                    status = "fail"

                results.append({
                    "check": f"flux_ratio_{epoch_label}",
                    "value": round(median_ratio, 4),
                    "status": status,
                    "detail": f"Median DSA/NVSS = {median_ratio:.3f} from {len(ratios)} sources",
                })
            else:
                results.append({
                    "check": f"flux_ratio_{epoch_label}",
                    "value": len(ratios),
                    "status": "fail",
                    "detail": f"Only {len(ratios)} valid ratios (need {MIN_DETECTIONS})",
                })

        except Exception as e:
            results.append({
                "check": f"photometry_read_{epoch_label}",
                "value": None,
                "status": "fail",
                "detail": f"Error reading CSV: {e}",
            })

    return results


def check_mosaic_stats(mosaic_dir: Path, date: str) -> list[dict]:
    """Check basic image statistics of each epoch mosaic."""
    results = []

    try:
        from astropy.io import fits
        from astropy.stats import mad_std
    except ImportError:
        results.append({
            "check": "mosaic_stats",
            "value": None,
            "status": "skip",
            "detail": "astropy not available",
        })
        return results

    mosaics = sorted(mosaic_dir.glob(f"{date}T*_mosaic.fits"))
    for mosaic_path in mosaics:
        epoch_label = mosaic_path.stem.replace("_mosaic", "")
        try:
            with fits.open(str(mosaic_path)) as hdul:
                data = hdul[0].data
                if data is None:
                    results.append({
                        "check": f"mosaic_data_{epoch_label}",
                        "value": None,
                        "status": "fail",
                        "detail": "No image data in primary HDU",
                    })
                    continue

                # Squeeze extra dimensions (Stokes, freq)
                data = np.squeeze(data)
                finite = data[np.isfinite(data)]

                if len(finite) == 0:
                    results.append({
                        "check": f"mosaic_data_{epoch_label}",
                        "value": 0,
                        "status": "fail",
                        "detail": "All pixels are NaN",
                    })
                    continue

                peak = float(np.nanmax(np.abs(finite)))
                rms = float(mad_std(finite)) * 1e3  # mJy/beam

                # Peak check
                results.append({
                    "check": f"mosaic_peak_{epoch_label}",
                    "value": round(peak, 4),
                    "status": "pass" if peak >= MIN_PEAK_JYB else "warn",
                    "detail": f"Peak = {peak:.4f} Jy/beam",
                })

                # RMS check
                results.append({
                    "check": f"mosaic_rms_{epoch_label}",
                    "value": round(rms, 2),
                    "status": "pass" if rms < MAX_RMS_MJYB else "warn",
                    "detail": f"RMS = {rms:.2f} mJy/beam",
                })

        except Exception as e:
            results.append({
                "check": f"mosaic_read_{epoch_label}",
                "value": None,
                "status": "fail",
                "detail": f"Error reading FITS: {e}",
            })

    return results


def check_qa_summary(products_dir: Path, date: str) -> list[dict]:
    """Check qa_summary.csv for this date's entries."""
    results = []
    qa_csv = products_dir / "qa_summary.csv"

    if not qa_csv.exists():
        results.append({
            "check": "qa_summary_exists",
            "value": None,
            "status": "warn",
            "detail": "qa_summary.csv not found",
        })
        return results

    try:
        import csv as csv_mod
        with open(qa_csv) as f:
            reader = csv_mod.DictReader(f)
            date_rows = [r for r in reader if r.get("date", "") == date]

        n_pass = sum(1 for r in date_rows if r.get("qa_result") == "pass")
        n_warn = sum(1 for r in date_rows if r.get("qa_result") == "warn")
        n_fail = sum(1 for r in date_rows if r.get("qa_result") == "fail")
        n_total = len(date_rows)

        results.append({
            "check": "qa_summary_epochs",
            "value": n_total,
            "status": "pass" if n_total > 0 else "warn",
            "detail": f"{n_total} epochs in qa_summary.csv ({n_pass} pass, {n_warn} warn, {n_fail} fail)",
        })

        if n_fail > 0:
            results.append({
                "check": "qa_summary_failures",
                "value": n_fail,
                "status": "fail" if n_fail > n_total // 2 else "warn",
                "detail": f"{n_fail}/{n_total} epochs failed QA",
            })

    except Exception as e:
        results.append({
            "check": "qa_summary_read",
            "value": None,
            "status": "fail",
            "detail": f"Error reading qa_summary.csv: {e}",
        })

    return results


def check_calibration_tables(ms_dir: Path, date: str) -> list[dict]:
    """Check that bandpass and gain tables exist for the date."""
    results = []
    bp_table = ms_dir / f"{date}T22:26:05_0~23.b"
    g_table = ms_dir / f"{date}T22:26:05_0~23.g"

    for label, path in [("bandpass", bp_table), ("gain", g_table)]:
        exists = path.exists() or path.is_symlink()
        results.append({
            "check": f"caltable_{label}",
            "value": str(path),
            "status": "pass" if exists else "fail",
            "detail": f"{label} table {'exists' if exists else 'MISSING'}: {path.name}",
        })

    return results


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(all_results: list[dict], date: str) -> int:
    """Print a formatted validation report. Returns exit code (0=pass, 1=issues)."""
    n_pass = sum(1 for r in all_results if r["status"] == "pass")
    n_warn = sum(1 for r in all_results if r["status"] == "warn")
    n_fail = sum(1 for r in all_results if r["status"] == "fail")
    n_skip = sum(1 for r in all_results if r["status"] == "skip")
    n_total = len(all_results)

    print(f"\n{'=' * 72}")
    print(f"  VALIDATION REPORT: {date}")
    print(f"{'=' * 72}")

    status_icons = {"pass": "OK", "warn": "!!", "fail": "XX", "skip": "--"}

    for r in all_results:
        icon = status_icons[r["status"]]
        print(f"  [{icon}] {r['check']:40s}  {r['detail']}")

    print(f"{'─' * 72}")
    print(f"  TOTAL: {n_total} checks — {n_pass} pass, {n_warn} warn, {n_fail} fail, {n_skip} skip")

    if n_fail == 0 and n_warn == 0:
        print(f"  RESULT: PASS")
    elif n_fail == 0:
        print(f"  RESULT: WARN (review warnings above)")
    else:
        print(f"  RESULT: FAIL")

    print(f"{'=' * 72}\n")

    return 1 if n_fail > 0 else 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Unified pipeline validation runner")
    parser.add_argument("--date", required=True, help="Observation date (YYYY-MM-DD)")
    parser.add_argument("--products-dir", default=os.environ.get("DSA110_PRODUCTS_BASE", "/data/dsa110-proc/products/mosaics").rsplit("/mosaics", 1)[0],
                        help="Products directory")
    parser.add_argument("--ms-dir", default="/stage/dsa110-contimg/ms",
                        help="Measurement Set directory")
    parser.add_argument("--json-out", default=None,
                        help="Write JSON report to this path")
    args = parser.parse_args()

    date = args.date
    products_dir = Path(args.products_dir)
    ms_dir = Path(args.ms_dir)
    mosaic_dir = products_dir / "mosaics" / date

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    all_results = []

    # Run all checks
    log.info("Checking calibration tables...")
    all_results.extend(check_calibration_tables(ms_dir, date))

    if mosaic_dir.exists():
        log.info("Checking mosaic files...")
        all_results.extend(check_mosaic_exists(mosaic_dir, date))

        log.info("Checking mosaic image statistics...")
        all_results.extend(check_mosaic_stats(mosaic_dir, date))

        log.info("Checking photometry CSVs...")
        all_results.extend(check_photometry_csvs(mosaic_dir, date))
    else:
        all_results.append({
            "check": "mosaic_dir_exists",
            "value": str(mosaic_dir),
            "status": "fail",
            "detail": f"Mosaic directory not found: {mosaic_dir}",
        })

    log.info("Checking QA summary...")
    all_results.extend(check_qa_summary(products_dir, date))

    # Print report
    exit_code = print_report(all_results, date)

    # Optional JSON output
    if args.json_out:
        report = {
            "date": date,
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "n_checks": len(all_results),
            "n_pass": sum(1 for r in all_results if r["status"] == "pass"),
            "n_warn": sum(1 for r in all_results if r["status"] == "warn"),
            "n_fail": sum(1 for r in all_results if r["status"] == "fail"),
            "checks": all_results,
        }
        Path(args.json_out).write_text(json.dumps(report, indent=2, default=str))
        log.info("JSON report written to %s", args.json_out)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
