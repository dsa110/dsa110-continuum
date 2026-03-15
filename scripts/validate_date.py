#!/opt/miniforge/envs/casa6/bin/python
"""
Validate a single date's pipeline readiness and output quality.

This script is the prescribed method for confirming that a date's data
can be reliably processed and that the resulting images are scientifically
sound.  It runs the same checks regardless of which date is provided —
no date is assumed to be "known good."

Stages:
  1. DATA INVENTORY     — verify MS files exist and are readable
  2. CAL ASSESSMENT     — evaluate cal table quality metrics
  3. TILE CHECK         — verify existing tile images (or note they're missing)
  4. MOSAIC CHECK       — inspect existing epoch mosaics
  5. SOURCE FINDING     — run BANE + Aegean blind source finding
  6. FLUX SCALE         — forced photometry cross-match against catalog
  7. REPORT             — write JSON report + diagnostic PNG

Usage:
    python scripts/validate_date.py --date 2026-01-25
    python scripts/validate_date.py --date 2026-02-12 --cal-date 2026-01-25
    python scripts/validate_date.py --date 2026-01-25 --skip-source-finding
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from casacore.tables import table

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MS_DIR = os.environ.get("DSA110_MS_DIR", "/stage/dsa110-contimg/ms")
STAGE_IMAGE_BASE = os.environ.get("DSA110_STAGE_IMAGE_BASE", "/stage/dsa110-contimg/images")
PRODUCTS_BASE = os.environ.get("DSA110_PRODUCTS_BASE", "/data/dsa110-proc/products/mosaics")
OUTPUT_DIR = "/data/dsa110-continuum/outputs/validation"


# ── Stage 1: Data inventory ──────────────────────────────────────────────────

def check_ms_inventory(date: str) -> dict:
    """Count and validate MS files for a date."""
    import glob
    pattern = f"{MS_DIR}/{date}T*.ms"
    candidates = sorted(glob.glob(pattern))
    candidates = [p for p in candidates if "meridian" not in p and "flagversion" not in p]

    valid = []
    corrupt = []
    broken_links = []
    for path in candidates:
        if os.path.islink(path) and not os.path.exists(path):
            broken_links.append(path)
            continue
        if not os.path.isdir(path):
            corrupt.append(path)
            continue
        try:
            with table(path, readonly=True, ack=False) as t:
                _ = t.nrows()
            valid.append(path)
        except Exception:
            corrupt.append(path)

    # Parse hours
    hours = set()
    for p in valid:
        try:
            ts = Path(p).stem.split("T")[1][:2]
            hours.add(int(ts))
        except (IndexError, ValueError):
            pass

    result = {
        "n_total": len(candidates),
        "n_valid": len(valid),
        "n_corrupt": len(corrupt),
        "n_broken_links": len(broken_links),
        "hours_covered": sorted(hours),
        "corrupt_files": [Path(p).name for p in corrupt],
    }

    if result["n_valid"] == 0 and len(broken_links) > 0:
        result["verdict"] = "UNAVAILABLE"
        log.warning("STAGE 1: %d MS entries are broken symlinks (raw data moved/deleted); "
                    "%d broken links found", len(broken_links), len(broken_links))
    elif result["n_valid"] == 0:
        result["verdict"] = "FAIL"
        log.error("STAGE 1: No valid MS files found for %s", date)
    elif result["n_corrupt"] > 0:
        result["verdict"] = "WARN"
        log.warning("STAGE 1: %d/%d MS files corrupt", result["n_corrupt"], result["n_total"])
    else:
        result["verdict"] = "OK"
        log.info("STAGE 1: %d MS files, hours %s", result["n_valid"],
                 [f"{h:02d}" for h in result["hours_covered"]])

    return result


# ── Stage 2: Cal table assessment ────────────────────────────────────────────

def check_cal_quality(date: str, cal_date: str) -> dict:
    """Assess calibration table quality."""
    bp_path = f"{MS_DIR}/{cal_date}T22:26:05_0~23.b"
    g_path = f"{MS_DIR}/{cal_date}T22:26:05_0~23.g"

    result = {
        "cal_date": cal_date,
        "cross_date": cal_date != date,
        "bp_table": bp_path,
        "g_table": g_path,
        "bp_exists": os.path.exists(bp_path),
        "g_exists": os.path.exists(g_path),
    }

    if not result["bp_exists"] or not result["g_exists"]:
        result["verdict"] = "FAIL"
        missing = []
        if not result["bp_exists"]:
            missing.append("BP")
        if not result["g_exists"]:
            missing.append("G")
        log.error("STAGE 2: Missing cal tables: %s", ", ".join(missing))
        return result

    # Compute quality metrics
    try:
        from dsa110_continuum.calibration.qa import compute_calibration_metrics

        issues = []
        for label, path in [("bp", bp_path), ("g", g_path)]:
            metrics = compute_calibration_metrics(path)
            d = metrics.to_dict()
            result[f"{label}_metrics"] = d

            if metrics.extraction_error:
                issues.append(f"{label.upper()}: extraction error — {metrics.extraction_error}")
                continue

            log.info("STAGE 2: %s — flag_frac=%.1f%%, phase_scatter=%.1f°, amp_scatter=%.1f%%",
                     label.upper(), metrics.flag_fraction * 100,
                     metrics.phase_scatter_deg, metrics.std_amplitude)

            if metrics.flag_fraction > 0.5:
                issues.append(f"{label.upper()} table {metrics.flag_fraction:.0%} flagged (> 50%)")
            if metrics.flag_fraction > 0.3:
                issues.append(f"{label.upper()} table {metrics.flag_fraction:.0%} flagged (> 30%, elevated)")

            if label == "g" and result["cross_date"] and metrics.phase_scatter_deg > 30.0:
                issues.append(
                    f"Cross-date G phase scatter {metrics.phase_scatter_deg:.1f}° > 30° — "
                    f"phase cal from {cal_date} may not transfer to {date}"
                )

        result["issues"] = issues
        if any("Cross-date" in i or "> 50%" in i for i in issues):
            result["verdict"] = "WARN"
        elif issues:
            result["verdict"] = "NOTE"
        else:
            result["verdict"] = "OK"

    except Exception as e:
        result["verdict"] = "ERROR"
        result["error"] = str(e)
        log.error("STAGE 2: Cal quality check failed: %s", e)

    return result


# ── Stage 3: Tile image check ────────────────────────────────────────────────

def check_tiles(date: str) -> dict:
    """Check existing tile images."""
    import glob
    stage_dir = f"{STAGE_IMAGE_BASE}/mosaic_{date}"
    tiles = sorted(glob.glob(f"{stage_dir}/*-image-pb.fits"))
    plain_tiles = sorted(glob.glob(f"{stage_dir}/*-image.fits"))
    # Exclude tiles that also have pb versions
    plain_only = [t for t in plain_tiles if t.replace("-image.fits", "-image-pb.fits") not in tiles]

    all_tiles = tiles + plain_only

    result = {
        "n_pb_corrected": len(tiles),
        "n_plain_only": len(plain_only),
        "n_total": len(all_tiles),
        "stage_dir": stage_dir,
        "tile_stats": [],
    }

    if not all_tiles:
        result["verdict"] = "NONE"
        log.info("STAGE 3: No tile images found — pipeline has not been run for %s", date)
        return result

    # Quick quality check on each tile
    n_good = 0
    n_bad = 0
    for tile_path in all_tiles:
        try:
            with fits.open(tile_path) as hdul:
                data = hdul[0].data.squeeze()
            finite = data[np.isfinite(data)]
            if len(finite) == 0:
                n_bad += 1
                continue
            peak = float(np.nanmax(data))
            rms = float(1.4826 * np.nanmedian(np.abs(finite - np.nanmedian(finite))))
            all_zero = np.allclose(finite, 0, atol=1e-10)
            if all_zero:
                n_bad += 1
            else:
                n_good += 1
                result["tile_stats"].append({
                    "name": Path(tile_path).name,
                    "peak_jyb": round(peak, 4),
                    "rms_mjyb": round(rms * 1000, 2),
                    "dr": round(peak / rms, 0) if rms > 0 else 0,
                })
        except Exception:
            n_bad += 1

    result["n_good"] = n_good
    result["n_bad"] = n_bad

    if n_bad > 0:
        result["verdict"] = "WARN"
        log.warning("STAGE 3: %d/%d tiles bad (all-zero or unreadable)", n_bad, len(all_tiles))
    else:
        result["verdict"] = "OK"
        log.info("STAGE 3: %d tiles, all good", n_good)

    # Summarize tile stats
    if result["tile_stats"]:
        rms_vals = [t["rms_mjyb"] for t in result["tile_stats"]]
        dr_vals = [t["dr"] for t in result["tile_stats"]]
        log.info("  Tile RMS: median=%.1f mJy/beam, range=%.1f–%.1f",
                 np.median(rms_vals), min(rms_vals), max(rms_vals))
        log.info("  Tile DR:  median=%.0f, range=%.0f–%.0f",
                 np.median(dr_vals), min(dr_vals), max(dr_vals))

    return result


# ── Stage 4: Mosaic check ────────────────────────────────────────────────────

def check_mosaics(date: str) -> dict:
    """Inspect existing epoch mosaics."""
    import glob
    stage_dir = f"{STAGE_IMAGE_BASE}/mosaic_{date}"
    products_dir = f"{PRODUCTS_BASE}/{date}"

    mosaics = sorted(
        glob.glob(f"{stage_dir}/*_mosaic.fits")
        + glob.glob(f"{products_dir}/*_mosaic.fits")
    )
    # Deduplicate by filename
    seen = set()
    unique_mosaics = []
    for m in mosaics:
        name = Path(m).name
        if name not in seen:
            seen.add(name)
            unique_mosaics.append(m)

    result = {
        "n_mosaics": len(unique_mosaics),
        "mosaics": [],
    }

    if not unique_mosaics:
        result["verdict"] = "NONE"
        log.info("STAGE 4: No mosaics found — run batch_pipeline.py first")
        return result

    for mosaic_path in unique_mosaics:
        try:
            with fits.open(mosaic_path) as hdul:
                data = hdul[0].data.squeeze()
                hdr = hdul[0].header
            finite = data[np.isfinite(data)]
            peak = float(np.nanmax(data))
            rms = float(1.4826 * np.nanmedian(np.abs(finite - np.nanmedian(finite))))
            nan_frac = float(np.mean(np.isnan(data)))

            wcs = WCS(hdr).celestial
            ny, nx = data.shape
            center = wcs.pixel_to_world(nx / 2.0, ny / 2.0)

            m_info = {
                "path": mosaic_path,
                "name": Path(mosaic_path).name,
                "shape": list(data.shape),
                "peak_jyb": round(peak, 4),
                "rms_mjyb": round(rms * 1000, 2),
                "dynamic_range": round(peak / rms, 0) if rms > 0 else 0,
                "nan_fraction": round(nan_frac, 3),
                "center_ra_deg": round(float(center.ra.deg), 3),
                "center_dec_deg": round(float(center.dec.deg), 3),
            }

            # Check for QA header cards
            for key in ["QARESULT", "QARMS", "QARAT", "CALDATE", "NTILES"]:
                if key in hdr:
                    m_info[key] = str(hdr[key])

            result["mosaics"].append(m_info)
            log.info("STAGE 4: %s — peak=%.3f Jy/beam, RMS=%.2f mJy/beam, DR=%.0f, center=(%.1f, %.1f)",
                     m_info["name"], peak, rms * 1000, peak / rms if rms > 0 else 0,
                     m_info["center_ra_deg"], m_info["center_dec_deg"])

        except Exception as e:
            result["mosaics"].append({"path": mosaic_path, "error": str(e)})
            log.error("STAGE 4: Could not read %s: %s", mosaic_path, e)

    result["verdict"] = "OK" if result["mosaics"] else "NONE"
    return result


# ── Stage 5: Source finding ──────────────────────────────────────────────────

def run_source_finding(mosaic_path: str, output_dir: str) -> dict:
    """Run BANE + Aegean on a mosaic."""
    from source_finding import run_bane, run_aegean, check_catalog

    catalog_out = os.path.join(output_dir, Path(mosaic_path).stem + "_sources.fits")

    # Temporarily set the module-level CATALOG_OUT
    import source_finding
    source_finding.CATALOG_OUT = catalog_out

    result = {"mosaic": mosaic_path}

    try:
        bkg_path, rms_path = run_bane(mosaic_path)
        result["bkg_path"] = bkg_path
        result["rms_path"] = rms_path
    except Exception as e:
        result["verdict"] = "ERROR"
        result["error"] = f"BANE failed: {e}"
        log.error("STAGE 5: BANE failed: %s", e)
        return result

    try:
        catalog_path = run_aegean(mosaic_path, bkg_path, rms_path)
        result["catalog_path"] = catalog_path
    except Exception as e:
        result["verdict"] = "ERROR"
        result["error"] = f"Aegean failed: {e}"
        log.error("STAGE 5: Aegean failed: %s", e)
        return result

    # Check catalog
    from astropy.table import Table
    try:
        t = Table.read(catalog_path)
        result["n_sources"] = len(t)
        if "peak_flux_jy" in t.colnames:
            fluxes = t["peak_flux_jy"]
            result["n_bright_1jy"] = int(np.sum(fluxes > 1.0))
            result["n_bright_100mjy"] = int(np.sum(fluxes > 0.1))
            result["flux_max_jy"] = round(float(np.max(fluxes)), 4)
            result["flux_median_jy"] = round(float(np.median(fluxes)), 4)
        log.info("STAGE 5: %d sources found (%d > 1 Jy, %d > 100 mJy)",
                 result["n_sources"], result.get("n_bright_1jy", 0),
                 result.get("n_bright_100mjy", 0))
        result["verdict"] = "OK" if result["n_sources"] > 0 else "WARN"
    except Exception as e:
        result["verdict"] = "ERROR"
        result["error"] = str(e)

    return result


# ── Stage 6: Flux scale via forced photometry ────────────────────────────────

def check_flux_scale(mosaic_path: str, output_dir: str) -> dict:
    """Run forced photometry and assess flux scale."""
    from forced_photometry import run_forced_photometry

    out_csv = os.path.join(output_dir, Path(mosaic_path).stem + "_forced_phot.csv")

    result = {"mosaic": mosaic_path}

    try:
        # Use NVSS for flux scale validation — DSA-110 observes at L-band (1.4 GHz),
        # same as NVSS.  The master catalog mixes frequencies (VLASS is 3 GHz),
        # which introduces spectral index bias into flux ratios.
        phot = run_forced_photometry(
            mosaic_path, output_csv=out_csv, catalog="nvss", min_flux_mjy=50.0,
        )
        result["n_sources"] = phot["n_sources"]
        result["median_ratio"] = round(phot["median_ratio"], 4) if np.isfinite(phot["median_ratio"]) else None
        result["csv_path"] = phot["csv_path"]

        # Read back CSV for detailed stats
        import csv
        with open(phot["csv_path"]) as f:
            rows = list(csv.DictReader(f))
        ratios = [float(r["flux_ratio"]) for r in rows
                  if r.get("flux_ratio") and r["flux_ratio"] != ""
                  and np.isfinite(float(r["flux_ratio"]))]

        if ratios:
            result["ratio_std"] = round(float(np.std(ratios)), 4)
            result["ratio_iqr"] = round(float(np.percentile(ratios, 75) - np.percentile(ratios, 25)), 4)
            result["n_outliers"] = sum(1 for r in ratios if r < 0.5 or r > 2.0)

            med = result["median_ratio"]
            if med is not None and 0.8 <= med <= 1.2:
                result["flux_scale_verdict"] = "GOOD"
            elif med is not None and 0.5 <= med <= 2.0:
                result["flux_scale_verdict"] = "ACCEPTABLE"
            else:
                result["flux_scale_verdict"] = "BAD"

            log.info("STAGE 6: %d sources, median ratio=%.3f (IQR=%.3f), %s",
                     result["n_sources"], med or 0, result["ratio_iqr"],
                     result["flux_scale_verdict"])

        result["verdict"] = "OK" if result.get("flux_scale_verdict") in ("GOOD", "ACCEPTABLE") else "WARN"

    except Exception as e:
        result["verdict"] = "ERROR"
        result["error"] = str(e)
        log.error("STAGE 6: Forced photometry failed: %s", e)

    return result


# ── Diagnostic plot ──────────────────────────────────────────────────────────

def make_diagnostic_png(report: dict, output_path: str) -> None:
    """Generate a single-page diagnostic PNG summarizing the validation."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        log.warning("matplotlib not available — skipping diagnostic PNG")
        return

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    date = report["date"]
    cal_date = report["cal_assessment"]["cal_date"]
    fig.suptitle(f"DSA-110 Validation: {date}  (cal: {cal_date})", fontsize=14, fontweight="bold")

    # Panel 1: Status summary
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis("off")
    stages = [
        ("Data Inventory", report["data_inventory"]["verdict"]),
        ("Cal Assessment", report["cal_assessment"]["verdict"]),
        ("Tile Check", report["tile_check"]["verdict"]),
        ("Mosaic Check", report["mosaic_check"]["verdict"]),
    ]
    if "source_finding" in report:
        stages.append(("Source Finding", report["source_finding"]["verdict"]))
    if "flux_scale" in report:
        stages.append(("Flux Scale", report["flux_scale"]["verdict"]))

    colors = {"OK": "#2ecc71", "GOOD": "#2ecc71", "WARN": "#f39c12", "NOTE": "#3498db",
              "FAIL": "#e74c3c", "ERROR": "#e74c3c", "NONE": "#95a5a6",
              "UNAVAILABLE": "#95a5a6", "ACCEPTABLE": "#f39c12"}

    for i, (name, verdict) in enumerate(stages):
        y = 0.9 - i * 0.13
        color = colors.get(verdict, "#95a5a6")
        ax1.text(0.05, y, f"  {verdict:6s}", fontsize=11, fontfamily="monospace",
                 color="white", backgroundcolor=color,
                 transform=ax1.transAxes, verticalalignment="center",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.9))
        ax1.text(0.45, y, name, fontsize=11, transform=ax1.transAxes,
                 verticalalignment="center")
    ax1.set_title("Stage Verdicts", fontsize=12)

    # Panel 2: Mosaic image (if available)
    ax2 = fig.add_subplot(gs[0, 1:])
    mosaic_info = report["mosaic_check"].get("mosaics", [])
    if mosaic_info and "path" in mosaic_info[0] and "error" not in mosaic_info[0]:
        try:
            with fits.open(mosaic_info[0]["path"]) as hdul:
                data = hdul[0].data.squeeze()
            finite = data[np.isfinite(data)]
            vmin = float(np.percentile(finite, 1))
            vmax = float(np.percentile(finite, 99.5))
            ax2.imshow(data, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax, aspect="auto")
            ax2.set_title(f"{mosaic_info[0]['name']}\n"
                          f"Peak={mosaic_info[0]['peak_jyb']:.3f} Jy/beam  "
                          f"RMS={mosaic_info[0]['rms_mjyb']:.1f} mJy/beam  "
                          f"DR={mosaic_info[0]['dynamic_range']:.0f}", fontsize=10)
        except Exception:
            ax2.text(0.5, 0.5, "Could not display mosaic", transform=ax2.transAxes,
                     ha="center", va="center")
    else:
        ax2.text(0.5, 0.5, "No mosaic available", transform=ax2.transAxes,
                 ha="center", va="center")
        ax2.set_title("Mosaic Image", fontsize=12)

    # Panel 3: Tile RMS distribution
    ax3 = fig.add_subplot(gs[1, 0])
    tile_stats = report["tile_check"].get("tile_stats", [])
    if tile_stats:
        rms_vals = [t["rms_mjyb"] for t in tile_stats]
        ax3.hist(rms_vals, bins=min(20, len(rms_vals)), color="#3498db", edgecolor="white")
        ax3.axvline(np.median(rms_vals), color="red", linestyle="--", label=f"median={np.median(rms_vals):.1f}")
        ax3.set_xlabel("Tile RMS (mJy/beam)")
        ax3.set_ylabel("Count")
        ax3.legend(fontsize=9)
    ax3.set_title("Tile RMS Distribution", fontsize=12)

    # Panel 4: Flux ratio histogram (if available)
    ax4 = fig.add_subplot(gs[1, 1])
    if "flux_scale" in report and "csv_path" in report["flux_scale"]:
        try:
            import csv as csv_mod
            with open(report["flux_scale"]["csv_path"]) as f:
                rows = list(csv_mod.DictReader(f))
            ratios = [float(r["flux_ratio"]) for r in rows
                      if r.get("flux_ratio") and r["flux_ratio"] != ""
                      and 0.01 < float(r["flux_ratio"]) < 10.0]
            if ratios:
                ax4.hist(ratios, bins=50, color="#2ecc71", edgecolor="white", range=(0, 3))
                med = float(np.median(ratios))
                ax4.axvline(med, color="red", linestyle="--", label=f"median={med:.3f}")
                ax4.axvline(1.0, color="black", linestyle=":", alpha=0.5, label="unity")
                ax4.set_xlabel("DSA / Catalog Flux Ratio")
                ax4.set_ylabel("Count")
                ax4.legend(fontsize=9)
        except Exception:
            pass
    ax4.set_title("Flux Scale (Forced Photometry)", fontsize=12)

    # Panel 5: Cal quality summary
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    cal = report["cal_assessment"]
    lines = [f"Cal date: {cal['cal_date']}"]
    if cal.get("cross_date"):
        lines.append("** CROSS-DATE CALIBRATION **")
    for label in ["bp", "g"]:
        m = cal.get(f"{label}_metrics", {})
        if m and "extraction_error" not in m:
            lines.append(f"{label.upper()}: flag={m.get('flag_fraction', 0):.0%}  "
                         f"phase={m.get('phase_scatter_deg', 0):.1f}°  "
                         f"amp={m.get('amplitude_scatter_pct', 0):.1f}%")
    if cal.get("issues"):
        lines.append("")
        for issue in cal["issues"][:4]:
            lines.append(f"⚠ {issue}")
    text = "\n".join(lines)
    ax5.text(0.05, 0.95, text, fontsize=9, fontfamily="monospace",
             transform=ax5.transAxes, verticalalignment="top")
    ax5.set_title("Calibration Quality", fontsize=12)

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Diagnostic PNG: %s", output_path)


# ── Overall verdict ──────────────────────────────────────────────────────────

def compute_overall_verdict(report: dict) -> str:
    """Compute overall verdict from all stage verdicts."""
    verdicts = [
        report["data_inventory"]["verdict"],
        report["cal_assessment"]["verdict"],
        report["tile_check"]["verdict"],
        report["mosaic_check"]["verdict"],
    ]
    if "source_finding" in report:
        verdicts.append(report["source_finding"]["verdict"])
    if "flux_scale" in report:
        verdicts.append(report["flux_scale"]["verdict"])

    if any(v in ("FAIL", "ERROR") for v in verdicts):
        return "FAIL"
    if any(v in ("NONE", "UNAVAILABLE") for v in verdicts):
        return "INCOMPLETE"
    if any(v == "WARN" for v in verdicts):
        return "WARN"
    return "OK"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate a date's pipeline readiness and output quality."
    )
    parser.add_argument("--date", required=True, help="Observation date (YYYY-MM-DD)")
    parser.add_argument("--cal-date", default=None, help="Calibration date (default: same as --date)")
    parser.add_argument("--skip-source-finding", action="store_true",
                        help="Skip BANE + Aegean (faster, no blind catalog)")
    parser.add_argument("--skip-photometry", action="store_true",
                        help="Skip forced photometry flux scale check")
    parser.add_argument("--mosaic", default=None,
                        help="Specific mosaic to validate (default: auto-detect)")
    args = parser.parse_args()

    date = args.date
    cal_date = args.cal_date or date
    t0 = time.time()

    out_dir = os.path.join(OUTPUT_DIR, date)
    os.makedirs(out_dir, exist_ok=True)

    log.info("=" * 70)
    log.info("  DSA-110 Date Validation: %s  (cal: %s)", date, cal_date)
    log.info("=" * 70)

    report = {
        "date": date,
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Stage 1
    log.info("\n--- Stage 1: Data Inventory ---")
    report["data_inventory"] = check_ms_inventory(date)

    # Stage 2
    log.info("\n--- Stage 2: Cal Assessment ---")
    report["cal_assessment"] = check_cal_quality(date, cal_date)

    # Stage 3
    log.info("\n--- Stage 3: Tile Check ---")
    report["tile_check"] = check_tiles(date)

    # Stage 4
    log.info("\n--- Stage 4: Mosaic Check ---")
    report["mosaic_check"] = check_mosaics(date)

    # Pick mosaic for stages 5–6
    mosaic_path = args.mosaic
    if mosaic_path is None:
        mosaic_candidates = report["mosaic_check"].get("mosaics", [])
        for m in mosaic_candidates:
            if "error" not in m and "path" in m:
                mosaic_path = m["path"]
                break

    # Stage 5
    if mosaic_path and not args.skip_source_finding:
        log.info("\n--- Stage 5: Source Finding ---")
        report["source_finding"] = run_source_finding(mosaic_path, out_dir)

    # Stage 6
    if mosaic_path and not args.skip_photometry:
        log.info("\n--- Stage 6: Flux Scale ---")
        report["flux_scale"] = check_flux_scale(mosaic_path, out_dir)

    # Overall
    report["overall_verdict"] = compute_overall_verdict(report)
    report["wall_time_sec"] = round(time.time() - t0, 1)

    # Save report JSON
    report_path = os.path.join(out_dir, f"{date}_validation.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("\nReport written: %s", report_path)

    # Diagnostic PNG
    png_path = os.path.join(out_dir, f"{date}_validation.png")
    make_diagnostic_png(report, png_path)

    # Print summary
    log.info("\n" + "=" * 70)
    log.info("  OVERALL VERDICT: %s", report["overall_verdict"])
    log.info("=" * 70)
    for stage_name in ["data_inventory", "cal_assessment", "tile_check",
                       "mosaic_check", "source_finding", "flux_scale"]:
        if stage_name in report:
            v = report[stage_name]["verdict"]
            log.info("  %-20s %s", stage_name.replace("_", " ").title(), v)
    log.info("  Wall time: %.0fs", report["wall_time_sec"])
    log.info("  Report: %s", report_path)
    log.info("  Diagnostic: %s", png_path)

    return report


if __name__ == "__main__":
    main()
