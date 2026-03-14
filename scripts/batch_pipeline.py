#!/opt/miniforge/envs/casa6/bin/python
"""
Batch pipeline: calibrate → image → hourly-epoch mosaics → forced photometry.

Usage:
    python scripts/batch_pipeline.py [--date DATE] [--cal-date DATE] [--keep-intermediates] [--skip-photometry]

Steps:
    1. Find all valid MS files for DATE and process each one:
       phaseshift → applycal → WSClean image  (skip if tile FITS already exists)
    2. Bin tile images into 1-hour epochs by observation timestamp.
       Each epoch mosaic also includes the last 2 tiles from the previous epoch
       and the first 2 tiles from the next epoch (~4-tile / ~20-min overlap).
       The first and last epochs of the day have overlap on one side only.
    3. For each epoch (skip if output mosaic already exists):
       a. Build mosaic FITS  →  {stage}/mosaic_{date}/{date}T{HH}00_mosaic.fits
       b. Run QA (noise consistency)
       c. Forced photometry against master catalog → {products}/{date}T{HH}00_forced_phot.csv
    4. Print per-epoch summary + overall totals.

Output layout (after mosaic move to products/):
    /data/dsa110-continuum/products/mosaics/{date}/
        {date}T{HH}00_mosaic.fits
        {date}T{HH}00_forced_phot.csv
        ...
"""
import argparse
import csv
import json
import logging
import os
import shutil
import sys
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from pathlib import Path

# ── Load scripts/.env before anything else ───────────────────────────────────
_ENV_FILE = Path(__file__).parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if _line.startswith("export "):
            _line = _line[len("export "):]
        if "=" in _line and not _line.startswith("#"):
            _key, _, _val = _line.partition("=")
            os.environ.setdefault(_key.strip(), _val.strip())

# ── Project root + scripts/ on path ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))  # enables `import mosaic_day`

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from dsa110_continuum.photometry.epoch_qa import EpochQAResult, measure_epoch_qa
from dsa110_continuum.photometry.epoch_qa_plot import plot_epoch_qa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_DATE = "2026-01-25"
MS_DIR = os.environ.get("DSA110_MS_DIR", "/stage/dsa110-contimg/ms")
STAGE_IMAGE_BASE = os.environ.get("DSA110_STAGE_IMAGE_BASE", "/stage/dsa110-contimg/images")
PRODUCTS_BASE = os.environ.get("DSA110_PRODUCTS_BASE", "/data/dsa110-proc/products/mosaics")
CELL_ARCSEC = 6.0  # must match mosaic_day.py
TILE_TIMEOUT_SEC = 1800  # 30 min max per tile before we kill & skip

# QA summary CSV schema (expanded for three-gate epoch QA)
QA_SUMMARY_CSV = os.environ.get(
    "DSA110_QA_SUMMARY",
    "/data/dsa110-proc/products/qa_summary.csv",
)
QA_CSV_FIELDS = [
    "date", "epoch_utc", "mosaic_path",
    "n_catalog", "n_recovered", "completeness_frac",
    "median_ratio", "ratio_gate", "completeness_gate",
    "rms_gate", "mosaic_rms_mjy",
    "qa_result", "gaincal_used",
]


def get_paths(date: str) -> dict:
    return {
        "ms_dir": MS_DIR,
        "stage_dir": f"{STAGE_IMAGE_BASE}/mosaic_{date}",
        "products_dir": f"{PRODUCTS_BASE}/{date}",
    }


def epoch_mosaic_path(paths: dict, date: str, hour: int) -> str:
    return f"{paths['stage_dir']}/{date}T{hour:02d}00_mosaic.fits"


def epoch_phot_path(paths: dict, date: str, hour: int) -> str:
    return f"{paths['products_dir']}/{date}T{hour:02d}00_forced_phot.csv"


# ── Timestamp parsing ─────────────────────────────────────────────────────────

def timestamp_from_fits(fits_path: str) -> datetime | None:
    """Extract UTC datetime from a tile FITS path like .../2026-01-25T21:17:33-image-pb.fits."""
    name = Path(fits_path).name  # e.g. 2026-01-25T21:17:33-image-pb.fits
    ts_str = name.split("-image")[0]  # e.g. 2026-01-25T21:17:33
    try:
        return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# ── Epoch binning ─────────────────────────────────────────────────────────────

def bin_tiles_by_hour(tile_fits: list[str]) -> dict[int, list[str]]:
    """Group tile FITS paths by the UTC hour of their observation timestamp.

    Returns a dict mapping hour (0–23) → sorted list of tile paths.
    Tiles whose timestamp cannot be parsed are dropped with a warning.
    """
    epochs: dict[int, list[str]] = {}
    for path in tile_fits:
        dt = timestamp_from_fits(path)
        if dt is None:
            log.warning("Cannot parse timestamp from %s — skipping", Path(path).name)
            continue
        h = dt.hour
        epochs.setdefault(h, []).append(path)
    # Sort within each epoch (tiles arrive in time order within an hour)
    for h in epochs:
        epochs[h].sort()
    return epochs


def build_epoch_tile_sets(epochs: dict[int, list[str]]) -> list[tuple[int, list[str]]]:
    """Return list of (hour, tiles_with_overlap) in chronological order.

    Each epoch's tile list is:
        last 2 tiles of previous epoch  (if exists)
      + all tiles for this epoch
      + first 2 tiles of next epoch     (if exists)

    The epoch hour label is the start of the 1-hour window; the center is hour+0.5h.
    """
    sorted_hours = sorted(epochs.keys())
    result = []
    for i, h in enumerate(sorted_hours):
        tiles = list(epochs[h])  # core tiles

        # Previous-epoch overlap
        if i > 0:
            prev_tiles = epochs[sorted_hours[i - 1]]
            tiles = prev_tiles[-2:] + tiles

        # Next-epoch overlap
        if i < len(sorted_hours) - 1:
            next_tiles = epochs[sorted_hours[i + 1]]
            tiles = tiles + next_tiles[:2]

        result.append((h, tiles))
    return result


# ── Per-epoch mosaic writer (path-explicit version of mosaic_day.write_mosaic) ─

def write_epoch_mosaic(
    mosaic: np.ndarray,
    out_wcs: WCS,
    ref_fits_paths: list[str],
    out_path: str,
    date: str,
    hour: int,
    n_tiles: int,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with fits.open(ref_fits_paths[0]) as ref:
        ref_hdr = ref[0].header.copy()

    new_hdr = fits.Header()
    new_hdr["SIMPLE"] = True
    new_hdr["BITPIX"] = -32
    new_hdr["NAXIS"] = 2
    new_hdr["NAXIS1"] = mosaic.shape[1]
    new_hdr["NAXIS2"] = mosaic.shape[0]
    for key in ("BMAJ", "BMIN", "BPA", "BUNIT", "RESTFRQ", "EQUINOX"):
        if key in ref_hdr:
            new_hdr[key] = ref_hdr[key]
    new_hdr.update(out_wcs.to_header())
    new_hdr["HISTORY"] = (
        f"DSA-110 hourly mosaic {date}T{hour:02d}:00 UTC, {n_tiles} tiles (with overlap)"
    )

    hdu = fits.PrimaryHDU(data=mosaic.astype(np.float32), header=new_hdr)
    hdu.writeto(out_path, overwrite=True)
    log.info("Epoch mosaic written: %s", out_path)


# ── Forced photometry (delegates to forced_photometry.run_forced_photometry) ──


# ── Mosaic stats helper ───────────────────────────────────────────────────────

def mosaic_stats(mosaic_path: str) -> tuple[float, float]:
    """Return (peak_jyb, rms_jyb) for a FITS mosaic."""
    with fits.open(mosaic_path) as hdul:
        data = hdul[0].data.squeeze()
    finite = data[np.isfinite(data)]
    peak = float(np.nanmax(data))
    rms = float(1.4826 * np.nanmedian(np.abs(finite - np.nanmedian(finite))))
    return peak, rms


# ── QA summary CSV ────────────────────────────────────────────────────────────

def write_qa_summary_row(
    date: str,
    epoch_label: str,
    mosaic_path: str,
    qa: EpochQAResult | None,
    gaincal_status: str,
) -> None:
    """Append one row to the QA summary CSV, creating the file if needed."""
    row = {
        "date": date,
        "epoch_utc": epoch_label,
        "mosaic_path": mosaic_path,
        "gaincal_used": gaincal_status,
    }
    if qa is not None:
        row.update(qa.to_dict())
    else:
        for field in QA_CSV_FIELDS:
            row.setdefault(field, "")

    file_exists = os.path.isfile(QA_SUMMARY_CSV)
    os.makedirs(os.path.dirname(QA_SUMMARY_CSV), exist_ok=True)
    try:
        with open(QA_SUMMARY_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=QA_CSV_FIELDS, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        log.warning("Could not write QA summary row: %s", e)


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(date: str, epoch_results: list[dict]) -> None:
    """Print a per-epoch table plus overall totals."""
    print("\n" + "=" * 88)
    print(f"  DSA-110 Batch Pipeline Summary — {date}")
    print("=" * 88)
    hdr = (
        f"  {'Epoch':12s}  {'Tiles':>5}  {'GainCal':>8}"
        f"  {'Peak(Jy/b)':>10}  {'RMS(mJy/b)':>10}  {'Sources':>7}  {'DSA/Cat':>8}  {'QA':>4}"
    )
    print(hdr)
    print("  " + "-" * 84)

    all_ratios: list[float] = []
    for r in epoch_results:
        status = r.get("status", "ok")
        gcal_str = r.get("gaincal_status", "n/a")[:8]
        if status == "skipped":
            print(f"  {r['label']:12s}  {'--':>5}  {gcal_str:>8}  {'(skipped)':>10}")
            continue
        if status == "failed":
            print(f"  {r['label']:12s}  {r['n_tiles']:>5}  {gcal_str:>8}  {'FAILED':>10}")
            continue

        peak_str = f"{r['peak']:.4f}" if r['peak'] is not None else "n/a"
        rms_str = f"{r['rms']*1000:.2f}" if r['rms'] is not None else "n/a"
        src_str = str(r['n_sources']) if r['n_sources'] is not None else "n/a"
        ratio = r.get('median_ratio')
        ratio_str = f"{ratio:.3f}" if ratio is not None else "n/a"
        if ratio is not None:
            all_ratios.append(ratio)

        qa_str = r.get("qa_result") or "n/a"
        print(
            f"  {r['label']:12s}  {r['n_tiles']:>5}  {gcal_str:>8}"
            f"  {peak_str:>10}  {rms_str:>10}  {src_str:>7}  {ratio_str:>8}  {qa_str:>4}"
        )

    print("  " + "-" * 84)
    if all_ratios:
        overall = float(np.median(all_ratios))
        flag = "  OK" if 0.8 <= overall <= 1.2 else "  WARNING: outside 0.8–1.2 target"
        print(f"  Median DSA/Cat ratio across all epochs: {overall:.3f}{flag}")
    total_tiles = sum(r.get("n_tiles", 0) for r in epoch_results if r.get("status") != "skipped")
    n_epochs = len(epoch_results)
    n_skipped = sum(1 for r in epoch_results if r.get("status") == "skipped")
    n_failed = sum(1 for r in epoch_results if r.get("status") == "failed")
    print(f"  Epochs: {n_epochs} total, {n_skipped} skipped, {n_failed} failed")
    # QA aggregate counts (separate from execution success/failure)
    qa_pass = sum(1 for r in epoch_results if r.get("qa_result") == "PASS")
    qa_fail = sum(1 for r in epoch_results if r.get("qa_result") == "FAIL")
    qa_none = n_epochs - qa_pass - qa_fail
    qa_parts = [f"{qa_pass} QA-pass", f"{qa_fail} QA-fail"]
    if qa_none:
        qa_parts.append(f"{qa_none} QA-unavailable")
    print(f"  QA:     {', '.join(qa_parts)}")
    print("=" * 78 + "\n")



# ── Tile execution: timeout + retry ──────────────────────────────────────────

def _run_process_ms(
    ms_path: str,
    keep: bool,
    force_recal: bool = False,
    g_table: str | None = None,
    bp_table: str | None = None,
) -> str | None:
    """Thin wrapper so process_ms can be submitted to a subprocess pool.

    Accepts optional *g_table* and *bp_table* so that the epoch-derived G table
    (set on the parent process's _md module) is propagated into the fresh
    ``mosaic_day`` import that runs inside the subprocess.
    """
    import mosaic_day as _md
    if g_table is not None:
        _md.G_TABLE = g_table
    if bp_table is not None:
        _md.BP_TABLE = bp_table
    return _md.process_ms(ms_path, keep_intermediates=keep, force_recal=force_recal)


def process_tile_safe(
    md,
    ms_path: str,
    keep: bool,
    timeout_sec: int,
    retry: bool,
    force_recal: bool = False,
    g_table: str | None = None,
    bp_table: str | None = None,
) -> str | None:
    """Run md.process_ms with a hard timeout and optional single retry.

    If the tile hangs beyond *timeout_sec*, any CASA/WSClean subprocesses are
    killed with SIGKILL and None is returned.  With *retry=True* a second
    attempt is made after a 60-second cool-down.
    """
    tag = Path(ms_path).stem

    def _attempt() -> str | None:
        with ProcessPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_run_process_ms, ms_path, keep, force_recal, g_table, bp_table)
            try:
                return fut.result(timeout=timeout_sec)
            except FuturesTimeoutError:
                log.error("[%s] TIMEOUT after %ds — killing CASA/WSClean", tag, timeout_sec)
                for pattern in ["applycal", "wsclean", "mpicasa"]:
                    subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True)
                return None

    result = _attempt()
    if result is None and retry:
        log.warning("[%s] First attempt failed — waiting 60s then retrying once", tag)
        time.sleep(60)
        result = _attempt()
        if result is None:
            log.error("[%s] Retry also failed — skipping tile", tag)
    return result


# ── Dec-strip guard ───────────────────────────────────────────────────────────

def check_dec_strip(
    observed_dec: float,
    expected_dec: float,
    threshold_deg: float = 5.0,
) -> None:
    """Abort if the observed Dec strip differs from the expected calibration strip.

    DSA-110 observes at different declination strips on different nights. Calibration
    tables are strip-specific — applying tables from one strip to another silently
    produces near-zero flux (confirmed: median DSA/NVSS ≈ 0.06 for cross-strip runs).
    """
    delta = abs(observed_dec - expected_dec)
    if delta > threshold_deg:
        log.error(
            "ABORT: observed Dec %.1f° differs from expected %.1f° by %.1f° "
            "(threshold %.1f°). Cal tables were derived at Dec≈%.1f° — "
            "applying them at Dec≈%.1f° will produce invalid flux scale. "
            "Re-run with --expected-dec %.1f once cal tables for that strip exist.",
            observed_dec, expected_dec, delta, threshold_deg,
            expected_dec, observed_dec, observed_dec,
        )
        sys.exit(1)
    log.info(
        "Dec-strip check passed: observed %.1f° vs expected %.1f° (Δ=%.1f°)",
        observed_dec, expected_dec, delta,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _main_start = time.time()
    parser = argparse.ArgumentParser(
        description="Hourly-epoch mosaic pipeline for DSA-110 drift observations."
    )
    parser.add_argument("--date", default=DEFAULT_DATE, help="Observation date (YYYY-MM-DD)")
    parser.add_argument(
        "--cal-date",
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "Date whose calibration tables (BP/gain) to use. "
            "Defaults to --date if not provided. "
            "Use this when processing a new date whose cal tables are symlinked "
            "from 2026-01-25 (see CLAUDE.md)."
        ),
    )
    parser.add_argument(
        "--expected-dec",
        type=float,
        default=16.1,
        metavar="DEG",
        help=(
            "Expected pointing declination for this cal-table strip (default: 16.1°). "
            "Pipeline aborts if the first MS differs by more than 5° (DEC_CHANGE_THRESHOLD_DEG). "
            "Set this explicitly when processing a non-default Dec strip once cal tables exist."
        ),
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        default=False,
        help="Keep *_meridian.ms files (useful for debugging).",
    )
    parser.add_argument(
        "--skip-photometry",
        action="store_true",
        default=False,
        help="Skip forced photometry step.",
    )
    parser.add_argument(
        "--skip-epoch-gaincal",
        action="store_true",
        default=False,
        help=(
            "Skip per-epoch gain calibration. "
            "Falls back to the static daily G table (--cal-date). "
            "Use for debugging or when cal tables already exist."
        ),
    )
    parser.add_argument(
        "--start-hour",
        type=int,
        default=None,
        metavar="H",
        help="Only process MS files with timestamp >= this UTC hour (0–23). Default: all hours.",
    )
    parser.add_argument(
        "--end-hour",
        type=int,
        default=None,
        metavar="H",
        help="Only process MS files with timestamp < this UTC hour (0–23). Default: all hours.",
    )
    parser.add_argument(
        "--tile-timeout",
        type=int,
        default=TILE_TIMEOUT_SEC,
        metavar="SECONDS",
        help=f"Hard timeout per tile (applycal + WSClean). Default: {TILE_TIMEOUT_SEC}s (30 min). "
             "If a tile exceeds this, CASA/WSClean are killed and the tile is skipped (or retried).",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        default=False,
        help="Retry each failed tile once (60s cool-down between attempts). "
             "Useful for transient CASA crashes or memory pressure.",
    )
    parser.add_argument(
        "--force-recal",
        action="store_true",
        default=False,
        help=(
            "Force full re-calibration and re-imaging of every tile, even when FITS outputs "
            "already exist. Also clears the epoch_gaincal ap.G cache so the fallback check "
            "runs fresh. Use when re-running a date after code changes (e.g. BP-only fallback)."
        ),
    )
    args = parser.parse_args()

    date = args.date
    cal_date = args.cal_date if args.cal_date is not None else date

    # ── Cal-table validation ───────────────────────────────────────────────
    _bp = f"{MS_DIR}/{cal_date}T22:26:05_0~23.b"
    _ga = f"{MS_DIR}/{cal_date}T22:26:05_0~23.g"
    _missing = [t for t in [_bp, _ga] if not os.path.exists(t)]
    if _missing:
        for _t in _missing:
            log.error("ABORT: calibration table not found: %s", _t)
        log.error("Available .b tables in %s:", MS_DIR)
        for _f in sorted(os.listdir(MS_DIR)):
            if _f.endswith(".b"):
                log.error("  %s", _f)
        sys.exit(1)
    log.info("Cal tables verified for %s", cal_date)

    # ── Dec-strip validation ───────────────────────────────────────────────────
    from dsa110_continuum.calibration.dec_utils import read_ms_dec as _read_ms_dec
    _first_ms_list = sorted(
        f for f in os.listdir(MS_DIR)
        if f.endswith(".ms") and f.startswith(date) and "meridian" not in f
    )
    if _first_ms_list:
        try:
            _obs_dec = _read_ms_dec(os.path.join(MS_DIR, _first_ms_list[0]))
            check_dec_strip(_obs_dec, args.expected_dec)
        except RuntimeError as _e:
            log.warning("Could not determine observed Dec (%s) — skipping dec-strip check", _e)
    else:
        log.warning("No MS files found for %s yet — dec-strip check skipped", date)
    # ─────────────────────────────────────────────────────────────────────────

    # ───────────────────────────────────────────────────────────────────────
    keep = args.keep_intermediates
    paths = get_paths(date)

    log.info("=== DSA-110 Batch Pipeline — %s ===", date)
    if cal_date != date:
        log.info("Calibration tables from: %s", cal_date)
    log.info("Stage dir:    %s", paths["stage_dir"])
    log.info("Products dir: %s", paths["products_dir"])

    os.makedirs(paths["stage_dir"], exist_ok=True)
    os.makedirs(paths["products_dir"], exist_ok=True)

    # ── Migrate stale qa_summary.csv if schema doesn't match ─────────────────
    if os.path.isfile(QA_SUMMARY_CSV):
        try:
            with open(QA_SUMMARY_CSV) as _f:
                existing_header = _f.readline().strip()
            expected_header = ",".join(QA_CSV_FIELDS)
            if existing_header != expected_header:
                archive_path = QA_SUMMARY_CSV + ".pre_phase0.bak"
                if not os.path.exists(archive_path):
                    shutil.copy2(QA_SUMMARY_CSV, archive_path)
                    log.info("Archived old qa_summary.csv to %s", archive_path)
                os.remove(QA_SUMMARY_CSV)
                log.info("Removed stale qa_summary.csv (old schema)")
        except Exception as e:
            log.warning("Could not check/migrate qa_summary.csv: %s", e)

    # ── Import mosaic_day and patch constants for this date ───────────────────
    import mosaic_day as _md  # type: ignore  (scripts/ is on sys.path)

    _md.DATE = date
    _md.IMAGE_DIR = paths["stage_dir"]
    _md.MOSAIC_OUT = f"{paths['stage_dir']}/full_mosaic.fits"  # not used, but keeps _md consistent
    _md.PRODUCTS_DIR = paths["products_dir"]
    _md.BP_TABLE = f"{MS_DIR}/{cal_date}T22:26:05_0~23.b"
    _md.G_TABLE = f"{MS_DIR}/{cal_date}T22:26:05_0~23.g"

    # ── Phase 1: Find + validate MS files ────────────────────────────────────
    ms_list = _md.find_valid_ms()
    if not ms_list:
        log.error("No valid MS files found for %s — aborting", date)
        sys.exit(1)
    log.info("Found %d valid MS files", len(ms_list))

    # Apply date filter (--date), then --start-hour / --end-hour
    def _ms_ts(ms_path: str):
        ts_str = Path(ms_path).stem  # e.g. 2026-01-25T21:17:33
        try:
            return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return None

    before = len(ms_list)
    ms_list = [p for p in ms_list if (t := _ms_ts(p)) is not None and t.strftime("%Y-%m-%d") == date]
    log.info("Date filter (%s): %d → %d MS files", date, before, len(ms_list))
    if not ms_list:
        log.error("No MS files for date %s — aborting", date)
        sys.exit(1)

    start_hour = args.start_hour
    end_hour = args.end_hour
    if start_hour is not None or end_hour is not None:
        before = len(ms_list)
        ms_list = [
            p for p in ms_list
            if (t := _ms_ts(p)) is not None
            and (start_hour is None or t.hour >= start_hour)
            and (end_hour is None or t.hour < end_hour)
        ]
        log.info(
            "Hour filter [%s, %s): %d → %d MS files",
            f"{start_hour:02d}" if start_hour is not None else "*",
            f"{end_hour:02d}" if end_hour is not None else "*",
            before,
            len(ms_list),
        )
        if not ms_list:
            log.error("No MS files remain after hour filter — aborting")
            sys.exit(1)

    # ── Phase 0: Per-epoch gain calibration ───────────────────────────────────
    epoch_gaincal_dir = os.path.join(paths["stage_dir"], "epoch_gaincal")
    os.makedirs(epoch_gaincal_dir, exist_ok=True)

    # --force-recal: purge cached gaincal products so the solve runs fresh
    if args.force_recal:
        import glob as _glob
        _cached_ap = _glob.glob(os.path.join(epoch_gaincal_dir, "*.ap.G"))
        for _f in _cached_ap:
            try:
                import shutil
                shutil.rmtree(_f) if os.path.isdir(_f) else os.remove(_f)
                log.info("--force-recal: removed cached ap.G: %s", _f)
            except Exception as _e:
                log.warning("--force-recal: could not remove %s: %s", _f, _e)

        # Purge stale per-tile *_meridian.ms intermediate Measurement Sets so
        # that a corrupt MS cannot cause applycal to fail on the next run.
        _stale_meridian = _glob.glob(os.path.join(MS_DIR, f"{date}*_meridian.ms"))
        for _p in _stale_meridian:
            try:
                shutil.rmtree(_p)
                log.info("--force-recal: removed stale meridian MS: %s", _p)
            except Exception as _e:
                log.warning("--force-recal: could not remove meridian MS %s: %s", _p, _e)

    _epoch_g_table: str | None = None
    if not args.skip_epoch_gaincal:
        log.info("=== Phase 0/3: Per-epoch gain calibration ===")
        try:
            from dsa110_continuum.calibration.epoch_gaincal import calibrate_epoch
            from dsa110_continuum.calibration.mosaic_constants import MOSAIC_TILE_COUNT
            _epoch_ms = ms_list[:MOSAIC_TILE_COUNT] if len(ms_list) >= MOSAIC_TILE_COUNT else ms_list
            if len(_epoch_ms) >= 2:
                _epoch_g_table = calibrate_epoch(
                    epoch_ms_paths=_epoch_ms,
                    bp_table=_bp,
                    work_dir=epoch_gaincal_dir,
                    refant="103",
                )
                if _epoch_g_table is not None:
                    log.info("Epoch gaincal SUCCESS: %s", _epoch_g_table)
                    _md.G_TABLE = _epoch_g_table
                    _epoch_gaincal_status = "ok"
                else:
                    log.warning(
                        "Epoch gaincal failed — falling back to static daily G table (%s)", _ga
                    )
                    _epoch_gaincal_status = "fallback"
            else:
                log.warning(
                    "Epoch gaincal skipped: need at least 2 MS files, found %d", len(_epoch_ms)
                )
                _epoch_gaincal_status = "skipped"
        except Exception as _eg_exc:
            log.error("Epoch gaincal error: %s — using static table", _eg_exc)
            _epoch_gaincal_status = "error"
    else:
        log.info("--skip-epoch-gaincal set: using static daily G table (%s)", _ga)
        _epoch_gaincal_status = "skipped"
    # ──────────────────────────────────────────────────────────────────────────

    # ── Phase 1: Calibrate + image all tiles ──────────────────────────────────
    tile_timeout = args.tile_timeout
    retry_failed = args.retry_failed
    checkpoint_path = os.path.join(paths["stage_dir"], ".tile_checkpoint.json")

    # --force-recal means "fresh rerun", so discard any old checkpoint state
    if args.force_recal and os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            log.info("--force-recal: removed stale checkpoint: %s", checkpoint_path)
        except Exception as e:
            log.warning("--force-recal: could not remove checkpoint %s: %s", checkpoint_path, e)

    # Load completed tiles from a previous (crashed) run
    tile_fits: list[str] = []
    if os.path.exists(checkpoint_path):
        try:
            ck = json.load(open(checkpoint_path))
            tile_fits = [p for p in ck.get("completed", []) if os.path.exists(p)]
            if tile_fits:
                log.info("Checkpoint: resuming with %d previously completed tiles", len(tile_fits))
        except Exception as e:
            log.warning("Could not read checkpoint file: %s", e)

    completed_fits = set(tile_fits)
    log.info("=== Phase 1/3: Calibrate + Image all tiles (timeout=%ds, retry=%s) ===",
             tile_timeout, retry_failed)
    n_imaged = n_skipped_tiles = n_failed_tiles = 0

    for i, ms_path in enumerate(ms_list, 1):
        tag = Path(ms_path).stem
        log.info("[%d/%d] %s", i, len(ms_list), tag)
        t0 = time.time()
        result = process_tile_safe(
            _md, ms_path, keep, tile_timeout, retry_failed,
            force_recal=(args.force_recal or _epoch_gaincal_status == "ok"),
            g_table=_epoch_g_table,
            bp_table=_bp,
        )
        elapsed = time.time() - t0

        if result is None:
            log.error("  FAILED after %.0fs", elapsed)
            n_failed_tiles += 1
        else:
            if result not in completed_fits:
                tile_fits.append(result)
                completed_fits.add(result)
            if elapsed < 2.0:
                n_skipped_tiles += 1
            else:
                log.info("  Done in %.0fs → %s", elapsed, Path(result).name)
                n_imaged += 1

            # Write checkpoint after every successful tile
            try:
                with open(checkpoint_path, "w") as ck_f:
                    json.dump({"date": date, "cal_date": cal_date, "completed": tile_fits}, ck_f, indent=2)
            except Exception as e:
                log.warning("Could not write checkpoint: %s", e)

    log.info(
        "Tiles: %d imaged, %d already done, %d failed",
        n_imaged, n_skipped_tiles, n_failed_tiles,
    )

    if len(tile_fits) < 2:
        log.error("Too few tiles to mosaic (%d) — aborting", len(tile_fits))
        sys.exit(1)

    # ── Phase 3: Bin tiles into 1-hour epochs with overlap ────────────────────
    log.info("=== Phase 2/3: Build hourly-epoch mosaics ===")
    by_hour = bin_tiles_by_hour(tile_fits)
    epoch_sets = build_epoch_tile_sets(by_hour)  # [(hour, [tile_paths...]), ...]

    log.info(
        "Epochs: %d  (hours: %s)",
        len(epoch_sets),
        ", ".join(f"{h:02d}h" for h, _ in epoch_sets),
    )

    epoch_results: list[dict] = []

    for hour, epoch_tiles in epoch_sets:
        label = f"{date}T{hour:02d}00"
        n_core = len(by_hour.get(hour, []))
        n_overlap = len(epoch_tiles) - n_core
        log.info(
            "--- Epoch %s: %d core + %d overlap = %d tiles ---",
            label, n_core, n_overlap, len(epoch_tiles),
        )

        mosaic_path = epoch_mosaic_path(paths, date, hour)
        phot_csv_path = epoch_phot_path(paths, date, hour)
        mosaic_fits_dst = Path(paths["products_dir"]) / Path(mosaic_path).name

        # --force-recal: remove stale epoch-level outputs before rebuilding
        if args.force_recal:
            for stale_path in (mosaic_path, phot_csv_path, str(mosaic_fits_dst)):
                if os.path.exists(stale_path):
                    try:
                        os.remove(stale_path)
                        log.info("  --force-recal: removed stale output %s", stale_path)
                    except Exception as e:
                        log.warning("  --force-recal: could not remove %s: %s", stale_path, e)

        # Skip if mosaic already exists (unless --force-recal)
        if os.path.exists(mosaic_path) and not args.force_recal:
            log.info("  Mosaic already exists — skipping epoch %s", label)
            epoch_results.append({"label": label, "status": "skipped", "n_tiles": len(epoch_tiles), "gaincal_status": _epoch_gaincal_status})
            continue

        # Build mosaic
        try:
            out_wcs, ny, nx = _md.build_common_wcs(epoch_tiles)
            mosaic_arr = _md.coadd_tiles(epoch_tiles, out_wcs, ny, nx)
            write_epoch_mosaic(mosaic_arr, out_wcs, epoch_tiles, mosaic_path, date, hour, len(epoch_tiles))
        except Exception as e:
            log.error("  Mosaic failed for epoch %s: %s", label, e)
            epoch_results.append({"label": label, "status": "failed", "n_tiles": len(epoch_tiles), "gaincal_status": _epoch_gaincal_status})
            continue

        # QA
        _md.check_mosaic_quality(mosaic_path)
        peak, rms = mosaic_stats(mosaic_path)
        log.info("  Peak: %.4f Jy/beam  RMS: %.2f mJy/beam  DR: %.0f", peak, rms * 1000, peak / rms if rms else 0)

        # Forced photometry
        n_sources: int | None = None
        median_ratio: float | None = None

        if not args.skip_photometry:
            try:
                from forced_photometry import run_forced_photometry
                phot_result = run_forced_photometry(
                    mosaic_path, output_csv=phot_csv_path, min_flux_mjy=10.0,
                )
                n_sources = phot_result["n_sources"]
                median_ratio = phot_result["median_ratio"]
                if np.isfinite(median_ratio):
                    log.info("  Median DSA/Cat ratio: %.3f  (%d sources)", median_ratio, n_sources)
            except Exception as e:
                log.error("  Forced photometry failed: %s", e)

        # ── Epoch QA (three-gate) ──────────────────────────────────────────────
        epoch_qa: EpochQAResult | None = None
        try:
            epoch_qa = measure_epoch_qa(mosaic_path)
            log.info(
                "  Epoch QA: ratio=%.3f [%s] | compl=%.1f%% [%s] | rms=%.1f mJy [%s] → %s",
                epoch_qa.median_ratio, epoch_qa.ratio_gate,
                epoch_qa.completeness_frac * 100, epoch_qa.completeness_gate,
                epoch_qa.mosaic_rms_mjy, epoch_qa.rms_gate,
                epoch_qa.qa_result,
            )
        except Exception as e:
            log.warning("  Epoch QA failed: %s", e)

        # ── Diagnostic PNG ────────────────────────────────────────────────────
        if epoch_qa is not None:
            diag_png = mosaic_path.replace(".fits", "_qa_diag.png")
            try:
                tile_rms_list = []
                for tp in epoch_tiles:
                    if os.path.exists(tp):
                        _, trms = mosaic_stats(tp)
                        tile_rms_list.append(trms * 1000.0)
                plot_epoch_qa(
                    epoch_qa,
                    epoch_qa.ratios or [],
                    tile_rms_list,
                    diag_png,
                    epoch_label=label,
                )
                log.info("  QA diagnostic PNG: %s", diag_png)
            except Exception as e:
                log.warning("  Could not generate QA PNG: %s", e)

        # ── QA summary CSV row ────────────────────────────────────────────────
        try:
            write_qa_summary_row(date, label, mosaic_path, epoch_qa, _epoch_gaincal_status)
        except Exception as e:
            log.warning("  Could not write QA summary: %s", e)

        # Archive epoch mosaic FITS alongside CSV
        mosaic_fits_src = Path(mosaic_path)
        if mosaic_fits_src.exists() and (not mosaic_fits_dst.exists() or args.force_recal):
            shutil.copy2(str(mosaic_fits_src), str(mosaic_fits_dst))
            log.info("Archived mosaic FITS: %s", mosaic_fits_dst)

        epoch_results.append({
            "label": label,
            "status": "ok",
            "n_tiles": len(epoch_tiles),
            "n_core": n_core,
            "n_overlap": n_overlap,
            "peak": peak,
            "rms": rms,
            "n_sources": n_sources,
            "median_ratio": median_ratio,
            "mosaic_path": mosaic_path,
            "gaincal_status": _epoch_gaincal_status,
            "qa_result": epoch_qa.qa_result if epoch_qa else None,
        })

    # ── Print summary ─────────────────────────────────────────────────────────
    print_summary(date, epoch_results)


    emit_run_summary(date, cal_date, epoch_results, time.time() - _main_start)

def emit_run_summary(date: str, cal_date: str, epoch_results: list, wall_time_sec: float) -> None:
    """Write /tmp/pipeline_last_run.json and optionally POST to DSA_NOTIFY_URL."""
    import json as _json
    from datetime import datetime as _dt

    epochs_list = epoch_results
    n_exec_ok = sum(1 for v in epoch_results if v.get("status") == "ok")
    n_exec_fail = sum(1 for v in epoch_results if v.get("status") != "ok")
    n_qa_pass = sum(1 for v in epoch_results if v.get("qa_result") == "PASS")
    n_qa_fail = sum(1 for v in epoch_results if v.get("qa_result") == "FAIL")

    payload = {
        "date": date,
        "cal_date": cal_date,
        "finished_at": _dt.utcnow().isoformat() + "Z",
        "wall_time_sec": round(wall_time_sec),
        "n_epochs": len(epoch_results),
        "n_pass": n_exec_ok,
        "n_fail": n_exec_fail,
        "n_qa_pass": n_qa_pass,
        "n_qa_fail": n_qa_fail,
        "epochs": epochs_list,
    }

    summary_path = "/tmp/pipeline_last_run.json"
    with open(summary_path, "w") as _f:
        _json.dump(payload, _f, indent=2)

    log.info(
        "Run complete — date=%s cal=%s  epochs=%d  exec_ok=%d exec_fail=%d"
        "  qa_pass=%d qa_fail=%d  wall=%.0fm  → %s",
        date, cal_date, len(epoch_results), n_exec_ok, n_exec_fail,
        n_qa_pass, n_qa_fail, wall_time_sec / 60, summary_path,
    )

    notify_url = os.environ.get("DSA_NOTIFY_URL")
    if notify_url:
        try:
            import requests as _req
            _req.post(notify_url, json=payload, timeout=10)
        except Exception:
            pass  # notification is best-effort


if __name__ == "__main__":
    main()
