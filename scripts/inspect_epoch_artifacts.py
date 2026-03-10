#!/opt/miniforge/envs/casa6/bin/python
r"""Epoch artifact inspector — quickly verify what a pipeline run actually produced.

Given a date (and optionally an epoch hour), reports the presence, size, and
science-facing state of each expected pipeline artifact:

  • Epoch mosaic FITS        (products/ and/or stage/)
  • QA diagnostic PNG        (stage/, beside the mosaic)
  • Forced-photometry CSV    (products/)
  • qa_summary.csv row       (products/)
  • Gaincal table status     (present / borrowed / missing)
  • Basic FITS stats         (peak Jy/beam, RMS mJy/beam, NaN fraction)

This is a read-only diagnostic tool — it never writes or modifies anything.

Usage:
    # All epochs for a date
    python scripts/inspect_epoch_artifacts.py --date 2026-01-25

    # One epoch only (UTC hour as two digits)
    python scripts/inspect_epoch_artifacts.py --date 2026-01-25 --epoch 22

    # Override default search roots
    python scripts/inspect_epoch_artifacts.py --date 2026-01-25 \
        --products-dir /my/products --stage-dir /my/stage/images

    # Inspect one explicit mosaic FITS path
    python scripts/inspect_epoch_artifacts.py \
        --mosaic-path /data/dsa110-continuum/products/mosaics/2026-01-25/2026-01-25T2200_mosaic.fits
"""
import argparse
import csv
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# ── Default search roots ───────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PRODUCTS_BASE = REPO_ROOT / "products" / "mosaics"
DEFAULT_QA_SUMMARY = REPO_ROOT / "products" / "qa_summary.csv"
DEFAULT_STAGE_BASE = Path("/stage/dsa110-contimg/images")
DEFAULT_MS_DIR = Path("/stage/dsa110-contimg/ms")


# ── File helpers ──────────────────────────────────────────────────────────────

def _size_str(path: Path) -> str:
    """Return human-readable file size, or '—' if not accessible."""
    try:
        b = path.stat().st_size
        if b < 1024:
            return f"{b} B"
        elif b < 1024 ** 2:
            return f"{b / 1024:.1f} KB"
        elif b < 1024 ** 3:
            return f"{b / 1024**2:.1f} MB"
        return f"{b / 1024**3:.2f} GB"
    except OSError:
        return "—"


def _mtime_str(path: Path) -> str:
    """Return mtime as a readable string."""
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return "—"


def _file_status(path: Path) -> str:
    """One-line status: 'OK (size, mtime)' or 'MISSING'."""
    if path.exists():
        return f"OK  ({_size_str(path)}, {_mtime_str(path)})"
    return "MISSING"


# ── FITS stats ────────────────────────────────────────────────────────────────

def _fits_stats(fits_path: Path) -> dict | None:
    """Return {peak_jy, rms_mjy, nan_frac} or None if unavailable."""
    try:
        import numpy as np
        from astropy.io import fits

        with fits.open(str(fits_path)) as hdul:
            raw = hdul[0].data
        while raw is not None and raw.ndim > 2:
            raw = raw[0]
        if raw is None:
            return None
        data = raw.astype(float)
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            # All-NaN mosaic: return sentinel so caller can report it explicitly
            return {"peak_jy": float("nan"), "rms_mjy": float("nan"), "nan_frac": 1.0}
        peak = float(np.nanmax(np.abs(finite)))
        # MAD-based RMS (robust to bright sources)
        med = float(np.median(finite))
        rms_jy = float(1.4826 * np.median(np.abs(finite - med)))
        nan_frac = float(np.sum(~np.isfinite(data))) / data.size
        return {
            "peak_jy": peak,
            "rms_mjy": rms_jy * 1000.0,
            "nan_frac": nan_frac,
        }
    except Exception:
        return None


# ── QA summary reader ─────────────────────────────────────────────────────────

def _load_qa_summary(qa_csv: Path) -> list[dict]:
    """Load qa_summary.csv as a list of row dicts."""
    if not qa_csv.exists():
        return []
    try:
        with qa_csv.open() as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _find_qa_row(rows: list[dict], date: str, epoch_label: str) -> dict | None:
    """Find the first qa_summary row matching date and epoch_utc prefix."""
    for row in rows:
        if row.get("date", "") == date and row.get("epoch_utc", "").startswith(epoch_label):
            return row
    return None


# ── Gaincal status ────────────────────────────────────────────────────────────

def _gaincal_status(ms_dir: Path, date: str) -> dict:
    """Check bandpass/gain calibration table presence and provenance."""
    bp_path = ms_dir / f"{date}T22:26:05_0~23.b"
    g_path = ms_dir / f"{date}T22:26:05_0~23.g"

    def _table_info(p: Path) -> dict:
        if p.is_symlink():
            try:
                target = os.readlink(str(p))
                return {"status": "symlink", "target": target}
            except OSError:
                return {"status": "broken_symlink"}
        elif p.exists():
            return {"status": "present"}
        return {"status": "missing"}

    return {
        "bandpass": _table_info(bp_path),
        "gain": _table_info(g_path),
        "bp_path": str(bp_path),
        "g_path": str(g_path),
    }


def _fmt_cal(info: dict) -> str:
    status = info.get("status", "?")
    if status == "present":
        return "present (own)"
    if status == "symlink":
        target = info.get("target", "?")
        # Try to extract source date from target path
        tpath = Path(target)
        tname = tpath.name
        # e.g. 2026-01-25T22:26:05_0~23.b → date is 2026-01-25
        borrowed_date = tname[:10] if len(tname) >= 10 else target
        return f"symlink → {borrowed_date}"
    if status == "missing":
        return "MISSING"
    return status


def _infer_context_from_mosaic_path(mosaic_path: Path, products_base: Path, stage_base: Path) -> dict:
    """Infer epoch/date context from a specific mosaic FITS path."""
    match = re.search(r"(\d{4}-\d{2}-\d{2})T(\d{2})(\d{2})_mosaic\.fits$", mosaic_path.name)
    if not match:
        raise ValueError(
            "mosaic path must end with YYYY-MM-DDTHHMM_mosaic.fits so the epoch can be inferred"
        )

    date = match.group(1)
    hour = int(match.group(2))
    minute = match.group(3)
    label = f"{date}T{hour:02d}{minute}"
    products_dir = products_base / date
    stage_dir = stage_base / f"mosaic_{date}"
    prod_mosaic = products_dir / f"{label}_mosaic.fits"
    stage_mosaic = stage_dir / f"{label}_mosaic.fits"

    return {
        "date": date,
        "hour": hour,
        "label": label,
        "products_dir": products_dir,
        "stage_dir": stage_dir,
        "prod_mosaic": prod_mosaic,
        "stage_mosaic": stage_mosaic,
        "active_mosaic": mosaic_path,
        "qa_png": stage_dir / f"{label}_mosaic_qa_diag.png",
        "phot_csv": products_dir / f"{label}_forced_phot.csv",
    }


# ── Per-epoch report ──────────────────────────────────────────────────────────

def _report_epoch_context(context: dict, ms_dir: Path, qa_rows: list[dict]) -> None:
    """Print the artifact report for one epoch context."""
    date = context["date"]
    hour = context["hour"]
    label = context["label"]
    prod_mosaic = context["prod_mosaic"]
    stage_mosaic = context["stage_mosaic"]
    active_mosaic = context["active_mosaic"]
    qa_png = context["qa_png"]
    phot_csv = context["phot_csv"]
    mosaic_available = active_mosaic.exists() or prod_mosaic.exists() or stage_mosaic.exists()

    print(f"\n  ── Epoch {label} ──")

    # 1. Mosaic FITS
    if prod_mosaic.exists():
        print(f"    Mosaic (products):  {_file_status(prod_mosaic)}")
    else:
        print(f"    Mosaic (products):  MISSING  [{prod_mosaic}]")
    if stage_mosaic.exists():
        print(f"    Mosaic (stage):     {_file_status(stage_mosaic)}")
    elif active_mosaic == stage_mosaic:
        print(f"    Mosaic (stage):     MISSING  [{stage_mosaic}]")
    if active_mosaic not in {prod_mosaic, stage_mosaic}:
        print(f"    Mosaic (input):     {_file_status(active_mosaic)}  [{active_mosaic}]")

    # 2. QA diagnostic PNG
    print(f"    QA diagnostic PNG:  {_file_status(qa_png)}")
    if not qa_png.exists():
        print(f"      (expected: {qa_png})")

    # 3. Forced-phot CSV
    phot_status = _file_status(phot_csv)
    if phot_csv.exists():
        # Count rows
        try:
            with phot_csv.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            phot_status += f"  [{len(rows)} sources]"
        except Exception:
            pass
    print(f"    Forced-phot CSV:    {phot_status}")

    # 4. qa_summary.csv row
    qa_row = _find_qa_row(qa_rows, date, f"{date}T{hour:02d}")
    if qa_row:
        qa_result = qa_row.get("qa_result", "?").upper()
        median_r = qa_row.get("median_ratio", "?")
        n_det = qa_row.get("n_detections", qa_row.get("n_recovered", "?"))
        gaincal_used = qa_row.get("gaincal_used", "?")
        print(f"    qa_summary row:     qa={qa_result}  ratio={median_r}  n={n_det}  cal={gaincal_used}")
    else:
        print(f"    qa_summary row:     not found for epoch {label}")

    # 5. FITS stats (if mosaic exists)
    if mosaic_available:
        import math

        stats = _fits_stats(active_mosaic)
        if stats is None:
            print("    FITS stats:         (read error)")
        elif math.isnan(stats["peak_jy"]):
            print("    FITS stats:         ALL NaN — mosaic is empty (bad epoch)")
        else:
            nan_pct = stats["nan_frac"] * 100.0
            print(
                f"    FITS stats:         peak={stats['peak_jy']:.4f} Jy/bm  "
                f"rms={stats['rms_mjy']:.2f} mJy/bm  "
                f"NaN={nan_pct:.1f}%"
            )
    else:
        print("    FITS stats:         (no mosaic found)")


def _report_epoch(
    date: str,
    hour: int,
    products_dir: Path,
    stage_dir: Path,
    ms_dir: Path,
    qa_rows: list[dict],
) -> None:
    """Build canonical paths for a date/hour epoch and print the report."""
    label = f"{date}T{hour:02d}00"
    prod_mosaic = products_dir / f"{label}_mosaic.fits"
    stage_mosaic = stage_dir / f"{label}_mosaic.fits"
    active_mosaic = prod_mosaic if prod_mosaic.exists() else stage_mosaic
    context = {
        "date": date,
        "hour": hour,
        "label": label,
        "products_dir": products_dir,
        "stage_dir": stage_dir,
        "prod_mosaic": prod_mosaic,
        "stage_mosaic": stage_mosaic,
        "active_mosaic": active_mosaic,
        "qa_png": stage_dir / f"{label}_mosaic_qa_diag.png",
        "phot_csv": products_dir / f"{label}_forced_phot.csv",
    }
    _report_epoch_context(context, ms_dir, qa_rows)


# ── Date-level report ─────────────────────────────────────────────────────────

def _report_date(
    date: str,
    epoch_filter: int | None,
    products_base: Path,
    stage_base: Path,
    ms_dir: Path,
    qa_summary: Path,
) -> None:
    products_dir = products_base / date
    stage_dir = stage_base / f"mosaic_{date}"

    qa_rows = _load_qa_summary(qa_summary)
    cal_info = _gaincal_status(ms_dir, date)

    print(f"\n{'=' * 72}")
    print(f"  Epoch Artifact Inspector — {date}")
    print(f"{'=' * 72}")
    print(f"\n  Products dir:  {products_dir}")
    print(f"  Stage dir:     {stage_dir}")
    print(f"  Bandpass cal:  {_fmt_cal(cal_info['bandpass'])}  ({cal_info['bp_path']})")
    print(f"  Gain cal:      {_fmt_cal(cal_info['gain'])}  ({cal_info['g_path']})")
    print(f"  qa_summary:    {_file_status(qa_summary)}  [{len(qa_rows)} total rows]")

    # Discover epochs: scan both dirs for *T??00_mosaic.fits or *T??00_forced_phot.csv
    import re
    found_hours: set[int] = set()
    hour_pattern = re.compile(rf"{date}T(\d{{2}})00")

    for d in (products_dir, stage_dir):
        if d.exists():
            for f in d.iterdir():
                m = hour_pattern.search(f.name)
                if m:
                    found_hours.add(int(m.group(1)))

    # Also add hours from qa_summary rows for this date
    for row in qa_rows:
        if row.get("date", "") == date:
            ep = row.get("epoch_utc", "")
            m = re.search(r"T(\d{2})", ep)
            if m:
                found_hours.add(int(m.group(1)))

    if epoch_filter is not None:
        hours = [epoch_filter]
        if epoch_filter not in found_hours:
            print(f"\n  WARNING: no artifacts found for hour {epoch_filter:02d} — showing anyway")
    else:
        hours = sorted(found_hours)
        if not hours:
            print("\n  No epoch artifacts found for this date.")
            print(f"  Check: products dir = {products_dir}")
            print(f"         stage dir    = {stage_dir}")
            print(f"{'=' * 72}\n")
            return

    for h in hours:
        _report_epoch(date, h, products_dir, stage_dir, ms_dir, qa_rows)

    print(f"\n{'=' * 72}\n")


def _report_mosaic_path(
    mosaic_path: Path,
    products_base: Path,
    stage_base: Path,
    ms_dir: Path,
    qa_summary: Path,
) -> None:
    """Inspect artifacts for one explicit mosaic FITS path."""
    context = _infer_context_from_mosaic_path(mosaic_path, products_base, stage_base)
    qa_rows = _load_qa_summary(qa_summary)
    cal_info = _gaincal_status(ms_dir, context["date"])

    print(f"\n{'=' * 72}")
    print(f"  Epoch Artifact Inspector — {context['label']}")
    print(f"{'=' * 72}")
    print(f"\n  Input mosaic:   {mosaic_path}")
    print(f"  Products dir:   {context['products_dir']}")
    print(f"  Stage dir:      {context['stage_dir']}")
    print(f"  Bandpass cal:   {_fmt_cal(cal_info['bandpass'])}  ({cal_info['bp_path']})")
    print(f"  Gain cal:       {_fmt_cal(cal_info['gain'])}  ({cal_info['g_path']})")
    print(f"  qa_summary:     {_file_status(qa_summary)}  [{len(qa_rows)} total rows]")

    _report_epoch_context(context, ms_dir, qa_rows)
    print(f"\n{'=' * 72}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Report pipeline artifact state for a given date/epoch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--date",
        help="Observation date (YYYY-MM-DD)",
    )
    mode.add_argument(
        "--mosaic-path",
        default=None,
        help="Direct path to one epoch mosaic FITS (YYYY-MM-DDTHHMM_mosaic.fits)",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="UTC hour (0–23) to inspect; valid with --date",
    )
    parser.add_argument(
        "--products-dir",
        default=str(DEFAULT_PRODUCTS_BASE),
        help=f"Products mosaics base dir (default: {DEFAULT_PRODUCTS_BASE})",
    )
    parser.add_argument(
        "--stage-dir",
        default=str(DEFAULT_STAGE_BASE),
        help=f"Stage images base dir (default: {DEFAULT_STAGE_BASE})",
    )
    parser.add_argument(
        "--ms-dir",
        default=str(DEFAULT_MS_DIR),
        help=f"Measurement Set directory (for cal table check, default: {DEFAULT_MS_DIR})",
    )
    parser.add_argument(
        "--qa-summary",
        default=str(DEFAULT_QA_SUMMARY),
        help=f"Path to qa_summary.csv (default: {DEFAULT_QA_SUMMARY})",
    )
    args = parser.parse_args()

    products_base = Path(args.products_dir)
    stage_base = Path(args.stage_dir)
    ms_dir = Path(args.ms_dir)
    qa_summary = Path(args.qa_summary)

    if args.mosaic_path:
        if args.epoch is not None:
            parser.error("--epoch is only valid together with --date")
        _report_mosaic_path(
            mosaic_path=Path(args.mosaic_path),
            products_base=products_base,
            stage_base=stage_base,
            ms_dir=ms_dir,
            qa_summary=qa_summary,
        )
        return

    _report_date(
        date=args.date,
        epoch_filter=args.epoch,
        products_base=products_base,
        stage_base=stage_base,
        ms_dir=ms_dir,
        qa_summary=qa_summary,
    )


if __name__ == "__main__":
    main()
