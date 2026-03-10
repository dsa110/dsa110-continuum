#!/opt/miniforge/envs/casa6/bin/python
"""Canary history tracker — record and compare canary QA runs across code changes.

Each `record` run measures the canary tile, appends a JSONL entry, then prints
the result plus a delta against the previous record.  Use this after any change
to QA, photometry, or mosaic code to confirm science-facing behaviour is stable.

Canary tile: 2026-01-25T22:26:05 (3C454.3 field, ~12.5 Jy/beam peak)
Acceptance:  ratio ∈ [0.85, 1.15] | n_recovered >= 3 | RMS <= 17.1 mJy/beam

Usage:
    # Measure the canary tile and append to history log
    python scripts/canary_history.py record

    # Use a different FITS (e.g. after re-imaging)
    python scripts/canary_history.py record --fits /path/to/tile.fits

    # Show latest entry + delta from previous
    python scripts/canary_history.py show

    # Show last N entries
    python scripts/canary_history.py show --n 5

    # Override the log file location
    python scripts/canary_history.py record --log /tmp/canary.jsonl
"""
import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CANARY_FITS_PB = (
    "/stage/dsa110-contimg/images/mosaic_2026-01-25/2026-01-25T22:26:05-image-pb.fits"
)
DEFAULT_CANARY_FITS = (
    "/stage/dsa110-contimg/images/mosaic_2026-01-25/2026-01-25T22:26:05-image.fits"
)
DEFAULT_NVSS_DB = "/data/dsa110-contimg/state/catalogs/nvss_full.sqlite3"
DEFAULT_LOG = REPO_ROOT / "outputs" / "dev_tools" / "canary_history.jsonl"

# Acceptance thresholds (mirrored from run_canary.sh)
RATIO_LO = 0.85
RATIO_HI = 1.15
N_RECOVERED_MIN = 3
RMS_MAX_MJY = 17.1


# ── Git helper ────────────────────────────────────────────────────────────────

def _git_commit() -> str:
    """Return the current HEAD short hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=REPO_ROOT, timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ── Log I/O ───────────────────────────────────────────────────────────────────

def _load_log(log_path: Path) -> list[dict]:
    """Load all JSONL entries; returns [] if log doesn't exist."""
    if not log_path.exists():
        return []
    entries = []
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def _append_log(log_path: Path, entry: dict) -> None:
    """Append one JSONL record to the log (creates file + parents if needed)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Measurement ───────────────────────────────────────────────────────────────

def _pick_fits(fits_override: str | None) -> str:
    """Return the FITS path to use, with pb-corrected preferred."""
    if fits_override:
        return fits_override
    if Path(DEFAULT_CANARY_FITS_PB).exists():
        return DEFAULT_CANARY_FITS_PB
    return DEFAULT_CANARY_FITS


def _canary_pass(entry: dict) -> bool:
    ratio = entry.get("median_ratio", float("nan"))
    n = entry.get("n_recovered", 0)
    rms = entry.get("rms_mjy", float("inf"))
    ratio_ok = not math.isnan(ratio) and RATIO_LO <= ratio <= RATIO_HI
    return ratio_ok and n >= N_RECOVERED_MIN and rms <= RMS_MAX_MJY


def _measure(fits_path: str, nvss_db: str) -> dict:
    """Run epoch QA on the canary FITS and return a log-ready dict."""
    from dsa110_continuum.photometry.epoch_qa import measure_epoch_qa

    result = measure_epoch_qa(fits_path, nvss_db)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "fits_path": fits_path,
        "median_ratio": result.median_ratio,
        "n_recovered": result.n_recovered,
        "n_catalog": result.n_catalog,
        "rms_mjy": result.mosaic_rms_mjy,
        "ratio_gate": result.ratio_gate,
        "completeness_gate": result.completeness_gate,
        "rms_gate": result.rms_gate,
        "epoch_qa": result.qa_result,
        "canary_pass": _canary_pass({
            "median_ratio": result.median_ratio,
            "n_recovered": result.n_recovered,
            "rms_mjy": result.mosaic_rms_mjy,
        }),
    }


# ── Display helpers ───────────────────────────────────────────────────────────

def _fmt_entry(entry: dict, label: str = "") -> str:
    lines = []
    ts = entry.get("timestamp", "?")
    commit = entry.get("git_commit", "?")
    verdict = "PASS" if entry.get("canary_pass") else "FAIL"
    if label:
        lines.append(f"  {label} [{commit}]  {ts}")
    else:
        lines.append(f"  commit={commit}  {ts}")
    lines.append(f"    ratio:       {entry.get('median_ratio', float('nan')):.4f}  [{entry.get('ratio_gate', '?')}]")
    lines.append(f"    n_recovered: {entry.get('n_recovered', '?')} / {entry.get('n_catalog', '?')}")
    lines.append(f"    RMS:         {entry.get('rms_mjy', float('nan')):.2f} mJy/beam  [{entry.get('rms_gate', '?')}]")
    lines.append(f"    CANARY:      {verdict}")
    return "\n".join(lines)


def _fmt_delta(prev: dict, curr: dict) -> str:
    def _d(key: str, fmt: str = ".4f") -> str:
        p = prev.get(key, float("nan"))
        c = curr.get(key, float("nan"))
        try:
            delta = c - p  # type: ignore[operator]
            sign = "+" if delta >= 0 else ""
            return f"{sign}{delta:{fmt}}"
        except TypeError:
            return "n/a"

    p_pass = prev.get("canary_pass")
    c_pass = curr.get("canary_pass")
    status_change = ""
    if p_pass != c_pass:
        status_change = f"  *** CANARY STATUS CHANGED: {'FAIL→PASS' if c_pass else 'PASS→FAIL'} ***"

    lines = [
        f"  Delta ({prev.get('git_commit', '?')} → {curr.get('git_commit', '?')}):",
        f"    Δratio:       {_d('median_ratio')}",
        f"    Δn_recovered: {_d('n_recovered', 'd')}",
        f"    ΔRMS:         {_d('rms_mjy', '.2f')} mJy/beam",
    ]
    if status_change:
        lines.append(status_change)
    return "\n".join(lines)


def _assess_change(prev: dict, curr: dict) -> dict[str, object]:
    """Classify the latest canary drift as stable, notable, or concerning."""
    reasons: list[str] = []

    prev_pass = bool(prev.get("canary_pass"))
    curr_pass = bool(curr.get("canary_pass"))
    if prev_pass != curr_pass:
        reasons.append("canary status changed")

    try:
        ratio_delta = abs(float(curr.get("median_ratio", float("nan"))) - float(prev.get("median_ratio", float("nan"))))
        if not math.isnan(ratio_delta) and ratio_delta > 0.05:
            reasons.append("median ratio shifted by > 0.05")
    except (TypeError, ValueError):
        pass

    try:
        recovered_delta = int(curr.get("n_recovered", 0)) - int(prev.get("n_recovered", 0))
        if recovered_delta <= -2:
            reasons.append("recovered-source count dropped by 2 or more")
    except (TypeError, ValueError):
        pass

    try:
        rms_delta = float(curr.get("rms_mjy", float("nan"))) - float(prev.get("rms_mjy", float("nan")))
        if not math.isnan(rms_delta) and rms_delta > 2.0:
            reasons.append("image RMS increased by > 2 mJy/beam")
    except (TypeError, ValueError):
        pass

    concerning = len(reasons) > 0
    if concerning:
        summary = "Concerning drift: " + "; ".join(reasons)
    else:
        summary = "Stable relative to previous canary"

    return {"concerning": concerning, "summary": summary, "reasons": reasons}


# ── Subcommands ───────────────────────────────────────────────────────────────

def cmd_record(args: argparse.Namespace) -> int:
    """Measure the canary tile and append result to the history log."""
    log_path = Path(args.log)
    fits_path = _pick_fits(args.fits)

    if not Path(fits_path).exists():
        print(f"ERROR: canary FITS not found: {fits_path}", file=sys.stderr)
        print("  Run the pipeline first or supply --fits /path/to/tile.fits", file=sys.stderr)
        return 1
    if not Path(args.nvss_db).exists():
        print(f"ERROR: NVSS database not found: {args.nvss_db}", file=sys.stderr)
        return 1

    print(f"Measuring canary: {fits_path}")
    try:
        entry = _measure(fits_path, args.nvss_db)
    except Exception as e:
        print(f"ERROR: measurement failed: {e}", file=sys.stderr)
        return 1

    # Show result
    print()
    print(_fmt_entry(entry, label="NEW"))

    # Load history for delta
    history = _load_log(log_path)
    if history:
        prev = history[-1]
        print()
        print(_fmt_delta(prev, entry))
        assessment = _assess_change(prev, entry)
        print(f"  Assessment:    {assessment['summary']}")

    # Append to log
    _append_log(log_path, entry)
    print(f"\n  Appended to {log_path}")

    return 0 if entry["canary_pass"] else 1


def cmd_show(args: argparse.Namespace) -> int:
    """Print recent canary history and the delta between the two latest runs."""
    log_path = Path(args.log)
    history = _load_log(log_path)

    if not history:
        print(f"  No canary history found at {log_path}")
        print("  Run: python scripts/canary_history.py record")
        return 0

    n = min(args.n, len(history))
    recent = history[-n:]

    print(f"\n{'=' * 72}")
    print(f"  Canary History  (log: {log_path})")
    print(f"{'=' * 72}\n")

    for i, entry in enumerate(recent):
        label = "LATEST" if i == len(recent) - 1 else f"  #{len(history) - n + i + 1}"
        print(_fmt_entry(entry, label=label))
        print()

    if len(recent) >= 2:
        prev, curr = recent[-2], recent[-1]
        print(f"{'─' * 72}")
        print(_fmt_delta(prev, curr))
        assessment = _assess_change(prev, curr)
        print(f"  Assessment:    {assessment['summary']}")
        print()

    print(f"{'=' * 72}\n")
    return 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Track canary QA results across code changes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--log",
        default=str(DEFAULT_LOG),
        help=f"JSONL log file (default: {DEFAULT_LOG})",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_record = sub.add_parser("record", help="Measure canary tile and append to log")
    p_record.add_argument("--fits", default=None, help="Override canary FITS path")
    p_record.add_argument("--nvss-db", default=DEFAULT_NVSS_DB, help="NVSS SQLite DB")

    p_show = sub.add_parser("show", help="Show recent history and delta")
    p_show.add_argument("--n", type=int, default=2, help="Number of entries to show (default 2)")

    args = parser.parse_args()

    if args.cmd == "record":
        sys.exit(cmd_record(args))
    elif args.cmd == "show":
        sys.exit(cmd_show(args))


if __name__ == "__main__":
    main()
