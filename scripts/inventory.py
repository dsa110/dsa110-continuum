#!/opt/miniforge/envs/casa6/bin/python
"""
Inventory HDF5 data in /data/incoming/ and MS files in /stage/dsa110-contimg/ms/.

Outputs:
  - Human-readable summary to stdout
  - /data/dsa110-continuum/inventory.csv
"""
import os
import re
import sys
import csv
from pathlib import Path
from collections import defaultdict

# ── Configuration ─────────────────────────────────────────────────────────────
HDF5_DIR   = Path("/data/incoming")
MS_DIR     = Path("/stage/dsa110-contimg/ms")
OUT_CSV    = Path("/data/dsa110-continuum/inventory.csv")

# Regex for valid HDF5 filenames: 2026-01-25T22:26:05_sb03.hdf5
HDF5_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})_sb(\d+)\.hdf5$"
)

# ── Scan HDF5 dir ─────────────────────────────────────────────────────────────

def scan_hdf5(hdf5_dir: Path) -> dict:
    """
    Walk hdf5_dir (non-recursively) and collect per-timestamp info.

    Returns:
        {timestamp_str: {"date": str, "subbands": set[int], "size_bytes": int}}
    """
    obs: dict = defaultdict(lambda: {"date": "", "subbands": set(), "size_bytes": 0})
    skipped = []

    for entry in hdf5_dir.iterdir():
        if not entry.is_file():
            continue  # skip subdirectories
        m = HDF5_RE.match(entry.name)
        if not m:
            # Report genuinely unexpected names (not dot-files / temp files)
            if not entry.name.startswith(".") and not entry.name.endswith(".corrupted"):
                skipped.append(entry.name)
            continue
        date_str, time_str, sb_str = m.group(1), m.group(2), m.group(3)
        ts = f"{date_str}T{time_str}"
        obs[ts]["date"] = date_str
        obs[ts]["subbands"].add(int(sb_str))
        obs[ts]["size_bytes"] += entry.stat().st_size

    if skipped:
        print("WARNING: unexpected files in HDF5 dir (not matched by pattern):")
        for name in sorted(skipped)[:20]:
            print(f"  {name}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")
        print()

    return dict(obs)


def determine_expected_subbands(obs: dict) -> int:
    """
    Infer the expected number of subbands from the data.

    Strategy: the first (earliest) observation that has the maximum
    subband count is treated as "complete"; its count becomes the
    expected value for all observations.
    """
    if not obs:
        return 0
    return max(len(v["subbands"]) for v in obs.values())


# ── Scan MS dir ───────────────────────────────────────────────────────────────

def scan_ms(ms_dir: Path) -> set:
    """Return set of timestamp strings that already have a raw MS file."""
    converted = set()
    if not ms_dir.exists():
        return converted
    for entry in ms_dir.iterdir():
        if entry.suffix == ".ms" and "meridian" not in entry.name and "flagversion" not in entry.name:
            converted.add(entry.stem)
    return converted


# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not HDF5_DIR.exists():
        print(f"ERROR: HDF5 directory not found: {HDF5_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {HDF5_DIR} ...")
    obs = scan_hdf5(HDF5_DIR)

    if not obs:
        print("ERROR: No valid HDF5 files found — check naming convention.", file=sys.stderr)
        sys.exit(1)

    n_expected = determine_expected_subbands(obs)
    print(f"Expected subbands per observation: {n_expected}\n")

    converted_ts = scan_ms(MS_DIR)

    # ── Build per-date groups ─────────────────────────────────────────────────
    by_date: dict = defaultdict(list)
    for ts, info in obs.items():
        by_date[info["date"]].append(ts)

    # ── Build CSV rows ────────────────────────────────────────────────────────
    csv_rows = []
    for ts, info in sorted(obs.items()):
        n_found = len(info["subbands"])
        is_complete = (n_found == n_expected)
        csv_rows.append({
            "date": info["date"],
            "timestamp": ts,
            "n_subbands_found": n_found,
            "n_subbands_expected": n_expected,
            "is_complete": is_complete,
            "size_bytes": info["size_bytes"],
            "has_ms": ts in converted_ts,
        })

    # ── Write CSV ─────────────────────────────────────────────────────────────
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["date", "timestamp", "n_subbands_found", "n_subbands_expected",
                  "is_complete", "size_bytes", "has_ms"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # ── Human-readable report ─────────────────────────────────────────────────
    grand_total_obs = 0
    grand_complete  = 0
    grand_bytes     = 0
    grand_converted = 0

    all_dates = sorted(by_date.keys())

    print("=" * 70)
    print("DSA-110 HDF5 DATA INVENTORY")
    print("=" * 70)

    for date in all_dates:
        timestamps = sorted(by_date[date])
        date_obs      = len(timestamps)
        date_complete = sum(1 for ts in timestamps if len(obs[ts]["subbands"]) == n_expected)
        date_incomplete = date_obs - date_complete
        date_bytes    = sum(obs[ts]["size_bytes"] for ts in timestamps)
        date_converted = sum(1 for ts in timestamps if ts in converted_ts)

        print(f"\n── {date}  ({date_obs} obs | {date_complete} complete | "
              f"{date_incomplete} incomplete | {fmt_bytes(date_bytes)} | "
              f"{date_converted} converted to MS)")
        print(f"   {'TIMESTAMP':<22}  {'FOUND':>5}  {'EXPCT':>5}  {'OK':>3}  {'MS':>3}  SIZE")
        print(f"   {'-'*22}  {'-----':>5}  {'-----':>5}  {'---':>3}  {'---':>3}  ----")

        for ts in timestamps:
            n_found = len(obs[ts]["subbands"])
            is_complete = n_found == n_expected
            has_ms = ts in converted_ts
            flag = "" if is_complete else "  *** INCOMPLETE"
            print(
                f"   {ts:<22}  {n_found:>5}  {n_expected:>5}  "
                f"{'Y' if is_complete else 'N':>3}  "
                f"{'Y' if has_ms else 'N':>3}  "
                f"{fmt_bytes(obs[ts]['size_bytes'])}{flag}"
            )

        grand_total_obs += date_obs
        grand_complete  += date_complete
        grand_bytes     += date_bytes
        grand_converted += date_converted

    print()
    print("=" * 70)
    print("GRAND TOTAL")
    print("=" * 70)
    print(f"  Date range         : {all_dates[0]}  →  {all_dates[-1]}")
    print(f"  Dates with data    : {len(all_dates)}")
    print(f"  Total observations : {grand_total_obs}")
    print(f"  Complete           : {grand_complete}  ({100*grand_complete/grand_total_obs:.1f}%)")
    print(f"  Incomplete         : {grand_total_obs - grand_complete}")
    print(f"  Converted to MS    : {grand_converted} / {grand_total_obs}")
    print(f"  Still to process   : {grand_total_obs - grand_converted}")
    print(f"  Total data volume  : {fmt_bytes(grand_bytes)}")
    print(f"\nCSV written to: {OUT_CSV}")


if __name__ == "__main__":
    main()
