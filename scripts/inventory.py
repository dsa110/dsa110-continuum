#!/opt/miniforge/envs/casa6/bin/python
"""
Inventory HDF5 data in /data/incoming/ and MS files in /stage/dsa110-contimg/ms/.

Grouping strategy (two-tier):
  1. Indexed observations: read directly from pipeline.sqlite3 tables
     ``hdf5_files`` and ``group_time_ranges``.  These groups were built using
     exact ``time_array[0]`` (Julian Date) matching from HDF5 headers, so they
     are authoritative.
  2. Unindexed observations: fall back to filename-based grouping
     ({YYYY-MM-DDTHH:MM:SS}_sb{NN}.hdf5).  Used for files not yet in the DB.

Outputs:
  - Human-readable summary to stdout
  - /data/dsa110-continuum/inventory.csv
"""
import csv
import os
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
HDF5_DIR   = Path("/data/incoming")
MS_DIR     = Path("/stage/dsa110-contimg/ms")
PIPELINE_DB = Path("/data/dsa110-contimg/state/db/pipeline.sqlite3")
OUT_CSV    = Path("/data/dsa110-continuum/inventory.csv")

# Expected subbands per complete observation (hard-wired DSA-110 value,
# verified from the first full day in the DB).
N_SUBBANDS_EXPECTED = 16

# Regex for valid HDF5 filenames: 2026-01-25T22:26:05_sb03.hdf5
HDF5_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})_sb(\d+)\.hdf5$"
)


# ── Source 1: pipeline SQLite DB ───────────────────────────────────────────────

def load_from_db(db_path: Path) -> tuple[dict, set]:
    """
    Read group metadata from pipeline.sqlite3.

    Returns
    -------
    obs_db : dict
        {timestamp: {"date": str, "n_subbands": int, "size_bytes": int,
                     "is_complete": bool, "source": "db"}}
    indexed_timestamps : set[str]
        All timestamps present in hdf5_files (even if group_time_ranges
        doesn't have a row yet — shouldn't happen, but defensive).
    """
    obs_db: dict = {}
    indexed_timestamps: set = set()

    if not db_path.exists():
        return obs_db, indexed_timestamps

    conn = sqlite3.connect(str(db_path), timeout=10)

    # --- per-group summary from group_time_ranges ---------------------------
    rows = conn.execute(
        "SELECT group_id, file_count FROM group_time_ranges"
    ).fetchall()
    for group_id, file_count in rows:
        date = group_id[:10]  # "2026-01-25"
        obs_db[group_id] = {
            "date": date,
            "n_subbands": file_count,
            "size_bytes": 0,          # filled below
            # Use the same criterion as the filesystem scan: exactly N_SUBBANDS_EXPECTED.
            # The DB's own `complete` flag uses >= 16, which marks synthetic test groups
            # (file_count=240, 80) as complete — we want a stricter, uniform definition.
            "is_complete": (file_count == N_SUBBANDS_EXPECTED),
            "source": "db",
        }

    # --- per-file sizes aggregated to group ---------------------------------
    size_rows = conn.execute(
        "SELECT group_id, SUM(file_size_bytes) FROM hdf5_files GROUP BY group_id"
    ).fetchall()
    for group_id, total_bytes in size_rows:
        if group_id in obs_db:
            obs_db[group_id]["size_bytes"] = int(total_bytes or 0)
        indexed_timestamps.add(group_id)

    # Timestamps in hdf5_files not yet in group_time_ranges (edge case)
    all_groups = conn.execute(
        "SELECT DISTINCT group_id FROM hdf5_files"
    ).fetchall()
    for (g,) in all_groups:
        indexed_timestamps.add(g)

    conn.close()
    return obs_db, indexed_timestamps


# ── Source 2: filesystem scan (for unindexed files) ───────────────────────────

def scan_hdf5_unindexed(hdf5_dir: Path, skip_timestamps: set) -> dict:
    """
    Walk hdf5_dir non-recursively and collect per-timestamp info for files
    whose timestamp is NOT already in skip_timestamps (i.e. not in the DB).

    Returns
    -------
    obs_fs : dict
        {timestamp: {"date": str, "n_subbands": int, "size_bytes": int,
                     "is_complete": bool, "source": "fs"}}
    """
    raw: dict = defaultdict(lambda: {"date": "", "subbands": set(), "size_bytes": 0})
    skipped = []

    for entry in hdf5_dir.iterdir():
        if not entry.is_file():
            continue
        m = HDF5_RE.match(entry.name)
        if not m:
            if not entry.name.startswith(".") and not entry.name.endswith(".corrupted"):
                skipped.append(entry.name)
            continue
        date_str, time_str, sb_str = m.group(1), m.group(2), m.group(3)
        ts = f"{date_str}T{time_str}"
        if ts in skip_timestamps:
            continue  # already authoritative from DB
        raw[ts]["date"] = date_str
        raw[ts]["subbands"].add(int(sb_str))
        raw[ts]["size_bytes"] += entry.stat().st_size

    if skipped:
        print("WARNING: unexpected files in HDF5 dir (not matched by pattern):")
        for name in sorted(skipped)[:20]:
            print(f"  {name}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")
        print()

    obs_fs = {}
    for ts, info in raw.items():
        n = len(info["subbands"])
        obs_fs[ts] = {
            "date": info["date"],
            "n_subbands": n,
            "size_bytes": info["size_bytes"],
            "is_complete": (n == N_SUBBANDS_EXPECTED),
            "source": "fs",
        }
    return obs_fs


# ── Scan MS dir ───────────────────────────────────────────────────────────────

def scan_ms(ms_dir: Path) -> set:
    """Return set of timestamp strings that already have a raw MS file."""
    converted: set = set()
    if not ms_dir.exists():
        return converted
    for entry in ms_dir.iterdir():
        if (entry.suffix == ".ms"
                and "meridian" not in entry.name
                and "flagversion" not in entry.name):
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

    # ── Load groups ───────────────────────────────────────────────────────────
    print(f"Loading indexed groups from {PIPELINE_DB} ...")
    obs_db, indexed_ts = load_from_db(PIPELINE_DB)
    print(f"  {len(obs_db)} groups from DB ({len(indexed_ts)} timestamps indexed)")

    print(f"Scanning filesystem for unindexed files in {HDF5_DIR} ...")
    obs_fs = scan_hdf5_unindexed(HDF5_DIR, skip_timestamps=indexed_ts)
    print(f"  {len(obs_fs)} unindexed groups from filesystem\n")

    obs = {**obs_db, **obs_fs}  # DB takes priority for any overlap

    if not obs:
        print("ERROR: No observations found.", file=sys.stderr)
        sys.exit(1)

    converted_ts = scan_ms(MS_DIR)

    # ── Build per-date groups ─────────────────────────────────────────────────
    by_date: dict = defaultdict(list)
    for ts, info in obs.items():
        by_date[info["date"]].append(ts)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["date", "timestamp", "n_subbands_found", "n_subbands_expected",
                  "is_complete", "size_bytes", "has_ms", "group_source"]
    csv_rows = []
    for ts, info in sorted(obs.items()):
        csv_rows.append({
            "date": info["date"],
            "timestamp": ts,
            "n_subbands_found": info["n_subbands"],
            "n_subbands_expected": N_SUBBANDS_EXPECTED,
            "is_complete": info["is_complete"],
            "size_bytes": info["size_bytes"],
            "has_ms": ts in converted_ts,
            "group_source": info["source"],
        })
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # ── Human-readable report ─────────────────────────────────────────────────
    grand_total = grand_complete = grand_bytes = grand_converted = 0
    all_dates = sorted(by_date.keys())

    print("=" * 72)
    print("DSA-110 HDF5 DATA INVENTORY")
    print(f"  (complete = all {N_SUBBANDS_EXPECTED} subbands present)")
    print("=" * 72)

    for date in all_dates:
        timestamps = sorted(by_date[date])
        d_total    = len(timestamps)
        d_complete = sum(1 for ts in timestamps if obs[ts]["is_complete"])
        d_bytes    = sum(obs[ts]["size_bytes"] for ts in timestamps)
        d_conv     = sum(1 for ts in timestamps if ts in converted_ts)

        print(f"\n── {date}  "
              f"({d_total} obs | {d_complete} complete | "
              f"{d_total - d_complete} incomplete | "
              f"{fmt_bytes(d_bytes)} | {d_conv} MS)")
        print(f"   {'TIMESTAMP':<22}  {'SRC':>3}  {'FOUND':>5}  {'OK':>3}  {'MS':>3}  SIZE")
        print(f"   {'-'*22}  {'---':>3}  {'-----':>5}  {'---':>3}  {'---':>3}  ----")

        for ts in timestamps:
            info = obs[ts]
            flag = "" if info["is_complete"] else "  *** INCOMPLETE"
            print(
                f"   {ts:<22}  {info['source']:>3}  "
                f"{info['n_subbands']:>5}  "
                f"{'Y' if info['is_complete'] else 'N':>3}  "
                f"{'Y' if ts in converted_ts else 'N':>3}  "
                f"{fmt_bytes(info['size_bytes'])}{flag}"
            )

        grand_total    += d_total
        grand_complete += d_complete
        grand_bytes    += d_bytes
        grand_converted += d_conv

    print()
    print("=" * 72)
    print("GRAND TOTAL")
    print("=" * 72)
    print(f"  Date range         : {all_dates[0]}  →  {all_dates[-1]}")
    print(f"  Dates with data    : {len(all_dates)}")
    print(f"  Total observations : {grand_total}")
    print(f"  Complete           : {grand_complete}  ({100*grand_complete/grand_total:.1f}%)")
    print(f"  Incomplete         : {grand_total - grand_complete}")
    print(f"  Converted to MS    : {grand_converted} / {grand_total}")
    print(f"  Still to process   : {grand_total - grand_converted}")
    print(f"  Total data volume  : {fmt_bytes(grand_bytes)}")
    print(f"\nCSV written to: {OUT_CSV}")


if __name__ == "__main__":
    main()
