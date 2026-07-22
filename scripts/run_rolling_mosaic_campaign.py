#!/usr/bin/env python3
# ruff: noqa: D103
"""Resumable Jan-Apr 2026 rolling mosaic campaign for H17."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INPUT_DIR = Path(os.environ.get("DSA110_INPUT_DIR", "/data/incoming"))
MS_DIR = Path(os.environ.get("DSA110_MS_DIR", "/stage/dsa110-continuum/ms"))
PROC_MS_DIR = Path(os.environ.get("DSA110_PROC_MS_DIR", "/data/dsa110-proc/ms"))
PRODUCTS_MOSAIC_DIR = Path(
    os.environ.get("DSA110_PRODUCTS_BASE", "/data/dsa110-proc/products/mosaics")
)
PRODUCTS_DIR = PRODUCTS_MOSAIC_DIR.parent
IMAGE_DIR = Path(
    os.environ.get("DSA110_STAGE_IMAGE_BASE", "/stage/dsa110-continuum/images")
)
DB_PATH = Path(os.environ.get("PIPELINE_DB", REPO / "state/db/pipeline.sqlite3"))
CAMPAIGN_DIR = Path(
    os.environ.get(
        "DSA110_CAMPAIGN_DIR",
        REPO / "outputs/jan-apr-mosaic-campaign-2026-07-21",
    )
)
STATUS_PATH = CAMPAIGN_DIR / "status.json"
FILE_RE = re.compile(r"^(2026-(?:01|02|03|04)-\d{2})T.*_sb\d+\.hdf5$")
NVME_RESERVE_BYTES = 20 * 1024**3


def write_status(**updates: object) -> dict:
    CAMPAIGN_DIR.mkdir(parents=True, exist_ok=True)
    status = json.loads(STATUS_PATH.read_text()) if STATUS_PATH.exists() else {}
    status.update(updates, updated_at=datetime.now().astimezone().isoformat())
    temp = STATUS_PATH.with_suffix(".tmp")
    temp.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    temp.replace(STATUS_PATH)
    return status


def index_inventory() -> None:
    from dsa110_continuum.database.hdf5_index import index_subband_files
    from dsa110_continuum.database.unified import init_unified_db

    files_by_date: dict[str, list[Path]] = defaultdict(list)
    with os.scandir(INPUT_DIR) as entries:
        for entry in entries:
            match = FILE_RE.match(entry.name)
            if match and entry.is_file(follow_symlinks=False):
                files_by_date[match.group(1)].append(Path(entry.path))

    database = init_unified_db(DB_PATH)
    try:
        for date, files in sorted(files_by_date.items()):
            added = index_subband_files(database.conn, sorted(files))
            print(f"indexed {date}: {added} new rows", flush=True)
    finally:
        database.close()


def complete_hours() -> dict[str, list[int]]:
    query = """
        WITH groups_by_hour AS (
            SELECT obs_date, CAST(substr(timestamp_iso, 12, 2) AS INTEGER) AS hour,
                   group_id, COUNT(DISTINCT subband_num) AS subbands
            FROM hdf5_files
            WHERE obs_date BETWEEN '2026-01-01' AND '2026-04-30'
            GROUP BY obs_date, hour, group_id
        )
        SELECT obs_date, hour
        FROM groups_by_hour
        WHERE subbands = 16
        GROUP BY obs_date, hour
        ORDER BY obs_date, hour
    """
    result: dict[str, list[int]] = defaultdict(list)
    with sqlite3.connect(DB_PATH) as connection:
        for date, hour in connection.execute(query):
            result[str(date)].append(int(hour))
    return dict(result)


def rolling_window(hours: list[int], index: int) -> tuple[int, int]:
    start = hours[index - 1] if index else hours[index]
    end = hours[index + 1] + 1 if index + 1 < len(hours) else hours[index] + 1
    return start, end


def iso_bound(date: str, hour: int) -> str:
    return (datetime.strptime(date, "%Y-%m-%d") + timedelta(hours=hour)).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)


def _select_ms_for_mosaic_hour(ms_paths: list[str], target_hour: int) -> list[str]:
    """Keep one hourly core plus two tiles from each adjacent available hour."""
    by_hour: dict[int, list[str]] = defaultdict(list)
    for path in sorted(ms_paths):
        try:
            hour = datetime.strptime(Path(path).stem, "%Y-%m-%dT%H:%M:%S").hour
        except ValueError:
            continue
        by_hour[hour].append(path)
    if target_hour not in by_hour:
        return []
    hours = sorted(by_hour)
    index = hours.index(target_hour)
    selected = list(by_hour[target_hour])
    if index:
        selected = by_hour[hours[index - 1]][-2:] + selected
    if index + 1 < len(hours):
        selected.extend(by_hour[hours[index + 1]][:2])
    return selected


def _measurement_set_is_readable(ms_path: str) -> bool:
    """Return whether the main table and essential subtables can be read."""
    from dsa110_continuum.adapters import casa_tables as ct

    try:
        with ct.table(ms_path, readonly=True, ack=False) as table:
            if table.nrows() == 0 or not {"DATA", "FLAG"}.issubset(table.colnames()):
                return False
        for subtable in ("FIELD", "SPECTRAL_WINDOW"):
            with ct.table(f"{ms_path}::{subtable}", readonly=True, ack=False) as table:
                if table.nrows() == 0:
                    return False
    except Exception:
        return False
    return True


def _working_set_group_ids(
    date: str,
    target_hour: int,
    start_hour: int,
    end_hour: int,
) -> set[str]:
    query = """
        SELECT group_id
        FROM hdf5_files
        WHERE timestamp_iso >= ? AND timestamp_iso < ?
        GROUP BY group_id
        HAVING COUNT(DISTINCT subband_num) = 16
        ORDER BY MIN(timestamp_iso)
    """
    with sqlite3.connect(DB_PATH) as connection:
        group_ids = [
            str(row[0])
            for row in connection.execute(
                query,
                (iso_bound(date, start_hour), iso_bound(date, end_hour)),
            )
        ]
    inventory_paths = [f"/inventory/{group_id}.ms" for group_id in group_ids]
    return {
        Path(path).stem
        for path in _select_ms_for_mosaic_hour(inventory_paths, target_hour)
    }


def _is_base_ms(path: Path) -> bool:
    try:
        datetime.strptime(path.stem, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return False
    return path.suffix == ".ms"


def demote_inactive_nvme_ms(date: str, active_stems: set[str]) -> None:
    for path in sorted(MS_DIR.glob(f"{date}T*.ms")):
        if not _is_base_ms(path) or path.stem in active_stems or path.is_symlink():
            continue
        target = PROC_MS_DIR / path.name
        if not target.is_dir() or not _measurement_set_is_readable(str(target)):
            continue
        temp_link = path.with_name(f".{path.name}.slow-link-{os.getpid()}")
        backup = path.with_name(f".{path.name}.nvme-backup-{os.getpid()}")
        if (
            temp_link.exists()
            or temp_link.is_symlink()
            or backup.exists()
            or backup.is_symlink()
        ):
            raise RuntimeError(f"refusing stale NVMe demotion path for {path.name}")
        temp_link.symlink_to(target)
        path.rename(backup)
        try:
            temp_link.rename(path)
        except Exception:
            backup.rename(path)
            temp_link.unlink(missing_ok=True)
            raise
        shutil.rmtree(backup)
        print(f"demoted inactive {path.name} to slow-storage symlink", flush=True)


def ensure_conversion_capacity(group_ids: set[str]) -> None:
    missing = [group_id for group_id in group_ids if not (MS_DIR / f"{group_id}.ms").exists()]
    if not missing:
        return
    samples = []
    for path in sorted(PROC_MS_DIR.glob("*.ms")):
        if _is_base_ms(path) and path.is_dir():
            samples.append(_allocated_bytes(path))
            if len(samples) == 3:
                break
    per_group = sorted(samples)[len(samples) // 2] if samples else 3 * 1024**3
    required = len(missing) * per_group
    free = shutil.disk_usage(MS_DIR).free
    if required + NVME_RESERVE_BYTES > free:
        raise RuntimeError(
            f"conversion working set estimates {required / 1024**3:.1f} GiB plus "
            f"{NVME_RESERVE_BYTES / 1024**3:.1f} GiB reserve; "
            f"only {free / 1024**3:.1f} GiB is free"
        )


def convert_window(date: str, target_hour: int, start_hour: int, end_hour: int) -> None:
    from dsa110_continuum.conversion.conversion_orchestrator import (
        convert_subband_groups_to_ms,
    )

    group_ids = _working_set_group_ids(date, target_hour, start_hour, end_hour)
    demote_inactive_nvme_ms(date, group_ids)
    ensure_conversion_capacity(group_ids)
    result = convert_subband_groups_to_ms(
        str(INPUT_DIR),
        str(MS_DIR),
        iso_bound(date, start_hour),
        iso_bound(date, end_hour),
        skip_incomplete=True,
        skip_existing=True,
        group_ids=group_ids,
    )
    if result["failed"]:
        raise RuntimeError(f"conversion failures: {result['failed']}")


def _active_ms_paths(
    date: str,
    target_hour: int,
    start_hour: int,
    end_hour: int,
) -> list[Path]:
    candidates = []
    for path in MS_DIR.glob(f"{date}T*.ms"):
        try:
            hour = datetime.strptime(path.stem, "%Y-%m-%dT%H:%M:%S").hour
        except ValueError:
            continue
        if start_hour <= hour < end_hour:
            candidates.append(str(path))
    return [Path(path) for path in _select_ms_for_mosaic_hour(candidates, target_hour)]


def _allocated_bytes(path: Path) -> int:
    total = 0
    for root, dirs, files in os.walk(path):
        for name in [*dirs, *files]:
            total += (Path(root) / name).lstat().st_blocks * 512
    return total


def promote_working_set_to_nvme(
    date: str,
    target_hour: int,
    start_hour: int,
    end_hour: int,
) -> None:
    paths = _active_ms_paths(date, target_hour, start_hour, end_hour)
    slow_links: list[tuple[Path, Path, int]] = []
    for path in paths:
        if not path.is_symlink():
            continue
        target = path.resolve(strict=True)
        if target.parent != PROC_MS_DIR:
            raise RuntimeError(f"refusing unexpected MS source target: {target}")
        slow_links.append((path, target, _allocated_bytes(target)))

    required = sum(size for _path, _target, size in slow_links)
    free = shutil.disk_usage(MS_DIR).free
    if required + NVME_RESERVE_BYTES > free:
        raise RuntimeError(
            f"NVMe working set needs {required / 1024**3:.1f} GiB plus "
            f"{NVME_RESERVE_BYTES / 1024**3:.1f} GiB reserve; "
            f"only {free / 1024**3:.1f} GiB is free"
        )

    for path, target, _size in slow_links:
        started = time.perf_counter()
        print(f"promoting {path.name} to NVMe", flush=True)
        temp = path.with_name(f".{path.name}.nvme-copy-{os.getpid()}")
        backup = path.with_name(f".{path.name}.slow-link-{os.getpid()}")
        if temp.exists() or temp.is_symlink() or backup.exists() or backup.is_symlink():
            raise RuntimeError(f"refusing stale NVMe promotion path for {path.name}")
        try:
            shutil.copytree(target, temp, symlinks=True)
            if not _measurement_set_is_readable(str(temp)):
                raise RuntimeError(f"copied Measurement Set is unreadable: {temp}")
            path.rename(backup)
            try:
                temp.rename(path)
            except Exception:
                backup.rename(path)
                raise
            backup.unlink()
            print(
                f"promoted {path.name} in {time.perf_counter() - started:.1f}s",
                flush=True,
            )
        finally:
            if temp.exists():
                shutil.rmtree(temp)


def batch_command(date: str, target_hour: int, start_hour: int, end_hour: int) -> list[str]:
    command = [
        "/opt/miniforge/envs/casa6/bin/python",
        "scripts/batch_pipeline.py",
        "--date",
        date,
        "--start-hour",
        str(start_hour),
        "--mosaic-hour",
        str(target_hour),
        "--rfi-mode",
        "conditional",
        "--quarantine-after-failures",
        "3",
        "--tile-timeout",
        "1800",
        "--retry-failed",
        "--tile-workers",
        "2",
        "--photometry-workers",
        "4",
        "--photometry-chunk-size",
        "0",
    ]
    if end_hour < 24:
        command.extend(["--end-hour", str(end_hour)])
    return command


def mosaic_paths(date: str, hour: int) -> tuple[Path, Path]:
    mosaic = IMAGE_DIR / f"mosaic_{date}" / f"{date}T{hour:02d}00_mosaic.fits"
    return mosaic, mosaic.with_suffix(".weights.fits")


def _manifest_paths(date: str, hour: int) -> tuple[Path, Path]:
    preserved = CAMPAIGN_DIR / f"{date}T{hour:02d}_{date}_manifest.json"
    current = PRODUCTS_MOSAIC_DIR / date / f"{date}_manifest.json"
    return preserved, current


def strict_qa_passed(date: str, hour: int) -> bool:
    for manifest_path in _manifest_paths(date, hour):
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for epoch in manifest.get("epochs", []):
            if int(epoch.get("hour", -1)) == hour:
                return epoch.get("status") == "ok" and epoch.get("qa_result") == "PASS"
    return False


def mosaic_is_valid(date: str, hour: int) -> bool:
    from dsa110_continuum.mosaic.production import weight_map_is_valid

    mosaic, weight = mosaic_paths(date, hour)
    return weight_map_is_valid(weight, mosaic) and strict_qa_passed(date, hour)


def preserve_run_metadata(date: str, hour: int) -> None:
    products = PRODUCTS_MOSAIC_DIR / date
    names = (f"{date}_manifest.json", f"{date}_run_summary.json", "run_report.md")
    missing = [name for name in names if not (products / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing run metadata for {date}T{hour:02d}: {missing}")
    for name in names:
        source = products / name
        shutil.copy2(source, CAMPAIGN_DIR / f"{date}T{hour:02d}_{name}")


def preserved_run_metadata_complete(date: str, hour: int) -> bool:
    prefix = CAMPAIGN_DIR / f"{date}T{hour:02d}_"
    manifest_path = Path(f"{prefix}{date}_manifest.json")
    summary_path = Path(f"{prefix}{date}_run_summary.json")
    report_path = Path(f"{prefix}run_report.md")
    if not all(path.is_file() for path in (manifest_path, summary_path, report_path)):
        return False
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    label = f"{date}T{hour:02d}00"
    summary_matches = any(
        epoch.get("label") == label
        and epoch.get("status") == "ok"
        and epoch.get("qa_result") == "PASS"
        for epoch in summary.get("epochs", [])
    )
    report_matches = f"| {hour:02d} | ok | PASS |" in report_path.read_text()
    return strict_qa_passed(date, hour) and summary_matches and report_matches


def photometry_path(date: str, hour: int) -> Path:
    return PRODUCTS_MOSAIC_DIR / date / f"{date}T{hour:02d}00_forced_phot.csv"


def accepted_artifacts_complete(date: str, hour: int) -> bool:
    forced_phot = photometry_path(date, hour)
    stacked = PRODUCTS_DIR / "lightcurves/lightcurves.parquet"
    return (
        mosaic_is_valid(date, hour)
        and preserved_run_metadata_complete(date, hour)
        and forced_phot.is_file()
        and forced_phot.stat().st_size > 0
        and stacked.is_file()
        and stacked.stat().st_mtime_ns >= forced_phot.stat().st_mtime_ns
    )


def accepted_products_ready(date: str, hour: int) -> tuple[bool, str | None]:
    try:
        if not preserved_run_metadata_complete(date, hour):
            preserve_run_metadata(date, hour)
        if not preserved_run_metadata_complete(date, hour):
            return False, "preserved run metadata does not match the accepted epoch"
    except (OSError, ValueError) as exc:
        return False, f"run metadata preservation failed: {exc}"
    forced_phot = photometry_path(date, hour)
    if not forced_phot.is_file() or forced_phot.stat().st_size == 0:
        return False, "forced-photometry product is missing or empty"
    if not refresh_lightcurves():
        return False, "light-curve stack was not produced"
    stacked = PRODUCTS_DIR / "lightcurves/lightcurves.parquet"
    if not stacked.is_file() or stacked.stat().st_mtime_ns < forced_phot.stat().st_mtime_ns:
        return False, "light-curve stack is older than forced photometry"
    return True, None


def refresh_lightcurves() -> bool:
    if not any((PRODUCTS_DIR / "mosaics").glob("*/*_forced_phot.csv")):
        return False
    run(
        [
            "/opt/miniforge/envs/casa6/bin/python",
            "scripts/stack_lightcurves.py",
            "--products-dir",
            str(PRODUCTS_DIR),
        ]
    )
    return (PRODUCTS_DIR / "lightcurves/lightcurves.parquet").is_file()


def _last_base_ms_stems(date: str, hour: int, count: int = 2) -> list[str]:
    stems = []
    for path in sorted(MS_DIR.glob(f"{date}T{hour:02d}:*.ms")):
        try:
            datetime.strptime(path.stem, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
        stems.append(path.stem)
    return stems[-count:]


def prune_hour(date: str, hour: int, retain_stems: set[str] | None = None) -> None:
    retain_stems = retain_stems or set()
    for path in MS_DIR.glob(f"{date}T{hour:02d}:*"):
        if any(path.name.startswith(stem) for stem in retain_stems):
            continue
        if path.name.endswith((".ms", ".ms.flagversions")):
            if path.is_symlink():
                target = path.resolve(strict=False)
                if target.parent != PROC_MS_DIR:
                    raise RuntimeError(f"refusing to prune unexpected MS target: {target}")
                if target.exists():
                    shutil.rmtree(target) if target.is_dir() else target.unlink()
                path.unlink()
            else:
                shutil.rmtree(path) if path.is_dir() else path.unlink()

            proc_copy = PROC_MS_DIR / path.name
            if proc_copy.exists():
                shutil.rmtree(proc_copy) if proc_copy.is_dir() else proc_copy.unlink()

    stage = IMAGE_DIR / f"mosaic_{date}"
    for path in stage.glob(f"{date}T{hour:02d}:*"):
        if any(path.name.startswith(stem) for stem in retain_stems):
            continue
        shutil.rmtree(path) if path.is_dir() else path.unlink()


def run_campaign(plan_only: bool) -> None:
    os.environ.update(
        PYTHONPATH=str(REPO),
        PIPELINE_DB=str(DB_PATH),
        CONTIMG_BASE_DIR=str(REPO),
        CONTIMG_STATE_DIR=str(REPO / "state"),
        CONTIMG_TMPFS_DIR="/dev/shm/dsa110-continuum",
        CONTIMG_SCRATCH_DIR="/dev/shm/dsa110-continuum",
        DSA110_CATALOG_DIR=str(REPO / "state/catalogs"),
        DSA110_MS_DIR=str(MS_DIR),
        DSA110_STAGE_IMAGE_BASE=str(IMAGE_DIR),
        DSA110_PRODUCTS_BASE=str(PRODUCTS_MOSAIC_DIR),
    )
    write_status(state="indexing", error=None)
    index_inventory()
    inventory = complete_hours()
    accepted_existing = [
        f"{date}T{hour:02d}00"
        for date, hours in inventory.items()
        for hour in hours
        if accepted_artifacts_complete(date, hour)
    ]
    status = write_status(state="planned", dates=inventory, completed=accepted_existing)
    completed_epochs = set(accepted_existing)
    failed_epochs = {
        str(item["epoch"]): item
        for item in status.get("failed_epochs", [])
        if isinstance(item, dict) and item.get("epoch")
    }
    if plan_only:
        return

    for date, hours in inventory.items():
        for index, target_hour in enumerate(hours):
            start_hour, end_hour = rolling_window(hours, index)
            epoch = f"{date}T{target_hour:02d}00"
            accepted = True
            if mosaic_is_valid(date, target_hour):
                print(f"validated existing mosaic {epoch}", flush=True)
            else:
                write_status(
                    state="converting",
                    date=date,
                    target_hour=target_hour,
                    window=[start_hour, end_hour],
                )
                convert_window(date, target_hour, start_hour, end_hour)
                promote_working_set_to_nvme(date, target_hour, start_hour, end_hour)
                command = batch_command(date, target_hour, start_hour, end_hour)
                write_status(state="dry_run", date=date, target_hour=target_hour)
                run([*command, "--dry-run"])
                write_status(state="imaging", date=date, target_hour=target_hour)
                run(command)
                if not mosaic_is_valid(date, target_hour):
                    accepted = False
                    failed_epochs[epoch] = {
                        "epoch": epoch,
                        "reason": "mosaic failed product integrity or strict QA",
                    }
                    write_status(
                        state="epoch_rejected",
                        date=date,
                        target_hour=target_hour,
                        failed_epochs=list(failed_epochs.values()),
                    )

            if accepted:
                accepted, reason = accepted_products_ready(date, target_hour)
                if not accepted:
                    failed_epochs[epoch] = {"epoch": epoch, "reason": reason}
                    completed_epochs.discard(epoch)
                    write_status(
                        state="epoch_rejected",
                        date=date,
                        target_hour=target_hour,
                        completed=sorted(completed_epochs),
                        failed_epochs=list(failed_epochs.values()),
                    )

            if accepted:
                failed_epochs.pop(epoch, None)
                completed_epochs.add(epoch)
                write_status(
                    state="validated",
                    date=date,
                    target_hour=target_hour,
                    completed=sorted(completed_epochs),
                    failed_epochs=list(failed_epochs.values()),
                )
            if index:
                previous_epoch = f"{date}T{hours[index - 1]:02d}00"
                if previous_epoch not in failed_epochs:
                    prune_hour(date, hours[index - 1])
            if accepted:
                retain_stems = (
                    set(_last_base_ms_stems(date, target_hour))
                    if index + 1 < len(hours)
                    else set()
                )
                prune_hour(date, target_hour, retain_stems)
    write_status(
        state="complete_with_failures" if failed_epochs else "complete",
        failed_epochs=list(failed_epochs.values()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    try:
        run_campaign(args.plan_only)
    except Exception as error:
        write_status(state="failed", error=str(error))
        raise


if __name__ == "__main__":
    sys.path.insert(0, str(REPO))
    main()
