"""Read-only filesystem and process probes for one continuum observing hour."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class HourStateConfig:
    """Locations and epoch selected for an observability snapshot."""

    date: str = field(default_factory=lambda: os.environ.get("DSA110_OBSERVE_DATE", "2026-07-13"))
    hour: int = field(default_factory=lambda: int(os.environ.get("DSA110_OBSERVE_HOUR", "11")))
    stage: Path = field(
        default_factory=lambda: Path(os.environ.get("DSA110_STAGE", "/stage/dsa110-contimg"))
    )
    products: Path = field(
        default_factory=lambda: Path(
            os.environ.get("DSA110_PRODUCTS_BASE", "/data/dsa110-proc/products/mosaics")
        )
    )
    incoming: Path = field(
        default_factory=lambda: Path(os.environ.get("DSA110_INCOMING", "/data/incoming"))
    )
    campaign_outputs: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "DSA110_CAMPAIGN_OUTPUTS",
                "/data/dsa110-continuum/outputs/slowvis-mosaic-campaign-2026-07-14",
            )
        )
    )
    disk_roots: tuple[Path, ...] = (Path("/stage"), Path("/data"))

    def __post_init__(self) -> None:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.date):
            raise ValueError(f"Invalid observing date: {self.date}")
        try:
            datetime.strptime(self.date, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"Invalid observing date: {self.date}") from exc
        if not 0 <= self.hour <= 23:
            raise ValueError(f"Invalid observing hour: {self.hour}")


def _file_record(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def _latest(paths: list[Path]) -> Path | None:
    files = [path for path in paths if path.is_file()]
    return max(files, key=lambda path: path.stat().st_mtime) if files else None


def _tail(path: Path | None, lines: int = 24) -> list[str]:
    if path is None:
        return []
    try:
        with path.open(errors="replace") as stream:
            return [line.rstrip() for line in stream.readlines()[-lines:]]
    except OSError as exc:
        return [f"Unable to read {path}: {exc}"]


def _process_status() -> list[dict]:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,etimes=,stat=,args="],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [{"error": str(exc)}]
    matches = []
    for line in result.stdout.splitlines():
        lower = line.lower()
        if not any(
            marker in lower for marker in ("scripts/batch_pipeline", " wsclean ", " aoflagger ")
        ):
            continue
        fields = line.strip().split(maxsplit=3)
        if len(fields) == 4:
            matches.append(
                {
                    "pid": int(fields[0]),
                    "elapsed_seconds": int(fields[1]),
                    "state": fields[2],
                    "command": fields[3],
                }
            )
    return matches


def _pid_hints(directory: Path) -> list[dict]:
    hints = []
    if not directory.is_dir():
        return hints
    for path in sorted(directory.glob("*.pid")):
        try:
            content = path.read_text(errors="replace").strip()
        except OSError:
            continue
        labelled = re.findall(r"([A-Z_]*PID)=(\d+)", content)
        values = labelled or [
            (path.stem.upper(), value) for value in re.findall(r"\b\d+\b", content)
        ]
        for label, raw_pid in values:
            pid = int(raw_pid)
            hints.append(
                {
                    "label": label,
                    "pid": pid,
                    "visible": Path(f"/proc/{pid}").exists(),
                    "source": str(path),
                }
            )
    return hints


def _disk_status(paths: tuple[Path, ...]) -> dict:
    disks = {}
    for path in paths:
        try:
            usage = shutil.disk_usage(path)
            disks[str(path)] = {
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
                "pct_used": round(usage.used / usage.total * 100, 1),
            }
        except OSError as exc:
            disks[str(path)] = {"error": str(exc)}
    return disks


def campaign_hour_logs(directory: Path, hour: int) -> list[Path]:
    """Campaign logs whose filename hour equals `hour` (padded or unpadded form)."""
    pattern = re.compile(rf"batch_run_h0?{hour}(?!\d)")
    return sorted(path for path in directory.glob("batch_run_h*.log") if pattern.match(path.name))


def collect_hour_state(config: HourStateConfig | None = None) -> dict:
    """Return a bounded, read-only snapshot of one hourly-epoch campaign."""
    config = config or HourStateConfig()
    prefix = f"{config.date}T{config.hour:02d}"
    ms_dir = config.stage / "ms"
    image_dir = config.stage / f"images/mosaic_{config.date}"
    product_dir = config.products / config.date

    measurement_sets = sorted(ms_dir.glob(f"{prefix}:*.ms")) if ms_dir.is_dir() else []
    bandpass = sorted(ms_dir.glob(f"{prefix}:*.b")) if ms_dir.is_dir() else []
    gains = sorted(ms_dir.glob(f"{prefix}:*.g")) if ms_dir.is_dir() else []
    tiles = sorted(image_dir.glob(f"{prefix}:*-image.fits")) if image_dir.is_dir() else []
    mosaic = image_dir / f"{config.date}T{config.hour:02d}00_mosaic.fits"
    incoming = (
        sorted(config.incoming.glob(f"{prefix}:*_sb*.hdf5")) if config.incoming.is_dir() else []
    )
    logs = list(product_dir.glob("run_*.log")) if product_dir.is_dir() else []
    if config.campaign_outputs.is_dir():
        logs.extend(campaign_hour_logs(config.campaign_outputs, config.hour))
    latest_log = _latest(list(dict.fromkeys(logs)))

    state = {
        "date": config.date,
        "hour": config.hour,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "campaign": {
            "processes": _process_status(),
            "pid_hints": _pid_hints(config.campaign_outputs),
            "log": _file_record(latest_log),
            "log_tail": _tail(latest_log),
        },
        "measurement_sets": {
            "count": len(measurement_sets),
            "paths": [str(path) for path in measurement_sets],
        },
        "calibration": {
            "bandpass_count": len(bandpass),
            "bandpass_paths": [str(path) for path in bandpass],
            "gain_count": len(gains),
            "gain_paths": [str(path) for path in gains],
        },
        "tiles": {
            "count": len(tiles),
            "latest": _file_record(_latest(tiles)),
        },
        "mosaic": _file_record(mosaic),
        "incoming": {
            "count": len(incoming),
            "directory": str(config.incoming),
            "latest": _file_record(_latest(incoming)),
        },
        "run_products": {
            "manifest": _file_record(product_dir / f"{config.date}_manifest.json"),
            "summary": _file_record(product_dir / f"{config.date}_run_summary.json"),
            "report": _file_record(product_dir / "run_report.md"),
        },
        "disks": _disk_status(config.disk_roots),
    }
    process_visible = any("pid" in item for item in state["campaign"]["processes"])
    pid_visible = any(item["visible"] for item in state["campaign"]["pid_hints"])
    campaign_artifacts_present = state["campaign"]["log"] is not None or any(
        state["run_products"].values()
    )
    campaign_state = (
        "running"
        if process_visible or pid_visible
        else "finished"
        if campaign_artifacts_present
        else "absent"
    )
    state["campaign"]["state"] = campaign_state
    state["summary"] = {
        "campaign_state": campaign_state,
        "campaign_process_visible": process_visible,
        "campaign_pid_visible": pid_visible,
        "measurement_sets_present": bool(measurement_sets),
        "calibration_present": bool(bandpass and gains),
        "tiles_present": bool(tiles),
        "mosaic_present": state["mosaic"] is not None,
        "incoming_hdf5_present": bool(incoming),
    }
    return state
