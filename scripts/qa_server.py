"""DSA-110 continuum QA and operations dashboard."""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from dsa110_continuum.observability import control as pipeline_control
from dsa110_continuum.observability.hour_state import campaign_hour_logs
from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from matplotlib.colors import PowerNorm
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DashboardConfig:
    """Filesystem and campaign settings for one dashboard instance."""

    stage: Path = Path(os.environ.get("DSA110_STAGE", "/stage/dsa110-contimg"))
    products: Path = Path(
        os.environ.get("DSA110_PRODUCTS_BASE", "/data/dsa110-proc/products/mosaics")
    )
    incoming: Path = Path(os.environ.get("DSA110_INCOMING", "/data/incoming"))
    thumb_dir: Path = Path(os.environ.get("DSA110_QA_THUMBS", "/tmp/qa_thumbs"))
    campaign_outputs: Path = Path(
        os.environ.get(
            "DSA110_CAMPAIGN_OUTPUTS",
            "/data/dsa110-continuum/outputs/slowvis-mosaic-campaign-2026-07-14",
        )
    )
    campaign_date: str = os.environ.get("DSA110_CAMPAIGN_DATE", "2026-07-13")
    campaign_hour: int = int(os.environ.get("DSA110_CAMPAIGN_HOUR", "11"))


EPOCHS = [
    ("2026-01-25", "T0200"),
    ("2026-02-12", "T0000"),
    ("2026-02-15", "T0000"),
    ("2026-02-23", "T0000"),
    ("2026-02-25", "T0000"),
    ("2026-02-26", "T0000"),
]

DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
EPOCH_RE = re.compile(r"T\d{4}")
PROCESS_NAMES = ("batch_pipeline.py", "wsclean", "aoflagger")


def _config(request: Request) -> DashboardConfig:
    return request.app.state.dashboard_config


def _validate_date_epoch(date: str, epoch: str | None = None) -> None:
    if not DATE_RE.fullmatch(date) or (epoch is not None and not EPOCH_RE.fullmatch(epoch)):
        raise HTTPException(status_code=400, detail="Invalid date or epoch")


def find_mosaic(config: DashboardConfig, date: str, epoch: str) -> Path | None:
    """Return the canonical hourly-epoch mosaic path when it exists."""
    path = config.stage / f"images/mosaic_{date}" / f"{date}{epoch}_mosaic.fits"
    return path if path.is_file() else None


def find_csv(config: DashboardConfig, date: str, epoch: str) -> Path | None:
    """Return the newest matching forced-photometry table when present."""
    directory = config.products / date
    if not directory.is_dir():
        return None
    matches = list(directory.glob(f"*{epoch}*phot.csv"))
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def cal_tables(config: DashboardConfig) -> set[str]:
    """Return the observation prefixes with bandpass tables."""
    directory = config.stage / "ms"
    if not directory.is_dir():
        return set()
    return {path.stem.split("_")[0] for path in directory.glob("*.b")}


MOSAIC_NAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})(T\d{4})_mosaic\.fits")
PHOT_NAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})(T\d{4})_forced_phot\.csv")


def lightcurve_points(
    config: DashboardConfig, ra_deg: float, dec_deg: float, radius_arcsec: float = 30.0
) -> list[dict]:
    """Nearest forced-photometry match per epoch within the match radius."""
    points: list[dict] = []
    if not config.products.is_dir():
        return points
    radius_deg = radius_arcsec / 3600.0
    cos_dec = max(float(np.cos(np.radians(dec_deg))), 1e-6)
    for csv_path in sorted(config.products.glob("*/*_forced_phot.csv")):
        match = PHOT_NAME_RE.fullmatch(csv_path.name)
        if not match:
            continue
        try:
            frame = pd.read_csv(csv_path)
        except Exception as exc:
            logger.warning("Lightcurve CSV error %s: %s", csv_path, exc)
            continue
        if not {"ra_deg", "dec_deg", "dsa_peak_jyb"}.issubset(frame.columns):
            continue
        frame = frame[np.isfinite(frame["dsa_peak_jyb"])]
        if not len(frame):
            continue
        separation = np.hypot((frame["ra_deg"] - ra_deg) * cos_dec, frame["dec_deg"] - dec_deg)
        index = separation.idxmin()
        if not np.isfinite(separation[index]) or separation[index] > radius_deg:
            continue
        row = frame.loc[index]
        error = row.get("dsa_peak_err_jyb")
        points.append(
            {
                "epoch": match.group(1) + match.group(2),
                "date": match.group(1),
                "epoch_token": match.group(2),
                "flux_jy": float(row["dsa_peak_jyb"]),
                "flux_err_jy": float(error) if error is not None else float("nan"),
                "separation_arcsec": float(separation[index] * 3600.0),
                "csv": str(csv_path),
            }
        )
    points.sort(key=lambda point: point["epoch"])
    return points


def _lightcurve_metrics(points: list[dict]) -> dict:
    """Compute eta/V from the matched series via the canonical formulas."""
    if len(points) < 2:
        return {"eta": None, "v": None}
    from dsa110_continuum.photometry.metrics import calculate_eta_metric, calculate_v_metric

    fluxes = np.array([point["flux_jy"] for point in points], dtype=float)
    errors = np.array([point["flux_err_jy"] for point in points], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(np.isfinite(errors) & (errors > 0), 1.0 / errors**2, 1.0)
    try:
        return {
            "eta": float(calculate_eta_metric(fluxes, weights)),
            "v": float(calculate_v_metric(fluxes)),
        }
    except Exception as exc:
        logger.warning("Variability metric error: %s", exc)
        return {"eta": None, "v": None}


def _validate_lightcurve_query(ra: float, dec: float, radius_arcsec: float) -> None:
    if not 0 <= ra < 360:
        raise HTTPException(status_code=400, detail="ra must be in [0, 360)")
    if not -90 <= dec <= 90:
        raise HTTPException(status_code=400, detail="dec must be in [-90, 90]")
    if not 1 <= radius_arcsec <= 300:
        raise HTTPException(status_code=400, detail="radius_arcsec must be in [1, 300]")


def discover_epochs(config: DashboardConfig, limit: int = 24) -> list[tuple[str, str]]:
    """List (date, epoch) for every hourly-epoch mosaic on stage, newest first."""
    root = config.stage / "images"
    found = []
    if root.is_dir():
        for path in root.glob("mosaic_*/*_mosaic.fits"):
            match = MOSAIC_NAME_RE.fullmatch(path.name)
            if match:
                found.append((path.stat().st_mtime, match.group(1), match.group(2)))
    found.sort(reverse=True)
    discovered = [(date, epoch) for _, date, epoch in found[:limit]]
    return discovered or EPOCHS


def get_metrics(config: DashboardConfig, date: str, epoch: str) -> dict:
    """Measure the legacy QA fields for one hourly-epoch mosaic."""
    fits_path = find_mosaic(config, date, epoch)
    csv_path = find_csv(config, date, epoch)
    metrics = {
        "date": date,
        "epoch": epoch,
        "fits": fits_path is not None,
        "csv": csv_path is not None,
        "mosaic_path": str(fits_path) if fits_path else None,
        "photometry_path": str(csv_path) if csv_path else None,
        "thumbnail_url": f"/artifacts/mosaic/{date}/{epoch}/thumb.png",
        "peak": None,
        "rms": None,
        "dr": None,
        "ratio": None,
        "n_bright": 0,
        "status": "missing",
    }
    if not fits_path:
        return metrics
    try:
        with fits.open(fits_path, memmap=True) as hdus:
            data = hdus[0].data.squeeze().astype(np.float32)
        finite = data[np.isfinite(data)]
        metrics["peak"] = float(np.nanmax(finite)) if finite.size else None
        background = data[np.isfinite(data) & (np.abs(data) < 0.05)]
        metrics["rms"] = float(np.nanstd(background)) * 1000 if background.size else None
        if metrics["peak"] is not None and metrics["rms"]:
            metrics["dr"] = metrics["peak"] / (metrics["rms"] / 1000)
    except Exception as exc:
        logger.warning("FITS error %s %s: %s", date, epoch, exc)
    if csv_path:
        try:
            frame = pd.read_csv(csv_path)
            flux = frame.get("nvss_flux_jy", pd.Series(index=frame.index, dtype=float))
            bright = frame[flux > 0.06]
            metrics["n_bright"] = len(bright)
            if len(bright) and "dsa_nvss_ratio" in bright:
                metrics["ratio"] = float(bright["dsa_nvss_ratio"].median())
        except Exception as exc:
            logger.warning("CSV error %s %s: %s", date, epoch, exc)
    ratio = metrics["ratio"]
    if ratio is not None and not np.isfinite(ratio):
        metrics["ratio"] = None
        ratio = None
    if ratio is not None:
        metrics["status"] = "pass" if 0.8 <= ratio <= 1.2 else "fail"
    else:
        metrics["status"] = "no_phot"
    return metrics


def make_thumbnail(config: DashboardConfig, date: str, epoch: str) -> Path | None:
    """Build or reuse a cached PNG for one hourly-epoch mosaic."""
    fits_path = find_mosaic(config, date, epoch)
    if not fits_path:
        return None
    key = hashlib.md5(f"{fits_path}{fits_path.stat().st_mtime}".encode()).hexdigest()[:8]
    output = config.thumb_dir / f"{date}_{epoch}_{key}.png"
    if output.exists():
        return output
    for old in config.thumb_dir.glob(f"{date}_{epoch}_*.png"):
        old.unlink(missing_ok=True)
    try:
        with fits.open(fits_path, memmap=True) as hdus:
            data = hdus[0].data.squeeze().astype(np.float32)
        finite = data[np.isfinite(data)]
        if not finite.size:
            return None
        vmax = float(np.nanpercentile(finite, 99.5))
        background = data[np.isfinite(data) & (np.abs(data) < 0.05)]
        rms = float(np.nanstd(background)) if background.size else 0.0
        peak = float(np.nanmax(finite))
        figure, axis = plt.subplots(figsize=(12, 4), dpi=90)
        axis.imshow(
            data,
            origin="lower",
            cmap="inferno",
            norm=PowerNorm(gamma=0.35, vmin=0, vmax=max(vmax, np.finfo(float).eps)),
            aspect="auto",
        )
        dynamic_range = f"{peak / rms:.0f}" if rms else "--"
        axis.set_title(
            f"{date}  |  Peak {peak:.2f} Jy  RMS {rms * 1000:.1f} mJy/beam  DR {dynamic_range}",
            color="white",
            fontsize=10,
            pad=4,
        )
        axis.axis("off")
        figure.patch.set_facecolor("#111")
        plt.tight_layout(pad=0.2)
        plt.savefig(output, dpi=90, bbox_inches="tight", facecolor="#111")
        plt.close(figure)
        return output
    except Exception as exc:
        logger.error("Thumbnail error %s %s: %s", date, epoch, exc)
        return None


def _file_record(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def _latest(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.is_file()]
    return max(existing, key=lambda path: path.stat().st_mtime) if existing else None


def _tail(path: Path | None, lines: int = 24) -> list[str]:
    if path is None:
        return []
    try:
        with path.open(errors="replace") as stream:
            return stream.readlines()[-lines:]
    except OSError as exc:
        return [f"Unable to read {path}: {exc}"]


def disk_status() -> dict:
    """Report read-only capacity information for the data volumes."""
    result = {}
    for path in (Path("/stage"), Path("/data")):
        try:
            usage = shutil.disk_usage(path)
            result[str(path)] = {
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
                "pct_used": round(usage.used / usage.total * 100, 1),
            }
        except OSError as exc:
            result[str(path)] = {"error": str(exc)}
    return result


def process_status() -> list[dict]:
    """List only continuum pipeline processes relevant to operators."""
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
    processes = []
    for line in result.stdout.splitlines():
        if not any(name in line.lower() for name in PROCESS_NAMES):
            continue
        parts = line.strip().split(maxsplit=3)
        if len(parts) == 4:
            processes.append(
                {
                    "pid": int(parts[0]),
                    "elapsed_seconds": int(parts[1]),
                    "state": parts[2],
                    "command": parts[3],
                }
            )
    return processes


def _pid_hints(config: DashboardConfig) -> list[dict]:
    hints = []
    if not config.campaign_outputs.is_dir():
        return hints
    for path in sorted(config.campaign_outputs.glob("*.pid")):
        try:
            content = path.read_text(errors="replace")
        except OSError:
            continue
        for label, raw_pid in re.findall(r"([A-Z_]*PID)=(\d+)", content):
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


def campaign_status(config: DashboardConfig, date: str, hour: int) -> dict:
    """Collect current filesystem, log, disk, and process state for an epoch."""
    _validate_date_epoch(date)
    if not 0 <= hour <= 23:
        raise HTTPException(status_code=400, detail="Hour must be between 0 and 23")
    prefix = f"{date}T{hour:02d}"
    ms_dir = config.stage / "ms"
    image_dir = config.stage / f"images/mosaic_{date}"
    product_dir = config.products / date
    ms_files = sorted(ms_dir.glob(f"{prefix}:*.ms")) if ms_dir.is_dir() else []
    bandpass = sorted(ms_dir.glob(f"{prefix}:*.b")) if ms_dir.is_dir() else []
    gains = sorted(ms_dir.glob(f"{prefix}:*.g")) if ms_dir.is_dir() else []
    tile_images = sorted(image_dir.glob(f"{prefix}:*-image.fits")) if image_dir.is_dir() else []
    mosaic = image_dir / f"{date}T{hour:02d}00_mosaic.fits"
    incoming_files = (
        sorted(config.incoming.glob(f"{prefix}:*_sb*.hdf5")) if config.incoming.is_dir() else []
    )
    logs = list(product_dir.glob("run_*.log")) if product_dir.is_dir() else []
    if config.campaign_outputs.is_dir():
        logs.extend(campaign_hour_logs(config.campaign_outputs, hour))
    latest_log = _latest(list(dict.fromkeys(logs)))
    manifest = product_dir / f"{date}_manifest.json"
    summary = product_dir / f"{date}_run_summary.json"
    report = product_dir / "run_report.md"
    processes = process_status()
    stages = [
        {
            "name": "Measurement Sets",
            "state": "ready" if ms_files else "not_yet",
            "count": len(ms_files),
        },
        {
            "name": "Calibration tables",
            "state": "ready" if bandpass and gains else "not_yet",
            "count": len(bandpass) + len(gains),
        },
        {
            "name": "Tile images",
            "state": "ready" if tile_images else "not_yet",
            "count": len(tile_images),
        },
        {
            "name": "Hourly-epoch mosaic",
            "state": "ready" if mosaic.is_file() else "not_yet",
            "count": int(mosaic.is_file()),
        },
    ]
    return {
        "date": date,
        "hour": hour,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stages": stages,
        "measurement_sets": {"count": len(ms_files), "paths": [str(path) for path in ms_files]},
        "calibration": {
            "bandpass": [_file_record(path) for path in bandpass],
            "gains": [_file_record(path) for path in gains],
        },
        "tiles": {"count": len(tile_images), "latest": _file_record(_latest(tile_images))},
        "mosaic": _file_record(mosaic),
        "run_products": {
            "manifest": _file_record(manifest),
            "summary": _file_record(summary),
            "report": _file_record(report),
        },
        "log": {"file": _file_record(latest_log), "tail": _tail(latest_log)},
        "incoming": {
            "directory": str(config.incoming),
            "hdf5_count": len(incoming_files),
            "latest": _file_record(_latest(incoming_files)),
        },
        "disks": disk_status(),
        "processes": processes,
        "pid_hints": _pid_hints(config),
    }


def _format_number(value, decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:.{decimals}f}"


def _human_bytes(value: int | None) -> str:
    if value is None:
        return "—"
    number = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if number < 1024 or unit == "PB":
            return f"{number:.1f} {unit}"
        number /= 1024
    return "—"


def _badge(state: str, label: str | None = None) -> str:
    colors = {
        "pass": "#41c97a",
        "ready": "#41c97a",
        "active": "#4eb8ff",
        "fail": "#ff6470",
        "missing": "#69717d",
        "not_yet": "#d99b35",
        "no_phot": "#d99b35",
    }
    text = label or state.replace("_", " ")
    return f'<span class="badge" style="--badge:{colors.get(state, "#69717d")}">{html.escape(text.upper())}</span>'


def _ratio_cell(ratio) -> str:
    if ratio is None or (isinstance(ratio, float) and np.isnan(ratio)):
        return "—"
    color = "#41c97a" if 0.8 <= ratio <= 1.2 else "#ff6470"
    return f'<span style="color:{color};font-weight:700">{ratio:.3f}</span>'


_CONTROL_SCRIPT = """<script>
function controlToken(){return document.getElementById("control-token").value.trim();}
function showOutput(text){document.getElementById("run-output").textContent=text;}
async function postRuns(dryRun){
  const form=document.getElementById("run-form");
  const body={date:form.date.value,dry_run:dryRun};
  if(form.cal_date.value)body.cal_date=form.cal_date.value;
  if(form.start_hour.value)body.start_hour=parseInt(form.start_hour.value,10);
  if(form.end_hour.value)body.end_hour=parseInt(form.end_hour.value,10);
  if(form.rfi_mode.value)body.rfi_mode=form.rfi_mode.value;
  for(const name of["retry_failed","force_recal","skip_photometry","lenient_qa","clear_quarantine"]){
    if(form[name].checked)body[name]=true;
  }
  showOutput("Submitting...");
  try{
    const response=await fetch("/api/runs",{method:"POST",
      headers:{"Content-Type":"application/json","Authorization":"Bearer "+controlToken()},
      body:JSON.stringify(body)});
    const payload=await response.json();
    if(!response.ok){showOutput("Error "+response.status+": "+JSON.stringify(payload.detail));return;}
    showOutput(payload.plan?payload.plan:JSON.stringify(payload,null,2));
    if(!dryRun)setTimeout(()=>location.reload(),1500);
  }catch(error){showOutput("Request failed: "+error);}
}
async function terminateRun(runId){
  if(!confirm("Terminate run "+runId+"?"))return;
  try{
    const response=await fetch("/api/runs/"+runId+"/terminate",{method:"POST",
      headers:{"Authorization":"Bearer "+controlToken()}});
    const payload=await response.json();
    if(!response.ok){showOutput("Error "+response.status+": "+JSON.stringify(payload.detail));return;}
    showOutput(JSON.stringify(payload,null,2));
    setTimeout(()=>location.reload(),1500);
  }catch(error){showOutput("Request failed: "+error);}
}
</script>"""


def _render_control_section() -> str:
    """Render the pipeline-control panel: recent runs table + launch form."""
    try:
        runs = pipeline_control.list_runs(pipeline_control.ControlConfig(), limit=10)
    except Exception as exc:
        logger.warning("Run registry unavailable: %s", exc)
        runs = []
    if runs:
        run_rows = []
        for run in runs:
            state_key = {
                "running": "active",
                "succeeded": "pass",
                "failed": "fail",
                "terminated": "not_yet",
                "orphaned": "missing",
            }.get(run["status"], "missing")
            exit_code = run["exit_code"] if run["exit_code"] is not None else "—"
            action = (
                f"<button onclick=\"terminateRun('{html.escape(run['run_id'])}')\">"
                "Terminate</button>"
                if run["status"] == "running"
                else ""
            )
            run_rows.append(
                f'<tr><td><a href="/control/runs/{html.escape(run["run_id"])}">'
                f"{html.escape(run['run_id'])}</a></td>"
                f"<td>{_badge(state_key, run['status'])}</td>"
                f"<td>{html.escape(run['created_at'][:19])}</td>"
                f"<td>{exit_code}</td><td>{action}</td></tr>"
            )
        runs_table = (
            '<div class="table-wrap"><table><thead><tr><th>Run</th><th>Status</th>'
            "<th>Started (UTC)</th><th>Exit</th><th></th></tr></thead>"
            f"<tbody>{''.join(run_rows)}</tbody></table></div>"
        )
    else:
        runs_table = '<p class="empty-row">No launcher-owned runs recorded yet.</p>'
    form = """
<form id="run-form" onsubmit="return false" style="margin-top:14px">
<div class="info-grid">
<div class="info-card"><span>Date</span><input name="date" required pattern="\\d{4}-\\d{2}-\\d{2}" placeholder="YYYY-MM-DD"></div>
<div class="info-card"><span>Cal date</span><input name="cal_date" pattern="\\d{4}-\\d{2}-\\d{2}" placeholder="optional"></div>
<div class="info-card"><span>Hours</span><input name="start_hour" type="number" min="0" max="23" placeholder="start" style="width:45%">
<input name="end_hour" type="number" min="0" max="23" placeholder="end" style="width:45%"></div>
<div class="info-card"><span>RFI mode</span><select name="rfi_mode"><option value="">default</option><option>full</option><option>conditional</option><option>off</option></select></div>
</div>
<div style="margin:10px 0;display:flex;gap:16px;flex-wrap:wrap;font-size:.8rem">
<label><input type="checkbox" name="retry_failed"> retry failed</label>
<label><input type="checkbox" name="force_recal"> force recal</label>
<label><input type="checkbox" name="skip_photometry"> skip photometry</label>
<label><input type="checkbox" name="lenient_qa"> lenient QA</label>
<label><input type="checkbox" name="clear_quarantine"> clear quarantine</label>
</div>
<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
<input id="control-token" type="password" placeholder="control token" autocomplete="off">
<button onclick="postRuns(true)">Dry-run preview</button>
<button onclick="postRuns(false)" style="background:#274b63">Launch</button>
</div>
</form>
<pre id="run-output" style="margin-top:12px;min-height:40px"></pre>"""
    return (
        '<section class="campaign"><div class="campaign-head"><div>'
        "<h2>Pipeline control</h2>"
        "<p>Launch or re-run batch_pipeline.py — token required for mutating actions; "
        "dry-run preview shows the execution plan without writing anything</p></div></div>"
        f"{runs_table}{form}{_CONTROL_SCRIPT}</section>"
    )


def render_dashboard(config: DashboardConfig) -> str:
    """Render the consolidated QA and operations dashboard."""
    epochs = discover_epochs(config)
    all_metrics = [get_metrics(config, date, epoch) for date, epoch in epochs]
    cals = cal_tables(config)
    campaign = campaign_status(config, config.campaign_date, config.campaign_hour)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_pass = sum(metric["status"] == "pass" for metric in all_metrics)
    n_fail = sum(metric["status"] == "fail" for metric in all_metrics)
    n_missing = sum(metric["status"] in ("missing", "no_phot") for metric in all_metrics)

    rows = []
    cards = []
    for metric in all_metrics:
        has_cal = any(metric["date"] in name for name in cals)
        rows.append(
            f'<tr><td><b><a href="/runs/{metric["date"]}">{metric["date"]}</a></b>'
            f"<small>{metric['epoch']}</small></td>"
            f"<td>{_badge(metric['status'])}</td>"
            f"<td>{_badge('ready' if has_cal else 'missing', 'yes' if has_cal else 'no')}</td>"
            f"<td>{_format_number(metric['peak'])}</td>"
            f"<td>{_format_number(metric['rms'], 1)}</td>"
            f"<td>{_format_number(metric['dr'], 0)}</td>"
            f"<td>{_ratio_cell(metric['ratio'])}</td><td>{metric['n_bright']}</td></tr>"
        )
        if metric["fits"]:
            image = (
                f'<img src="{metric["thumbnail_url"]}" alt="{metric["date"]} mosaic" '
                'loading="lazy">'
            )
        else:
            image = '<div class="empty">No mosaic</div>'
        cards.append(
            f'<article class="mosaic-card status-{metric["status"]}"><header><b>{metric["date"]}</b>'
            f"{_badge(metric['status'])}</header>{image}<footer>Peak {_format_number(metric['peak'])} Jy"
            f"<span>RMS {_format_number(metric['rms'], 1)} mJy</span>"
            f"<span>Ratio {_ratio_cell(metric['ratio'])}</span></footer></article>"
        )

    stage_cards = []
    for stage in campaign["stages"]:
        stage_cards.append(
            f'<div class="stage"><span>{html.escape(stage["name"])}</span>'
            f"{_badge(stage['state'])}<strong>{stage['count']}</strong></div>"
        )
    disk_cards = []
    for path, disk in campaign["disks"].items():
        if "error" in disk:
            detail = html.escape(disk["error"])
            percent = "unknown"
        else:
            detail = (
                f"{_human_bytes(disk['free_bytes'])} free of {_human_bytes(disk['total_bytes'])}"
            )
            percent = f"{disk['pct_used']}% used"
        disk_cards.append(
            f'<div class="info-card"><span>{html.escape(path)}</span><strong>{percent}</strong><small>{detail}</small></div>'
        )
    processes = campaign["processes"]
    if processes:
        process_rows = "".join(
            f"<li><code>PID {process.get('pid', '—')}</code><span>{html.escape(process.get('command', process.get('error', '')))}</span></li>"
            for process in processes
        )
    elif campaign["pid_hints"]:
        process_rows = "".join(
            f"<li><code>PID {hint['pid']}</code><span>{html.escape(hint['label'])} recorded in "
            f"{html.escape(hint['source'])}; "
            f"{'visible' if hint['visible'] else 'not visible in the server process namespace'}</span></li>"
            for hint in campaign["pid_hints"]
        )
    else:
        process_rows = (
            '<li class="empty-row">No batch_pipeline, WSClean, or AOFlagger process visible</li>'
        )
    log_record = campaign["log"]["file"]
    log_title = html.escape(log_record["path"]) if log_record else "No campaign log found"
    log_tail = html.escape("".join(campaign["log"]["tail"]) or "No log lines available")
    mosaic_path = campaign["mosaic"]["path"] if campaign["mosaic"] else "Not yet produced"
    run_summary = campaign["run_products"]["summary"]
    run_summary_path = run_summary["path"] if run_summary else "Not yet produced"
    incoming = campaign["incoming"]
    control_section = _render_control_section()

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DSA-110 Continuum Observatory</title>
<style>
:root{{--ink:#e8edf2;--muted:#87919d;--panel:#171b20;--line:#2a3038;--blue:#4eb8ff;--green:#41c97a}}
*{{box-sizing:border-box}} body{{background:#0d1014;color:var(--ink);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0}}
body:before{{content:"";display:block;height:3px;background:linear-gradient(90deg,#4eb8ff,#41c97a 45%,#d99b35)}}
.shell{{max-width:1500px;margin:auto;padding:22px}} h1{{font-size:1.5rem;margin:0;letter-spacing:.01em}} h2{{font-size:1rem;margin:0 0 14px;text-transform:uppercase;letter-spacing:.12em;color:#bcc6d0}}
.subtitle{{color:var(--muted);font-size:.82rem;margin:5px 0 24px}} section{{margin-bottom:34px}} .summary,.stage-grid,.info-grid{{display:grid;gap:10px}}
.summary{{grid-template-columns:repeat(4,minmax(110px,1fr));max-width:620px}} .stat,.stage,.info-card{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:13px 16px}}
.stat strong{{display:block;font-size:1.7rem}} .stat span,.stage span,.info-card span{{color:var(--muted);font-size:.75rem;text-transform:uppercase;letter-spacing:.08em}}
.campaign{{border:1px solid #2f4553;background:linear-gradient(135deg,#131a20,#14181d);border-radius:12px;padding:18px}} .campaign-head{{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;margin-bottom:16px}}
.campaign-head h2{{margin:0;color:#dce9f2}} .campaign-head p{{margin:4px 0 0;color:var(--muted);font-size:.82rem}} .stage-grid{{grid-template-columns:repeat(4,1fr)}}
.stage{{display:grid;grid-template-columns:1fr auto;gap:10px;align-items:center}} .stage strong{{grid-column:1/-1;font-size:1.8rem}} .info-grid{{grid-template-columns:repeat(3,1fr);margin-top:10px}}
.info-card strong,.info-card small{{display:block;margin-top:7px}} .info-card small{{color:var(--muted)}} .path{{font-family:ui-monospace,SFMono-Regular,monospace;word-break:break-all;color:#a9c8dd}}
.badge{{display:inline-block;background:var(--badge);color:#081014;padding:3px 8px;border-radius:999px;font-size:.68rem;font-weight:800;letter-spacing:.05em;white-space:nowrap}}
.ops-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}} .panel{{background:#101419;border:1px solid var(--line);border-radius:8px;padding:14px;min-width:0}}
.panel h3{{font-size:.76rem;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin:0 0 10px}} .process-list{{list-style:none;padding:0;margin:0;max-height:180px;overflow:auto}} .process-list li{{display:flex;gap:10px;padding:7px 0;border-top:1px solid var(--line);font-size:.76rem}} .process-list li:first-child{{border:0}} .process-list span{{word-break:break-all;color:#bdc6cf}}
pre{{background:#090b0e;color:#b9c6d0;border-radius:6px;padding:12px;margin:0;max-height:250px;overflow:auto;font-size:.72rem;line-height:1.45;white-space:pre-wrap}} .log-path{{color:var(--muted);font-size:.68rem;word-break:break-all;margin-bottom:8px}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:9px}} table{{width:100%;border-collapse:collapse;font-size:.82rem}} th,td{{padding:10px 12px;text-align:center;border-bottom:1px solid var(--line);white-space:nowrap}} th{{background:#171b20;color:#9da8b4;font-weight:600}} td:first-child{{text-align:left}} td small{{display:block;color:var(--muted);margin-top:2px}} tr:last-child td{{border:0}} tr:hover{{background:#14191f}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px}} .mosaic-card{{background:var(--panel);border:1px solid var(--line);border-radius:9px;overflow:hidden}} .mosaic-card header,.mosaic-card footer{{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:9px 12px}} .mosaic-card img{{display:block;width:100%;min-height:140px;object-fit:cover;background:#090b0e}} .mosaic-card footer{{font-size:.75rem;color:#aeb7c0;flex-wrap:wrap}} .empty{{display:grid;place-items:center;min-height:140px;background:#101318;color:#5d6670}} .empty-row{{color:var(--muted)}}
@media(max-width:850px){{.stage-grid,.info-grid{{grid-template-columns:repeat(2,1fr)}}.ops-grid{{grid-template-columns:1fr}}}} @media(max-width:560px){{.shell{{padding:16px}}.summary,.stage-grid,.info-grid{{grid-template-columns:1fr 1fr}}.grid{{grid-template-columns:1fr}}}}
</style><script>setTimeout(()=>{{const token=document.getElementById("control-token");if(!token||!token.value)location.reload();}},30000)</script></head>
<body><main class="shell"><h1>DSA-110 Continuum Observatory</h1><div class="subtitle">Updated {now} · auto-refresh 30s · read-only</div>
<section class="campaign"><div class="campaign-head"><div><h2>Active campaign · {campaign["date"]} hour {campaign["hour"]:02d}</h2><p>Hourly-epoch mosaic progress from incoming HDF5 through science products</p></div>{_badge("active" if processes else "not_yet", "process visible" if processes else ("PID recorded" if campaign["pid_hints"] else "process not visible"))}</div>
<div class="stage-grid">{"".join(stage_cards)}</div><div class="info-grid">{"".join(disk_cards)}
<div class="info-card"><span>Incoming slow-vis</span><strong>{incoming["hdf5_count"]} HDF5</strong><small class="path">{html.escape(incoming["directory"])}</small></div></div>
<div class="info-grid"><div class="info-card"><span>MS root</span><strong>{campaign["measurement_sets"]["count"]} hour-11 MS</strong><small class="path">{html.escape(str(config.stage / "ms"))}</small></div>
<div class="info-card"><span>Tile root</span><strong>{campaign["tiles"]["count"]} image FITS</strong><small class="path">{html.escape(str(config.stage / f"images/mosaic_{campaign['date']}"))}</small></div>
<div class="info-card"><span>Hourly-epoch mosaic</span><strong>{"available" if campaign["mosaic"] else "not yet"}</strong><small class="path">{html.escape(mosaic_path)}</small></div>
<div class="info-card"><span>Run summary</span><strong>{"available" if run_summary else "not yet"}</strong><small class="path">{html.escape(run_summary_path)}</small></div></div>
<div class="ops-grid"><div class="panel"><h3>Pipeline heartbeat</h3><ul class="process-list">{process_rows}</ul></div>
<div class="panel"><h3>Latest batch log</h3><div class="log-path">{log_title}</div><pre>{log_tail}</pre></div></div></section>
{control_section}
<section><h2>Historical hourly-epoch QA</h2><div class="summary"><div class="stat"><strong style="color:#41c97a">{n_pass}</strong><span>Pass</span></div><div class="stat"><strong style="color:#ff6470">{n_fail}</strong><span>Fail</span></div><div class="stat"><strong style="color:#d99b35">{n_missing}</strong><span>Missing / pending photometry</span></div><div class="stat"><strong>{len(epochs)}</strong><span>Total</span></div></div>
<div class="table-wrap"><table><thead><tr><th>Date</th><th>Status</th><th>Cal tables</th><th>Peak (Jy)</th><th>RMS (mJy)</th><th>Dyn range</th><th>DSA/NVSS</th><th>Bright sources</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></section>
<section><h2>Mosaic thumbnails</h2><div class="grid">{"".join(cards)}</div></section>
<section><h2>Light curve lookup</h2>
<form action="/sources/lightcurve" method="get" style="display:flex;gap:10px;flex-wrap:wrap;align-items:center">
<input name="ra" type="number" step="any" min="0" max="359.9999" required placeholder="RA (deg)">
<input name="dec" type="number" step="any" min="-90" max="90" required placeholder="Dec (deg)">
<input name="radius_arcsec" type="number" step="any" min="1" max="300" value="30" title="match radius (arcsec)">
<button type="submit">Show light curve</button>
</form></section></main></body></html>"""


mosaic_router = APIRouter(prefix="/artifacts/mosaic", tags=["mosaic artifacts"])
ops_router = APIRouter(prefix="/api", tags=["read-only operations"])
legacy_router = APIRouter(include_in_schema=False)
control_router = APIRouter(prefix="/api/runs", tags=["pipeline control"])

CONTROL_TOKEN_ENV = "DSA110_CONTROL_TOKEN"


class RunRequestBody(BaseModel):
    """JSON body for launching or previewing a batch_pipeline run."""

    date: str
    cal_date: str | None = None
    start_hour: int | None = None
    end_hour: int | None = None
    rfi_mode: str | None = None
    tile_timeout: int | None = None
    quarantine_after_failures: int | None = None
    photometry_workers: int | None = None
    retry_failed: bool = False
    force_recal: bool = False
    skip_photometry: bool = False
    lenient_qa: bool = False
    clear_quarantine: bool = False
    dry_run: bool = False


def _require_control_token(request: Request) -> None:
    expected = os.environ.get(CONTROL_TOKEN_ENV, "")
    provided = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if not expected or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=403, detail="control token missing or invalid")


def _audit(
    config: pipeline_control.ControlConfig, action: str, request: Request, payload: dict
) -> None:
    config.control_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "time": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "remote": request.client.host if request.client else None,
        "payload": payload,
    }
    with (config.control_dir / "audit.jsonl").open("a") as stream:
        stream.write(json.dumps(entry) + "\n")


@control_router.get("")
def control_list_runs():
    """List the newest launcher-owned pipeline runs (read-only)."""
    return {"runs": pipeline_control.list_runs(pipeline_control.ControlConfig())}


@control_router.get("/{run_id}")
def control_get_run(run_id: str, tail: int = 40):
    """Return one run's registry record plus a bounded log tail (read-only)."""
    config = pipeline_control.ControlConfig()
    if not config.db_path.is_file():
        raise HTTPException(status_code=404, detail="unknown run")
    try:
        record = pipeline_control.get_run(run_id, config)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown run") from None
    record["log_tail"] = _tail(Path(record["log_path"]), max(1, min(tail, 500)))
    return record


@control_router.post("")
def control_launch(body: RunRequestBody, request: Request):
    """Launch batch_pipeline.py (or preview with dry_run) — token required."""
    _require_control_token(request)
    config = pipeline_control.ControlConfig()
    try:
        run_request = pipeline_control.RunRequest(**body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    _audit(config, "dry_run" if body.dry_run else "launch", request, body.model_dump())
    if body.dry_run:
        try:
            plan = pipeline_control.run_dry_run(run_request, config)
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="dry-run timed out") from None
        return {"plan": plan}
    try:
        return pipeline_control.launch_run(run_request, config)
    except pipeline_control.RunConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None


control_page_router = APIRouter(include_in_schema=False)


@control_page_router.get("/control/runs/{run_id}", response_class=HTMLResponse)
def control_run_page(run_id: str):
    """Render one launcher-owned run: registry record plus escaped log tail."""
    config = pipeline_control.ControlConfig()
    if not config.db_path.is_file():
        raise HTTPException(status_code=404, detail="run registry unavailable")
    try:
        record = pipeline_control.get_run(run_id, config)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown run") from None
    state_key = {
        "running": "active",
        "succeeded": "pass",
        "failed": "fail",
        "terminated": "not_yet",
        "orphaned": "missing",
    }.get(record["status"], "missing")
    log_tail = html.escape("".join(_tail(Path(record["log_path"]), 200)) or "No log output yet")
    argv = html.escape(" ".join(json.loads(record["argv_json"])))
    finished = record["finished_at"] or "—"
    exit_code = record["exit_code"] if record["exit_code"] is not None else "—"
    return HTMLResponse(
        f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Run {html.escape(run_id)}</title>
<style>body{{background:#0d1014;color:#e8edf2;font-family:Inter,-apple-system,sans-serif;margin:0}}
.shell{{max-width:1100px;margin:auto;padding:22px}} a{{color:#4eb8ff}}
table{{border-collapse:collapse;font-size:.85rem;margin:14px 0}} td,th{{padding:7px 14px;border-bottom:1px solid #2a3038;text-align:left}}
.badge{{display:inline-block;background:var(--badge);color:#081014;padding:3px 8px;border-radius:999px;font-size:.68rem;font-weight:800}}
pre{{background:#090b0e;color:#b9c6d0;border-radius:6px;padding:12px;max-height:560px;overflow:auto;font-size:.74rem;line-height:1.45;white-space:pre-wrap}}</style>
<script>setTimeout(()=>location.reload(),15000)</script></head>
<body><main class="shell"><p><a href="/">← Dashboard</a></p>
<h1>Run {html.escape(run_id)}</h1>
<table>
<tr><th>Status</th><td>{_badge(state_key, record["status"])}</td></tr>
<tr><th>Started (UTC)</th><td>{html.escape(record["created_at"])}</td></tr>
<tr><th>Finished (UTC)</th><td>{html.escape(str(finished))}</td></tr>
<tr><th>Exit code</th><td>{exit_code}</td></tr>
<tr><th>Command</th><td><code>{argv}</code></td></tr>
<tr><th>Log file</th><td><code>{html.escape(record["log_path"])}</code></td></tr>
</table>
<h2>Log tail</h2><pre>{log_tail}</pre></main></body></html>"""
    )


sources_router = APIRouter(prefix="/sources", tags=["science sources"])


@sources_router.get("/lightcurve", response_class=HTMLResponse)
def lightcurve_page(request: Request, ra: float, dec: float, radius_arcsec: float = 30.0):
    """Render the positional light curve across all epochs with metrics."""
    _validate_lightcurve_query(ra, dec, radius_arcsec)
    config = _config(request)
    points = lightcurve_points(config, ra, dec, radius_arcsec)
    metrics = _lightcurve_metrics(points)
    if points:
        point_rows = "".join(
            f'<tr><td><a href="/artifacts/mosaic/{point["date"]}/{point["epoch_token"]}/status">'
            f"{point['epoch']}</a></td>"
            f"<td>{point['flux_jy']:.4f}</td>"
            f"<td>{point['flux_err_jy']:.4f}</td>"
            f"<td>{point['separation_arcsec']:.1f}</td></tr>"
            for point in points
        )
        plot = (
            f'<img src="/sources/lightcurve.png?ra={ra}&dec={dec}'
            f'&radius_arcsec={radius_arcsec}" alt="light curve" '
            'style="max-width:100%;border-radius:8px">'
        )
    else:
        point_rows = '<tr><td colspan="4">No matching source in any epoch</td></tr>'
        plot = ""
    eta = f"{metrics['eta']:.3f}" if metrics["eta"] is not None else "—"
    v_metric = f"{metrics['v']:.3f}" if metrics["v"] is not None else "—"
    return HTMLResponse(
        f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Light curve {ra:.4f} {dec:+.4f}</title>
<style>body{{background:#0d1014;color:#e8edf2;font-family:Inter,-apple-system,sans-serif;margin:0}}
.shell{{max-width:1100px;margin:auto;padding:22px}} a{{color:#4eb8ff}} h2{{font-size:1rem;text-transform:uppercase;letter-spacing:.1em;color:#bcc6d0}}
table{{border-collapse:collapse;font-size:.83rem;margin:12px 0}} td,th{{padding:8px 14px;border-bottom:1px solid #2a3038;text-align:left}} th{{background:#171b20;color:#9da8b4}}</style></head>
<body><main class="shell"><p><a href="/">← Dashboard</a></p>
<h1>Light curve · RA {ra:.4f}° Dec {dec:+.4f}° (r={radius_arcsec:.0f}&Prime;)</h1>
<p>eta = {eta} · V = {v_metric} · {len(points)} epochs matched</p>
{plot}
<h2>Forced photometry</h2>
<table><thead><tr><th>Epoch</th><th>Flux (Jy/beam)</th><th>Error</th><th>Sep (&Prime;)</th></tr></thead>
<tbody>{point_rows}</tbody></table></main></body></html>"""
    )


@sources_router.get("/lightcurve.png")
def lightcurve_png(request: Request, ra: float, dec: float, radius_arcsec: float = 30.0):
    """Render the light-curve plot as PNG."""
    _validate_lightcurve_query(ra, dec, radius_arcsec)
    config = _config(request)
    points = lightcurve_points(config, ra, dec, radius_arcsec)
    if not points:
        return Response(status_code=404)
    labels = [point["epoch"] for point in points]
    fluxes = [point["flux_jy"] for point in points]
    errors = [
        point["flux_err_jy"] if np.isfinite(point["flux_err_jy"]) else 0.0 for point in points
    ]
    figure, axis = plt.subplots(figsize=(10, 4), dpi=100)
    axis.errorbar(range(len(points)), fluxes, yerr=errors, fmt="o-", color="#4eb8ff", capsize=3)
    axis.set_xticks(range(len(points)))
    axis.set_xticklabels(labels, rotation=30, ha="right", fontsize=7, color="white")
    axis.set_ylabel("Peak flux (Jy/beam)", color="white")
    axis.tick_params(colors="white")
    axis.set_title(f"RA {ra:.4f}  Dec {dec:+.4f}", color="white", fontsize=10)
    for spine in axis.spines.values():
        spine.set_color("#2a3038")
    axis.set_facecolor("#111")
    figure.patch.set_facecolor("#111")
    plt.tight_layout()
    from io import BytesIO

    buffer = BytesIO()
    plt.savefig(buffer, format="png", facecolor="#111")
    plt.close(figure)
    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={"Cache-Control": "max-age=60"},
    )


def _manifest_value(value, decimals: int | None = None) -> str:
    if value is None:
        return "—"
    if decimals is not None and isinstance(value, (int, float)):
        return f"{value:.{decimals}f}"
    return html.escape(str(value))


@control_page_router.get("/runs/{date}", response_class=HTMLResponse)
def run_provenance_page(date: str, request: Request):
    """Render one date's manifest: verdict, gates, epochs, and the run report."""
    _validate_date_epoch(date)
    config = _config(request)
    manifest_path = config.products / date / f"{date}_manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(status_code=404, detail="no manifest for this date")
    try:
        manifest = json.loads(manifest_path.read_text(errors="replace"))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"unreadable manifest: {exc}") from None
    if not isinstance(manifest, dict):
        manifest = {}
    verdict = str(manifest.get("pipeline_verdict") or "UNKNOWN")
    verdict_badge = _badge("pass" if verdict == "CLEAN" else "fail", verdict)
    gates = manifest.get("gates") or []
    gate_rows = (
        "".join(
            f"<tr><td>{_manifest_value(gate.get('gate'))}</td>"
            f"<td>{_manifest_value(gate.get('verdict'))}</td>"
            f"<td>{_manifest_value(gate.get('reason'))}</td></tr>"
            for gate in gates
            if isinstance(gate, dict)
        )
        or '<tr><td colspan="3">No gates triggered</td></tr>'
    )
    epoch_rows = []
    for epoch in manifest.get("epochs") or []:
        if not isinstance(epoch, dict):
            continue
        hour = epoch.get("hour")
        token = f"T{hour:02d}00" if isinstance(hour, int) else "—"
        link = (
            f'<a href="/artifacts/mosaic/{date}/{token}/status">{token}</a>'
            if token != "—"
            else token
        )
        qa_result = str(epoch.get("qa_result") or "—")
        qa_badge = _badge({"PASS": "pass", "FAIL": "fail"}.get(qa_result, "missing"), qa_result)
        epoch_rows.append(
            f"<tr><td>{link}</td><td>{_manifest_value(epoch.get('n_tiles'))}</td>"
            f"<td>{_manifest_value(epoch.get('status'))}</td><td>{qa_badge}</td>"
            f"<td>{_manifest_value(epoch.get('peak'), 2)}</td>"
            f"<td>{_manifest_value(epoch.get('rms'), 4)}</td>"
            f"<td>{_manifest_value(epoch.get('n_sources'))}</td>"
            f"<td>{_manifest_value(epoch.get('median_ratio'), 3)}</td></tr>"
        )
    epoch_table = "".join(epoch_rows) or '<tr><td colspan="8">No epochs recorded</td></tr>'
    report_path = config.products / date / "run_report.md"
    report_text = (
        html.escape(report_path.read_text(errors="replace"))
        if report_path.is_file()
        else "No run report found"
    )
    return HTMLResponse(
        f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Run {html.escape(date)}</title>
<style>body{{background:#0d1014;color:#e8edf2;font-family:Inter,-apple-system,sans-serif;margin:0}}
.shell{{max-width:1250px;margin:auto;padding:22px}} a{{color:#4eb8ff}} h2{{font-size:1rem;text-transform:uppercase;letter-spacing:.1em;color:#bcc6d0}}
table{{border-collapse:collapse;font-size:.83rem;margin:12px 0;width:100%}} td,th{{padding:8px 12px;border-bottom:1px solid #2a3038;text-align:left}} th{{background:#171b20;color:#9da8b4}}
.badge{{display:inline-block;background:var(--badge);color:#081014;padding:3px 8px;border-radius:999px;font-size:.68rem;font-weight:800}}
code{{color:#a9c8dd;word-break:break-all}}
pre{{background:#090b0e;color:#b9c6d0;border-radius:6px;padding:12px;max-height:520px;overflow:auto;font-size:.74rem;line-height:1.45;white-space:pre-wrap}}</style></head>
<body><main class="shell"><p><a href="/">← Dashboard</a></p>
<h1>Pipeline run · {html.escape(date)} {verdict_badge}</h1>
<p><code>{_manifest_value(manifest.get("command_line"))}</code></p>
<p>cal date {_manifest_value(manifest.get("cal_date"))} · git {_manifest_value(manifest.get("git_sha"))} · log {_manifest_value(manifest.get("run_log"))}</p>
<h2>QA gates</h2>
<table><thead><tr><th>Gate</th><th>Verdict</th><th>Reason</th></tr></thead><tbody>{gate_rows}</tbody></table>
<h2>Epochs</h2>
<table><thead><tr><th>Epoch</th><th>Tiles</th><th>Status</th><th>QA</th><th>Peak (Jy)</th><th>RMS (Jy)</th><th>Sources</th><th>Median ratio</th></tr></thead><tbody>{epoch_table}</tbody></table>
<h2>Run report</h2><pre>{report_text}</pre></main></body></html>"""
    )


@control_router.post("/{run_id}/terminate")
def control_terminate(run_id: str, request: Request):
    """Terminate a running pipeline run's whole process group — token required."""
    _require_control_token(request)
    config = pipeline_control.ControlConfig()
    _audit(config, "terminate", request, {"run_id": run_id})
    try:
        return pipeline_control.terminate_run(run_id, config)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown run") from None
    except pipeline_control.RunConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None


@mosaic_router.get("/{date}/{epoch}/status")
def mosaic_status(date: str, epoch: str, request: Request):
    """Return the data used by the mosaic dashboard card."""
    _validate_date_epoch(date, epoch)
    return get_metrics(_config(request), date, epoch)


@mosaic_router.get("/{date}/{epoch}/thumb.png")
def mosaic_thumbnail(date: str, epoch: str, request: Request):
    """Return a cached or newly rendered mosaic PNG."""
    _validate_date_epoch(date, epoch)
    path = make_thumbnail(_config(request), date, epoch)
    if not path:
        return Response(status_code=404)
    return Response(
        content=path.read_bytes(),
        media_type="image/png",
        headers={"Cache-Control": "max-age=60"},
    )


@legacy_router.get("/thumb/{date}/{epoch}.png")
def legacy_thumbnail(date: str, epoch: str, request: Request):
    """Preserve the original thumbnail URL."""
    return mosaic_thumbnail(date, epoch, request)


@ops_router.get("/status")
def operations_status(request: Request, date: str | None = None, hour: int | None = None):
    """Return the read-only campaign and telescope operations snapshot."""
    config = _config(request)
    return campaign_status(
        config,
        date or config.campaign_date,
        config.campaign_hour if hour is None else hour,
    )


def create_app(config: DashboardConfig | None = None) -> FastAPI:
    """Create a routed dashboard application."""
    dashboard_config = config or DashboardConfig()
    dashboard_config.thumb_dir.mkdir(parents=True, exist_ok=True)
    application = FastAPI(title="DSA-110 Continuum Observatory", version="2.0")
    application.state.dashboard_config = dashboard_config
    application.include_router(mosaic_router)
    application.include_router(ops_router)
    application.include_router(control_router)
    application.include_router(control_page_router)
    application.include_router(sources_router)
    application.include_router(legacy_router)

    @application.get("/", response_class=HTMLResponse)
    def dashboard():
        return HTMLResponse(render_dashboard(dashboard_config))

    @application.get("/health")
    def health():
        return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8767, log_level="warning")
