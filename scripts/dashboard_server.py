#!/usr/bin/env python3
"""DSA-110 Pipeline Console — unified monitoring + control dashboard.

Tracer-bullet for the live observability stack (issues #48-#62): one routed
server that exposes every pipeline stage (ingest → conversion → calibration
→ imaging → mosaicking → QA → photometry) per date, with artifact browsing
and token-gated control actions.

Design rules
------------
- Read-only routes are open; every mutating route requires the
  ``X-DSA110-Token`` header to match ``DSA110_DASH_TOKEN``. If the env var
  is unset, control is DISABLED (fail closed) and the UI says so.
- No raw shell passthrough (unlike ``monitor_server.py`` ``POST /exec``).
  Control actions build fixed argv lists from validated fields.
- All filesystem roots are env-configurable so the server runs against
  H17 production paths by default and against synthetic trees in tests.

Run (H17)::

    PYTHONPATH=/data/dsa110-continuum DSA110_DASH_TOKEN=<secret> \
      /opt/miniforge/envs/casa6/bin/python -m uvicorn \
      scripts.dashboard_server:app --host 0.0.0.0 --port 8766

Run (anywhere, synthetic tree)::

    DSA110_INCOMING_DIR=/tmp/demo/data/incoming ... python -m uvicorn ...
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(title="DSA-110 Pipeline Console", version="0.1")

# --------------------------------------------------------------------------
# Configuration (env-overridable; defaults are H17 production paths)
# --------------------------------------------------------------------------


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


INCOMING_DIR = _env_path("DSA110_INCOMING_DIR", "/data/incoming")
MS_DIR = _env_path("DSA110_MS_DIR", "/stage/dsa110-contimg/ms")
IMAGE_BASE = _env_path("DSA110_IMAGE_BASE", "/stage/dsa110-contimg/images")
PRODUCTS_BASE = _env_path("DSA110_PRODUCTS_BASE", "/data/dsa110-proc/products/mosaics")
PIPELINE_DB = _env_path("PIPELINE_DB", "/data/dsa110-contimg/state/db/pipeline.sqlite3")
REPO_DIR = _env_path("DSA110_REPO_DIR", str(Path(__file__).resolve().parent.parent))
PIPELINE_PYTHON = os.environ.get("DSA110_PYTHON", "/opt/miniforge/envs/casa6/bin/python")
DASH_TOKEN = os.environ.get("DSA110_DASH_TOKEN", "")
JOB_DIR = _env_path("DSA110_JOB_DIR", "/tmp/dsa110_dash_jobs")
THUMB_DIR = _env_path("DSA110_THUMB_DIR", "/tmp/dsa110_dash_thumbs")
LOG_GLOBS = os.environ.get(
    "DSA110_LOG_GLOBS",
    "/tmp/batch_pipeline_*.log:/tmp/convert_*.log:/tmp/dsa110-convert/*.log:"
    "/data/dsa110-continuum/*.log:/data/dsa110-contimg/*.log",
).split(":")
DISK_PATHS = os.environ.get("DSA110_DISK_PATHS", "/:/data:/stage").split(":")

HDF5_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})_sb(\d+)\.hdf5$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
N_SUBBANDS = 16

_metrics_cache: dict[str, tuple[float, dict]] = {}
_cache_lock = threading.Lock()

# Load casatools before any request thread can import pandas/matplotlib:
# they resolve the system libstdc++, whose CXXABI is too old for casatools'
# libsakura, so a later casatools import fails (CXXABI_1.3.15 not found)
# and antenna health silently degrades to "none". Best-effort by design:
# hosts without the repo package or casatools just skip it.
try:  # pragma: no cover - environment-dependent
    import sys as _sys

    if str(REPO_DIR) not in _sys.path:
        _sys.path.insert(0, str(REPO_DIR))
    from dsa110_continuum.adapters import casa_tables as _casa_tables  # noqa: F401
except Exception:  # pragma: no cover
    pass

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def bytes_to_human(n: float) -> str:
    """Format a byte count as a human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _allowed_roots() -> list[Path]:
    roots = [INCOMING_DIR, MS_DIR, IMAGE_BASE, PRODUCTS_BASE, JOB_DIR]
    for pattern in LOG_GLOBS:
        parent = Path(pattern).parent
        if "*" not in str(parent):
            roots.append(parent)
    return roots


def _resolve_safe(raw: str) -> Path:
    """Resolve *raw* and require it to live under an allowed root."""
    p = Path(raw).resolve()
    for root in _allowed_roots():
        try:
            p.relative_to(root.resolve())
            return p
        except ValueError:
            continue
    raise HTTPException(status_code=403, detail="Path outside allowed roots")


def _require_token(x_dsa110_token: str | None) -> None:
    if not DASH_TOKEN:
        raise HTTPException(
            status_code=403,
            detail="Control disabled: DSA110_DASH_TOKEN is not set on the server",
        )
    if x_dsa110_token != DASH_TOKEN:
        raise HTTPException(status_code=403, detail="Bad or missing X-DSA110-Token")


def _fits_stats(path: Path) -> dict:
    """Peak/RMS/DR for a FITS image, cached on (path, mtime)."""
    key = str(path)
    mtime = path.stat().st_mtime
    with _cache_lock:
        hit = _metrics_cache.get(key)
        if hit and hit[0] == mtime:
            return hit[1]
    from astropy.io import fits as afits

    out: dict = {"peak": None, "rms_mjy": None, "dr": None}
    try:
        with afits.open(path, memmap=True) as hdul:
            d = np.asarray(hdul[0].data).squeeze().astype(np.float32)
        peak = float(np.nanmax(d))
        # RMS on a stride-4 subsample: indistinguishable for noise stats on
        # 4800-px mosaics, ~10x faster on the fuseblk mount. Peak stays full-res.
        sub = d[::4, ::4] if d.ndim == 2 and min(d.shape) >= 512 else d
        quiet = sub[np.abs(sub) < 0.05]
        rms = float(np.nanstd(quiet)) if quiet.size else float(np.nanstd(sub))
        out = {
            "peak": round(peak, 3),
            "rms_mjy": round(rms * 1000, 2) if np.isfinite(rms) else None,
            "dr": round(peak / rms) if rms else None,
        }
    except Exception as e:  # pragma: no cover - defensive
        out["error"] = str(e)
    with _cache_lock:
        _metrics_cache[key] = (mtime, out)
    return out


# --------------------------------------------------------------------------
# Per-stage scanners
# --------------------------------------------------------------------------


def scan_incoming() -> dict[str, dict]:
    """Summarize incoming HDF5 groups per date (timestamps, files, complete groups)."""
    out: dict[str, dict] = {}
    if not INCOMING_DIR.exists():
        return out
    groups: dict[str, dict[str, int]] = {}
    for f in INCOMING_DIR.iterdir():
        m = HDF5_RE.match(f.name)
        if not m:
            continue
        date, ts = m.group(1), m.group(2)
        groups.setdefault(date, {}).setdefault(ts, 0)
        groups[date][ts] += 1
    for date, tsmap in groups.items():
        out[date] = {
            "timestamps": len(tsmap),
            "files": sum(tsmap.values()),
            "complete_groups": sum(1 for n in tsmap.values() if n >= N_SUBBANDS),
        }
    return out


def scan_indexed(date: str) -> dict | None:
    """Summarize indexed HDF5 groups for one date from the pipeline database."""
    if not PIPELINE_DB.is_file():
        return None
    try:
        conn = sqlite3.connect(f"file:{PIPELINE_DB}?mode=ro", uri=True, timeout=1)
        row = conn.execute(
            """
            SELECT COALESCE(SUM(n_subbands), 0), COUNT(*),
                   COALESCE(SUM(CASE WHEN n_subbands >= ? THEN 1 ELSE 0 END), 0)
            FROM (
                SELECT group_id, COUNT(DISTINCT subband_num) AS n_subbands
                FROM hdf5_files
                WHERE obs_date = ? AND stored = 1
                GROUP BY group_id
            )
            """,
            (N_SUBBANDS, date),
        ).fetchone()
        conn.close()
    except (OSError, sqlite3.Error) as exc:
        logger.warning("could not read pipeline index for %s: %s", date, exc)
        return None
    return {
        "files": int(row[0]),
        "groups": int(row[1]),
        "complete_groups": int(row[2]),
    }


def scan_ms() -> dict[str, list[str]]:
    """Group Measurement Set names by observation date."""
    out: dict[str, list[str]] = {}
    if not MS_DIR.exists():
        return out
    for f in sorted(MS_DIR.glob("*.ms")):
        date = f.name[:10]
        if DATE_RE.match(date):
            out.setdefault(date, []).append(f.name)
    return out


def scan_cal() -> dict[str, dict]:
    """List bandpass (.b) and gain (.g) tables per date."""
    out: dict[str, dict] = {}
    if not MS_DIR.exists():
        return out
    for ext in ("b", "g"):
        for p in MS_DIR.glob(f"*.{ext}"):
            date = p.name[:10]
            if not DATE_RE.match(date):
                continue
            rec = out.setdefault(date, {"bandpass": [], "gain": []})
            rec["bandpass" if ext == "b" else "gain"].append(p.name)
    return out


def _stage_dir(date: str) -> Path:
    return IMAGE_BASE / f"mosaic_{date}"


def scan_tiles(date: str) -> dict:
    """Report per-tile FITS images and the tile checkpoint for a date."""
    sd = _stage_dir(date)
    if not sd.exists():
        return {
            "n_tiles": 0,
            "tiles": [],
            "n_artifacts": 0,
            "artifacts": [],
            "checkpoint": None,
        }
    artifacts = [f.name for f in sorted(sd.glob("*.fits")) if "_mosaic" not in f.name]
    ck = None
    ck_path = sd / ".tile_checkpoint.json"
    if ck_path.exists():
        try:
            ck = json.loads(ck_path.read_text())
        except Exception as e:
            ck = {"error": f"unreadable checkpoint: {e}"}
    completed = ck.get("completed", []) if isinstance(ck, dict) else []
    if completed:
        tiles = [Path(item).name for item in completed]
    else:
        tiles = [
            name
            for name in artifacts
            if name.endswith("-image-pb.fits") or name.endswith("_image.fits")
        ]
    return {
        "n_tiles": len(tiles),
        "tiles": tiles,
        "n_artifacts": len(artifacts),
        "artifacts": artifacts,
        "checkpoint": ck,
    }


def scan_mosaics(date: str) -> list[dict]:
    """List epoch mosaics (and weight-map presence) for a date."""
    sd = _stage_dir(date)
    out = []
    if not sd.exists():
        return out
    for p in sorted(sd.glob("*_mosaic.fits")):
        w = p.with_suffix(".weights.fits")
        out.append(
            {
                "name": p.name,
                "path": str(p),
                "size": bytes_to_human(p.stat().st_size),
                "weights": w.exists(),
                "mtime": datetime.utcfromtimestamp(p.stat().st_mtime).isoformat() + "Z",
            }
        )
    return out


def scan_products(date: str) -> dict:
    """Collect manifest, run summary, report, photometry CSVs and logs for a date."""
    pd_dir = PRODUCTS_BASE / date
    out: dict = {
        "dir": str(pd_dir),
        "manifest": None,
        "run_summary": None,
        "run_report": None,
        "phot_csvs": [],
        "archived_mosaics": [],
        "logs": [],
    }
    if not pd_dir.exists():
        return out
    man = pd_dir / f"{date}_manifest.json"
    if man.exists():
        try:
            out["manifest"] = json.loads(man.read_text())
            out["manifest_path"] = str(man)
        except Exception as e:
            out["manifest"] = {"error": str(e)}
    summ = pd_dir / f"{date}_run_summary.json"
    if summ.exists():
        try:
            out["run_summary"] = json.loads(summ.read_text())
        except Exception as e:
            out["run_summary"] = {"error": str(e)}
    rep = pd_dir / "run_report.md"
    if rep.exists():
        out["run_report"] = str(rep)
    for c in sorted(pd_dir.glob("*_forced_phot.csv")):
        rec = {"name": c.name, "path": str(c)}
        try:
            import pandas as pd

            df = pd.read_csv(c)
            rec["n_sources"] = int(len(df))
            if "dsa_nvss_ratio" in df.columns and "nvss_flux_jy" in df.columns:
                bright = df[df["nvss_flux_jy"] > 0.06]
                rec["n_bright"] = int(len(bright))
                if len(bright):
                    rec["median_ratio"] = round(float(bright["dsa_nvss_ratio"].median()), 3)
        except Exception as e:
            rec["error"] = str(e)
        out["phot_csvs"].append(rec)
    out["archived_mosaics"] = [p.name for p in sorted(pd_dir.glob("*_mosaic.fits"))]
    out["logs"] = [str(p) for p in sorted(pd_dir.glob("run_*.log"))]
    return out


def stage_states(
    incoming: dict | None,
    indexed: dict | None,
    n_ms: int,
    cal: dict,
    tiles: dict,
    n_mosaics: int,
    manifest: dict,
    n_phot: int,
    n_archived: int,
) -> list[dict]:
    """Return operator-facing state for every intake-to-science stage."""
    raw_files = int((incoming or {}).get("files", 0))
    indexed_files = int((indexed or {}).get("files", 0))
    indexed_groups = int((indexed or {}).get("groups", 0))
    indexed_complete = int((indexed or {}).get("complete_groups", 0))
    expected_ms = indexed_complete
    conversion_backlog = max(expected_ms - n_ms, 0)
    n_tiles = int(tiles.get("n_tiles", 0))
    failures = (tiles.get("checkpoint") or {}).get("failed", [])
    archive_blocked = any(
        gate.get("gate") == "archive" and gate.get("verdict") == "BLOCKED"
        for gate in manifest.get("gates", [])
    )
    verdict = manifest.get("pipeline_verdict")

    if raw_files == 0:
        ingest_state, ingest_detail = "waiting", "no raw HDF5 files visible"
    elif indexed is None or indexed_files < raw_files:
        ingest_state = "partial"
        ingest_detail = f"{raw_files} raw files; group completeness pending index"
    elif indexed_complete < indexed_groups:
        ingest_state = "partial"
        ingest_detail = f"{indexed_complete}/{indexed_groups} complete 16-subband groups"
    else:
        ingest_state = "complete"
        ingest_detail = f"{indexed_complete}/{indexed_groups} complete 16-subband groups"

    if indexed is None:
        index_state, index_detail = "unknown", "pipeline index unavailable"
    elif indexed_files == 0 and raw_files:
        index_state, index_detail = "waiting", f"0/{raw_files} raw files indexed"
    elif indexed_files < raw_files:
        index_state, index_detail = (
            "partial",
            f"{indexed_files}/{raw_files} files indexed; {indexed_complete} complete groups",
        )
    else:
        index_state, index_detail = (
            "complete",
            f"{indexed_files} files / {indexed_groups} groups indexed",
        )

    if n_ms and conversion_backlog == 0:
        conversion_state = "complete"
    elif n_ms:
        conversion_state = "partial"
    elif expected_ms:
        conversion_state = "waiting"
    else:
        conversion_state = "waiting"

    if n_tiles and n_tiles >= n_ms and not failures:
        imaging_state = "complete"
    elif n_tiles:
        imaging_state = "partial"
    elif failures:
        imaging_state = "failed"
    else:
        imaging_state = "waiting"

    if n_tiles:
        flag_state = "complete" if imaging_state == "complete" else "partial"
        flag_detail = f"inferred from {n_tiles} tiles reaching imaging"
    elif failures:
        flag_state, flag_detail = "failed", f"{len(failures)} recorded tile failures"
    else:
        flag_state, flag_detail = "waiting", "no standalone flagging artifact recorded"

    if cal.get("bandpass") and cal.get("gain"):
        cal_state = "complete"
    elif cal.get("bandpass") or cal.get("gain"):
        cal_state = "partial"
    else:
        cal_state = "waiting"

    qa_state = (
        "complete"
        if verdict == "CLEAN"
        else "blocked"
        if verdict == "DEGRADED"
        else "failed"
        if verdict == "FAILED"
        else "waiting"
    )
    phot_state = "complete" if n_phot else "blocked" if qa_state == "blocked" else "waiting"
    archive_state = "complete" if n_archived else "blocked" if archive_blocked else "waiting"
    return [
        {
            "key": "ingest",
            "label": "Raw intake",
            "state": ingest_state,
            "detail": ingest_detail,
        },
        {"key": "index", "label": "Index", "state": index_state, "detail": index_detail},
        {
            "key": "conversion",
            "label": "Conversion",
            "state": conversion_state,
            "detail": f"{n_ms}/{expected_ms} MS ready; {conversion_backlog} group backlog",
        },
        {
            "key": "flagging",
            "label": "Flagging",
            "state": flag_state,
            "detail": flag_detail,
            "inferred": True,
        },
        {
            "key": "calibration",
            "label": "Calibration",
            "state": cal_state,
            "detail": f"{cal.get('bandpass', 0)} bandpass / {cal.get('gain', 0)} gain tables",
        },
        {
            "key": "imaging",
            "label": "Imaging",
            "state": imaging_state,
            "detail": f"{n_tiles}/{n_ms} science tiles; {tiles.get('n_artifacts', 0)} FITS artifacts",
        },
        {
            "key": "mosaic",
            "label": "Mosaicking",
            "state": "complete" if n_mosaics else "waiting",
            "detail": f"{n_mosaics} hourly-epoch mosaics",
        },
        {
            "key": "qa",
            "label": "QA gate",
            "state": qa_state,
            "detail": verdict or "no manifest verdict",
        },
        {
            "key": "photometry",
            "label": "Photometry",
            "state": phot_state,
            "detail": f"{n_phot} forced-photometry products"
            if n_phot
            else "withheld by strict QA"
            if qa_state == "blocked"
            else "not run",
        },
        {
            "key": "archive",
            "label": "Archive",
            "state": archive_state,
            "detail": f"{n_archived} archived mosaics"
            if n_archived
            else "blocked by QA"
            if archive_blocked
            else "not archived",
        },
    ]


# --------------------------------------------------------------------------
# Read-only API
# --------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict:
    """Liveness probe with configured roots and control state."""
    return {
        "status": "ok",
        "time": _utcnow(),
        "control_enabled": bool(DASH_TOKEN),
        "roots": {
            "incoming": str(INCOMING_DIR),
            "ms": str(MS_DIR),
            "images": str(IMAGE_BASE),
            "products": str(PRODUCTS_BASE),
            "pipeline_db": str(PIPELINE_DB),
            "pipeline_repo": str(REPO_DIR),
        },
    }


@app.get("/api/system")
def system() -> dict:
    """Disk usage plus pipeline-related process list."""
    disk = {}
    for path in DISK_PATHS:
        try:
            u = shutil.disk_usage(path)
            disk[path] = {
                "total": bytes_to_human(u.total),
                "free": bytes_to_human(u.free),
                "pct_used": round(u.used / u.total * 100, 1),
            }
        except Exception as e:
            disk[path] = {"error": str(e)}
    procs: list[str] = []
    try:
        r = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=10)
        kw = ["wsclean", "dsa110", "casa", "bane", "aegean", "batch_pipeline"]
        procs = [
            line
            for line in r.stdout.splitlines()
            if any(k in line.lower() for k in kw)
            and "grep" not in line
            and "dashboard_server" not in line
        ]
    except Exception as e:
        procs = [f"error: {e}"]
    return {"time": _utcnow(), "disk": disk, "processes": procs, "n_processes": len(procs)}


@app.get("/api/dates")
def dates() -> dict:
    """Build the pipeline coverage matrix: per-date state of every stage."""
    incoming = scan_incoming()
    ms = scan_ms()
    cal = scan_cal()
    all_dates: set[str] = set(incoming) | set(ms) | set(cal)
    if IMAGE_BASE.exists():
        for p in IMAGE_BASE.glob("mosaic_*"):
            if p.is_dir() and DATE_RE.match(p.name[7:]):
                all_dates.add(p.name[7:])
    if PRODUCTS_BASE.exists():
        for p in PRODUCTS_BASE.iterdir():
            if p.is_dir() and DATE_RE.match(p.name):
                all_dates.add(p.name)

    rows = []
    for date in sorted(all_dates, reverse=True):
        indexed = scan_indexed(date)
        tiles = scan_tiles(date)
        mosaics = scan_mosaics(date)
        prod = scan_products(date)
        ck = tiles.get("checkpoint") or {}
        failures = ck.get("failed", []) if isinstance(ck, dict) else []
        manifest = prod.get("manifest") or {}
        cal_counts = {
            "bandpass": len(cal.get(date, {}).get("bandpass", [])),
            "gain": len(cal.get(date, {}).get("gain", [])),
        }
        stages = stage_states(
            incoming.get(date),
            indexed,
            len(ms.get(date, [])),
            cal_counts,
            tiles,
            len(mosaics),
            manifest,
            len(prod["phot_csvs"]),
            len(prod["archived_mosaics"]),
        )
        rows.append(
            {
                "date": date,
                "incoming": incoming.get(date),
                "indexed": indexed,
                "n_ms": len(ms.get(date, [])),
                "cal": cal_counts,
                "n_tiles": tiles["n_tiles"],
                "n_tile_artifacts": tiles["n_artifacts"],
                "n_failures": len(failures),
                "n_quarantine_risk": sum(
                    1 for f in failures if int(f.get("failure_count", 0)) >= 3
                ),
                "n_mosaics": len(mosaics),
                "verdict": manifest.get("pipeline_verdict") or None,
                "n_phot": len(prod["phot_csvs"]),
                "n_archived": len(prod["archived_mosaics"]),
                "stages": stages,
            }
        )
    return {"time": _utcnow(), "dates": rows}


@app.get("/api/date/{date}")
def date_detail(date: str) -> dict:
    """Full per-stage detail for one date, including FITS metrics."""
    if not DATE_RE.match(date):
        raise HTTPException(status_code=400, detail="Bad date format")
    incoming = scan_incoming().get(date)
    indexed = scan_indexed(date)
    ms = scan_ms().get(date, [])
    cal = scan_cal().get(date, {"bandpass": [], "gain": []})
    tiles = scan_tiles(date)
    mosaics = scan_mosaics(date)
    for m in mosaics:
        m.update(_fits_stats(Path(m["path"])))
    prod = scan_products(date)
    cal_counts = {"bandpass": len(cal["bandpass"]), "gain": len(cal["gain"])}
    stages = stage_states(
        incoming,
        indexed,
        len(ms),
        cal_counts,
        tiles,
        len(mosaics),
        prod.get("manifest") or {},
        len(prod["phot_csvs"]),
        len(prod["archived_mosaics"]),
    )
    return {
        "date": date,
        "time": _utcnow(),
        "incoming": incoming,
        "indexed": indexed,
        "ms": ms,
        "cal": cal,
        "tiles": tiles,
        "mosaics": mosaics,
        "products": prod,
        "stages": stages,
    }


@app.get("/api/thumb/{date}/{name}")
def thumb(date: str, name: str):
    """Render (and cache) a PNG thumbnail of a FITS product."""
    if not DATE_RE.match(date) or "/" in name or not name.endswith(".png"):
        raise HTTPException(status_code=400, detail="Bad thumb request")
    fits_path = _stage_dir(date) / name[:-4]
    if not fits_path.exists() or fits_path.suffix != ".fits":
        raise HTTPException(status_code=404, detail="No such FITS")
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(f"{fits_path}{fits_path.stat().st_mtime}".encode()).hexdigest()[:8]
    out = THUMB_DIR / f"{date}_{fits_path.stem}_{key}.png"
    if not out.exists():
        for old in THUMB_DIR.glob(f"{date}_{fits_path.stem}_*.png"):
            old.unlink(missing_ok=True)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from astropy.io import fits as afits
        from matplotlib.colors import PowerNorm

        try:
            with afits.open(fits_path, memmap=True) as hdul:
                d = np.asarray(hdul[0].data).squeeze().astype(np.float32)
            # Render at most ~2400 px on the long side; full-res imshow of a
            # 4800-px mosaic costs ~9 s of pure matplotlib resampling.
            if d.ndim == 2:
                step = max(1, max(d.shape) // 2400)
                d = d[::step, ::step]
            fig, ax = plt.subplots(figsize=(10, 4), dpi=80)
            finite = d[np.isfinite(d)]
            vmax = float(np.nanpercentile(finite, 99.5)) if finite.size else 1.0
            ax.imshow(
                d,
                origin="lower",
                cmap="inferno",
                norm=PowerNorm(gamma=0.35, vmin=0, vmax=max(vmax, 1e-6)),
                aspect="auto",
            )
            ax.axis("off")
            fig.patch.set_facecolor("#0d1017")
            plt.tight_layout(pad=0.1)
            plt.savefig(out, dpi=80, bbox_inches="tight", facecolor="#0d1017")
            plt.close(fig)
        except Exception as e:
            logger.error("thumb failed for %s: %s", fits_path, e)
            raise HTTPException(status_code=500, detail=f"Thumbnail failed: {e}")
    return Response(
        content=out.read_bytes(),
        media_type="image/png",
        headers={"Cache-Control": "max-age=60"},
    )


@app.get("/api/artifact")
def artifact(path: str = Query(...), lines: int = Query(200, le=5000)) -> Response:
    """Safe artifact viewer: text tail, CSV head, JSON, FITS header, PNG."""
    p = _resolve_safe(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    suffix = p.suffix.lower()
    if suffix == ".png":
        return Response(content=p.read_bytes(), media_type="image/png")
    if suffix == ".json":
        try:
            return JSONResponse({"path": str(p), "json": json.loads(p.read_text())})
        except Exception:
            pass  # fall through to text
    if suffix == ".fits":
        from astropy.io import fits as afits

        with afits.open(p, memmap=True) as hdul:
            header = repr(hdul[0].header)
        return JSONResponse({"path": str(p), "fits_header": header, **_fits_stats(p)})
    if suffix == ".csv":
        text = p.read_text(errors="replace").splitlines()
        return JSONResponse({"path": str(p), "n_lines": len(text), "head": text[: min(lines, 60)]})
    text = p.read_text(errors="replace").splitlines()
    return JSONResponse({"path": str(p), "n_lines": len(text), "tail": text[-lines:]})


@app.get("/api/logs")
def logs(lines: int = Query(80, le=2000)) -> dict:
    """Tail the most recent pipeline log matched by DSA110_LOG_GLOBS."""
    found: list[str] = []
    for pat in LOG_GLOBS:
        found.extend(glob.glob(pat))
    if not found:
        return {"message": "No log files found", "files": []}
    found.sort(key=os.path.getmtime, reverse=True)
    latest = found[0]
    tail = Path(latest).read_text(errors="replace").splitlines()[-lines:]
    return {
        "file": latest,
        "modified": datetime.utcfromtimestamp(os.path.getmtime(latest)).isoformat() + "Z",
        "tail": tail,
        "files": found[:20],
    }


# --------------------------------------------------------------------------
# Control (token-gated)
# --------------------------------------------------------------------------

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
RFI_MODES = {"conditional", "cflag", "full", "off"}


class RunRequest(BaseModel):
    """Validated parameters for launching the full flow or science batch."""

    date: str
    flow: str = "batch"
    rfi_mode: str = "conditional"
    start_hour: int | None = Field(default=None, ge=0, le=23)
    end_hour: int | None = Field(default=None, ge=0, le=23)
    dry_run: bool = True
    retry_failed: bool = False
    lenient_qa: bool = False
    clear_quarantine: bool = False
    photometry_workers: int | None = Field(default=None, ge=1, le=32)


def _pipeline_argv(req: RunRequest) -> list[str]:
    if req.flow not in {"batch", "end_to_end"}:
        raise HTTPException(status_code=400, detail="flow must be batch or end_to_end")
    if req.rfi_mode not in RFI_MODES:
        raise HTTPException(status_code=400, detail=f"rfi_mode must be one of {sorted(RFI_MODES)}")
    if req.flow == "end_to_end":
        if req.dry_run:
            raise HTTPException(status_code=400, detail="end-to-end flow has no dry-run mode")
        if req.start_hour is not None or req.end_hour is not None:
            raise HTTPException(
                status_code=400, detail="hour bounds apply only to science-batch mode"
            )
        script = str(REPO_DIR / "scripts" / "auto_pipeline.py")
        return [
            PIPELINE_PYTHON,
            script,
            "--date",
            req.date,
            "--rfi-mode",
            req.rfi_mode,
        ]
    script = str(REPO_DIR / "scripts" / "batch_pipeline.py")
    argv = [PIPELINE_PYTHON, script, "--date", req.date]
    argv += ["--rfi-mode", req.rfi_mode]
    if req.start_hour is not None:
        argv += ["--start-hour", str(req.start_hour)]
    if req.end_hour is not None:
        argv += ["--end-hour", str(req.end_hour)]
    if req.dry_run:
        argv.append("--dry-run")
    if req.retry_failed:
        argv.append("--retry-failed")
    if req.lenient_qa:
        argv.append("--lenient-qa")
    if req.clear_quarantine:
        argv.append("--clear-quarantine")
    if req.photometry_workers:
        argv += ["--photometry-workers", str(req.photometry_workers)]
    return argv


def _reap(job: dict) -> None:
    """Update job status once its process has exited."""
    proc: subprocess.Popen = job["proc"]
    rc = proc.poll()
    if rc is not None and job["status"] == "running":
        job["status"] = "completed" if rc == 0 else "failed"
        job["returncode"] = rc
        job["finished_at"] = _utcnow()


@app.post("/api/control/run")
def control_run(req: RunRequest, x_dsa110_token: str | None = Header(default=None)) -> dict:
    """Launch a validated pipeline flow as a tracked background job."""
    _require_token(x_dsa110_token)
    if not DATE_RE.match(req.date):
        raise HTTPException(status_code=400, detail="Bad date format")
    argv = _pipeline_argv(req)
    if not Path(argv[1]).is_file():
        raise HTTPException(status_code=503, detail=f"pipeline entrypoint missing: {argv[1]}")
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex[:12]
    log_path = JOB_DIR / f"{job_id}.log"
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(REPO_DIR))
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(argv, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    job = {
        "id": job_id,
        "kind": (
            "end-to-end" if req.flow == "end_to_end" else "dry-run" if req.dry_run else "batch-run"
        ),
        "argv": argv,
        "date": req.date,
        "log": str(log_path),
        "proc": proc,
        "pid": proc.pid,
        "status": "running",
        "started_at": _utcnow(),
    }
    with _jobs_lock:
        _jobs[job_id] = job
    logger.info("job %s started: %s", job_id, " ".join(argv))
    return {k: v for k, v in job.items() if k != "proc"}


@app.post("/api/control/clear-quarantine")
def control_clear_quarantine(req: dict, x_dsa110_token: str | None = Header(default=None)) -> dict:
    """Zero quarantine failure counts in a date's tile checkpoint (token-gated)."""
    _require_token(x_dsa110_token)
    date = str(req.get("date", ""))
    if not DATE_RE.match(date):
        raise HTTPException(status_code=400, detail="Bad date format")
    ck_path = _stage_dir(date) / ".tile_checkpoint.json"
    if not ck_path.exists():
        raise HTTPException(status_code=404, detail="No checkpoint for that date")
    ck = json.loads(ck_path.read_text())
    n = 0
    for rec in ck.get("failed", []):
        if int(rec.get("failure_count", 0)) != 0:
            rec["failure_count"] = 0
            n += 1
    tmp = str(ck_path) + ".tmp"
    with open(tmp, "w") as ck_f:
        json.dump(ck, ck_f, indent=2)
    os.replace(tmp, ck_path)
    return {"date": date, "cleared": n, "checkpoint": str(ck_path)}


@app.get("/api/jobs")
def jobs() -> dict:
    """List all jobs launched this server session."""
    with _jobs_lock:
        for job in _jobs.values():
            _reap(job)
        return {
            "jobs": [
                {k: v for k, v in j.items() if k != "proc"}
                for j in sorted(_jobs.values(), key=lambda j: j["started_at"], reverse=True)
            ]
        }


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str, lines: int = Query(100, le=2000)) -> dict:
    """Job record plus a tail of its captured log."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="No such job")
        _reap(job)
        out = {k: v for k, v in job.items() if k != "proc"}
    log_path = Path(out["log"])
    if log_path.exists():
        out["log_tail"] = log_path.read_text(errors="replace").splitlines()[-lines:]
    return out


@app.post("/api/jobs/{job_id}/kill")
def job_kill(job_id: str, x_dsa110_token: str | None = Header(default=None)) -> dict:
    """Terminate a running job (token-gated)."""
    _require_token(x_dsa110_token)
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="No such job")
        proc: subprocess.Popen = job["proc"]
        if proc.poll() is None:
            proc.terminate()
            job["status"] = "killed"
            job["finished_at"] = _utcnow()
    return {"id": job_id, "status": job["status"]}


# --------------------------------------------------------------------------
# Telescope state: sky, antennas (home page data)
# --------------------------------------------------------------------------

OVRO_LON_DEG = float(os.environ.get("DSA110_LON_DEG", "-118.2817"))
OVRO_LAT_DEG = float(os.environ.get("DSA110_LAT_DEG", "37.2339"))
DEC_STRIP_DEG = float(os.environ.get("DSA110_EXPECTED_DEC", "16.1"))
N_ANT = int(os.environ.get("DSA110_N_ANT", "110"))
ANT_STATUS_JSON = os.environ.get(
    "DSA110_ANT_STATUS_JSON", "/data/dsa110-continuum/state/ant_status.json"
)
VLA_CAL_DB = os.environ.get(
    "DSA110_VLA_CAL_DB", "/data/dsa110-contimg/state/catalogs/vla_calibrators.sqlite3"
)
MASTER_CAT_DB = os.environ.get("DSA110_MASTER_CAT_DB", "")

# Bright 1.4-GHz sky: A-team, flux standards / VLA calibrators, notable sources.
# (name, ra_deg, dec_deg, flux_jy, kind)
BRIGHT_SOURCES: list[tuple[str, float, float, float, str]] = [
    ("Cas A", 350.850, 58.815, 1700.0, "ateam"),
    ("Cyg A", 299.868, 40.734, 1579.0, "ateam"),
    ("Tau A", 83.633, 22.015, 875.0, "ateam"),
    ("Vir A", 187.706, 12.391, 212.0, "ateam"),
    ("Her A", 252.784, 4.992, 45.0, "ateam"),
    ("Hyd A", 139.524, -12.096, 43.0, "ateam"),
    ("Pic A", 79.957, -45.779, 66.0, "ateam"),
    ("For A", 50.674, -37.208, 150.0, "ateam"),
    ("3C84", 49.951, 41.512, 14.0, "src"),
    ("3C48", 24.422, 33.160, 16.0, "cal"),
    ("3C123", 69.268, 29.670, 47.0, "src"),
    ("3C138", 80.291, 16.639, 8.5, "cal"),
    ("3C147", 85.651, 49.852, 22.0, "cal"),
    ("3C161", 96.792, -5.885, 19.0, "src"),
    ("3C196", 123.400, 48.217, 14.0, "src"),
    ("3C273", 187.278, 2.052, 45.0, "src"),
    ("3C286", 202.785, 30.509, 15.0, "cal"),
    ("3C295", 212.836, 52.203, 22.0, "cal"),
    ("3C353", 260.117, -0.980, 57.0, "src"),
    ("3C380", 277.382, 48.746, 14.0, "src"),
    ("0834+555", 128.729, 55.570, 8.0, "cal"),
]


def _gmst_hours(when: datetime) -> float:
    """Greenwich mean sidereal time (hours) from a UTC datetime."""
    jd = when.timestamp() / 86400.0 + 2440587.5
    d = jd - 2451545.0
    return (18.697374558 + 24.06570982441908 * d) % 24.0


def lst_hours(when: datetime | None = None) -> float:
    """Local mean sidereal time at OVRO (hours)."""
    when = when or datetime.now(timezone.utc)
    return (_gmst_hours(when) + OVRO_LON_DEG / 15.0) % 24.0


def _sun_position() -> dict | None:
    """Return current Sun RA/Dec (deg); None if astropy is unavailable."""
    try:
        from astropy.coordinates import get_sun
        from astropy.time import Time

        s = get_sun(Time.now())
        return {
            "name": "Sun",
            "ra_deg": float(s.ra.deg),
            "dec_deg": float(s.dec.deg),
            "flux_jy": None,
            "kind": "sun",
        }
    except Exception as e:  # pragma: no cover - env without astropy coords
        logger.warning("sun position unavailable: %s", e)
        return None


def _catalog_sources(min_flux_jy: float = 2.0, limit: int = 400) -> list[dict]:
    """Flux-limited sources from optional catalog SQLite DBs (best-effort)."""
    out: list[dict] = []
    import sqlite3

    for db, kind in ((MASTER_CAT_DB, "catalog"), (VLA_CAL_DB, "cal")):
        if not db or not Path(db).exists():
            continue
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5.0)
            # Views included: the VLA calibrator DB exposes its per-band
            # flux join as the vla_20cm VIEW, not a table.
            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
                )
            ]
            for t in tables:
                cols = {r[1].lower() for r in conn.execute(f"PRAGMA table_info({t})")}
                ra_c = next((c for c in ("ra_deg", "ra") if c in cols), None)
                dec_c = next((c for c in ("dec_deg", "dec") if c in cols), None)
                fx_c = next(
                    (c for c in ("flux_jy", "flux_20_cm", "s1400", "flux") if c in cols), None
                )
                nm_c = next((c for c in ("name", "source", "j2000") if c in cols), None)
                if not (ra_c and dec_c and fx_c):
                    continue
                q = (
                    f"SELECT {nm_c or 'NULL'},{ra_c},{dec_c},{fx_c} FROM {t} "
                    f"WHERE {fx_c}>=? ORDER BY {fx_c} DESC LIMIT ?"
                )
                for nm, ra, dec, fx in conn.execute(q, (min_flux_jy, limit)):
                    out.append(
                        {
                            "name": nm or "",
                            "ra_deg": float(ra),
                            "dec_deg": float(dec),
                            "flux_jy": float(fx),
                            "kind": kind,
                        }
                    )
                break  # first usable table per DB
            conn.close()
        except Exception as e:
            logger.warning("catalog read failed for %s: %s", db, e)
    return out


def _latest_incoming_ts() -> str | None:
    """Most recent incoming HDF5 timestamp string, if any."""
    if not INCOMING_DIR.exists():
        return None
    latest = None
    for f in INCOMING_DIR.iterdir():
        m = HDF5_RE.match(f.name)
        if m:
            ts = f"{m.group(1)}T{m.group(2)}"
            if latest is None or ts > latest:
                latest = ts
    return latest


@app.get("/api/sky")
def sky() -> dict:
    """Report current sky state: LST, meridian, Dec strip, sources, Sun."""
    now = datetime.now(timezone.utc)
    lst = lst_hours(now)
    sources = [
        {"name": n, "ra_deg": r, "dec_deg": d, "flux_jy": f, "kind": k}
        for n, r, d, f, k in BRIGHT_SOURCES
    ]
    seen = {s["name"] for s in sources}
    for s in _catalog_sources():
        if s["name"] not in seen:
            sources.append(s)
    sun = _sun_position()
    if sun:
        sources.append(sun)
    return {
        "time": _utcnow(),
        "lst_hours": round(lst, 5),
        "meridian_ra_deg": round(lst * 15.0, 3),
        "dec_strip_deg": DEC_STRIP_DEG,
        "site": {"lon_deg": OVRO_LON_DEG, "lat_deg": OVRO_LAT_DEG},
        "latest_data_ts": _latest_incoming_ts(),
        "n_sources": len(sources),
        "sources": sources,
    }


def _ant_from_json() -> dict | None:
    p = Path(ANT_STATUS_JSON)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text())
        ants = raw.get("antennas", raw)
        recs = []
        if isinstance(ants, dict):
            for name, v in ants.items():
                if isinstance(v, str):
                    recs.append({"name": str(name), "status": v})
                else:
                    recs.append({"name": str(name), **v})
        else:
            recs = [dict(a) for a in ants]
        asof = raw.get("asof") if isinstance(raw, dict) else None
        asof = asof or datetime.utcfromtimestamp(p.stat().st_mtime).isoformat() + "Z"
        return {"source": "json", "asof": asof, "antennas": recs}
    except Exception as e:
        logger.warning("ant status json unreadable: %s", e)
        return None


def _ant_from_caltable() -> dict | None:
    """Per-antenna health from the newest bandpass table's solution flags."""
    if not MS_DIR.exists():
        return None
    tables = sorted(MS_DIR.glob("*.b"), key=lambda p: p.stat().st_mtime)
    if not tables:
        return None
    bp = tables[-1]
    try:
        from dsa110_continuum.adapters.casa_tables import table as _table

        recs: dict[int, list[float]] = {}
        with _table(str(bp)) as tb:
            ants = tb.getcol("ANTENNA1")
            flags = tb.getcol("FLAG")
        flags = np.asarray(flags)
        ants = np.asarray(ants).ravel()
        # collapse all non-row axes to a per-row flagged fraction
        row_axis = 0 if flags.shape[0] == ants.size else -1
        flat = (
            flags.reshape(flags.shape[0], -1)
            if row_axis == 0
            else flags.reshape(-1, flags.shape[-1]).T
        )
        for ant, row in zip(ants, flat):
            recs.setdefault(int(ant), []).append(float(np.mean(row)))
        antennas = []
        for ant in sorted(recs):
            ff = float(np.mean(recs[ant]))
            status = "good" if ff < 0.2 else ("warn" if ff < 0.8 else "bad")
            antennas.append({"name": str(ant + 1), "status": status, "flag_frac": round(ff, 3)})
        asof = datetime.utcfromtimestamp(bp.stat().st_mtime).isoformat() + "Z"
        return {"source": f"caltable:{bp.name}", "asof": asof, "antennas": antennas}
    except Exception as e:
        logger.info("caltable antenna health unavailable: %s", e)
        return None


@app.get("/api/antennas")
def antennas() -> dict:
    """Antenna liveness: ops JSON if present, else newest bandpass table flags."""
    data = _ant_from_json() or _ant_from_caltable()
    if data is None:
        data = {
            "source": "none",
            "asof": None,
            "antennas": [{"name": str(i + 1), "status": "unknown"} for i in range(N_ANT)],
        }
    else:
        known = {a["name"] for a in data["antennas"]}
        for i in range(N_ANT):
            if str(i + 1) not in known:
                data["antennas"].append({"name": str(i + 1), "status": "unknown"})
        data["antennas"].sort(key=lambda a: int(a["name"]) if str(a["name"]).isdigit() else 0)
    pos = _ant_positions()
    n_pos = 0
    for a in data["antennas"]:
        hit = pos.get(_ant_key(a["name"]))
        if hit:
            a.update(hit)
            n_pos += 1
    data["has_positions"] = n_pos >= 3
    counts: dict[str, int] = {}
    for a in data["antennas"]:
        counts[a["status"]] = counts.get(a["status"], 0) + 1
    data["counts"] = counts
    data["n"] = len(data["antennas"])
    return data


@app.get("/api/science")
def science(limit: int = Query(8, le=30)) -> dict:
    """Recent science products across dates: mosaics with QA, photometry summaries."""
    matrix = dates()["dates"]
    out: list[dict] = []
    for row in matrix[:limit]:
        date = row["date"]
        mosaics = scan_mosaics(date)
        prod = scan_products(date)
        man = prod.get("manifest") or {}
        epochs = man.get("epochs", []) if isinstance(man, dict) else []
        for m in mosaics:
            m.update(_fits_stats(Path(m["path"])))
            ep = next(
                (e for e in epochs if str(e.get("mosaic_path", "")).endswith(m["name"])), None
            )
            m["qa_result"] = ep.get("qa_result") if ep else None
        out.append(
            {
                "date": date,
                "verdict": row["verdict"],
                "n_tiles": row["n_tiles"],
                "mosaics": mosaics,
                "phot": prod["phot_csvs"],
                "run_report": prod.get("run_report"),
            }
        )
    return {"time": _utcnow(), "dates": out}


# --------------------------------------------------------------------------
# Antenna positions + lightcurves / variability
# --------------------------------------------------------------------------

ANTPOS_CSV = os.environ.get("DSA110_ANTPOS_CSV", "")
ANTPOS_GLOB = os.environ.get("DSA110_ANTPOS_GLOB", "/data/dsa110-antpos/ant_ids*.csv")


def _ant_key(name) -> str:
    """Normalize an antenna name to its digit core ('DSA-001', 1.0 -> '1')."""
    try:
        return str(int(float(name)))
    except (TypeError, ValueError):
        pass
    digits = "".join(ch for ch in str(name) if ch.isdigit())
    return str(int(digits)) if digits else str(name).strip()


def _antpos_cols(columns) -> dict | None:
    """Pick antenna-position columns from a header; None if unusable."""
    cols = {str(c).lower().strip(): c for c in columns}

    def pick(*names):
        return next((cols[n] for n in names if n in cols), None)

    sel = {
        "name": pick(
            "ant", "antenna", "ant_id", "ant_ids", "pad", "station", "station number", "name"
        ),
        "x": pick("x_m", "x", "east", "east_m", "easting"),
        "y": pick("y_m", "y", "north", "north_m", "northing"),
        "lat": pick("lat", "latitude", "lat_deg"),
        "lon": pick("lon", "longitude", "lon_deg", "long"),
    }
    if (sel["x"] and sel["y"]) or (sel["lat"] and sel["lon"]):
        return sel
    return None


def _ant_positions() -> dict[str, dict]:
    """Antenna positions keyed by normalized name, from a best-effort CSV read.

    Accepts x/y (m), east/north, or lat/lon columns; lat/lon are projected to
    local east-north metres about the array centroid. Excel-export CSVs that
    bury the header under title rows (e.g. DSA110_Station_Coordinates.csv)
    are handled by sniffing the first usable header line.
    """
    path = ANTPOS_CSV
    if not path:
        hits = sorted(glob.glob(ANTPOS_GLOB))
        path = hits[0] if hits else ""
    if not path or not Path(path).exists():
        return {}
    try:
        import pandas as pd

        df = pd.read_csv(path)
        sel = _antpos_cols(df.columns)
        if sel is None:
            with open(path, errors="replace") as fh:
                head = [fh.readline() for _ in range(40)]
            for i, line in enumerate(head):
                if _antpos_cols(line.split(",")) is not None:
                    df = pd.read_csv(path, header=i)
                    sel = _antpos_cols(df.columns)
                    break
        if sel is None:
            return {}
        name_c = sel["name"]
        out: dict[str, dict] = {}
        if sel["x"] and sel["y"]:
            df = df.dropna(subset=[sel["x"], sel["y"]])
            for i, (_, row) in enumerate(df.iterrows()):
                nm = _ant_key(row[name_c]) if name_c else str(i + 1)
                out[nm] = {"x_m": float(row[sel["x"]]), "y_m": float(row[sel["y"]])}
        else:
            lat_c, lon_c = sel["lat"], sel["lon"]
            df = df.dropna(subset=[lat_c, lon_c])
            lat0 = float(df[lat_c].mean())
            lon0 = float(df[lon_c].mean())
            kx = 111320.0 * np.cos(np.radians(lat0))
            for i, (_, row) in enumerate(df.iterrows()):
                nm = _ant_key(row[name_c]) if name_c else str(i + 1)
                out[nm] = {
                    "x_m": (float(row[lon_c]) - lon0) * kx,
                    "y_m": (float(row[lat_c]) - lat0) * 110540.0,
                }
        return out
    except Exception as e:
        logger.warning("antpos CSV unreadable (%s): %s", path, e)
        return {}


def _variability_table(max_dates: int = 30, top: int = 40) -> dict:
    """Stack per-epoch forced-photometry CSVs into per-source lightcurves.

    Uses the canonical eta/V formulas from ``dsa110_continuum.photometry.metrics``
    when importable, with equivalent local fallbacks otherwise.
    """
    import pandas as pd

    epochs: list[tuple[str, Path]] = []
    if PRODUCTS_BASE.exists():
        for ddir in sorted(PRODUCTS_BASE.iterdir(), reverse=True):
            if not (ddir.is_dir() and DATE_RE.match(ddir.name)):
                continue
            for c in sorted(ddir.glob("*_forced_phot.csv")):
                epochs.append((c.name.split("_forced_phot")[0], c))
            if len({e[0][:10] for e in epochs}) >= max_dates:
                break
    epochs.sort(key=lambda t: t[0])
    if len(epochs) < 2:
        return {"n_epochs": len(epochs), "sources": []}

    flux: dict[str, dict[str, tuple[float, float]]] = {}
    for label, csv_path in epochs:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        fx_c = next(
            (c for c in ("flux_jy", "dsa_peak_jyb", "measured_flux_jy") if c in df.columns), None
        )
        id_c = next((c for c in ("source_id", "source_name") if c in df.columns), None)
        if fx_c is None or id_c is None:
            continue
        er_c = next((c for c in ("flux_err_jy", "dsa_peak_err_jyb") if c in df.columns), None)
        err = df[er_c] if er_c else df[fx_c] * 0.1
        for sid, f, e in zip(df[id_c], df[fx_c], err):
            if np.isfinite(f):
                flux.setdefault(str(sid), {})[label] = (
                    float(f),
                    float(e) if np.isfinite(e) and e > 0 else 0.1,
                )

    try:
        import sys as _sys

        if str(REPO_DIR) not in _sys.path:
            _sys.path.insert(0, str(REPO_DIR))
        from dsa110_continuum.photometry.metrics import (
            calculate_eta_metric,
            calculate_v_metric,
        )

        def eta_fn(f, w):
            return float(calculate_eta_metric(f, w))

        def v_fn(f):
            return float(calculate_v_metric(f))

        canonical = True
    except Exception:  # pragma: no cover - repo package not importable

        def eta_fn(f, w):
            n = len(f)
            return (n / (n - 1)) * (np.mean(w * f**2) - np.mean(w * f) ** 2 / np.mean(w))

        def v_fn(f):
            return float(np.std(f) / np.mean(f)) if np.mean(f) else 0.0

        canonical = False

    labels = [label for label, _ in epochs]
    rows = []
    for sid, series in flux.items():
        if len(series) < 2:
            continue
        f = np.array([series[la][0] for la in labels if la in series])
        e = np.array([max(series[la][1], 1e-6) for la in labels if la in series])
        w = 1.0 / e**2
        mean_f = float(np.mean(f))
        if mean_f <= 0:
            continue
        rows.append(
            {
                "source_id": sid,
                "n_epochs": int(len(f)),
                "mean_flux_jy": round(mean_f, 4),
                "v": round(v_fn(f), 4),
                "eta": round(eta_fn(f, w), 3),
                "epochs": [la for la in labels if la in series],
                "flux_jy": [round(float(x), 5) for x in f],
                "flux_err_jy": [round(float(x), 5) for x in e],
            }
        )
    rows.sort(key=lambda r: r["eta"], reverse=True)
    return {
        "n_epochs": len(epochs),
        "epoch_labels": labels,
        "canonical_metrics": canonical,
        "n_sources": len(rows),
        "sources": rows[:top],
    }


@app.get("/api/variability")
def variability(top: int = Query(40, le=200)) -> dict:
    """Top variable sources (eta-ranked) from stacked forced-photometry epochs."""
    return _variability_table(top=top)


@app.get("/api/lightcurve/{source_id}")
def lightcurve(source_id: str) -> dict:
    """Full stacked lightcurve for one source across all epochs."""
    table = _variability_table(top=100000)
    for row in table["sources"]:
        if row["source_id"] == source_id:
            return row
    raise HTTPException(status_code=404, detail=f"No lightcurve for {source_id}")


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

PIPELINE_PAGE = r"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DSA-110 Pipeline Console</title>
<style>
:root{
  --bg:#0b0d10;--surface:#0f1216;--card:#12161c;--line:#1d222b;--line2:#262d38;
  --tx:#e7ebf3;--mut:#8b95a7;--dim:#57607000;--dim2:#5a6474;
  --acc:#5aa2ff;--ok:#3ecf8e;--warn:#d9a13f;--bad:#e0564f;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
html{-webkit-font-smoothing:antialiased}
body{margin:0;background:var(--bg);color:var(--tx);
  font:13.5px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
a{color:var(--acc);text-decoration:none;cursor:pointer}
a:hover{text-decoration:underline;text-underline-offset:3px}

/* ---------- header ---------- */
header{display:flex;align-items:baseline;gap:14px;padding:20px 28px 0;max-width:1560px;margin:0 auto}
header .mark{font-size:15px;font-weight:650;letter-spacing:.2px}
header .mark span{color:var(--mut);font-weight:400}
header .spacer{flex:1}
header .meta{font:11.5px var(--mono);color:var(--dim2)}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:6px;
  vertical-align:1px;background:var(--dim2)}
.dot.ok{background:var(--ok)}.dot.bad{background:var(--bad)}.dot.warn{background:var(--warn)}
.dot.run{background:var(--warn);animation:pulse 1.6s ease-in-out infinite}
@keyframes pulse{50%{opacity:.35}}

/* ---------- layout ---------- */
main{max-width:1560px;margin:0 auto;padding:10px 28px 48px}
section{margin-top:26px}
.microlabel{font-size:10.5px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;
  color:var(--dim2);margin:0 0 10px}
.rule{border-top:1px solid var(--line);margin:26px 0 0}
.grid2{display:grid;grid-template-columns:5fr 6fr;gap:36px}
@media(max-width:1100px){.grid2{grid-template-columns:1fr}}

/* ---------- system strip ---------- */
.sys{display:flex;align-items:center;gap:28px;flex-wrap:wrap;padding:14px 0 0;
  font:12px var(--mono);color:var(--mut)}
.sys .d{display:flex;align-items:center;gap:10px}
.sys .lbl{color:var(--dim2)}
.meter{width:110px;height:3px;border-radius:2px;background:var(--line2);overflow:hidden}
.meter i{display:block;height:100%;background:var(--acc);opacity:.85}
.meter i.hot{background:var(--bad)}
.sys .err{color:var(--bad);opacity:.8}

/* ---------- matrix ---------- */
table{width:100%;border-collapse:collapse}
.matrix th{font-size:10.5px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
  color:var(--dim2);text-align:right;padding:6px 14px;border-bottom:1px solid var(--line)}
.matrix th:first-child,.matrix th.l{text-align:left;padding-left:10px}
.matrix td{padding:9px 14px;border-bottom:1px solid var(--line);text-align:right;
  font:12.5px var(--mono);font-variant-numeric:tabular-nums;color:var(--tx)}
.matrix td:first-child{text-align:left;padding-left:10px;font-weight:600}
.matrix td.l{text-align:left}
.matrix tbody tr{cursor:pointer;transition:background .12s}
.matrix tbody tr:hover{background:var(--surface)}
.matrix tbody tr.sel{background:var(--surface);box-shadow:inset 2px 0 0 var(--acc)}
.matrix .z{color:var(--dim2)}
.matrix .neg{color:var(--bad)}
.matrix .att{color:var(--warn)}
.frac{color:var(--mut)}.frac b{color:var(--tx);font-weight:500}

/* stage glyph */
.stages{display:inline-flex;gap:4px}
.seg{width:15px;height:5px;border-radius:2.5px;background:var(--line2)}
.seg.ok{background:var(--acc);opacity:.9}
.seg.good{background:var(--ok)}
.seg.warn{background:var(--warn)}
.seg.bad{background:var(--bad)}

.verdict{font:11px var(--mono);letter-spacing:.06em}
.v-clean{color:var(--ok)}.v-degraded{color:var(--warn)}.v-failed{color:var(--bad)}.v-none{color:var(--dim2)}

/* ---------- detail ---------- */
#detail{display:none}
.kv{display:grid;grid-template-columns:150px 1fr;row-gap:7px;font-size:12.5px}
.kv dt{color:var(--mut)}
.kv dd{margin:0;font:12.5px var(--mono);color:var(--tx);overflow-wrap:anywhere}
.subsec{margin-top:22px}
.quiet{color:var(--dim2);font-size:12.5px}

.failrow{display:grid;grid-template-columns:1fr auto auto;gap:14px;align-items:baseline;
  padding:8px 0;border-bottom:1px solid var(--line);font:12px var(--mono)}
.failrow .err{color:var(--mut);grid-column:1/-1;font-size:11.5px;padding-left:2px}
.count{font-weight:700}
.count.bad{color:var(--bad)}.count.warn{color:var(--warn)}

.gate{padding:7px 0;border-bottom:1px solid var(--line);font-size:12.5px}
.gate b{font-weight:600}
.gate .why{color:var(--mut)}
.stage-row{display:grid;grid-template-columns:110px 70px 1fr;gap:12px;padding:6px 0;
  border-bottom:1px solid var(--line);font-size:12px}
.stage-row .state{font:10.5px var(--mono);letter-spacing:.05em;text-transform:uppercase}
.stage-row .complete{color:var(--ok)}.stage-row .partial{color:var(--warn)}
.stage-row .blocked,.stage-row .failed{color:var(--bad)}
.stage-row .waiting,.stage-row .unknown{color:var(--dim2)}
.stage-row .detail{color:var(--mut);font:11.5px var(--mono)}

.chips{display:flex;flex-wrap:wrap;gap:8px}
.chip{font:11.5px var(--mono);color:var(--tx);background:var(--card);border:1px solid var(--line2);
  border-radius:6px;padding:4px 10px;cursor:pointer;transition:border-color .12s}
.chip:hover{border-color:var(--acc);text-decoration:none}

/* mosaic cards */
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
.mcard{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.mcard .hd{display:flex;align-items:center;gap:8px;padding:8px 12px;font:11.5px var(--mono)}
.mcard .hd .name{color:var(--tx)}
.mcard .hd .qa{margin-left:auto;font-size:10.5px;letter-spacing:.08em}
.mcard img{width:100%;display:block;min-height:56px;background:#000}
.mcard .ft{display:flex;gap:14px;padding:7px 12px;font:11px var(--mono);color:var(--mut)}
.mcard .ft b{color:var(--tx);font-weight:500}

/* photometry */
.phot td,.phot th{padding:7px 12px;border-bottom:1px solid var(--line);text-align:right;
  font:12px var(--mono);font-variant-numeric:tabular-nums}
.phot th{font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--dim2);font-family:inherit}
.phot td:first-child,.phot th:first-child{text-align:left;padding-left:0}
.ratio.ok{color:var(--ok)}.ratio.bad{color:var(--bad)}

/* viewer */
pre{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px;
  font:11.5px/1.6 var(--mono);color:#c6cedd;max-height:360px;overflow:auto;white-space:pre-wrap;margin:0}

/* ---------- control ---------- */
.ctl{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-end}
.f label{display:block;font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--dim2);margin-bottom:5px}
input[type=text],input[type=number],select{background:var(--surface);border:1px solid var(--line2);
  color:var(--tx);border-radius:7px;padding:7px 10px;font:12.5px var(--mono);width:130px;outline:none;
  transition:border-color .12s}
input:focus{border-color:var(--acc)}
input.hr{width:64px}
select{width:260px}
.toggle{display:inline-flex;align-items:center;gap:7px;font-size:12px;color:var(--mut);
  cursor:pointer;padding:8px 0;user-select:none}
.toggle input{accent-color:var(--acc)}
button{background:var(--acc);border:none;color:#0b0d10;font-weight:650;border-radius:7px;
  padding:8px 18px;cursor:pointer;font-size:12.5px;transition:opacity .12s}
button:hover{opacity:.88}
button.ghost{background:transparent;color:var(--mut);border:1px solid var(--line2)}
button.ghost:hover{color:var(--tx);border-color:var(--dim2);opacity:1}
button:disabled{opacity:.35;cursor:not-allowed}
.note{font-size:11.5px;color:var(--dim2);margin-top:10px}
#c-token{width:230px}

/* ---------- jobs ---------- */
.job{display:flex;align-items:center;gap:14px;padding:8px 2px;border-bottom:1px solid var(--line);
  font:12px var(--mono);cursor:pointer}
.job:hover{background:var(--surface)}
.job .id{color:var(--mut)}
.job .when{color:var(--dim2);margin-left:auto}
.job a{font-size:11px}
.toast{position:fixed;bottom:20px;right:24px;background:var(--card);border:1px solid var(--line2);
  border-left:2px solid var(--acc);border-radius:8px;padding:10px 16px;font-size:12.5px;display:none;z-index:20}
.toast.err{border-left-color:var(--bad)}
</style></head><body>
<header>
  <div class="mark">DSA-110 <span>/ pipeline</span></div>
  <nav style="display:flex;gap:18px;font-size:12.5px"><a href="/" style="color:var(--mut)">telescope</a><a href="/pipeline" style="color:var(--tx);font-weight:600;text-decoration:none">pipeline</a><a href="/science" style="color:var(--mut)">science</a></nav>
  <div class="spacer"></div>
  <div class="meta" id="hd-ctl"><span class="dot"></span>…</div>
  <div class="meta" id="hd-time">—</div>
</header>
<main>
  <div class="sys" id="system"></div>
  <div class="rule"></div>

  <section>
    <p class="microlabel">Pipeline coverage</p>
    <table class="matrix"><thead><tr>
      <th>Date</th><th class="l">Stages</th><th>Raw</th><th>MS ready</th><th>Cal</th>
      <th>Tiles</th><th>Fail</th><th>Mosaics</th><th>Phot</th><th>Verdict</th>
    </tr></thead><tbody id="matrix-body"><tr><td colspan="10" class="z">Loading…</td></tr></tbody></table>
  </section>

  <section id="detail">
    <p class="microlabel" id="detail-title">Detail</p>
    <div class="grid2">
      <div>
        <div class="subsec" style="margin-top:0"><p class="microlabel">Stage walkthrough</p><div id="detail-stages"></div></div>
        <dl class="kv" id="detail-kv"></dl>
        <div class="subsec"><p class="microlabel">Failures · quarantine</p><div id="detail-failures"></div></div>
        <div class="subsec"><p class="microlabel">QA gates</p><div id="detail-gates"></div></div>
        <div class="subsec"><p class="microlabel">Artifacts</p><div class="chips" id="detail-artifacts"></div></div>
      </div>
      <div>
        <p class="microlabel">Epoch mosaics</p>
        <div class="cards" id="detail-mosaics"></div>
        <div class="subsec"><p class="microlabel">Forced photometry</p><div id="detail-phot"></div></div>
      </div>
    </div>
    <div class="subsec"><p class="microlabel">Viewer</p><pre id="viewer">Select an artifact…</pre></div>
  </section>

  <div class="rule"></div>
  <section class="grid2">
    <div>
      <p class="microlabel">Control · validated pipeline entrypoints</p>
      <div class="ctl">
        <div class="f"><label>Flow</label><select id="c-flow">
          <option value="batch">Science batch · MS → products</option>
          <option value="end_to_end">Full flow · raw → science products</option>
        </select></div>
        <div class="f"><label>Date</label><input type="text" id="c-date" placeholder="YYYY-MM-DD"></div>
        <div class="f"><label>Start</label><input type="number" id="c-sh" min="0" max="23" class="hr"></div>
        <div class="f"><label>End</label><input type="number" id="c-eh" min="0" max="23" class="hr"></div>
        <div class="f"><label>RFI</label><select id="c-rfi">
          <option value="conditional">Conditional · production default</option>
          <option value="cflag">cflag · dynamic amplitude</option>
          <option value="full">AOFlagger · full chain</option>
          <option value="off">Off · diagnostic only</option>
        </select></div>
        <label class="toggle"><input type="checkbox" id="c-dry" checked>dry-run</label>
        <label class="toggle"><input type="checkbox" id="c-retry">retry-failed</label>
        <label class="toggle"><input type="checkbox" id="c-lenient">lenient-QA</label>
        <button id="c-go">Launch</button>
        <button class="ghost" id="c-cq">Clear quarantine</button>
      </div>
      <div class="ctl" style="margin-top:14px">
        <div class="f"><label>Token</label><input type="text" id="c-token" placeholder="X-DSA110-Token"></div>
      </div>
      <div class="note" id="c-note"></div>
    </div>
    <div>
      <p class="microlabel">Jobs</p>
      <div id="jobs" class="quiet">No jobs yet.</div>
      <pre id="job-log" style="display:none;margin-top:12px"></pre>
    </div>
  </section>
</main>
<div class="toast" id="toast"></div>
<script>
const $=id=>document.getElementById(id);
let selDate=null, selJob=null, controlEnabled=false;
const j=async(u,opt)=>{const r=await fetch(u,opt);const d=await r.json().catch(()=>({}));
  if(!r.ok)throw new Error(d.detail||r.status);return d};
function toast(msg,ok=true){const t=$('toast');t.textContent=msg;
  t.className='toast'+(ok?'':' err');t.style.display='block';
  clearTimeout(t._h);t._h=setTimeout(()=>t.style.display='none',4200)}
const fmt=(v,alt='—')=>v==null?alt:v;
const dim=v=>v?`${v}`:`<span class="z">0</span>`;
const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function verdictCell(v){if(!v)return '<span class="verdict v-none">—</span>';
  return `<span class="verdict v-${v.toLowerCase()}"><span class="dot ${
    v==='CLEAN'?'ok':(v==='DEGRADED'?'warn':'bad')}"></span>${v}</span>`}

function stageGlyph(r){
  const cls={complete:'good',partial:'warn',blocked:'bad',failed:'bad',waiting:'',unknown:''};
  return `<span class="stages">${r.stages.map(s=>
    `<i class="seg ${cls[s.state]||''}" title="${esc(s.label+': '+s.detail)}"></i>`).join('')}</span>`}

async function loadHealth(){const h=await j('/api/health');controlEnabled=h.control_enabled;
  $('hd-time').textContent=h.time;
  $('hd-ctl').innerHTML=`<span class="dot ${controlEnabled?'ok':'bad'}"></span>control ${
    controlEnabled?'enabled':'disabled'}`;
  $('c-note').textContent=controlEnabled
    ?'Full flow indexes raw HDF5, converts complete groups, then runs strict-QA science processing. Mutating actions require the control token.'
    :'DSA110_DASH_TOKEN is unset on the server — control actions are disabled (fail closed).';
  $('c-go').disabled=$('c-cq').disabled=!controlEnabled}

async function loadSystem(){const s=await j('/api/system');
  $('system').innerHTML=Object.entries(s.disk).map(([k,v])=>v.error
    ?`<div class="d"><span class="lbl">${k}</span><span class="err">unavailable</span></div>`
    :`<div class="d"><span class="lbl">${k}</span>
       <span class="meter"><i class="${v.pct_used>85?'hot':''}" style="width:${v.pct_used}%"></i></span>
       <span>${v.pct_used}%</span><span class="lbl">${v.free} free</span></div>`
  ).join('')+`<div class="d"><span class="lbl">procs</span><span>${s.n_processes}</span></div>`}

async function loadMatrix(){const d=await j('/api/dates');
  $('matrix-body').innerHTML=d.dates.length?d.dates.map(r=>{
    const indexedComplete=r.incoming&&r.indexed&&r.indexed.files>=r.incoming.files;
    const inc=indexedComplete?`<span class="frac"><b>${r.indexed.complete_groups}</b>/${r.indexed.groups}</span>`:
      (r.incoming?`<span class="frac"><b>${r.incoming.files}</b> files</span>`:'<span class="z">—</span>');
    const cal=(r.cal.bandpass&&r.cal.gain)?`${r.cal.bandpass}B ${r.cal.gain}G`
      :((r.cal.bandpass||r.cal.gain)?`<span class="att">${r.cal.bandpass}B ${r.cal.gain}G</span>`:'<span class="neg">none</span>');
    const fail=r.n_failures?`<span class="${r.n_quarantine_risk?'neg':'att'}">${r.n_failures}${
      r.n_quarantine_risk?'·'+r.n_quarantine_risk+'q':''}</span>`:'<span class="z">0</span>';
    const expected=(r.indexed&&r.indexed.complete_groups)||(r.incoming&&r.incoming.complete_groups)||0;
    const ms=expected?`<span class="frac"><b>${r.n_ms}</b>/${expected}</span>`:dim(r.n_ms);
    return `<tr data-d="${r.date}" class="${r.date===selDate?'sel':''}">
      <td>${r.date}</td><td class="l">${stageGlyph(r)}</td><td>${inc}</td>
      <td>${ms}</td><td>${cal}</td><td>${dim(r.n_tiles)}</td><td>${fail}</td>
      <td>${dim(r.n_mosaics)}</td><td>${dim(r.n_phot)}</td><td>${verdictCell(r.verdict)}</td></tr>`
  }).join('')
  :'<tr><td colspan="10" class="z">No dates found under configured roots.</td></tr>';
  document.querySelectorAll('#matrix-body tr[data-d]').forEach(tr=>
    tr.onclick=()=>openDate(tr.dataset.d))}

async function openDate(date){selDate=date;$('c-date').value=date;
  const d=await j('/api/date/'+date);
  $('detail').style.display='block';
  $('detail-title').textContent='Detail · '+date;
  const indexCoversRaw=d.incoming&&d.indexed&&d.indexed.files>=d.incoming.files;
  const inc=d.incoming?`${d.incoming.files} files${indexCoversRaw?` · ${d.indexed.complete_groups}/${d.indexed.groups} complete groups`:''}`:'none visible';
  const man=d.products.manifest||{};
  $('detail-stages').innerHTML=d.stages.map(s=>`<div class="stage-row">
    <b>${esc(s.label)}</b><span class="state ${s.state}">${esc(s.state)}</span>
    <span class="detail">${esc(s.detail)}${s.inferred?' · inferred':''}</span></div>`).join('');
  $('detail-kv').innerHTML=`
    <dt>Incoming HDF5</dt><dd>${inc}</dd>
    <dt>Pipeline index</dt><dd>${d.indexed?`${d.indexed.files} files · ${d.indexed.complete_groups}/${d.indexed.groups} complete groups`:'unavailable'}</dd>
    <dt>Measurement sets</dt><dd>${d.ms.length}${d.ms.length?' · latest '+d.ms[d.ms.length-1]:''}</dd>
    <dt>Bandpass</dt><dd>${d.cal.bandpass.join(', ')||'—'}</dd>
    <dt>Gain</dt><dd>${d.cal.gain.join(', ')||'—'}</dd>
    <dt>Tiles imaged</dt><dd>${d.tiles.n_tiles} science tiles · ${d.tiles.n_artifacts} FITS artifacts</dd>
    <dt>Run verdict</dt><dd>${man.pipeline_verdict||'—'}${man.gaincal_status?' · gaincal '+man.gaincal_status:''}</dd>`;
  const ck=d.tiles.checkpoint;
  $('detail-failures').innerHTML=(ck&&ck.failed&&ck.failed.length)?
    ck.failed.map(f=>`<div class="failrow">
      <span>${(f.ms_path||'').split('/').pop()}</span>
      <span class="quiet">${f.stage||'?'}</span>
      <span class="count ${f.failure_count>=3?'bad':(f.failure_count?'warn':'quiet')}">${f.failure_count?'×'+f.failure_count:'cleared'}</span>
      <span class="err">${(f.error||'').slice(0,110)}</span></div>`).join('')
    :'<span class="quiet">No recorded tile failures.</span>';
  $('detail-gates').innerHTML=(man.gates&&man.gates.length)?
    man.gates.map(g=>`<div class="gate"><span class="dot ${g.verdict==='FAIL'?'bad':'warn'}"></span>
      <b>${g.gate}</b> <span class="why">— ${g.reason}</span></div>`).join('')
    :'<span class="quiet">No gates triggered.</span>';
  const arts=[];
  if(d.products.manifest_path)arts.push(['manifest',d.products.manifest_path]);
  if(d.products.run_report)arts.push(['run_report.md',d.products.run_report]);
  d.products.logs.forEach(l=>arts.push([l.split('/').pop(),l]));
  d.mosaics.forEach(m=>arts.push([m.name.replace('_mosaic.fits','')+' header',m.path]));
  $('detail-artifacts').innerHTML=arts.length?arts.map(([n,p])=>
    `<span class="chip" onclick="viewArtifact('${p}')">${n}</span>`).join(''):'<span class="quiet">—</span>';
  $('detail-mosaics').innerHTML=d.mosaics.length?d.mosaics.map(m=>{
    const ep=(man.epochs||[]).find(e=>e.mosaic_path&&e.mosaic_path.endsWith(m.name));
    const qa=ep?`<span class="qa ${ep.qa_result==='PASS'?'v-clean':'v-failed'}">${ep.qa_result||'?'}</span>`
      :'<span class="qa v-none">NO QA</span>';
    const hour=(m.name.match(/T(\d{2})00_mosaic/)||[])[1];
    return `<div class="mcard"><div class="hd"><span class="name">${hour!=null?hour+':00 UTC':m.name}</span>${qa}</div>
      <img loading="lazy" src="/api/thumb/${date}/${m.name}.png">
      <div class="ft"><span>peak <b>${fmt(m.peak)}</b> Jy</span><span>rms <b>${fmt(m.rms_mjy)}</b> mJy</span>
      <span>DR <b>${fmt(m.dr)}</b></span><span>wt ${m.weights?'✓':'<span style="color:var(--bad)">✗</span>'}</span></div></div>`
  }).join(''):'<span class="quiet">No epoch mosaics yet.</span>';
  $('detail-phot').innerHTML=d.products.phot_csvs.length?
    '<table class="phot"><tr><th>Epoch CSV</th><th>Sources</th><th>Bright</th><th>Median ratio</th></tr>'+
    d.products.phot_csvs.map(c=>{
      const r=c.median_ratio, ok=(r>=0.8&&r<=1.2);
      return `<tr><td><a onclick="viewArtifact('${c.path}')">${c.name}</a></td>
      <td>${fmt(c.n_sources)}</td><td>${fmt(c.n_bright)}</td>
      <td class="ratio ${r!=null?(ok?'ok':'bad'):''}">${fmt(r)}</td></tr>`}).join('')+'</table>'
    :'<span class="quiet">No forced-photometry products.</span>';
  loadMatrix();
  $('detail').scrollIntoView({behavior:'smooth'})}

async function viewArtifact(path){const v=$('viewer');v.textContent='Loading '+path+' …';
  try{const d=await j('/api/artifact?path='+encodeURIComponent(path));
    if(d.fits_header)v.textContent=`# ${d.path}\npeak ${d.peak} Jy · rms ${d.rms_mjy} mJy · DR ${d.dr}\n\n`+d.fits_header;
    else if(d.json)v.textContent=JSON.stringify(d.json,null,2);
    else v.textContent=(d.head||d.tail||[]).join('\n')||'(empty)';
  }catch(e){v.textContent='Error: '+e.message}}

function token(){return $('c-token').value.trim()}
$('c-flow').onchange=()=>{
  const full=$('c-flow').value==='end_to_end';
  $('c-sh').disabled=$('c-eh').disabled=$('c-dry').disabled=
    $('c-retry').disabled=$('c-lenient').disabled=full;
  if(full){$('c-sh').value=$('c-eh').value='';$('c-dry').checked=false;
    $('c-retry').checked=true;$('c-lenient').checked=false}
  else{$('c-dry').checked=true;$('c-retry').checked=false}
};
$('c-go').onclick=async()=>{
  const body={date:$('c-date').value.trim(),flow:$('c-flow').value,dry_run:$('c-dry').checked,
    rfi_mode:$('c-rfi').value,retry_failed:$('c-retry').checked,
    lenient_qa:$('c-lenient').checked};
  if($('c-sh').value!=='')body.start_hour=+$('c-sh').value;
  if($('c-eh').value!=='')body.end_hour=+$('c-eh').value;
  const label=body.flow==='end_to_end'?'FULL raw-to-products flow':'REAL science batch';
  if(!body.dry_run&&!confirm('Launch '+label+' for '+body.date+' with strict QA'+
    (body.lenient_qa?' overridden':'')+'?'))return;
  try{const d=await j('/api/control/run',{method:'POST',
    headers:{'Content-Type':'application/json','X-DSA110-Token':token()},
    body:JSON.stringify(body)});
    toast(`Job ${d.id} started (${d.kind})`);loadJobs();selJob=d.id;
  }catch(e){toast('Launch failed: '+e.message,false)}};
$('c-cq').onclick=async()=>{
  const date=$('c-date').value.trim();
  if(!confirm('Clear quarantine failure counts for '+date+'?'))return;
  try{const d=await j('/api/control/clear-quarantine',{method:'POST',
    headers:{'Content-Type':'application/json','X-DSA110-Token':token()},
    body:JSON.stringify({date})});
    toast(`Cleared ${d.cleared} failure count(s)`);if(selDate)openDate(selDate);
  }catch(e){toast('Clear failed: '+e.message,false)}};

async function loadJobs(){const d=await j('/api/jobs');
  $('jobs').className=d.jobs.length?'':'quiet';
  $('jobs').innerHTML=d.jobs.length?d.jobs.map(x=>{
    const cls=x.status==='running'?'run':(x.status==='completed'?'ok':'bad');
    return `<div class="job" data-id="${x.id}"><span class="dot ${cls}"></span>
      <span>${x.status}</span><span class="id">${x.id}</span><span>${x.kind}</span>
      <span>${x.date}</span><span class="when">${x.started_at}</span>
      ${x.status==='running'?`<a data-kill="${x.id}">kill</a>`:''}</div>`}).join('')
    :'No jobs yet.';
  document.querySelectorAll('.job').forEach(el=>el.onclick=async ev=>{
    if(ev.target.dataset.kill){
      try{await j('/api/jobs/'+ev.target.dataset.kill+'/kill',{method:'POST',
        headers:{'X-DSA110-Token':token()}});toast('Kill sent');loadJobs()}
      catch(e){toast('Kill failed: '+e.message,false)}
      return}
    selJob=el.dataset.id;showJobLog()});
  if(selJob)showJobLog()}
async function showJobLog(){if(!selJob)return;
  try{const d=await j('/api/jobs/'+selJob);
    const p=$('job-log');p.style.display='block';
    p.textContent=`# job ${d.id} · ${d.status} · ${d.argv.join(' ')}\n\n`+(d.log_tail||[]).join('\n')}
  catch(e){}}

loadHealth();loadSystem();loadMatrix();loadJobs();
setInterval(()=>{loadSystem();loadMatrix();loadJobs();loadHealth()},30000);
setInterval(showJobLog,5000);
</script></body></html>
"""


HOME_PAGE = r"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DSA-110 Pipeline Console</title>
<style>
:root{
  --bg:#0b0d10;--surface:#0f1216;--card:#12161c;--line:#1d222b;--line2:#262d38;
  --tx:#e7ebf3;--mut:#8b95a7;--dim2:#5a6474;
  --acc:#5aa2ff;--ok:#3ecf8e;--warn:#d9a13f;--bad:#e0564f;--sun:#f2c14e;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
html{-webkit-font-smoothing:antialiased}
body{margin:0;background:var(--bg);color:var(--tx);
  font:13.5px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
a{color:var(--acc);text-decoration:none;cursor:pointer}
a:hover{text-decoration:underline;text-underline-offset:3px}
header{display:flex;align-items:baseline;gap:22px;padding:20px 28px 0;max-width:1560px;margin:0 auto}
header .mark{font-size:15px;font-weight:650;letter-spacing:.2px}
header .mark span{color:var(--mut);font-weight:400}
nav{display:flex;gap:18px;font-size:12.5px}
nav a{color:var(--mut)}
nav a.active{color:var(--tx);font-weight:600}
nav a:hover{color:var(--tx);text-decoration:none}
header .spacer{flex:1}
header .meta{font:11.5px var(--mono);color:var(--dim2)}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:6px;
  vertical-align:1px;background:var(--dim2)}
.dot.ok{background:var(--ok)}.dot.bad{background:var(--bad)}.dot.warn{background:var(--warn)}
main{max-width:1560px;margin:0 auto;padding:10px 28px 48px}
section{margin-top:26px}
.microlabel{font-size:10.5px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;
  color:var(--dim2);margin:0 0 10px}
.rule{border-top:1px solid var(--line);margin:26px 0 0}
.quiet{color:var(--dim2);font-size:12.5px}

/* status tiles */
.tiles{display:flex;gap:40px;flex-wrap:wrap;padding-top:14px}
.tile .v{font:20px var(--mono);font-variant-numeric:tabular-nums}
.tile .v small{font-size:12px;color:var(--mut)}
.tile .k{font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--dim2);margin-top:2px}
.tile .age{font:10.5px var(--mono);color:var(--dim2)}

/* sky map */
.skywrap{background:linear-gradient(180deg,#10151d,#0c1016);border:1px solid var(--line);
  border-radius:12px;padding:14px 16px 8px;box-shadow:inset 0 1px 0 rgba(255,255,255,.025)}
.skyhead{display:flex;gap:8px 18px;align-items:baseline;flex-wrap:wrap;font:11px var(--mono);color:var(--mut);
  padding:0 2px 8px}
.skyhead .spacer{flex:1}
.legend{display:flex;gap:6px 14px;align-items:center;flex-wrap:wrap}
.legend i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;vertical-align:-1px}
.skycanvas{position:relative;overflow:hidden;border-radius:8px;background:#07101a}
#skymap svg{width:100%;display:block}
.sky-source{cursor:crosshair;outline:none}
.sky-source .source-halo{fill:var(--source-color);filter:url(#sourceGlow);opacity:.38;
  pointer-events:none}
.sky-source .source-core{fill:var(--source-color);stroke:rgba(255,255,255,.14);stroke-width:.7;
  transition:stroke .12s,stroke-width .12s,opacity .12s}
.sky-source .source-hit{fill:transparent;pointer-events:all}
.sky-source:hover .source-core,.sky-source:focus .source-core{stroke:#fff;stroke-width:1.8;opacity:1}
.sky-source:focus-visible .source-hit{stroke:var(--acc);stroke-width:1;stroke-dasharray:2 2}
.sky-tip{position:absolute;z-index:4;min-width:178px;max-width:260px;padding:9px 11px;
  border:1px solid #425269;border-radius:7px;background:rgba(8,12,18,.96);color:var(--tx);
  box-shadow:0 12px 34px rgba(0,0,0,.42);font:11px/1.45 var(--mono);pointer-events:none;
  opacity:0;transform:translateY(3px);transition:opacity .1s,transform .1s}
.sky-tip.show{opacity:1;transform:translateY(0)}
.sky-tip strong{display:block;font-size:12px;color:#f4f7fb;margin-bottom:2px}
.sky-tip .source-kind{text-transform:uppercase;letter-spacing:.11em;font-size:9px;color:var(--acc)}
.sky-tip .source-coords{display:block;color:var(--dim2);margin-top:3px}
svg text{font:9.5px var(--mono);fill:var(--dim2)}
svg .lbl{fill:#aab6c9;font-size:10px}
svg .lbl.big{fill:#dfe6f2;font-size:10.5px}
@media(prefers-reduced-motion:reduce){.sky-source .source-core,.sky-tip{transition:none}}

/* antennas */
.antwrap{display:flex;gap:36px;align-items:flex-start;flex-wrap:wrap}
.antgrid{display:grid;grid-template-columns:repeat(auto-fill,24px);gap:4px}
#antscatter svg{width:100%;display:block;background:var(--surface);border:1px solid var(--line);border-radius:10px}
.ant{width:24px;height:24px;border-radius:4px;background:var(--line2);position:relative;
  display:flex;align-items:center;justify-content:center;
  font:8.5px var(--mono);color:transparent;transition:transform .08s}
.ant:hover{transform:scale(1.25);color:#0b0d10;z-index:2}
.ant.good{background:var(--ok);opacity:.85}
.ant.warn{background:var(--warn);opacity:.9}
.ant.bad{background:var(--bad);opacity:.9}
.ant.unknown{background:#151a21;border:1px solid var(--line2)}
.antside{min-width:200px;font:12px var(--mono);color:var(--mut)}
.antside .row{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid var(--line)}
.antside .n{color:var(--tx)}

/* overhead list */
.chips{display:flex;flex-wrap:wrap;gap:8px}
.chip{font:11.5px var(--mono);color:var(--tx);background:var(--card);border:1px solid var(--line2);
  border-radius:6px;padding:4px 10px}
.chip b{font-weight:600}
.chip .in{color:var(--dim2)}
.grid2{display:grid;grid-template-columns:2fr 1fr;gap:36px}
@media(max-width:1100px){.grid2{grid-template-columns:1fr}.antgrid{min-width:0}}
</style></head><body>
<header>
  <div class="mark">DSA-110 <span>/ telescope</span></div>
  <nav><a class="active" href="/">telescope</a><a href="/pipeline">pipeline</a><a href="/science">science</a></nav>
  <div class="spacer"></div>
  <div class="meta" id="hd-ctl"><span class="dot"></span>…</div>
  <div class="meta" id="hd-utc">—</div>
</header>
<main>
  <div class="tiles">
    <div class="tile"><div class="v" id="t-lst">—</div><div class="k">LST · OVRO</div></div>
    <div class="tile"><div class="v" id="t-mer">—</div><div class="k">Meridian RA</div></div>
    <div class="tile"><div class="v" id="t-dec">—</div><div class="k">Dec strip</div></div>
    <div class="tile"><div class="v" id="t-fresh">—</div><div class="k">Latest data</div>
      <div class="age" id="t-fresh-age"></div></div>
    <div class="tile"><div class="v" id="t-ant">—</div><div class="k">Antennas good</div>
      <div class="age" id="t-ant-age"></div></div>
  </div>

  <section>
    <p class="microlabel">Simulated 1.4 GHz sky over OVRO — live meridian + catalog overlay</p>
    <div class="skywrap">
      <div class="skyhead">
        <span id="sky-note">—</span><span class="spacer"></span>
        <span class="legend">
          <span><i style="background:var(--sun)"></i>Sun</span>
          <span><i style="background:var(--acc)"></i>calibrator</span>
          <span><i style="background:#aab6c9"></i>source</span>
          <span><i style="background:#b98760;box-shadow:0 0 6px #b98760"></i>Galactic plane</span>
          <span><i style="background:none;border:1px solid var(--acc);border-radius:2px;width:10px;height:6px"></i>Dec strip</span>
        </span>
      </div>
      <div class="skycanvas">
        <div id="skymap"></div>
        <div class="sky-tip" id="sky-tip" role="tooltip" aria-hidden="true"></div>
      </div>
    </div>
  </section>

  <section class="grid2">
    <div>
      <p class="microlabel">Transiting now · next two hours</p>
      <div class="chips" id="overhead">—</div>
    </div>
    <div>
      <p class="microlabel">Observing</p>
      <div class="quiet" id="obs-note">—</div>
    </div>
  </section>

  <div class="rule"></div>
  <section>
    <p class="microlabel" id="ant-label">Antennas</p>
    <div class="antwrap">
      <div style="flex:1;min-width:520px"><div id="antscatter" style="display:none"></div><div class="antgrid" id="antgrid"></div></div>
      <div class="antside" id="antside"></div>
    </div>
  </section>
</main>
<script>
const $=id=>document.getElementById(id);
const j=async u=>{const r=await fetch(u);if(!r.ok)throw new Error(r.status);return r.json()};
let SKY=null;

/* client-side sidereal time so the meridian moves without polling */
function lstHours(){
  const d=(Date.now()/86400000)+2440587.5-2451545.0;
  const gmst=(18.697374558+24.06570982441908*d)%24;
  return ((gmst+(SKY?SKY.site.lon_deg:-118.2817)/15)%24+24)%24}
const hfmt=h=>{const H=Math.floor(h),M=Math.floor((h-H)*60),S=Math.floor(((h-H)*60-M)*60);
  return `${String(H).padStart(2,'0')}:${String(M).padStart(2,'0')}:${String(S).padStart(2,'0')}`}

const W=1240,H=330,DECMIN=-45,DECMAX=90;
const X=ra=>(360-ra)/360*W;
const Y=dec=>(DECMAX-dec)/(DECMAX-DECMIN)*H;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fluxLabel=f=>f==null?'flux unavailable':(f>=1000?`${(f/1000).toFixed(2)} kJy`:
  f>=10?`${f.toFixed(1)} Jy`:`${f.toFixed(2)} Jy`);

function galToEq(l,b=0){
  const lr=l*Math.PI/180,br=b*Math.PI/180,cb=Math.cos(br);
  const xg=cb*Math.cos(lr),yg=cb*Math.sin(lr),zg=Math.sin(br);
  const xe=-.0548755604*xg+.4941094279*yg-.8676661490*zg;
  const ye=-.8734370902*xg-.4448296300*yg-.1980763734*zg;
  const ze=-.4838350155*xg+.7469822445*yg+.4559837762*zg;
  return {ra:(Math.atan2(ye,xe)*180/Math.PI+360)%360,
    dec:Math.asin(Math.max(-1,Math.min(1,ze)))*180/Math.PI}}

function galacticPlanePaths(b=0){
  const paths=[];let d='',prev=null;
  for(let l=0;l<=360;l+=2){
    const q=galToEq(l,b),x=X(q.ra),y=Y(q.dec);
    if(prev!=null&&Math.abs(x-prev)>W/2){if(d)paths.push(d);d=`M${x.toFixed(1)},${y.toFixed(1)}`}
    else d+=`${d?' L':'M'}${x.toFixed(1)},${y.toFixed(1)}`;
    prev=x}
  if(d)paths.push(d);return paths}

function seededRandom(seed){return function(){seed|=0;seed=seed+0x6D2B79F5|0;
  let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;
  return ((t^t>>>14)>>>0)/4294967296}}

let SIM_SKY='';
function simulatedSky(){
  const rand=seededRandom(110),dots=[];
  for(let i=0;i<280;i++){
    let q,plane=i<165;
    if(plane){const b=(rand()+rand()+rand()-1.5)*9;q=galToEq(rand()*360,b)}
    else{const lo=Math.sin(DECMIN*Math.PI/180),hi=Math.sin(DECMAX*Math.PI/180);
      q={ra:rand()*360,dec:Math.asin(lo+rand()*(hi-lo))*180/Math.PI}}
    if(q.dec<DECMIN||q.dec>DECMAX)continue;
    const bright=Math.pow(rand(),5),r=.22+bright*(plane?1.05:.65),op=.12+bright*.38;
    dots.push(`<circle cx="${X(q.ra).toFixed(1)}" cy="${Y(q.dec).toFixed(1)}" r="${r.toFixed(2)}"
      fill="${plane?'#d7b08a':'#8fb2cf'}" opacity="${op.toFixed(2)}"/>`)}
  return `<g class="simulated-sources" pointer-events="none">${dots.join('')}</g>`}

const pathMarkup=(paths,attrs)=>paths.map(d=>`<path d="${d}" ${attrs}/>`).join('');
function diffuseSky(){
  const core=galacticPlanePaths(0),north=galacticPlanePaths(9),south=galacticPlanePaths(-9);
  return pathMarkup(north.concat(south),'fill="none" stroke="#557c99" stroke-width="34" opacity=".08" filter="url(#planeBlur)"')+
    pathMarkup(core,'fill="none" stroke="#c38d61" stroke-width="46" opacity=".14" filter="url(#planeBlur)"')+
    pathMarkup(core,'fill="none" stroke="#e0b488" stroke-width="13" opacity=".18" filter="url(#planeSoft)"')+
    pathMarkup(core,'fill="none" stroke="#e8c39e" stroke-width="1.2" opacity=".28"')}

function sourceMarkup(src){
  if(src.dec_deg<DECMIN||src.dec_deg>DECMAX)return '';
  const col={sun:'var(--sun)',ateam:'var(--bad)',cal:'var(--acc)',src:'#b8c6d8',catalog:'#74859b'};
  const x=X(src.ra_deg),y=Y(src.dec_deg),f=src.flux_jy;
  const r=src.kind==='sun'?7:Math.max(1.6,Math.min(6.5,1.2+Math.log10(Math.max(f||1,1))*1.9));
  const name=src.name||'Catalog source',kind=src.kind||'source';
  const aria=`${name}, ${fluxLabel(f)}, right ascension ${src.ra_deg.toFixed(2)} degrees, declination ${src.dec_deg.toFixed(2)} degrees`;
  const halo=kind==='catalog'?'':`<circle class="source-halo" cx="${x}" cy="${y}" r="${r*2.1}"/>`;
  return `<g class="sky-source kind-${esc(kind)}" tabindex="0" role="img" aria-label="${esc(aria)}"
    data-name="${esc(name)}" data-kind="${esc(kind)}" data-flux="${esc(fluxLabel(f))}"
    data-ra="${src.ra_deg.toFixed(2)}" data-dec="${src.dec_deg.toFixed(2)}"
    style="--source-color:${col[kind]||col.src}">${halo}
    <circle class="source-core" cx="${x}" cy="${y}" r="${r}" opacity="${kind==='catalog'?.62:.94}"/>
    <circle class="source-hit" cx="${x}" cy="${y}" r="${Math.max(8,r+4)}"/></g>`}

function updateSkyReadout(){if(!SKY)return;const lst=lstHours(),mer=lst*15;
  $('t-lst').textContent=hfmt(lst);$('t-mer').innerHTML=`${mer.toFixed(1)}<small>°</small>`}

function drawSky(){
  if(!SKY)return;
  const lst=lstHours(), mer=lst*15, strip=SKY.dec_strip_deg;
  if(!SIM_SKY)SIM_SKY=simulatedSky();
  let s=`<svg viewBox="0 0 ${W} ${H}" aria-label="Simulated 1.4 gigahertz sky in equatorial coordinates">
    <defs>
      <linearGradient id="skyBg" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#0b1826"/>
        <stop offset=".56" stop-color="#08121d"/><stop offset="1" stop-color="#050a11"/></linearGradient>
      <radialGradient id="gcGlow"><stop stop-color="#d69b68" stop-opacity=".26"/><stop offset="1" stop-color="#d69b68" stop-opacity="0"/></radialGradient>
      <filter id="planeBlur" x="-25%" y="-60%" width="150%" height="220%"><feGaussianBlur stdDeviation="12"/></filter>
      <filter id="planeSoft" x="-15%" y="-40%" width="130%" height="180%"><feGaussianBlur stdDeviation="4"/></filter>
      <filter id="sourceGlow" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="3"/></filter>
    </defs>
    <rect width="${W}" height="${H}" fill="url(#skyBg)"/>
    <ellipse cx="${X(266.4)}" cy="${Y(-29)}" rx="88" ry="54" fill="url(#gcGlow)" filter="url(#planeSoft)"/>
    ${diffuseSky()}${SIM_SKY}`;
  // graticule
  for(let ra=0;ra<=360;ra+=30){
    s+=`<line x1="${X(ra)}" y1="0" x2="${X(ra)}" y2="${H}" stroke="#26364a" stroke-opacity=".38" stroke-width="1"/>`;
    if(ra<360)s+=`<text x="${X(ra)-3}" y="${H-5}" text-anchor="end">${ra/15}h</text>`}
  for(let dec=-30;dec<=60;dec+=30){
    s+=`<line x1="0" y1="${Y(dec)}" x2="${W}" y2="${Y(dec)}" stroke="#26364a" stroke-opacity=".38"/>`;
    s+=`<text x="5" y="${Y(dec)-4}">${dec>0?'+':''}${dec}°</text>`}
  // horizon limit (dec < lat-90 never rises)
  const horizon=SKY.site.lat_deg-90;
  if(horizon>DECMIN)s+=`<rect x="0" y="${Y(horizon)}" width="${W}" height="${H-Y(horizon)}"
     fill="#0a0c0f" opacity=".75"/><text x="${W-8}" y="${Y(horizon)+14}" text-anchor="end">never rises</text>`;
  // dec strip band
  s+=`<rect x="0" y="${Y(strip+1.6)}" width="${W}" height="${Y(strip-1.6)-Y(strip+1.6)}"
     fill="var(--acc)" opacity=".10"/>
     <line x1="0" y1="${Y(strip+1.6)}" x2="${W}" y2="${Y(strip+1.6)}" stroke="var(--acc)" opacity=".45" stroke-dasharray="3 4"/>
     <line x1="0" y1="${Y(strip-1.6)}" x2="${W}" y2="${Y(strip-1.6)}" stroke="var(--acc)" opacity=".45" stroke-dasharray="3 4"/>`;
  // overhead window (meridian ± 7.5°) intersecting the strip
  const wx=X(mer+7.5), ww=X(mer-7.5)-X(mer+7.5);
  s+=`<rect x="${wx}" y="${Y(strip+1.6)}" width="${ww}" height="${Y(strip-1.6)-Y(strip+1.6)}"
     fill="var(--acc)" opacity=".28" rx="2"/>`;
  // meridian
  s+=`<line x1="${X(mer)}" y1="0" x2="${X(mer)}" y2="${H}" stroke="#e7ebf3" opacity=".55" stroke-width="1.2"/>
     <text x="${X(mer)+5}" y="12" class="lbl">meridian ${hfmt(lst)} LST</text>`;
  s+=SKY.sources.map(sourceMarkup).join('')+'</svg>';
  $('skymap').innerHTML=s;
  updateSkyReadout();
  // overhead / next-hour chips
  const soon=[],now=[];
  for(const src of SKY.sources){
    let dra=(src.ra_deg-mer+540)%360-180; // + = transits later
    const hrs=dra/15*0.9972695663;
    if(Math.abs(src.dec_deg-strip)<=2.5||src.kind==='sun'||(src.flux_jy||0)>=100){
      if(Math.abs(hrs)<=0.25)now.push(src);
      else if(hrs>0&&hrs<=2.0)soon.push([hrs,src])}}
  soon.sort((a,b)=>a[0]-b[0]);
  $('overhead').innerHTML=(now.map(s2=>`<span class="chip"><b>${esc(s2.name||'src')}</b> <span class="in">on meridian</span></span>`)
    .concat(soon.slice(0,10).map(([hq,s2])=>`<span class="chip">${esc(s2.name||'src')} <span class="in">in ${Math.round(hq*60)}m</span></span>`)))
    .join('')||'<span class="quiet">Nothing notable in the strip this hour.</span>'}

const skyTip=$('sky-tip'),skyCanvas=document.querySelector('.skycanvas');
const kindLabel={sun:'solar system',ateam:'very bright radio source',cal:'calibrator',src:'bright source',catalog:'catalog source'};
const skySource=e=>e.target.closest?e.target.closest('.sky-source'):null;
function placeSkyTip(source,event){
  const canvas=skyCanvas.getBoundingClientRect(),box=source.getBoundingClientRect();
  const x=event?.clientX!=null?event.clientX-canvas.left:box.left+box.width/2-canvas.left;
  const y=event?.clientY!=null?event.clientY-canvas.top:box.top-canvas.top;
  const left=Math.max(8,Math.min(canvas.width-skyTip.offsetWidth-8,x+13));
  const top=Math.max(8,Math.min(canvas.height-skyTip.offsetHeight-8,y-skyTip.offsetHeight-12));
  skyTip.style.left=`${left}px`;skyTip.style.top=`${top}px`}
function showSkyTip(source,event){
  skyTip.innerHTML=`<strong>${esc(source.dataset.name)}</strong>
    <span class="source-kind">${esc(kindLabel[source.dataset.kind]||source.dataset.kind)}</span>
    <span> · ${esc(source.dataset.flux)}</span>
    <span class="source-coords">RA ${esc(source.dataset.ra)}° · Dec ${esc(source.dataset.dec)}°</span>`;
  skyTip.classList.add('show');skyTip.setAttribute('aria-hidden','false');placeSkyTip(source,event)}
function hideSkyTip(){skyTip.classList.remove('show');skyTip.setAttribute('aria-hidden','true')}
$('skymap').addEventListener('pointerover',e=>{const source=skySource(e);if(source)showSkyTip(source,e)});
$('skymap').addEventListener('pointermove',e=>{const source=skySource(e);if(source)placeSkyTip(source,e)});
$('skymap').addEventListener('pointerout',e=>{const source=skySource(e);
  if(source&&!source.contains(e.relatedTarget))hideSkyTip()});
$('skymap').addEventListener('focusin',e=>{const source=skySource(e);if(source)showSkyTip(source)});
$('skymap').addEventListener('focusout',hideSkyTip);
skyCanvas.addEventListener('pointerleave',hideSkyTip);

async function loadSky(){SKY=await j('/api/sky');
  $('t-dec').innerHTML=`${SKY.dec_strip_deg}<small>°</small>`;
  $('sky-note').textContent=`${SKY.n_sources} sources · strip Dec ${SKY.dec_strip_deg}° ± 1.6° · window ±30 min`;
  const ts=SKY.latest_data_ts;
  if(ts){$('t-fresh').textContent=ts.slice(5,16).replace('T',' ');
    const age=(Date.now()-Date.parse(ts+'Z'))/3.6e6;
    $('t-fresh-age').textContent=age<48?`${age.toFixed(1)} h ago`:`${(age/24).toFixed(1)} d ago`;
    $('t-fresh-age').style.color=age>24?'var(--bad)':(age>3?'var(--warn)':'var(--dim2)');
    $('obs-note').innerHTML=`Latest HDF5 group <b style="color:var(--tx)">${ts}</b>. The highlighted window is the
      tile currently transiting; hourly-epoch mosaics assemble ~12 sequential tiles along the Dec ${SKY.dec_strip_deg}° strip.
      See <a href="/pipeline">pipeline</a> for stage state.`}
  else{$('t-fresh').textContent='none';$('obs-note').textContent='No incoming HDF5 visible under the configured root.'}
  drawSky()}

async function loadAnts(){const a=await j('/api/antennas');
  const good=a.counts.good||0;
  $('t-ant').innerHTML=`${good}<small>/${a.n}</small>`;
  $('t-ant-age').textContent=a.asof?('as of '+a.asof.slice(0,16).replace('T',' ')):'no status source';
  $('ant-label').textContent=`Antennas · source: ${a.source}${a.asof?' · '+a.asof.slice(0,16).replace('T',' '):''}`;
  const withPos=a.antennas.filter(x=>x.x_m!=null&&x.y_m!=null);
  if(a.has_positions&&withPos.length>=3){
    $('antgrid').style.display='none';$('antscatter').style.display='block';
    const xs=withPos.map(x=>x.x_m),ys=withPos.map(x=>x.y_m);
    const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
    const W2=1180,H2=380,pad=34;
    const sc=Math.min((W2-2*pad)/Math.max(x1-x0,1),(H2-2*pad)/Math.max(y1-y0,1));
    const px=v=>pad+(v-x0)*sc+((W2-2*pad)-(x1-x0)*sc)/2;
    const py=v=>H2-pad-(v-y0)*sc-((H2-2*pad)-(y1-y0)*sc)/2;
    const cmap={good:'var(--ok)',warn:'var(--warn)',bad:'var(--bad)',unknown:'#39414d'};
    $('antscatter').innerHTML=`<svg viewBox="0 0 ${W2} ${H2}">`+
      `<text x="${W2-12}" y="${H2-10}" text-anchor="end">east →</text>`+
      `<text x="14" y="18">north ↑</text>`+
      withPos.map(x=>`<circle cx="${px(x.x_m)}" cy="${py(x.y_m)}" r="5" fill="${cmap[x.status]||cmap.unknown}"
        opacity=".92"><title>ant ${x.name} · ${x.status}${
        x.flag_frac!=null?' · flagged '+(x.flag_frac*100).toFixed(0)+'%':''} · E ${x.x_m.toFixed(0)} m, N ${x.y_m.toFixed(0)} m</title></circle>`).join('')+
      '</svg>';
  }else{
    $('antscatter').style.display='none';$('antgrid').style.display='grid';
    $('antgrid').innerHTML=a.antennas.map(x=>
      `<div class="ant ${x.status}" title="ant ${x.name} · ${x.status}${
       x.flag_frac!=null?' · flagged '+(x.flag_frac*100).toFixed(0)+'%':''}">${x.name}</div>`).join('');
  }
  const rows=[['good','var(--ok)'],['warn','var(--warn)'],['bad','var(--bad)'],['unknown','var(--line2)']];
  $('antside').innerHTML=rows.map(([k,c])=>
    `<div class="row"><span><span class="dot" style="background:${c}"></span>${k}</span>
     <span class="n">${a.counts[k]||0}</span></div>`).join('')+
    `<div class="row"><span>total</span><span class="n">${a.n}</span></div>`}

async function loadHealth(){try{const h=await j('/api/health');
  $('hd-ctl').innerHTML=`<span class="dot ${h.control_enabled?'ok':'bad'}"></span>control ${h.control_enabled?'enabled':'disabled'}`}catch(e){}}

function tick(){$('hd-utc').textContent=new Date().toISOString().slice(0,19)+'Z';updateSkyReadout()}
loadSky();loadAnts();loadHealth();
setInterval(tick,1000);
setInterval(drawSky,60000);
setInterval(loadSky,300000);setInterval(loadAnts,120000);
</script></body></html>"""

SCIENCE_PAGE = r"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DSA-110 Pipeline Console</title>
<style>
:root{
  --bg:#0b0d10;--surface:#0f1216;--card:#12161c;--line:#1d222b;--line2:#262d38;
  --tx:#e7ebf3;--mut:#8b95a7;--dim2:#5a6474;
  --acc:#5aa2ff;--ok:#3ecf8e;--warn:#d9a13f;--bad:#e0564f;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
html{-webkit-font-smoothing:antialiased}
body{margin:0;background:var(--bg);color:var(--tx);
  font:13.5px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
a{color:var(--acc);text-decoration:none;cursor:pointer}
a:hover{text-decoration:underline;text-underline-offset:3px}
header{display:flex;align-items:baseline;gap:22px;padding:20px 28px 0;max-width:1560px;margin:0 auto}
header .mark{font-size:15px;font-weight:650;letter-spacing:.2px}
header .mark span{color:var(--mut);font-weight:400}
nav{display:flex;gap:18px;font-size:12.5px}
nav a{color:var(--mut)}
nav a.active{color:var(--tx);font-weight:600}
nav a:hover{color:var(--tx);text-decoration:none}
header .spacer{flex:1}
header .meta{font:11.5px var(--mono);color:var(--dim2)}
main{max-width:1560px;margin:0 auto;padding:10px 28px 48px}
section{margin-top:26px}
.microlabel{font-size:10.5px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;
  color:var(--dim2);margin:0 0 10px}
.rule{border-top:1px solid var(--line);margin:26px 0 0}
.quiet{color:var(--dim2);font-size:12.5px}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:6px;
  vertical-align:1px;background:var(--dim2)}
.dot.ok{background:var(--ok)}.dot.bad{background:var(--bad)}.dot.warn{background:var(--warn)}
.daterow{display:flex;align-items:baseline;gap:14px;margin:26px 0 12px}
.daterow .d{font:14px var(--mono);font-weight:600}
.daterow .v{font:11px var(--mono);letter-spacing:.06em}
.v-clean{color:var(--ok)}.v-degraded{color:var(--warn)}.v-failed{color:var(--bad)}.v-none{color:var(--dim2)}
.daterow .meta{font:11.5px var(--mono);color:var(--dim2)}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
.mcard{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.mcard .hd{display:flex;align-items:center;gap:8px;padding:8px 12px;font:11.5px var(--mono)}
.mcard .hd .qa{margin-left:auto;font-size:10.5px;letter-spacing:.08em}
.mcard img{width:100%;display:block;min-height:56px;background:#000;cursor:zoom-in}
.mcard .ft{display:flex;gap:14px;padding:7px 12px;font:11px var(--mono);color:var(--mut)}
.mcard .ft b{color:var(--tx);font-weight:500}
table{border-collapse:collapse}
.phot td,.phot th{padding:6px 14px 6px 0;border-bottom:1px solid var(--line);text-align:right;
  font:12px var(--mono);font-variant-numeric:tabular-nums}
.phot th{font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--dim2);font-family:inherit}
.phot td:first-child,.phot th:first-child{text-align:left}
.ratio.ok{color:var(--ok)}.ratio.bad{color:var(--bad)}
.lightbox{position:fixed;inset:0;background:rgba(5,6,8,.88);display:none;
  align-items:center;justify-content:center;z-index:30;cursor:zoom-out}
.lightbox img{max-width:94vw;max-height:92vh;border-radius:8px}
.grid2v{display:grid;grid-template-columns:minmax(0,7fr) minmax(0,5fr);gap:32px}
@media(max-width:1100px){.grid2v{grid-template-columns:1fr}}
.var td,.var th{padding:6px 12px 6px 0;border-bottom:1px solid var(--line);text-align:right;
  font:12px var(--mono);font-variant-numeric:tabular-nums}
.var th{font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--dim2);font-family:inherit}
.var td:first-child,.var th:first-child{text-align:left}
.var tbody tr{cursor:pointer}.var tbody tr:hover{background:var(--surface)}
.var tbody tr.sel{background:var(--surface);box-shadow:inset 2px 0 0 var(--acc)}
#lcpanel svg{width:100%;background:var(--surface);border:1px solid var(--line);border-radius:10px}
#lcpanel .t{font:12px var(--mono);color:var(--mut);margin-bottom:8px}
</style></head><body>
<header>
  <div class="mark">DSA-110 <span>/ science products</span></div>
  <nav><a href="/">telescope</a><a href="/pipeline">pipeline</a><a class="active" href="/science">science</a></nav>
  <div class="spacer"></div>
  <div class="meta" id="hd-utc">—</div>
</header>
<main>
  <p class="quiet" style="margin-top:14px">Hourly-epoch mosaics with QA, forced photometry per epoch, and run
  reports, newest first. Detail and control live in <a href="/pipeline">pipeline</a>.</p>
  <section><p class="microlabel">Variability · eta-ranked across stacked epochs</p>
  <div class="grid2v"><div id="vartable" class="quiet">Loading…</div><div id="lcpanel" class="quiet">Select a source for its lightcurve.</div></div></section>
  <div class="rule"></div>
  <div id="feed">Loading…</div>
</main>
<div class="lightbox" id="lightbox" onclick="this.style.display='none'"><img id="lightbox-img"></div>
<script>
const $=id=>document.getElementById(id);
const j=async u=>{const r=await fetch(u);if(!r.ok)throw new Error(r.status);return r.json()};
const fmt=(v,alt='—')=>v==null?alt:v;
function zoom(src){$('lightbox-img').src=src;$('lightbox').style.display='flex'}

async function load(){const d=await j('/api/science');
  $('feed').innerHTML=d.dates.length?d.dates.map(row=>{
    const v=row.verdict;
    const cards=row.mosaics.length?row.mosaics.map(m=>{
      const hour=(m.name.match(/T(\d{2})00_mosaic/)||[])[1];
      const qa=m.qa_result?`<span class="qa ${m.qa_result==='PASS'?'v-clean':'v-failed'}">${m.qa_result}</span>`
        :'<span class="qa v-none">NO QA</span>';
      const url=`/api/thumb/${row.date}/${m.name}.png`;
      return `<div class="mcard"><div class="hd"><span>${hour!=null?hour+':00 UTC':m.name}</span>${qa}</div>
        <img loading="lazy" src="${url}" onclick="zoom('${url}')">
        <div class="ft"><span>peak <b>${fmt(m.peak)}</b> Jy</span><span>rms <b>${fmt(m.rms_mjy)}</b> mJy</span>
        <span>DR <b>${fmt(m.dr)}</b></span><span>wt ${m.weights?'✓':'✗'}</span></div></div>`}).join('')
      :'<span class="quiet">No mosaics.</span>';
    const phot=row.phot.length?
      '<table class="phot"><tr><th>Epoch CSV</th><th>Sources</th><th>Bright</th><th>Median ratio</th></tr>'+
      row.phot.map(c=>{const r=c.median_ratio,ok=(r>=0.8&&r<=1.2);
        return `<tr><td>${c.name}</td><td>${fmt(c.n_sources)}</td><td>${fmt(c.n_bright)}</td>
        <td class="ratio ${r!=null?(ok?'ok':'bad'):''}">${fmt(r)}</td></tr>`}).join('')+'</table>'
      :'';
    return `<div class="daterow"><span class="d">${row.date}</span>
      <span class="v ${v?'v-'+v.toLowerCase():'v-none'}">${v||'in flight'}</span>
      <span class="meta">${row.n_tiles} tiles</span></div>
      <div class="cards">${cards}</div>
      ${phot?'<div style="margin-top:12px">'+phot+'</div>':''}` }).join('<div class="rule"></div>')
    :'<span class="quiet">No products found.</span>'}

let VAR=null,selSrc=null;
function spark(f){const w=96,h=22,mn=Math.min(...f),mx=Math.max(...f),rng=(mx-mn)||1;
  const pts=f.map((v,i)=>`${(i/Math.max(f.length-1,1))*w},${h-2-((v-mn)/rng)*(h-4)}`).join(' ');
  return `<svg width="${w}" height="${h}" style="vertical-align:middle"><polyline points="${pts}"
    fill="none" stroke="var(--acc)" stroke-width="1.3"/></svg>`}
function drawLC(row){selSrc=row.source_id;
  const W=640,H=230,padL=54,padR=14,padT=16,padB=44;
  const f=row.flux_jy,e=row.flux_err_jy,n=f.length;
  const lo=Math.min(...f.map((v,i)=>v-e[i])),hi=Math.max(...f.map((v,i)=>v+e[i]));
  const rng=(hi-lo)||1;
  const X=i=>padL+(i/Math.max(n-1,1))*(W-padL-padR);
  const Y=v=>padT+(1-(v-lo)/rng)*(H-padT-padB);
  let g='';
  for(let k=0;k<4;k++){const v=lo+rng*k/3;
    g+=`<line x1="${padL}" y1="${Y(v)}" x2="${W-padR}" y2="${Y(v)}" stroke="#161b22"/>
        <text x="${padL-6}" y="${Y(v)+3}" text-anchor="end">${v.toFixed(v<0.1?3:2)}</text>`}
  const pts=f.map((v,i)=>`${X(i)},${Y(v)}`).join(' ');
  const bars=f.map((v,i)=>`<line x1="${X(i)}" y1="${Y(v-e[i])}" x2="${X(i)}" y2="${Y(v+e[i])}"
    stroke="var(--acc)" opacity=".45"/>`).join('');
  const dots=f.map((v,i)=>`<circle cx="${X(i)}" cy="${Y(v)}" r="3" fill="var(--acc)">
    <title>${row.epochs[i]} · ${v.toFixed(4)} ± ${e[i].toFixed(4)} Jy</title></circle>`).join('');
  const xt=row.epochs.map((ep,i)=>(i===0||i===n-1||n<=6)?
    `<text x="${X(i)}" y="${H-8}" text-anchor="${i===n-1?'end':(i===0?'start':'middle')}">${ep.slice(5,10)}</text>`:'').join('');
  $('lcpanel').className='';
  $('lcpanel').innerHTML=`<div class="t"><b style="color:var(--tx)">${row.source_id}</b>
    · ${row.n_epochs} epochs · ⟨S⟩ ${row.mean_flux_jy} Jy · V ${row.v} · η ${row.eta}</div>
    <svg viewBox="0 0 ${W} ${H}">${g}<polyline points="${pts}" fill="none" stroke="var(--acc)"
    stroke-width="1.2" opacity=".6"/>${bars}${dots}${xt}
    <text x="14" y="${H/2}" transform="rotate(-90 14 ${H/2})" text-anchor="middle">Jy</text></svg>`;
  renderVar()}
function renderVar(){if(!VAR)return;
  $('vartable').className='';
  $('vartable').innerHTML=VAR.sources.length?
    `<table class="var"><thead><tr><th>Source</th><th>N</th><th>⟨S⟩ Jy</th><th>V</th><th style="text-transform:none">η</th>
     <th style="text-align:left;padding-left:12px">Lightcurve</th></tr></thead><tbody>`+
    VAR.sources.slice(0,18).map(r=>
      `<tr data-s="${r.source_id}" class="${r.source_id===selSrc?'sel':''}">
       <td>${r.source_id}</td><td>${r.n_epochs}</td><td>${r.mean_flux_jy}</td>
       <td>${r.v}</td><td>${r.eta}</td>
       <td style="text-align:left;padding-left:12px">${spark(r.flux_jy)}</td></tr>`).join('')+
    '</tbody></table>'
    :`Need ≥2 photometry epochs for variability (have ${VAR.n_epochs}).`;
  document.querySelectorAll('.var tbody tr').forEach(tr=>tr.onclick=()=>{
    const row=VAR.sources.find(x=>x.source_id===tr.dataset.s);if(row)drawLC(row)})}
async function loadVar(){try{VAR=await j('/api/variability');renderVar();
  if(VAR.sources.length&&!selSrc)drawLC(VAR.sources[0])}catch(e){$('vartable').textContent='—'}}

load();loadVar();
setInterval(loadVar,120000);
setInterval(()=>{$('hd-utc').textContent=new Date().toISOString().slice(0,19)+'Z'},1000);
setInterval(load,60000);
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the telescope-status home page."""
    return HTMLResponse(HOME_PAGE)


@app.get("/pipeline", response_class=HTMLResponse)
def pipeline_page() -> HTMLResponse:
    """Serve the pipeline coverage + control console."""
    return HTMLResponse(PIPELINE_PAGE)


@app.get("/science", response_class=HTMLResponse)
def science_page() -> HTMLResponse:
    """Serve the science-products gallery."""
    return HTMLResponse(SCIENCE_PAGE)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8766, log_level="info")
