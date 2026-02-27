#!/usr/bin/env python3
"""
Lightweight read-only HTTP monitoring server for DSA-110 pipeline on H17.
Run with: uvicorn scripts.monitor_server:app --host 0.0.0.0 --port 8765
Expose externally with: cloudflared tunnel --url http://localhost:8765
"""

import subprocess
import shutil
import os
import glob
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="DSA-110 H17 Monitor", version="1.0")

MS_DIR = Path("/stage/dsa110-contimg/ms")
IMAGES_DIR = Path("/stage/dsa110-contimg/images")
PRODUCTS_DIR = Path("/data/dsa110-contimg/products")
INCOMING_DIR = Path("/data/incoming")

def bytes_to_human(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}

@app.get("/disk")
def disk():
    results = {}
    for label, path in [("/", "/"), ("/data", "/data"), ("/stage", "/stage")]:
        try:
            usage = shutil.disk_usage(path)
            results[label] = {
                "total": bytes_to_human(usage.total),
                "used": bytes_to_human(usage.used),
                "free": bytes_to_human(usage.free),
                "pct_used": round(usage.used / usage.total * 100, 1),
            }
        except Exception as e:
            results[label] = {"error": str(e)}
    return results

@app.get("/ms")
def ms_files():
    if not MS_DIR.exists():
        return {"error": f"{MS_DIR} does not exist"}

    ms_files = sorted(MS_DIR.glob("*.ms"))
    by_date: dict = {}
    for f in ms_files:
        name = f.name
        date = name[:10] if len(name) >= 10 else "unknown"
        by_date.setdefault(date, []).append(name)

    return {
        "total": len(ms_files),
        "by_date": {date: len(files) for date, files in sorted(by_date.items())},
        "latest": ms_files[-1].name if ms_files else None,
    }


@app.get("/processes")
def processes():
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.splitlines()
        keywords = ["wsclean", "dsa110", "python", "casa", "bane", "aegean", "uvicorn"]
        relevant = [
            l for l in lines
            if any(kw in l.lower() for kw in keywords)
            and "grep" not in l
            and "monitor_server" not in l
        ]
        return {"processes": relevant, "count": len(relevant)}
    except Exception as e:
        return {"error": str(e)}

@app.get("/images")
def images():
    if not IMAGES_DIR.exists():
        return {"error": f"{IMAGES_DIR} does not exist"}

    mosaics = sorted(IMAGES_DIR.iterdir()) if IMAGES_DIR.exists() else []
    result = []
    for m in mosaics:
        if m.is_dir():
            size = sum(f.stat().st_size for f in m.rglob("*") if f.is_file())
            fits_files = list(m.glob("*.fits"))
            result.append({
                "name": m.name,
                "size": bytes_to_human(size),
                "fits_count": len(fits_files),
            })
    return {"mosaics": result}


@app.get("/logs")
def logs(lines: int = 50):
    """Tail recent log files from common locations."""
    log_candidates = [
        "/tmp/dsa110-convert/.log",
        "/data/dsa110-continuum/.log",
        "/data/dsa110-contimg/*.log",
    ]
    found = []
    for pattern in log_candidates:
        found.extend(glob.glob(pattern))

    if not found:
        return {"logs": [], "message": "No log files found"}

    latest = max(found, key=os.path.getmtime)
    try:
        result = subprocess.run(
            ["tail", f"-{lines}", latest],
            capture_output=True, text=True, timeout=5
        )
        return {
            "file": latest,
            "modified": datetime.utcfromtimestamp(os.path.getmtime(latest)).isoformat() + "Z",
            "lines": result.stdout.splitlines(),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/status")
def status():
    """Combined summary: disk + MS counts + active processes."""
    disk_info = disk()
    ms_info = ms_files()
    proc_info = processes()

    active_jobs = [p for p in proc_info.get("processes", [])
                   if any(kw in p.lower() for kw in ["wsclean", "dsa110 convert", "bane"])]

    return {
        "time": datetime.utcnow().isoformat() + "Z",
        "disk": disk_info,
        "ms_files": ms_info,
        "active_pipeline_jobs": len(active_jobs),
        "active_jobs_detail": active_jobs,
    }

