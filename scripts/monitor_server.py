#!/usr/bin/env python3
import subprocess, shutil, os, glob
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
app = FastAPI(title="DSA-110 H17 Monitor", version="1.1")
MS_DIR = Path("/stage/dsa110-contimg/ms")
IMAGES_DIR = Path("/stage/dsa110-contimg/images")
EXEC_SECRET = os.environ.get("TIDY_EXEC_SECRET", "")
def bytes_to_human(n):
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024: return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"
@app.get("/health")
def health(): return {"status":"ok","time":datetime.utcnow().isoformat()+"Z"}
@app.get("/disk")
def disk():
    results = {}
    for label, path in [("/","/"),("/data","/data"),("/stage","/stage")]:
        try:
            u = shutil.disk_usage(path)
            results[label] = {"total":bytes_to_human(u.total),"used":bytes_to_human(u.used),"free":bytes_to_human(u.free),"pct_used":round(u.used/u.total*100,1)}
        except Exception as e: results[label] = {"error":str(e)}
    return results
@app.get("/ms")
def ms_files():
    files = sorted(MS_DIR.glob("*.ms")) if MS_DIR.exists() else []
    by_date = {}
    for f in files:
        d = f.name[:10]; by_date.setdefault(d,[]).append(f.name)
    return {"total":len(files),"by_date":{d:len(v) for d,v in sorted(by_date.items())},"latest":files[-1].name if files else None}
@app.get("/processes")
def processes():
    try:
        r = subprocess.run(["ps","aux"],capture_output=True,text=True,timeout=10)
        kw = ["wsclean","dsa110","python","casa","bane","aegean","uvicorn"]
        lines = [l for l in r.stdout.splitlines() if any(k in l.lower() for k in kw) and "grep" not in l and "monitor_server" not in l]
        return {"processes":lines,"count":len(lines)}
    except Exception as e: return {"error":str(e)}
@app.get("/images")
def images():
    if not IMAGES_DIR.exists(): return {"error":f"{IMAGES_DIR} does not exist"}
    result = []
    for m in sorted(IMAGES_DIR.iterdir()):
        if m.is_dir():
            size = sum(f.stat().st_size for f in m.rglob("*") if f.is_file())
            result.append({"name":m.name,"size":bytes_to_human(size),"fits_count":len(list(m.glob("*.fits")))})
    return {"mosaics":result}
@app.get("/logs")
def logs(lines: int = 50):
    found = []
    for pat in ["/tmp/batch_pipeline_*.log","/tmp/convert_*.log","/tmp/dsa110-convert/*.log","/data/dsa110-continuum/*.log","/data/dsa110-contimg/*.log"]:
        found.extend(glob.glob(pat))
    if not found: return {"message":"No log files found","logs":[]}
    latest = max(found, key=os.path.getmtime)
    try:
        r = subprocess.run(["tail",f"-{lines}",latest],capture_output=True,text=True,timeout=5)
        return {"file":latest,"modified":datetime.utcfromtimestamp(os.path.getmtime(latest)).isoformat()+"Z","lines":r.stdout.splitlines()}
    except Exception as e: return {"error":str(e)}
class ExecRequest(BaseModel):
    command: str; secret: str; timeout: int = 60
@app.post("/exec")
def exec_command(req: ExecRequest):
    if not EXEC_SECRET or req.secret != EXEC_SECRET:
        return JSONResponse(status_code=403, content={"error":"Forbidden"})
    try:
        r = subprocess.run(req.command,shell=True,capture_output=True,text=True,timeout=req.timeout)
        return {"stdout":r.stdout,"stderr":r.stderr,"returncode":r.returncode,"command":req.command}
    except subprocess.TimeoutExpired: return {"error":f"Timed out after {req.timeout}s","command":req.command}
    except Exception as e: return {"error":str(e),"command":req.command}
@app.get("/status")
def status():
    d,m,p = disk(),ms_files(),processes()
    active = [x for x in p.get("processes",[]) if any(k in x.lower() for k in ["wsclean","dsa110 convert","bane"])]
    return {"time":datetime.utcnow().isoformat()+"Z","disk":d,"ms_files":m,"active_pipeline_jobs":len(active),"active_jobs_detail":active}
