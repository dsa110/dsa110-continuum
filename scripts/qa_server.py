"""
DSA-110 Continuum Pipeline QA Server
Serves a live dashboard at http://lxd110h17:8767
"""
import glob, os, hashlib, logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from astropy.io import fits

from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DSA-110 QA Dashboard")

# Paths
STAGE       = Path("/stage/dsa110-contimg")
PRODUCTS    = Path("/data/dsa110-continuum/products/mosaics")
THUMB_DIR   = Path("/tmp/qa_thumbs")
THUMB_DIR.mkdir(exist_ok=True)

EPOCHS = [
    ("2026-01-25", "T0200"),
    ("2026-02-12", "T0000"),
    ("2026-02-15", "T0000"),
    ("2026-02-23", "T0000"),
    ("2026-02-25", "T0000"),
    ("2026-02-26", "T0000"),
]

def find_mosaic(date, epoch):
    p = STAGE / f"images/mosaic_{date}" / f"{date}{epoch}_mosaic.fits"
    return p if p.exists() else None

def find_csv(date, epoch):
    d = PRODUCTS / date
    if not d.exists(): return None
    matches = sorted(d.glob(f"*{epoch}*phot.csv"))
    return matches[-1] if matches else None

def cal_tables():
    bs = {Path(p).stem.split("_")[0] for p in glob.glob(str(STAGE/"ms/*.b"))}
    return bs  # set of date strings like "2026-02-12T22:26:05"

def get_metrics(date, epoch):
    fits_p = find_mosaic(date, epoch)
    csv_p  = find_csv(date, epoch)
    m = dict(date=date, epoch=epoch, fits=fits_p is not None, csv=csv_p is not None,
             peak=None, rms=None, dr=None, ratio=None, n_bright=0, status="missing")
    if not fits_p: return m
    try:
        d = fits.open(fits_p)[0].data.squeeze().astype(np.float32)
        m["peak"] = float(np.nanmax(d))
        m["rms"]  = float(np.nanstd(d[np.abs(d) < 0.05])) * 1000
        m["dr"]   = m["peak"] / (m["rms"] / 1000) if m["rms"] else None
    except Exception as e:
        logger.warning(f"FITS error {date}: {e}")
    if csv_p:
        try:
            df = pd.read_csv(csv_p)
            bright = df[df.get("nvss_flux_jy", pd.Series(dtype=float)) > 0.06]
            m["n_bright"] = len(bright)
            if len(bright):
                m["ratio"] = float(bright["dsa_nvss_ratio"].median())
        except Exception as e:
            logger.warning(f"CSV error {date}: {e}")
    r = m["ratio"]
    if r is not None and not np.isnan(r):
        m["status"] = "pass" if 0.8 <= r <= 1.2 else "fail"
    elif m["fits"]:
        m["status"] = "no_phot"
    return m

def make_thumbnail(date, epoch):
    fits_p = find_mosaic(date, epoch)
    if not fits_p: return None
    mtime = fits_p.stat().st_mtime
    key = hashlib.md5(f"{fits_p}{mtime}".encode()).hexdigest()[:8]
    out = THUMB_DIR / f"{date}_{epoch}_{key}.png"
    if out.exists(): return out
    # Clear old thumbs for this date
    for old in THUMB_DIR.glob(f"{date}_{epoch}_*.png"):
        old.unlink(missing_ok=True)
    try:
        d = fits.open(fits_p)[0].data.squeeze().astype(np.float32)
        fig, ax = plt.subplots(figsize=(12, 4), dpi=90)
        vmax = float(np.nanpercentile(d[np.isfinite(d)], 99.5))
        rms  = float(np.nanstd(d[np.abs(d) < 0.05]))
        ax.imshow(d, origin="lower", cmap="inferno",
                  norm=PowerNorm(gamma=0.35, vmin=0, vmax=vmax), aspect="auto")
        peak = float(np.nanmax(d))
        ax.set_title(f"{date}  |  Peak {peak:.2f} Jy  RMS {rms*1000:.1f} mJy/beam  DR {peak/rms:.0f}",
                     color="white", fontsize=10, pad=4)
        ax.axis("off")
        fig.patch.set_facecolor("#111")
        plt.tight_layout(pad=0.2)
        plt.savefig(out, dpi=90, bbox_inches="tight", facecolor="#111")
        plt.close()
        return out
    except Exception as e:
        logger.error(f"Thumbnail error {date}: {e}")
        return None

@app.get("/thumb/{date}/{epoch}.png")
def thumbnail(date: str, epoch: str):
    p = make_thumbnail(date, epoch)
    if not p: return Response(status_code=404)
    return Response(content=p.read_bytes(), media_type="image/png",
                    headers={"Cache-Control": "max-age=60"})

@app.get("/", response_class=HTMLResponse)
def dashboard():
    all_metrics = [get_metrics(d, e) for d, e in EPOCHS]
    cals = cal_tables()
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    n_pass = sum(1 for m in all_metrics if m["status"] == "pass")
    n_fail = sum(1 for m in all_metrics if m["status"] == "fail")
    n_miss = sum(1 for m in all_metrics if m["status"] in ("missing", "no_phot"))

    def sc(v, dec=2): return f"{v:.{dec}f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "—"
    def badge(s):
        cols = {"pass":"#4caf50","fail":"#f44336","missing":"#888","no_phot":"#ff9800"}
        return f'<span style="background:{cols.get(s,"#888")};color:#fff;padding:3px 8px;border-radius:10px;font-size:.85em">{s.upper()}</span>'
    def cal_badge(date):
        has = any(date in c for c in cals)
        return f'<span style="background:{"#4caf50" if has else "#f44336"};color:#fff;padding:2px 6px;border-radius:8px;font-size:.8em">{"✓ yes" if has else "✗ no"}</span>'
    def ratio_cell(r):
        if r is None or (isinstance(r, float) and np.isnan(r)): return "—"
        ok = 0.8 <= r <= 1.2
        c = "#4caf50" if ok else "#f44336"
        return f'<span style="color:{c};font-weight:bold">{r:.3f}</span>'

    rows = ""
    cards = ""
    for m in all_metrics:
        rows += f"""<tr>
            <td><b>{m['date']}</b></td>
            <td>{badge(m['status'])}</td>
            <td>{cal_badge(m['date'])}</td>
            <td>{sc(m['peak'])}</td>
            <td>{sc(m['rms'],1)}</td>
            <td>{sc(m['dr'],0)}</td>
            <td>{ratio_cell(m['ratio'])}</td>
            <td>{m['n_bright']}</td>
        </tr>"""
        thumb_url = f"/thumb/{m['date']}/{m['epoch']}.png"
        has_thumb = find_mosaic(m['date'], m['epoch']) is not None
        img_html = f'<img src="{thumb_url}" style="width:100%;border-radius:4px" loading="lazy">' if has_thumb else '<div style="color:#666;padding:20px;text-align:center">No mosaic</div>'
        status_bar = {"pass":"#4caf50","fail":"#f44336","missing":"#555","no_phot":"#ff9800"}.get(m['status'],"#555")
        cards += f"""<div style="background:#1a1a1a;border-radius:8px;overflow:hidden;border:1px solid {status_bar}">
            <div style="background:{status_bar};padding:6px 12px;font-size:.9em;color:#fff;display:flex;justify-content:space-between">
                <b>{m['date']}</b><span>{m['status'].upper()}</span>
            </div>
            {img_html}
            <div style="padding:8px 12px;font-size:.85em;color:#aaa">
                Peak: {sc(m['peak'])} Jy &nbsp;|&nbsp; RMS: {sc(m['rms'],1)} mJy &nbsp;|&nbsp; Ratio: {ratio_cell(m['ratio'])}
            </div>
        </div>"""

    html = f"""<!DOCTYPE html><html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DSA-110 QA Dashboard</title>
<style>
  body{{background:#111;color:#eee;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;padding:16px}}
  h1{{color:#ddd;font-size:1.4em;margin:0 0 4px}}
  .subtitle{{color:#666;font-size:.85em;margin-bottom:20px}}
  .summary{{display:flex;gap:12px;margin-bottom:24px;flex-wrap:wrap}}
  .stat{{background:#1a1a1a;border-radius:8px;padding:12px 20px;text-align:center;min-width:80px}}
  .stat .n{{font-size:2em;font-weight:bold}}
  .stat .l{{font-size:.8em;color:#888;margin-top:2px}}
  table{{width:100%;border-collapse:collapse;font-size:.88em;margin-bottom:32px}}
  th,td{{padding:10px 12px;text-align:center;border-bottom:1px solid #2a2a2a}}
  th{{background:#1a1a1a;color:#aaa;font-weight:600}}
  tr:hover{{background:#1a1a1a}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px;margin-top:8px}}
  @media(max-width:600px){{.grid{{grid-template-columns:1fr}}}}
</style>
<script>setTimeout(()=>location.reload(),60000)</script>
</head><body>
<h1>DSA-110 Continuum QA</h1>
<div class="subtitle">Updated: {now} &nbsp;·&nbsp; Auto-refresh: 60s &nbsp;·&nbsp; Target ratio: 0.8–1.2</div>
<div class="summary">
  <div class="stat"><div class="n" style="color:#4caf50">{n_pass}</div><div class="l">PASS</div></div>
  <div class="stat"><div class="n" style="color:#f44336">{n_fail}</div><div class="l">FAIL</div></div>
  <div class="stat"><div class="n" style="color:#888">{n_miss}</div><div class="l">MISSING</div></div>
  <div class="stat"><div class="n">{len(EPOCHS)}</div><div class="l">TOTAL</div></div>
</div>
<table>
  <tr><th>Date</th><th>Status</th><th>Cal Tables</th><th>Peak (Jy)</th><th>RMS (mJy)</th><th>Dyn Range</th><th>DSA/NVSS Ratio</th><th>Bright Srcs</th></tr>
  {rows}
</table>
<h2 style="color:#bbb;font-size:1.1em;margin-bottom:12px">Mosaic Thumbnails</h2>
<div class="grid">{cards}</div>
</body></html>"""
    return HTMLResponse(html)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8767, log_level="warning")
