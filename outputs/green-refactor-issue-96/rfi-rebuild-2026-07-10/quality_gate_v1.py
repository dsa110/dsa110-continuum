"""dsa110-mosaic-quality-gate/v1 — recovered verbatim from the 2026-07-10 Codex
session (rollout-2026-07-09T17-48-28, call_d8EVWxIRamwbrKanR97wgGBU) that
produced pr1-quality-gate-2026-07-10/2026-01-25T2200_quality_gate.json.
Method and thresholds are unchanged; only parameterized via argv.

Usage: python3 quality_gate_v1.py IMAGE.fits WEIGHTS.fits OUT.json
"""
import sys
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import numpy as np
from scipy import ndimage
from astropy.io import fits

image_path = Path(sys.argv[1])
weight_path = Path(sys.argv[2])
out_json = Path(sys.argv[3])
out_json.parent.mkdir(parents=True, exist_ok=True)

with fits.open(image_path, memmap=True) as h:
    image = np.squeeze(h[0].data).astype(float, copy=False)
    hdr = h[0].header
with fits.open(weight_path, memmap=True) as h:
    weight = np.squeeze(h[0].data).astype(float, copy=False)

support = np.isfinite(image) & np.isfinite(weight) & (weight > 0)
scale_deg = abs(float(hdr.get("CDELT1", hdr.get("CD1_1", 1 / 180))))


def rms(x):
    x = x[np.isfinite(x)]
    m = np.median(x)
    return float(1.4826 * np.median(np.abs(x - m)))


def herr(window):
    x = np.array(window, float, copy=True)
    good = np.isfinite(x)
    med = np.median(x[good])
    sig = rms(x[good])
    x[~good] = med
    x = np.clip(x, med - 5 * sig, med + 5 * sig) - med
    x *= np.hanning(x.shape[0])[:, None] * np.hanning(x.shape[1])[None, :]
    p = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(x.shape[0], d=scale_deg))
    fx = np.fft.fftshift(np.fft.fftfreq(x.shape[1], d=scale_deg))
    xx, yy = np.meshgrid(fx, fy)
    rr = np.hypot(xx, yy)
    ang = (np.arctan2(yy, xx) + np.pi) % np.pi
    ann = (rr >= 12) & (rr <= 45)
    idx = np.minimum((ang[ann] / np.pi * 180).astype(int), 179)
    sums = np.bincount(idx, weights=p[ann], minlength=180)
    cnt = np.bincount(idx, minlength=180)
    directional = sums / np.maximum(cnt, 1)
    return {
        "anisotropy_peak_to_median": float(np.max(directional) / np.median(directional)),
        "fringe_bandpower_to_lowfreq_median": float(np.median(directional) / np.median(p[rr < 5])),
        "source_clip_sigma": 5,
        "hanning": True,
        "annulus_cycles_per_degree": [12, 45],
        "orientation_bins": 180,
    }


labels, n = ndimage.label(support)
components = []
for lab in range(1, n + 1):
    yy, xx = np.where(labels == lab)
    if len(xx) < 10000:
        continue
    cx = int(np.median(xx))
    cy = int(np.median(yy))
    half = 350
    y0 = max(0, cy - half)
    y1 = min(image.shape[0], cy + half)
    x0 = max(0, cx - half)
    x1 = min(image.shape[1], cx + half)
    win = image[y0:y1, x0:x1]
    components.append({
        "label": int(lab),
        "pixels": int(len(xx)),
        "window_bounds_xyxy": [x0, y0, x1, y1],
        "central_rms_mjy_per_beam": rms(win) * 1e3,
        "herr": herr(win),
    })

dist = ndimage.distance_transform_edt(support)
edge = support & (dist <= 64)
interior = support & (dist >= 128)
edge_rms = rms(image[edge])
interior_rms = rms(image[interior])
local_noise = 1 / np.sqrt(weight[support])

record = {
    "schema": "dsa110-mosaic-quality-gate/v1",
    "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    "input": {
        "image": str(image_path),
        "weight": str(weight_path),
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "weight_sha256": hashlib.sha256(weight_path.read_bytes()).hexdigest(),
        "shape": list(image.shape),
        "pixel_scale_deg": scale_deg,
    },
    "thresholds": {
        "central_rms_mjy_per_beam_max": 8.0,
        "herr_anisotropy_max": 5000.0,
        "edge_to_interior_rms_ratio_max": 2.0,
        "min_positive_weight_fraction": 0.5,
    },
    "metrics": {
        "positive_weight_fraction": float(support.mean()),
        "effective_noise_mjy_per_beam": {
            "median": float(np.median(local_noise) * 1e3),
            "p95": float(np.percentile(local_noise, 95) * 1e3),
        },
        "edge": {
            "band_width_pixels": 64,
            "edge_rms_mjy_per_beam": edge_rms * 1e3,
            "interior_rms_mjy_per_beam": interior_rms * 1e3,
            "edge_to_interior_rms_ratio": edge_rms / interior_rms,
        },
        "components": components,
    },
    "method": {
        "herr": "central component window; 5-sigma source clip; Hanning; FFT annulus 12-45 cycles/degree; peak/median orientation bandpower",
        "edge": "64-pixel in-support boundary band relative to >=128-pixel interior",
    },
}

checks = []
for c in components:
    label = c["label"]
    crms = c["central_rms_mjy_per_beam"]
    aniso = c["herr"]["anisotropy_peak_to_median"]
    checks.append({"name": "component_%s_central_rms" % label, "value": crms,
                   "threshold": 8.0, "operator": "<=", "passed": crms <= 8.0})
    checks.append({"name": "component_%s_herr" % label, "value": aniso,
                   "threshold": 5000.0, "operator": "<=", "passed": aniso <= 5000.0})
checks += [
    {"name": "edge_to_interior_rms_ratio", "value": edge_rms / interior_rms,
     "threshold": 2.0, "operator": "<=", "passed": edge_rms / interior_rms <= 2.0},
    {"name": "positive_weight_fraction", "value": float(support.mean()),
     "threshold": 0.5, "operator": ">=", "passed": float(support.mean()) >= 0.5},
]
record["checks"] = checks
record["verdict"] = {
    "status": "PASS" if all(c["passed"] for c in checks) else "FAIL",
    "science_ready": all(c["passed"] for c in checks),
    "failed_checks": [c["name"] for c in checks if not c["passed"]],
}
out_json.write_text(json.dumps(record, indent=2) + "\n")
print(out_json)
print(json.dumps(record["verdict"], indent=2))
for c in checks:
    print(c["name"], c["value"], "PASS" if c["passed"] else "FAIL")
