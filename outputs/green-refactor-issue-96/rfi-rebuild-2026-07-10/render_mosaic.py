"""Render mosaic / weight FITS to PNG with a provenance sidecar, matching the
pr1-validation render format (asinh stretch, 0.5/99.5 percentiles).

Usage: python3 render_mosaic.py FITS OUT_PNG {image|weight} "NOTE"
"""
import sys, json, hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

src = Path(sys.argv[1])
out_png = Path(sys.argv[2])
kind = sys.argv[3]
note = sys.argv[4]

with fits.open(src, memmap=True) as h:
    data = np.squeeze(h[0].data).astype(float, copy=False)

finite = data[np.isfinite(data)]
med = np.median(finite)
robust_rms = float(1.4826 * np.median(np.abs(finite - med)))
vmin, vmax = np.percentile(finite, [0.5, 99.5])

disp = np.array(data, float, copy=True)
if kind == "image":
    a = np.arcsinh(np.clip(disp, vmin, vmax) / (3 * robust_rms if robust_rms > 0 else 1))
    stretch = "asinh"
else:
    a = np.clip(disp, vmin, vmax)
    stretch = "linear"

fig, ax = plt.subplots(figsize=(18, 5), dpi=150)
im = ax.imshow(a, origin="lower", cmap="gray", aspect="equal")
ax.set_title(f"{src.name} [{kind}] rendered {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%MZ')}\n"
             f"robust RMS {robust_rms*1e3:.3f} mJy/beam, stretch {stretch} p[0.5,99.5]",
             fontsize=9)
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(out_png)
plt.close(fig)

sidecar = {
    "artifact": str(out_png),
    "source_fits": str(src),
    "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
    "source_fits_mtime_utc": datetime.fromtimestamp(src.stat().st_mtime, timezone.utc).isoformat(),
    "rendered_at_utc": datetime.now(timezone.utc).isoformat(),
    "shape": list(data.shape),
    "display": {
        "stretch": stretch,
        "percentiles": [0.5, 99.5],
        "vmin_jy_per_beam": float(vmin),
        "vmax_jy_per_beam": float(vmax),
    },
    "robust_rms_jy_per_beam": robust_rms,
    "note": note,
}
Path(str(out_png).replace(".png", ".json")).write_text(json.dumps(sidecar, indent=2) + "\n")
print(out_png)
print("robust_rms_mjy", robust_rms * 1e3)
