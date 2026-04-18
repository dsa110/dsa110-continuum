"""Step 4 display: calibrated tile dirty image with source markers at correct positions.

The dirty image may contain grating lobes (PSF artifacts) due to the sparse DSA-110
array. We display it with source markers showing the injected source positions, and
annotate the peak flux at each source location.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dsa110_continuum.simulation.plot_style import apply_pipeline_style

OUT_DIR = Path("/home/user/workspace/dsa110-continuum/pipeline_outputs/step4")

# Sources injected into tile 0 (from the simulation)
SOURCES = [
    ("S0", 344.9797, 17.5769, 0.349),
    ("S1", 343.9332, 16.9334, 4.011),
    ("S2", 345.2440, 17.0082, 1.442),
    ("S3", 344.7405, 15.0343, 2.853),
    ("S4", 342.8566, 16.0012, 0.542),
]

apply_pipeline_style()

# Load dirty and restored images
dirty_path   = OUT_DIR / "wsclean_out" / "wsclean-dirty.fits"
restored_path = OUT_DIR / "wsclean_out" / "wsclean-image.fits"

with fits.open(str(dirty_path)) as hdul:
    dirty_hdr  = hdul[0].header
    dirty_data = hdul[0].data.squeeze()

with fits.open(str(restored_path)) as hdul:
    rest_data  = hdul[0].data.squeeze()

wcs = WCS(dirty_hdr, naxis=2)

# ── Figure layout: dirty + restored side by side ──────────────────────────────
fig = plt.figure(figsize=(14, 6.5))

for col_idx, (data, title_suffix) in enumerate(
    [(dirty_data, "Dirty"), (rest_data, "Restored")]
):
    ax = fig.add_subplot(1, 2, col_idx + 1, projection=wcs)

    # Scale: clip at 2×RMS to avoid sidelobe domination of color scale
    rms  = np.nanstd(data)
    vmax = 3.0 * rms
    vmin = -0.5 * rms

    im = ax.imshow(data, origin="lower", cmap="inferno",
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.04)
    cbar.set_label("Flux density (Jy/beam)", fontsize=9)

    # Mark each source
    for name, ra, dec, flux in SOURCES:
        px, py = wcs.all_world2pix(ra, dec, 0)
        in_img = (0 <= px < dirty_data.shape[1]) and (0 <= py < dirty_data.shape[0])
        color  = "cyan" if in_img else "gray"
        marker = "+" if in_img else "x"
        ax.plot(px, py, marker, color=color, ms=14, mew=2.0, zorder=6)

        if in_img:
            # Sample image value at source position
            px_i = int(np.round(float(px)))
            py_i = int(np.round(float(py)))
            px_i = np.clip(px_i, 0, data.shape[1]-1)
            py_i = np.clip(py_i, 0, data.shape[0]-1)
            val = float(data[py_i, px_i])
            label = f"{name}  {flux:.2f} Jy inj.\n   {val:.2f} Jy/beam"
        else:
            label = f"{name}  {flux:.2f} Jy\n   (outside)"

        ax.annotate(label, xy=(px, py), xytext=(8, 4),
                    textcoords="offset points", color=color,
                    fontsize=7, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5, ec="none"))

    ax.set_xlabel("RA (J2000)")
    ax.set_ylabel("Dec (J2000)")
    ax.set_title(f"Tile 0 — {title_suffix} image\n"
                 f"Phase centre RA≈344.124°  Dec=16.15°\n"
                 f"RMS = {rms:.3f} Jy/beam",
                 fontsize=10)

    # Add a 1° scale bar annotation
    ax.text(0.02, 0.04, f"Cell = 20\"  Image = 2.84°", transform=ax.transAxes,
            fontsize=8, color="white",
            bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.6, ec="none"))

fig.suptitle("Step 4 — Calibrated Tile 0 Dirty Image  (32 ant · 24 int · 1 SB · 5° noise=1 Jy)",
             fontsize=11, y=1.01)

fig.tight_layout()
out_fig = OUT_DIR / "step4_tile0_image.png"
fig.savefig(str(out_fig), dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Saved → {out_fig}")
print(f"\n=== Source flux at image pixels ===")
for name, ra, dec, flux in SOURCES:
    px, py = wcs.all_world2pix(ra, dec, 0)
    in_img = (0 <= px < dirty_data.shape[1]) and (0 <= py < dirty_data.shape[0])
    if in_img:
        px_i = int(np.clip(np.round(float(px)), 0, dirty_data.shape[1]-1))
        py_i = int(np.clip(np.round(float(py)), 0, dirty_data.shape[0]-1))
        val_d = float(dirty_data[py_i, px_i])
        val_r = float(rest_data[py_i, px_i])
        print(f"  {name}: injected={flux:.3f} Jy  dirty={val_d:.3f} Jy/beam  restored={val_r:.3f} Jy/beam  ratio={val_d/flux:.2f}")
    else:
        print(f"  {name}: OUTSIDE image")
