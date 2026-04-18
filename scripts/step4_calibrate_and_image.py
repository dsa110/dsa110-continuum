"""Step 4 continuation: calibrate + WSClean on existing meridian MS.

Assumes these files exist:
  pipeline_outputs/step4/tile00_corrupted_meridian.ms
  pipeline_outputs/step4/sim_cal_combined.uvh5
"""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astropy.wcs import WCS
from astropy.io import fits

from dsa110_continuum.simulation.pipeline import SimulatedPipeline
from dsa110_continuum.simulation.plot_style import apply_pipeline_style

OUT_DIR = Path("/home/user/workspace/dsa110-continuum/pipeline_outputs/step4")

CAL_FLUX_JY = 10.0
median_ra   = 344.124   # from tile 0 parameters
dec_deg     = 16.15

ms_phased       = OUT_DIR / "tile00_corrupted_meridian.ms"
cal_uvh5_combined = OUT_DIR / "sim_cal_combined.uvh5"

assert ms_phased.exists(),        f"Missing: {ms_phased}"
assert cal_uvh5_combined.exists(), f"Missing: {cal_uvh5_combined}"

print(f"Using:\n  MS:  {ms_phased}\n  Cal: {cal_uvh5_combined}")

# ── Calibrate ─────────────────────────────────────────────────────────────────
pipeline = SimulatedPipeline(
    work_dir=OUT_DIR,
    niter=10000,
    cell_arcsec=20.0,
    image_size=512,
    wsclean_mem_gb=4.0,
)

print("\nRunning Jacobi gain calibration…")
pipeline._calibrate(
    target_ms=ms_phased,
    cal_uvh5=cal_uvh5_combined,
    cal_flux_jy=CAL_FLUX_JY,
    work_dir=OUT_DIR,
)
print("  Done")

# Free cal file
cal_uvh5_combined.unlink(missing_ok=True)
print(f"  Freed calibrator UVH5")

# ── WSClean MFS imaging ────────────────────────────────────────────────────────
print("\nRunning WSClean MFS imaging…")

img_dir = OUT_DIR / "wsclean_out"
img_dir.mkdir(exist_ok=True)
prefix  = str(img_dir / "wsclean")

cmd = [
    "wsclean",
    "-name", prefix,
    "-size", "512", "512",
    "-scale", "20asec",
    "-weight", "briggs", "0.0",
    "-join-channels",
    "-channels-out", "4",
    "-niter", "10000",
    "-mgain", "0.8",
    "-auto-threshold", "3.0",
    "-auto-mask", "5.0",
    "-pol", "I",
    "-data-column", "CORRECTED_DATA",
    "-make-psf",
    "-no-update-model-required",
    "-abs-mem", "4",
    str(ms_phased),
]
print("  WSClean command:", " ".join(cmd[:15]), "...")
r = subprocess.run(cmd, capture_output=True, text=True)
print("  Return code:", r.returncode)
if r.returncode != 0:
    print("STDERR:", r.stderr[-3000:])
    sys.exit(1)

for line in r.stdout.split('\n'):
    if any(k in line for k in ['beam', 'peak', 'Peak', 'iter', 'threshold', 'DONE',
                                'major', 'Stopped', 'mask', 'Restored', 'Maximum',
                                'Fitting', 'Theoretic', 'Predicted', 'Deconvolving']):
        print("  WSClean:", line.strip())

# ── Find output FITS ──────────────────────────────────────────────────────────
restored_path = Path(f"{prefix}-MFS-image.fits")
dirty_path    = Path(f"{prefix}-MFS-dirty.fits")
psf_path      = Path(f"{prefix}-MFS-psf.fits")

print(f"\n  Restored FITS: {restored_path} (exists={restored_path.exists()})")
print(f"  Dirty FITS:    {dirty_path}    (exists={dirty_path.exists()})")
print(f"  PSF FITS:      {psf_path}      (exists={psf_path.exists()})")

if not restored_path.exists():
    print("ERROR: restored image not found. Available FITS:")
    for f in img_dir.glob("*.fits"):
        print(f"  {f}")
    sys.exit(1)

# ── Plot ──────────────────────────────────────────────────────────────────────
apply_pipeline_style()

# Sky model source positions (Tile 0)
SOURCES = [
    ("S0", 344.9797, 17.5769, 0.349),
    ("S1", 343.9332, 16.9334, 4.011),
    ("S2", 345.2440, 17.0082, 1.442),
    ("S3", 344.7405, 15.0343, 2.853),
    ("S4", 342.8566, 16.0012, 0.542),
]

def plot_fits(fits_path: Path, title: str, ax, *, vmin=None, vmax=None, annotate_sources=True):
    hdul = fits.open(fits_path)
    # WSClean produces (1,1,Ny,Nx) or (Ny,Nx)
    data = hdul[0].data
    while data.ndim > 2:
        data = data[0]
    hdr = hdul[0].header

    # Build WCS (ignore extra axes)
    wcs_full = WCS(hdr)
    wcs2d = wcs_full.celestial

    peak = float(np.nanmax(np.abs(data)))
    print(f"  {title}: peak={peak:.4f} Jy/beam, shape={data.shape}")

    # Robust color scale
    if vmax is None:
        vmax = float(np.nanpercentile(data, 99.5))
    if vmin is None:
        vmin = float(np.nanpercentile(data,  0.5))

    im = ax.imshow(data, origin="lower", cmap="inferno",
                   vmin=vmin, vmax=vmax,
                   aspect="auto")
    plt.colorbar(im, ax=ax, label="Jy/beam", fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")

    if annotate_sources:
        for name, ra, dec, flux in SOURCES:
            try:
                # Convert sky coords to pixel
                import astropy.units as u
                from astropy.coordinates import SkyCoord
                coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs")
                px, py = wcs2d.world_to_pixel(coord)
                if 0 <= px < data.shape[1] and 0 <= py < data.shape[0]:
                    ax.plot(px, py, "x", color="cyan", ms=10, mew=2)
                    ax.text(px+5, py+5, f"{name}\n{flux:.2f}Jy",
                            color="cyan", fontsize=7, ha="left")
            except Exception as e:
                print(f"  Warn: could not annotate {name}: {e}")
    hdul.close()
    return data

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Step 4: DSA-110 Tile 0 CLEAN Image (96 antennas, MFS 1311–1498 MHz)", fontsize=13)

plot_fits(psf_path,      "PSF",            axes[0], annotate_sources=False)
plot_fits(dirty_path,    "Dirty Image",    axes[1])
plot_fits(restored_path, "CLEAN Restored", axes[2])

plt.tight_layout()
out_png = OUT_DIR / "step4_clean_image.png"
plt.savefig(str(out_png), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_png}")
