"""
Step 6: Final mosaic of 4 drift-scan tiles.

Strategy: noise-weighted linear mosaic on a common RA/Dec grid.
  1. Reproject each tile onto a shared WCS output grid (spanning all 4 tiles).
  2. Weight each reprojected tile by 1/rms^2 (inverse-variance weighting).
  3. Sum the weighted images and divide by the sum of weights to get the coadd.
  4. Produce a publication-quality figure showing the mosaic with source annotations.
  5. Save the mosaic FITS and PNG.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as au
from astropy.coordinates import SkyCoord
from astropy.time import Time as ATime
import astropy.units as _u
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs

from dsa110_continuum.simulation.harness import SimulationHarness

# ── Parameters ────────────────────────────────────────────────────────────────
STEP5_DIR = Path("/home/user/workspace/dsa110-continuum/pipeline_outputs/step5")
OUT_DIR   = Path("/home/user/workspace/dsa110-continuum/pipeline_outputs/step6")
DOCS_DIR  = Path("/home/user/workspace/dsa110-continuum/docs/images")
N_TILES   = 4
N_ANT     = 117
N_INT     = 24
_T_INT    = 12.884902
_OVRO_LON = -118.2825
_TILE_T0  = ATime("2026-01-25T22:26:05", format="isot", scale="utc")
_T_TILE_S = N_INT * _T_INT
DEC_DEG   = 16.15

OUT_DIR.mkdir(parents=True, exist_ok=True)

def tile_params(idx: int) -> float:
    t_start = ATime(_TILE_T0.jd + idx * _T_TILE_S / 86400.0, format="jd", scale="utc")
    t_mid   = ATime(t_start.jd + _T_TILE_S / 2 / 86400.0, format="jd", scale="utc")
    return float(t_mid.sidereal_time("apparent", longitude=_OVRO_LON * _u.deg).deg)

# ── Load tile FITS headers & data ─────────────────────────────────────────────
print("Loading tile images…", flush=True)
tile_hdus   = []   # list of (HDU, rms)
tile_skies  = []   # sky models for annotation

for tile_idx in range(N_TILES):
    path = STEP5_DIR / f"tile{tile_idx:02d}" / "wsclean_out" / "wsclean-MFS-image.fits"
    with fits.open(str(path)) as h:
        hdr  = h[0].header.copy()
        data = h[0].data.copy()
    while data.ndim > 2: data = data[0]
    rms  = float(np.nanstd(data))
    peak = float(np.nanmax(data))
    print(f"  Tile {tile_idx}: peak={peak:.3f}  rms={rms:.4f}  SNR={peak/rms:.0f}", flush=True)

    # Rebuild as 2D FITS HDU (needed for reproject)
    hdr2 = hdr.copy()
    # Strip non-celestial axes so WCS is strictly 2D
    for key in ["NAXIS3", "NAXIS4", "CTYPE3", "CRVAL3", "CDELT3", "CRPIX3", "CUNIT3",
                "CTYPE4", "CRVAL4", "CDELT4", "CRPIX4", "CUNIT4"]:
        hdr2.remove(key, ignore_missing=True)
    hdr2["NAXIS"] = 2
    hdr2["NAXIS1"] = data.shape[1]
    hdr2["NAXIS2"] = data.shape[0]
    hdu2d = fits.PrimaryHDU(data=data.astype(np.float32), header=hdr2)
    tile_hdus.append((hdu2d, rms))

    # Sky model for annotation
    median_ra = tile_params(tile_idx)
    harness = SimulationHarness(n_antennas=N_ANT, n_integrations=N_INT, n_sky_sources=5,
                                 noise_jy=1.0, seed=42 + tile_idx, use_real_positions=True)
    harness.pointing_ra_deg  = median_ra
    harness.pointing_dec_deg = DEC_DEG
    sky = harness.make_sky_model(fov_deg=3.0)
    tile_skies.append((tile_idx, median_ra, sky))

# ── Build optimal output WCS spanning all tiles ───────────────────────────────
print("\nFinding optimal mosaic WCS…", flush=True)
wcs_out, shape_out = find_optimal_celestial_wcs(
    [hdu for hdu, _ in tile_hdus],
    resolution=20 * au.arcsec,
    projection="TAN",
)
print(f"  Output grid: {shape_out[0]}×{shape_out[1]} pixels", flush=True)
print(f"  RA range:  {wcs_out.pixel_to_world(0, shape_out[0]//2).ra.deg:.3f}°  →  "
      f"{wcs_out.pixel_to_world(shape_out[1]-1, shape_out[0]//2).ra.deg:.3f}°", flush=True)
print(f"  Dec range: {wcs_out.pixel_to_world(shape_out[1]//2, 0).dec.deg:.3f}°  →  "
      f"{wcs_out.pixel_to_world(shape_out[1]//2, shape_out[0]-1).dec.deg:.3f}°", flush=True)

# ── Reproject and co-add with inverse-variance weights ────────────────────────
print("\nReprojecting tiles…", flush=True)
sum_weighted = np.zeros(shape_out, dtype=np.float64)
sum_weights  = np.zeros(shape_out, dtype=np.float64)

for tile_idx, (hdu, rms) in enumerate(tile_hdus):
    print(f"  Tile {tile_idx}…", end=" ", flush=True)
    reproj, footprint = reproject_interp(hdu, wcs_out, shape_out=shape_out)
    weight = footprint / (rms ** 2)   # 1/sigma^2 where data is valid
    reproj = np.nan_to_num(reproj, nan=0.0)
    sum_weighted += reproj * weight
    sum_weights  += weight
    print("done", flush=True)

# Avoid division by zero
valid = sum_weights > 0
mosaic = np.where(valid, sum_weighted / sum_weights, np.nan)

rms_mosaic  = float(np.nanstd(mosaic))
peak_mosaic = float(np.nanmax(mosaic))
print(f"\nMosaic: peak={peak_mosaic:.3f}  rms={rms_mosaic:.4f}  SNR={peak_mosaic/rms_mosaic:.0f}", flush=True)

# ── Save mosaic FITS ──────────────────────────────────────────────────────────
mosaic_fits = OUT_DIR / "step6_mosaic.fits"
mosaic_hdr  = wcs_out.to_header()
mosaic_hdr["BUNIT"]   = "Jy/beam"
mosaic_hdr["OBJECT"]  = "DSA110_SIM_MOSAIC"
mosaic_hdr["HISTORY"] = "Inverse-variance mosaic of 4 drift-scan tiles (Step 6)"
fits.writeto(str(mosaic_fits), mosaic.astype(np.float32), mosaic_hdr, overwrite=True)
print(f"Saved FITS → {mosaic_fits}", flush=True)

# ── Coverage map (number of tiles contributing per pixel) ────────────────────
coverage = np.zeros(shape_out, dtype=np.int16)
for tile_idx, (hdu, rms) in enumerate(tile_hdus):
    reproj, footprint = reproject_interp(hdu, wcs_out, shape_out=shape_out)
    coverage += (footprint > 0.5).astype(np.int16)

# ── Publication figure ────────────────────────────────────────────────────────
print("\nGenerating mosaic figure…", flush=True)

with plt.style.context(["science", "notebook"]):
    fig = plt.figure(figsize=(16, 7))
    gs  = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.35)

    # --- Left: mosaic image ---
    ax_mosaic = fig.add_subplot(gs[0])
    vmax = float(np.nanpercentile(mosaic, 99.8))
    vmin = float(np.nanpercentile(mosaic,  0.2))
    im = ax_mosaic.imshow(mosaic, origin="lower", cmap="inferno",
                           vmin=vmin, vmax=vmax, aspect="auto")
    cb = plt.colorbar(im, ax=ax_mosaic, label="Jy/beam", fraction=0.03, pad=0.02)
    cb.ax.tick_params(labelsize=8)

    ax_mosaic.set_title(
        "DSA-110 Step 6 — Inverse-Variance Mosaic\n"
        f"4 drift-scan tiles · 117 antennas · 8-subband MFS · Dec +16.15°\n"
        f"peak={peak_mosaic:.3f} Jy/beam  rms={rms_mosaic:.4f} Jy/beam  SNR={peak_mosaic/rms_mosaic:.0f}",
        fontsize=10,
    )
    ax_mosaic.set_xlabel("RA (pixel)"); ax_mosaic.set_ylabel("Dec (pixel)")

    # Tile boundary lines (vertical lines at each tile edge)
    tile_ra_centres = [tile_params(i) for i in range(N_TILES)]
    for ra_c in tile_ra_centres:
        coord_l = SkyCoord(ra=(ra_c + 512*0.005556/2) * au.deg, dec=DEC_DEG * au.deg, frame="icrs")
        coord_r = SkyCoord(ra=(ra_c - 512*0.005556/2) * au.deg, dec=DEC_DEG * au.deg, frame="icrs")
        for coord, ls in [(coord_l, "--"), (coord_r, ":")]:
            try:
                px, py = wcs_out.world_to_pixel(coord)
                ax_mosaic.axvline(float(px), color="white", lw=0.6, ls=ls, alpha=0.4)
            except Exception:
                pass

    # Annotate all sources from all tiles
    already_annotated = set()
    for tile_idx, median_ra, sky in tile_skies:
        for k in range(sky.Ncomponents):
            try:
                coord = SkyCoord(ra=sky.ra[k].deg * au.deg,
                                 dec=sky.dec[k].deg * au.deg, frame="icrs")
                px, py = wcs_out.world_to_pixel(coord)
                px, py = float(px), float(py)
                if not (0 <= px < shape_out[1] and 0 <= py < shape_out[0]):
                    continue
                flux = float(sky.stokes[0, 0, k].value)
                label_key = (round(sky.ra[k].deg, 3), round(sky.dec[k].deg, 3))
                if label_key in already_annotated:
                    continue
                already_annotated.add(label_key)
                ax_mosaic.plot(px, py, "+", color="cyan", ms=12, mew=1.5, zorder=5)
                # Edge-aware label offset
                dx = 6 if px < shape_out[1] - 80 else -75
                dy = 6 if py < shape_out[0] - 55 else -25
                ax_mosaic.text(px + dx, py + dy,
                               f"T{tile_idx}·S{k}\n{flux:.2f} Jy",
                               color="cyan", fontsize=6.5, zorder=6,
                               ha="left" if dx > 0 else "right",
                               bbox=dict(boxstyle="round,pad=0.2", fc="black",
                                         alpha=0.6, ec="none"),
                               clip_on=True)
            except Exception:
                pass

    # --- Right: coverage map ---
    ax_cov = fig.add_subplot(gs[1])
    cov_im = ax_cov.imshow(coverage, origin="lower", cmap="Blues",
                            vmin=0, vmax=N_TILES, aspect="auto")
    cb2 = plt.colorbar(cov_im, ax=ax_cov, label="N tiles", fraction=0.06, pad=0.04,
                        ticks=list(range(N_TILES + 1)))
    cb2.ax.tick_params(labelsize=8)
    ax_cov.set_title("Tile Coverage\n(# tiles per pixel)", fontsize=9)
    ax_cov.set_xlabel("RA (pixel)"); ax_cov.set_ylabel("Dec (pixel)")

    out_png = OUT_DIR / "step6_mosaic.png"
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PNG → {out_png}", flush=True)

# Copy figure to docs/images for the reference site
import shutil
shutil.copy(str(out_png), str(DOCS_DIR / "step6_mosaic.png"))
print(f"Copied to docs/images/", flush=True)

print("\nDone.", flush=True)
