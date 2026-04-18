"""Plot Step 4 result: PSF | Dirty | CLEAN restored, with source annotations."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as au

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dsa110_continuum.simulation.harness import SimulationHarness
from astropy.time import Time as ATime
import astropy.units as _u

OUT_DIR   = Path("/home/user/workspace/dsa110-continuum/pipeline_outputs/step4")
IMG_DIR   = OUT_DIR / "wsclean_out"
prefix    = str(IMG_DIR / "wsclean")

restored_path = Path(f"{prefix}-MFS-image.fits")
dirty_path    = Path(f"{prefix}-MFS-dirty.fits")
psf_path      = Path(f"{prefix}-MFS-psf.fits")

# Reconstruct sky model (same seed → identical sources)
_OVRO_LON = -118.2825
_TILE_T0  = ATime("2026-01-25T22:26:05", format="isot", scale="utc")
_T_TILE_S = 24 * 12.884902
t_start   = _TILE_T0
t_mid     = ATime(t_start.jd + _T_TILE_S / 2 / 86400.0, format="jd", scale="utc")
median_ra = float(t_mid.sidereal_time("apparent", longitude=_OVRO_LON * _u.deg).deg)
dec_deg   = 16.15

harness = SimulationHarness(
    n_antennas=117, n_integrations=24, n_sky_sources=5,
    noise_jy=1.0, seed=42, use_real_positions=True,
)
harness.pointing_ra_deg  = median_ra
harness.pointing_dec_deg = dec_deg
sky = harness.make_sky_model(fov_deg=3.0)

print(f"Sky model ({sky.Ncomponents} sources):")
for k in range(sky.Ncomponents):
    print(f"  S{k}: RA={sky.ra[k].deg:.4f}°  Dec={sky.dec[k].deg:.4f}°  "
          f"I={float(sky.stokes[0,0,k].value):.3f} Jy")

# Image statistics
print("\n=== Image statistics ===")
for path, name in [(restored_path, "Restored"), (dirty_path, "Dirty")]:
    with fits.open(str(path)) as hdul:
        d = hdul[0].data.copy()
    while d.ndim > 2:
        d = d[0]
    rms  = float(np.nanstd(d))
    peak = float(np.nanmax(d))
    print(f"  {name}: peak={peak:.4f} Jy/beam  rms={rms:.4f} Jy/beam  "
          f"SNR={peak/rms:.1f}  shape={d.shape}")

# Plot
with plt.style.context(["science", "notebook"]):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "DSA-110 Step 4 — Tile 0: 117 antennas · 8 subbands MFS "
        "(1311–1450 MHz) · Dec +16.15°",
        fontsize=12,
    )

    for ax, fpath, title, do_src in [
        (axes[0], psf_path,      "PSF",            False),
        (axes[1], dirty_path,    "Dirty Image",    True),
        (axes[2], restored_path, "CLEAN Restored", True),
    ]:
        with fits.open(str(fpath)) as hdul:
            data = hdul[0].data.copy()
            hdr  = hdul[0].header
        while data.ndim > 2:
            data = data[0]
        wcs2d = WCS(hdr, naxis=2)

        vmax = float(np.nanpercentile(data, 99.5))
        vmin = float(np.nanpercentile(data,  0.5))

        im = ax.imshow(data, origin="lower", cmap="inferno",
                       vmin=vmin, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, label="Jy/beam", fraction=0.046, pad=0.04)

        rms  = float(np.nanstd(data))
        peak = float(np.nanmax(data))
        ax.set_title(f"{title}\npeak={peak:.3f}  rms={rms:.3f} Jy/beam", fontsize=10)
        ax.set_xlabel("RA (pixel)")
        ax.set_ylabel("Dec (pixel)")

        if do_src:
            for k in range(sky.Ncomponents):
                try:
                    coord = SkyCoord(ra=sky.ra[k].deg * au.deg,
                                     dec=sky.dec[k].deg * au.deg, frame="icrs")
                    px, py = wcs2d.world_to_pixel(coord)
                    px, py = float(px), float(py)
                    flux = float(sky.stokes[0, 0, k].value)
                    in_img = (0 <= px < data.shape[1]) and (0 <= py < data.shape[0])
                    color = "cyan" if in_img else "gray"
                    ax.plot(px, py, "+", color=color, ms=14, mew=2, zorder=5)
                    ax.text(px + 5, py + 5, f"S{k}\n{flux:.2f}Jy",
                            color=color, fontsize=7, zorder=5,
                            bbox=dict(boxstyle="round,pad=0.2", fc="black",
                                      alpha=0.55, ec="none"))
                except Exception as e:
                    print(f"  warn: S{k} annotation failed: {e}")

    plt.tight_layout()
    out_png = OUT_DIR / "step4_tile0_clean.png"
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_png}")
