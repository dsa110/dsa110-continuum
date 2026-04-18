"""
Step 5: Image all 4 consecutive drift-scan tiles and produce a 2×2 panel figure.

Each tile is processed independently (sequential) to stay within the disk budget:
  generate 8 subbands → corrupt → phaseshift → calibrate → WSClean MFS → free MS

Final output: a 2×2 grid showing PSF | Dirty | CLEAN for each tile, plus a
summary 2×2 panel of just the four CLEAN restored images.
"""
from __future__ import annotations

import os
import sys
import shutil
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astropy.time import Time as ATime
import astropy.units as _u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as au
import pyradiosky
from astropy.coordinates import Longitude, Latitude

from dsa110_continuum.simulation.harness import SimulationHarness
from dsa110_continuum.simulation.pipeline import SimulatedPipeline
from dsa110_continuum.simulation.plot_style import apply_pipeline_style

# ── Parameters ────────────────────────────────────────────────────────────────
OUT_DIR        = Path("/home/user/workspace/dsa110-continuum/pipeline_outputs/step5")
N_ANT          = 117
N_INT          = 24
N_SUBBANDS     = 8          # every-other subband (SBs 0,2,4,6,8,10,12,14) — proven to fit
SB_INDICES     = list(range(0, 16, 2))   # [0,2,4,6,8,10,12,14]
AMP_SCATTER    = 0.05
PHASE_SCATTER  = 5.0
CAL_FLUX_JY    = 10.0
N_TILES        = 4
_T_INT         = 12.884902
_OVRO_LON      = -118.2825
_TILE_T0       = ATime("2026-01-25T22:26:05", format="isot", scale="utc")
_T_TILE_S      = N_INT * _T_INT

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Per-tile geometry ─────────────────────────────────────────────────────────
def tile_params(idx: int):
    t_start = ATime(_TILE_T0.jd + idx * _T_TILE_S / 86400.0, format="jd", scale="utc")
    t_mid   = ATime(t_start.jd + _T_TILE_S / 2 / 86400.0, format="jd", scale="utc")
    median_ra = float(t_mid.sidereal_time("apparent", longitude=_OVRO_LON * _u.deg).deg)
    return t_start, median_ra

DEC_DEG = 16.15

print(f"Step 5: processing {N_TILES} tiles", flush=True)
for i in range(N_TILES):
    t, ra = tile_params(i)
    print(f"  Tile {i}: start={t.isot}  median_ra={ra:.3f}°", flush=True)

# ── Shared gain seed — same gains across all tiles (coherent corruption) ──────
rng_gains = np.random.default_rng(seed=100)
amp_g = rng_gains.normal(1.0, AMP_SCATTER, N_ANT).astype(np.float32)
ph_g  = rng_gains.normal(0.0, np.radians(PHASE_SCATTER), N_ANT).astype(np.float32)
G     = amp_g * np.exp(1j * ph_g)

def apply_gains(uv, g_vec):
    a1 = uv.ant_1_array; a2 = uv.ant_2_array
    g1 = g_vec[a1][:, np.newaxis, np.newaxis]
    g2 = g_vec[a2][:, np.newaxis, np.newaxis]
    uv.data_array = (g1 * uv.data_array * np.conj(g2)).astype(np.complex64)

# ── Process each tile ─────────────────────────────────────────────────────────
tile_results = []   # list of dicts: {idx, median_ra, sky, restored_path, dirty_path, psf_path}

for tile_idx in range(N_TILES):
    print(f"\n{'='*60}", flush=True)
    print(f"TILE {tile_idx}", flush=True)
    print(f"{'='*60}", flush=True)

    t_start, median_ra = tile_params(tile_idx)
    tile_dir = OUT_DIR / f"tile{tile_idx:02d}"
    tile_dir.mkdir(exist_ok=True)

    # Build harness for this tile
    harness = SimulationHarness(
        n_antennas=N_ANT,
        n_integrations=N_INT,
        n_sky_sources=5,
        noise_jy=1.0,
        seed=42 + tile_idx,   # different sky per tile
        use_real_positions=True,
    )
    harness.pointing_ra_deg  = median_ra
    harness.pointing_dec_deg = DEC_DEG
    sky = harness.make_sky_model(fov_deg=3.0)

    print(f"  Sky ({sky.Ncomponents} sources):", flush=True)
    for k in range(sky.Ncomponents):
        print(f"    S{k}: RA={sky.ra[k].deg:.3f}°  Dec={sky.dec[k].deg:.3f}°  "
              f"I={float(sky.stokes[0,0,k].value):.3f} Jy", flush=True)

    pipeline = SimulatedPipeline(
        work_dir=tile_dir,
        niter=50000,
        cell_arcsec=20.0,
        image_size=512,
        wsclean_mem_gb=4.0,
    )

    ms_list = []
    for sb in SB_INDICES:
        print(f"  SB {sb:02d}…", end=" ", flush=True)

        # Generate tile visibilities
        uv = harness._build_uvdata(sb, t_start, sky, drift_scan=True)
        apply_gains(uv, G)

        # Write + phaseshift
        ms_raw = tile_dir / f"sb{sb:02d}_raw.ms"
        uv.write_ms(str(ms_raw))
        del uv

        ms_phased = pipeline._phaseshift_to_median(
            ms_path=ms_raw,
            median_ra_deg=median_ra,
            dec_deg=DEC_DEG,
            work_dir=tile_dir,
        )
        shutil.rmtree(str(ms_raw), ignore_errors=True)

        # Generate calibrator for this SB
        freq_hz = float(harness.subband_freqs(sb).mean())
        stokes  = np.zeros((4, 1, 1))
        stokes[0, 0, 0] = CAL_FLUX_JY
        sky_cal = pyradiosky.SkyModel(
            name=np.array(["SIM_CAL"]),
            ra=Longitude([median_ra], unit="deg"),
            dec=Latitude([DEC_DEG], unit="deg"),
            stokes=stokes * au.Jy,
            spectral_type="spectral_index",
            reference_frequency=np.array([freq_hz]) * au.Hz,
            spectral_index=np.array([0.0]),
            frame="icrs",
        )
        orig_noise = harness.noise_jy; orig_nint = harness.n_integrations
        harness.noise_jy = 0.0; harness.n_integrations = 1
        uv_cal = harness._build_uvdata(sb, t_start, sky_cal, drift_scan=False)
        harness.noise_jy = orig_noise; harness.n_integrations = orig_nint
        apply_gains(uv_cal, G)
        for key in uv_cal.phase_center_catalog:
            uv_cal.phase_center_catalog[key]["cat_name"] = "SIM_CAL"

        cal_uvh5 = tile_dir / f"sb{sb:02d}_cal.uvh5"
        uv_cal.write_uvh5(str(cal_uvh5))
        del uv_cal

        # Calibrate
        pipeline._calibrate(
            target_ms=ms_phased,
            cal_uvh5=cal_uvh5,
            cal_flux_jy=CAL_FLUX_JY,
            work_dir=tile_dir,
        )
        cal_uvh5.unlink(missing_ok=True)
        ms_list.append(ms_phased)

        stat = os.statvfs("/home/user/workspace")
        free_gb = stat.f_bavail * stat.f_frsize / 1e9
        print(f"free={free_gb:.1f}GB", flush=True)

    # WSClean
    print(f"\n  Running WSClean on {len(ms_list)} MS files…", flush=True)
    img_dir = tile_dir / "wsclean_out"
    img_dir.mkdir(exist_ok=True)
    prefix = str(img_dir / "wsclean")

    cmd = [
        "wsclean",
        "-name", prefix,
        "-size", "512", "512",
        "-scale", "20asec",
        "-weight", "briggs", "0.0",
        "-join-channels", "-channels-out", "2",
        "-niter", "50000",
        "-mgain", "0.8",
        "-auto-threshold", "3.0",
        "-auto-mask", "5.0",
        "-pol", "I",
        "-data-column", "CORRECTED_DATA",
        "-make-psf",
        "-no-update-model-required",
        "-no-reorder",
        "-abs-mem", "4",
    ] + [str(m) for m in ms_list]

    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f"  WSClean exit: {r.returncode}", flush=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:])
        sys.exit(1)

    for line in r.stdout.split("\n"):
        if any(k in line for k in ["Fitting beam", "major iter", "Deconvolving",
                                    "Stopped", "Restored", "Rendering", "peak"]):
            print(f"    {line.strip()}", flush=True)

    # Free MS files
    for ms in ms_list:
        shutil.rmtree(str(ms), ignore_errors=True)

    restored_path = Path(f"{prefix}-MFS-image.fits")
    dirty_path    = Path(f"{prefix}-MFS-dirty.fits")
    psf_path      = Path(f"{prefix}-MFS-psf.fits")

    if not restored_path.exists():
        print(f"  ERROR: {restored_path} missing")
        sys.exit(1)

    with fits.open(str(restored_path)) as h:
        d = h[0].data.copy()
    while d.ndim > 2: d = d[0]
    rms = float(np.nanstd(d)); peak = float(np.nanmax(d))
    print(f"  Restored: peak={peak:.3f}  rms={rms:.4f}  SNR={peak/rms:.0f}", flush=True)

    tile_results.append(dict(
        idx=tile_idx, median_ra=median_ra, sky=sky,
        restored_path=restored_path, dirty_path=dirty_path, psf_path=psf_path,
    ))

print(f"\n✓ All {N_TILES} tiles imaged.", flush=True)

# ── 2×2 summary figure: CLEAN restored for each tile ─────────────────────────
print("\nGenerating 2×2 summary figure…", flush=True)

with plt.style.context(["science", "notebook"]):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        "DSA-110 Step 5 — 4 Consecutive Drift-Scan Tiles\n"
        "CLEAN Restored · 117 antennas · 8-subband MFS · Dec +16.15°",
        fontsize=13,
    )

    for res in tile_results:
        row, col = divmod(res["idx"], 2)
        ax = axes[row][col]
        sky = res["sky"]

        with fits.open(str(res["restored_path"])) as h:
            data = h[0].data.copy()
            hdr  = h[0].header
        while data.ndim > 2: data = data[0]
        wcs2d = WCS(hdr, naxis=2)

        rms  = float(np.nanstd(data))
        peak = float(np.nanmax(data))

        vmax = float(np.nanpercentile(data, 99.5))
        vmin = float(np.nanpercentile(data,  0.5))
        im = ax.imshow(data, origin="lower", cmap="inferno",
                       vmin=vmin, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, label="Jy/beam", fraction=0.046, pad=0.04)

        ax.set_title(
            f"Tile {res['idx']}  RA={res['median_ra']:.2f}°\n"
            f"peak={peak:.3f}  rms={rms:.4f} Jy/beam  SNR={peak/rms:.0f}",
            fontsize=9,
        )
        ax.set_xlabel("RA (pixel)"); ax.set_ylabel("Dec (pixel)")

        # Annotate sources
        for k in range(sky.Ncomponents):
            try:
                coord = SkyCoord(ra=sky.ra[k].deg * au.deg,
                                 dec=sky.dec[k].deg * au.deg, frame="icrs")
                px, py = wcs2d.world_to_pixel(coord)
                px, py = float(px), float(py)
                flux = float(sky.stokes[0, 0, k].value)
                in_img = (0 <= px < data.shape[1]) and (0 <= py < data.shape[0])
                color = "cyan" if in_img else "gray"
                ax.plot(px, py, "+", color=color, ms=12, mew=2, zorder=5)
                ax.text(px + 4, py + 4, f"S{k}\n{flux:.2f}Jy",
                        color=color, fontsize=6.5, zorder=5,
                        bbox=dict(boxstyle="round,pad=0.2", fc="black",
                                  alpha=0.55, ec="none"))
            except Exception:
                pass

    plt.tight_layout()
    out_summary = OUT_DIR / "step5_four_tiles.png"
    fig.savefig(str(out_summary), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_summary}", flush=True)

print("\nDone.", flush=True)
