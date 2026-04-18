"""
Step 4: single calibrated tile CLEAN image — multi-MS approach.

Generates each of the 16 subbands as a separate small MS (~125 MB each),
applies gain corruption + calibration per-subband, phase-shifts, then passes
all 16 MS files to WSClean as a multi-MS MFS run.

This avoids the large in-memory concatenation that caused OOM/timeout in
the single-concatenated-MS approach.

Total disk budget:
  16 × ~250 MB (MS with CORRECTED_DATA) = ~4 GB peak, freed after imaging.
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
import pyradiosky
from astropy.coordinates import Longitude, Latitude, SkyCoord
import astropy.units as au

from dsa110_continuum.simulation.harness import SimulationHarness
from dsa110_continuum.simulation.pipeline import SimulatedPipeline
from dsa110_continuum.simulation.plot_style import apply_pipeline_style

# ── Parameters ────────────────────────────────────────────────────────────────
OUT_DIR        = Path("/home/user/workspace/dsa110-continuum/pipeline_outputs/step4")
N_ANT          = 117
N_INT          = 24
N_SUBBANDS     = 16
AMP_SCATTER    = 0.05
PHASE_SCATTER  = 5.0        # degrees
CAL_FLUX_JY    = 10.0
_T_INT         = 12.884902
_OVRO_LON      = -118.2825
_TILE_T0       = ATime("2026-01-25T22:26:05", format="isot", scale="utc")
TILE_IDX       = 0

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Tile geometry ─────────────────────────────────────────────────────────────
_T_TILE_S = N_INT * _T_INT
t_start   = ATime(_TILE_T0.jd + TILE_IDX * _T_TILE_S / 86400.0, format="jd", scale="utc")
t_mid     = ATime(t_start.jd + _T_TILE_S / 2 / 86400.0, format="jd", scale="utc")
median_ra = float(t_mid.sidereal_time("apparent", longitude=_OVRO_LON * _u.deg).deg)
dec_deg   = 16.15

print(f"Tile {TILE_IDX}: start={t_start.isot}  median_ra={median_ra:.3f}°  dec={dec_deg}°", flush=True)

# ── Build harness + sky model ─────────────────────────────────────────────────
harness = SimulationHarness(
    n_antennas=N_ANT,
    n_integrations=N_INT,
    n_sky_sources=5,
    noise_jy=1.0,
    seed=42,
    use_real_positions=True,
)
harness.pointing_ra_deg  = median_ra
harness.pointing_dec_deg = dec_deg

sky = harness.make_sky_model(fov_deg=3.0)
print(f"Sky model: {sky.Ncomponents} sources", flush=True)
for k in range(sky.Ncomponents):
    print(f"  S{k}: RA={sky.ra[k].deg:.4f}°  Dec={sky.dec[k].deg:.4f}°  "
          f"I={float(sky.stokes[0,0,k].value):.3f} Jy", flush=True)

# ── Draw gain errors once, shared across all subbands ────────────────────────
rng   = np.random.default_rng(seed=100)
amp_g = rng.normal(1.0, AMP_SCATTER, N_ANT).astype(np.float32)
ph_g  = rng.normal(0.0, np.radians(PHASE_SCATTER), N_ANT).astype(np.float32)
g     = amp_g * np.exp(1j * ph_g)

def apply_gains(uv, g_vec):
    a1 = uv.ant_1_array
    a2 = uv.ant_2_array
    g1 = g_vec[a1][:, np.newaxis, np.newaxis]
    g2 = g_vec[a2][:, np.newaxis, np.newaxis]
    uv.data_array = (g1 * uv.data_array * np.conj(g2)).astype(np.complex64)

# ── Pipeline object (for phaseshift + calibrate) ──────────────────────────────
pipeline = SimulatedPipeline(
    work_dir=OUT_DIR,
    niter=10000,
    cell_arcsec=20.0,
    image_size=512,
    wsclean_mem_gb=4.0,
)

# ── Per-subband loop: generate → corrupt → write MS → calibrate ───────────────
ms_list = []

for sb in range(N_SUBBANDS):
    print(f"\n── Subband {sb:02d}/{N_SUBBANDS-1} ──────────────────────────────", flush=True)

    # 1. Generate tile visibilities for this SB
    uv = harness._build_uvdata(sb, t_start, sky, drift_scan=True)
    print(f"  tile: {uv.Nblts} blts × {uv.Nfreqs} ch", flush=True)

    # 2. Apply gain corruption
    apply_gains(uv, g)

    # 3. Write corrupted MS
    ms_raw = OUT_DIR / f"sb{sb:02d}_raw.ms"
    uv.write_ms(str(ms_raw))
    del uv
    print(f"  wrote {ms_raw.stat().st_size/1e6:.0f} MB", flush=True)

    # 4. Phase-shift to median meridian
    ms_phased = pipeline._phaseshift_to_median(
        ms_path=ms_raw,
        median_ra_deg=median_ra,
        dec_deg=dec_deg,
        work_dir=OUT_DIR,
    )
    shutil.rmtree(str(ms_raw), ignore_errors=True)
    print(f"  phased → {ms_phased.name}  ({ms_phased.stat().st_size/1e6:.0f} MB)", flush=True)

    # 5. Generate matching calibrator for this SB (1 integration, no noise)
    freq_hz = float(harness.subband_freqs(sb).mean())
    stokes  = np.zeros((4, 1, 1))
    stokes[0, 0, 0] = CAL_FLUX_JY
    sky_cal = pyradiosky.SkyModel(
        name=np.array(["SIM_CAL"]),
        ra=Longitude([median_ra], unit="deg"),
        dec=Latitude([dec_deg], unit="deg"),
        stokes=stokes * au.Jy,
        spectral_type="spectral_index",
        reference_frequency=np.array([freq_hz]) * au.Hz,
        spectral_index=np.array([0.0]),
        frame="icrs",
    )
    orig_noise = harness.noise_jy
    orig_nint  = harness.n_integrations
    harness.noise_jy      = 0.0
    harness.n_integrations = 1
    uv_cal = harness._build_uvdata(sb, t_start, sky_cal, drift_scan=False)
    harness.noise_jy      = orig_noise
    harness.n_integrations = orig_nint
    apply_gains(uv_cal, g)
    for key in uv_cal.phase_center_catalog:
        uv_cal.phase_center_catalog[key]["cat_name"] = "SIM_CAL"

    cal_uvh5 = OUT_DIR / f"sb{sb:02d}_cal.uvh5"
    uv_cal.write_uvh5(str(cal_uvh5))
    del uv_cal
    print(f"  calibrator: {cal_uvh5.stat().st_size/1e6:.1f} MB", flush=True)

    # 6. Apply calibration (writes CORRECTED_DATA into ms_phased in-place)
    pipeline._calibrate(
        target_ms=ms_phased,
        cal_uvh5=cal_uvh5,
        cal_flux_jy=CAL_FLUX_JY,
        work_dir=OUT_DIR,
    )
    cal_uvh5.unlink(missing_ok=True)
    ms_list.append(ms_phased)

    # Disk check
    stat = os.statvfs("/home/user/workspace")
    free_gb = stat.f_bavail * stat.f_frsize / 1e9
    print(f"  disk free: {free_gb:.1f} GB", flush=True)

print(f"\n✓ All {N_SUBBANDS} subbands ready: {[m.name for m in ms_list]}", flush=True)

# ── WSClean multi-MS MFS imaging ─────────────────────────────────────────────
print("\nRunning WSClean multi-MS MFS imaging…", flush=True)

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
    "-niter", "50000",
    "-mgain", "0.8",
    "-auto-threshold", "3.0",
    "-auto-mask", "5.0",
    "-pol", "I",
    "-data-column", "CORRECTED_DATA",
    "-make-psf",
    "-no-update-model-required",
    "-abs-mem", "4",
] + [str(m) for m in ms_list]

print("  cmd:", " ".join(cmd[:16]), "...", flush=True)
r = subprocess.run(cmd, capture_output=True, text=True)
print(f"  return code: {r.returncode}", flush=True)

if r.returncode != 0:
    print("STDERR:", r.stderr[-3000:])
    sys.exit(1)

for line in r.stdout.split("\n"):
    if any(k in line for k in ["beam", "Peak", "peak", "iter", "threshold",
                                "major", "Stopped", "Restored", "Maximum",
                                "Fitting", "Deconvolving", "Mask"]):
        print("  WSClean:", line.strip(), flush=True)

# ── Free per-SB MS files now that imaging is done ────────────────────────────
for ms in ms_list:
    shutil.rmtree(str(ms), ignore_errors=True)
print("  Freed per-SB MS files.", flush=True)

# ── Find output FITS ──────────────────────────────────────────────────────────
restored_path = Path(f"{prefix}-MFS-image.fits")
dirty_path    = Path(f"{prefix}-MFS-dirty.fits")
psf_path      = Path(f"{prefix}-MFS-psf.fits")

print(f"\n  Restored: {restored_path.exists()}  {restored_path}", flush=True)
print(f"  Dirty:    {dirty_path.exists()}  {dirty_path}", flush=True)
print(f"  PSF:      {psf_path.exists()}  {psf_path}", flush=True)

if not restored_path.exists():
    print("ERROR: restored image not found. FITS in wsclean_out:")
    for f in sorted(img_dir.glob("*.fits")):
        print(f"  {f.name}  {f.stat().st_size/1e6:.1f} MB")
    sys.exit(1)

# ── Image statistics ──────────────────────────────────────────────────────────
print("\n=== Image statistics ===")
for path, name in [(restored_path, "Restored"), (dirty_path, "Dirty")]:
    with fits.open(str(path)) as hdul:
        d = hdul[0].data.squeeze()
    rms  = float(np.nanstd(d))
    peak = float(np.nanmax(d))
    print(f"  {name}: peak={peak:.4f} Jy/beam  rms={rms:.4f} Jy/beam  SNR={peak/rms:.1f}")

# ── Produce 3-panel figure (PSF | Dirty | Restored) ──────────────────────────
print("\nGenerating output figure…", flush=True)

with plt.style.context(["science", "notebook"]):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"DSA-110 Step 4: Tile 0 — 117 antennas · MFS 1311–1498 MHz · Dec +16.15°",
        fontsize=12,
    )

    for ax, fpath, title, do_sources in [
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
        ax.set_xlabel("RA pixel")
        ax.set_ylabel("Dec pixel")

        if do_sources:
            for k in range(sky.Ncomponents):
                ra_s  = sky.ra[k].deg
                dec_s = sky.dec[k].deg
                flux  = float(sky.stokes[0, 0, k].value)
                try:
                    coord = SkyCoord(ra=ra_s * au.deg, dec=dec_s * au.deg, frame="icrs")
                    px, py = wcs2d.world_to_pixel(coord)
                    px, py = float(px), float(py)
                    if 0 <= px < data.shape[1] and 0 <= py < data.shape[0]:
                        ax.plot(px, py, "+", color="cyan", ms=12, mew=2)
                        ax.text(px + 4, py + 4, f"S{k} {flux:.2f}Jy",
                                color="cyan", fontsize=7)
                except Exception:
                    pass

    plt.tight_layout()
    out_png = OUT_DIR / "step4_tile0_clean.png"
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_png}", flush=True)

print("\nDone.", flush=True)
