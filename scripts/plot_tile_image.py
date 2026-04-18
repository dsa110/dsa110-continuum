"""Step 4: single calibrated tile dirty + CLEAN image.

Uses all 16 DSA-110 subbands (1311–1498 MHz) for multi-frequency synthesis
with the full 96-antenna array.  All intermediate UVData objects are kept
in memory (never written to disk as UVH5) to avoid filling the 22 GB sandbox
disk.  Only the final concatenated MS and the CLEAN output are written to disk.

Pipeline:
  1. Generate drift-scan tile visibilities (96 ants, 24 integrations, 16 SBs)
     → in-memory UVData objects, then concatenated and written as one MS
  2. Corrupt with per-antenna gain errors
  3. Generate matching calibrator for each SB (in-memory); run Jacobi gain
     solve + apply
  4. Phase-shift all 24 per-integration fields to the median meridian RA
  5. Run WSClean with MFS (-join-channels) and CLEAN deconvolution
  6. Display the restored image with known source positions overlaid
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astropy.time import Time as ATime
import astropy.units as _u
from astropy.wcs import WCS
from astropy.io import fits

from dsa110_continuum.simulation.harness import SimulationHarness
from dsa110_continuum.simulation.pipeline import SimulatedPipeline
from dsa110_continuum.simulation.plot_style import apply_pipeline_style
import pyuvdata
import shutil

OUT_DIR = Path("/home/user/workspace/dsa110-continuum/pipeline_outputs/step4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
N_ANT          = 117      # all 117 allocated DSA-110 station slots (correct geometry).
                          # Using only first-96 rows misses all outriggers; using 117
                          # gives the correct T-array with max baselines ~2.2 km.
                          # See docs/GROUND_TRUTH.md §1.4 for rationale.
N_INT          = 24
N_SUBBANDS     = 16       # all 16 DSA-110 subbands → 1311–1498 MHz MFS
AMP_SCATTER    = 0.05
PHASE_SCATTER  = 5.0
CAL_FLUX_JY    = 10.0
_T_INT         = 12.884902
_OVRO_LON      = -118.2825
_TILE_T0       = ATime("2026-01-25T22:26:05", format="isot", scale="utc")
_T_TILE_S      = N_INT * _T_INT

def tile_start(idx: int) -> ATime:
    return ATime(_TILE_T0.jd + idx * _T_TILE_S / 86400.0, format="jd", scale="utc")

def tile_median_ra(idx: int) -> float:
    t_mid = ATime(tile_start(idx).jd + _T_TILE_S / 2 / 86400.0, format="jd", scale="utc")
    return float(t_mid.sidereal_time("apparent", longitude=_OVRO_LON * _u.deg).deg)

TILE_IDX  = 0
t_start   = tile_start(TILE_IDX)
median_ra = tile_median_ra(TILE_IDX)
dec_deg   = 16.15

print(f"Tile {TILE_IDX}: start={t_start.isot}  median_ra={median_ra:.3f}°  dec={dec_deg}°")

# ── 1. Build harness ─────────────────────────────────────────────────────────
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
print(f"Sky model: {sky.Ncomponents} sources")
for k in range(sky.Ncomponents):
    print(f"  S{k}: RA={sky.ra[k].deg:.4f}°  Dec={sky.dec[k].deg:.4f}°  "
          f"I={float(sky.stokes[0,0,k].value):.3f} Jy")

# ── 2. Generate drift-scan tile in memory (all 16 subbands) ─────────────────
print(f"\nGenerating drift-scan tile in memory ({N_SUBBANDS} subbands, {N_ANT} antennas)…")

uvs_tile = []
for sb in range(N_SUBBANDS):
    uv = harness._build_uvdata(sb, t_start, sky, drift_scan=True)
    uvs_tile.append(uv)
    print(f"  SB {sb:02d}: {uv.Nblts} blts × {uv.Nfreqs} ch", flush=True)

print(f"  Concatenating {N_SUBBANDS} subbands in memory…")
uv_all = uvs_tile[0]
for uv_extra in uvs_tile[1:]:
    uv_all.fast_concat(uv_extra, axis="freq", inplace=True)
del uvs_tile
print(f"  Concatenated: {uv_all.Nblts} blts × {uv_all.Nfreqs} ch")

# ── 3. Apply gain corruption in memory ───────────────────────────────────────
print("Applying gain corruption…")
# Apply in-memory corruption per-baseline
import numpy as np

rng = np.random.default_rng(seed=100)
n_ant = N_ANT
# Draw per-antenna gain errors
amp_g  = rng.normal(1.0, AMP_SCATTER, n_ant).astype(np.float32)
phase_g = rng.normal(0.0, np.radians(PHASE_SCATTER), n_ant).astype(np.float32)
g = amp_g * np.exp(1j * phase_g)  # (n_ant,)

# Apply baseline gains: V_corrupted = g[a1] * V * conj(g[a2])
# data_array stores conj(vis) per harness convention, so same formula applies
a1 = uv_all.ant_1_array
a2 = uv_all.ant_2_array
g1 = g[a1][:, np.newaxis, np.newaxis]   # (nblts, 1, 1)
g2 = g[a2][:, np.newaxis, np.newaxis]
uv_all.data_array = (g1 * uv_all.data_array * np.conj(g2)).astype(np.complex64)
print(f"  Applied per-antenna gains (amp scatter={AMP_SCATTER}, phase scatter={PHASE_SCATTER}°)")

# ── 4. Write corrupted tile MS ────────────────────────────────────────────────
ms_path = OUT_DIR / "tile00_corrupted.ms"
print(f"Writing corrupted MS → {ms_path}")
uv_all.write_ms(str(ms_path))
print(f"  MS written ({ms_path.stat().st_size/1e6:.0f} MB)")
del uv_all

# ── 5. Generate calibrators in memory, per-SB ─────────────────────────────────
print("Generating calibrators in memory…")
import pyradiosky
from astropy.coordinates import Longitude, Latitude
from astropy import units as au_units

cal_uvs = []
for sb in range(N_SUBBANDS):
    freq_hz = float(harness.subband_freqs(sb).mean())
    stokes = np.zeros((4, 1, 1), dtype=float)
    stokes[0, 0, 0] = CAL_FLUX_JY
    sky_cal = pyradiosky.SkyModel(
        name=np.array(["SIM_CAL"]),
        ra=Longitude([median_ra], unit="deg"),
        dec=Latitude([dec_deg], unit="deg"),
        stokes=stokes * au_units.Jy,
        spectral_type="spectral_index",
        reference_frequency=np.array([freq_hz]) * au_units.Hz,
        spectral_index=np.array([0.0]),
        frame="icrs",
    )

    # Temporarily disable noise and use 1 integration for calibrator (saves ~1.2 GB disk)
    orig_noise  = harness.noise_jy
    orig_n_src  = harness.n_sky_sources
    orig_n_int  = harness.n_integrations
    harness.noise_jy      = 0.0
    harness.n_sky_sources = 0
    harness.n_integrations = 1
    uv_cal = harness._build_uvdata(sb, t_start, sky_cal, drift_scan=False)
    harness.noise_jy      = orig_noise
    harness.n_sky_sources = orig_n_src
    harness.n_integrations = orig_n_int

    # Apply same gain errors to calibrator
    a1_c = uv_cal.ant_1_array
    a2_c = uv_cal.ant_2_array
    g1_c = g[a1_c][:, np.newaxis, np.newaxis]
    g2_c = g[a2_c][:, np.newaxis, np.newaxis]
    uv_cal.data_array = (g1_c * uv_cal.data_array * np.conj(g2_c)).astype(np.complex64)

    # Harmonise phase_center_catalog cat_name for concatenation
    for key in uv_cal.phase_center_catalog:
        uv_cal.phase_center_catalog[key]["cat_name"] = "SIM_CAL"

    cal_uvs.append(uv_cal)
    print(f"  Cal SB {sb:02d}: {uv_cal.Nblts} blts × {uv_cal.Nfreqs} ch", flush=True)

print("  Concatenating calibrator subbands…")
cal_combined = cal_uvs[0]
if len(cal_uvs) > 1:
    cal_combined.fast_concat(cal_uvs[1:], axis="freq", inplace=True)
del cal_uvs

# Write combined calibrator UVH5 (temporary, small relative to the MS)
cal_uvh5_combined = OUT_DIR / "sim_cal_combined.uvh5"
cal_combined.write_uvh5(str(cal_uvh5_combined))
print(f"  Combined calibrator → {cal_uvh5_combined}  ({cal_combined.Nfreqs} channels, "
      f"{cal_uvh5_combined.stat().st_size/1e6:.0f} MB)")
del cal_combined

# ── 6. Phaseshift + calibrate ─────────────────────────────────────────────────
pipeline = SimulatedPipeline(
    work_dir=OUT_DIR,
    niter=10000,
    cell_arcsec=20.0,
    image_size=512,
    wsclean_mem_gb=4.0,
)

print(f"\nPhase-shifting to median RA {median_ra:.3f}°…")
ms_phased = pipeline._phaseshift_to_median(
    ms_path=ms_path,
    median_ra_deg=median_ra,
    dec_deg=dec_deg,
    work_dir=OUT_DIR,
)
print(f"  → {ms_phased}")

# Free the raw MS AFTER phaseshift has written the meridian MS
print(f"  Removing raw MS {ms_path} to free disk…")
shutil.rmtree(str(ms_path), ignore_errors=True)
print(f"  Freed. Disk after rmtree:")
import os
os.system("df -h /home/user/workspace")

print("Running Jacobi gain calibration…")
pipeline._calibrate(
    target_ms=ms_phased,
    cal_uvh5=cal_uvh5_combined,
    cal_flux_jy=CAL_FLUX_JY,
    work_dir=OUT_DIR,
)
print("  Done")

cal_uvh5_combined.unlink(missing_ok=True)

# ── 7. WSClean MFS imaging ────────────────────────────────────────────────────
print("\nRunning WSClean MFS imaging…")
import subprocess

img_dir = OUT_DIR / "wsclean_out"
img_dir.mkdir(exist_ok=True)
prefix  = str(img_dir / "wsclean")

cmd = [
    "wsclean",
    "-name", prefix,
    "-size", "512", "512",
    "-scale", "20asec",
    "-weight", "briggs", "0.0",
    "-join-channels",        # MFS: combine all channels for deconvolution
    "-channels-out", "4",   # 4 sub-band images + MFS combination
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
    print("STDERR:", r.stderr[-2000:])
    sys.exit(1)

for line in r.stdout.split('\n'):
    if any(k in line for k in ['beam', 'peak', 'Peak', 'iter', 'threshold', 'DONE',
                                'major', 'Stopped', 'mask', 'Restored', 'Maximum',
                                'Fitting', 'Theoretic']):
        print("  WSClean:", line.strip())

# ── 8. Find output FITS ───────────────────────────────────────────────────────
# With -join-channels and -channels-out 4, WSClean produces:
#   PREFIX-MFS-image.fits   (MFS restored)
#   PREFIX-MFS-dirty.fits   (MFS dirty)
restored_path = Path(f"{prefix}-MFS-image.fits")
dirty_path    = Path(f"{prefix}-MFS-dirty.fits")
psf_path      = Path(f"{prefix}-MFS-psf.fits")

print(f"\n  Restored FITS: {restored_path} (exists={restored_path.exists()})")
print(f"  Dirty FITS:    {dirty_path} (exists={dirty_path.exists()})")

# ── 9. Plot ───────────────────────────────────────────────────────────────────
apply_pipeline_style()

for path, label, tag in [
    (restored_path, "CLEAN restored", "restored"),
    (dirty_path,    "Dirty",          "dirty"),
]:
    if not path.exists():
        print(f"  SKIP: {path} not found")
        continue

    with fits.open(str(path)) as hdul:
        hdr  = hdul[0].header
        data = hdul[0].data.squeeze()

    wcs = WCS(hdr, naxis=2)
    rms  = float(np.nanstd(data))
    peak = float(np.nanmax(data))

    fig = plt.figure(figsize=(8, 7))
    ax  = fig.add_subplot(1, 1, 1, projection=wcs)

    vmax = min(peak, 5 * rms)
    vmin = -rms
    im = ax.imshow(data, origin="lower", cmap="inferno",
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Flux density (Jy/beam)")

    for k in range(sky.Ncomponents):
        ra_src  = sky.ra[k].deg
        dec_src = sky.dec[k].deg
        flux    = float(sky.stokes[0, 0, k].value)
        px_arr, py_arr = wcs.all_world2pix(ra_src, dec_src, 0)
        px, py = float(px_arr), float(py_arr)
        in_img  = (0 <= px < data.shape[1]) and (0 <= py < data.shape[0])
        color   = "cyan" if in_img else "gray"
        ax.plot(px, py, "+", color=color, ms=14, mew=2, zorder=5)
        ax.annotate(
            f"S{k}  {flux:.2f} Jy",
            xy=(px, py), xytext=(6, 6), textcoords="offset points",
            color=color, fontsize=8, zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5, ec="none"),
        )

    ax.set_xlabel("RA (J2000)")
    ax.set_ylabel("Dec (J2000)")
    ax.set_title(
        f"Tile 0 — {label}  (MFS 16 SB, 1311–1498 MHz)\n"
        f"RA={median_ra:.3f}°  Dec={dec_deg}°  "
        f"({N_ANT} ant · {N_INT} int)   RMS={rms:.3f} Jy/beam"
    )
    fig.tight_layout()
    out_fig = OUT_DIR / f"step4_tile0_{tag}.png"
    fig.savefig(str(out_fig), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_fig}")

# ── 10. Image statistics ──────────────────────────────────────────────────────
print("\n=== Image statistics ===")
for path, name in [(restored_path, "Restored"), (dirty_path, "Dirty")]:
    if path.exists():
        with fits.open(str(path)) as hdul:
            d = hdul[0].data.squeeze()
        rms = np.nanstd(d)
        peak = np.nanmax(d)
        print(f"  {name}: peak={peak:.4f} Jy/beam  rms={rms:.4f} Jy/beam  SNR={peak/rms:.1f}")
        if psf_path.exists():
            with fits.open(str(psf_path)) as hdul:
                psf_data = hdul[0].data.squeeze()
            cy, cx = psf_data.shape[0]//2, psf_data.shape[1]//2
            from numpy import mgrid
            Y, X = mgrid[0:psf_data.shape[0], 0:psf_data.shape[1]]
            main_mask = (Y-cy)**2 + (X-cx)**2 < 20**2
            sl_peak = float(psf_data[~main_mask].max())
            print(f"  PSF sidelobe peak: {sl_peak:.4f}  (main lobe = 1.000)")

print("\nDone.")
