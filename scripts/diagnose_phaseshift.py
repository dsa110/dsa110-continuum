"""Deep diagnostic: trace exactly why phaseshift produces 104° phase scatter.

Strategy: simulate a SINGLE source at the EXACT phase centre, generate a 
short (4 int) drift-scan, check visibilities at each stage.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/user/workspace/dsa110-continuum")

import pyuvdata
from astropy.time import Time as ATime
import astropy.units as u

from dsa110_continuum.simulation.harness import SimulationHarness
from dsa110_continuum.simulation.pipeline import SimulatedPipeline
import pyradiosky

OUT = Path("/home/user/workspace/dsa110-continuum/pipeline_outputs/diag_phaseshift")
OUT.mkdir(parents=True, exist_ok=True)

# ── Minimal config: 4 antennas, 4 integrations, 1 subband ────────────────────
N_ANT = 8
N_INT = 4
T_INT = 12.884902
T0    = ATime("2026-01-25T22:26:05", format="isot", scale="utc")
OVRO_LON = -118.2825

# Compute LSTs for each integration
lsts_deg = []
for i in range(N_INT):
    t_mid = ATime(T0.jd + (i + 0.5) * T_INT / 86400.0, format="jd", scale="utc")
    lst = float(t_mid.sidereal_time("apparent", longitude=OVRO_LON * u.deg).deg)
    lsts_deg.append(lst)

lsts_deg = np.array(lsts_deg)
median_ra = float(np.median(lsts_deg))
dec_deg   = 16.15

print(f"LSTs (deg): {lsts_deg}")
print(f"Median RA:  {median_ra:.4f}°")
print(f"LST spread: {lsts_deg[-1]-lsts_deg[0]:.4f}°")

# ── 1. Place a single source exactly at the MEDIAN RA (common phase centre) ──
harness = SimulationHarness(
    n_antennas=N_ANT,
    n_integrations=N_INT,
    n_sky_sources=0,
    noise_jy=0.0,
    seed=42,
    use_real_positions=True,
)
harness.pointing_ra_deg  = median_ra
harness.pointing_dec_deg = dec_deg

# Build a sky with exactly 1 source at the median RA (phase centre)
sky_at_centre = pyradiosky.SkyModel(
    ra=pyradiosky.utils.longitude_type([median_ra], unit="deg"),
    dec=pyradiosky.utils.latitude_type([dec_deg], unit="deg"),
    stokes=pyradiosky.stokes_utils.jy_to_stokes(np.array([[[[1.0]]]])),
    spectral_type="flat",
    name=np.array(["centre_source"]),
)

print("\n--- Stage A: Generate drift-scan UVH5 ---")
uvh5_paths = harness.generate_subbands(
    output_dir=OUT / "raw_centre",
    n_subbands=1,
    start_time=T0,
    sky=sky_at_centre,
    drift_scan=True,
)
print(f"  Generated {len(uvh5_paths)} UVH5")

# Read the raw UVH5
uv_raw = pyuvdata.UVData()
uv_raw.read(str(uvh5_paths[0]))

print(f"\n  n_times in raw UVH5: {uv_raw.Ntimes}")
print(f"  n_phase_centers: {len(uv_raw.phase_center_catalog)}")
print(f"  Phase centers: {list(uv_raw.phase_center_catalog.values())[:4]}")

# Check first integration raw data
for t_idx in range(N_INT):
    t_mask = uv_raw.time_array == np.unique(uv_raw.time_array)[t_idx]
    data_t = uv_raw.data_array[t_mask, :, 0]  # XX pol
    print(f"  t={t_idx}: mean amp={np.mean(np.abs(data_t)):.4f}  "
          f"phase={np.mean(np.angle(data_t, deg=True)):.1f}°  "
          f"phase_std={np.std(np.angle(data_t, deg=True)):.1f}°")

# ── 2. Write to MS ────────────────────────────────────────────────────────────
print("\n--- Stage B: Write UVH5 → MS ---")
ms_path = OUT / "centre.ms"
uv_raw.write_ms(str(ms_path))
print(f"  MS written to {ms_path}")

# Read back the MS and check
import casacore.tables as ct
t = ct.table(str(ms_path), readonly=True)
data_ms = t.getcol("DATA")   # (nrow, nchan, npol)
print(f"  MS shape: {data_ms.shape}")
print(f"  MS mean amp: {np.mean(np.abs(data_ms[:,:,0])):.4f}")
print(f"  MS phase std: {np.std(np.angle(data_ms[:,:,0], deg=True)):.1f}°")
t.close()

# ── 3. Apply pyuvdata.phase() ────────────────────────────────────────────────
print("\n--- Stage C: pyuvdata.phase() → median RA ---")
uv_phased = pyuvdata.UVData()
uv_phased.read(str(ms_path))

print(f"  Before phase: n_phase_centers={len(uv_phased.phase_center_catalog)}")
for k, v in uv_phased.phase_center_catalog.items():
    ra_deg = np.degrees(v['cat_lon'])
    dec_deg_ = np.degrees(v['cat_lat'])
    print(f"    id={k}: cat_name={v['cat_name']}  RA={ra_deg:.4f}°  Dec={dec_deg_:.4f}°")

# Check UVW before
print(f"  UVW max before: {np.max(np.abs(uv_phased.uvw_array)):.1f} m")

uv_phased.phase(
    ra=np.radians(median_ra),
    dec=np.radians(dec_deg),
    cat_name="median_meridian",
    use_ant_pos=True,
)

print(f"  After phase: n_phase_centers={len(uv_phased.phase_center_catalog)}")
print(f"  UVW max after: {np.max(np.abs(uv_phased.uvw_array)):.1f} m")

# Check data after phase
data_phased = uv_phased.data_array[:, :, 0]
print(f"  After phase: mean amp={np.mean(np.abs(data_phased)):.4f}")
print(f"  After phase: mean phase={np.mean(np.angle(data_phased, deg=True)):.2f}°")
print(f"  After phase: phase std={np.std(np.angle(data_phased, deg=True)):.2f}°")

# Per integration
for t_idx in range(N_INT):
    t_mask = uv_phased.time_array == np.unique(uv_phased.time_array)[t_idx]
    data_t = uv_phased.data_array[t_mask, :, 0]
    print(f"  t={t_idx}: mean phase={np.mean(np.angle(data_t, deg=True)):.2f}°  "
          f"std={np.std(np.angle(data_t, deg=True)):.2f}°")

# ── 4. Write phased MS and read back with casacore ───────────────────────────
print("\n--- Stage D: Write phased UVData → MS ---")
phased_ms = OUT / "centre_phased.ms"
uv_phased.write_ms(str(phased_ms))

t = ct.table(str(phased_ms), readonly=True)
data_phased_ms = t.getcol("DATA")
print(f"  Phased MS shape: {data_phased_ms.shape}")
print(f"  Phased MS mean amp: {np.mean(np.abs(data_phased_ms[:,:,0])):.4f}")
print(f"  Phased MS phase std: {np.std(np.angle(data_phased_ms[:,:,0], deg=True)):.2f}°")
t.close()

# ── 5. Now try: place a source at a DIFFERENT position (off-centre) ──────────
print("\n--- Stage E: Source at FIRST LST (not median) —should NOT be coherent ---")
lst0 = lsts_deg[0]
sky_at_lst0 = pyradiosky.SkyModel(
    ra=pyradiosky.utils.longitude_type([lst0], unit="deg"),
    dec=pyradiosky.utils.latitude_type([dec_deg], unit="deg"),
    stokes=pyradiosky.stokes_utils.jy_to_stokes(np.array([[[[1.0]]]])),
    spectral_type="flat",
    name=np.array(["lst0_source"]),
)

uvh5_paths2 = harness.generate_subbands(
    output_dir=OUT / "raw_lst0",
    n_subbands=1,
    start_time=T0,
    sky=sky_at_lst0,
    drift_scan=True,
)
uv2 = pyuvdata.UVData()
uv2.read(str(uvh5_paths2[0]))
uv2.write_ms(str(OUT / "lst0.ms"))

uv2_p = pyuvdata.UVData()
uv2_p.read(str(OUT / "lst0.ms"))
uv2_p.phase(ra=np.radians(median_ra), dec=np.radians(dec_deg),
             cat_name="median_meridian", use_ant_pos=True)
data2 = uv2_p.data_array[:, :, 0]
print(f"  Off-centre source: phase std={np.std(np.angle(data2, deg=True)):.2f}°  (expected large)")

print("\n--- SUMMARY ---")
print("If Stage C shows phase_std ≈ 0°: phaseshift is working correctly in pyuvdata")
print("If Stage C shows phase_std >> 0°: pyuvdata.phase() is broken for multi-field MS")
