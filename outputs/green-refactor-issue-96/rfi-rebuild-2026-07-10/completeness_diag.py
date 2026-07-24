"""Diagnose the 44.9% completeness FAIL on the 2026-01-25T2200 RFI rebuild.

Replicates measure_epoch_qa()'s catalog query and recovery test, then breaks
the 383 catalog sources down by *why* they were or weren't recovered:
  - source lands on NaN / zero-weight pixels (never observed -> unfair miss)
  - source on valid pixels but peak <= 5*local RMS (true miss)
  - recovered
Also reports completeness restricted to covered (finite-pixel) sources, and
the sensitivity of the verdict to that coverage correction.
"""
import sys, json
import numpy as np
import sqlite3
from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, "/data/dsa110-continuum")
from dsa110_continuum.photometry.epoch_qa import (
    _image_rms_mad, _local_rms, _peak_in_box, _sky_footprint,
    QA_MIN_FLUX_MJY, QA_RECOVERY_SIGMA, DEFAULT_NVSS_DB,
)

mosaic = "/stage/dsa110-contimg/images/mosaic_2026-01-25/2026-01-25T2200_mosaic.fits"
weights = "/stage/dsa110-contimg/images/mosaic_2026-01-25/2026-01-25T2200_mosaic.weights.fits"

with fits.open(mosaic) as h:
    hdr = h[0].header
    raw = h[0].data
while raw.ndim > 2:
    raw = raw[0]
data = raw.astype(np.float64)
wcs = WCS(hdr, naxis=2)
with fits.open(weights) as h:
    w = np.squeeze(h[0].data).astype(np.float64)

ny, nx = data.shape
ra_min, ra_max, dec_min, dec_max = _sky_footprint(wcs, ny, nx)
print(f"footprint deg: RA [{ra_min:.3f}, {ra_max:.3f}] Dec [{dec_min:.3f}, {dec_max:.3f}]")

con = sqlite3.connect(DEFAULT_NVSS_DB)
rows = con.execute(
    "SELECT ra_deg, dec_deg, flux_mjy FROM sources "
    "WHERE ra_deg BETWEEN ? AND ? AND dec_deg BETWEEN ? AND ? AND flux_mjy >= ?",
    (ra_min, ra_max, dec_min, dec_max, QA_MIN_FLUX_MJY)).fetchall()
con.close()
print("n_catalog:", len(rows))

cats = {"off_grid": 0, "nan_pixel": 0, "zero_weight": 0, "miss_valid": 0, "recovered": 0}
miss_details = []
flux_recovered, flux_missed = [], []
for ra, dec, fmjy in rows:
    pix = wcs.world_to_pixel_values(ra, dec)
    cx, cy = int(round(float(pix[0]))), int(round(float(pix[1])))
    if not (2 <= cy < ny - 2 and 2 <= cx < nx - 2):
        cats["off_grid"] += 1
        continue
    if not np.isfinite(data[cy, cx]):
        cats["nan_pixel"] += 1
        continue
    if not (np.isfinite(w[cy, cx]) and w[cy, cx] > 0):
        cats["zero_weight"] += 1
        continue
    local = _local_rms(data, cy, cx)
    peak = _peak_in_box(data, cy, cx, half=1)
    if peak > QA_RECOVERY_SIGMA * local:
        cats["recovered"] += 1
        flux_recovered.append(fmjy)
    else:
        cats["miss_valid"] += 1
        flux_missed.append(fmjy)
        miss_details.append({
            "ra": ra, "dec": dec, "flux_mjy": fmjy,
            "peak_mjy": peak * 1e3, "local_rms_mjy": local * 1e3,
            "snr": peak / local if local > 0 else float("nan"),
            "weight_rel": float(w[cy, cx] / np.nanmax(w)),
        })

n_cat = len(rows)
n_covered = cats["recovered"] + cats["miss_valid"]
print(json.dumps(cats, indent=1))
print(f"raw completeness (as gate computes): {cats['recovered']}/{n_cat} = {cats['recovered']/n_cat:.3f}")
print(f"coverage-corrected completeness:     {cats['recovered']}/{n_covered} = {cats['recovered']/max(n_covered,1):.3f}")
if flux_missed:
    fm = np.array(flux_missed)
    print(f"missed-on-valid flux distribution mJy: median {np.median(fm):.0f}, "
          f"p25 {np.percentile(fm,25):.0f}, p75 {np.percentile(fm,75):.0f}, max {fm.max():.0f}")
if flux_recovered:
    fr = np.array(flux_recovered)
    print(f"recovered flux distribution mJy:       median {np.median(fr):.0f}, "
          f"p25 {np.percentile(fr,25):.0f}, p75 {np.percentile(fr,75):.0f}")
miss_details.sort(key=lambda d: -d["flux_mjy"])
print("\nTop 15 brightest misses on valid pixels:")
for d in miss_details[:15]:
    print(f"  {d['flux_mjy']:8.0f} mJy cat | peak {d['peak_mjy']:7.1f} mJy | "
          f"local_rms {d['local_rms_mjy']:6.1f} mJy | snr {d['snr']:5.1f} | w/wmax {d['weight_rel']:.2f} "
          f"| ra {d['ra']:.3f} dec {d['dec']:.3f}")
# SNR distribution of valid misses: near-threshold or hopeless?
if miss_details:
    snrs = np.array([d["snr"] for d in miss_details])
    print(f"\nmiss SNR distribution: median {np.median(snrs):.1f}, "
          f"n(snr>3) {(snrs>3).sum()}, n(snr>4) {(snrs>4).sum()}, n(4<snr<=5) {((snrs>4)&(snrs<=5)).sum()}")
