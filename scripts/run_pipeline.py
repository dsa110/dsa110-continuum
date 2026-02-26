#!/opt/miniforge/envs/casa6/bin/python
"""Run pipeline: phaseshift → apply cal → image."""
import os
import sys

# Ensure the project root is on the path so dsa110_continuum is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MS = "/stage/dsa110-contimg/ms/2026-01-25T22:26:05.ms"
BP_TABLE = "/stage/dsa110-contimg/ms/2026-01-25T22:26:05_0~23.b"
G_TABLE = "/stage/dsa110-contimg/ms/2026-01-25T22:26:05_0~23.g"
OUTPUT_DIR = "/stage/dsa110-contimg/images/3c454"
MERIDIAN_MS = "/stage/dsa110-contimg/ms/2026-01-25T22:26:05_meridian.ms"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Phaseshift to median meridian
print("=== Step 1: Phaseshift to median meridian ===")
if os.path.exists(MERIDIAN_MS):
    print(f"  Already exists: {MERIDIAN_MS}")
else:
    from dsa110_continuum.calibration.runner import phaseshift_ms
    meridian_ms, info = phaseshift_ms(
        ms_path=MS,
        mode="median_meridian",
        output_ms=MERIDIAN_MS,
    )
    print(f"  Created: {meridian_ms}")
    print(f"  Info: {info}")

# Step 2: Apply calibration to meridian MS
print("\n=== Step 2: Apply calibration ===")
from casacore.tables import table
import numpy as np

with table(MERIDIAN_MS, readonly=True, ack=False) as t:
    cols = t.colnames()
    has_corrected = 'CORRECTED_DATA' in cols
    if has_corrected:
        raw = t.getcol('DATA', nrow=1000)
        corr = t.getcol('CORRECTED_DATA', nrow=1000)
        flag = t.getcol('FLAG', nrow=1000)
        good = ~flag
        ratio = np.mean(np.abs(corr[good])) / np.mean(np.abs(raw[good]))
        print(f"  CORRECTED_DATA ratio: {ratio:.2f}")
        needs_cal = ratio < 5.0  # If ratio is close to 1, cal not applied
    else:
        print("  No CORRECTED_DATA — applying calibration")
        needs_cal = True

if needs_cal:
    from dsa110_continuum.calibration.applycal import apply_to_target
    apply_to_target(
        ms_target=MERIDIAN_MS,
        field="",
        gaintables=[BP_TABLE, G_TABLE],
        interp=["linear", "linear"],
    )
    print("  Calibration applied.")
else:
    print("  Calibration already applied (ratio > 5).")

# Verify after applying
with table(MERIDIAN_MS, readonly=True, ack=False) as t:
    raw = t.getcol('DATA', nrow=10000)
    corr = t.getcol('CORRECTED_DATA', nrow=10000)
    flag = t.getcol('FLAG', nrow=10000)
    good = ~flag
    ratio = np.mean(np.abs(corr[good])) / np.mean(np.abs(raw[good]))
    print(f"  Final CORRECTED/DATA ratio: {ratio:.2f}")

# Step 3: Image
print("\n=== Step 3: Image with WSClean ===")
from dsa110_continuum.imaging.cli_imaging import image_ms

imagename = os.path.join(OUTPUT_DIR, "3c454")
print(f"  Imagename: {imagename}")
print(f"  MS: {MERIDIAN_MS}")

image_ms(
    ms_path=MERIDIAN_MS,
    imagename=imagename,
    imsize=2400,
    cell_arcsec=6.0,
    weighting="briggs",
    robust=0.5,
    niter=1000,
    threshold="0.005Jy",
    pbcor=True,
    gridder="wgridder",
    backend="wsclean",
    use_unicat_mask=False,
)
print("  Imaging complete.")

# Step 4: Check peak flux
print("\n=== Step 4: Verify peak flux ===")
import glob
fits_files = glob.glob(f"{imagename}*.fits")
print(f"  FITS files: {fits_files}")

for fits_file in fits_files:
    try:
        from astropy.io import fits
        import numpy as np
        with fits.open(fits_file) as hdu:
            data = hdu[0].data
            if data is not None:
                peak = np.nanmax(np.abs(data))
                print(f"  {os.path.basename(fits_file)}: peak = {peak:.4f} Jy/beam")
    except Exception as e:
        print(f"  Error reading {fits_file}: {e}")
