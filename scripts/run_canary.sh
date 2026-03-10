#!/usr/bin/env bash
# Pre-existing-tile QA canary — fast regression smoke test.
#
# This script does NOT re-run calibration or imaging. It validates the QA
# measurement path (measure_epoch_qa) on a known-good, pre-existing canary
# FITS tile (2026-01-25T22:26:05, contains 3C454.3 at ~12.5 Jy/beam).
#
# Use this after changes to QA code or photometry helpers to confirm the
# QA pipeline still produces sensible results on a well-characterised tile.
# For a full pipeline execution test, run batch_pipeline.py on the canary date.
#
# Expected: median_ratio ~1.0, n_recovered >= 3, RMS <= 17.1 mJy/beam
# Runtime: < 1 minute (reads pre-existing FITS, no reprocessing).
#
# Usage:
#   bash scripts/run_canary.sh               # uses default canary tile
#   bash scripts/run_canary.sh /path/to.fits  # test arbitrary mosaic
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="/opt/miniforge/envs/casa6/bin/python"
NVSS_DB="/data/dsa110-contimg/state/catalogs/nvss_full.sqlite3"

# Allow overriding canary FITS via first argument
CANARY_FITS="${1:-/stage/dsa110-contimg/images/mosaic_2026-01-25/2026-01-25T22:26:05-image-pb.fits}"
# Fallback to non-pb-corrected if pb version doesn't exist
if [[ ! -f "$CANARY_FITS" ]]; then
    CANARY_FITS="${1:-/stage/dsa110-contimg/images/mosaic_2026-01-25/2026-01-25T22:26:05-image.fits}"
fi

echo "=== DSA-110 Canary QA Smoke Test (pre-existing tile) ==="
echo "Tile:    $CANARY_FITS"
echo "NVSS DB: $NVSS_DB"
echo ""

# Validate inputs exist
if [[ ! -f "$CANARY_FITS" ]]; then
    echo "ERROR: Canary FITS not found: $CANARY_FITS"
    echo "  Run the pipeline first: $PYTHON scripts/mosaic_day.py --date 2026-01-25"
    exit 1
fi
if [[ ! -f "$NVSS_DB" ]]; then
    echo "ERROR: NVSS database not found: $NVSS_DB"
    exit 1
fi

# Run epoch QA and apply canary-specific acceptance criteria.
# Canary criteria (from Phase 0 plan) are looser than full epoch QA:
#   median_ratio ∈ [0.85, 1.15]
#   n_recovered ≥ 3
#   RMS ≤ 17.1 mJy/beam
"$PYTHON" -c "
import sys, math
sys.path.insert(0, '${REPO_ROOT}')
from dsa110_continuum.photometry.epoch_qa import measure_epoch_qa

result = measure_epoch_qa('${CANARY_FITS}', '${NVSS_DB}')

print('--- Canary QA Results ---')
print(f'  Median DSA/NVSS ratio: {result.median_ratio:.3f}')
print(f'  Recovered sources:     {result.n_recovered} / {result.n_catalog}')
print(f'  Completeness:          {result.completeness_frac:.1%}')
print(f'  Mosaic RMS:            {result.mosaic_rms_mjy:.2f} mJy/beam')
print()
print('  Three-gate epoch QA (informational):')
print(f'    Flux scale gate:     {result.ratio_gate}')
print(f'    Completeness gate:   {result.completeness_gate}')
print(f'    Noise floor gate:    {result.rms_gate}')
print(f'    Overall epoch QA:    {result.qa_result}')

# Canary-specific acceptance criteria
ratio_ok = not math.isnan(result.median_ratio) and 0.85 <= result.median_ratio <= 1.15
n_ok = result.n_recovered >= 3
rms_ok = result.mosaic_rms_mjy <= 17.1
canary_pass = ratio_ok and n_ok and rms_ok

print()
print('  Canary acceptance criteria:')
print(f'    ratio in [0.85, 1.15]:   {\"PASS\" if ratio_ok else \"FAIL\"}  ({result.median_ratio:.3f})')
print(f'    n_recovered >= 3:        {\"PASS\" if n_ok else \"FAIL\"}  ({result.n_recovered})')
print(f'    RMS <= 17.1 mJy/beam:    {\"PASS\" if rms_ok else \"FAIL\"}  ({result.mosaic_rms_mjy:.2f})')
print(f'  ─────────────────────────')
print(f'  CANARY: {\"PASS\" if canary_pass else \"FAIL\"}')

if not canary_pass:
    sys.exit(1)
"

echo ""
echo "=== Canary QA Smoke Test Complete ==="
