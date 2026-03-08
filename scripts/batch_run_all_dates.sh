#!/usr/bin/env bash
# batch_run_all_dates.sh — run the continuum pipeline + verify_sources QA for multiple dates.
#
# Usage:
#   ./scripts/batch_run_all_dates.sh [--dates "2026-01-25 2026-02-12 2026-02-15"] \
#                                     [--cal-date 2026-01-25] \
#                                     [--qa-csv products/qa_summary.csv]
#
# For each date:
#   1. Run batch_pipeline.py (calibrate, image, mosaic)
#   2. Run verify_sources.py on each epoch mosaic found
#   3. Append a row to $QA_CSV
#
# Exit code: 0 if all dates passed QA (median_ratio >= 0.70), non-zero otherwise.
#
# Environment:
#   PYTHON            - Python interpreter (default: /opt/miniforge/envs/casa6/bin/python)
#   PRODUCTS_BASE     - output base directory (default: /data/dsa110-continuum/products/mosaics)
#   NVSS_DB           - path to nvss_full.sqlite3
#   MASTER_DB         - path to master_sources.sqlite3
#   ATNF_DB           - path to atnf_full.sqlite3
#   DSA110_MIN_FLUX_JY- minimum catalog flux for QA (default: 0.050)

set -euo pipefail

PYTHON="${PYTHON:-/opt/miniforge/envs/casa6/bin/python}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PRODUCTS_BASE="${PRODUCTS_BASE:-${REPO_ROOT}/products/mosaics}"
NVSS_DB="${NVSS_DB:-/data/dsa110-contimg/state/catalogs/nvss_full.sqlite3}"
MASTER_DB="${MASTER_DB:-/data/dsa110-contimg/state/catalogs/master_sources.sqlite3}"
ATNF_DB="${ATNF_DB:-/data/dsa110-contimg/state/catalogs/atnf_full.sqlite3}"
MIN_FLUX_JY="${DSA110_MIN_FLUX_JY:-0.050}"

# ── Argument parsing ─────────────────────────────────────────────────────────
DATES=""
CAL_DATE="2026-01-25"
QA_CSV="${REPO_ROOT}/products/qa_summary.csv"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dates)        DATES="$2";       shift 2 ;;
        --cal-date)     CAL_DATE="$2";    shift 2 ;;
        --qa-csv)       QA_CSV="$2";      shift 2 ;;
        --min-flux-jy)  MIN_FLUX_JY="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Default: use all dates that have mosaic products
if [[ -z "$DATES" ]]; then
    DATES="$(ls "$PRODUCTS_BASE" 2>/dev/null || true)"
fi

if [[ -z "$DATES" ]]; then
    echo "No dates found in $PRODUCTS_BASE and no --dates provided."
    exit 1
fi

# ── Initialise QA CSV ────────────────────────────────────────────────────────
mkdir -p "$(dirname "$QA_CSV")"
if [[ ! -f "$QA_CSV" ]]; then
    echo "date,epoch_utc,mosaic_path,median_ratio,n_detections,gaincal_used,qa_result" > "$QA_CSV"
fi

# ── Per-date loop ────────────────────────────────────────────────────────────
overall_fail=0

for DATE in $DATES; do
    echo "======================================================================"
    echo "DATE: $DATE  (cal-date: $CAL_DATE)"
    echo "======================================================================"

    STAGE_DIR="${PRODUCTS_BASE}/${DATE}"
    LOG_DIR="${REPO_ROOT}/logs"
    mkdir -p "$LOG_DIR"
    PIPELINE_LOG="${LOG_DIR}/batch_pipeline_${DATE}.log"

    # ── Step 1: Run batch_pipeline.py ──────────────────────────────────────
    echo "[1/2] Running batch_pipeline.py for $DATE ..."
    set +e
    "$PYTHON" "${REPO_ROOT}/scripts/batch_pipeline.py" \
        --date "$DATE" \
        --cal-date "$CAL_DATE" \
        --skip-photometry \
        2>&1 | tee "$PIPELINE_LOG"
    PIPELINE_EXIT=$?
    set -e

    if [[ $PIPELINE_EXIT -ne 0 ]]; then
        echo "WARNING: batch_pipeline.py exited $PIPELINE_EXIT for $DATE — QA may be incomplete"
    fi

    # ── Step 2: Run verify_sources.py on each epoch mosaic ─────────────────
    echo "[2/2] Running verify_sources.py on epoch mosaics for $DATE ..."

    # Find epoch mosaics produced today (pattern: YYYY-MM-DDTHH00_mosaic.fits)
    mapfile -t MOSAIC_PATHS < <(find "$STAGE_DIR" -name "*_mosaic.fits" 2>/dev/null | sort)

    if [[ ${#MOSAIC_PATHS[@]} -eq 0 ]]; then
        echo "  No mosaic FITS found in $STAGE_DIR — skipping QA for $DATE"
        echo "${DATE},,${STAGE_DIR},nan,0,unknown,no_mosaics" >> "$QA_CSV"
        overall_fail=1
        continue
    fi

    for MOSAIC in "${MOSAIC_PATHS[@]}"; do
        EPOCH_UTC="$(basename "$MOSAIC" | sed 's/_mosaic\.fits//')"
        VERIFY_CSV="${STAGE_DIR}/${EPOCH_UTC}_verify.csv"
        VERIFY_LOG="${LOG_DIR}/verify_${EPOCH_UTC}.log"

        echo "  Verifying: $EPOCH_UTC"
        set +e
        QA_LINE="$("$PYTHON" "${REPO_ROOT}/scripts/verify_sources.py" \
            --fits     "$MOSAIC" \
            --master-db "$MASTER_DB" \
            --nvss-db   "$NVSS_DB" \
            --atnf-db   "$ATNF_DB" \
            --out       "$VERIFY_CSV" \
            --min-flux-jy "$MIN_FLUX_JY" \
            --box-pix   3 \
            2>"$VERIFY_LOG")"
        VERIFY_EXIT=$?
        set -e

        echo "  $QA_LINE"

        # Parse median_ratio and n_sources from the QA line
        # Format: VERIFY PASS|WARN|FAIL: median_ratio=X.XXX n_sources=NNN
        MEDIAN_RATIO="$(echo "$QA_LINE" | grep -oP 'median_ratio=\K[^ ]+' || echo 'nan')"
        N_SOURCES="$(echo "$QA_LINE"    | grep -oP 'n_sources=\K[0-9]+'   || echo '0')"
        if   echo "$QA_LINE" | grep -q "PASS"; then QA_RESULT="pass"
        elif echo "$QA_LINE" | grep -q "WARN"; then QA_RESULT="warn"
        else                                         QA_RESULT="fail"; overall_fail=1
        fi

        # Determine whether gaincal was used (check for "fallback" in pipeline log)
        if grep -q "BP-only\|bandpass-only\|fallback" "$PIPELINE_LOG" 2>/dev/null; then
            GAINCAL_USED="bp_only"
        else
            GAINCAL_USED="epoch_ap"
        fi

        echo "${DATE},${EPOCH_UTC},${MOSAIC},${MEDIAN_RATIO},${N_SOURCES},${GAINCAL_USED},${QA_RESULT}" >> "$QA_CSV"
    done
done

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "QA SUMMARY written to: $QA_CSV"
cat "$QA_CSV"
echo "======================================================================"

if [[ $overall_fail -ne 0 ]]; then
    echo "OVERALL: FAIL (one or more epochs did not pass QA)"
    exit 1
else
    echo "OVERALL: PASS (all epochs passed QA)"
    exit 0
fi
