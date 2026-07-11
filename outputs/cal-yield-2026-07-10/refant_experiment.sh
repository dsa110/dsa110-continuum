#!/bin/bash
# Controlled refant experiment: 2026-01-25 3C454.3 solve with refants 103/104/105.
# Each run works on its own scratch copy of the base cal MS; production
# /stage tables and MS are never written.
set -u
BASE=/stage/dsa110-contimg/ms/2026-01-25T22:26:05.ms
EXP=/data/dsa110-continuum/outputs/cal-yield-2026-07-10
PY=/opt/miniforge/envs/casa6/bin/python
export PYTHONPATH=/data/dsa110-continuum

for R in 103 104 105; do
  D=$EXP/refant_$R
  mkdir -p "$D"
  if [ ! -d "$D/cal.ms" ]; then
    echo "[$(date -u +%H:%M:%S)] copying MS for refant $R ..."
    cp -r "$BASE" "$D/cal.ms"
  fi
  echo "[$(date -u +%H:%M:%S)] solving refant=$R ..."
  $PY - "$R" "$D" > "$D/solve.log" 2>&1 <<'PYEOF'
import sys
refant, d = sys.argv[1], sys.argv[2]
sys.path.insert(0, "/data/dsa110-continuum")
from dsa110_continuum.calibration.runner import run_calibrator
tables = run_calibrator(
    f"{d}/cal.ms",
    cal_field="0~23",
    refant=refant,
    do_flagging=True,
    do_k=False,
    table_prefix=f"{d}/solve",
    calibrator_name="3C454.3",
    do_phaseshift=True,
)
print("TABLES:", tables)
PYEOF
  echo "[$(date -u +%H:%M:%S)] refant $R exit=$? (log: $D/solve.log)"
done
echo "ALL DONE"
