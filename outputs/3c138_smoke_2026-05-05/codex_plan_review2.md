# Codex Plan Review #2

## 1. Verdict

**NO-GO.** The conversion command is fixable, but the calibration acquisition path still cannot guarantee a 2026-02-15 hour-03 3C138 primary anchor and can fall back without tripping a hard gate.

## 2. Plan-blocking risks

- `/data/dsa110-contimg/backend/src/dsa110_contimg/interfaces/cli/commands/convert.py:87` and `pyproject.toml:148`: the live conversion CLI is `dsa110 convert`; `python -m dsa110_continuum.cli convert` has no module in this checkout. Change Step 2 to `/opt/miniforge/envs/casa6/bin/dsa110 convert ...`.

- `dsa110_continuum/conversion/calibrator_ms_generator.py:493-501` and `/data/dsa110-contimg/backend/src/dsa110_contimg/infrastructure/database/hdf5_index.py:1624-1636`: `generate_from_transit()` ignores `transit_time` for selection and calls a spatial selector over all stored HDF5 groups with no date/window predicate. Change to date/time-scope the selector or bypass auto-cal and manually calibrate the known 2026-02-15T03:33:38 MS.

- `dsa110_continuum/conversion/calibrator_ms_generator.py:512` plus `dsa110_continuum/conversion/calibrator_ms_generator.py:374-383`: the selector returns timestamp strings, but `convert_groups()` treats each group as a file list and derives `Path(files[0])`, i.e. the first character of the timestamp. Change `convert_groups()` to accept group IDs properly or pass real `SubbandGroup.files`.

- `dsa110_continuum/calibration/ensure.py:772-779`: `generate_bandpass_tables()` accepts a generated MS based only on `success` and `ms_path`; it ignores `calibrator_in_ms=False`. Change it to fail unless the selected MS verifies the calibrator.

## 3. Silent wrong-results risks

- `scripts/batch_pipeline.py:1157-1159`: `--force-recal` is still not passed as `force=True` to `ensure_bandpass()`. If any same-date `2026-02-15T*_0~23.{b,g}` table remains, `find_cal_tables()` can reuse the first real glob match (`dsa110_continuum/calibration/ensure.py:340-355`) instead of generating fresh 3C138 tables.

- `dsa110_continuum/calibration/ensure.py:989-994`: if primary generation fails, `ensure_bandpass()` borrows nearest real tables. `scripts/batch_pipeline.py:903-923` only gates cal-table quality, not `selection_pool`, `flux_anchor`, or `calibrator_name`, so a non-primary/borrowed output can still "succeed."

- `scripts/batch_pipeline.py:1165-1177`: if auto-cal raises, batch falls back to filesystem table resolution. With stale same-date tables present, this can silently proceed outside the 3C138 primary-anchor contract.

- Positive checks: primary selection itself would rank by Dec proximity (`dsa110_continuum/calibration/ensure.py:592-602`), and 3C138 is a primary at Dec +16.64 with 8.36 Jy (`dsa110_continuum/calibration/fluxscale.py:97-103`). The blocker is the downstream generator, not the ranking.

## 4. Concrete commands to add/change

Use the live CLI and avoid the `/tmp` path-policy warning:

```bash
mkdir -p /dev/shm/dsa110-convert
DSA110_SKIP_DOTENV=1 TMPDIR=/dev/shm/dsa110-convert CONTIMG_TMPFS_DIR=/dev/shm/dsa110-contimg \
  /opt/miniforge/envs/casa6/bin/dsa110 convert \
  --input-dir /data/incoming \
  --output-dir /stage/dsa110-contimg/ms \
  --start-time 2026-02-15T03:00:00 \
  --end-time 2026-02-15T04:00:00 \
  --execution-mode inprocess \
  --no-diagnostics
```

Back up all same-date BP/G globs, not only the known T22 pair:

```bash
for ext in b g; do
  for f in /stage/dsa110-contimg/ms/2026-02-15T*_0~23.$ext; do
    [ -e "$f" ] && mv "$f" "$f.bak-pre-3c138-demo-20260505"
  done
done
```

Do not use auto-cal for the real batch run until the generator is fixed. Generate/verify a fresh 3C138 table from the known in-hour MS, write the provenance sidecar, then run batch with `--skip-auto-cal`:

```bash
/opt/miniforge/envs/casa6/bin/python -m dsa110_continuum.calibration.cli calibrate \
  --ms /stage/dsa110-contimg/ms/2026-02-15T03:33:38.ms \
  --calibrator 3C138 --field 0~23 --refant 103 \
  --output-prefix /stage/dsa110-contimg/ms/2026-02-15T03:33:38_0~23

/opt/miniforge/envs/casa6/bin/python - <<PY2
from dsa110_continuum.calibration.ensure import write_provenance_sidecar
bp = "/stage/dsa110-contimg/ms/2026-02-15T03:33:38_0~23.b"
g = "/stage/dsa110-contimg/ms/2026-02-15T03:33:38_0~23.g"
write_provenance_sidecar(bp, {
    "selection_mode": "manual_primary",
    "selection_pool": "primary",
    "flux_anchor": "perley_butler_primary",
    "obs_dec_deg_used": 16.19,
    "selection_dec_tolerance_deg": 10.0,
    "calibrator_name": "3C138",
    "calibrator_ra_deg": 80.29119,
    "calibrator_dec_deg": 16.63946,
    "calibrator_flux_jy": 8.36,
    "calibrator_dec_offset_deg": 0.45,
    "transit_time_iso": "2026-02-15 03:34:49.095",
    "source": "generated",
    "cal_date": "2026-02-15",
    "bp_table": bp,
    "g_table": g,
})
PY2
```

Real batch should add `--skip-auto-cal --cal-date 2026-02-15`. Pass `--expected-dec 16.19` for clarity. `16.2` and the default `16.1` both pass the 5 deg guard (`scripts/batch_pipeline.py:873-899`), but `16.19` matches the verified strip.

11 tiles are acceptable: batch only aborts below 2 tiles (`scripts/batch_pipeline.py:1609-1611`), epoch gaincal uses all MSs if fewer than `MOSAIC_TILE_COUNT=12` (`scripts/batch_pipeline.py:1359-1361`), and epoch mosaics use whatever tiles are in the hour (`scripts/batch_pipeline.py:591-617`).

`--force-recal` does clear cached epoch `*.ap.G` independently of BP/G presence (`scripts/batch_pipeline.py:1316-1327`), but it does not force daily BP/G regeneration.

## 5. Confidence

**Medium.** The blockers are grounded in code, but I did not run conversion or the pipeline, and the live `dsa110` CLI is served from the sibling legacy checkout.
