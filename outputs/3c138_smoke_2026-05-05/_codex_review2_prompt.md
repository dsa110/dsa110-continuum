# Codex review #2: revised plan after R2 ruled out and R1-on-2026-01-25 blocked

You are a senior reviewer. **Read-only.** Do not modify any code, run the pipeline, or create artifacts other than the review file at the path requested below.

## Context recap (read fast — these are decided, not for relitigation)

- Goal: science-ready ~12-tile primary-anchored mosaic. User has fully delegated; I am driving with you as collaborator.
- Codex review #1 (yours, on the 2026-01-25 hour-04 plan) returned NO-GO with 4 findings:
  1. `--force-recal` doesn't propagate `force=True` into `ensure_bandpass()`.
  2. `batch_pipeline.py` doesn't convert HDF5→MS for tiles; it only globs existing MSs.
  3. Hour 04 of 2026-01-25 has only 6 MSs; HDF5 itself stops at 04:29 — the 04:57 transit *is not in the data*.
  4. `CalibratorMSGenerator.generate_from_transit()` ignores `transit_time` for selection (position-only).
- Findings filed as issues #70, #71, #72.

## Discovery results since #1

I queried HDF5 inventory + DB transits + array pointing for all indexed dates:

- **R2 ruled out**: only 2026-01-25 has substantial pre-converted MSs on disk; other indexed dates have ≤11 MSs total across 1–2 hours.
- The `daily_calibrator_transits` DB table lists transits for *all* dec strips per date, regardless of where the array was actually pointing. Verified by reading `phase_center_app_dec` from raw HDF5 at each primary's transit timestamp.
- **Actual primary-cal transits with the array pointing in-beam** (within 2°):
  - 2026-01-25 04:57 — 3C138 (pointing 16.27°) — **HDF5 stops at 04:29, transit not observed**
  - 2026-02-12 03:46 — 3C138 (pointing 16.26°) — full HDF5 hour 03 coverage
  - 2026-02-15 03:34 — 3C138 (pointing 16.19°) — full HDF5 hour 03 coverage ← chosen
  - 2026-02-22 23:26 — 3C48 (pointing 33.15°) — UTC day-boundary problem
  - 2026-02-23 03:03 — 3C138 (pointing 16.27°) — full hour 03 coverage

## Chosen target

- **Date**: 2026-02-15
- **Hour**: 03 (UTC `[03:00, 04:00)`)
- **Source**: 3C138 (J2000 0521+166, RA 80.29°, Dec +16.64°). Perley-Butler primary; ~8.3 Jy at 1.4 GHz averaged across L-band.
- **Transit**: 2026-02-15T03:34:49.095 UTC (closest HDF5 timestamp 03:33:38, well inside hour 03)
- **Array pointing in hour**: dec +16.188° throughout (verified 11 of 11 timestamps consistent).
- **Pre-flight state**:
  - **No** hour-03 MSs exist on disk yet (need conversion).
  - 11 HDF5 timestamps × 16 subbands = 176 raw files for hour 03.
  - Same-date BP/G tables exist at `/stage/dsa110-contimg/ms/2026-02-15T22:26:05_0~23.{b,g}` — must be bak'd to dodge priority-1 same-date acquisition.

## Updated execution plan

Step 1 — bak existing BP/G:
```bash
mv /stage/dsa110-contimg/ms/2026-02-15T22:26:05_0~23.b{,.bak-pre-3c138-demo-20260505}
mv /stage/dsa110-contimg/ms/2026-02-15T22:26:05_0~23.g{,.bak-pre-3c138-demo-20260505}
```

Step 2 — convert HDF5 → MS for hour 03:
```bash
/opt/miniforge/envs/casa6/bin/python -m dsa110_continuum.cli convert \
  --input-dir /data/incoming \
  --output-dir /stage/dsa110-contimg/ms \
  --start-time 2026-02-15T03:00:00 \
  --end-time   2026-02-15T04:00:00 \
  --execution-mode auto
```
(or whatever the canonical `dsa110 convert` invocation is)

Step 3 — dry-run batch_pipeline:
```bash
/opt/miniforge/envs/casa6/bin/python /data/dsa110-continuum/scripts/batch_pipeline.py \
  --date 2026-02-15 --start-hour 3 --end-hour 4 \
  --force-recal --skip-photometry --keep-intermediates \
  --retry-failed --archive-all \
  --tile-timeout 1800 --quarantine-after-failures 3 \
  --expected-dec 16.2 \
  --dry-run
```

Step 4 — real run (background): same as above sans `--dry-run`, redirect to `outputs/3c138_smoke_2026-05-05/run_<utc>.log`.

Acceptance bar (unchanged from previous review):
1. Cal provenance: `flux_anchor=perley_butler_primary`, `selection_pool=primary`, derived from 3C138 03:34 transit. Hard gate.
2. Pipeline completes; manifest + run_report.md emitted. Hard gate.
3. Mosaic FITS at `/stage/dsa110-contimg/images/mosaic_2026-02-15/2026-02-15T0300_mosaic.fits`. Hard gate.
4. 3C138 detected at `(80.29°, +16.64°)`, peak ±10% of ~8.3 Jy/beam. Hard gate ±10%; warn ±20%.
5. No silent-failure-invariant hits (PHASE_DIR / REFERENCE_DIR / TELESCOPE_NAME warnings). Soft gate.

## What I want from this review (terse)

Write a markdown file at `/data/dsa110-continuum/outputs/3c138_smoke_2026-05-05/codex_plan_review2.md` with:

1. **Verdict**: GO / GO-WITH-CHANGES / NO-GO. One sentence.
2. **Plan-blocking risks** (must fix before launch). For each: `path:line`, what the code does, what to change.
3. **Risks that would cause silent wrong-results** (would let the run "succeed" but produce non-primary-anchored output). For each: `path:line`, why it bites.
4. **Concrete commands to add/change**: e.g., correct `dsa110 convert` invocation if mine is wrong, additional flags, additional bak operations.
5. **Confidence**: high / medium / low with one-sentence reason.

Specific things I want you to verify by reading code (not by guessing):

- (a) **Conversion CLI form**: what's the exact command/path that produces MSs at `/stage/dsa110-contimg/ms/{date}T{hh:mm:ss}.ms`? `dsa110 convert` help shows `--start-time`/`--end-time`/`--input-dir`/`--output-dir`. Do those map to the correct `dsa110_continuum.conversion.*` entrypoint? Are there any CONTIMG_* env vars I need to set first? (Earlier `--help` printed `Filesystem path policy violation for TMPDIR=/tmp/dsa110-convert` — is that just a warning or will conversion fail?)
- (b) **Cal-acquisition routing**: with same-date BP/G bak'd, what does priority-2 do for `obs_dec_deg=16.2` on date `2026-02-15`? Specifically — does `_lookup_calibrator_coords` (`dsa110_continuum/calibration/ensure.py:647`) hit 3C138 first based on dec proximity? And does the resulting `generate_from_transit("3C138", ...)` (despite finding #4) actually pick the 03:34 transit MS, given the generator's date scope is implicit?
- (c) **Hour-03 windowing**: when `batch_pipeline.py` discovers MSs after my conversion run, it'll see 11 MSs in `[03:00, 04:00)`. Are 11 tiles enough for the mosaic builder, or does it require exactly 12? Memory note says "≈ ~12 sequential tiles" — review the actual mosaic builder code for any hardcoded count assumption.
- (d) **`--force-recal` + bak interaction**: with bak'd BP/G + `--force-recal`, will the per-epoch gain cache (`epoch_gaincal ap.G`) be cleared correctly? Or does `--force-recal` only clear it if BP/G are *present*?
- (e) **`--expected-dec 16.2`**: Default is 16.1° per `--help`. Real pointing is 16.188°. Threshold `DEC_CHANGE_THRESHOLD_DEG=5°`. Is 16.2 the right value to pass, or should I pass 16.19? Or omit the flag entirely and let the default 16.1 work (well within the 5° tolerance)?

Constraints:
- You are running with `-s workspace-write` for the `outputs/3c138_smoke_2026-05-05/` directory only. Do NOT modify code, only write the review file.
- Be terse. Cite `path:line`. Skip verdicts you can't ground in code.
- Print only the absolute path of the review + the one-sentence verdict to stdout when done.
