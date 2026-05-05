# Codex review: 3C138 fundamentals-smoke plan for DSA-110 continuum pipeline

You are a senior reviewer. **Read-only.** Do not modify any files, run the pipeline, or create artifacts other than the review document this prompt asks for. Your job is to find logic gaps, false assumptions about code paths, and risks in the plan below before the user (Jakob, senior PhD at Caltech) launches a multi-hour orchestrated run.

## Repository context (durable)

- Repo: `/data/dsa110-continuum`. Canonical package: `dsa110_continuum/`. Branch: `main`.
- Python env: `/opt/miniforge/envs/casa6/bin/python` (CASA, casacore, casatools, pyuvdata, astropy). Do **not** invoke system `python3`.
- Pipeline DB: `/data/dsa110-contimg/state/db/pipeline.sqlite3`.
- VLA calibrator catalog DB: `/data/dsa110-contimg/state/catalogs/vla_calibrators.sqlite3`.
- Read `CLAUDE.md` (root) and `docs/reference/calibration.md` first; they encode the BP/G acquisition policy ladder, three silent-failure invariants, and the strict-QA defaults.

## Science goal of the plan

Confirm the *fundamental* radio-interferometry methods of the pipeline are in order via a single primary-anchored mosaic that contains the bandpass calibrator. This is a *smoke test* of the load-bearing chain — convert + flag + populate model + K/B/G solve + applycal + phaseshift + WSClean + mosaic — not a science run.

## Committed plan (do not relitigate the choice; review for soundness)

### Target

- **Source**: 3C138 (J2000 0521+166, RA 80.29°, Dec +16.64°). Perley-Butler primary; ~8.3 Jy at 1.4 GHz averaged across L-band 1.31–1.50 GHz BW.
- **Date**: 2026-01-25.
- **Transit (UTC)**: 04:57:50.359088 (per `daily_calibrator_transits` row, rank=2 for that date, dec_strip 16°).
- **Hourly bin**: hour 04 (UTC `[04:00, 05:00)`).
- **Cal anchor**: solve BP/G from scratch on this transit; do NOT replay the existing `_0~23.{b,g}` tables that live at `/stage/dsa110-contimg/ms/2026-01-25T22:26:05_0~23.{b,g}` (those were anchored at hour 22 / 3C454.3 / variable, not a primary).

### Dataset state (verified)

- 2026-01-25 has **2894 indexed HDF5 files** in `/data/incoming/` covering ~24 h, 181 distinct timestamps × 16 subbands.
- 2026-01-25 row in `daily_calibrator_transits`:
  - rank=1: 2253+161 (3C454.3) at 22:33:31 UTC
  - **rank=2: 0521+166 (3C138) at 04:57:50 UTC** ← our target
  - rank=3: 1230+123 (3C273) at 12:08:40 UTC
- `bandpass_calibrators` table is **empty** in the pipeline DB. This is a flagged risk — see Risks below.

### Pre-flight (cheap, non-mutating)

1. Confirm `_0~23.{b,g}` exist on disk where memory says.
2. Confirm 04:57:50 transit MS or 04:50–05:05 HDF5 group is on disk.
3. Confirm cal-acquisition path can identify 3C138 as a primary even with `bandpass_calibrators` empty.

### Cal-acquisition forcing (Q4b option i + force-recal)

- Move-aside (bak):
  ```bash
  mv /stage/dsa110-contimg/ms/2026-01-25T22:26:05_0~23.b{,.bak-pre-3c138-demo}
  mv /stage/dsa110-contimg/ms/2026-01-25T22:26:05_0~23.g{,.bak-pre-3c138-demo}
  ```
- Plus `--force-recal` flag to clear the `epoch_gaincal ap.G` cache and re-image every tile.

### Orchestration

```bash
/opt/miniforge/envs/casa6/bin/python /data/dsa110-continuum/scripts/batch_pipeline.py \
  --date 2026-01-25 --start-hour 4 --end-hour 5 \
  --force-recal --skip-photometry --keep-intermediates \
  --retry-failed --archive-all \
  --tile-timeout 1800 --quarantine-after-failures 3 \
  --dry-run    # first; then drop --dry-run for real run
```

- Default `--strict-qa` (cal-gate aborts on warn).
- `--archive-all` (so mosaic FITS lands on disk regardless of catalog-completeness gate).
- `--expected-dec` defaults to 16.1° per the orchestrator help — matches dec strip 16°.
- `--cal-date` not passed (defaults to `--date`).
- Real run launched via Bash `run_in_background=true`; harness fires one notification on exit.
- Tiles process **serially** (no `--workers`-style flag exists). 12 tiles × ~30 min worst-case = up to 6 h wall.

### Acceptance bar (5 checks, ±10% flux tolerance)

1. **Cal provenance**: dry-run + manifest show `flux_anchor=perley_butler_primary`, `selection_pool=primary`, derived from 3C138 04:57:50 transit. Hard gate.
2. **Pipeline completes without exception**, run_report.md emitted, manifest present. Hard gate.
3. **Mosaic FITS exists** at `/stage/dsa110-contimg/images/mosaic_2026-01-25/2026-01-25T0400_mosaic.fits` (or equivalent single-strip naming). Hard gate.
4. **3C138 detected in mosaic** at `(80.29°, +16.64°)`, peak within **±10% of ~8.3 Jy/beam** (catalog-anchored expected value, beam-corrected). Hard gate at ±10%, warn at ±20%.
5. **No silent-failure-invariant violations** in run log: no `PHASE_DIR mismatch`, no `MODEL_DATA at wrong position`, no `TELESCOPE_NAME=OVRO_MMA at WSClean`. Soft gate (these are silent by definition; we may need to instrument).

### Failure-mode protocol (committed)

- Dry-run lands on bright_fallback / borrows from another date → stop, regrill.
- Real run cal-gate fails → orchestrator self-aborts; restore bak'd tables; regrill.
- 1–3 tiles fail after retry → continue; mosaic of 9–11 still meets check 3.
- ≥4 tiles fail → stop demo.
- 3C138 flux in 10–15% deviation → soft fail; investigate; record DEGRADED.
- 3C138 flux >15% → hard fail; abort demo.
- Silent-failure-invariant grep finds a hit → hard fail.

### Cleanup on success

- Restore bak'd `_0~23.{b,g}`.
- Copy mosaic + manifest + run_report.md into `outputs/3c138_smoke_2026-05-05/`.
- Update `~/.claude/projects/-data-dsa110-continuum/memory/MEMORY.md` with provenance.

## Known risks already on my radar (review for completeness, but focus on what I might be missing)

1. **`bandpass_calibrators` table empty** — if the cal-acquisition policy ladder uses this table to identify "primary" sources, priority-2 won't fire and the run will fall through to bright_fallback or borrow. Verify by reading `dsa110_continuum/calibration/runner.py`, `caltables.py`, `presets.py`, `model.py::populate_model_from_catalog`, and any `flux_anchor` / `selection_pool` decision code. Where does the orchestrator decide a transit is "primary"? Is it J2000-name table, hardcoded list, VLA catalog quality codes, or `daily_calibrator_transits.rank`?
2. **`--force-recal` semantics** — help text says it clears the `epoch_gaincal ap.G` cache and re-images tiles, but does it actually re-derive **BP/G** (the K/B/G chain on a *new transit*), or only re-derive the per-epoch G? Read the relevant code path.
3. **MS for 3C138 transit may not be pre-converted** — the 04:57:50 hour requires HDF5 → MS conversion. The orchestrator handles this, but: does it know to convert the *transit-windowed* MS or the *hourly-binned* MSs? `--start-hour 4 --end-hour 5` selects MSs whose timestamps fall in [04:00, 05:00). What governs which transit MS the orchestrator picks for the BP/G solve?
4. **Path policy violations on stdout** — I observed `Filesystem path policy violation for TMPDIR=/tmp/dsa110-convert` printed during `--help`. Is this only a warning, or will the conversion stage refuse to run? Check `CONTIMG_ALLOWED_ROOTS / CONTIMG_FORBIDDEN_ROOTS` env handling.
5. **Mosaic output naming for hour 04** — memory says hour 22 produced `2026-01-25T2200_mosaic.fits`. By symmetry hour 04 should be `2026-01-25T0400_mosaic.fits`. Confirm the format string in `mosaic/builder.py` or `batch_pipeline.py`.
6. **Pre-existing tile FITS / checkpoint state** — the date may have prior run artifacts (checkpoints, partial tiles) from previous sessions that `--force-recal` should clean but might miss edge cases.
7. **Quarantine state for 2026-01-25 MSs** — `--quarantine-after-failures 3` means an MS that failed 3 times in prior runs is skipped. Need to confirm hour 04 MSs are not pre-quarantined.

## What I want from this review

Write a markdown file at `/data/dsa110-continuum/outputs/3c138_smoke_2026-05-05/codex_plan_review.md` with these sections:

1. **Verdict**: GO / GO-WITH-CHANGES / NO-GO. One sentence.
2. **Risks I flagged that hold up under code reading**: for each, cite file:line and one-line summary of what the code actually does.
3. **Risks I flagged that don't hold up**: for each, cite file:line and one-line "why this isn't actually a risk."
4. **Risks I missed**: anything else the code reading turns up that would block or compromise the plan.
5. **Concrete plan-modifying recommendations**: bullet list, each one specific enough to incorporate without re-discussion (e.g., "add `--cal-date 2026-01-25` to force same-date acquisition lookup," or "before bak'ing tables, also `mv` the `*_meridian.ms` files because they cache phaseshift state").
6. **Confidence level**: high / medium / low, with one sentence on what would push it to high.

When you finish, **print to stdout**: the absolute path of the review file + the one-sentence verdict. Nothing else.

## Operational constraints

- You are running with `-s read-only`. Do not edit, write, or run pipeline code. The only file you may create is the review markdown above.
- Cite file paths with line numbers (`path:line`) wherever possible.
- Do not relitigate the high-level plan choices (3C138 vs 3C48, hour 04 vs hour 11, batch_pipeline vs hand-rolled). Those are committed. Review them only for *technical soundness*, not for *strategic preference*.
- Be terse. The user prefers concision over completeness when there's tension.
