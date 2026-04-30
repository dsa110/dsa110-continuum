# Pipeline validation from scratch

This spec replaces the implicit "trust 2026-01-25 hour 22" validation story with
an explicit, two-layer process for deciding when a `(date, hour)` window from
`scripts/batch_pipeline.py` should be trusted as a science baseline. It is the
canonical reference for any future operator who needs to bring a new date into
production without inheriting unverified assumptions from prior runs.

The companion artifacts in `outputs/validation-from-scratch/` apply the
methodology defined here to the dates available on this host as of the spec's
authoring date.

## Audience and scope

- **Operators** who need to choose `(date, hour)` for a regression run and decide
  whether the result earns "trusted baseline" status.
- **Maintainers** who need to evaluate whether a code change preserves
  pipeline behavior on previously-promoted hours.

In scope: `scripts/batch_pipeline.py` end-to-end orchestration (`conversion`
results assumed present) → calibration → imaging → mosaic → photometry, plus
the manifest / run report / checkpoint artifacts produced under the products
directory. Out of scope: the `dsa110_contimg` import-cleanup roadmap, the
self-cal subsystem, and any infrastructural rework of the dry-run output
format itself (see _Future enhancements_).

## Two-layer validation model

A `(date, hour)` window participates in **two independent checklists**, each
serving a different purpose. The intent of separating them is to keep day-to-day
engineering cheap while making "this hour is science-ready" a deliberate,
auditable decision.

### Layer 1 — Per-run regression (consistency)

Used after any code change, dependency upgrade, or operational re-run on a
window whose prior outcome is recorded in version control or on disk. Compares
the *kind* and *class* of artifacts and gates against the recorded baseline,
not byte-for-byte equality.

A regression run **passes** when every row in the per-run regression checklist
matches the recorded baseline. A run that exits 0 but produces a different
gate class, a different cal-table provenance, or a new quarantine entry **does
not pass** — exit code is necessary but never sufficient.

### Layer 2 — Promotion to trusted baseline

Used when an operator wants to *bless* a new `(date, hour)` window as a
reference others can compare against. Stricter than the regression layer and
explicitly evaluated against first-principles criteria, not against a prior
run's record.

A window **earns trusted-baseline status** only when every promotion checklist
item passes for documented reasons. A window that fails any item may still be
recorded as **comparator-only** (with the failing item named) — this gives the
operator a partial baseline to compare against without claiming science
correctness.

### Why separate them

- A regression run on a previously-promoted window proves "the code did not
  break this," not "this window is correct" — the latter is established at
  promotion time.
- A promotion candidate on a never-before-run window has no prior record to
  compare against — only first-principles criteria apply.
- Collapsing the layers means either (a) every code change re-litigates
  scientific correctness (slow, expensive) or (b) promotion silently degrades
  to "matches yesterday's run, which matched the day before, …" — the
  privileged-canary problem this spec exists to fix.

## Promotable unit

The **promotable unit is `(date UTC, hour 0–23)`**, matching:

- `scripts/batch_pipeline.py --start-hour H --end-hour H+1` (per-epoch boundary),
- mosaic filename convention `<date>T<HH>00_mosaic.fits` under
  `/stage/dsa110-contimg/images/mosaic_<date>/`,
- per-epoch QA gates in the manifest produced under
  `<products>/mosaics/<date>/<date>_manifest.json`.

A **date is never promoted as a unit**; constituent hours are promoted
independently. A date enters the operational queue when at least one of its
hours has been promoted.

A **contiguous-hour RA strip** is a *reporting view* that rolls up consecutive
promoted hours along a drift track. It is not a separate gate; it has no
distinct manifest entry and is not used as a promotion key.

The canonical key in tables and side-car files is `YYYY-MM-DD` + `HH` (00–23).
The matching mosaic-filename label is `YYYY-MM-DDTHH00`.

## Calibration policy axes

Daily-table provenance and per-epoch gain outcome are **independent axes**.
Both must be recorded to characterize a window honestly, and trusted-baseline
status requires both to be at their cleanest values.

### Axis 1 — Daily BP/G provenance tier

Aligned with the calibration table policy in `CLAUDE.md` ("Calibration tables —
Operational policy for obtaining BP/G tables"). Three tiers, in decreasing
order of cleanliness:

| Tier | Source                                                              | Eligible for trusted baseline? |
|------|---------------------------------------------------------------------|--------------------------------|
| A    | Same-date BP/G generated on this date's calibrator MS               | Yes                            |
| B    | Same-date BP, daily G borrowed from a strip-compatible donor date   | No — comparator only           |
| C    | Both BP and G borrowed from a strip-compatible donor date           | No — comparator / diagnostic only |

Tier-B and tier-C runs are not invalid; they remain useful for diagnostic and
operational purposes. Their exclusion from "trusted baseline" reflects the
reality that borrowed cal cannot anchor a *new* science claim — only repeat one
the donor already established.

The tier label is determined by inspecting the resolved BP/G paths in the
dry-run plan output. Tier A requires the BP/G filenames to start with the same
date as `--date`; otherwise the run is tier B (G differs) or tier C (both
differ).

### Axis 2 — Per-epoch gaincal outcome

The per-epoch gaincal runs as Phase 0 of `batch_pipeline.py`. Its outcome on
each `(date, hour)` window must be recorded as one of:

| Value                                | Meaning                                                                                              | Eligible for trusted baseline? |
|--------------------------------------|------------------------------------------------------------------------------------------------------|--------------------------------|
| `solved`                             | Phase 0 produced a valid `_meridian.ms` epoch G table for this hour                                  | Yes                            |
| `fell_back_to_static_with_reason`    | Phase 0 detected a code-path failure and silently used the static daily G; the reason is captured   | No — bug/code path; investigate |
| `skipped_intentionally`              | Operator passed `--skip-epoch-gaincal`                                                               | No — comparator only           |
| `skipped_or_failed_low_snr`          | Phase 0 attempted but legitimately could not solve due to data quality (flagged baselines, low S/N) | No — astrophysical/operational limit |

`fell_back_to_static_with_reason` is the only state that signals a defect. The
other non-`solved` states record legitimate operational decisions and must not
be conflated with the defect state.

### Trusted-baseline rule

`promotion_class = "trusted_baseline"` requires:

```
daily_cal_tier == "A"  AND  epoch_gaincal_state == "solved"
```

Any other combination yields:

- `promotion_class = "comparator_only"` if any other axis value is recorded but
  the window passed all dataset-completeness and gate checks; or
- `promotion_class = "weak_baseline"` if only a tier-3 photometric anchor (see
  *Photometric anchors* below) was available even though tier-A cal and
  `solved` epoch gaincal both passed.

## Photometric anchors

Each promotion record must name **which independent comparator** the window's
photometry was checked against. There are three tiers, evaluated in order; the
first that fires earns its label, but tier-3 is always evaluated as a sanity
floor when the pipeline exposes it.

| Tier | Anchor                                       | Notes                                                                                  |
|------|----------------------------------------------|----------------------------------------------------------------------------------------|
| 1    | Named primary calibrator in the coadd        | Only when geometry actually places the source in the mosaic for this `(date, hour)`. Do not assert from documentation tradition — verify from the mosaic FITS WCS or coordinates. |
| 2    | Catalog cross-match (NVSS / RACS) median     | Use ≥ N matched sources; record N, the catalog used, the median ratio, and the robust scatter. |
| 3    | Tile self-consistency across overlap regions | Same source seen in adjacent tiles must agree within Y%. Required floor for any promotion class. |

The `anchor` field in the side-car JSON (see *Promotion record schema*)
records all three tier slots so a future reader can see which fired and which
did not. A window where tier-1 fails (no named primary in field), tier-2 fires
weakly (small N), but tier-3 passes earns `promotion_class = "weak_baseline"`
even if tier-A cal and `solved` epoch gaincal both passed.

## Dataset completeness pre-gate

**Before** any photometric tier is evaluated, the window must pass a
dataset-completeness pre-gate that checks the *raw inputs* were intact. This
catches the failure mode where partial-subband or zero-byte HDF5 files silently
narrow the band, producing a "successful" run on degraded data.

Pre-gate items (any failure → window is **not eligible** for promotion at any
class; remains in the evidence table as `eligible_for_trusted_baseline=false`
with reason recorded):

1. Every MS in the hour window has all 16 subbands' worth of channels (no
   incomplete-subband neighbors per `scripts/inventory.py`).
2. No zero-byte HDF5 placeholder files in the hour window.
3. Tile count after corrupt-MS filtering ≥ 8.
4. No quarantined MS in the hour window
   (`failure_count >= --quarantine-after-failures`, default `3`).
5. Declination strip matches `--expected-dec` (default 16.1°) within
   `DEC_CHANGE_THRESHOLD_DEG` (5°). The dry-run plan flags a mismatch
   explicitly; treat that flag as a hard fail.

The 8-tile floor is empirical: legacy `2026-01-25 hour 22` ran with 11 tiles,
and most candidate hours on this host show 8–12 tiles per hour after corrupt
filtering. If the dry-run sweep on a host shows no candidate ever reaches 10
tiles, raise the floor to whatever empirical 25th percentile is documented in
`outputs/validation-from-scratch/evidence.md` and update this section.

## Per-run regression checklist

For a `(date, hour)` window with a recorded prior baseline. Compare each item
below to the recorded values; any divergence requires written justification or
fails the regression.

1. **Same arguments**: `--date`, `--cal-date`, `--expected-dec`,
   `--start-hour`, `--end-hour`, `--skip-epoch-gaincal`, `--force-recal`,
   `--quarantine-after-failures` match the recorded
   `batch_pipeline_invocation`.
2. **Same daily cal tier**: resolved BP/G filenames produce the same
   `daily_cal_tier ∈ {A, B, C}` as the recorded baseline.
3. **Same epoch gaincal state**: `epoch_gaincal_state` matches the recorded
   value. A change from `solved` → `fell_back_…` is a regression. A change in
   the *opposite* direction (a previously-failing hour now solves) is **not**
   a regression but is a *promotion candidate* — re-evaluate under Layer 2.
4. **Same gate classes**: every QA gate's class
   (`PASS / WARN / FAIL / DEGRADED / SKIP`) matches the recorded baseline. A
   gate that was `WARN` becoming `FAIL` is a regression even if the underlying
   metric value drifted only slightly.
5. **Same `manifest_verdict`**: `CLEAN`, `DEGRADED`, or other terminal verdict
   matches the recorded baseline.
6. **No new quarantine entries**: any MS in the hour window that newly enters
   `quarantine_state` is a regression unless an operator note documents the
   underlying corrupt-MS event.
7. **Tile count is within tolerance**: ±1 tile from the recorded value (allows
   for transient inventory drift); >1 difference is a regression.
8. **Mosaic file present at the same path**: a missing or relocated mosaic is
   a regression even when all gates pass.
9. **Side-car JSON present and valid**: schema matches; required fields are
   non-null.
10. **Run report renders**: `run_report.md` exists and references the same
    artifact paths as the baseline.

Exit code 0 alone is **not** sufficient. A run that produces an empty mosaic,
a quarantine cascade, or a different cal tier — but exits cleanly — fails the
regression checklist.

## Promotion-to-trusted-baseline checklist

For a `(date, hour)` window being evaluated as a *new* baseline. Every item is
mandatory. Items that fail downgrade the promotion class but do not invalidate
the entire record — the operator records the failure reason, not "blocked."

1. **Dataset completeness pre-gate** passes (all 5 items above).
2. **`daily_cal_tier == "A"`** for `promotion_class = "trusted_baseline"`.
   Tier B or C records the window as `comparator_only` with reason.
3. **`epoch_gaincal_state == "solved"`** for `promotion_class =
   "trusted_baseline"`. Other states record the window as `comparator_only`
   with the gaincal state as the reason.
4. **All QA gates** in `qa/run_report.py`'s output are either `PASS` or have
   a documented `WARN`/`FAIL` reason that the operator explicitly accepts.
5. **Mosaic FITS exists** at the expected path and opens cleanly (header valid,
   data array non-empty, beam parameters present).
6. **At least one photometric anchor tier passes**:
   - Tier 1 (named primary) if geometry permits.
   - Tier 2 (catalog xmatch) with N ≥ 10 and median residual ratio within
     20% of unity.
   - Tier 3 (tile self-consistency) with median disagreement ≤ 10% across
     overlapping tiles.
7. **`pipeline_verdict`** at promotion time is recorded (typically `CLEAN`;
   `DEGRADED` is acceptable for `comparator_only` only).
8. **Side-car JSON written** to the products directory; markdown ledger row
   appended.
9. **`git_sha`** of this repo at promotion time is recorded.
10. **Operator name and notes** field captures the human decision context.

## Anti-patterns

The following patterns must not appear in validation work or be cited as
evidence of a window's trustworthiness.

- **"Exit 0 alone."** A successful batch_pipeline.py exit means the orchestrator
  did not crash. It says nothing about whether the per-tile imaging succeeded,
  whether the cal path was the intended one, or whether the mosaic contains
  meaningful data. Always check `manifest_verdict`, gate classes, and artifact
  presence.
- **"Single canonical date worship."** Treating one historically-cited
  `(date, hour)` as the privileged baseline for all comparisons. The legacy
  `2026-01-25 hour 22` was originally chosen because the calibration tables
  happened to be generated for that meridian transit, not because it had been
  scientifically promoted. Under this spec it is at most a *historical
  comparator*, never automatically a *trusted baseline*.
- **"Mixing calibration policies across comparisons."** A regression of run
  X (tier-A, `--cal-date == --date`) against run Y (tier-C, `--cal-date`
  pointing to a borrowed donor) is meaningless. The regression checklist's
  item 1 ("same arguments") and item 2 ("same daily cal tier") exist to
  prevent this.
- **"Promoting a date as a unit."** A date with one promoted hour is not a
  promoted date; the other hours retain their own status. A regression that
  passes on hour 22 but fails on hour 02 is a partial regression, not a date
  regression.
- **"Treating a borrowed-cal mosaic as a science product."** Tier-B and
  tier-C runs are operational and diagnostic; their photometry must not be
  cited as variability or transient detections.
- **"Inferring science success from operational symbols."** A green
  `pipeline_verdict=CLEAN` does not imply the pixel values are scientifically
  correct; it implies all gates the orchestrator knows about passed.
  Promotion is the gate that bridges operational success to scientific trust.

## Selection methodology — how to choose candidate windows

This methodology produces the **ranked shortlist** in
`outputs/validation-from-scratch/evidence.md` and is the canonical procedure
for selecting future regression and promotion candidates.

### Step 1 — Inventory the date axis

```
/opt/miniforge/envs/casa6/bin/python scripts/inventory.py
```

The output enumerates dates with raw HDF5 in `/data/incoming/` and reports
per-timestamp completeness. Cross-check against `/stage/dsa110-contimg/ms/` to
identify which dates have staged MS (a date with raw HDF5 but no MS is a
*conversion-blocked* candidate; document but do not run dry-run).

### Step 2 — Cross-reference cal-table availability

Same-date BP/G tables typically appear at
`/stage/dsa110-contimg/ms/<date>T22:26:05_0~23.{b,g}`. Their presence
determines tier-A eligibility. Their absence means any candidate must rely on
borrowed tables (tier B or C) or on a calibrator-MS pass first.

### Step 3 — Classify dates into pool roles

| Class               | Definition                                                                      | Used in shortlist?                          |
|---------------------|---------------------------------------------------------------------------------|---------------------------------------------|
| `candidate`         | Staged MS present + same-date BP/G present + no prior mosaic dir                | Yes — primary shortlist source              |
| `control`           | Staged MS present + same-date BP/G present + prior mosaic dir present           | Documented; included only as cross-check    |
| `blocked_conversion`| Raw HDF5 present + no staged MS                                                 | Documented; not eligible until conversion runs |
| `anomaly`           | Staged MS present without raw HDF5 context                                      | Documented; never a candidate               |

### Step 4 — Run dry-run sweep

For each `candidate` and `control` date, run:

```
/opt/miniforge/envs/casa6/bin/python scripts/batch_pipeline.py \
  --date $DATE \
  --dry-run \
  --quarantine-after-failures 3 \
  > outputs/validation-from-scratch/dry_runs/dry_run_$DATE.log 2>&1
```

The dry-run plan emits a clearly-marked block:

```
=== DRY RUN — DSA-110 batch_pipeline ===
Date:           ...
Cal date:       ...
Obs Dec:        ...°
Stage dir:      ...
Products dir:   ...
Cal tables (BP): <path>  [exists | missing]
Cal tables (G):  <path>  [exists | missing]
MS files (post-filter): N
Epoch hours: [H1, H2, ...]
Prior manifest: absent | present
Checkpoint: completed=N  failed=N
Quarantine: N MS (threshold=N)
Resume plan:
  hour H: rebuild | skip (...)
Phase 0 (gaincal):    would run | would skip
Phase 1 (per-tile):   would attempt N tiles (Q quarantined)
Phase 2 (mosaic):     R rebuild, S skip
Phase 3 (photometry): would run; QA gating=strict | lenient
Pipeline NOT executed (--dry-run set). No products written.
```

These lines are the parseable basis for the shortlist; the noise tokens
counted by the rank function appear above this block.

### Step 5 — Apply hard disqualifiers

A candidate `(date, hour)` is **not eligible** for the primary shortlist if any
of:

- `tile_count_after_filter < 8` for the hour;
- any incomplete-subband or zero-byte MS appears in the hour ± 1;
- any MS in the hour shows `failure_count >= 3`
  (matching `--quarantine-after-failures` default);
- the dry-run plan flags a `--expected-dec` strip mismatch;
- same-date BP/G is absent → exclude from primary shortlist but **keep the
  row** with `eligible_for_trusted_baseline = false` and the reason recorded.

### Step 6 — Rank qualified candidates

Sort ascending (lower = better) by:

1. Count of dry-run noise tokens in the log: `WARNING`, `Skipping corrupt MS`,
   `read_ms_dec: MS read failed`, `Could not determine observed Dec`,
   `Resume plan: ... rebuild`, `mismatch`, `quarantine`.
2. Count of incomplete or corrupt MS in the hour ± 1 (proxy for local
   observing trouble).
3. Negative tile count after filter (more = better).
4. Distance from hour 22 (anti-legacy bias) — **tiebreak only**.

### Step 7 — Apply anti-legacy bias to the #1 slot

The **#1 recommendation** must have `hour != 22` if any qualified non-22 hour
exists in the candidate pool. If only hour-22 hours qualify, #1 = best
hour-22, and the evidence file explicitly notes "no non-22 passed floor."

#2 and #3 slots are unrestricted; a legitimate hour-22 candidate may legitimately
appear there.

### Step 8 — Emit the shortlist

Top-3 `(date, hour)` recommendations with explicit `recommendation_role`:

- `primary_regression` — #1, must be `hour != 22` when possible.
- `legacy_isomorphic_alt` — first hour-22 entry in top-3, if any.
- `diversity_alt` — #3 if it lands on a different date than #1, else omitted
  with a note. (If diversity is awkward to satisfy, omit and note it as
  follow-up work.)

## Promotion record schema

Every promoted `(date, hour)` window writes a **side-car JSON** in the products
directory and an **append-only row** in the repo's promotion ledger.

### Side-car JSON

Path (site-neutral; adjust to local site policy if products root differs):

```
<products_root>/mosaics/<date>/promotion_<date>T<HH>00.json
```

On this host the products root resolves to `/data/dsa110-proc/products/`, so
the canonical path is:

```
/data/dsa110-proc/products/mosaics/<date>/promotion_<date>T<HH>00.json
```

Schema (all keys required unless noted; nullable values allowed where the
field's tier did not fire):

```json
{
  "date": "YYYY-MM-DD",
  "hour": 0,
  "promoted_at_utc": "YYYY-MM-DDTHH:MM:SSZ",
  "git_sha": "<short sha of this repo at promotion time>",
  "operator": "<name or handle>",
  "notes": "<free text — context the schema does not capture>",

  "manifest_path": "<absolute path to <date>_manifest.json>",
  "run_log_path": "<absolute path to run_<utc>.log>",
  "run_summary_path": "<absolute path to <date>_run_summary.json>",
  "mosaic_path": "<absolute path to <date>T<HH>00_mosaic.fits>",

  "batch_pipeline_invocation": [
    "/opt/miniforge/envs/casa6/bin/python",
    "scripts/batch_pipeline.py",
    "--date", "YYYY-MM-DD",
    "--cal-date", "YYYY-MM-DD",
    "--start-hour", "H",
    "--end-hour", "H+1",
    "--quarantine-after-failures", "3",
    "..."
  ],
  "manifest_verdict": "CLEAN | DEGRADED | ...",
  "pipeline_verdict":  "CLEAN | DEGRADED | ...",

  "daily_cal_tier": "A | B | C",
  "cal_provenance": {
    "bp": "<absolute path to .b table>",
    "g":  "<absolute path to .g table>",
    "borrowed_from": null | "YYYY-MM-DD"
  },
  "epoch_gaincal_state": "solved | fell_back_to_static_with_reason | skipped_intentionally | skipped_or_failed_low_snr",
  "epoch_gaincal_reason": null | "<short text>",

  "anchor": {
    "primary_model": null | "<source name e.g. 3C286>",
    "catalog_xmatch": null | {
      "catalog": "nvss | racs",
      "n": 0,
      "median_ratio": 0.0,
      "robust_scatter": 0.0
    },
    "tile_self_consistency": null | {
      "n_overlaps": 0,
      "median_disagreement_pct": 0.0
    }
  },

  "eligible_for_trusted_baseline": true,
  "eligible_for_trusted_baseline_reason": null | "<short text when false>",
  "promotion_class": "trusted_baseline | comparator_only | weak_baseline"
}
```

### Markdown ledger

Path: `docs/validation/promotion-log.md`. One append-only table; one row per
promoted hour; oldest at top. Required columns:

| date       | hour | class             | tier | epoch_gc | anchor                       | side-car (relative-to-products-root)              | git_sha |
|------------|------|-------------------|------|----------|------------------------------|---------------------------------------------------|---------|
| YYYY-MM-DD |  HH  | trusted_baseline  |  A   | solved   | catalog_xmatch_nvss_n=27     | mosaics/YYYY-MM-DD/promotion_YYYY-MM-DDTHH00.json | <sha>   |

The ledger is the human-discoverable record; the side-car JSON is the
machine-readable source of truth.

## Operator checklist

Each step has an **intent** (one line) and a **literal command** (copy-pastable;
substitute the placeholders).

### Preflight (before any compute)

1. **Intent**: confirm the casa6 conda env is the active interpreter.
   ```
   /opt/miniforge/envs/casa6/bin/python -c 'import casacore, numpy, astropy; print("ok")'
   ```

2. **Intent**: enumerate dates with raw HDF5 and per-date completeness.
   ```
   /opt/miniforge/envs/casa6/bin/python scripts/inventory.py
   ```

3. **Intent**: confirm same-date BP/G tables for the candidate date.
   ```
   ls /stage/dsa110-contimg/ms/${DATE}T22:26:05_0~23.{b,g}
   ```

4. **Intent**: confirm staged MS files exist for the date.
   ```
   ls /stage/dsa110-contimg/ms/${DATE}T*.ms 2>/dev/null | wc -l
   ```

### Dry-run (before any real compute)

5. **Intent**: produce the plan summary and capture noise tokens for the rank function.
   ```
   /opt/miniforge/envs/casa6/bin/python scripts/batch_pipeline.py \
     --date ${DATE} \
     --dry-run \
     --quarantine-after-failures 3 \
     > outputs/validation-from-scratch/dry_runs/dry_run_${DATE}.log 2>&1
   ```

6. **Intent**: confirm the plan block is present and identify the eligible hours.
   ```
   awk '/=== DRY RUN/,/Pipeline NOT executed/' \
     outputs/validation-from-scratch/dry_runs/dry_run_${DATE}.log
   ```

7. **Intent**: count corrupt-MS warnings (high counts indicate the candidate is risky).
   ```
   grep -c 'Skipping corrupt MS' outputs/validation-from-scratch/dry_runs/dry_run_${DATE}.log
   ```

8. **Intent**: rerun the dry-run bounded to the chosen hour to see the per-hour resume plan.
   ```
   /opt/miniforge/envs/casa6/bin/python scripts/batch_pipeline.py \
     --date ${DATE} \
     --start-hour ${H} --end-hour $((H+1)) \
     --dry-run \
     --quarantine-after-failures 3
   ```

### Real run (only after dry-run passes)

9. **Intent**: execute the per-hour batch with default-strict QA, retry-on-fail, parallel photometry.
   ```
   /opt/miniforge/envs/casa6/bin/python scripts/batch_pipeline.py \
     --date ${DATE} \
     --cal-date ${DATE} \
     --start-hour ${H} --end-hour $((H+1)) \
     --quarantine-after-failures 3 \
     --tile-timeout 1800 \
     --retry-failed \
     --photometry-workers 4 \
     --photometry-chunk-size 0
   ```

### Post-run artifact review

10. **Intent**: confirm the run wrote all four expected artifacts.
    ```
    ls /data/dsa110-proc/products/mosaics/${DATE}/{${DATE}_manifest.json,${DATE}_run_summary.json,run_report.md,run_*.log}
    ```

11. **Intent**: confirm the mosaic FITS exists at the expected path.
    ```
    ls /stage/dsa110-contimg/images/mosaic_${DATE}/${DATE}T${H}00_mosaic.fits
    ```

12. **Intent**: read the manifest's verdict and gate classes.
    ```
    /opt/miniforge/envs/casa6/bin/python -c "import json,sys; \
      m=json.load(open('/data/dsa110-proc/products/mosaics/${DATE}/${DATE}_manifest.json')); \
      print('verdict:', m.get('pipeline_verdict')); print('gates:', list(m.get('gates', {}).items()))"
    ```

13. **Intent**: identify the daily cal tier from the resolved BP/G paths.
    ```
    grep -E 'Cal tables \(BP|G\)' /data/dsa110-proc/products/mosaics/${DATE}/run_*.log | head -2
    ```

14. **Intent**: identify the epoch gaincal state (solved vs fell-back).
    ```
    grep -E 'epoch_gaincal|FALLBACK|fell_back|gaincal' /data/dsa110-proc/products/mosaics/${DATE}/run_*.log
    ```

### Promotion (only after the operator decides to bless the window)

The pipeline auto-emits a side-car JSON and appends a ledger row at end-of-run
(via `dsa110_continuum.qa.promotion.emit_for_run`). The auto-emitted record
sets `promotion_class = "auto_emitted_pending_review"` until the operator
evaluates an anchor. Operator promotion steps:

15. **Intent**: read the auto-emitted side-car at
    `/data/dsa110-proc/products/mosaics/${DATE}/promotion_${DATE}T${H}00.json`
    and confirm `daily_cal_tier`, `epoch_gaincal_state`, and
    `eligible_for_trusted_baseline` match expectation.

16. **Intent**: evaluate the photometric anchor for this window. Fill in the
    `anchor.primary_model` field if a named primary calibrator is in the
    coadd, or the `anchor.catalog_xmatch` block (NVSS / RACS) using
    `scripts/forced_photometry.py` results, or the
    `anchor.tile_self_consistency` block if neither tier 1 nor tier 2 is
    available.

17. **Intent**: finalize `promotion_class` in the side-car. If
    `eligible_for_trusted_baseline=true` and any anchor tier passes, set to
    `trusted_baseline`. Otherwise set to `comparator_only` or `weak_baseline`
    per the spec rules.

18. **Intent**: re-render the ledger row from the finalized side-car (the
    auto-appended ledger row is overwritten or amended once the side-car
    promotion class changes from `auto_emitted_pending_review`).

## Future enhancements (out of scope for this spec)

These are natural next implementation tasks that automate or extend the spec.

1. **Pipeline-emitted promotion record (IMPLEMENTED).** `dsa110_continuum.qa.promotion`
   now auto-emits a side-car JSON per `(date, hour)` at end-of-run and appends a
   row to `docs/validation/promotion-log.md`. Calibration tier and epoch
   gaincal state are derived from the manifest; the photometric anchor block
   is left null and the auto-emitted record carries
   `promotion_class = "auto_emitted_pending_review"` until the operator
   evaluates an anchor and finalizes the class. The original
   "today: operator-authored, future: pipeline-emitted" framing has thus
   collapsed: operator action is now limited to the anchor evaluation step.
2. **Dry-run plan as machine-readable JSON**: today the plan summary is
   parseable text; emitting it as JSON alongside the human-readable lines
   would eliminate fragile regex-based parsing.
3. **Tile self-consistency tooling**: the tier-3 photometric anchor requires
   a tool that compares per-source flux across overlapping tiles within a
   window. `scripts/forced_photometry.py` is the natural extension point.
4. **Catalog xmatch as a first-class gate**: tier-2 photometric anchor
   evaluation should run automatically post-mosaic and surface as a manifest
   gate, populating `anchor.catalog_xmatch` in the auto-emitted side-car so
   the operator's only remaining task is the geometry check for tier-1.
5. **Conversion-blocked discovery**: a date with raw HDF5 but no staged MS
   should produce a structured "needs conversion" entry the operator can act on
   without re-discovering it via inspection.

## Companion artifacts

- `outputs/validation-from-scratch/evidence.md` — human-readable evidence
  table and the ranked top-3 shortlist applying this methodology to dates
  available on this host as of the spec's authoring date.
- `outputs/validation-from-scratch/candidates.json` — machine-readable
  full candidate table.
- `outputs/validation-from-scratch/dry_runs/` — raw dry-run logs captured by
  the sweep, one file per evaluated date.
