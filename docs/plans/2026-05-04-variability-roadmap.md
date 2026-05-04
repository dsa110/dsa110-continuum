# Variability-Science Roadmap — 2026-05-04

Converged roadmap from a three-round review cycle on how to get from the current
pipeline state to a trustworthy 22-date variability sweep. This file is the
pickup point for the next session. See `CLAUDE.md` and `docs/reference/` for
validated science constraints referenced below.

## One-line version

Fix calibration per-date, record its provenance, define what a trustworthy
product looks like *before* producing 22 dates of them, pilot 3 dates, then
sweep — and treat silent failures more harshly than noisy ones, because
variability science can survive noise but not corruption.

## Tiers

### Tier 1 — Smoke run (H17, single tile)

Plumbing-only validation. The post-smoke report must explicitly frame this as
proving end-to-end connectivity, **not** calibration stability, flux scale, or
mosaic suitability. Those are separate gates in later tiers.

### Tier 2 — Per-date gain calibration

Fix calibration-selection logic so same-date tables are preferred, fallbacks
are explicit, and provenance is recorded in every output.

**5-value calibration-provenance enum (lands in code before Tier 2.5):**

1. `SAME_DATE_VALID` — valid same-date table
2. `PRIMARY_TRANSIT_GENERATED` — model-anchored (`flux_anchor = perley_butler_primary`)
3. `BRIGHT_FALLBACK_GENERATED` — catalog-anchored (`flux_anchor = vla_catalog`)
4. `BORROWED_VALIDATED` — nearest validated table; requires strip-compatibility
5. `NO_VIABLE_TABLE` — fail loudly

**Acceptance criteria (hard requirements, not nice-to-haves):**

- Calibration provenance recorded per-output using the enum above.
- Same-date preferred and verified when available.
- Gain-quality summary per solve: median |gain|, phase RMS per antenna,
  flagged-antenna count, solution SNR, refant-stability across SPWs.
- **Two-gate dynamic-range comparison:**
  1. *Absolute floor* pinned on reference tile `3C454.3 @ 2026-01-25T22:26:05`
     (verified-working, 12.5 Jy/beam). First Tier 2 PR measures the delta and
     pins the number; later PRs regression-test against it.
  2. *Non-regression on a hard date*: same-date-cal must not be worse than
     borrowed-cal on at least one fallback-class or high-flag-fraction date.
     Prevents the easy reference tile from hiding regressions.

### Tier 2.5 — Product validation contract

Schema + writer + validator + one test. **Scope is time-boxed** — no
source-finding changes in the same PR. If it grows beyond that, defer the
extra surface to a follow-up.

**Schema required fields:**

- Calibration provenance (the 5-value enum above).
- `days_since_reference_cal` — numeric proxy for `flux_scale_confidence`.
  (Do not split `BORROWED_VALIDATED`; keep the enum discrete and the degradation
  continuous.)
- Image/mosaic QA: RMS, beam, FLAG fraction, number of contributing tiles,
  edge/depth masks.
- Source measurement metadata: flux, uncertainty, local RMS, position, epoch,
  mosaic ID.
- Rejection flags with `HARD | SOFT` severity tags.

**Rejection-flag taxonomy (with severities):**

| Flag                           | Severity | Why                                        |
|--------------------------------|----------|--------------------------------------------|
| `NO_VIABLE_TABLE`              | HARD     | No calibration at all                      |
| `BAD_CALIBRATION`              | HARD     | Structural calibration failure             |
| `DEC_STRIP_INCOMPATIBLE`       | HARD     | Wrong beam geometry; not weight-recoverable |
| `FIELD_PHASE_DIR_NOT_SYNCED`   | HARD     | Position-corrupted (CASA `ft()` reads `REFERENCE_DIR`, not `PHASE_DIR`) |
| `TELESCOPE_NAME_OVRO_AT_IMAGING` | HARD   | Wrong primary beam applied; flux-scale corrupted |
| `HIGH_FLAG_FRACTION`           | SOFT     | Noisy but measurable                       |
| `LARGE_BEAM`                   | SOFT     | Resolvable downstream                      |
| `EDGE_PROXIMITY`               | SOFT     | Geometric, weight-recoverable              |
| `LOW_COVERAGE`                 | SOFT     | Variance-weighted downstream               |

**Governing principle for the taxonomy:** *Silent failures from `CLAUDE.md`
are always HARD, because their failure mode is wrong-but-plausible output, not
noisy output.* Use this rule when new flags are added to keep the taxonomy
from drifting.

**Two-tier validator behaviour:**

- **HARD** → drop the product, record a loud reason in a run-summary sidecar
  manifest. Passing structurally-bad products through poisons downstream
  analysis.
- **SOFT** → flag-and-pass. The Mooley η/Vs/m variability metrics already
  handle variance via their variance terms. Silent dropping of SOFT failures
  creates selection bias *correlated with the astrophysical conditions the
  analysis is trying to detect* (RFI epochs, antenna outages, edge geometry).

**Companion change in the same PR:** pre-WSClean assertion that
`OBSERVATION::TELESCOPE_NAME == 'DSA_110'`. `set_ms_telescope_name()` already
runs inside `run_wsclean()`; this is defense-in-depth that the fix-call
actually took effect. The rejection flag remains as second-line catch for
products produced before the assertion existed.

### Tier 3 — Pilot → sweep → sources → variability

1. **Pilot (2–3 deliberately-chosen dates):** one easy same-date-cal, one
   fallback-class, one edge case. Not chronologically next.
2. **Calibration-stability gate lives inside the pilot.** Compute the
   per-antenna phase-RMS distribution *from the pilot itself*; flag tail
   outliers (>3σ from cohort median). Tolerance is an *output* of the pilot,
   not an input — prevents tuning-to-pass.
3. Full 22-date sweep only after the pilot passes.
4. Source finding.
5. Variability metrics (Mooley η/Vs/m).

### Tier 4 — Deferred: `dsa110_contimg` → `dsa110_continuum` migration

Not active. **Tripwire:** if any Tier 2 / 2.5 / 3 PR either

- adds a new `from dsa110_contimg…` import, **or**
- extends `_compat.py` by more than ~50 LOC for a single subsystem,

the migration is promoted to active. Without the tripwire, "deferred" silently
becomes "never."

## Ready-to-start PRs

Either can begin first; they are independent.

**Tier 2 PR**
- Gain solving
- Same-date selection logic
- 5-value fallback enum landed in calibration provenance
- DR floor measured and pinned on 3C454.3 reference tile
- Non-regression check on one hard date

**Tier 2.5 PR**
- Schema (Pydantic or dataclass)
- One writer (runner record-emitter)
- One validator with two-tier HARD/SOFT behaviour driven by per-flag severity
- `flux_scale_confidence` / `days_since_reference_cal` field
- Pre-WSClean telescope-name assertion
- One test

## Review-process note

Three rounds of review produced one substantive correction each:

1. Tier 2.5 (product validation contract) inserted between Tier 2 and Tier 3.
2. 5-value enum (not 4), falsifiable migration tripwire, pinned-on-measurement
   DR floor (not picked threshold).
3. Two-tier validator with HARD/SOFT severity — explicitly not a binary
   drop-or-pass global switch.

If the next review round produces zero substantive points, the right answer
is "ship the Tier 2.5 schema PR" rather than manufacturing more roadmap
detail. Implementation friction will surface the next set of real questions.
