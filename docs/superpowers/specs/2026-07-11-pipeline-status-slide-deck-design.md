# Design: DSA-110 continuum pipeline status slide deck

**Date:** 2026-07-11  
**Audience:** science collaborators  
**Length:** ~20 slides  
**Tone:** balanced (celebrate validated pieces; blockers are first-class)  
**Approach:** pipeline walkthrough (Approach A)

## Goal

A shareable HTML slide deck that answers: *Where does the dsa110-continuum imaging pipeline stand today, and can collaborators use products for science yet?*

Reference epoch for concrete evidence: **2026-01-25T2200** Dec-strip mosaic.

## Constraints

- Use Poimandres HTML template (same family as `outputs/astrometry-2026-01-25/slides/`).
- Vocabulary from `CONTEXT.md`: *tile*, *hourly-epoch mosaic*, *Dec strip* (not “daily mosaic”, “snapshot”, or “RA strip”).
- No new pipeline runs; reuse existing diagnostic PNGs/JSON under `outputs/`.
- Do not ship MS/FITS in the deck folder — figures and HTML only.
- Do not claim photometry/ESE is production-validated on the reference epoch.
- Explicit takeaway: imaging + astrometry look credible; **variability science products are not unblocked yet** under strict QA.

## Deliverables

| Path | Role |
| --- | --- |
| `outputs/pipeline-status-2026-07-11/spec.md` | Slide markdown source (`* * *` separators) |
| `outputs/pipeline-status-2026-07-11/slides/index.html` | Interactive deck |
| `outputs/pipeline-status-2026-07-11/slides/*-standalone.html` | Single-file shareable (images inlined) |
| `outputs/pipeline-status-2026-07-11/README.md` | How to open / what figures mean |

Figures copied (or symlinked locally, then copied for standalone) from:

- `outputs/mosaic-visual-qa-2026-07-03/2026-01-25T2200_mosaic_full.png` (+ zoom if useful)
- `outputs/green-refactor-issue-96/rfi-rebuild-2026-07-10/2026-01-25T2200_mosaic_rfi-rebuild_full.png`
- `outputs/astrometry-2026-01-25/diagnostic_nvss_bright20_cutouts.png`
- `outputs/astrometry-2026-01-25/offset_scatter.png`
- `outputs/astrometry-2026-01-25/quiver_sky.png`
- Optional: `hist_separation.png`, `diagnostic_offset_vs_radius.png`

## Slide outline (20)

| # | Title | Content intent |
| --- | --- | --- |
| 1 | Title | DSA-110 continuum imaging — where we stand · Jul 2026 |
| 2 | Science goal | Compact-source variability / ESEs via forced photometry on hourly-epoch mosaics |
| 3 | What “done” means | Dec strip → ~5-min tiles → ~1-hour mosaics with overlap → light curves (not a 24-hour mosaic) |
| 4 | Instrument snapshot | Drift-scan; L-band 1.31–1.50 GHz; 16 subbands; ~96 active antennas |
| 5 | Data flow | Mermaid: HDF5 → MS → cal → tile → hourly-epoch mosaic → photometry |
| 6 | Calibration status | BP/G order (same-date → primary → bright fallback → borrow); fail loud |
| 7 | Tile imaging status | 4800² @ 3″; WSClean; phase-centre / EveryBeam invariants; 3C454.3 ~12.5 Jy check |
| 8 | Mosaicking status | Batch UTC-hour bins + ±2 tile overlap; Quicklook image-domain coadd in production |
| 9 | Reference product | Image: `2026-01-25T2200` full mosaic |
| 10 | Image quality (RFI) | ~40–45% RMS reduction after pre-applycal RFI; flux preserved; lattice + coverage pinch remain |
| 11 | Epoch QA (3 gates) | Flux scale, catalog completeness, noise floor; strict QA skips photometry on FAIL |
| 12 | QA on reference epoch | Completeness ~172/383 ≈ 45% FAIL (raw); coverage-corrected ~71% noted as investigation; not yet production gate |
| 13 | Astrometry | Seeded NVSS/FIRST/RACS PASS; RMS 3–5″ vs gate 8.82″ |
| 14 | Astrometry evidence | Bright NVSS cutouts figure |
| 15 | Astrometry evidence | Offset scatter + quiver (no strong radius-dependent failure once scaled) |
| 16 | Photometry / variability | Condon forced phot + Mooley η/Vs/m exist; **gated off** for this epoch under strict QA |
| 17 | Catalog caveat | Forced-phot “master” is effectively VLASS-only; survey DBs used for astrometry truth |
| 18 | Solid vs not | Two-column readiness summary |
| 19 | Near-term plan | Completeness/coverage gate; unblock photometry; master catalog rebuild; science/deep mosaic later |
| 20 | Takeaway | Imaging+astrometry credible on reference epoch; **not science-ready for variability products yet** |

## Key numbers (must stay accurate)

From existing artifacts (do not invent):

- Astrometry gate: RMS ≤ √(BMAJ×BMIN)/5 ≈ **8.82″**
- NVSS: N=176, median Δ=1.56″, RMS=4.69″ **PASS**
- FIRST: N=16, median Δ=2.22″, RMS=2.98″ **PASS**
- RACS/RAX: N=215, median Δ=1.86″, RMS=4.58″ **PASS**
- RFI rebuild: global robust RMS 8.92 → 5.10 mJy/beam (~−43%); peak flux ~16.2 Jy preserved
- Epoch QA completeness (production gate): **172/383 ≈ 44.9% FAIL** (≥60% required)
- Coverage-corrected completeness (investigation only): **~171/241 ≈ 71%** — call out as *not yet the production gate*

## Non-goals

- Monitor / observability recovery narrative
- Full calibration parameter tables
- Blind Aegean path (timed out / skipped on reference run)
- Claiming Sault/deep coadd is the production science product

## Implementation plan (after spec approval)

1. Create `outputs/pipeline-status-2026-07-11/` with copied figures + `spec.md`.
2. Build `slides/index.html` from the html-presentations template + spec.
3. Run standalone bundler for shareable HTML.
4. Add short README; do not commit unless requested.

## Success criteria

- Collaborator can state in one sentence whether products are science-ready.
- Stage narrative is correct per `CONTEXT.md`.
- Every quantitative claim cites an existing output artifact path in README or speaker notes equivalent (README table).
