# DSA-110 continuum pipeline status — July 2026

Science-collaborator slide deck: where the imaging pipeline stands, with balanced honesty about what’s solid vs not yet ready.

Reference epoch: **2026-01-25T2200** hourly-epoch mosaic.

## Open the deck

- PDF (21 slides): [`DSA110-continuum-pipeline-status-2026-07-11.pdf`](DSA110-continuum-pipeline-status-2026-07-11.pdf)
- Interactive: [`slides/index.html`](slides/index.html)
- Shareable (images embedded): [`slides/pipeline-status-2026-07-11-standalone.html`](slides/pipeline-status-2026-07-11-standalone.html)

Keyboard: ←/→ or Space · Home/End

## Design / outline

[`docs/superpowers/specs/2026-07-11-pipeline-status-slide-deck-design.md`](../../docs/superpowers/specs/2026-07-11-pipeline-status-slide-deck-design.md)

## Figure provenance

| Slide asset | Source |
| --- | --- |
| `09_mosaic_full.png` / `09_mosaic_zoom.png` | `outputs/mosaic-visual-qa-2026-07-03/` |
| `10_rfi_rebuild_full.png` | `outputs/green-refactor-issue-96/rfi-rebuild-2026-07-10/` |
| `14_nvss_bright20_cutouts.png` | `outputs/astrometry-2026-01-25/diagnostic_nvss_bright20_cutouts.png` |
| `15_offset_scatter.png` / `15_quiver_sky.png` | `outputs/astrometry-2026-01-25/` |

## Key numbers (as of deck build)

| Claim | Value | Source |
| --- | --- | --- |
| Astrometry gate | RMS ≤ 8.82″ | `outputs/astrometry-2026-01-25/summary.json` |
| NVSS / FIRST / RACS | PASS (RMS 4.69″ / 2.98″ / 4.58″) | same |
| Completeness (production gate) | 172/383 ≈ 45% **FAIL** | epoch QA / RFI rebuild comparison notes |
| Coverage-corrected completeness | ~71% (investigation only) | `rfi-rebuild-2026-07-10/COMPLETENESS_INVESTIGATION.md` |
| RFI RMS change | 8.92 → 5.10 mJy/beam (~−43%) | `rfi-rebuild-2026-07-10/COMPARISON.md` |

## Takeaway

Imaging and astrometry look credible on the reference epoch. Variability science products are **not** ready until completeness / photometry clear under strict QA.
