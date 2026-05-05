# MS-on-disk inventory under `/stage/dsa110-contimg/ms/`

Generated 2026-05-05 during 3C138 fundamentals-mosaic plan triage.

## Counts by (date, UTC hour)

| Date | Hour | n_MS | Notes |
|---|---|---|---|
| 2026-01-25 | 00 | 10 | |
| 2026-01-25 | 01 | 10 | |
| 2026-01-25 | 02 | 11 | |
| 2026-01-25 | 03 | 10 | |
| 2026-01-25 | **04** | **6** | **gap 04:30–05:00 — does not cover 3C138 transit at 04:57:50** |
| 2026-01-25 | 11 | 10 | |
| 2026-01-25 | 12 | 11 | |
| 2026-01-25 | 13 | 11 | |
| 2026-01-25 | 14 | 10 | |
| 2026-01-25 | 15 | 10 | |
| 2026-01-25 | 16 | 11 | |
| 2026-01-25 | 17 | 10 | |
| 2026-01-25 | 18 | 10 | |
| 2026-01-25 | 19 | 11 | |
| 2026-01-25 | 20 | 10 | |
| 2026-01-25 | 21 | 10 | |
| 2026-01-25 | 22 | 11 | hour 22 mosaic was prior `--force-recal` rebuild target |
| 2026-01-25 | 23 | 8 | gap |
| 2026-02-12 | 00 | 5 | |
| 2026-02-12 | 01 | 8 | |
| 2026-02-15 | 00 | 3 | |
| 2026-02-15 | 01 | 3 | |
| 2026-02-23 | 00 | 8 | |
| 2026-02-25 | 00 | 11 | |
| 2026-02-26 | 00 | 10 | |
| 2026-03-16 | 04 | 2 | stale phantom — no HDF5 on disk |

## Implication for primary-anchored mosaic plan

Cross-reference with `daily_calibrator_transits` Perley-Butler primaries (3C48 0137+331, 3C138 0521+166, 3C147 0542+498, 3C286 1331+305):

- **2026-01-25 hour 04 / 3C138 at 04:57:50 UTC, dec_strip 16°** — 6 MSs, transit not covered.
- All other indexed dates with primary cal transits (2026-02-12..27) have ≤11 MSs across 1–2 hours total — no convertible 12-tile hour anywhere.

Conclusion: no R2-style "ready-to-go" date+hour exists. R1 (convert the 04:30–05:00 gap on 2026-01-25 hour 04 from HDF5) is the unique viable path to a primary-anchored 12-tile mosaic.
