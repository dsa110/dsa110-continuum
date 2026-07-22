# Promotion ledger

Auto-appended by `dsa110_continuum.qa.promotion.append_promotion_ledger`. One row per `(date, hour)` window at end-of-run. Operator finalizes `promotion_class` after evaluating the photometric anchor block in the side-car JSON.

| date       | hour | class                          | tier | epoch_gc                          | anchor                                                | side-car (relative to products root)               | git_sha  |
|------------|------|--------------------------------|------|------------------------------------|--------------------------------------------------------|----------------------------------------------------|----------|
| 2026-01-25 | 22   | auto_emitted_pending_review    | A    | skipped_or_failed_low_snr          | —                                                      | mosaics/2026-01-25/promotion_2026-01-25T2200.json  | 2bd3e4c  |
| 2026-07-15 | 11   | auto_emitted_pending_review    | C    | solved                             | —                                                      | products/2026-07-15/promotion_2026-07-15T1100.json | cbe629e4 |
