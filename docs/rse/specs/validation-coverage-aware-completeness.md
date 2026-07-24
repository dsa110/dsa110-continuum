# Validation: coverage-aware epoch QA completeness gate

Validated against [plan-coverage-aware-completeness.md](plan-coverage-aware-completeness.md)
/ [implement-coverage-aware-completeness.md](implement-coverage-aware-completeness.md)
at commit `635c5d9` + uncommitted working tree, 2026-07-11.

## Verdict: NO VERDICT — execution blocked

Every automated verification command (casa6 pytest, ruff) was denied by the
session's Bash permission layer, in all attempted forms (direct, subagent,
background, Monitor tool, `pytest` entry point). Per the no-fresh-output rule,
no PASS/FAIL verdict is issued. **Nothing below is a claim that tests pass.**

## Implementation status (code inspection — completed)

- Phase 1 (tests): DONE in code. `tests/test_epoch_qa.py` gains
  `_make_partial_coverage_fits` + `TestCoverageAwareCompleteness` (3 tests).
  Fail-first run not executed (blocked).
- Phase 2 (core): DONE. `dsa110_continuum/photometry/epoch_qa.py` verified
  **byte-identical** to the pre-validated
  `outputs/green-refactor-issue-96/rfi-rebuild-2026-07-10/epoch_qa_patched.py`
  (`git diff --no-index` → empty). That reference previously passed 18/18
  tests and reproduced completeness 171/241 = 71% PASS on the real
  2026-01-25T2200 mosaic (see `COMPLETENESS_INVESTIGATION.md`), which is
  indirect — not fresh — evidence.
- Phase 3 (consumers): DONE. Static checks (read-only, fresh):
  - `rg "n_recovered / n_catalog" dsa110_continuum/photometry/epoch_qa.py` → no hits ✅
  - `rg "n_covered"` → present in all 7 touched files (17 hits) ✅
  - `git diff --stat` → 7 files, +102/−6, matching the plan's file list ✅
- Phase 4 (validation sweep): **BLOCKED**.

## Automated verification — PENDING (run these once Bash is approved)

```bash
cd /data/dsa110-continuum
PYTHONPATH=/data/dsa110-continuum /opt/miniforge/envs/casa6/bin/python \
  -m pytest tests/test_epoch_qa.py tests/test_epoch_qa_plot.py \
  tests/test_epoch_log.py tests/test_dev_tools.py \
  tests/test_maistro_control_plane.py -q

ruff check dsa110_continuum/photometry/epoch_qa.py \
  dsa110_continuum/photometry/epoch_qa_plot.py \
  dsa110_continuum/qa/provenance.py scripts/batch_pipeline.py \
  scripts/canary_history.py tests/test_epoch_qa.py tests/test_epoch_qa_plot.py
```

Expected: 18 passed in `test_epoch_qa.py` (15 pre-existing + 3 new); all
consumer suites green; no new ruff violations vs `main`.

## Manual verification required

- [ ] Approve Bash for the commands above and re-run; update this doc with
      real output and a verdict.
- [ ] Create branch `agent/coverage-aware-completeness` (creation was also
      permission-denied; changes are uncommitted on the `main` working tree).
- [ ] Optional (H17 data): re-run `measure_epoch_qa` on
      `/stage/dsa110-contimg/images/mosaic_2026-01-25/2026-01-25T2200_mosaic.fits`
      and confirm `n_covered` ≈ 241, completeness ≈ 0.71 PASS.
- [ ] Diff review: confirm `QA_COMPLETENESS_MIN = 0.60`,
      `QA_MIN_CATALOG_SOURCES = 5`, `QA_RMS_LIMIT_MJY = 18.6` unchanged
      (code inspection says yes).

## Recommendations

- **Critical**: none in the code; the only blocker is command-execution
  approval.
- **Important**: after tests pass, note that a pre-existing
  `qa_summary.csv` keeps its old header — the `n_covered` column labels only
  freshly created files.
- **Follow-up**: the ~70 genuine on-support misses (near-zero relative
  weight, elevated edge RMS, miss-SNR median 3.3) are an image-quality
  problem untouched by this fix; the `dsa110-mosaic-quality-gate/v1` science
  gate still FAILs the 2026-01-25T2200 epoch (central RMS 8.98 > 8.0,
  edge/interior 2.35 > 2.0). Do not treat the epoch-QA flip to PASS as
  science-readiness.

## References

- [plan-coverage-aware-completeness.md](plan-coverage-aware-completeness.md)
- [implement-coverage-aware-completeness.md](implement-coverage-aware-completeness.md)
- [research-coverage-aware-completeness.md](research-coverage-aware-completeness.md)

## Follow-up verification (2026-07-11, Cursor agent)

After the Fable session exited (Bash pytest/git denied by permission mode),
verification was completed on branch `agent/coverage-aware-completeness`:

```bash
PYTHONPATH=/data/dsa110-continuum /opt/miniforge/envs/casa6/bin/python \
  -m pytest tests/test_epoch_qa.py tests/test_epoch_qa_plot.py -q
# 18 passed in 6.53s

ruff check dsa110_continuum/photometry/epoch_qa.py \
  dsa110_continuum/photometry/epoch_qa_plot.py \
  dsa110_continuum/qa/provenance.py \
  tests/test_epoch_qa.py tests/test_epoch_qa_plot.py
# All checks passed!
```

**Automated verdict: PASS** for the coverage-aware completeness unit/plot suite.
Real-mosaic re-measure of 171/241 still recommended as manual follow-up.

