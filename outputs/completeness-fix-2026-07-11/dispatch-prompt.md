# Task: coverage-aware epoch QA completeness gate

Use the `ai-research-workflows` plugin skills from this session's `--plugin-dir`.
Follow `ai-research-workflows:using-research-workflows` routing.

## Interaction mode
**Direct** — enough context is provided; do not ask clarifying questions unless blocked.
Do not commit or push unless asked.

## Goal
Land the coverage-aware completeness fix for
`dsa110_continuum/photometry/epoch_qa.py` so catalog sources on non-finite
mosaic pixels are excluded from the completeness denominator.

## Known diagnosis (already done)
Read completely:
- `outputs/green-refactor-issue-96/rfi-rebuild-2026-07-10/COMPLETENESS_INVESTIGATION.md`
- `outputs/green-refactor-issue-96/rfi-rebuild-2026-07-10/epoch_qa_patched.py`
- `outputs/green-refactor-issue-96/rfi-rebuild-2026-07-10/completeness_diag.py`
- current `dsa110_continuum/photometry/epoch_qa.py` (main still uses `n_recovered / n_catalog`)
- existing tests under `tests/test_epoch_qa.py` if present

Root cause: production gate divides by all NVSS sources in the RA/Dec bbox,
including blank/NaN pixels (~37% of 383 on 2026-01-25T2200). Coverage-corrected
completeness is ~171/241 = 71% (PASS at 60%). Threshold stays 60%.

## Required workflow chain
1. `ai-research-workflows:researching` — confirm main vs patched state; write
   `docs/rse/specs/research-coverage-aware-completeness.md`
2. `ai-research-workflows:planning-implementations` — write
   `docs/rse/specs/plan-coverage-aware-completeness.md` (no placeholders / open questions)
3. `ai-research-workflows:implementing-plans` — execute the plan on a branch
   `agent/coverage-aware-completeness` (create if needed)
4. `ai-research-workflows:validating-implementations` — run relevant pytest with
   `/opt/miniforge/envs/casa6/bin/python` and `PYTHONPATH=/data/dsa110-continuum`;
   write `docs/rse/specs/validation-coverage-aware-completeness.md`

## Implementation constraints
- Add `n_covered` to `EpochQAResult`; exclude non-finite center pixels from denominator
- Gate SKIP if `n_covered < QA_MIN_CATALOG_SOURCES` (5)
- Keep `QA_COMPLETENESS_MIN = 0.60` unchanged
- Update callers/provenance/CSV fields if they assume old schema
- Add/port `TestCoverageAwareCompleteness` tests
- Use casa6 python for all tests
- Do NOT bulk-fix unrelated ruff issues
- Do NOT claim the mosaic is science-ready — stricter image QA may still FAIL

## Done when
- Research + plan + validation docs exist under `docs/rse/specs/`
- Code on branch implements coverage-aware completeness
- Targeted tests pass
- Final message summarizes diff, test results, and remaining genuine-miss work (~70 on-support misses)
