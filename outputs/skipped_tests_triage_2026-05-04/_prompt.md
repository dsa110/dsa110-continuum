You are doing READ-ONLY analysis in `/data/dsa110-continuum`, NOT a fix
session. Do not edit any source or test files.

# Goal

Produce a single markdown document — suitable for pasting as a GitHub
issue — that triages every currently-skipped test in this repo. The
ideal end state has zero skipped tests; for each one we want to either
fix it or delete it. This document is the input to that decision.

# Environment

- Use `/opt/miniforge/envs/casa6/bin/python` for any pytest invocations.
- Do NOT use `python3` or `PYTHONPATH=/workspace`.
- The full default suite currently reports `1037 passed, 11 skipped`.
- Do NOT write anything to `/tmp/` — `/tmp` is wiped regularly and
  storage is tight. The ONLY output path you may write is:
      /data/dsa110-continuum/outputs/skipped_tests_triage_2026-05-04/skipped_tests_issue.md

# Workflow

1. Enumerate every skipped test with its reason. Use:
       /opt/miniforge/envs/casa6/bin/python -m pytest tests/ \
         --ignore=tests/test_mosaic_ra_wrap.py -v -rs --no-header \
         --timeout=60 --timeout-method=thread 2>&1 \
         | grep -E "SKIPPED|^tests/.*skipped"
   Cross-check the count is 11. If different, use the actual number.

2. For each skipped test, read the test file + relevant source to
   understand:
   - Why is it skipped? (marker, importorskip, pytest.skip call, fixture
     skip, env gate, etc.)
   - Is the underlying functionality still in scope for the project?
   - What would it take to UN-skip it (fix path), or is the test
     obsolete/replaced (delete path)?

3. Group skips by mechanism (e.g. `@pytest.mark.slow`,
   `pytest.importorskip("scattering")`, env-only `casacore stub`) so the
   reader can see clusters.

4. Write the document to:
       /data/dsa110-continuum/outputs/skipped_tests_triage_2026-05-04/skipped_tests_issue.md
   Format (verbatim section structure):

   # Skipped tests triage — YYYY-MM-DD

   Default `pytest tests/` run currently reports `N passed, M skipped`.
   The ideal end state has zero skips; this doc enumerates each skip
   with a recommendation: **fix** or **delete**.

   ## Summary
   | # | test id | mechanism | recommendation | effort |
   |---|---------|-----------|----------------|--------|
   ...

   ## Group 1 — <mechanism>
   ### `tests/<file>::<test>`
   **Skip reason.** ...
   **Why it's skipped.** ...
   **Underlying functionality in scope?** Yes / No / Unclear — ...
   **Recommendation: fix | delete.** ...
   **Effort estimate.** S / M / L (with one-line justification).

   ## Group 2 — <mechanism>
   ...

   ## Proposed action plan
   - Bulleted list, ordered by effort and value.
   - Call out any skips that should be deleted outright.
   - Note any cases where the right answer is "fix the test infra"
     (e.g. install an optional dep) vs "fix the production code".

5. After writing the file, print to stdout:
   - the absolute path
   - the document's word count
   - the recommendation tally (e.g. "5 fix, 4 delete, 2 unclear")
   Nothing else — no diffs, no extra commentary.

# Constraints

- Read-only — do not modify any file under `dsa110_continuum/`,
  `scripts/`, or `tests/`.
- The ONLY file you may write is the markdown document at the path
  above. Do NOT write any other files anywhere — no scratch files in
  `/tmp`, no extra logs, no helper scripts.
- Do not run git commands beyond `git log` / `git blame` for context.
- Do not commit anything.
- Do not push anything.
- If a skip's root cause requires a non-trivial code investigation that
  doesn't fit a triage doc, say so explicitly in that test's section
  rather than guessing.
