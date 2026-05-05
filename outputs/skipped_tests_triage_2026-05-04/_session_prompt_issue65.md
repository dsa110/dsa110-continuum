# Session brief: tackle issue #65 (Step 6 mosaic skipped tests)

You are a fresh Claude Code session opening on `/data/dsa110-continuum` (H17). Your job is to fully resolve [GitHub issue #65](https://github.com/dsa110/dsa110-continuum/issues/65) — six tests in `tests/test_two_stage_photometry.py` skipped because of a missing `pipeline_outputs/step6/step6_mosaic.fits` artifact — in a single session, end to end, no human intervention. **3 to delete, 3 to fix.**

## Pre-flight (≤5 min)

```bash
gh issue view 65 --repo dsa110/dsa110-continuum
cat outputs/skipped_tests_triage_2026-05-04/skipped_tests_issue.md      # full triage; read Group 2
cat tests/test_two_stage_photometry.py                                  # whole file — you'll be editing most of it
git log --oneline -5
```

Skills to invoke up front:
- `superpowers:test-driven-development` (for every fix)
- `superpowers:verification-before-completion` (before every commit; never claim green without running pytest)

Environment (overrides AGENTS.md cloud-VM section):
- Use `/opt/miniforge/envs/casa6/bin/python` for ALL pytest invocations.
- NEVER use `python3` or `PYTHONPATH=/workspace`.
- Default suite: 1037 passed, 11 skipped. **Target: 5 skipped (slow-marker only, untouched), 0 failed.**
- Never write artifacts to `/tmp` or `/dev/shm`. Use `outputs/<topic>_2026-05-05/` if needed.

## Pre-committed policy decisions (do NOT pause to ask)

| Decision | Choice | Rationale |
|---|---|---|
| Delete vs fix per test | Per the triage table at the top of issue #65. Don't relitigate. | The 3 delete tests are confirmed redundant with existing synthetic-FITS coverage. |
| FITS fabrication | Build a small (e.g. 200×200) FITS in `tmp_path` using `astropy.io.fits` + `astropy.wcs.WCS` with BMAJ/BMIN headers and 2–3 Gaussian sources. | Removes the disk-artifact dependency; matches existing patterns. |
| Reuse vs invent | Search the repo for FITS-fabrication helpers (`rg "fits\.PrimaryHDU\|WCS\(\)" tests/`) before writing new code. Reuse if anything fits. | Cheaper + consistent. |
| CLI test cwd | `cwd = Path(__file__).parents[1]` (computed repo root) — NEVER the hardcoded `/home/user/workspace/dsa110-continuum`. | Cross-checkout-safe. |
| Python in subprocess | `sys.executable`, not a hardcoded path. | Same. |
| Commit cadence | One commit per logical fix (one for the 3 deletes, one per fix). Push at the very end. | Clean history. |

## Workflow

### 1. Delete the 3 redundant tests (single commit)

In `tests/test_two_stage_photometry.py`, remove:
- `test_coarse_pass_returns_finite`
- `test_snr_gate_all_pass_with_low_rms`
- `test_snr_gate_all_fail_with_high_rms`

Verify the rest of the file still collects and the non-skipped tests still pass:

```bash
/opt/miniforge/envs/casa6/bin/python -m pytest tests/test_two_stage_photometry.py -v
```

Commit: `fix(test): remove redundant Step 6-artifact-dependent coarse-pass tests`

### 2. Fix `test_beam_correction_ratio_bright_sources`

Replace the disk artifact with a generated synthetic FITS in `tmp_path`. Required ingredients:
- WCS centred so the seed=42 simulation source positions land inside the image.
- BMAJ/BMIN/BPA headers (use a plausible synth beam, e.g. 30 arcsec circular).
- 2–3 bright Gaussian sources matching the catalog the test uses.
- Recompute the expected beam-correction ratio for the new image and document it inline (one comment line — *why*, not *what*).

```bash
/opt/miniforge/envs/casa6/bin/python -m pytest \
    tests/test_two_stage_photometry.py::test_beam_correction_ratio_bright_sources -v
```

Commit: `fix(test): generate synthetic FITS in tmp_path for beam-correction test`

### 3. Fix the two CLI tests

For each of `test_cli_simple_peak_sim_produces_csv` and `test_cli_two_stage_sim_produces_coarse_snr_column`:

1. Generate a small FITS mosaic in `tmp_path` (reuse the helper from step 2 — extract it to a fixture if it makes sense).
2. Use `sys.executable` for the subprocess call.
3. Compute repo root: `repo_root = Path(__file__).parents[1]` and pass that as `cwd=`.
4. Write `--output` under `tmp_path`.
5. If `--sim` source positions don't land inside the small generated mosaic:
   - **First fallback:** add a test-only catalog file in `tmp_path` and pass it via the script's catalog arg (read the script source to find which arg).
   - **Last fallback:** call `run_forced_photometry()` directly for column-shape coverage, and keep ONE subprocess CLI smoke that just verifies argument wiring (no source recovery).

Verify each:

```bash
/opt/miniforge/envs/casa6/bin/python -m pytest tests/test_two_stage_photometry.py -k cli -v
```

Commits (one each):
- `fix(test): tmp_path FITS + sys.executable for simple-peak CLI test`
- `fix(test): tmp_path FITS + sys.executable for two-stage CLI test`

### 4. Optional cleanup (15-min timebox)

```bash
rg "/home/user/workspace/dsa110-continuum" -- tests/ scripts/
```

Patch any other stale-cwd hits with the same `Path(__file__).parents[1]` pattern. Skip if none found or it expands beyond 15 min — note in the issue close comment.

If you patch any: separate commit, `fix(scripts): replace stale /home/user/workspace cwd with computed repo root`.

### 5. Final verification (mandatory before push)

```bash
/opt/miniforge/envs/casa6/bin/python -m pytest tests/ \
    --ignore=tests/test_mosaic_ra_wrap.py -q --timeout=60 \
    --timeout-method=thread
```

Acceptance:
- 5 skipped (slow-marker only, untouched).
- 0 failures. (No regressions in the 9 codex-fixed tests from earlier this week.)
- No remaining references to the dead artifact:

```bash
rg "step6_mosaic\.fits|/home/user/workspace" tests/   # should print 0 hits
```

If new failures appear, **revert the offending commit and stop** — partial fixes that don't regress beat full fixes that do.

### 6. Push + close

```bash
git push origin main

SUMMARY=$(git log --oneline 69ec1d3..HEAD)  # adjust base to the prior tip on main
gh issue comment 65 --repo dsa110/dsa110-continuum --body "Resolved in commits:
$SUMMARY

Final suite: <paste the X passed, Y skipped line>

\`rg step6_mosaic.fits tests/\` returns 0 hits."

gh issue close 65 --repo dsa110/dsa110-continuum
```

## Subagents / Codex CLI usage

- This work is mechanical. **Do NOT spawn codex.**
- For repo searches ("how is FITS fabricated elsewhere in tests?"), use `rg` directly — don't escalate to the `Explore` agent unless you've already searched and come up empty.

## Hard stop conditions

Stop the session and comment on issue #65 if any of these trigger:
1. A test fix would require non-trivial production-code change.
2. New failures you can't trivially revert.
3. >2 hours wall on this session.
4. Genuine user-judgement question not covered above.

## Out of scope

- Issue #64 (slow-marker skips) — separate session.
- Generating a real `pipeline_outputs/step6/step6_mosaic.fits` artifact — explicitly NOT what this issue wants. The whole point is to drop the artifact dependency.
- Force-pushing, rewriting history, modifying CI.
- Installing pip packages.
