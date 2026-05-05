# Session brief: tackle issue #64 (slow-marker skipped tests)

You are a fresh Claude Code session opening on `/data/dsa110-continuum` (H17). Your job is to fully resolve [GitHub issue #64](https://github.com/dsa110/dsa110-continuum/issues/64) — five tests gated by `@pytest.mark.slow`, all recommended **fix** (not delete) — in a single session, end to end, no human intervention.

## Pre-flight (≤5 min)

Run these in order before touching anything:

```bash
gh issue view 64 --repo dsa110/dsa110-continuum
cat outputs/skipped_tests_triage_2026-05-04/skipped_tests_issue.md      # full triage; read Group 1
sed -n '20,45p' tests/conftest.py                                       # understand the slow gate
git log --oneline -5
```

Skills to invoke up front:
- `superpowers:test-driven-development` (for every test edit)
- `superpowers:verification-before-completion` (before every commit; never claim green without running pytest)

Environment (overrides AGENTS.md cloud-VM section):
- Use `/opt/miniforge/envs/casa6/bin/python` for ALL pytest invocations.
- NEVER use `python3` or `PYTHONPATH=/workspace` — that env doesn't exist on H17.
- Default suite currently: 1037 passed, 11 skipped. **Target after this session: ≤6 skipped, 0 failed.**
- Never write artifacts to `/tmp` or `/dev/shm` (volatile). Use `outputs/<topic>_2026-05-05/` if needed.

## Pre-committed policy decisions (do NOT pause to ask)

These keep the session autonomous:

| Decision | Choice | Rationale |
|---|---|---|
| Scattering test dependency | **Mock approach.** Rewrite to follow the mocked-scattering pattern already used elsewhere in `tests/test_scattering_qa.py`. Do NOT install the non-PyPI `scattering` package. | "Don't install pip packages without asking" (global CLAUDE.md). |
| L-effort test (full pipeline) | **45-min timebox.** See section 4 below for the exact triage tree. | Avoids open-ended production debugging. |
| Commit cadence | One commit per test (S/M) or per logical fix. **Push at the very end,** not after each commit. | Keeps history clean; lets a final `git push` reflect the session as a unit. |

## Workflow — cheapest first

### 1. S — `test_full_pipeline_result_is_serializable`

Pull this out of `class TestEndToEnd` in `tests/test_simulated_pipeline.py`. Rewrite as a pure unit test that constructs `SimulatedPipelineResult` and `SourceFluxResult` directly and asserts `dataclasses.asdict()` returns a `dict`. No fixtures, no slow class.

```bash
/opt/miniforge/envs/casa6/bin/python -m pytest tests/test_simulated_pipeline.py -k test_full_pipeline_result_is_serializable -v
```

Commit: `fix(test): convert test_full_pipeline_result_is_serializable to unit test`

### 2. M — `test_check_tile_scattering_integration`

Read the surrounding tests in `tests/test_scattering_qa.py` to find the existing mocked-scattering pattern. Apply it to this test so `check_tile_scattering()` is exercised against a synthetic FITS but the scattering calculator is mocked. The assertion shape should be unchanged from the original intent.

```bash
/opt/miniforge/envs/casa6/bin/python -m pytest tests/test_scattering_qa.py -v
```

Commit: `fix(test): mock scattering calculator in check_tile_scattering integration`

### 3. M (×2) — `test_full_96_antenna_subband` and `test_all_16_subbands_closure`

For each, run with `--run-slow` first to measure runtime:

```bash
/opt/miniforge/envs/casa6/bin/python -m pytest --run-slow \
    tests/test_integration_e2e.py::TestSlow::<test_name> -v --durations=5
```

Branch by result:
- Runtime <60s, passes → **drop the `slow` marker**, done.
- Runtime ≥60s → shrink fixture parameters (`n_integrations`, `n_sky_sources`, image size, etc.) until <60s while preserving the metadata/closure invariants. Drop the marker. Done.
- Fails for non-timing reasons → **STOP that test only**, leave the slow marker, file a followup issue (see section 4 fallback) — do not block the rest of the session.

Verify each: `pytest tests/test_integration_e2e.py::TestSlow::<test_name> -v` (without `--run-slow` to confirm marker is gone).

Commits (one each):
- `fix(test): inline slow 96-antenna subband test under default budget`
- `fix(test): inline slow 16-subband closure test under default budget`

### 4. L — `test_full_pipeline_recovers_sources`

**Timebox: 45 minutes wall.** Run with `--run-slow` and capture the failure mode:

```bash
/opt/miniforge/envs/casa6/bin/python -m pytest --run-slow \
    tests/test_simulated_pipeline.py::TestEndToEnd::test_full_pipeline_recovers_sources \
    -v --durations=0 --timeout=180 --timeout-method=thread 2>&1 | tee \
    outputs/issue64_session_2026-05-05/full_pipeline_run.log
```

(Create `outputs/issue64_session_2026-05-05/` first; never log to `/tmp`.)

Triage:

- **Passes, runtime <60s** → drop the slow marker. Commit. Done.
- **Passes, runtime ≥60s** → shrink imsize / niter / tile count / subband count until <60s, preserve at least one recovered-source assertion. Commit.
- **Fails for any non-timing reason** → **STOP this test only.** Leave the slow marker in place. File a followup:

  ```bash
  gh issue create --repo dsa110/dsa110-continuum \
      --title "Followup: test_full_pipeline_recovers_sources fails under --run-slow" \
      --body "$(cat <<'EOF'
  Discovered while resolving #64. The test fails for non-timing reasons under \`--run-slow\`. Failure trace and runtime captured at \`outputs/issue64_session_2026-05-05/full_pipeline_run.log\`.

  Out of scope for #64 (single-session test cleanup); needs deeper investigation into the pipeline component that's failing.
  EOF
  )"
  ```

  Note the followup issue number in your final session summary. Mark this test as **deferred** in the issue #64 close comment; do NOT close #64 if this happens — leave it open with the deferred test called out.

### 5. Final verification (mandatory before push)

```bash
/opt/miniforge/envs/casa6/bin/python -m pytest tests/ \
    --ignore=tests/test_mosaic_ra_wrap.py -q --timeout=60 \
    --timeout-method=thread
```

Acceptance:
- Skip count ≤6 (was 11; you fixed 5, or 4 if you deferred the L task).
- Zero new failures. (The 9 pre-existing failures fixed by codex earlier this week should stay fixed.)

If the suite has any new failure, **revert the offending commit and stop** — better to ship 4 fixes than 5 + a regression.

### 6. Push + close

```bash
git push origin main

# Build a comment with your commits and the final pytest line
SUMMARY=$(git log --oneline e6c1f72..HEAD)  # adjust base if codex commits have moved
gh issue comment 64 --repo dsa110/dsa110-continuum --body "Resolved in commits:
$SUMMARY

Final suite: <paste the X passed, Y skipped line>"

# Only if ALL 5 tests resolved (no L-task deferral):
gh issue close 64 --repo dsa110/dsa110-continuum
```

## Subagents / Codex CLI usage

- **Default: do not spawn codex.** All five fixes are mechanical test edits — they fit comfortably in your tool budget.
- **Exception: if section 4's L task needs deep WSClean / pipeline investigation,** you may delegate the *investigation* (not the fix) via the codex-wait wrapper. See your global CLAUDE.md "Spawning Codex CLI subagents" section. Use `-s read-only` and have codex write a triage note to `outputs/issue64_session_2026-05-05/`. Do NOT let codex modify production code in this session.
- For broad codebase questions during the M-task ("how is `SimulatedPipelineResult` constructed elsewhere?"), use the `Explore` agent — fast read-only, no side effects.

## Hard stop conditions

If any of these trigger, stop the session and write a comment on issue #64 with what you did and what's blocked:
1. A test fix would require a non-trivial production-code change (anything in `dsa110_continuum/` that isn't a 1-line bugfix).
2. The default suite has new failures you can't trivially revert.
3. You've spent >2.5 hours wall on this session.
4. You hit something that genuinely requires user judgement and isn't covered by the policy decisions above.

## Out of scope

- Issue #65 (Step 6 mosaic skips) — separate session, separate prompt.
- Force-pushing, rewriting history, modifying CI.
- Installing pip packages.
