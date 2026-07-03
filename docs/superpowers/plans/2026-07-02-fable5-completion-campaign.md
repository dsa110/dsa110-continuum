# Fable 5 Completion Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete `dsa110-continuum` through a controlled, evidence-driven PR stack using Claude Fable 5 for long-running orchestration and bounded implementation lanes.

**Architecture:** Treat Fable 5 as the lead autonomous engineer for isolated branches, not as a single unreviewed mega-branch. Each lane starts with repository-context reading, produces a small PR with tests, and stops at explicit H17-only validation when local macOS evidence is insufficient.

**Tech Stack:** Python 3.12, CASA 6 via `/opt/miniforge/envs/casa6/bin/python`, pytest, ruff, GitHub Issues/PRs, Claude Code `claude-fable-5`, DSA-110 HDF5/MS/FITS pipeline data on H17.

---

## Verified Host State

Initial H17 access was verified from the local macOS checkout on 2026-07-02:

- SSH host alias: `h17`
- Resolved host: `lxd110h17`
- SSH user: `ubuntu`
- Primary checkout: `/data/dsa110-continuum`
- H17 branch state at verification: `main...origin/main`
- H17 dirty state at verification: untracked `.emdash/` directory, classified as generated/tool state and intentionally preserved
- CASA Python: `/opt/miniforge/envs/casa6/bin/python`
- CASA Python version: `3.12.12`

Do not delete, clean, stage, or overwrite `.emdash/` as part of this campaign unless a separate task proves it is stale and user-approved for removal.

## Global Rules For Every Agent

- Read `AGENTS.md`, `CONTEXT.md`, `docs/agents/`, and the relevant `docs/skills/` plus `docs/reference/` files before editing code.
- Use repo vocabulary exactly: tile, hourly-epoch mosaic, Dec strip, MS, HDF5, FITS.
- Use `dsa110_continuum.*` imports for new code.
- Do not add new `dsa110_contimg` imports.
- Do not change legacy `__init__.py` re-export layers unless a task explicitly reopens that design.
- Preserve the silent-failure invariants around `FIELD::PHASE_DIR`, `FIELD::REFERENCE_DIR`, and `TELESCOPE_NAME = DSA_110`.
- Treat `scripts/qa_server.py` and `scripts/monitor_server.py` as live surfaces; changes to `POST /exec` are security-relevant.
- Use `/opt/miniforge/envs/casa6/bin/python` on H17 for tests and pipeline checks.
- Keep large stage/correlation products out of git.
- Commit only validated, task-scoped changes.
- If a task needs H17 data, run the H17 validation or stop with the exact H17 command and missing external condition.

## Standard Dispatch Command

Use subscription/OAuth Claude Code auth only. Do not pass Anthropic API key environment variables.

```bash
env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN \
  claude --model claude-fable-5 --effort xhigh \
  --permission-mode auto \
  --name "dsa110-continuum-<lane>" \
  -p "$PROMPT" \
  < /dev/null
```

For read-only review or planning tasks, include `Do not edit files.` in the prompt.

## Required Closeout

Each implementation lane must produce:

- A branch name and commit list.
- A short changed-file summary.
- Tests run, exact commands, and pass/fail output summary.
- Any H17-only validation performed or explicitly still pending.
- Dirty-state classification for every touched repo.
- Restart/reload inventory if runtime services or long-lived processes changed.
- A GitHub issue or PR comment with evidence and residual risk.

Run closeout checks when the tool is available:

```bash
mskill tool agent-closeout-check \
  --repo /data/dsa110-continuum \
  --touched <changed-paths>
```

## Phase 0: Completion Map And Issue Triage

**Goal:** Convert the open issue set into an ordered, dependency-aware PR stack.

**Inputs:**

- `gh issue list --repo dsa110/dsa110-continuum --state open`
- `docs/agents/issue-tracker.md`
- `docs/agents/triage-labels.md`
- `CONTEXT.md`
- `docs/skills/`
- `docs/reference/`

**Output:**

- Classify open issues as `ready-for-agent`, `needs-human-decision`, `blocked-on-H17-data`, `duplicate/stale`, or `must-fix-before-production`.
- Comment on the campaign tracking issue with the classification.
- Apply GitHub labels only when the classification is directly supported by issue contents and repo evidence.

**First command:**

```bash
env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN \
  claude --model claude-fable-5 --effort xhigh \
  --permission-mode auto \
  --name "dsa110-continuum-completion-map" \
  -p "Read AGENTS.md, CONTEXT.md, docs/agents, docs/skills, docs/reference, and the open GitHub issues for dsa110/dsa110-continuum. Do not edit files. Produce a completion map: lanes, dependencies, first 10 PRs, required tests, H17-only validation, and issues that need human decisions. Use repo terminology exactly." \
  < /dev/null
```

## Phase 1: Correctness Gates Before Features

**Goal:** Fix issues that can silently break science products or production orchestration.

**Initial issue set:**

- #47 flaky `.pytest_tmp` cleanup race in `test_batch_e1_hygiene`
- #70 `batch_pipeline.py`: `--force-recal` does not propagate `force=True` to `ensure_bandpass`
- #71 `batch_pipeline.py` does not convert HDF5 to MS for hourly tiles
- #72 `CalibratorMSGenerator.generate_from_transit()` ignores `transit_time`
- #73 bad-pol detection label inversion under amplitude-imbalanced single-pol failure
- #79 replace silent `table=None` fallback in `calibration/solver_common.py`

**Validation floor:**

```bash
PYTHONPATH=/data/dsa110-continuum \
  /opt/miniforge/envs/casa6/bin/python -m pytest <targeted-tests> -q

ruff check <touched-paths>
```

Use H17 for any test requiring CASA, casatools, telescope data paths, or real staged products.

## Phase 2: Mosaic Architecture And Migration

**Goal:** Choose and wire the canonical hourly-epoch mosaic implementation without preserving duplicate orchestration paths.

**Initial issue set:**

- #75 defer legacy Dagster bootstrap so mosaic imports do not require `/dev/shm/dsa110-contimg/`
- #76 ADR: choose canonical hourly-epoch coadd implementation
- #77 migrate `batch_pipeline.py` coadd from `mosaic_day` to canonical implementation
- #78 wire `SlidingWindowTrigger` to streaming hourly-epoch mosaic driver
- #80 audit and rename/delete `scripts/mosaic_day.py`
- #74 fix or delete `dsa110_continuum/mosaic/__main__.py`

**Stop condition:**

Write or update an ADR before changing production mosaic behavior. Do not migrate `batch_pipeline.py` until #76 is resolved.

## Phase 3: Live Observability Stack

**Goal:** Land an operator-facing observability path in tracer-bullet slices without weakening live service security.

**Initial issue set:**

- #48 ADR: live observability server architecture
- #49 ADR: auth posture
- #50 ADR: interactive image/FITS exploration vs pre-rendered diagnostics
- #51 tracer-bullet routed live observability server
- #52 end-to-end stage event to browser card
- #53 through #60 artifact browser and QA views
- #61 mutating mosaic-on-demand routes
- #62 retire or migrate `scripts/monitor_server.py`

**Security rule:**

Do not change `scripts/monitor_server.py` `POST /exec` behavior without an explicit security note in the PR and a restart inventory.

## Phase 4: Evidence Matrix And Scientific Validation

**Goal:** Make pipeline readiness measurable with reproducible evidence instead of static claims.

**Initial issue set:**

- #35 fix cloud pytest collection blockers
- #36 restore coverage XML evidence
- #37 add executable conversion-stage evidence
- #38 define H17/CASA calibration and imaging smoke-evidence workflow
- #39 add production-mosaic evidence
- #40 add multi-epoch photometry and light-curve evidence fixture
- #41 add source-finding and catalog database evidence path
- #42 harden the evidence matrix
- #64 skipped slow-test triage
- #68 canonical 3C286 BP/G smoke demo
- #69 slow full-pipeline recovery hang

**Evidence rule:**

Do not claim H17 production readiness from macOS-only tests. H17 evidence must include exact command, input date/path, output artifact path, and verdict.

## Phase 5: Photometry And VAST Methodology

**Goal:** Finish the VAST-to-DSA methodology wiring without duplicating already-existing metric and multi-epoch code.

**Initial scope:**

- Consolidate `photometry/metrics.py`, `photometry/variability.py`, and `lightcurves/metrics.py`.
- Add only the true missing association module: `photometry/association.py`.
- Reuse existing `photometry/multi_epoch.py` position averaging and new-source significance helpers.
- Wire orphaned multi-epoch stats into batch execution only after unit tests establish canonical formulas.

**Baseline tests:**

```bash
PYTHONPATH=/data/dsa110-continuum \
  /opt/miniforge/envs/casa6/bin/python -m pytest \
  tests/test_variability_metrics.py tests/test_lightcurves.py -v
```

## Campaign Tracking Checklist

- [ ] Publish this plan as a GitHub PR.
- [ ] Create a GitHub tracking issue linking this plan.
- [ ] Pull or fetch the plan branch on H17.
- [ ] Run Phase 0 completion-map dispatch.
- [ ] Review Phase 0 output and choose first implementation branch.
- [ ] Dispatch Phase 1 correctness lane in an isolated branch/worktree.
- [ ] Run targeted H17 validation for the first merged fix.
- [ ] Keep the tracking issue updated after every PR.
