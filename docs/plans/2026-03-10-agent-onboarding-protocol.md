# Agent Onboarding Protocol Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a canonical agent-onboarding document to the Quarto docs site so new AI tools and collaborators can gain the right scientific and operational context quickly without rediscovering the pipeline state from scratch.

**Architecture:** The onboarding protocol should live as a human-maintained Quarto page in `docs/quarto/`, linked from the site nav and the docs home page. It should point readers to the authoritative sources of truth (`CLAUDE.md`, `TODOS.md`, and the existing Quarto pages), separate stable context from volatile context, and define what evidence an agent must gather before claiming success. Do **not** build the machine-generated context snapshot in this task; document it as a future extension only.

**Tech Stack:** Quarto `.qmd`, Markdown, existing Quarto site config in `docs/quarto/_quarto.yml`

---

### Task 1: Create the onboarding page skeleton

**Files:**
- Create: `docs/quarto/agent-onboarding.qmd`
- Reference: `docs/quarto/index.qmd`
- Reference: `docs/quarto/pipeline-overview.qmd`
- Reference: `CLAUDE.md`
- Reference: `TODOS.md`

**Step 1: Verify the page does not already exist**

Run: `rg "agent-onboarding\.qmd|Agent Onboarding" docs/quarto -n`

Expected: no matches for the new page filename.

**Step 2: Draft the page structure before filling content**

Use this exact section outline:

```markdown
---
title: "Agent Onboarding"
---

## Purpose

## What This Pipeline Is Scientifically Trying To Do

## What Is Currently Trusted

## What Is Not Yet Trusted

## Authoritative Files To Read First

## Stable Context vs Volatile Context

## Standard Onboarding Protocol

## Evidence Required Before Claiming Success

## Future Extension: Machine-Generated Context Snapshot
```

**Step 3: Write the minimal page skeleton**

Create `docs/quarto/agent-onboarding.qmd` with the front matter and headings from Step 2, plus a short opening paragraph like:

```markdown
This page explains how a new agent or AI tool should acquire context in
`dsa110-continuum` without confusing validated science behavior, active
development state, and future plans.
```

**Step 4: Verify the file exists with the expected title**

Run: `rg "^title:|^## " docs/quarto/agent-onboarding.qmd -n`

Expected: the title plus all planned headings appear.

**Step 5: Commit**

```bash
git add docs/quarto/agent-onboarding.qmd
git commit -m "docs: add agent onboarding page skeleton"
```

---

### Task 2: Fill in the scientific and operational context

**Files:**
- Modify: `docs/quarto/agent-onboarding.qmd`
- Reference: `CLAUDE.md`
- Reference: `TODOS.md`
- Reference: `docs/quarto/pipeline-overview.qmd`
- Reference: `docs/quarto/epoch-qa.qmd`
- Reference: `docs/quarto/outputs-and-artifacts.qmd`

**Step 1: Write the scientific framing section**

Add a short section that explains, in astronomy terms:
- the pipeline exists to turn drift-scan visibilities into calibrated images, mosaics, QA-qualified epoch products, and eventually light curves
- the key present scientific question is whether fluxes and known-source recovery are trustworthy enough to support multi-epoch variability work
- the current validated regime is limited and should be described honestly

Use concrete language such as:

```markdown
The central scientific question is not only whether the software runs, but
whether the flux scale, source recovery, and QA verdicts are attributable to
the code in this repository and are trustworthy enough for multi-epoch
variability analysis.
```

**Step 2: Write "trusted" and "not yet trusted" sections**

Document both of these explicitly:

- **Currently trusted**
  - single-epoch/single-strip production is possible in the validated strip
  - epoch QA infrastructure exists
  - known bright sources can be recovered and inspected
- **Not yet fully trusted**
  - bulk multi-epoch production
  - all import paths being local until the import migration is complete
  - performance-sensitive parts of mosaicking
  - wider-strip calibration coverage

Keep the wording concise and evidence-based.

**Step 3: Add the authoritative-files section**

List these files with one-line purposes:

```markdown
- `CLAUDE.md`: top-level repository truth, runtime environment, silent-failure risks
- `TODOS.md`: current milestone state, next priorities, and known open work
- `docs/quarto/pipeline-overview.qmd`: architecture and stage-level pipeline understanding
- `docs/quarto/epoch-qa.qmd`: QA gates, canary logic, and threshold definitions
- `docs/quarto/outputs-and-artifacts.qmd`: where products land and what to inspect when debugging
```

**Step 4: Add the stable-vs-volatile-context section**

Document:
- **Stable context**: architecture, silent-failure risks, reference docs, key data paths
- **Volatile context**: latest milestone state, most recent canary result, current blockers, run summaries, current dirty worktree state

Include a sentence that the onboarding protocol should always combine both.

**Step 5: Commit**

```bash
git add docs/quarto/agent-onboarding.qmd
git commit -m "docs: add scientific and operational onboarding context"
```

---

### Task 3: Write the onboarding protocol itself

**Files:**
- Modify: `docs/quarto/agent-onboarding.qmd`
- Reference: `docs/quarto/index.qmd`
- Reference: `docs/quarto/pipeline-overview.qmd`
- Reference: `docs/quarto/epoch-qa.qmd`
- Reference: `docs/quarto/outputs-and-artifacts.qmd`

**Step 1: Add the standard onboarding checklist**

Write a concrete, ordered checklist for any new agent/tool. It should say that the agent must determine:

1. the scientific goal
2. what pipeline state is already trusted
3. what code paths are known silent-failure risks
4. what outputs/artifacts matter
5. what task is in scope right now
6. what evidence is required before success can be claimed

Use a compact numbered list, not prose only.

**Step 2: Add an "evidence before success" section**

State that agents should not claim completion without recording:
- files changed
- commands actually run
- tests/lints actually run
- what remains unresolved
- what the result means scientifically

Include a short example block like:

```markdown
- **What changed**: exact files edited
- **What proves it**: exact commands/tests run
- **What remains unresolved**: known caveats
- **Scientific significance**: what this does and does not let us trust
```

**Step 3: Add the deferred machine-snapshot section**

Document the future extension without implementing it yet. Say that a later script such as
`scripts/generate_context_snapshot.py` could emit `outputs/context/current_state.json` containing:
- git commit / branch
- dirty files
- latest canary result
- latest QA summary highlights
- active milestone / blockers
- key artifact paths

Be explicit that this is deferred until the manually maintained protocol has proven useful.

**Step 4: Add brief cross-links back to the existing docs**

At the bottom, add a short "See also" list linking back to:
- `index.qmd`
- `pipeline-overview.qmd`
- `epoch-qa.qmd`
- `outputs-and-artifacts.qmd`

**Step 5: Commit**

```bash
git add docs/quarto/agent-onboarding.qmd
git commit -m "docs: add agent onboarding protocol"
```

---

### Task 4: Wire the page into the Quarto site

**Files:**
- Modify: `docs/quarto/_quarto.yml`
- Modify: `docs/quarto/index.qmd`
- Reference: `docs/quarto/agent-onboarding.qmd`

**Step 1: Update the sidebar navigation**

Add the new page to `docs/quarto/_quarto.yml` in a sensible reading order. Recommended order:

```yaml
sidebar:
  style: "docked"
  contents:
    - index.qmd
    - agent-onboarding.qmd
    - pipeline-overview.qmd
    - epoch-qa.qmd
    - outputs-and-artifacts.qmd
```

**Step 2: Update the docs home page**

In `docs/quarto/index.qmd`, add the new page to the `## Pages` table with a description such as:

```markdown
| [Agent Onboarding](agent-onboarding.qmd) | How a new AI tool or collaborator should acquire trustworthy pipeline context |
```

Optionally add one sentence near the top of the page saying that new agents should start there before reading the deeper docs.

**Step 3: Verify navigation references**

Run: `rg "agent-onboarding\.qmd|Agent Onboarding" docs/quarto -n`

Expected: matches in `_quarto.yml`, `index.qmd`, and `agent-onboarding.qmd`.

**Step 4: Commit**

```bash
git add docs/quarto/_quarto.yml docs/quarto/index.qmd docs/quarto/agent-onboarding.qmd
git commit -m "docs: add agent onboarding page to Quarto nav"
```

---

### Task 5: Validate the docs change without Quarto

**Files:**
- Modify if needed: `docs/quarto/agent-onboarding.qmd`
- Modify if needed: `docs/quarto/index.qmd`
- Modify if needed: `docs/quarto/_quarto.yml`

**Step 1: Run structural checks**

Run these commands:

```bash
rg "^title:|^## |agent-onboarding\.qmd|^\-\s+index\.qmd|^\-\s+agent-onboarding\.qmd" docs/quarto -n
```

Expected:
- the new page has a title and planned headings
- the new page is linked from `_quarto.yml`
- the new page is listed on the home page

**Step 2: Check for broken obvious cross-links**

Run:

```bash
rg "\]\((index|pipeline-overview|epoch-qa|outputs-and-artifacts|agent-onboarding)\.qmd" docs/quarto -n
```

Expected: only links to files that actually exist in `docs/quarto/`.

**Step 3: If Quarto is available, render; otherwise document the limitation**

If `quarto` is installed:

```bash
cd docs/quarto && quarto render
```

Expected: successful site build with no broken-page errors.

If Quarto is not installed, add or preserve a short note in the page or final report saying that the change was validated structurally but not rendered on this machine.

**Step 4: Run a light lint/format check if needed**

If there is a markdown or docs lint command already in repo practice, run it.
If not, skip and say why in the final report.

**Step 5: Commit**

```bash
git add docs/quarto/_quarto.yml docs/quarto/index.qmd docs/quarto/agent-onboarding.qmd
git commit -m "docs: validate agent onboarding docs wiring"
```

---

### Task 6: Optional follow-up (do not implement in this task unless requested)

**Files:**
- Future create: `scripts/generate_context_snapshot.py`
- Future create: `outputs/context/current_state.json`

**Step 1: Do not implement this yet**

Leave this as a documented future extension only.

**Step 2: Capture the future direction in the final report**

State that if the onboarding page proves useful in real usage, the next logical extension is a machine-generated snapshot that captures:
- git state
- latest canary result
- latest QA summary highlights
- active milestone / blockers
- key artifact paths

This should remain secondary to the human-maintained onboarding protocol.

---

## Implementation notes for the executing agent

- Use `docs/quarto/agent-onboarding.qmd` as the canonical artifact for now.
- Do **not** create a Python script in this task.
- Keep the page concise and operator/developer-facing, not academic.
- Prefer astronomy-facing language over generic SWE language wherever possible.
- Do not overclaim what the pipeline can currently do.
- Reuse the style already established in the Quarto docs: short sections, direct prose, and explicit caveats where needed.
- If you need to soften certainty, do it in favor of scientific honesty.

