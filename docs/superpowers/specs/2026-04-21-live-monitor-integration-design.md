# Live Monitor Integration Design (`dsa110-continuum`)

## Goal

Integrate the `dsacamera-monitor` near-real-time dashboard into the existing `dsa110-continuum` Quarto docs site so operators can access pipeline documentation and live incoming-file monitoring from one GitHub Pages deployment.

## Scope

### In scope

- Add a `Live Monitor` entry to the Quarto sidebar.
- Publish the generated monitor app at `.../live-monitor/` inside the same Pages artifact as Quarto docs.
- Add near-real-time scheduled refreshes (5-minute cadence) in the docs deployment workflow.
- Support a manual full refresh mode that includes file size and mtime metadata.
- Add CI checks that fail fast if monitor artifacts are missing or malformed.

### Out of scope

- Rewriting the dashboard UI as native Quarto pages.
- Building a streaming backend or websocket service.
- Changing the monitor data model beyond current `manifest.json` schema compatibility.

## Architecture

Use a hybrid static architecture:

1. **Quarto docs** remain the primary site framework and navigation shell.
2. **Monitor app** is generated as static files and mounted under `live-monitor/` in the final `_site`.
3. **Quarto wrapper page** (`live-monitor.qmd`) provides user context, embedding, and fallback navigation.
4. **Single Pages artifact** is deployed, containing both docs content and monitor content, preventing cross-workflow overwrite.

This preserves the existing monitor frontend behavior while giving users a native docs entry point.

## Components and Responsibilities

### Quarto content layer

- Update Quarto sidebar config in `docs/quarto/_quarto.yml`.
- Add `docs/quarto/live-monitor.qmd` with:
  - brief operational context,
  - iframe embed to `/live-monitor/`,
  - fallback direct link to `/live-monitor/` for browsers or settings that block embedding.

### Monitor generation layer

- Reuse `dsacamera-incoming-scan` to generate monitor output.
- Scheduled and push-triggered runs use `--no-stat` for speed.
- Manual dispatch supports `full_scan=true` to run without `--no-stat`.

### Workflow orchestration layer

In `.github/workflows/docs.yml`:

- Add `schedule: "*/5 * * * *"` and `workflow_dispatch` input `full_scan` (boolean).
- Use self-hosted runner labels that can read `/data/incoming` (for example `self-hosted`, `linux`, `dsacamera`).
- Build monitor output in a staging path.
- Render Quarto.
- Copy monitor staging artifacts into `docs/quarto/_site/live-monitor/`.
- Upload and deploy one Pages artifact.

## Build and Data Flow

1. Workflow starts (push/schedule/manual).
2. Checkout repository.
3. Install Python tooling needed for scanner invocation.
4. Build monitor files:
   - `full_scan=true` and manual dispatch: full scan.
   - otherwise: fast scan (`--no-stat`).
5. Render Quarto docs to `docs/quarto/_site`.
6. Validate monitor artifact integrity before merge into final site.
7. Copy monitor directory into `docs/quarto/_site/live-monitor/`.
8. Upload/deploy Pages artifact (non-PR only).

Result: one coherent site with docs and monitor updated together.

## Error Handling and Reliability

- **Fail closed on scan errors:** if scanner cannot read `/data/incoming` or exits non-zero, fail build and skip deploy.
- **Artifact contract checks:** require presence of:
  - `live-monitor/index.html`
  - `live-monitor/manifest.json`
  - expected static asset folders (`css/`, `js/`) or equivalents from scanner output
- **Event gating:** PRs run render/validation without deployment; non-PR events deploy.
- **Concurrency control:** only one deployment in flight for Pages target to avoid race conditions.

## Testing and Verification Strategy

### CI verification

- Add workflow checks for:
  - Quarto render success,
  - monitor artifact existence,
  - basic manifest structure assertions (schema/version and expected top-level keys).

### Runtime smoke tests

- Confirm wrapper page is linked in sidebar.
- Confirm wrapper page loads and embeds `/live-monitor/`.
- Confirm direct route `.../live-monitor/` resolves and assets load with relative paths.

### Operational validation

- Scheduled runs update roughly every 5 minutes plus Pages latency.
- Manual `full_scan=true` path is exercised periodically to prevent drift/failure in the slow path.

## Success Criteria

- `Live Monitor` appears in docs sidebar.
- `.../live-monitor/` is deployed in the same artifact as docs.
- Wrapper page embed and fallback link both work.
- Scheduled near-real-time refreshes are stable.
- CI blocks deployment when monitor generation or artifact integrity checks fail.

## Risks and Mitigations

- **Risk:** self-hosted runner unavailable.
  - **Mitigation:** explicit runner labels, alerting on schedule failures, manual rerun procedure.
- **Risk:** increased workflow runtime due to frequent schedule.
  - **Mitigation:** default `--no-stat`, keep checks lightweight.
- **Risk:** embed blocked by browser policies.
  - **Mitigation:** always include direct-link fallback to `/live-monitor/`.
- **Risk:** docs and monitor path conflicts.
  - **Mitigation:** reserve `live-monitor/` namespace and validate target directory before copy.

## Implementation Boundary for Next Phase

The next phase (`writing-plans`) should produce a task-by-task implementation plan covering:

1. Quarto nav/page additions,
2. workflow trigger and runner updates,
3. monitor build + copy + validation steps,
4. CI verification commands and smoke-test checklist.
