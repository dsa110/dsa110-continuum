# Dashboard operations (qa_server + pipeline control)

The dashboard on port 8767 (`scripts/qa_server.py`) is the single operator surface: read-only
monitoring is open; every mutating route (`POST /api/runs`, `POST /api/runs/{id}/terminate`)
requires a bearer token and is audited. The Dagster tracer-bullet on :3212 remains a separate
read-only metadata view (see `outputs/observability-dashboard-2026-07-14/REPRODUCE.md`).

## Control token

Generate once, store in the environment file the systemd unit loads:

```bash
mkdir -p ~/.config/dsa110
/opt/miniforge/envs/casa6/bin/python -c "import secrets; print(secrets.token_urlsafe(32))" \
  | xargs -I{} printf 'DSA110_CONTROL_TOKEN=%s\n' '{}' > ~/.config/dsa110/dashboard.env
chmod 600 ~/.config/dsa110/dashboard.env
```

The env file must live on a real POSIX filesystem (`~` is ext4). Do NOT put it under `/data`
or `/stage` — both are fuseblk there and ignore chmod, leaving the token world-readable.

Fail-closed: if `DSA110_CONTROL_TOKEN` is unset in the server environment, every mutating
request returns 403 and the dashboard is effectively read-only. Never commit the env file.

## Service installation (one-time, requires sudo)

```bash
sudo cp ops/systemd/dsa110-dashboard.service ops/systemd/dsa110-autopipeline.service \
        ops/systemd/dsa110-autopipeline.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now dsa110-dashboard
sudo systemctl enable --now dsa110-autopipeline.timer
```

Health checks: `systemctl status dsa110-dashboard`, `systemctl list-timers 'dsa110-*'`,
`curl -s http://127.0.0.1:8767/health`.

## Worktree topology

Both production services run `main` from the dedicated worktree at
`/data/dsa110-continuum-production`. Agent work stays in separate worktrees and never changes the
code used by a running service. Create the production worktree once with:

```bash
git -C /data/dsa110-continuum worktree add /data/dsa110-continuum-production main
```

Deploy a new `main` revision explicitly from that worktree:

```bash
git -C /data/dsa110-continuum-production pull --ff-only
sudo cp /data/dsa110-continuum-production/ops/systemd/dsa110-dashboard.service \
        /data/dsa110-continuum-production/ops/systemd/dsa110-autopipeline.service \
        /data/dsa110-continuum-production/ops/systemd/dsa110-autopipeline.timer \
        /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl restart dsa110-dashboard
sudo systemctl restart dsa110-autopipeline.timer
```

Per-artifact QA pages: `/artifacts/caltable/` (BP/G/K tables with acquisition provenance),
`/artifacts/tile/` (single-tile FITS with the image QA gate), `/artifacts/ms/`
(per-MS conversion/UVW/RFI diagnostics). Detail pages render plots lazily and cache them in
`DSA110_QA_THUMBS` keyed on artifact mtime; a plot card returning HTTP 424 names the missing
dependency or product rather than failing the page.

The timer runs `scripts/auto_pipeline.py` daily at 02:00 UTC, which launches
`batch_pipeline.py --date <yesterday UTC> --retry-failed --quarantine-after-failures 3
--photometry-workers 4` through the same registry the dashboard uses. If a launcher-owned run
is still active the timer tick reports `skipped_running` to the journal and does not queue.

## Manual intervention runbook

1. Open `http://lxd110h17:8767/` → Pipeline control panel.
2. Fill in date (and hours / flags as needed), paste the control token.
3. **Dry-run preview** — shows `batch_pipeline.py --dry-run`'s full rebuild/skip/quarantine
   plan; writes nothing.
4. **Launch** — the run appears in the runs table; click its ID for live log tail
   (`/control/runs/{run_id}`, 15 s refresh).
5. **Terminate** — SIGTERMs the whole process group (batch driver, CASA workers, WSClean),
   escalating to SIGKILL after 10 s. Verify with `pgrep -g <pid>` if paranoid.
6. Per-date QA and provenance: `/runs/{date}` (verdict, gate reasons, per-epoch QA).
7. Light curves: `/sources/lightcurve?ra=<deg>&dec=<deg>` or the dashboard lookup form.

Equivalent curl:

```bash
curl -s -X POST http://127.0.0.1:8767/api/runs \
  -H "Authorization: Bearer $DSA110_CONTROL_TOKEN" -H 'Content-Type: application/json' \
  -d '{"date":"2026-01-25","start_hour":22,"end_hour":23,"force_recal":true,"dry_run":true}'
```

## Audit and registry

- Run registry: `/data/dsa110-proc/products/control/runs.sqlite3` (`runs` table; each row
  carries the full request + argv JSON, pid, log path, status, exit code).
- Per-run logs: `/data/dsa110-proc/products/control/run_<id>.log`.
- Audit log: `/data/dsa110-proc/products/control/audit.jsonl` — one JSON line per mutating
  request (timestamp, action, remote address, payload). The token is never written.
- Env overrides: `DSA110_CONTROL_DIR`, `DSA110_REPO_ROOT`, `DSA110_PIPELINE_PYTHON`.

## Network posture

The Cloudflare tunnel (see `outputs/observability-dashboard-2026-07-14/CLOUDFLARE.md`) points
at this origin. Mutating routes are safe to expose only because of the token gate; for
defense in depth, enable Cloudflare Access (SSO) on the tunnel hostname before advertising
the URL beyond the group. LAN access needs no extra step. The former host-ops service
(`scripts/monitor_server.py`, incl. its `POST /exec` shell hook) was deleted under issue #62 —
auth posture is recorded in `docs/adr/0002-observability-auth-posture.md`; no shell-execution
endpoint exists in this repo.
