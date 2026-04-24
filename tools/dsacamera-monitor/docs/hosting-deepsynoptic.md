# Hosting on `code.deepsynoptic.org` (or nearby)

## GitHub Pages (same idea as `dsa110-continuum`)

The [DSA-110 Continuum](http://code.deepsynoptic.org/dsa110-continuum/) docs use **GitHub Actions** to build a site and deploy to **GitHub Pages**; this repository includes [`.github/workflows/pages.yml`](../.github/workflows/pages.yml) in the same style: build → `upload-pages-artifact` → `deploy-pages`.

After you enable **Pages → GitHub Actions** in the `dsacamera-monitor` repository settings, the site is typically available under your org’s GitHub Pages host, e.g. **`https://code.deepsynoptic.org/dsacamera-monitor/`** if the custom domain and project paths match your existing continuum project.

This repository is now configured for a **self-hosted runner on dsacamera** (`runs-on: [self-hosted, linux, dsacamera]`) so the workflow performs a **real** scan:

- Scheduled every 5 minutes (`cron: */5 * * * *`) using `--no-stat` for speed.
- Push-triggered deploys also use `--no-stat` for fast turnaround.
- Manual runs can set `full_scan=true` to include file sizes/mtime (slower).

### Self-hosted runner setup checklist

1. In GitHub repo settings, create a self-hosted runner and install it on `dsacamera`.
2. Add labels: `linux`, `dsacamera` (plus default `self-hosted`).
3. Ensure runner service account can read `/data/incoming`.
4. Ensure Python 3.11+ and `pip` are available on the runner host.
5. Keep disk headroom for `_site` artifact creation during each run.

### About “real-time”

GitHub Pages deployments are **not true streaming real-time**. With a 5-minute schedule you typically get updates in:

- scan time
- plus artifact upload/deploy latency

So “near real-time” is realistic; sub-minute updates usually require a direct web service rather than Pages.

---

`pip install` only installs the **scanner** and static **site templates**. It does **not** register a URL on your GitLab host. You get a public URL only after you **publish** the generated `public/` (or `out/`) directory through one of the patterns below.

The dashboard uses **relative** asset paths (`manifest.json`, `css/`, `js/`), so it works when served from:

- the **root** of a site (`https://example.com/`)
- a **subpath** (`https://example.com/dsacamera/`) as long as the server maps that path to the folder that contains `index.html` and `manifest.json` together

---

## Option 1: Subdomain via Cloudflare Tunnel (good for “run on dsacamera”)

Typical pattern: **`https://dsacamera.code.deepsynoptic.org`** (or `dsacamera.deepsynoptic.org`) pointing at a **small local HTTP server** on the camera machine.

1. On `dsacamera`, regenerate the site periodically (cron) into a directory, e.g. `/var/lib/dsacamera-dashboard/public` or `/home/you/dashboard-public`.
2. Run a static server bound to **localhost** only:

   ```bash
   cd /path/to/public && python -m http.server 8765 --bind 127.0.0.1
   ```

3. Install [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) and create a **Cloudflare Tunnel** in the Zero Trust dashboard.
4. Add a **public hostname** for your chosen name, e.g. `dsacamera.code.deepsynoptic.org`, with service **`http://127.0.0.1:8765`**.
5. Add **Cloudflare Access** (SSO) on that hostname if the dashboard should not be world-readable.

**Why not `code.deepsynoptic.org/dsacamera` with Tunnel alone?**  
Cloudflare Tunnel routes by **hostname** by default. Serving a **path** on the **same** hostname as GitLab usually requires either **Cloudflare Workers** path routing, or **nginx** on the GitLab server — that is an **org/infrastructure** change, not something this repo can “turn on” from Python.

---

## Option 2: Path `https://code.deepsynoptic.org/dsacamera/`

You need whoever runs **`code.deepsynoptic.org`** to add a reverse-proxy rule, for example:

- Map **`/dsacamera/`** to a static file root (where you rsync the scanner output), or
- Map **`/dsacamera/`** to an internal origin that serves the same files.

GitLab itself does not expose arbitrary path prefixes for random static trees on the main UI hostname; that is normal **front-door nginx/Apache** configuration.

---

## Option 3: GitLab Pages (project URL, not necessarily `/dsacamera`)

If your instance has **GitLab Pages** enabled, a common pattern is:

- URL like `https://<namespace>.pages.code.deepsynoptic.org/<project>/`  
  (exact pattern depends on GitLab version and admin settings — check **Deploy → Pages** in the project.)

The pipeline must produce a **`public/`** artifact. Because `/data/incoming` exists only on the camera, you usually need either:

- a **GitLab Runner with tag** `dsacamera` (or similar) that runs on that machine, or  
- a job that publishes **pre-built** `public/` committed or uploaded as an artifact (less ideal for huge manifests).

See [`examples/gitlab-ci.pages-dsacamera.yml`](../examples/gitlab-ci.pages-dsacamera.yml) for a runner-on-dsacamera template.

---

## Local helper

[`scripts/serve_dashboard.sh`](../scripts/serve_dashboard.sh) rebuilds the site and serves it on `127.0.0.1` for quick checks; pair it with **Tunnel** for a stable HTTPS URL.

---

## Summary

| Goal | Typical approach |
|------|------------------|
| Stable HTTPS from dsacamera | Cloudflare Tunnel → `http://127.0.0.1:PORT` |
| Under `code.deepsynoptic.org/dsacamera/` | Org reverse proxy + static root |
| GitLab-native URL | GitLab Pages + CI (runner on dsacamera or published `public/`) |
