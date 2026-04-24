# dsacamera-monitor

Operator-facing **static** inventory for DSA-110 `/data/incoming`: a JSON manifest plus a small HTML/JS page (Chart.js) for daily counts, cumulative files, a day-by-day heatmap, gap highlights, and per-beam totals.

This repository was split out from `dsa110-FLITS`; the CLI was renamed from `dsa110-incoming-scan` to **`dsacamera-incoming-scan`**.

## Manifest schema

`manifest.json` is **schema version 2** (v1 is still valid for old snapshots). Key fields:

| Field | Description |
|--------|-------------|
| `schema_version` | Integer (currently `2`) |
| `generated_at` | ISO8601 UTC when the scan finished |
| `source_root` | Directory that was scanned |
| `options.no_stat` | If true, sizes and mtime freshness were not collected |
| `options.hdf5_metadata` | If false (`--no-hdf5-metadata`), HDF5 files are not opened; no `pointing` block |
| `options.pointing_timeseries` | If true, `pointing_timeseries.json` may be emitted (see below) |
| `totals` | `file_count`, `total_bytes` (bytes zero when `no_stat`) |
| `by_day` | `{ date, count, bytes, dec_deg_min?, dec_deg_max?, dec_unique_count? }[]` |
| `by_beam` | `{ beam, count, bytes }[]` sorted by beam id |
| `gaps` | `{ start, end, days }` ranges with zero files between first/last day with data |
| `freshness` | Earliest/latest timestamp from filenames; mtime range when not `no_stat` |
| `pointing` | When metadata scan is on: global Dec min/max, unique rounded strips, file counts |
| `pointing_timeseries` | When `--pointing-timeseries` is used: `{ file, row_count, truncated }` pointing at `pointing_timeseries.json` |

Filenames must match:

`YYYY-MM-DDTHH:MM:SS_sbNN.hdf5`

## Install

From this repository root:

```bash
pip install -e ".[dev]"
```

## Build

```bash
dsacamera-incoming-scan --root /data/incoming --out /path/to/out
```

Or:

```bash
python -m dsacamera_monitor.scan --root /data/incoming --out /path/to/out
```

- Writes `/path/to/out/manifest.json` and copies static assets from `dsacamera_monitor/site/` into `out/`.
- By default each matching `.hdf5` is opened once to read phase-center **Declination** from UVH5 headers (cheap metadata only; no visibilities). Use `--no-hdf5-metadata` to skip that (faster on huge trees, no Dec in manifest).
- Optional `--pointing-timeseries` writes `pointing_timeseries.json` (per file: median time, RA, Dec) with rows capped by `--pointing-timeseries-max-files` (default 5000). RA may be derived from `ha_phase_center` + LST at median time, matching the main pipeline’s HDF5 convention.
- **Performance:** the scan is O(number of files): one `h5py.File` open per file for metadata. On slow NFS, prefer cron spacing or metadata-off for smoke tests.
- Open `out/index.html` in a browser (or serve the directory with any static file server).

### Fast count-only (no `stat`)

Large trees: skip `stat()` to reduce I/O; bytes and mtime will be empty/zero:

```bash
dsacamera-incoming-scan --root /data/incoming --out /path/to/out --no-stat
```

### Output location

If `/data` is full, point `--out` at a filesystem with free space (e.g. `/run/user/$UID/...` or `/tmp/...`).

## Automation

Example cron (hourly scan):

```cron
0 * * * * /usr/bin/env dsacamera-incoming-scan --root /data/incoming --out /home/you/public/incoming-dashboard/ >> /tmp/incoming-scan.log 2>&1
```

Example **systemd** oneshot:

```ini
[Unit]
Description=Regenerate DSA-110 incoming dashboard manifest

[Service]
Type=oneshot
ExecStart=/usr/bin/env dsacamera-incoming-scan --root /data/incoming --out /var/www/incoming-dashboard/
```

## Viewing locally (then expose with a URL)

After `pip install -e .`, the dashboard is just files on disk. Build and serve on **localhost**:

```bash
./scripts/serve_dashboard.sh              # default: scan /data/incoming → ./public, port 8765
./scripts/serve_dashboard.sh --no-stat  # faster scan, no sizes/mtime
PORT=8765 OUT_DIR=/tmp/dash ./scripts/serve_dashboard.sh --no-stat
```

Open `http://127.0.0.1:8765/` (or your `PORT`).

**`pip install` does not create `https://code.deepsynoptic.org/...` by itself.** To get a dedicated URL you typically:

- use **Cloudflare Tunnel** from `dsacamera` to a hostname like `dsacamera.code.deepsynoptic.org` pointing at that local server, and/or
- ask infra for a **reverse-proxy path** such as `code.deepsynoptic.org/dsacamera/`, and/or
- use **GitLab Pages** with CI (see `docs/hosting-deepsynoptic.md` and `examples/gitlab-ci.pages-dsacamera.yml`).

Charts load **Chart.js from jsDelivr CDN**; the page needs outbound HTTPS for the script (or vendor Chart.js into `site/` for offline/air-gapped use).

Full notes: [docs/hosting-deepsynoptic.md](docs/hosting-deepsynoptic.md).

## GitHub Pages (like `dsa110-continuum`)

This repo can publish the static dashboard the same way continuum publishes Quarto docs: **GitHub Actions** builds **`_site/`** and deploys with `actions/upload-pages-artifact` + `actions/deploy-pages` (see [`.github/workflows/pages.yml`](.github/workflows/pages.yml)).

1. On GitHub: **Settings → Pages → Build and deployment → Source: GitHub Actions** (not “Deploy from a branch” unless you prefer that).
2. Register a **self-hosted runner on dsacamera** with labels `self-hosted`, `linux`, and `dsacamera`.
3. Push to **`main`** or **`master`**; the workflow runs on code changes and on a **5-minute schedule**.

**URL shape** matches your other project sites, e.g. `https://code.deepsynoptic.org/dsacamera-monitor/`, once your org’s GitHub / custom domain is set up the same way as [DSA-110 Continuum](http://code.deepsynoptic.org/dsa110-continuum/) (separate from this repo; same Pages pattern).

**Near real-time mode:** scheduled runs use `--no-stat` for faster updates and lower I/O overhead. This gives near-real-time freshness (about every 5 minutes + deployment latency).

**Manual full refresh:** use **Actions → Build and deploy GitHub Pages → Run workflow**, set `full_scan=true` to include bytes + mtime stats (slower, higher I/O).

For one-off local scans on dsacamera:

```bash
dsacamera-incoming-scan --root /data/incoming --out /path/to/_site --no-stat
```

## Tests

```bash
pytest -v
```
