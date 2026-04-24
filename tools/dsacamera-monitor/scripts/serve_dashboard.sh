#!/usr/bin/env bash
# Regenerate the static dashboard and serve it on localhost (127.0.0.1).
# Pair with Cloudflare Tunnel (or SSH -L) for a stable external URL.
#
# Environment (optional):
#   SCAN_ROOT   default /data/incoming
#   OUT_DIR     default ./public
#   PORT        default 8765
#
# Any extra arguments are passed to dsacamera-incoming-scan (e.g. --no-stat).
#
# Examples:
#   ./scripts/serve_dashboard.sh
#   ./scripts/serve_dashboard.sh --no-stat
#   PORT=9000 OUT_DIR=/tmp/dash ./scripts/serve_dashboard.sh --no-stat

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SCAN_ROOT="${SCAN_ROOT:-/data/incoming}"
OUT_DIR="${OUT_DIR:-./public}"
PORT="${PORT:-8765}"

if ! command -v dsacamera-incoming-scan >/dev/null 2>&1; then
  echo "error: dsacamera-incoming-scan not on PATH (pip install -e .)" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
dsacamera-incoming-scan --root "$SCAN_ROOT" --out "$OUT_DIR" "$@"

echo "Serving $OUT_DIR at http://127.0.0.1:$PORT/ (Ctrl+C to stop)"
cd "$OUT_DIR"
exec python -m http.server "$PORT" --bind 127.0.0.1
