#!/usr/bin/env bash
# Headless Claude Fable 5 via Max OAuth (NOT ANTHROPIC_API_KEY).
set -euo pipefail
cd /data/dsa110-continuum
PLUGIN="/home/ubuntu/.cursor/plugins/marketplaces/github.com/uw-ssec/rse-plugins/db8e73b77528d01d9bad8b99d5d61d42d7601698/plugins/ai-research-workflows"
PROMPT_FILE="/data/dsa110-continuum/outputs/completeness-fix-2026-07-11/dispatch-prompt.md"
LOG_DIR="/data/dsa110-continuum/outputs/completeness-fix-2026-07-11"
mkdir -p "$LOG_DIR"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$LOG_DIR/claude-fable5-$STAMP.log"

# Critical: env ANTHROPIC_API_KEY forces pay-as-you-go and reports "credit balance too low"
# even when Max subscription OAuth has Fable usage remaining.
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN || true

exec env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN   claude -p   --model claude-fable-5   --effort high   --permission-mode acceptEdits   --plugin-dir "$PLUGIN"   --output-format text   --append-system-prompt "You MUST invoke ai-research-workflows skills before acting. Start with using-research-workflows routing, then researching → planning-implementations → implementing-plans → validating-implementations. Direct mode. Repo root is /data/dsa110-continuum. Always use /opt/miniforge/envs/casa6/bin/python with PYTHONPATH=/data/dsa110-continuum. Do not commit or push unless explicitly asked."   "$(cat "$PROMPT_FILE")"   </dev/null 2>&1 | tee "$LOG"
