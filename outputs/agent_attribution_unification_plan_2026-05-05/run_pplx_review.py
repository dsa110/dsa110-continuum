#!/opt/miniforge/envs/casa6/bin/python
"""Run perplexity_research + perplexity_reason on the plan and save outputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/ubuntu/.claude/skills")
from pplx_mcp_client import _call  # type: ignore[import-not-found]

PLAN_PATH = Path("/data/dsa110-continuum/outputs/agent_attribution_unification_plan_2026-05-05/plan.md")
OUT_DIR = PLAN_PATH.parent

mode = sys.argv[1]  # "research" or "reason"
plan = PLAN_PATH.read_text()
prompt = (
    f"Critically review the following plan for unifying GitHub contributors-list attribution "
    f"to a single user via git-filter-repo. Identify technical errors, missing safety steps, "
    f"better alternatives, and edge cases not considered. Be specific about line-level issues "
    f"with the proposed git-filter-repo invocation, the message-callback regex, the mailmap "
    f"format, the force-push strategy, and the branch-reconciliation approach. Cite sources.\n\n"
    f"PLAN:\n\n{plan}"
)

tool = f"perplexity_{mode}"
timeout = 360 if mode == "research" else 240
result = _call(tool, prompt, timeout=timeout)

out_path = OUT_DIR / f"review_pplx_{mode}.json"
out_path.write_text(json.dumps(result, indent=2))
print(f"wrote {out_path}")
