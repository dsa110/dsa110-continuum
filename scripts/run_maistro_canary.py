#!/opt/miniforge/envs/casa6/bin/python
"""Run the DSA-110 QA canary and mirror the result into Maistro."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dsa110_continuum.orchestration.maistro import (  # noqa: E402
    MaistroClient,
    MaistroUnavailable,
    MaistroWriteError,
    parse_canary_output,
    record_canary_result,
)


def _default_run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"dsa110-canary-{stamp}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run scripts/run_canary.sh and write a Maistro control-plane record."
    )
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--date", default="2026-01-25")
    parser.add_argument("--rpc-url", default=os.environ.get("ORCH_RPC_URL") or os.environ.get("MAISTRO_RPC_URL"))
    parser.add_argument("--rpc-token", default=os.environ.get("ORCH_RPC_TOKEN") or os.environ.get("MAISTRO_RPC_TOKEN"))
    parser.add_argument("--strict-maistro", action="store_true", help="fail if the Maistro write fails")
    parser.add_argument("--no-maistro", action="store_true", help="run and parse canary without RPC writes")
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="optional command after --; defaults to bash scripts/run_canary.sh",
    )
    return parser.parse_args()


def _normalize_command(raw: list[str]) -> list[str]:
    if raw and raw[0] == "--":
        raw = raw[1:]
    return raw or ["bash", "scripts/run_canary.sh"]


def main() -> int:
    """Run the canary command, parse output, and write Maistro state."""
    args = _parse_args()
    command = _normalize_command(args.command)

    started = time.monotonic()
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    duration = time.monotonic() - started

    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)

    qa = parse_canary_output(proc.stdout)
    summary: dict[str, object] = {
        "run_id": args.run_id,
        "date": args.date,
        "command": command,
        "exit_code": proc.returncode,
        "duration_sec": round(duration, 3),
        "qa": qa,
    }

    if args.no_maistro:
        summary["maistro"] = {"status": "disabled"}
    else:
        try:
            client = MaistroClient.from_env(base_url=args.rpc_url, token=args.rpc_token)
            summary["maistro"] = record_canary_result(
                client,
                run_id=args.run_id,
                date=args.date,
                command=command,
                exit_code=proc.returncode,
                duration_sec=duration,
                qa=qa,
                stdout_tail=proc.stdout,
                stderr_tail=proc.stderr,
            )
        except (MaistroUnavailable, MaistroWriteError) as exc:
            summary["maistro"] = {"status": "error", "error": str(exc)}
            if args.strict_maistro:
                print(json.dumps(summary, sort_keys=True))
                return 1

    print(json.dumps(summary, sort_keys=True))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
