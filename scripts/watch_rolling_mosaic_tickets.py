#!/usr/bin/env python3
"""Close pipeline-optimization tickets only when their live evidence exists."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

REPO = "dsa110/dsa110-continuum"
MAP_ISSUE = 144
_CAMPAIGN_DIR = Path(
    __import__("os").environ.get(
        "DSA110_CAMPAIGN_DIR",
        str(Path(__file__).resolve().parent.parent / "outputs/jan-apr-mosaic-campaign-2026-07-21"),
    )
)
STATUS_PATH = _CAMPAIGN_DIR / "status.json"
HOUR22_PHOTOMETRY = Path(
    "/data/dsa110-proc/products/mosaics/2026-01-25/2026-01-25T2200_forced_phot.csv"
)
POLL_SECONDS = 60

FIRST_EPOCH_TICKETS = {
    171: (
        "Keep the rolling Measurement Set working set on NVMe",
        "The immediate campaign produced its first strict-QA/photometry epoch with the "
        "bounded NVMe working set and reserve policy.",
    ),
    172: (
        "Canonicalize NVSS and VLASS catalog ownership and lookup",
        "The canonical current-pipeline catalogs produced a live strict-QA epoch without "
        "lenient thresholds.",
    ),
    173: (
        "Certify two-process tile concurrency on H17",
        "The certified two-process mode produced a live strict-QA/photometry epoch.",
    ),
}


def run_gh(*args: str, input_text: str | None = None) -> str:
    result = subprocess.run(
        ["gh", *args],
        input=input_text,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def append_map_decision(issue: int, title: str, gist: str) -> None:
    issue_url = f"https://github.com/dsa110/dsa110-continuum/issues/{issue}"
    line = f"- [{title}]({issue_url}) — {gist}"
    payload = json.loads(run_gh("api", f"repos/{REPO}/issues/{MAP_ISSUE}"))
    body = payload["body"]
    if line in body:
        return
    marker = "\n## Not yet specified"
    if marker not in body:
        raise RuntimeError("Wayfinder map is missing its Not yet specified section")
    body = body.replace(marker, f"\n{line}\n{marker}", 1)
    run_gh(
        "api",
        "--method",
        "PATCH",
        "--input",
        "-",
        f"repos/{REPO}/issues/{MAP_ISSUE}",
        input_text=json.dumps({"body": body}),
    )


def close_ticket(issue: int, title: str, gist: str, evidence: str) -> None:
    state = json.loads(
        run_gh("issue", "view", str(issue), "--repo", REPO, "--json", "state")
    )["state"]
    if state == "CLOSED":
        return
    run_gh(
        "issue",
        "close",
        str(issue),
        "--repo",
        REPO,
        "--comment",
        f"{gist} Evidence: {evidence}",
    )
    append_map_decision(issue, title, gist)


def resolve_ready_tickets(status: dict) -> None:
    completed = set(status.get("completed", []))
    if completed:
        first_epoch = sorted(completed)[0]
        evidence = (
            f"campaign status records {first_epoch} complete after strict QA and photometry; "
            "see outputs/pipeline-optimization-2026-07-21/RESULTS.md"
        )
        for issue, (title, gist) in FIRST_EPOCH_TICKETS.items():
            close_ticket(issue, title, gist, evidence)

    if "2026-01-25T2200" in completed and HOUR22_PHOTOMETRY.is_file():
        close_ticket(
            157,
            "Restore strict flux-scale QA on 2026-01-25 hour 22",
            "The same hour-22 checkpoint passed strict epoch QA and wrote forced photometry.",
            str(HOUR22_PHOTOMETRY),
        )


def main() -> None:
    while True:
        if STATUS_PATH.exists():
            status = json.loads(STATUS_PATH.read_text())
            resolve_ready_tickets(status)
            if status.get("state") in {"complete", "failed"}:
                return
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
