"""Maistro control-plane bridge for DSA-110 continuum pipeline runs.

This module deliberately stays out of the compute path. Pipeline scripts keep
owning CASA/WSClean work; Maistro only receives deterministic run facts and
staged review candidates.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any

Transport = Callable[[str, dict[str, Any]], dict[str, Any]]


class MaistroUnavailable(RuntimeError):
    """Raised when Maistro RPC is not configured for this process."""


class MaistroWriteError(RuntimeError):
    """Raised when the Maistro RPC writer rejects a request."""


class MaistroClient:
    """Tiny HTTP client for the Maistro RPC writer.

    The pipeline is Python-only on h17, while Maistro itself is TypeScript. This
    client mirrors the small subset of ``RemoteMemory`` needed by operational
    wrappers without introducing a Node dependency into the CASA environment.
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_sec: float = 10.0,
        transport: Transport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_sec = timeout_sec
        self._transport = transport

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        base_url: str | None = None,
        token: str | None = None,
        timeout_sec: float = 10.0,
    ) -> "MaistroClient":
        """Build a client from explicit values or Maistro environment variables."""
        source = env if env is not None else os.environ
        resolved_url = base_url or source.get("ORCH_RPC_URL") or source.get("MAISTRO_RPC_URL")
        resolved_token = token or source.get("ORCH_RPC_TOKEN") or source.get("MAISTRO_RPC_TOKEN")
        if not resolved_url or not resolved_token:
            raise MaistroUnavailable("Maistro requires ORCH_RPC_URL and ORCH_RPC_TOKEN")
        return cls(resolved_url, resolved_token, timeout_sec=timeout_sec)

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST a JSON request to one RPC endpoint."""
        if self._transport is not None:
            return self._transport(path, body)

        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "authorization": f"Bearer {self.token}",
                "content-type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:
                payload = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise MaistroWriteError(f"{path} {exc.code}: {detail}") from exc
        except OSError as exc:
            raise MaistroWriteError(f"{path}: {exc}") from exc

        return json.loads(payload) if payload else {}

    def write_batch(
        self,
        run_id: str,
        items: list[dict[str, Any]],
        *,
        agent_id: str = "dsa110-continuum",
        correlation_id: str | None = None,
    ) -> dict[str, Any]:
        """Write several deterministic projection updates in one RPC transaction."""
        body: dict[str, Any] = {"run_id": run_id, "items": items, "agent_id": agent_id}
        if correlation_id:
            body["correlation_id"] = correlation_id
        return self.post("/write.batch", body)

    def stage(
        self,
        run_id: str,
        kind: str,
        payload: dict[str, Any],
        *,
        derived_from: list[int] | None = None,
    ) -> dict[str, Any]:
        """Stage a derived or review-required candidate in Maistro."""
        return self.post(
            "/stage",
            {
                "run_id": run_id,
                "kind": kind,
                "payload": payload,
                "derived_from": derived_from or [],
            },
        )


def _match_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, re.MULTILINE)
    return float(match.group(1)) if match else None


def _match_int_pair(pattern: str, text: str) -> tuple[int, int] | None:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _match_word(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, re.MULTILINE)
    return match.group(1).upper() if match else None


def parse_canary_output(output: str) -> dict[str, Any]:
    """Extract stable QA fields from ``scripts/run_canary.sh`` output."""
    parsed: dict[str, Any] = {}
    tile_match = re.search(r"^Tile:\s*(\S+)", output, re.MULTILINE)
    if tile_match:
        parsed["mosaic_path"] = tile_match.group(1)

    median_ratio = _match_float(r"Median DSA/NVSS ratio:\s*([0-9.+\-eE]+)", output)
    if median_ratio is not None:
        parsed["median_ratio"] = median_ratio

    recovered = _match_int_pair(r"Recovered sources:\s*(\d+)\s*/\s*(\d+)", output)
    if recovered:
        parsed["n_recovered"], parsed["n_catalog"] = recovered

    completeness_percent = _match_float(r"Completeness:\s*([0-9.+\-eE]+)%", output)
    if completeness_percent is not None:
        parsed["completeness_frac"] = round(completeness_percent / 100.0, 4)

    rms = _match_float(r"Mosaic RMS:\s*([0-9.+\-eE]+)\s*mJy/beam", output)
    if rms is not None:
        parsed["mosaic_rms_mjy"] = rms

    gate_patterns = {
        "ratio_gate": r"Flux scale gate:\s*(PASS|FAIL|SKIP)",
        "completeness_gate": r"Completeness gate:\s*(PASS|FAIL|SKIP)",
        "rms_gate": r"Noise floor gate:\s*(PASS|FAIL|SKIP)",
        "qa_result": r"Overall epoch QA:\s*(PASS|FAIL|SKIP)",
        "canary_result": r"CANARY:\s*(PASS|FAIL)",
    }
    for key, pattern in gate_patterns.items():
        value = _match_word(pattern, output)
        if value is not None:
            parsed[key] = value
    return parsed


def _set_state(key: str, value: Any) -> dict[str, Any]:
    return {"op": "setState", "key": key, "value": value}


def _tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _qa_failed(exit_code: int, qa: Mapping[str, Any]) -> bool:
    return exit_code != 0 or qa.get("canary_result") == "FAIL" or qa.get("qa_result") == "FAIL"


def record_canary_result(
    client: MaistroClient,
    *,
    run_id: str,
    date: str,
    command: Sequence[str],
    exit_code: int,
    duration_sec: float,
    qa: Mapping[str, Any],
    stdout_tail: str,
    stderr_tail: str,
) -> dict[str, Any]:
    """Write canary state and stage failed/ambiguous outcomes for review."""
    status = "failed" if _qa_failed(exit_code, qa) else "completed"
    now_ms = int(time.time() * 1000)
    command_list = list(command)
    verdict = "FAIL" if status == "failed" else qa.get("canary_result") or qa.get("qa_result") or "UNKNOWN"

    items = [
        _set_state("pipeline.kind", "dsa110-continuum-canary"),
        _set_state("pipeline.date", date),
        _set_state("pipeline.command", command_list),
        _set_state(
            "stage.canary.status",
            {
                "state": status,
                "exit_code": exit_code,
                "duration_sec": round(duration_sec, 3),
                "updated_at_ms": now_ms,
            },
        ),
        _set_state("qa.verdict", verdict),
        _set_state(
            "qa.gates",
            {
                "ratio": qa.get("ratio_gate"),
                "completeness": qa.get("completeness_gate"),
                "rms": qa.get("rms_gate"),
                "epoch": qa.get("qa_result"),
            },
        ),
        _set_state(
            "qa.metrics",
            {
                "median_ratio": qa.get("median_ratio"),
                "n_recovered": qa.get("n_recovered"),
                "n_catalog": qa.get("n_catalog"),
                "completeness_frac": qa.get("completeness_frac"),
                "mosaic_rms_mjy": qa.get("mosaic_rms_mjy"),
            },
        ),
        _set_state("artifact.stdout_tail", _tail(stdout_tail)),
        _set_state("artifact.stderr_tail", _tail(stderr_tail)),
    ]
    if qa.get("mosaic_path"):
        items.append(_set_state("artifact.mosaic_path", qa["mosaic_path"]))

    client.write_batch(run_id, items, agent_id="dsa110-continuum:maistro-canary")

    response: dict[str, Any] = {"status": "recorded"}
    if _qa_failed(exit_code, qa):
        review_value = {
            "date": date,
            "command": command_list,
            "exit_code": exit_code,
            "qa_result": qa.get("qa_result"),
            "canary_result": qa.get("canary_result"),
            "metrics": dict(qa),
            "reason": "canary failed or returned a failed QA gate",
        }
        staged = client.stage(
            run_id,
            "procedural",
            {
                "key": f"review.dsa110.canary.{now_ms}",
                "value": review_value,
            },
        )
        response["staged"] = staged
    return response

