from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from dsa110_continuum.orchestration.maistro import (
    MaistroClient,
    MaistroUnavailable,
    parse_canary_output,
    record_canary_result,
)


def test_parse_canary_output_extracts_qa_metrics():
    output = """
=== DSA-110 Canary QA Smoke Test (pre-existing tile) ===
Tile:    /stage/dsa110-contimg/images/mosaic_2026-01-25/example.fits
--- Canary QA Results ---
  Median DSA/NVSS ratio: 1.037
  Recovered sources:     7 / 9
  Completeness:          77.8%
  Mosaic RMS:            11.24 mJy/beam
    Flux scale gate:     PASS
    Completeness gate:   PASS
    Noise floor gate:    PASS
    Overall epoch QA:    PASS
  CANARY: PASS
"""

    parsed = parse_canary_output(output)

    assert parsed == {
        "mosaic_path": "/stage/dsa110-contimg/images/mosaic_2026-01-25/example.fits",
        "median_ratio": 1.037,
        "n_recovered": 7,
        "n_catalog": 9,
        "completeness_frac": 0.778,
        "mosaic_rms_mjy": 11.24,
        "ratio_gate": "PASS",
        "completeness_gate": "PASS",
        "rms_gate": "PASS",
        "qa_result": "PASS",
        "canary_result": "PASS",
    }


def test_record_canary_result_writes_state_batch_without_review_for_pass():
    calls: list[tuple[str, dict]] = []
    client = MaistroClient(
        "http://maistro.local",
        "token",
        transport=lambda p, b: calls.append((p, b)) or {"seqs": [1] * len(b["items"])},
    )
    qa = {
        "mosaic_path": "/stage/example.fits",
        "median_ratio": 1.01,
        "n_recovered": 8,
        "n_catalog": 10,
        "completeness_frac": 0.8,
        "mosaic_rms_mjy": 12.3,
        "ratio_gate": "PASS",
        "completeness_gate": "PASS",
        "rms_gate": "PASS",
        "qa_result": "PASS",
        "canary_result": "PASS",
    }

    result = record_canary_result(
        client,
        run_id="dsa110-canary-test",
        date="2026-01-25",
        command=["bash", "scripts/run_canary.sh"],
        exit_code=0,
        duration_sec=1.2,
        qa=qa,
        stdout_tail="CANARY: PASS",
        stderr_tail="",
    )

    assert result["status"] == "recorded"
    assert [path for path, _ in calls] == ["/write.batch"]
    body = calls[0][1]
    assert body["run_id"] == "dsa110-canary-test"
    by_key = {item["key"]: item["value"] for item in body["items"] if item["op"] == "setState"}
    assert by_key["pipeline.kind"] == "dsa110-continuum-canary"
    assert by_key["stage.canary.status"]["state"] == "completed"
    assert by_key["qa.verdict"] == "PASS"
    assert by_key["artifact.mosaic_path"] == "/stage/example.fits"


def test_record_canary_result_stages_failed_canary_for_review():
    calls: list[tuple[str, dict]] = []

    def transport(path: str, body: dict):
        calls.append((path, body))
        if path == "/write.batch":
            return {"seqs": [1] * len(body["items"])}
        if path == "/stage":
            return {"id": "candidate-1", "status": "pending"}
        raise AssertionError(path)

    client = MaistroClient("http://maistro.local", "token", transport=transport)
    qa = {
        "median_ratio": 1.6,
        "n_recovered": 2,
        "n_catalog": 9,
        "completeness_frac": 0.222,
        "mosaic_rms_mjy": 22.0,
        "ratio_gate": "FAIL",
        "completeness_gate": "FAIL",
        "rms_gate": "FAIL",
        "qa_result": "FAIL",
        "canary_result": "FAIL",
    }

    result = record_canary_result(
        client,
        run_id="dsa110-canary-test",
        date="2026-01-25",
        command=["bash", "scripts/run_canary.sh"],
        exit_code=1,
        duration_sec=2.0,
        qa=qa,
        stdout_tail="CANARY: FAIL",
        stderr_tail="",
    )

    assert result["staged"] == {"id": "candidate-1", "status": "pending"}
    batch_body = calls[0][1]
    by_key = {item["key"]: item["value"] for item in batch_body["items"] if item["op"] == "setState"}
    assert by_key["stage.canary.status"]["state"] == "failed"
    assert by_key["qa.verdict"] == "FAIL"
    stage_body = calls[1][1]
    assert calls[1][0] == "/stage"
    assert stage_body["kind"] == "procedural"
    assert stage_body["payload"]["key"].startswith("review.dsa110.canary.")
    assert stage_body["payload"]["value"]["qa_result"] == "FAIL"


def test_record_canary_result_marks_epoch_qa_failure_as_verdict_fail():
    calls: list[tuple[str, dict]] = []

    def transport(path: str, body: dict):
        calls.append((path, body))
        if path == "/write.batch":
            return {"seqs": [1] * len(body["items"])}
        if path == "/stage":
            return {"id": "candidate-epoch", "status": "pending"}
        raise AssertionError(path)

    client = MaistroClient("http://maistro.local", "token", transport=transport)
    qa = {
        "median_ratio": 1.13,
        "n_recovered": 22,
        "n_catalog": 60,
        "completeness_frac": 0.367,
        "mosaic_rms_mjy": 14.62,
        "ratio_gate": "PASS",
        "completeness_gate": "FAIL",
        "rms_gate": "PASS",
        "qa_result": "FAIL",
        "canary_result": "PASS",
    }

    record_canary_result(
        client,
        run_id="dsa110-canary-real-shape",
        date="2026-01-25",
        command=["bash", "scripts/run_canary.sh"],
        exit_code=0,
        duration_sec=34.3,
        qa=qa,
        stdout_tail="CANARY: PASS",
        stderr_tail="",
    )

    batch_body = calls[0][1]
    by_key = {item["key"]: item["value"] for item in batch_body["items"] if item["op"] == "setState"}
    assert by_key["stage.canary.status"]["state"] == "failed"
    assert by_key["qa.verdict"] == "FAIL"


def test_maistro_client_requires_url_and_token():
    with pytest.raises(MaistroUnavailable):
        MaistroClient.from_env(env={})


def test_run_maistro_canary_cli_nonfatal_when_maistro_unconfigured(tmp_path):
    command = tmp_path / "fake_canary.py"
    command.write_text(
        "import sys\n"
        "print('Tile:    /stage/example.fits')\n"
        "print('  Median DSA/NVSS ratio: 1.0')\n"
        "print('  Recovered sources:     3 / 3')\n"
        "print('  Completeness:          100.0%')\n"
        "print('  Mosaic RMS:            10.0 mJy/beam')\n"
        "print('    Flux scale gate:     PASS')\n"
        "print('    Completeness gate:   SKIP')\n"
        "print('    Noise floor gate:    PASS')\n"
        "print('    Overall epoch QA:    PASS')\n"
        "print('  CANARY: PASS')\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_maistro_canary.py",
            "--run-id",
            "cli-test",
            "--no-maistro",
            "--",
            sys.executable,
            str(command),
        ],
        cwd=Path(__file__).parent.parent,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0
    assert '"run_id": "cli-test"' in proc.stdout
    payload = json.loads(proc.stdout.rsplit("\n", 2)[-2])
    assert payload["maistro"]["status"] == "disabled"
    assert payload["qa"]["canary_result"] == "PASS"

