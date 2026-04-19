"""Tests for the epoch QA log (JSONL accumulator)."""
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from dsa110_continuum.qa.epoch_log import append_epoch_qa


def test_append_creates_file():
    """append_epoch_qa creates the JSONL file if it doesn't exist."""
    with tempfile.TemporaryDirectory() as d:
        log_path = Path(d) / "qa_log.jsonl"
        assert not log_path.exists()
        append_epoch_qa({"stage": "test", "value": 1}, log_path=log_path)
        assert log_path.exists()


def test_append_writes_valid_json():
    """Each appended record is a valid JSON object on its own line."""
    with tempfile.TemporaryDirectory() as d:
        log_path = Path(d) / "qa_log.jsonl"
        append_epoch_qa({"stage": "forced_photometry", "rms_mjy": 9.1}, log_path=log_path)
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["stage"] == "forced_photometry"
        assert abs(record["rms_mjy"] - 9.1) < 1e-6


def test_multiple_appends_accumulate():
    """Multiple appends produce one JSON record per line."""
    with tempfile.TemporaryDirectory() as d:
        log_path = Path(d) / "qa_log.jsonl"
        for i in range(5):
            append_epoch_qa({"stage": "test", "epoch": i}, log_path=log_path)
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 5
        epochs = [json.loads(l)["epoch"] for l in lines]
        assert epochs == list(range(5))


def test_readable_as_pandas_dataframe():
    """JSONL file is directly readable by pd.read_json(lines=True)."""
    with tempfile.TemporaryDirectory() as d:
        log_path = Path(d) / "qa_log.jsonl"
        append_epoch_qa({"stage": "source_finding", "n_sources": 42}, log_path=log_path)
        append_epoch_qa({"stage": "stage_c", "n_candidates": 3}, log_path=log_path)
        df = pd.read_json(str(log_path), lines=True)
        assert len(df) == 2
        assert "stage" in df.columns
