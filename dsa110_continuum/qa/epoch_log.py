"""Lightweight epoch-level QA log for the DSA-110 continuum pipeline.

Appends one JSON record per pipeline stage run to a JSONL file
(one JSON object per line). The log is trivially readable as a
time-series DataFrame:

    import pandas as pd
    df = pd.read_json("pipeline_outputs/qa_log.jsonl", lines=True)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)

_DEFAULT_LOG_PATH = Path("pipeline_outputs/qa_log.jsonl")


def append_epoch_qa(
    record: dict,
    log_path: str | Path = _DEFAULT_LOG_PATH,
) -> None:
    """Append one QA record to the epoch QA JSONL log.

    Parameters
    ----------
    record : dict
        Arbitrary key-value pairs for this pipeline stage run.
        A ``"timestamp"`` key is added automatically if not present.
    log_path : str or Path
        Path to the JSONL log file. Created (with parent dirs) if absent.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if "timestamp" not in record:
        record = {**record, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    try:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError as exc:
        log.warning("Could not write epoch QA log %s: %s", log_path, exc)
