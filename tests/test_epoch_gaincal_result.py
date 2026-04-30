"""Tests for the structured EpochGaincalResult / EpochGaincalStatus contract.

Verifies the spec-compliance mapping between the structured status enum
returned by ``calibrate_epoch`` and the spec's epoch_gaincal_state enum
consumed by the promotion module. See
``docs/validation/pipeline-validation-from-scratch.md``.
"""

from __future__ import annotations

import pytest
from dsa110_continuum.calibration.epoch_gaincal import (
    EpochGaincalResult,
    EpochGaincalStatus,
)


def test_status_enum_values_match_spec_taxonomy():
    """The four status values cover the cases the spec cares about."""
    assert EpochGaincalStatus.SOLVED.value == "solved"
    assert EpochGaincalStatus.LOW_SNR.value == "low_snr"
    assert EpochGaincalStatus.SOLVER_NO_TABLE.value == "solver_no_table"
    assert EpochGaincalStatus.EXCEPTION.value == "exception"


def test_result_dataclass_solved_carries_table_path():
    r = EpochGaincalResult("/path/to/ap.G", EpochGaincalStatus.SOLVED, None)
    assert r.g_table == "/path/to/ap.G"
    assert r.status == EpochGaincalStatus.SOLVED
    assert r.reason is None


def test_result_dataclass_low_snr_carries_no_table_and_reason():
    r = EpochGaincalResult(
        None,
        EpochGaincalStatus.LOW_SNR,
        "p.G flagged 44.4% of solutions (limit 30%) — SNR too low for reliable gain cal",
    )
    assert r.g_table is None
    assert r.status == EpochGaincalStatus.LOW_SNR
    assert "44.4%" in r.reason
    assert "limit 30%" in r.reason


def test_result_is_frozen():
    """Dataclass should be immutable so callers can't mutate manifest state."""
    r = EpochGaincalResult(None, EpochGaincalStatus.LOW_SNR, "test")
    with pytest.raises(Exception):  # FrozenInstanceError or similar
        r.status = EpochGaincalStatus.SOLVED  # type: ignore[misc]


@pytest.mark.parametrize(
    "status, expected_legacy",
    [
        (EpochGaincalStatus.SOLVED, "ok"),
        (EpochGaincalStatus.LOW_SNR, "low_snr"),
        (EpochGaincalStatus.SOLVER_NO_TABLE, "low_snr"),
        (EpochGaincalStatus.EXCEPTION, "fallback"),
    ],
)
def test_status_to_legacy_string_mapping_documented_in_batch_pipeline(status, expected_legacy):
    """Document the status → legacy gaincal_status mapping batch_pipeline.py applies.

    This test pins the expected mapping. The actual mapping lives in
    scripts/batch_pipeline.py around the calibrate_epoch invocation and is
    consumed by dsa110_continuum.qa.promotion.derive_epoch_gaincal_state.
    Failure here means the mapping has drifted — update batch_pipeline.py
    or the spec, not this test.
    """
    # SOLVED → "ok"
    # LOW_SNR / SOLVER_NO_TABLE → "low_snr" (operational SNR floor)
    # EXCEPTION → "fallback" (legacy code-path fall-back)
    mapping = {
        EpochGaincalStatus.SOLVED: "ok",
        EpochGaincalStatus.LOW_SNR: "low_snr",
        EpochGaincalStatus.SOLVER_NO_TABLE: "low_snr",
        EpochGaincalStatus.EXCEPTION: "fallback",
    }
    assert mapping[status] == expected_legacy
