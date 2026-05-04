"""Smoke validation of the receptor-aware FLAG fraction fix on real DSA-110 bandpass tables."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/data/dsa110-continuum')

from dsa110_continuum.adapters.casa_tables import table
from dsa110_continuum.calibration.calibration import (
    _check_flag_fraction,
    _flag_fraction_excluding_dead_receptors,
)


def buggy_axis_zero_count(flags: np.ndarray, dead_threshold: float = 0.99) -> int:
    """Reproduce the OLD buggy logic: treat axis 0 as antennas."""
    nant, nchan, npol = flags.shape
    per_ant_flags = np.sum(flags, axis=(1, 2))
    max_flags_per_ant = nchan * npol
    return int(np.sum(per_ant_flags >= dead_threshold * max_flags_per_ant))


def validate(path: str) -> None:
    print(f"\n{'=' * 78}\nReal bandpass table: {path}\n{'=' * 78}")
    with table(path, readonly=True, ack=False) as tb:
        flags = tb.getcol("FLAG")
        antenna_ids = tb.getcol("ANTENNA1")

    print(f"FLAG shape           : {flags.shape}")
    print(f"ANTENNA1 row count   : {len(antenna_ids)}")
    print(f"Unique antennas      : {len(set(antenna_ids))}")

    raw_flagged = int(np.sum(flags))
    print(f"Raw flagged cells    : {raw_flagged:,} / {flags.size:,} = {raw_flagged / flags.size * 100:.4f}%")

    # OLD buggy interpretation
    buggy_dead = buggy_axis_zero_count(flags)
    print(f"\nOLD (buggy) result : 'dead antennas' = {buggy_dead}  (would falsely fail strict QA if >>117)")

    # NEW receptor-aware interpretation
    result = _flag_fraction_excluding_dead_receptors(flags, antenna_ids)
    print(f"\nNEW (receptor-aware) result:")
    print(f"  effective_flag_fraction : {result['effective_flag_fraction']:.10f} = {result['effective_flag_fraction'] * 100:.4f}%")
    print(f"  dead_receptor_count     : {result['dead_receptor_count']}")
    print(f"  dead_antenna_count      : {result['dead_antenna_count']}")
    print(f"  working_receptor_count  : {result['working_receptor_count']}")
    print(f"  working_flagged         : {result['working_flagged']:,}")
    print(f"  working_total           : {result['working_total']:,}")

    # Confirm the QA gate now passes (5% threshold)
    threshold = 0.05
    passes = result["effective_flag_fraction"] <= threshold
    print(f"\nQA gate (5% threshold): {'PASS' if passes else 'FAIL'}")

    # Confirm the gate would have FAILED falsely under the bug
    # The buggy code computed effective_flag_fraction by excluding "dead antennas"
    # at axis 0. Replicate that:
    nant_ax0, nchan_ax0, npol_ax0 = flags.shape
    per_ant_flags = np.sum(flags, axis=(1, 2))
    max_flags_per_ant = nchan_ax0 * npol_ax0
    dead_mask = per_ant_flags >= 0.99 * max_flags_per_ant
    n_working = nant_ax0 - int(np.sum(dead_mask))
    if n_working > 0:
        buggy_eff = float(np.sum(flags[~dead_mask, :, :]) / (n_working * nchan_ax0 * npol_ax0))
    else:
        buggy_eff = float(raw_flagged / flags.size)
    print(f"\nReplay of OLD buggy 'effective_flag_fraction': {buggy_eff:.6f} = {buggy_eff * 100:.4f}%")
    print(f"  Would have {'PASSED' if buggy_eff <= threshold else 'FAILED'} 5% gate")

    return result


if __name__ == "__main__":
    paths = [
        "/stage/dsa110-contimg/ms/2026-01-25T22:26:05_0~23.b",
        "/stage/dsa110-contimg/ms/2026-02-15T22:26:05_0~23.b",
    ]
    for p in paths:
        if Path(p).exists():
            validate(p)
        else:
            print(f"SKIP (missing): {p}")
