"""Per-antenna complex gain corruption for simulation.

Multiplies each baseline visibility V_ij by G_i * conj(G_j), where
G_i = (1 + eps_amp_i) * exp(i * eps_phase_i).

This simulates direction-independent gain errors: the dominant error
source for a transit array during a ~5-min integration window.

The amplitude errors are drawn from a zero-mean normal distribution so
that E[|G_i|] = 1.0 by design; the gains are further normalized to
unit mean amplitude to remove the small-sample bias that arises with
O(10) antennas and 10% scatter.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pyuvdata

logger = logging.getLogger(__name__)


def corrupt_uvh5(
    uvh5_path: Path | str,
    *,
    amp_scatter: float = 0.05,
    phase_scatter_deg: float = 5.0,
    seed: int = 0,
    output_path: Path | str | None = None,
) -> Path:
    """Corrupt visibilities with per-antenna gain errors.

    Parameters
    ----------
    uvh5_path:
        Input UVH5 file path.
    amp_scatter:
        Standard deviation of amplitude error (fractional, e.g. 0.05 = 5%).
    phase_scatter_deg:
        Standard deviation of phase error in degrees.
    seed:
        RNG seed for reproducibility.
    output_path:
        Where to write the corrupted file. Defaults to
        ``<stem>_corrupted.uvh5`` alongside the input.

    Returns
    -------
    Path
        Path to the written corrupted UVH5 file.
    """
    uvh5_path = Path(uvh5_path)
    if output_path is None:
        output_path = uvh5_path.with_name(uvh5_path.stem + "_corrupted.uvh5")
    output_path = Path(output_path)

    rng = np.random.default_rng(seed)

    uv = pyuvdata.UVData()
    uv.read(str(uvh5_path))

    ant_nums = np.unique(
        np.concatenate([uv.ant_1_array, uv.ant_2_array])
    )  # actual antenna numbers in data

    # Draw one gain per antenna: G_i = (1 + eps_amp) * exp(i * eps_phase)
    amp_errors = 1.0 + rng.normal(0.0, amp_scatter, size=len(ant_nums))
    phase_errors = rng.normal(0.0, np.radians(phase_scatter_deg), size=len(ant_nums))
    gains = amp_errors * np.exp(1j * phase_errors)  # shape (n_ant_unique,)

    # Normalize so the mean amplitude is exactly 1.0.
    # This removes the small-sample bias that would otherwise cause the mean
    # baseline amplitude ratio to deviate from 1.0 for small arrays (e.g.
    # 4 antennas × 10 % scatter). The scatter (phase + fractional amplitude
    # variation around unity) is preserved; only the overall multiplicative
    # offset is removed.  This is equivalent to calibrating out the array
    # mean gain before applying the per-antenna perturbation.
    mean_amp = np.abs(gains).mean()
    if mean_amp > 0:
        gains = gains / mean_amp

    # Map antenna number -> gain index
    ant_to_idx = {int(a): i for i, a in enumerate(ant_nums)}

    # Vectorised application of G_i * conj(G_j) to cross-correlations only.
    # Autocorrelations (ant_1 == ant_2) are left unchanged — they measure
    # total power and are not affected by direction-independent gain errors
    # in interferometric calibration.
    idx1 = np.array([ant_to_idx[int(a)] for a in uv.ant_1_array])
    idx2 = np.array([ant_to_idx[int(a)] for a in uv.ant_2_array])
    cross_mask = idx1 != idx2  # True for cross-correlations
    factors = gains[idx1] * np.conj(gains[idx2])  # shape (Nblts,)
    data = uv.data_array.copy()  # shape (Nblts, Nfreqs, Npols)
    # Apply factor only to cross-correlation rows; autos untouched
    data[cross_mask] *= factors[cross_mask, None, None]
    uv.data_array = data.astype(np.complex64)

    # Store gain truth in extra_keywords for validation
    uv.extra_keywords["GAIN_AMP_SCATTER"] = amp_scatter
    uv.extra_keywords["GAIN_PHASE_SCATTER_DEG"] = phase_scatter_deg
    uv.extra_keywords["GAIN_SEED"] = seed

    uv.write_uvh5(str(output_path), clobber=True)
    logger.info(
        "Wrote corrupted UVH5 (%d antennas, amp_scatter=%.3f, phase_scatter=%.1f deg) -> %s",
        len(ant_nums), amp_scatter, phase_scatter_deg, output_path,
    )
    return output_path
