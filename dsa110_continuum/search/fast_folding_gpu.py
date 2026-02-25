"""GPU-accelerated Fast Folding Algorithm (FFA) for pulsar period searching.

Primary GPU backend: PyTorch CUDA (robust on H17).
CPU reference backend: NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np

try:
    import torch

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    _TORCH_AVAILABLE = False

Backend = Literal["cpu", "gpu"]


@dataclass(slots=True)
class FFAResult:
    trial_periods_s: np.ndarray
    score: np.ndarray
    best_period_s: float
    best_score: float
    backend_used: Backend
    runtime_s: float


def _fold_score_cpu(
    ts: np.ndarray,
    sample_dt_s: float,
    trial_periods_s: np.ndarray,
    n_phase_bins: int,
) -> np.ndarray:
    n = ts.size
    idx = np.arange(n, dtype=np.float64)
    out = np.empty(trial_periods_s.size, dtype=np.float64)

    for i, period_s in enumerate(trial_periods_s):
        phase = np.mod(idx * sample_dt_s, period_s) / period_s
        bins = np.minimum((phase * n_phase_bins).astype(np.int64), n_phase_bins - 1)
        prof = np.bincount(bins, weights=ts, minlength=n_phase_bins).astype(np.float64)
        mu = prof.mean()
        sigma = prof.std() + 1e-12
        out[i] = (prof.max() - mu) / sigma

    return out


def _fold_score_gpu_torch(
    ts: np.ndarray,
    sample_dt_s: float,
    trial_periods_s: np.ndarray,
    n_phase_bins: int,
    batch_size: int = 128,
    device: str = "cuda",
) -> np.ndarray:
    if not (_TORCH_AVAILABLE and torch.cuda.is_available()):
        raise RuntimeError("Torch CUDA backend unavailable")

    dtype = torch.float32
    ts_gpu = torch.as_tensor(ts, dtype=dtype, device=device)
    idx_gpu = torch.arange(ts.size, dtype=dtype, device=device)
    out = np.empty(trial_periods_s.size, dtype=np.float64)

    for start in range(0, trial_periods_s.size, batch_size):
        stop = min(start + batch_size, trial_periods_s.size)
        p = torch.as_tensor(trial_periods_s[start:stop], dtype=dtype, device=device)

        phase = torch.remainder(idx_gpu[None, :] * sample_dt_s, p[:, None]) / p[:, None]
        bins = torch.clamp((phase * n_phase_bins).to(torch.int64), 0, n_phase_bins - 1)

        prof = torch.zeros((stop - start, n_phase_bins), dtype=dtype, device=device)
        prof.scatter_add_(1, bins, ts_gpu.expand(stop - start, -1))

        mu = prof.mean(dim=1)
        sigma = prof.std(dim=1) + 1e-12
        score = (prof.max(dim=1).values - mu) / sigma
        out[start:stop] = score.detach().cpu().numpy().astype(np.float64)

    return out


def fast_fold_search(
    ts: np.ndarray,
    sample_dt_s: float,
    period_min_s: float,
    period_max_s: float,
    n_trials: int,
    n_phase_bins: int = 128,
    backend: Backend = "gpu",
    gpu_batch_size: int = 128,
) -> FFAResult:
    if ts.ndim != 1:
        raise ValueError("ts must be 1D")
    if n_trials < 2:
        raise ValueError("n_trials must be >= 2")
    if period_max_s <= period_min_s:
        raise ValueError("period_max_s must be > period_min_s")

    ts = np.asarray(ts, dtype=np.float32)
    trial_periods_s = np.linspace(period_min_s, period_max_s, n_trials, dtype=np.float64)

    t0 = perf_counter()
    backend_used: Backend

    if backend == "gpu" and _TORCH_AVAILABLE and torch.cuda.is_available():
        score = _fold_score_gpu_torch(
            ts=ts,
            sample_dt_s=sample_dt_s,
            trial_periods_s=trial_periods_s,
            n_phase_bins=n_phase_bins,
            batch_size=gpu_batch_size,
        )
        backend_used = "gpu"
    else:
        score = _fold_score_cpu(
            ts=ts,
            sample_dt_s=sample_dt_s,
            trial_periods_s=trial_periods_s,
            n_phase_bins=n_phase_bins,
        )
        backend_used = "cpu"

    runtime_s = perf_counter() - t0
    best_idx = int(np.argmax(score))

    return FFAResult(
        trial_periods_s=trial_periods_s,
        score=score,
        best_period_s=float(trial_periods_s[best_idx]),
        best_score=float(score[best_idx]),
        backend_used=backend_used,
        runtime_s=runtime_s,
    )


def make_synthetic_pulsar(
    n_samples: int,
    sample_dt_s: float,
    period_s: float,
    duty_cycle: float = 0.04,
    pulse_snr: float = 5.0,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=np.float64) * sample_dt_s
    phase = np.mod(t, period_s) / period_s

    width = max(duty_cycle, 1e-3)
    pulse = np.exp(-0.5 * ((phase - 0.2) / width) ** 2)
    pulse = pulse / (pulse.std() + 1e-12)

    noise = rng.normal(0.0, 1.0, n_samples)
    return (noise + pulse_snr * 0.1 * pulse).astype(np.float32)
