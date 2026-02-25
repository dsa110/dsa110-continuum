"""
GPU-accelerated calibration operations using CuPy.

This module provides GPU-accelerated gain application and simple gain solving
for visibility calibration. It complements the CASA-based calibration pipeline
by offering fast GPU implementations for common operations.

Note: We use CuPy instead of Numba CUDA due to driver constraints
(Driver 455.23.05 limits PTX version, blocking Numba CUDA compilation).
CuPy's ElementwiseKernel provides similar performance with better compatibility.

Primary Operations:
    - apply_gains_gpu: Apply complex gains to visibilities
    - solve_per_antenna_gains: Simple per-antenna gain solving
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from dsa110_contimg.common.utils.gpu_safety import (
    gpu_safe,
    memory_safe,
    safe_gpu_context,
)

logger = logging.getLogger(__name__)

# Try to import CuPy - graceful fallback if unavailable
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False
    logger.warning("CuPy not available - GPU calibration disabled")


@dataclass
class CalibrationConfig:
    """Configuration for GPU calibration operations."""

    gpu_id: int = 0
    chunk_size: int = 5_000_000
    n_antennas: int = 110
    n_channels: int = 1024
    n_polarizations: int = 4
    interpolation: str = "nearest"


@dataclass
class GainSolutionResult:
    """Result of gain solving."""

    gains: np.ndarray | None = None
    weights: np.ndarray | None = None
    n_iterations: int = 0
    converged: bool = False
    residual_rms: float = 0.0
    processing_time_s: float = 0.0
    gpu_id: int = 0
    error: str | None = None

    @property
    def success(self) -> bool:
        """Check if solving completed successfully."""
        return self.error is None and self.gains is not None


@dataclass
class ApplyCalResult:
    """Result of applying calibration."""

    n_vis_processed: int = 0
    n_vis_calibrated: int = 0
    n_vis_flagged: int = 0
    processing_time_s: float = 0.0
    gpu_id: int = 0
    error: str | None = None

    @property
    def success(self) -> bool:
        """Check if application completed successfully."""
        return self.error is None


# CuPy kernel for applying complex gains to visibilities
_APPLY_GAINS_KERNEL_CODE = """
extern "C" __global__
void apply_gains(
    const float* vis_real,      // Input visibility real parts
    const float* vis_imag,      // Input visibility imaginary parts
    const float* gain_real,     // Gain real parts (n_ant, n_chan, n_pol)
    const float* gain_imag,     // Gain imaginary parts
    const int* ant1,            // Antenna 1 indices
    const int* ant2,            // Antenna 2 indices
    const int* chan_ids,        // Channel indices (or NULL for all)
    float* out_real,            // Output corrected vis real
    float* out_imag,            // Output corrected vis imaginary
    int n_vis,                  // Number of visibilities
    int n_chan,                 // Number of channels
    int n_pol                   // Number of polarizations
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vis) return;

    // Get antenna indices for this baseline
    int a1 = ant1[idx];
    int a2 = ant2[idx];

    // Get channel (assume sequential if chan_ids is NULL)
    int chan = (chan_ids != NULL) ? chan_ids[idx] : idx % n_chan;

    // For each polarization
    for (int pol = 0; pol < n_pol; pol++) {
        // Gain indices: (ant, chan, pol) -> ant * n_chan * n_pol + chan * n_pol + pol
        int g1_idx = a1 * n_chan * n_pol + chan * n_pol + pol;
        int g2_idx = a2 * n_chan * n_pol + chan * n_pol + pol;

        // Get gains for both antennas
        float g1_re = gain_real[g1_idx];
        float g1_im = gain_imag[g1_idx];
        float g2_re = gain_real[g2_idx];
        float g2_im = gain_imag[g2_idx];

        // Compute g1 * conj(g2) = (g1_re + i*g1_im) * (g2_re - i*g2_im)
        // = g1_re*g2_re + g1_im*g2_im + i*(g1_im*g2_re - g1_re*g2_im)
        float gg_re = g1_re * g2_re + g1_im * g2_im;
        float gg_im = g1_im * g2_re - g1_re * g2_im;

        // Normalize: divide by |g1*conj(g2)|^2
        float gg_norm2 = gg_re * gg_re + gg_im * gg_im;
        if (gg_norm2 < 1e-10f) {
            // Flag this visibility (set to NaN)
            out_real[idx * n_pol + pol] = nanf("");
            out_imag[idx * n_pol + pol] = nanf("");
            continue;
        }

        // Get input visibility
        int vis_idx = idx * n_pol + pol;
        float v_re = vis_real[vis_idx];
        float v_im = vis_imag[vis_idx];

        // Corrected = vis / (g1 * conj(g2)) = vis * conj(g1*conj(g2)) / |g1*conj(g2)|^2
        // = (v_re + i*v_im) * (gg_re - i*gg_im) / gg_norm2
        float corr_re = (v_re * gg_re + v_im * gg_im) / gg_norm2;
        float corr_im = (v_im * gg_re - v_re * gg_im) / gg_norm2;

        out_real[vis_idx] = corr_re;
        out_imag[vis_idx] = corr_im;
    }
}
"""

# CuPy kernel for simple per-antenna gain solving using Stefcal iteration
_SOLVE_GAINS_KERNEL_CODE = """
extern "C" __global__
void accumulate_gain_numerator(
    const float* vis_real,      // Observed visibility real parts
    const float* vis_imag,      // Observed visibility imaginary parts
    const float* model_real,    // Model visibility real parts
    const float* model_imag,    // Model visibility imaginary parts
    const float* g_real,        // Current gain estimates real
    const float* g_imag,        // Current gain estimates imag
    const int* ant1,            // Antenna 1 indices
    const int* ant2,            // Antenna 2 indices
    const float* weights,       // Visibility weights
    float* num_real,            // Output numerator real (n_ant,)
    float* num_imag,            // Output numerator imag
    float* denom,               // Output denominator (n_ant,)
    int n_vis,
    int n_ant,
    int n_chan,
    int n_pol
) {
    // Each thread handles one antenna's accumulation across a subset of baselines
    int ant_idx = blockIdx.x;
    int tid = threadIdx.x;
    int n_threads = blockDim.x;

    if (ant_idx >= n_ant) return;

    // Shared memory for partial sums
    extern __shared__ float shared[];
    float* s_num_re = shared;
    float* s_num_im = shared + n_threads;
    float* s_denom = shared + 2 * n_threads;

    s_num_re[tid] = 0.0f;
    s_num_im[tid] = 0.0f;
    s_denom[tid] = 0.0f;

    // Loop over visibilities in strided fashion
    for (int idx = tid; idx < n_vis; idx += n_threads) {
        int a1 = ant1[idx];
        int a2 = ant2[idx];

        // Only process baselines involving this antenna
        if (a1 != ant_idx && a2 != ant_idx) continue;

        float w = weights[idx];
        if (w <= 0.0f) continue;

        // Determine partner antenna
        int partner = (a1 == ant_idx) ? a2 : a1;
        bool conjugate = (a2 == ant_idx);  // If we're antenna 2, need to conjugate

        // For simplicity, use first polarization and average over channels
        float v_re = vis_real[idx * n_pol];
        float v_im = vis_imag[idx * n_pol];
        float m_re = model_real[idx * n_pol];
        float m_im = model_imag[idx * n_pol];

        // Get partner gain (current estimate)
        float gp_re = g_real[partner];
        float gp_im = g_imag[partner];

        // If we're antenna 2, compute: g_i* = sum(V_ij * conj(M_ij) * conj(g_j)) / sum(|M_ij * g_j|^2)
        // If we're antenna 1, compute: g_i = sum(V_ij * M_ij * g_j) / sum(|M_ij * g_j|^2)
        // Actually for standard calibration: V_ij = g_i * M_ij * conj(g_j)
        // So: g_i = sum(V_ij * conj(g_j) * conj(M_ij)) / sum(|g_j|^2 * |M_ij|^2)

        // Model * conj(g_partner)
        float mg_re, mg_im;
        if (conjugate) {
            // M * g_partner
            mg_re = m_re * gp_re - m_im * gp_im;
            mg_im = m_re * gp_im + m_im * gp_re;
        } else {
            // M * conj(g_partner)
            mg_re = m_re * gp_re + m_im * gp_im;
            mg_im = -m_re * gp_im + m_im * gp_re;
        }

        // Numerator: V * conj(M * g_partner)
        float num_re = v_re * mg_re + v_im * mg_im;
        float num_im = v_im * mg_re - v_re * mg_im;

        // Denominator: |M * g_partner|^2
        float den = (mg_re * mg_re + mg_im * mg_im);

        s_num_re[tid] += w * num_re;
        s_num_im[tid] += w * num_im;
        s_denom[tid] += w * den;
    }

    __syncthreads();

    // Reduction
    for (int s = n_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_num_re[tid] += s_num_re[tid + s];
            s_num_im[tid] += s_num_im[tid + s];
            s_denom[tid] += s_denom[tid + s];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        atomicAdd(&num_real[ant_idx], s_num_re[0]);
        atomicAdd(&num_imag[ant_idx], s_num_im[0]);
        atomicAdd(&denom[ant_idx], s_denom[0]);
    }
}
"""

_COMPILED_KERNELS: dict[str, Any] = {}


def _get_kernel(name: str) -> Any:
    """Get or compile a CUDA kernel by name.

    Parameters
    ----------
    name :
        Kernel name ('apply_gains' or 'solve_gains')

    Returns
    -------
        Compiled CuPy RawKernel

    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available for GPU calibration")

    if name not in _COMPILED_KERNELS:
        if name == "apply_gains":
            kernel = cp.RawKernel(_APPLY_GAINS_KERNEL_CODE, "apply_gains")
        elif name == "accumulate":
            kernel = cp.RawKernel(_SOLVE_GAINS_KERNEL_CODE, "accumulate_gain_numerator")
        else:
            raise ValueError(f"Unknown kernel: {name}")
        _COMPILED_KERNELS[name] = kernel

    return _COMPILED_KERNELS[name]


def estimate_applycal_memory_gb(
    n_vis: int,
    n_channels: int,
    n_pols: int,
    n_antennas: int,
) -> tuple[float, float]:
    """Estimate memory requirements for applying calibration.

    Parameters
    ----------
    n_vis :
        Number of visibilities
    n_channels :
        Number of channels
    n_pols :
        Number of polarizations
    n_antennas :
        Number of antennas

    Returns
    -------
        Tuple of (gpu_memory_gb, system_memory_gb)

    """
    # Per-visibility data:
    # - Input vis: 2 * n_vis * n_pols * 4 bytes (float32 real/imag)
    # - Output vis: 2 * n_vis * n_pols * 4 bytes
    # - Antenna indices: 2 * n_vis * 4 bytes (int32)
    vis_bytes = 2 * n_vis * n_pols * 4 * 2 + 2 * n_vis * 4

    # Gains:
    # - (n_antennas, n_channels, n_pols) complex = 2 * n_ant * n_chan * n_pol * 4
    gain_bytes = 2 * n_antennas * n_channels * n_pols * 4

    gpu_gb = (vis_bytes + gain_bytes) / (1024**3)
    system_gb = gpu_gb * 2.5  # Account for CPU buffers and overhead

    return (gpu_gb, system_gb)


def estimate_solve_memory_gb(
    n_vis: int,
    n_antennas: int,
) -> tuple[float, float]:
    """Estimate memory requirements for gain solving.

    Parameters
    ----------
    n_vis :
        Number of visibilities
    n_antennas :
        Number of antennas

    Returns
    -------
        Tuple of (gpu_memory_gb, system_memory_gb)

    """
    # Observed and model vis (4 bytes each, float32)
    vis_bytes = 4 * n_vis * 4  # real, imag for obs and model

    # Gains and accumulators
    gain_bytes = n_antennas * 4 * 6  # real, imag, numerator_re, num_im, denom

    # Antenna indices and weights
    index_bytes = n_vis * 4 * 3  # ant1, ant2, weights

    gpu_gb = (vis_bytes + gain_bytes + index_bytes) / (1024**3)
    system_gb = gpu_gb * 2.5

    return (gpu_gb, system_gb)


@memory_safe()
@gpu_safe()
def apply_gains_gpu(
    vis: np.ndarray,
    gains: np.ndarray,
    ant1: np.ndarray,
    ant2: np.ndarray,
    *,
    config: CalibrationConfig | None = None,
    gpu_id: int = 0,
) -> ApplyCalResult:
    """Apply complex gains to visibilities on GPU.

    Corrects visibilities by dividing by g_i * conj(g_j) for baseline (i, j).

    Parameters
    ----------
    vis :
        Complex visibilities (n_vis, n_pol) or (n_vis,)
    gains :
        Complex gains (n_ant, n_chan, n_pol) or (n_ant,)
    ant1 :
        Antenna 1 indices (n_vis,)
    ant2 :
        Antenna 2 indices (n_vis,)
    config :
        Configuration (optional)
    gpu_id :
        GPU device ID
    vis: np.ndarray :

    gains: np.ndarray :

    ant1: np.ndarray :

    ant2: np.ndarray :

    * :

    config: Optional[CalibrationConfig] :
         (Default value = None)

    Returns
    -------
        ApplyCalResult with corrected visibilities in vis array (modified in place)

    """
    start_time = time.time()

    if not CUPY_AVAILABLE:
        return ApplyCalResult(error="CuPy not available for GPU calibration")

    if config is None:
        config = CalibrationConfig(gpu_id=gpu_id)

    # Validate inputs
    n_vis = len(vis)
    if len(ant1) != n_vis or len(ant2) != n_vis:
        return ApplyCalResult(
            error=f"Antenna arrays must match vis length: got {len(ant1)}, {len(ant2)}, {n_vis}"
        )

    # Reshape vis if 1D
    if vis.ndim == 1:
        vis = vis.reshape(-1, 1)
    n_pol = vis.shape[1] if vis.ndim > 1 else 1

    # Reshape gains if 1D (broadcast to all channels)
    if gains.ndim == 1:
        gains = gains.reshape(-1, 1, 1)
    n_ant = gains.shape[0]
    n_chan = gains.shape[1] if gains.ndim > 1 else 1

    logger.info(
        "Applying gains: %d vis, %d antennas, %d channels, %d pols", n_vis, n_ant, n_chan, n_pol
    )

    try:
        with safe_gpu_context(config.gpu_id):
            # Transfer to GPU
            vis_gpu = cp.asarray(vis.astype(np.complex64))
            vis_real = vis_gpu.real.astype(cp.float32).copy()
            vis_imag = vis_gpu.imag.astype(cp.float32).copy()

            gains_gpu = cp.asarray(gains.astype(np.complex64))
            gain_real = gains_gpu.real.astype(cp.float32).copy()
            gain_imag = gains_gpu.imag.astype(cp.float32).copy()

            ant1_gpu = cp.asarray(ant1.astype(np.int32))
            ant2_gpu = cp.asarray(ant2.astype(np.int32))

            # Output arrays
            out_real = cp.zeros_like(vis_real)
            out_imag = cp.zeros_like(vis_imag)

            # Launch kernel
            kernel = _get_kernel("apply_gains")
            threads = 256
            blocks = (n_vis + threads - 1) // threads

            kernel(
                (blocks,),
                (threads,),
                (
                    vis_real,
                    vis_imag,
                    gain_real.ravel(),
                    gain_imag.ravel(),
                    ant1_gpu,
                    ant2_gpu,
                    cp.int32(0),  # NULL for chan_ids
                    out_real,
                    out_imag,
                    np.int32(n_vis),
                    np.int32(n_chan),
                    np.int32(n_pol),
                ),
            )

            cp.cuda.stream.get_current_stream().synchronize()

            # Get result
            result_vis = out_real.get() + 1j * out_imag.get()

            # Count flagged (NaN)
            n_flagged = np.sum(np.isnan(result_vis))

            # Update vis array in place (excluding NaNs)
            valid_mask = ~np.isnan(result_vis)
            vis[valid_mask] = result_vis[valid_mask]

            processing_time = time.time() - start_time

            return ApplyCalResult(
                n_vis_processed=n_vis,
                n_vis_calibrated=int(np.sum(valid_mask)),
                n_vis_flagged=int(n_flagged),
                processing_time_s=processing_time,
                gpu_id=config.gpu_id,
            )

    except (RuntimeError, MemoryError, ValueError) as exc:
        logger.error("GPU gain application failed: %s", str(exc))
        return ApplyCalResult(
            error=str(exc),
            gpu_id=config.gpu_id,
        )


def apply_gains_cpu(
    vis: np.ndarray,
    gains: np.ndarray,
    ant1: np.ndarray,
    ant2: np.ndarray,
) -> tuple[np.ndarray, int]:
    """CPU fallback for gain application.

    Parameters
    ----------
    vis :
        Complex visibilities (n_vis, n_pol) or (n_vis,)
    gains :
        Complex gains (n_ant,) - per-antenna gains
    ant1 :
        Antenna 1 indices
    ant2 :
        Antenna 2 indices
    vis: np.ndarray :

    gains: np.ndarray :

    ant1: np.ndarray :

    ant2: np.ndarray :


    Returns
    -------
        Tuple of (corrected_vis, n_flagged)

    """
    # Reshape if needed
    if vis.ndim == 1:
        vis = vis.reshape(-1, 1)

    # Get gains for each baseline
    gain_ant1 = gains[ant1]  # (n_vis,)
    gain_ant2 = gains[ant2]  # (n_vis,)

    # gain_ant1 * conj(gain_ant2)
    gain_product = gain_ant1 * np.conj(gain_ant2)

    # Avoid division by zero
    gain_norm2 = np.abs(gain_product) ** 2
    small_gain = gain_norm2 < 1e-10

    # Corrected = vis / (g1 * conj(g2))
    with np.errstate(divide="ignore", invalid="ignore"):
        corrected = vis / gain_product.reshape(-1, 1)

    # Flag bad gains
    corrected[small_gain] = np.nan
    n_flagged = int(np.sum(small_gain))

    return corrected.squeeze(), n_flagged


def solve_per_antenna_gains_cpu(
    vis: np.ndarray,
    model: np.ndarray,
    ant1: np.ndarray,
    ant2: np.ndarray,
    weights: np.ndarray,
    n_antennas: int,
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
    refant: int = 0,
) -> GainSolutionResult:
    """Solve for per-antenna gains using iterative Stefcal-like algorithm (CPU).

    Uses the equation: V_ij = g_i * M_ij * conj(g_j)

    Parameters
    ----------
    vis :
        Observed visibilities (n_vis,) complex
    model :
        Model visibilities (n_vis,) complex
    ant1 :
        Antenna 1 indices
    ant2 :
        Antenna 2 indices
    weights :
        Visibility weights
    n_antennas :
        Number of antennas
    max_iter :
        Maximum iterations
    tol :
        Convergence tolerance
    refant :
        Reference antenna index (gain fixed to 1+0j)
    vis: np.ndarray :

    model: np.ndarray :

    ant1: np.ndarray :

    ant2: np.ndarray :

    weights: np.ndarray :

    Returns
    -------
        GainSolutionResult with solved gains

    """
    start_time = time.time()

    # Initialize gains to 1
    gains = np.ones(n_antennas, dtype=np.complex128)

    # Iterative solver
    for iteration in range(max_iter):
        gains_old = gains.copy()

        # Update each antenna
        for ant in range(n_antennas):
            if ant == refant:
                continue  # Reference antenna fixed

            # Find baselines involving this antenna
            mask1 = ant1 == ant
            mask2 = ant2 == ant

            # Accumulate numerator and denominator
            num = 0.0 + 0.0j
            denom = 0.0

            # Baselines where this is antenna 1
            for idx in np.where(mask1)[0]:
                if weights[idx] <= 0:
                    continue
                partner = ant2[idx]
                gp = gains[partner]
                mg = model[idx] * np.conj(gp)
                num += weights[idx] * vis[idx] * np.conj(mg)
                denom += weights[idx] * np.abs(mg) ** 2

            # Baselines where this is antenna 2
            for idx in np.where(mask2)[0]:
                if weights[idx] <= 0:
                    continue
                partner = ant1[idx]
                gp = gains[partner]
                mg = np.conj(model[idx]) * gp
                num += weights[idx] * np.conj(vis[idx]) * np.conj(mg)
                denom += weights[idx] * np.abs(mg) ** 2

            if denom > 1e-10:
                gains[ant] = num / denom

        # Check convergence
        diff = np.max(np.abs(gains - gains_old))
        if diff < tol:
            processing_time = time.time() - start_time
            return GainSolutionResult(
                gains=gains.reshape(n_antennas, 1, 1),
                weights=np.ones((n_antennas, 1, 1)),
                n_iterations=iteration + 1,
                converged=True,
                residual_rms=float(diff),
                processing_time_s=processing_time,
            )

    processing_time = time.time() - start_time
    return GainSolutionResult(
        gains=gains.reshape(n_antennas, 1, 1),
        weights=np.ones((n_antennas, 1, 1)),
        n_iterations=max_iter,
        converged=False,
        residual_rms=float(diff),
        processing_time_s=processing_time,
    )


@memory_safe()
@gpu_safe()
def solve_per_antenna_gains_gpu(
    vis: np.ndarray,
    model: np.ndarray,
    ant1: np.ndarray,
    ant2: np.ndarray,
    weights: np.ndarray,
    n_antennas: int,
    *,
    config: CalibrationConfig | None = None,
    max_iter: int = 50,
    tol: float = 1e-6,
    refant: int = 0,
) -> GainSolutionResult:
    """Solve for per-antenna gains on GPU using iterative algorithm.

    Parameters
    ----------
    vis :
        Observed visibilities (n_vis,) complex
    model :
        Model visibilities (n_vis,) complex
    ant1 :
        Antenna 1 indices
    ant2 :
        Antenna 2 indices
    weights :
        Visibility weights
    n_antennas :
        Number of antennas
    config :
        Configuration (optional)
    max_iter :
        Maximum iterations
    tol :
        Convergence tolerance
    refant :
        Reference antenna index
    vis: np.ndarray :

    model: np.ndarray :

    ant1: np.ndarray :

    ant2: np.ndarray :

    weights: np.ndarray :

    Returns
    -------
        GainSolutionResult with solved gains

    """
    start_time = time.time()

    # For now, fall back to CPU implementation
    # GPU kernel is more complex and needs careful testing
    logger.info("GPU gain solving: %d vis, %d antennas (using CPU fallback)", len(vis), n_antennas)

    result = solve_per_antenna_gains_cpu(
        vis, model, ant1, ant2, weights, n_antennas, max_iter=max_iter, tol=tol, refant=refant
    )

    # Update timing to include overhead
    result.processing_time_s = time.time() - start_time
    if config:
        result.gpu_id = config.gpu_id

    return result


def apply_gains(
    vis: np.ndarray,
    gains: np.ndarray,
    ant1: np.ndarray,
    ant2: np.ndarray,
    *,
    config: CalibrationConfig | None = None,
    use_gpu: bool = True,
) -> ApplyCalResult:
    """Apply gains with automatic GPU/CPU selection.

    Parameters
    ----------
    vis :
        Complex visibilities
    gains :
        Complex gains
    ant1 :
        Antenna 1 indices
    ant2 :
        Antenna 2 indices
    config :
        Configuration
    use_gpu :
        Whether to attempt GPU acceleration
    vis: np.ndarray :

    gains: np.ndarray :

    ant1: np.ndarray :

    ant2: np.ndarray :

    * :

    config: Optional[CalibrationConfig] :
         (Default value = None)

    Returns
    -------
        ApplyCalResult

    """
    if use_gpu and CUPY_AVAILABLE:
        return apply_gains_gpu(vis, gains, ant1, ant2, config=config)

    start_time = time.time()
    corrected, n_flagged = apply_gains_cpu(vis, gains, ant1, ant2)
    vis[:] = corrected
    return ApplyCalResult(
        n_vis_processed=len(vis),
        n_vis_calibrated=len(vis) - n_flagged,
        n_vis_flagged=n_flagged,
        processing_time_s=time.time() - start_time,
    )


def solve_per_antenna_gains(
    vis: np.ndarray,
    model: np.ndarray,
    ant1: np.ndarray,
    ant2: np.ndarray,
    weights: np.ndarray,
    n_antennas: int,
    *,
    config: CalibrationConfig | None = None,
    use_gpu: bool = True,
    max_iter: int = 50,
    tol: float = 1e-6,
    refant: int = 0,
) -> GainSolutionResult:
    """Solve for per-antenna gains with automatic GPU/CPU selection.

    Parameters
    ----------
    vis :
        Observed visibilities
    model :
        Model visibilities
    ant1 :
        Antenna 1 indices
    ant2 :
        Antenna 2 indices
    weights :
        Visibility weights
    n_antennas :
        Number of antennas
    config :
        Configuration
    use_gpu :
        Whether to attempt GPU acceleration
    max_iter :
        Maximum iterations
    tol :
        Convergence tolerance
    refant :
        Reference antenna
    vis: np.ndarray :

    model: np.ndarray :

    ant1: np.ndarray :

    ant2: np.ndarray :

    weights: np.ndarray :

    Returns
    -------
        GainSolutionResult

    """
    if use_gpu and CUPY_AVAILABLE:
        return solve_per_antenna_gains_gpu(
            vis,
            model,
            ant1,
            ant2,
            weights,
            n_antennas,
            config=config,
            max_iter=max_iter,
            tol=tol,
            refant=refant,
        )

    return solve_per_antenna_gains_cpu(
        vis, model, ant1, ant2, weights, n_antennas, max_iter=max_iter, tol=tol, refant=refant
    )
