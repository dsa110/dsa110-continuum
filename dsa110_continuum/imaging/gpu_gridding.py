"""
GPU-accelerated visibility gridding using CuPy.

This module implements UV-plane gridding on GPU for radio interferometry imaging.
It processes visibility data and grids it onto a 2D UV plane, followed by FFT
to produce dirty images.

Gridding Methods:
    - Nearest-neighbor: Simple, fast, lower quality
    - Convolutional: Uses gridding convolution function (GCF) for better quality
    - W-projection: Handles non-coplanar baselines (wide-field imaging)

Note: We use CuPy instead of Numba CUDA due to driver constraints
(Driver 455.23.05 limits PTX version, blocking Numba CUDA compilation).
CuPy's RawKernel provides similar performance with better compatibility.

Memory Safety:
    All GPU operations use safe_gpu_context from gpu_safety module to prevent
    OOM conditions that could crash the system.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from dsa110_contimg.common.utils.gpu_safety import (
    check_system_memory_available,
    gpu_safe,
    safe_gpu_context,
)

logger = logging.getLogger(__name__)

# Try to import CuPy - graceful fallback if unavailable
try:
    import cupy as cp
    from cupy import RawKernel

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    RawKernel = None
    CUPY_AVAILABLE = False
    logger.warning("CuPy not available - GPU gridding disabled")


# Try to import Numba
try:
    import numba

    HAVE_NUMBA = True
except ImportError:
    numba = None
    HAVE_NUMBA = False


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class GriddingConfig:
    """Configuration for GPU gridding."""

    image_size: int = 512
    cell_size_arcsec: float = 12.0
    gpu_id: int = 0
    support: int = 3
    oversampling: int = 128
    w_planes: int = 1
    use_w_projection: bool = False
    normalize: bool = True

    @property
    def cell_size_rad(self) -> float:
        """Cell size in radians."""
        return self.cell_size_arcsec * np.pi / (180.0 * 3600.0)


@dataclass
class GriddingResult:
    """Result of GPU gridding operation."""

    image: np.ndarray | None = None
    grid: np.ndarray | None = None
    weight_sum: float = 0.0
    n_vis: int = 0
    n_flagged: int = 0
    processing_time_s: float = 0.0
    gpu_id: int = 0
    error: str | None = None

    @property
    def success(self) -> bool:
        """Check if gridding completed successfully."""
        return self.error is None and self.image is not None


# =============================================================================
# Degridding Configuration and Results
# =============================================================================


@dataclass
class DegridConfig:
    """Configuration for GPU degridding (visibility prediction).

    Degridding is the inverse of gridding: given a sky model image,
    predict visibilities at specified UVW coordinates.

    For DSA-110 with ~200λ max baseline w-component:
    - w_planes=32 is sufficient
    - w_max=200.0 covers all baselines
    """

    image_size: int = 512
    cell_size_arcsec: float = 12.0
    gpu_id: int = 0
    support: int = 3
    oversampling: int = 128

    # W-projection parameters
    use_w_projection: bool = True
    w_planes: int = 32
    w_max: float = 200.0  # Maximum |w| in wavelengths

    @property
    def cell_size_rad(self) -> float:
        """Cell size in radians."""
        return self.cell_size_arcsec * np.pi / (180.0 * 3600.0)

    @property
    def cache_key(self) -> tuple:
        """Key for W-kernel cache lookup."""
        return (self.w_planes, self.w_max, self.support, self.image_size)


@dataclass
class DegridResult:
    """Result of GPU degridding (visibility prediction) operation."""

    vis_predicted: np.ndarray | None = None  # Complex predicted visibilities
    n_vis: int = 0
    n_skipped: int = 0  # Visibilities outside UV bounds
    processing_time_s: float = 0.0
    gpu_id: int = 0
    error: str | None = None

    @property
    def success(self) -> bool:
        """Check if degridding completed successfully."""
        return self.error is None and self.vis_predicted is not None


# =============================================================================
# CUDA Kernels (CuPy RawKernel)
# =============================================================================

# Nearest-neighbor gridding kernel
_GRID_NN_KERNEL = """
extern "C" __global__
void grid_nearest_neighbor(
    const float* uvw,           // (N, 3) UVW coordinates in wavelengths
    const float* vis_real,      // (N,) real part of visibilities
    const float* vis_imag,      // (N,) imaginary part of visibilities
    const float* weights,       // (N,) weights
    const int* flags,           // (N,) flags (1 = flagged, skip)
    float* grid_real,           // (size, size) real part of grid
    float* grid_imag,           // (size, size) imaginary part of grid
    float* weight_grid,         // (size, size) weight accumulator
    const int n_vis,            // Number of visibilities
    const int grid_size,        // Grid size in pixels
    const float cell_size       // Cell size in radians
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_vis) return;

    // Skip flagged data
    if (flags[idx] != 0) return;

    // Get UV coordinates (ignore W for now)
    float u = uvw[idx * 3 + 0];
    float v = uvw[idx * 3 + 1];

    // Convert to pixel coordinates
    // UV are in wavelengths, need to convert to pixel indices
    // u_pix = u / cell_size_rad + grid_size/2
    float u_pix_f = u * cell_size + grid_size / 2.0f;
    float v_pix_f = v * cell_size + grid_size / 2.0f;

    int u_pix = (int)(u_pix_f + 0.5f);  // Round to nearest
    int v_pix = (int)(v_pix_f + 0.5f);

    // Bounds check
    if (u_pix < 0 || u_pix >= grid_size || v_pix < 0 || v_pix >= grid_size) return;

    // Grid index
    int grid_idx = v_pix * grid_size + u_pix;

    // Weighted visibility
    float w = weights[idx];
    float vr = vis_real[idx] * w;
    float vi = vis_imag[idx] * w;

    // Atomic add to grid (handles race conditions)
    atomicAdd(&grid_real[grid_idx], vr);
    atomicAdd(&grid_imag[grid_idx], vi);
    atomicAdd(&weight_grid[grid_idx], w);
}
"""

# Convolutional gridding kernel with spheroidal function
_GRID_CONV_KERNEL = """
extern "C" __global__
void grid_convolutional(
    const float* uvw,           // (N, 3) UVW coordinates
    const float* vis_real,      // (N,) real part
    const float* vis_imag,      // (N,) imaginary part
    const float* weights,       // (N,) weights
    const int* flags,           // (N,) flags
    const float* gcf,           // (oversampling, support*2+1) gridding conv function
    float* grid_real,           // (size, size) real grid
    float* grid_imag,           // (size, size) imaginary grid
    float* weight_grid,         // (size, size) weight grid
    const int n_vis,
    const int grid_size,
    const float cell_size,
    const int support,
    const int oversampling
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_vis) return;
    if (flags[idx] != 0) return;

    float u = uvw[idx * 3 + 0];
    float v = uvw[idx * 3 + 1];

    // Convert to pixel coordinates
    float u_pix_f = u * cell_size + grid_size / 2.0f;
    float v_pix_f = v * cell_size + grid_size / 2.0f;

    int u_pix = (int)floorf(u_pix_f);
    int v_pix = (int)floorf(v_pix_f);

    // Fractional part for oversampling
    float u_frac = u_pix_f - u_pix;
    float v_frac = v_pix_f - v_pix;

    int u_off = (int)(u_frac * oversampling);
    int v_off = (int)(v_frac * oversampling);

    float w = weights[idx];
    float vr = vis_real[idx];
    float vi = vis_imag[idx];

    int gcf_width = 2 * support + 1;

    // Convolve onto grid
    for (int dv = -support; dv <= support; dv++) {
        int vv = v_pix + dv;
        if (vv < 0 || vv >= grid_size) continue;

        int gcf_v_idx = (dv + support) + v_off * gcf_width;
        float gcf_v = gcf[gcf_v_idx];

        for (int du = -support; du <= support; du++) {
            int uu = u_pix + du;
            if (uu < 0 || uu >= grid_size) continue;

            int gcf_u_idx = (du + support) + u_off * gcf_width;
            float gcf_u = gcf[gcf_u_idx];

            float conv_weight = gcf_u * gcf_v * w;

            int grid_idx = vv * grid_size + uu;

            atomicAdd(&grid_real[grid_idx], vr * conv_weight);
            atomicAdd(&grid_imag[grid_idx], vi * conv_weight);
            atomicAdd(&weight_grid[grid_idx], conv_weight);
        }
    }
}
"""

# Weight normalization kernel
_NORMALIZE_KERNEL = """
extern "C" __global__
void normalize_grid(
    float* grid_real,
    float* grid_imag,
    const float* weight_grid,
    const int size,
    const float epsilon
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size * size) return;

    float w = weight_grid[idx];
    if (w > epsilon) {
        grid_real[idx] /= w;
        grid_imag[idx] /= w;
    }
}
"""


# =============================================================================
# Degridding CUDA Kernels
# =============================================================================

# Degridding kernel with W-projection
# This is the inverse of gridding: sample the UV grid at specific (u,v,w) points
_DEGRID_W_KERNEL = """
extern "C" __global__
void degrid_w_projection(
    const float* uvw,               // (N, 3) UVW coordinates in wavelengths
    const float* grid_real,         // (size, size) UV grid real part
    const float* grid_imag,         // (size, size) UV grid imag part
    const float* gcf,               // (oversampling, 2*support+1) gridding conv function
    const float* w_kernels_real,    // (n_w_planes, kernel_size, kernel_size) W-kernels real
    const float* w_kernels_imag,    // (n_w_planes, kernel_size, kernel_size) W-kernels imag
    const float* w_values,          // (n_w_planes,) W values for interpolation
    float* vis_real,                // (N,) output predicted visibility real
    float* vis_imag,                // (N,) output predicted visibility imag
    int* valid_flags,               // (N,) 0=valid, 1=out of bounds
    const int n_vis,
    const int grid_size,
    const float cell_size,          // radians, for UV -> pixel conversion
    const int support,
    const int oversampling,
    const int n_w_planes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_vis) return;

    // Get UV coordinates
    float u = uvw[idx * 3 + 0];
    float v = uvw[idx * 3 + 1];
    float w = uvw[idx * 3 + 2];

    // Convert to pixel coordinates
    float u_pix_f = u * cell_size + grid_size / 2.0f;
    float v_pix_f = v * cell_size + grid_size / 2.0f;

    int u_pix = (int)floorf(u_pix_f);
    int v_pix = (int)floorf(v_pix_f);

    // Bounds check (need support pixels on each side)
    if (u_pix - support < 0 || u_pix + support >= grid_size ||
        v_pix - support < 0 || v_pix + support >= grid_size) {
        valid_flags[idx] = 1;  // Mark as invalid
        vis_real[idx] = 0.0f;
        vis_imag[idx] = 0.0f;
        return;
    }
    valid_flags[idx] = 0;  // Valid

    // Fractional part for GCF oversampling
    float u_frac = u_pix_f - u_pix;
    float v_frac = v_pix_f - v_pix;
    int u_off = (int)(u_frac * oversampling);
    int v_off = (int)(v_frac * oversampling);

    int gcf_width = 2 * support + 1;
    int kernel_size = gcf_width;

    // Find W-plane indices for interpolation
    // Clamp w to valid range
    float w_min = w_values[0];
    float w_max = w_values[n_w_planes - 1];
    float w_clamped = fminf(fmaxf(w, w_min), w_max);
    
    // Find interpolation indices and weight
    float w_frac_idx = (w_clamped - w_min) / (w_max - w_min) * (n_w_planes - 1);
    int w_idx_lo = (int)floorf(w_frac_idx);
    int w_idx_hi = w_idx_lo + 1;
    if (w_idx_hi >= n_w_planes) {
        w_idx_hi = n_w_planes - 1;
        w_idx_lo = w_idx_hi - 1;
    }
    if (w_idx_lo < 0) w_idx_lo = 0;
    float w_weight = w_frac_idx - w_idx_lo;

    // Accumulate visibility by convolving grid with GCF × W-kernel
    float acc_real = 0.0f;
    float acc_imag = 0.0f;

    for (int dv = -support; dv <= support; dv++) {
        int vv = v_pix + dv;
        int gcf_v_idx = (dv + support) + v_off * gcf_width;
        float gcf_v = gcf[gcf_v_idx];

        for (int du = -support; du <= support; du++) {
            int uu = u_pix + du;
            int gcf_u_idx = (du + support) + u_off * gcf_width;
            float gcf_u = gcf[gcf_u_idx];

            // GCF weight (separable)
            float gcf_weight = gcf_u * gcf_v;

            // W-kernel indices for this offset
            int wk_u = du + support;
            int wk_v = dv + support;
            int wk_idx = wk_v * kernel_size + wk_u;

            // Interpolate W-kernel between two planes
            int wk_offset_lo = w_idx_lo * kernel_size * kernel_size + wk_idx;
            int wk_offset_hi = w_idx_hi * kernel_size * kernel_size + wk_idx;

            float wk_real_lo = w_kernels_real[wk_offset_lo];
            float wk_imag_lo = w_kernels_imag[wk_offset_lo];
            float wk_real_hi = w_kernels_real[wk_offset_hi];
            float wk_imag_hi = w_kernels_imag[wk_offset_hi];

            float wk_real = wk_real_lo * (1.0f - w_weight) + wk_real_hi * w_weight;
            float wk_imag = wk_imag_lo * (1.0f - w_weight) + wk_imag_hi * w_weight;

            // Grid value at this pixel
            int grid_idx = vv * grid_size + uu;
            float g_real = grid_real[grid_idx];
            float g_imag = grid_imag[grid_idx];

            // Complex multiply: grid × conjugate(w_kernel) × gcf_weight
            // Note: For degridding, we use conjugate of W-kernel
            float prod_real = (g_real * wk_real + g_imag * wk_imag) * gcf_weight;
            float prod_imag = (g_imag * wk_real - g_real * wk_imag) * gcf_weight;

            acc_real += prod_real;
            acc_imag += prod_imag;
        }
    }

    vis_real[idx] = acc_real;
    vis_imag[idx] = acc_imag;
}
"""

# Simple degridding kernel without W-projection (for comparison/testing)
_DEGRID_SIMPLE_KERNEL = """
extern "C" __global__
void degrid_simple(
    const float* uvw,               // (N, 3) UVW coordinates
    const float* grid_real,         // (size, size) UV grid real
    const float* grid_imag,         // (size, size) UV grid imag
    const float* gcf,               // (oversampling, 2*support+1) GCF
    float* vis_real,                // (N,) output real
    float* vis_imag,                // (N,) output imag
    int* valid_flags,               // (N,) validity flags
    const int n_vis,
    const int grid_size,
    const float cell_size,
    const int support,
    const int oversampling
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_vis) return;

    float u = uvw[idx * 3 + 0];
    float v = uvw[idx * 3 + 1];

    float u_pix_f = u * cell_size + grid_size / 2.0f;
    float v_pix_f = v * cell_size + grid_size / 2.0f;

    int u_pix = (int)floorf(u_pix_f);
    int v_pix = (int)floorf(v_pix_f);

    if (u_pix - support < 0 || u_pix + support >= grid_size ||
        v_pix - support < 0 || v_pix + support >= grid_size) {
        valid_flags[idx] = 1;
        vis_real[idx] = 0.0f;
        vis_imag[idx] = 0.0f;
        return;
    }
    valid_flags[idx] = 0;

    float u_frac = u_pix_f - u_pix;
    float v_frac = v_pix_f - v_pix;
    int u_off = (int)(u_frac * oversampling);
    int v_off = (int)(v_frac * oversampling);

    int gcf_width = 2 * support + 1;

    float acc_real = 0.0f;
    float acc_imag = 0.0f;

    for (int dv = -support; dv <= support; dv++) {
        int vv = v_pix + dv;
        int gcf_v_idx = (dv + support) + v_off * gcf_width;
        float gcf_v = gcf[gcf_v_idx];

        for (int du = -support; du <= support; du++) {
            int uu = u_pix + du;
            int gcf_u_idx = (du + support) + u_off * gcf_width;
            float gcf_u = gcf[gcf_u_idx];

            float weight = gcf_u * gcf_v;

            int grid_idx = vv * grid_size + uu;
            acc_real += grid_real[grid_idx] * weight;
            acc_imag += grid_imag[grid_idx] * weight;
        }
    }

    vis_real[idx] = acc_real;
    vis_imag[idx] = acc_imag;
}
"""


# =============================================================================
# Compiled Kernels (lazy initialization)
# =============================================================================

_compiled_kernels = {}


def _get_kernel(name: str) -> RawKernel:
    """Get or compile a CUDA kernel.

    Parameters
    ----------
    name : str
        Kernel name: 'grid_nn', 'grid_conv', 'normalize', 'degrid_w', 'degrid_simple'
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available for GPU gridding")

    if name not in _compiled_kernels:
        kernel_code = {
            "grid_nn": _GRID_NN_KERNEL,
            "grid_conv": _GRID_CONV_KERNEL,
            "normalize": _NORMALIZE_KERNEL,
            "degrid_w": _DEGRID_W_KERNEL,
            "degrid_simple": _DEGRID_SIMPLE_KERNEL,
        }

        if name not in kernel_code:
            raise ValueError(f"Unknown kernel: {name}")

        kernel_name = {
            "grid_nn": "grid_nearest_neighbor",
            "grid_conv": "grid_convolutional",
            "normalize": "normalize_grid",
            "degrid_w": "degrid_w_projection",
            "degrid_simple": "degrid_simple",
        }[name]

        _compiled_kernels[name] = cp.RawKernel(
            kernel_code[name],
            kernel_name,
            options=("-std=c++11",),
        )

    return _compiled_kernels[name]


# =============================================================================
# Gridding Convolution Function (GCF)
# =============================================================================


def _compute_spheroidal_gcf(
    support: int = 3,
    oversampling: int = 128,
) -> np.ndarray:
    """Compute prolate spheroidal wave function for gridding.

    The spheroidal function minimizes aliasing in the image plane.

    Parameters
    ----------
    support :
        Half-width of convolution function in pixels
    oversampling :
        Oversampling factor

    Returns
    -------
        GCF array of shape (oversampling, 2*support+1)

    """
    width = 2 * support + 1
    gcf = np.zeros((oversampling, width), dtype=np.float32)

    # Approximation using Gaussian (simpler, still effective)
    # True spheroidal requires scipy.special.pro_ang1
    sigma = support / 2.5

    for i in range(oversampling):
        frac = i / oversampling
        for j in range(width):
            x = j - support + frac
            gcf[i, j] = np.exp(-0.5 * (x / sigma) ** 2)

        # Normalize each row
        gcf[i] /= gcf[i].sum()

    return gcf


# =============================================================================
# W-Projection Kernels for Degridding
# =============================================================================


def _compute_w_kernel(
    w_value: float,
    image_size: int,
    cell_size_rad: float,
    support: int,
    oversampling: int = 1,
) -> np.ndarray:
    """Compute W-projection kernel for a given w value.

    The W-kernel corrects for non-coplanar baselines by applying the
    appropriate phase shift in the UV plane. For degridding:

        V(u,v,w) = ∫∫ I(l,m) G(w,l,m) e^{-2πi(ul + vm)} dl dm

    where G(w,l,m) = e^{-2πi w(√(1-l²-m²) - 1)}

    Parameters
    ----------
    w_value : float
        W coordinate in wavelengths
    image_size : int
        Image size in pixels (determines l,m grid)
    cell_size_rad : float
        Cell size in radians
    support : int
        Half-width of kernel in pixels
    oversampling : int
        Oversampling factor (default 1 for W-kernels)

    Returns
    -------
    np.ndarray
        Complex W-kernel of shape (2*support+1, 2*support+1)
    """
    width = 2 * support + 1
    kernel = np.zeros((width, width), dtype=np.complex64)

    # Grid of l,m coordinates centered on kernel
    # Each pixel in UV corresponds to cell_size_rad in l,m
    for j in range(width):
        for i in range(width):
            # Position in l,m (relative to center)
            l = (i - support) * cell_size_rad
            m = (j - support) * cell_size_rad

            l2m2 = l * l + m * m

            # Check if within field of view (unit sphere constraint)
            if l2m2 < 1.0:
                n = np.sqrt(1.0 - l2m2)
                # W-projection phase: e^{-2πi w (n - 1)}
                phase = -2.0 * np.pi * w_value * (n - 1.0)
                kernel[j, i] = np.exp(1j * phase)

    # Normalize kernel
    kernel_sum = np.abs(kernel).sum()
    if kernel_sum > 0:
        kernel /= kernel_sum

    return kernel


class _WKernelCache:
    """LRU cache for W-projection kernels stored on GPU.

    Kernels are expensive to compute and transfer to GPU. This cache
    stores recently used kernels in GPU memory for reuse.

    The cache key is (w_planes, w_max, support, image_size) which
    uniquely identifies a set of W-kernels.
    """

    def __init__(self, max_entries: int = 8):
        """Initialize W-kernel cache.

        Parameters
        ----------
        max_entries : int
            Maximum number of kernel sets to cache (default 8)
        """
        self._cache: dict[tuple, tuple[np.ndarray, list]] = {}  # config -> (w_values, kernels)
        self._access_order: list[tuple] = []  # LRU tracking
        self._max_entries = max_entries
        self._lock = threading.Lock()

    def get_or_compute(
        self,
        config: DegridConfig,
        on_gpu: bool = True,
    ) -> tuple[np.ndarray, list]:
        """Get cached kernels or compute new ones.

        Parameters
        ----------
        config : DegridConfig
            Degridding configuration
        on_gpu : bool
            If True and CuPy available, return GPU arrays

        Returns
        -------
        tuple[np.ndarray, list]
            (w_values array, list of kernel arrays)
        """
        cache_key = config.cache_key

        with self._lock:
            if cache_key in self._cache:
                # Move to end of access order (most recently used)
                self._access_order.remove(cache_key)
                self._access_order.append(cache_key)
                logger.debug("W-kernel cache hit for config %s", cache_key)
                return self._cache[cache_key]

        # Compute new kernels (outside lock to allow parallelism)
        logger.info(
            "Computing W-kernels: %d planes, w_max=%.1f, support=%d",
            config.w_planes,
            config.w_max,
            config.support,
        )

        w_values = np.linspace(-config.w_max, config.w_max, config.w_planes)
        kernels_cpu = []

        for w_val in w_values:
            kernel = _compute_w_kernel(
                w_val,
                config.image_size,
                config.cell_size_rad,
                config.support,
            )
            kernels_cpu.append(kernel)

        # Convert to GPU if requested
        if on_gpu and CUPY_AVAILABLE:
            with cp.cuda.Device(config.gpu_id):
                w_values_out = cp.asarray(w_values.astype(np.float32))
                kernels_out = [cp.asarray(k) for k in kernels_cpu]
        else:
            w_values_out = w_values.astype(np.float32)
            kernels_out = kernels_cpu

        # Store in cache with lock
        with self._lock:
            # Evict oldest if at capacity
            while len(self._cache) >= self._max_entries:
                oldest_key = self._access_order.pop(0)
                evicted = self._cache.pop(oldest_key, None)
                # Free GPU memory if applicable
                if evicted is not None and CUPY_AVAILABLE:
                    try:
                        del evicted
                        cp.get_default_memory_pool().free_all_blocks()
                    except Exception:
                        pass
                logger.debug("Evicted W-kernel cache entry: %s", oldest_key)

            self._cache[cache_key] = (w_values_out, kernels_out)
            self._access_order.append(cache_key)

        return w_values_out, kernels_out

    def clear(self):
        """Clear all cached kernels and free GPU memory."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            if CUPY_AVAILABLE:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
        logger.info("W-kernel cache cleared")


# Global W-kernel cache instance
_w_kernel_cache = _WKernelCache(max_entries=8)


# =============================================================================
# Memory Estimation
# =============================================================================


def estimate_gridding_memory_gb(
    n_vis: int,
    image_size: int,
    support: int = 3,
    oversampling: int = 128,
) -> tuple[float, float]:
    """Estimate GPU and system memory required for gridding.

    Parameters
    ----------
    n_vis :
        Number of visibilities
    image_size :
        Image size in pixels
    support :
        Convolution support
    oversampling :
        Oversampling factor

    Returns
    -------
        Tuple of (gpu_gb, system_gb) memory estimates

    """
    bytes_per_gb = 1024**3

    # GPU memory:
    # - UVW: n_vis * 3 * 4 bytes (float32)
    # - vis_real, vis_imag: n_vis * 4 bytes each
    # - weights: n_vis * 4 bytes
    # - flags: n_vis * 4 bytes
    # - grid_real, grid_imag, weight_grid: image_size^2 * 4 bytes each
    # - GCF: oversampling * (2*support+1) * 4 bytes
    # - FFT workspace: ~2 * image_size^2 * 8 bytes (complex64)

    input_bytes = n_vis * (3 * 4 + 4 + 4 + 4 + 4)  # UVW, vis_r, vis_i, w, flags
    grid_bytes = image_size * image_size * 4 * 3  # 3 grids
    gcf_bytes = oversampling * (2 * support + 1) * 4
    fft_bytes = image_size * image_size * 8 * 2  # workspace

    gpu_gb = (input_bytes + grid_bytes + gcf_bytes + fft_bytes) / bytes_per_gb

    # System memory: input arrays + output image
    system_gb = (input_bytes + image_size * image_size * 8) / bytes_per_gb

    # Add 20% safety margin
    return gpu_gb * 1.2, system_gb * 1.2


# =============================================================================
# GPU Gridding Functions
# =============================================================================


def _grid_visibilities_cupy(
    uvw: np.ndarray,
    vis: np.ndarray,
    weights: np.ndarray,
    flags: np.ndarray | None,
    config: GriddingConfig,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Grid visibilities on GPU using CuPy.

    Parameters
    ----------
    uvw :
        (N, 3) UVW coordinates in wavelengths
    vis :
        (N,) complex visibilities
    weights :
        (N,) weights
    flags :
        (N,) flags (1 = flagged, 0 = valid) or None
    config :
        Gridding configuration
    uvw: np.ndarray :

    vis: np.ndarray :

    weights: np.ndarray :

    flags: Optional[np.ndarray] :

    Returns
    -------
        Tuple of (image, grid, weight_sum, n_flagged)

    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available for GPU gridding")

    n_vis = len(vis)
    image_size = config.image_size

    # Create flags if not provided
    if flags is None:
        flags = np.zeros(n_vis, dtype=np.int32)
    else:
        flags = flags.astype(np.int32)

    n_flagged = int(np.sum(flags))

    # Prepare visibility data (split complex)
    vis_real = np.real(vis).astype(np.float32)
    vis_imag = np.imag(vis).astype(np.float32)
    weights = weights.astype(np.float32)
    uvw = uvw.astype(np.float32)

    # Use safe GPU context for memory protection
    with safe_gpu_context(gpu_id=config.gpu_id, max_gpu_gb=9.0):
        with cp.cuda.Device(config.gpu_id):
            # Transfer to GPU
            uvw_gpu = cp.asarray(uvw)
            vis_real_gpu = cp.asarray(vis_real)
            vis_imag_gpu = cp.asarray(vis_imag)
            weights_gpu = cp.asarray(weights)
            flags_gpu = cp.asarray(flags)

            # Allocate grids
            grid_real_gpu = cp.zeros((image_size, image_size), dtype=cp.float32)
            grid_imag_gpu = cp.zeros((image_size, image_size), dtype=cp.float32)
            weight_grid_gpu = cp.zeros((image_size, image_size), dtype=cp.float32)

            # Select kernel based on configuration
            if config.support > 0:
                # Convolutional gridding
                gcf = _compute_spheroidal_gcf(config.support, config.oversampling)
                gcf_gpu = cp.asarray(gcf)

                kernel = _get_kernel("grid_conv")
                threads = 256
                blocks = (n_vis + threads - 1) // threads

                kernel(
                    (blocks,),
                    (threads,),
                    (
                        uvw_gpu,
                        vis_real_gpu,
                        vis_imag_gpu,
                        weights_gpu,
                        flags_gpu,
                        gcf_gpu,
                        grid_real_gpu,
                        grid_imag_gpu,
                        weight_grid_gpu,
                        np.int32(n_vis),
                        np.int32(image_size),
                        np.float32(config.cell_size_rad),
                        np.int32(config.support),
                        np.int32(config.oversampling),
                    ),
                )
            else:
                # Nearest-neighbor gridding
                kernel = _get_kernel("grid_nn")
                threads = 256
                blocks = (n_vis + threads - 1) // threads

                kernel(
                    (blocks,),
                    (threads,),
                    (
                        uvw_gpu,
                        vis_real_gpu,
                        vis_imag_gpu,
                        weights_gpu,
                        flags_gpu,
                        grid_real_gpu,
                        grid_imag_gpu,
                        weight_grid_gpu,
                        np.int32(n_vis),
                        np.int32(image_size),
                        np.float32(config.cell_size_rad),
                    ),
                )

            # Weight sum
            weight_sum = float(cp.sum(weight_grid_gpu))

            # Normalize if requested
            if config.normalize and weight_sum > 0:
                normalize_kernel = _get_kernel("normalize")
                n_pixels = image_size * image_size
                blocks_norm = (n_pixels + threads - 1) // threads

                normalize_kernel(
                    (blocks_norm,),
                    (threads,),
                    (
                        grid_real_gpu,
                        grid_imag_gpu,
                        weight_grid_gpu,
                        np.int32(image_size),
                        np.float32(1e-10),
                    ),
                )

            # Combine to complex grid
            grid_gpu = grid_real_gpu + 1j * grid_imag_gpu

            # FFT to image plane
            image_gpu = cp.fft.ifft2(cp.fft.ifftshift(grid_gpu))
            image_gpu = cp.fft.fftshift(image_gpu)

            # Transfer back to CPU
            image = cp.asnumpy(image_gpu)
            grid = cp.asnumpy(grid_gpu)

            # Cleanup GPU memory
            del uvw_gpu, vis_real_gpu, vis_imag_gpu, weights_gpu, flags_gpu
            del grid_real_gpu, grid_imag_gpu, weight_grid_gpu, grid_gpu, image_gpu
            if config.support > 0:
                del gcf_gpu
            cp.get_default_memory_pool().free_all_blocks()

    return image, grid, weight_sum, n_flagged


if HAVE_NUMBA:
    # NOTE: parallel=True is explicitly disabled here to prevent race conditions.
    # In a scatter-add operation (gridding), multiple threads might try to update
    # the same grid pixel simultaneously. Numba's parallel CPU backend does not
    # support atomic complex additions efficiently or safely by default.
    # While serial execution is slower, it guarantees scientific correctness.
    @numba.jit(nopython=True, parallel=False, fastmath=True)
    def _grid_convolutional_numba(
        u_pix_int, v_pix_int, u_off, v_off,
        vis_valid, w_valid,
        gcf, support, image_size,
        grid, weight_grid
    ):
        """Numba-accelerated convolutional gridding kernel.
        
        This replaces the slow Python loop for CPU-based gridding.
        """
        n_vis = len(u_pix_int)
        
        # Iterate over visibilities (serial for safety)
        for i in range(n_vis):
            ui = u_pix_int[i]
            vi = v_pix_int[i]
            
            # Bounds check (kernel footprint)
            if (ui - support < 0 or ui + support >= image_size or
                vi - support < 0 or vi + support >= image_size):
                continue
                
            w = w_valid[i]
            val = vis_valid[i] * w
            
            u_o = u_off[i]
            v_o = v_off[i]
            
            # Convolve (inner loops)
            for dv in range(-support, support + 1):
                vv = vi + dv
                gcf_v = gcf[v_o, dv + support]
                
                for du in range(-support, support + 1):
                    uu = ui + du
                    gcf_u = gcf[u_o, du + support]
                    
                    conv_weight = gcf_u * gcf_v
                    
                    grid[vv, uu] += val * conv_weight
                    weight_grid[vv, uu] += w * conv_weight
else:
    def _grid_convolutional_numba(*args, **kwargs):
        """Placeholder for when numba is not available."""
        pass


def _grid_visibilities_cpu(
    uvw: np.ndarray,
    vis: np.ndarray,
    weights: np.ndarray,
    flags: np.ndarray | None,
    config: GriddingConfig,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """CPU fallback for gridding (numpy/numba implementation).

    Supports both nearest-neighbor and convolutional gridding.

    Parameters
    ----------
    uvw :
        (N, 3) UVW coordinates in wavelengths
    vis :
        (N,) complex visibilities
    weights :
        (N,) weights
    flags :
        (N,) flags or None
    config :
        Gridding configuration

    Returns
    -------
        Tuple of (image, grid, weight_sum, n_flagged)

    """
    # Import numba lazily
    try:
        import numba
        HAVE_NUMBA = True
    except ImportError:
        HAVE_NUMBA = False

    n_vis = len(vis)
    image_size = config.image_size
    cell_size = config.cell_size_rad
    support = config.support
    oversampling = config.oversampling

    # Create flags if not provided
    if flags is None:
        flags = np.zeros(n_vis, dtype=bool)
    else:
        flags = flags.astype(bool)

    n_flagged = int(np.sum(flags))

    # Initialize grids
    grid = np.zeros((image_size, image_size), dtype=np.complex64)
    weight_grid = np.zeros((image_size, image_size), dtype=np.float32)

    # Mask valid data
    valid = ~flags
    u = uvw[valid, 0]
    v = uvw[valid, 1]
    vis_valid = vis[valid]
    w_valid = weights[valid]

    # Convert to pixel coordinates
    # u_pix_f = u / cell_size + image_size / 2
    u_pix_f = u * cell_size + image_size / 2.0
    v_pix_f = v * cell_size + image_size / 2.0

    if support == 0:
        # Nearest-neighbor gridding
        u_pix = (u_pix_f + 0.5).astype(np.int64)
        v_pix = (v_pix_f + 0.5).astype(np.int64)

        # Filter out-of-bounds
        in_bounds = (u_pix >= 0) & (u_pix < image_size) & (v_pix >= 0) & (v_pix < image_size)

        u_pix = u_pix[in_bounds]
        v_pix = v_pix[in_bounds]
        vis_valid = vis_valid[in_bounds]
        w_valid = w_valid[in_bounds]

        # Grid using numpy
        np.add.at(grid, (v_pix, u_pix), vis_valid * w_valid)
        np.add.at(weight_grid, (v_pix, u_pix), w_valid)

    else:
        # Convolutional gridding
        # Calculate integer and fractional parts
        u_pix_int = np.floor(u_pix_f).astype(np.int64)
        v_pix_int = np.floor(v_pix_f).astype(np.int64)
        
        u_frac = u_pix_f - u_pix_int
        v_frac = v_pix_f - v_pix_int
        
        # Calculate oversampling offsets
        u_off = (u_frac * oversampling).astype(np.int64)
        v_off = (v_frac * oversampling).astype(np.int64)
        
        # Pre-compute GCF
        gcf = _compute_spheroidal_gcf(support, oversampling)
        
        if HAVE_NUMBA:
            # Use JIT compiled kernel (defined locally to capture numba import)
            # We define it outside if possible, but here we do it dynamically if needed
            # or rely on the module-level one if numba was imported at top level.
            # To be safe, we'll assume the module level function handles the JIT
            # if numba is available.
            _grid_convolutional_numba(
                u_pix_int, v_pix_int, u_off, v_off,
                vis_valid, w_valid,
                gcf, support, image_size,
                grid, weight_grid
            )
        else:
            # Slow pure Python fallback
            gcf_width = 2 * support + 1
            for i in range(len(u_pix_int)):
                ui = u_pix_int[i]
                vi = v_pix_int[i]
                
                if (ui - support < 0 or ui + support >= image_size or
                    vi - support < 0 or vi + support >= image_size):
                    continue
                    
                w = w_valid[i]
                val = vis_valid[i] * w
                u_o = u_off[i]
                v_o = v_off[i]
                
                for dv in range(-support, support + 1):
                    vv = vi + dv
                    gcf_v = gcf[v_o, dv + support]
                    for du in range(-support, support + 1):
                        uu = ui + du
                        gcf_u = gcf[u_o, du + support]
                        conv_weight = gcf_u * gcf_v
                        grid[vv, uu] += val * conv_weight
                        weight_grid[vv, uu] += w * conv_weight

    weight_sum = float(weight_grid.sum())

    # Normalize
    if config.normalize and weight_sum > 0:
        nonzero = weight_grid > 1e-10
        grid[nonzero] /= weight_grid[nonzero]

    # FFT to image plane
    image = np.fft.ifft2(np.fft.ifftshift(grid))
    image = np.fft.fftshift(image)

    return image, grid, weight_sum, n_flagged


# =============================================================================
# Public API
# =============================================================================


@gpu_safe(max_gpu_gb=9.0, max_system_gb=6.0)
def gpu_grid_visibilities(
    uvw: np.ndarray,
    vis: np.ndarray,
    weights: np.ndarray,
    *,
    config: GriddingConfig | None = None,
    image_size: int | None = None,
    cell_size_arcsec: float | None = None,
    gpu_id: int | None = None,
    flags: np.ndarray | None = None,
    return_grid: bool = False,
) -> GriddingResult:
    """GPU-accelerated visibility gridding.

    Grids visibility data onto a UV plane and performs FFT to produce
    a dirty image. Uses CuPy for GPU acceleration with automatic
    memory safety guards.

    Parameters
    ----------
    uvw :

    vis :

    weights :

    config :
        Gridding configuration
    image_size :
        Override image size from config
    cell_size_arcsec :
        Override cell size from config
    gpu_id :
        Override GPU ID from config
    flags :

    return_grid :
        Include UV grid in result
    uvw: np.ndarray :

    vis: np.ndarray :

    weights: np.ndarray :

    * :

    config: Optional[GriddingConfig] :
         (Default value = None)
    image_size: Optional[int] :
         (Default value = None)
    cell_size_arcsec: Optional[float] :
         (Default value = None)
    gpu_id: Optional[int] :
         (Default value = None)
    flags: Optional[np.ndarray] :
         (Default value = None)

    Returns
    -------
    type
        GriddingResult with image and metadata

    """
    start_time = time.time()

    # Build configuration
    if config is None:
        config = GriddingConfig()
    if image_size is not None:
        config.image_size = image_size
    if cell_size_arcsec is not None:
        config.cell_size_arcsec = cell_size_arcsec
    if gpu_id is not None:
        config.gpu_id = gpu_id

    # Validate inputs
    if uvw.ndim != 2 or uvw.shape[1] != 3:
        return GriddingResult(
            error=f"UVW must be (N, 3), got {uvw.shape}",
            gpu_id=config.gpu_id,
        )

    if len(vis) != len(uvw):
        return GriddingResult(
            error=f"vis length {len(vis)} != uvw length {len(uvw)}",
            gpu_id=config.gpu_id,
        )

    if len(weights) != len(vis):
        return GriddingResult(
            error=f"weights length {len(weights)} != vis length {len(vis)}",
            gpu_id=config.gpu_id,
        )

    n_vis = len(vis)

    # Check memory requirements
    _gpu_gb, sys_gb = estimate_gridding_memory_gb(
        n_vis, config.image_size, config.support, config.oversampling
    )
    del _gpu_gb  # GPU memory checked by @gpu_safe decorator

    is_safe, reason = check_system_memory_available(sys_gb)
    if not is_safe:
        return GriddingResult(
            error=f"Insufficient system memory: {reason}",
            n_vis=n_vis,
            gpu_id=config.gpu_id,
        )

    # Run gridding
    try:
        if CUPY_AVAILABLE:
            logger.info(
                "GPU gridding %s visibilities to %dx%d image on GPU %d",
                f"{n_vis:,}",
                config.image_size,
                config.image_size,
                config.gpu_id,
            )
            image, grid, weight_sum, n_flagged = _grid_visibilities_cupy(
                uvw, vis, weights, flags, config
            )
        else:
            logger.warning("CuPy not available, using CPU fallback")
            image, grid, weight_sum, n_flagged = _grid_visibilities_cpu(
                uvw, vis, weights, flags, config
            )

        processing_time = time.time() - start_time

        logger.info(
            "Gridding complete: %s visibilities gridded (%s flagged) in %.2fs",
            f"{n_vis - n_flagged:,}",
            f"{n_flagged:,}",
            processing_time,
        )

        return GriddingResult(
            image=np.abs(image),  # Return amplitude image
            grid=grid if return_grid else None,
            weight_sum=weight_sum,
            n_vis=n_vis,
            n_flagged=n_flagged,
            processing_time_s=processing_time,
            gpu_id=config.gpu_id,
        )

    except MemoryError as err:
        logger.error("Memory error during gridding: %s", err)
        return GriddingResult(
            error=f"Memory error: {err}",
            n_vis=n_vis,
            gpu_id=config.gpu_id,
        )
    except RuntimeError as err:
        logger.error("Runtime error during gridding: %s", err)
        return GriddingResult(
            error=f"Runtime error: {err}",
            n_vis=n_vis,
            gpu_id=config.gpu_id,
        )


def cpu_grid_visibilities(
    uvw: np.ndarray,
    vis: np.ndarray,
    weights: np.ndarray,
    *,
    config: GriddingConfig | None = None,
    image_size: int | None = None,
    cell_size_arcsec: float | None = None,
    flags: np.ndarray | None = None,
    return_grid: bool = False,
) -> GriddingResult:
    """CPU-only visibility gridding (no GPU required).

    This is a fallback for systems without GPU support.

    Parameters
    ----------
    uvw :

    vis :

    weights :

    config :
        Gridding configuration
    image_size :
        Override image size from config
    cell_size_arcsec :
        Override cell size from config
    flags :

    return_grid :
        Include UV grid in result
    uvw: np.ndarray :

    vis: np.ndarray :

    weights: np.ndarray :

    * :

    config: Optional[GriddingConfig] :
         (Default value = None)
    image_size: Optional[int] :
         (Default value = None)
    cell_size_arcsec: Optional[float] :
         (Default value = None)
    flags: Optional[np.ndarray] :
         (Default value = None)

    Returns
    -------
    type
        GriddingResult with image and metadata

    """
    start_time = time.time()

    # Build configuration
    if config is None:
        config = GriddingConfig()
    if image_size is not None:
        config.image_size = image_size
    if cell_size_arcsec is not None:
        config.cell_size_arcsec = cell_size_arcsec

    # Validate inputs
    if uvw.ndim != 2 or uvw.shape[1] != 3:
        return GriddingResult(
            error=f"UVW must be (N, 3), got {uvw.shape}",
        )

    if len(vis) != len(uvw) or len(weights) != len(vis):
        return GriddingResult(
            error="Input array lengths must match",
        )

    n_vis = len(vis)

    try:
        logger.info(
            "CPU gridding %s visibilities to %dx%d image",
            f"{n_vis:,}",
            config.image_size,
            config.image_size,
        )

        image, grid, weight_sum, n_flagged = _grid_visibilities_cpu(
            uvw, vis, weights, flags, config
        )

        processing_time = time.time() - start_time

        logger.info(
            "CPU gridding complete: %s visibilities in %.2fs",
            f"{n_vis - n_flagged:,}",
            processing_time,
        )

        return GriddingResult(
            image=np.abs(image),
            grid=grid if return_grid else None,
            weight_sum=weight_sum,
            n_vis=n_vis,
            n_flagged=n_flagged,
            processing_time_s=processing_time,
            gpu_id=-1,  # Indicates CPU
        )

    except MemoryError as err:
        logger.error("Memory error during CPU gridding: %s", err)
        return GriddingResult(
            error=f"Memory error: {err}",
            n_vis=n_vis,
        )


# =============================================================================
# MS Integration
# =============================================================================


def grid_ms(
    ms_path: str,
    *,
    config: GriddingConfig | None = None,
    datacolumn: str = "DATA",
) -> GriddingResult:
    """Grid visibilities from a measurement set.

    Parameters
    ----------
    ms_path :
        Path to measurement set
    config :
        Gridding configuration
    datacolumn :
        Data column to grid (DATA, CORRECTED_DATA, MODEL_DATA)

    Returns
    -------
        GriddingResult with image

    """
    from pathlib import Path

    if not Path(ms_path).exists():
        return GriddingResult(
            error=f"MS not found: {ms_path}",
        )

    try:
        from casatools import table as tb
    except ImportError:
        return GriddingResult(
            error="casatools not available for MS reading",
        )

    if config is None:
        config = GriddingConfig()

    start_time = time.time()

    try:
        # Open MS and read data
        tbl = tb()
        tbl.open(ms_path)

        uvw = tbl.getcol("UVW").T  # Transpose to (N, 3)
        vis_data = tbl.getcol(datacolumn)  # (nchan, ncorr, nrow) or similar
        weights = tbl.getcol("WEIGHT")
        flags = tbl.getcol("FLAG")

        tbl.close()

        # Flatten multi-dimensional data
        # Assuming vis_data is (nchan, ncorr, nrow), flatten to (N,)
        if vis_data.ndim == 3:
            # Average channels and polarizations for continuum
            vis = vis_data.mean(axis=(0, 1))
            flag_any = flags.any(axis=(0, 1))
            # Expand UVW for each channel (if needed)
            # For continuum, just use the original UVW
            uvw_flat = uvw
            weights_flat = weights.mean(axis=0) if weights.ndim > 1 else weights
            flags_flat = flag_any.astype(np.int32)
        else:
            vis = vis_data.ravel()
            uvw_flat = uvw
            weights_flat = weights.ravel() if weights.ndim > 1 else weights
            flags_flat = flags.ravel().astype(np.int32) if flags.ndim > 1 else flags

        # Grid
        result = gpu_grid_visibilities(
            uvw_flat,
            vis,
            weights_flat,
            config=config,
            flags=flags_flat,
        )

        # Add MS path to result
        result.processing_time_s = time.time() - start_time

        return result

    except OSError as err:
        return GriddingResult(error=f"Error reading MS: {err}")
    except RuntimeError as err:
        return GriddingResult(error=f"Runtime error: {err}")


# =============================================================================
# GPU Degridding (Visibility Prediction)
# =============================================================================


def estimate_degridding_memory_gb(
    n_vis: int,
    image_size: int,
    config: DegridConfig,
) -> tuple[float, float]:
    """Estimate GPU and system memory required for degridding.

    Parameters
    ----------
    n_vis : int
        Number of visibilities to predict
    image_size : int
        Image size in pixels
    config : DegridConfig
        Degridding configuration

    Returns
    -------
    tuple[float, float]
        (gpu_gb, system_gb) memory estimates
    """
    bytes_per_gb = 1024**3

    # GPU memory:
    # - UVW: n_vis * 3 * 4 bytes
    # - grid_real, grid_imag: image_size^2 * 4 bytes each
    # - GCF: oversampling * (2*support+1) * 4 bytes
    # - W-kernels: w_planes * kernel_size^2 * 8 bytes (complex)
    # - vis_real, vis_imag: n_vis * 4 bytes each
    # - valid_flags: n_vis * 4 bytes

    kernel_size = 2 * config.support + 1
    input_bytes = n_vis * 3 * 4  # UVW
    grid_bytes = image_size * image_size * 4 * 2  # real + imag
    gcf_bytes = config.oversampling * kernel_size * 4
    w_kernel_bytes = config.w_planes * kernel_size * kernel_size * 8
    output_bytes = n_vis * (4 + 4 + 4)  # vis_real, vis_imag, flags

    gpu_gb = (input_bytes + grid_bytes + gcf_bytes + w_kernel_bytes + output_bytes) / bytes_per_gb

    # System memory: input + output arrays
    system_gb = (input_bytes + image_size * image_size * 8 + n_vis * 16) / bytes_per_gb

    return gpu_gb * 1.2, system_gb * 1.2


def _degrid_visibilities_cupy(
    uvw: np.ndarray,
    grid: np.ndarray,
    config: DegridConfig,
) -> tuple[np.ndarray, int]:
    """Degrid visibilities on GPU using CuPy.

    Parameters
    ----------
    uvw : np.ndarray
        (N, 3) UVW coordinates in wavelengths
    grid : np.ndarray
        (image_size, image_size) complex UV grid
    config : DegridConfig
        Degridding configuration

    Returns
    -------
    tuple[np.ndarray, int]
        (predicted_vis, n_skipped) where predicted_vis is complex (N,)
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available for GPU degridding")

    n_vis = len(uvw)
    image_size = config.image_size

    # Split grid to real/imag
    grid_real = np.real(grid).astype(np.float32)
    grid_imag = np.imag(grid).astype(np.float32)
    uvw = uvw.astype(np.float32)

    # Compute GCF
    gcf = _compute_spheroidal_gcf(config.support, config.oversampling)

    # Get or compute W-kernels
    w_values, w_kernels = _w_kernel_cache.get_or_compute(config, on_gpu=True)

    with safe_gpu_context(gpu_id=config.gpu_id, max_gpu_gb=9.0):
        with cp.cuda.Device(config.gpu_id):
            # Transfer to GPU
            uvw_gpu = cp.asarray(uvw)
            grid_real_gpu = cp.asarray(grid_real)
            grid_imag_gpu = cp.asarray(grid_imag)
            gcf_gpu = cp.asarray(gcf)

            # Stack W-kernels into contiguous array
            kernel_size = 2 * config.support + 1
            n_w_planes = config.w_planes

            # If w_kernels are already on GPU, stack them
            if isinstance(w_kernels[0], cp.ndarray):
                w_kernels_stacked = cp.stack(w_kernels, axis=0)  # (n_w, ks, ks)
            else:
                w_kernels_stacked = cp.asarray(np.stack(w_kernels, axis=0))

            w_kernels_real_gpu = cp.asarray(
                np.real(cp.asnumpy(w_kernels_stacked)).astype(np.float32)
                if isinstance(w_kernels_stacked, cp.ndarray)
                else np.real(w_kernels_stacked).astype(np.float32)
            )
            w_kernels_imag_gpu = cp.asarray(
                np.imag(cp.asnumpy(w_kernels_stacked)).astype(np.float32)
                if isinstance(w_kernels_stacked, cp.ndarray)
                else np.imag(w_kernels_stacked).astype(np.float32)
            )

            # w_values array
            if not isinstance(w_values, cp.ndarray):
                w_values_gpu = cp.asarray(w_values)
            else:
                w_values_gpu = w_values

            # Allocate outputs
            vis_real_gpu = cp.zeros(n_vis, dtype=cp.float32)
            vis_imag_gpu = cp.zeros(n_vis, dtype=cp.float32)
            valid_flags_gpu = cp.zeros(n_vis, dtype=cp.int32)

            # Select kernel based on W-projection setting
            threads = 256
            blocks = (n_vis + threads - 1) // threads

            if config.use_w_projection:
                kernel = _get_kernel("degrid_w")
                kernel(
                    (blocks,),
                    (threads,),
                    (
                        uvw_gpu,
                        grid_real_gpu,
                        grid_imag_gpu,
                        gcf_gpu,
                        w_kernels_real_gpu,
                        w_kernels_imag_gpu,
                        w_values_gpu,
                        vis_real_gpu,
                        vis_imag_gpu,
                        valid_flags_gpu,
                        np.int32(n_vis),
                        np.int32(image_size),
                        np.float32(config.cell_size_rad),
                        np.int32(config.support),
                        np.int32(config.oversampling),
                        np.int32(n_w_planes),
                    ),
                )
            else:
                kernel = _get_kernel("degrid_simple")
                kernel(
                    (blocks,),
                    (threads,),
                    (
                        uvw_gpu,
                        grid_real_gpu,
                        grid_imag_gpu,
                        gcf_gpu,
                        vis_real_gpu,
                        vis_imag_gpu,
                        valid_flags_gpu,
                        np.int32(n_vis),
                        np.int32(image_size),
                        np.float32(config.cell_size_rad),
                        np.int32(config.support),
                        np.int32(config.oversampling),
                    ),
                )

            # Transfer results back
            vis_real = cp.asnumpy(vis_real_gpu)
            vis_imag = cp.asnumpy(vis_imag_gpu)
            valid_flags = cp.asnumpy(valid_flags_gpu)

            # Cleanup
            del uvw_gpu, grid_real_gpu, grid_imag_gpu, gcf_gpu
            del w_kernels_real_gpu, w_kernels_imag_gpu, w_values_gpu
            del vis_real_gpu, vis_imag_gpu, valid_flags_gpu
            cp.get_default_memory_pool().free_all_blocks()

    # Combine to complex
    vis_predicted = vis_real + 1j * vis_imag
    n_skipped = int(valid_flags.sum())

    return vis_predicted, n_skipped


def _degrid_visibilities_cpu(
    uvw: np.ndarray,
    grid: np.ndarray,
    config: DegridConfig,
) -> tuple[np.ndarray, int]:
    """CPU fallback for degridding.

    Parameters
    ----------
    uvw : np.ndarray
        (N, 3) UVW coordinates in wavelengths
    grid : np.ndarray
        (image_size, image_size) complex UV grid
    config : DegridConfig
        Degridding configuration

    Returns
    -------
    tuple[np.ndarray, int]
        (predicted_vis, n_skipped)
    """
    n_vis = len(uvw)
    image_size = config.image_size
    cell_size = config.cell_size_rad
    support = config.support
    oversampling = config.oversampling

    # Compute GCF
    gcf = _compute_spheroidal_gcf(support, oversampling)

    # Compute W-kernels (CPU)
    w_values, w_kernels = _w_kernel_cache.get_or_compute(config, on_gpu=False)

    # Split grid
    grid_real = np.real(grid).astype(np.float32)
    grid_imag = np.imag(grid).astype(np.float32)

    # Output arrays
    vis_predicted = np.zeros(n_vis, dtype=np.complex64)
    n_skipped = 0

    for idx in range(n_vis):
        u, v, w = uvw[idx]

        # Convert to pixel coordinates
        u_pix_f = u * cell_size + image_size / 2.0
        v_pix_f = v * cell_size + image_size / 2.0

        u_pix = int(np.floor(u_pix_f))
        v_pix = int(np.floor(v_pix_f))

        # Bounds check
        if (u_pix - support < 0 or u_pix + support >= image_size or
            v_pix - support < 0 or v_pix + support >= image_size):
            n_skipped += 1
            continue

        # Fractional part for oversampling
        u_frac = u_pix_f - u_pix
        v_frac = v_pix_f - v_pix
        u_off = int(u_frac * oversampling)
        v_off = int(v_frac * oversampling)

        # Find W-kernel indices for interpolation
        w_min, w_max = w_values[0], w_values[-1]
        w_clamped = np.clip(w, w_min, w_max)
        w_frac_idx = (w_clamped - w_min) / (w_max - w_min) * (len(w_values) - 1)
        w_idx_lo = int(np.floor(w_frac_idx))
        w_idx_hi = min(w_idx_lo + 1, len(w_values) - 1)
        w_weight = w_frac_idx - w_idx_lo

        # Accumulate
        acc = 0.0 + 0.0j
        gcf_width = 2 * support + 1

        for dv in range(-support, support + 1):
            vv = v_pix + dv
            gcf_v = gcf[v_off, dv + support]

            for du in range(-support, support + 1):
                uu = u_pix + du
                gcf_u = gcf[u_off, du + support]
                gcf_weight = gcf_u * gcf_v

                # W-kernel interpolation
                wk_u = du + support
                wk_v = dv + support

                if config.use_w_projection:
                    wk_lo = w_kernels[w_idx_lo][wk_v, wk_u]
                    wk_hi = w_kernels[w_idx_hi][wk_v, wk_u]
                    wk = wk_lo * (1.0 - w_weight) + wk_hi * w_weight
                    wk_conj = np.conj(wk)
                else:
                    wk_conj = 1.0

                # Grid value
                g_val = grid_real[vv, uu] + 1j * grid_imag[vv, uu]

                acc += g_val * wk_conj * gcf_weight

        vis_predicted[idx] = acc

    return vis_predicted, n_skipped


@gpu_safe(max_gpu_gb=9.0, max_system_gb=6.0)
def gpu_degrid_visibilities(
    uvw: np.ndarray,
    sky_image: np.ndarray,
    *,
    config: DegridConfig | None = None,
    image_size: int | None = None,
    cell_size_arcsec: float | None = None,
    gpu_id: int | None = None,
) -> DegridResult:
    """GPU-accelerated visibility prediction (degridding).

    Given a sky model image, predict visibilities at specified UVW coordinates.
    This is the inverse of gridding - used for calibration with sky models.

    The workflow:
    1. FFT sky image to UV plane
    2. Sample UV grid at (u,v,w) coordinates using W-projection
    3. Return predicted complex visibilities

    Parameters
    ----------
    uvw : np.ndarray
        (N, 3) UVW coordinates in wavelengths
    sky_image : np.ndarray
        (image_size, image_size) sky model image (Stokes I)
    config : DegridConfig, optional
        Degridding configuration
    image_size : int, optional
        Override image size from config
    cell_size_arcsec : float, optional
        Override cell size from config
    gpu_id : int, optional
        Override GPU ID from config

    Returns
    -------
    DegridResult
        Result containing predicted visibilities and metadata
    """
    start_time = time.time()

    # Build configuration
    if config is None:
        config = DegridConfig()
    if image_size is not None:
        config.image_size = image_size
    if cell_size_arcsec is not None:
        config.cell_size_arcsec = cell_size_arcsec
    if gpu_id is not None:
        config.gpu_id = gpu_id

    # Validate inputs
    if uvw.ndim != 2 or uvw.shape[1] != 3:
        return DegridResult(
            error=f"UVW must be (N, 3), got {uvw.shape}",
            gpu_id=config.gpu_id,
        )

    if sky_image.ndim != 2:
        return DegridResult(
            error=f"sky_image must be 2D, got {sky_image.ndim}D",
            gpu_id=config.gpu_id,
        )

    n_vis = len(uvw)

    # Check memory requirements
    gpu_gb, sys_gb = estimate_degridding_memory_gb(n_vis, config.image_size, config)

    is_safe, reason = check_system_memory_available(sys_gb)
    if not is_safe:
        return DegridResult(
            error=f"Insufficient system memory: {reason}",
            n_vis=n_vis,
            gpu_id=config.gpu_id,
        )

    try:
        # FFT sky image to UV plane
        # fftshift to center, then FFT, then ifftshift
        sky_centered = np.fft.ifftshift(sky_image.astype(np.complex64))
        uv_grid = np.fft.fft2(sky_centered)
        uv_grid = np.fft.fftshift(uv_grid)

        if CUPY_AVAILABLE:
            logger.info(
                "GPU degridding %s visibilities from %dx%d image on GPU %d",
                f"{n_vis:,}",
                config.image_size,
                config.image_size,
                config.gpu_id,
            )
            vis_predicted, n_skipped = _degrid_visibilities_cupy(uvw, uv_grid, config)
        else:
            logger.warning("CuPy not available, using CPU fallback for degridding")
            vis_predicted, n_skipped = _degrid_visibilities_cpu(uvw, uv_grid, config)

        processing_time = time.time() - start_time

        logger.info(
            "Degridding complete: %s visibilities predicted (%s skipped) in %.3fs",
            f"{n_vis - n_skipped:,}",
            f"{n_skipped:,}",
            processing_time,
        )

        return DegridResult(
            vis_predicted=vis_predicted,
            n_vis=n_vis,
            n_skipped=n_skipped,
            processing_time_s=processing_time,
            gpu_id=config.gpu_id,
        )

    except MemoryError as err:
        logger.error("Memory error during degridding: %s", err)
        return DegridResult(
            error=f"Memory error: {err}",
            n_vis=n_vis,
            gpu_id=config.gpu_id,
        )
    except RuntimeError as err:
        logger.error("Runtime error during degridding: %s", err)
        return DegridResult(
            error=f"Runtime error: {err}",
            n_vis=n_vis,
            gpu_id=config.gpu_id,
        )


def clear_w_kernel_cache() -> None:
    """Clear the W-kernel cache and free GPU memory.

    Call this when degridding configuration changes significantly
    or to reclaim GPU memory.
    """
    _w_kernel_cache.clear()
