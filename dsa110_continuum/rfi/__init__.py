"""
RFI Detection Module.

GPU-accelerated Radio Frequency Interference detection for DSA-110.

This module provides CuPy-based RFI detection algorithms that run on GPU
for improved performance. Due to driver constraints (455.23.05, CUDA 11.1 max),
we use CuPy exclusively rather than Numba CUDA kernels.

Main Components:
    - gpu_detection: GPU-accelerated MAD-based outlier detection
    - strategies: Different RFI detection strategies (MAD, SumThreshold, etc.)
    - flagging: Apply flags to measurement sets

Usage:
    from dsa110_contimg.core.rfi import gpu_rfi_detection, RFIDetectionResult

    result = gpu_rfi_detection(
        ms_path="/path/to/data.ms",
        threshold=5.0,  # MAD units
        gpu_id=0
    )
    print(f"Flagged {result.flag_percent:.2f}% of data")
"""

from .gpu_detection import (
    RFIDetectionConfig,
    RFIDetectionResult,
    gpu_rfi_detection,
)

__all__ = [
    "gpu_rfi_detection",
    "RFIDetectionResult",
    "RFIDetectionConfig",
]
