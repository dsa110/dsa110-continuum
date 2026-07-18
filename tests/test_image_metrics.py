"""Tests for image-metric backend selection."""

import numpy as np


def test_gpu_allocation_failure_falls_back_to_numpy(monkeypatch):
    from dsa110_continuum.qa import image_metrics

    class BrokenGpu:
        @staticmethod
        def asarray(array):
            raise RuntimeError("CUDA initialization failed")

    monkeypatch.setattr(image_metrics, "get_array_module", lambda **kwargs: (BrokenGpu, True))
    array = np.ones((2, 2))

    backend_array, xp, is_gpu = image_metrics._maybe_to_gpu(array, min_elements=1)

    assert backend_array is array
    assert xp is np
    assert is_gpu is False


def test_gpu_kernel_failure_falls_back_to_numpy(monkeypatch):
    from dsa110_continuum.qa import image_metrics

    class BrokenGpu:
        @staticmethod
        def asarray(array):
            return array

        @staticmethod
        def abs(array):
            raise RuntimeError("NVRTC unavailable")

    monkeypatch.setattr(image_metrics, "get_array_module", lambda **kwargs: (BrokenGpu, True))
    array = np.ones((2, 2))

    backend_array, xp, is_gpu = image_metrics._maybe_to_gpu(array, min_elements=1)

    assert backend_array is array
    assert xp is np
    assert is_gpu is False
