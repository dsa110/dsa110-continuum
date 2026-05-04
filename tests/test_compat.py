"""Tests pinning dsa110_continuum._compat shim signatures.

These shims stand in for ``dsa110_contimg.common.utils.gpu_safety`` etc. when
the legacy package is not installed (cloud/CI). The shims must match the
real contimg contracts where call sites unpack or compose return values, or
they cause TypeErrors that the legacy module's actual presence would mask.

Reference signatures (from dsa110_contimg.common.utils.gpu_safety on H17):

    check_gpu_memory_available(required_gb: float, gpu_id: int = 0,
                               config: SafetyConfig | None = None)
        -> tuple[bool, str]
    is_gpu_available() -> bool
    gpu_safe(max_gpu_gb=None, max_system_gb=None, required_gpu_gb=None,
             gpu_id=0, timeout_seconds=None) -> Callable[[F], F]
    memory_safe(max_system_gb=None, required_gb=None,
                timeout_seconds=None) -> Callable[[F], F]
"""

from __future__ import annotations

import inspect

from dsa110_continuum import _compat


class TestGpuSafetyContract:
    def test_check_gpu_memory_available_returns_tuple_bool_str(self):
        result = _compat.check_gpu_memory_available(2.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        ok, reason = result
        assert isinstance(ok, bool)
        assert isinstance(reason, str)
        # When dsa110_contimg is unavailable the shim must report not-OK
        # so callers gate GPU paths off.
        assert ok is False

    def test_check_gpu_memory_available_accepts_real_contimg_kwargs(self):
        # The contimg signature is (required_gb, gpu_id=0, config=None);
        # the shim must accept all three call shapes without TypeError.
        ok, _ = _compat.check_gpu_memory_available(2.0)
        ok, _ = _compat.check_gpu_memory_available(2.0, 0)
        ok, _ = _compat.check_gpu_memory_available(2.0, gpu_id=0, config=None)
        assert ok is False

    def test_check_gpu_memory_available_unpacks_at_call_sites(self):
        # Callers in applycal.py and imaging/worker.py do
        # ``gpu_ok, gpu_reason = check_gpu_memory_available(2.0)``.
        # If the shim returns a scalar, this raises TypeError.
        gpu_ok, gpu_reason = _compat.check_gpu_memory_available(2.0)
        assert gpu_ok is False
        assert gpu_reason  # non-empty diagnostic string

    def test_is_gpu_available_returns_bool(self):
        assert _compat.is_gpu_available() is False

    def test_check_system_memory_available_matches_contimg_shape(self):
        # When the contimg helper exists it returns ``tuple[bool, str]``.
        # The shim must too — same unpack contract.
        ok, reason = _compat.check_system_memory_available(2.0)
        assert ok is False
        assert isinstance(reason, str)


class TestDecoratorStubs:
    def test_memory_safe_accepts_contimg_kwargs(self):
        @_compat.memory_safe(max_system_gb=4.0, required_gb=1.0, timeout_seconds=60)
        def f(x):
            return x + 1

        assert f(2) == 3

    def test_gpu_safe_accepts_contimg_kwargs(self):
        @_compat.gpu_safe(
            max_gpu_gb=2.0,
            max_system_gb=4.0,
            required_gpu_gb=1.0,
            gpu_id=0,
            timeout_seconds=60,
        )
        def f(x):
            return x * 2

        assert f(3) == 6

    def test_gpu_safe_works_as_plain_decorator(self):
        @_compat.gpu_safe
        def f(x):
            return x

        assert f(5) == 5

    def test_timed_works_with_and_without_label(self):
        @_compat.timed
        def a(x):
            return x

        @_compat.timed("label")
        def b(x):
            return x + 1

        assert a(1) == 1
        assert b(1) == 2


class TestAntennaStubs:
    def test_get_outrigger_and_core_partition_correctly(self):
        outriggers = _compat.get_outrigger_antenna_ids()
        cores = _compat.get_core_antenna_ids()
        assert set(outriggers).isdisjoint(set(cores))
        # The DSA-110 instrument has 117 array elements (per CLAUDE.md);
        # the union of stubs should match that or a subset, never overlap.
        assert len(set(outriggers) | set(cores)) == len(outriggers) + len(cores)


class TestSignatureParityWithCallers:
    """Pins shim signatures so a future _compat refactor cannot regress them.

    These tests are static: they introspect the shim signatures rather than
    calling them, so they fail loudly if a refactor renames a parameter or
    drops one that real call sites pass.
    """

    def test_check_gpu_memory_available_signature(self):
        sig = inspect.signature(_compat.check_gpu_memory_available)
        params = list(sig.parameters)
        # First positional parameter is the required-memory amount.
        assert params[0] in {"required_gb", "required_bytes"}
        # Must accept gpu_id (used by contimg call sites).
        assert any(p in sig.parameters for p in ("gpu_id",))
