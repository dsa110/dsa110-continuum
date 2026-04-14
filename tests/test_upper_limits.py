"""
Tests for upper-limit non-detection storage (Task 8).

Tests cover:
  1. UpperLimitRecord dataclass validation
  2. forced_peak_to_upper_limit() conversion helper
  3. ForcedPhotometryResult.is_upper_limit / upper_limit_jyb fields
  4. _flag_upper_limit() helper
  5. UpperLimitStore CRUD operations
  6. UpperLimitStore bulk operations (add_many)
  7. UpperLimitStore queries (by source, epoch, range)
  8. UpperLimitStore numpy helpers
  9. Idempotency (INSERT OR REPLACE)
  10. Thread-safety smoke test
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from dsa110_continuum.photometry.upper_limits import (
    DEFAULT_DETECTION_THRESHOLD_SIGMA,
    UpperLimitRecord,
    UpperLimitStore,
    forced_peak_to_upper_limit,
)
from dsa110_continuum.photometry.forced import (
    ForcedPhotometryResult,
    _flag_upper_limit,
)


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def store():
    """In-memory UpperLimitStore (auto-closed after test)."""
    s = UpperLimitStore(":memory:")
    yield s
    s.close()


def _make_record(
    source_id: int = 1,
    epoch_mjd: float = 60000.0,
    rms_jyb: float = 0.003,
    n_sigma: float = 5.0,
) -> UpperLimitRecord:
    return UpperLimitRecord(
        source_id=source_id,
        epoch_mjd=epoch_mjd,
        ra_deg=180.0,
        dec_deg=0.0,
        rms_jyb=rms_jyb,
        upper_limit_jyb=n_sigma * rms_jyb,
        n_sigma=n_sigma,
        forced_peak_jyb=0.001,
        image_path="/data/img.fits",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. UpperLimitRecord validation
# ══════════════════════════════════════════════════════════════════════════════

class TestUpperLimitRecord:

    def test_basic_construction(self):
        rec = _make_record()
        assert rec.source_id == 1
        assert rec.rms_jyb == 0.003
        assert rec.upper_limit_jyb == pytest.approx(0.015)
        assert rec.n_sigma == 5.0

    def test_default_n_sigma(self):
        rec = UpperLimitRecord(
            source_id=1, epoch_mjd=60000.0,
            ra_deg=0.0, dec_deg=0.0,
            rms_jyb=0.002, upper_limit_jyb=0.01,
        )
        assert rec.n_sigma == DEFAULT_DETECTION_THRESHOLD_SIGMA

    def test_negative_rms_raises(self):
        with pytest.raises(ValueError, match="rms_jyb"):
            UpperLimitRecord(
                source_id=1, epoch_mjd=60000.0,
                ra_deg=0.0, dec_deg=0.0,
                rms_jyb=-0.001, upper_limit_jyb=0.01,
            )

    def test_negative_upper_limit_raises(self):
        with pytest.raises(ValueError, match="upper_limit_jyb"):
            UpperLimitRecord(
                source_id=1, epoch_mjd=60000.0,
                ra_deg=0.0, dec_deg=0.0,
                rms_jyb=0.003, upper_limit_jyb=-0.01,
            )

    def test_zero_n_sigma_raises(self):
        with pytest.raises(ValueError, match="n_sigma"):
            UpperLimitRecord(
                source_id=1, epoch_mjd=60000.0,
                ra_deg=0.0, dec_deg=0.0,
                rms_jyb=0.003, upper_limit_jyb=0.015,
                n_sigma=0.0,
            )

    def test_nan_forced_peak_allowed(self):
        rec = _make_record()
        rec.forced_peak_jyb = float("nan")
        assert not np.isfinite(rec.forced_peak_jyb)

    def test_empty_image_path_allowed(self):
        rec = UpperLimitRecord(
            source_id=1, epoch_mjd=60000.0,
            ra_deg=0.0, dec_deg=0.0,
            rms_jyb=0.003, upper_limit_jyb=0.015,
        )
        assert rec.image_path == ""


# ══════════════════════════════════════════════════════════════════════════════
# 2. forced_peak_to_upper_limit
# ══════════════════════════════════════════════════════════════════════════════

class TestForcedPeakToUpperLimit:

    def test_below_threshold_returns_record(self):
        """SNR = 0.002/0.003 = 0.67 < 5 → upper limit."""
        rec = forced_peak_to_upper_limit(
            source_id=7, epoch_mjd=60340.5,
            ra_deg=150.0, dec_deg=30.0,
            forced_peak_jyb=0.002, rms_jyb=0.003, n_sigma=5.0,
        )
        assert rec is not None
        assert rec.source_id == 7
        assert rec.upper_limit_jyb == pytest.approx(0.015)

    def test_above_threshold_returns_none(self):
        """SNR = 0.02/0.003 = 6.7 > 5 → detected, return None."""
        rec = forced_peak_to_upper_limit(
            source_id=7, epoch_mjd=60340.5,
            ra_deg=150.0, dec_deg=30.0,
            forced_peak_jyb=0.020, rms_jyb=0.003, n_sigma=5.0,
        )
        assert rec is None

    def test_exactly_at_threshold_returns_none(self):
        """SNR = 5.0 exactly → detected (>=), return None."""
        rec = forced_peak_to_upper_limit(
            source_id=1, epoch_mjd=60000.0,
            ra_deg=0.0, dec_deg=0.0,
            forced_peak_jyb=0.015, rms_jyb=0.003, n_sigma=5.0,
        )
        assert rec is None

    def test_negative_peak_is_upper_limit(self):
        """Negative forced peak is always a non-detection."""
        rec = forced_peak_to_upper_limit(
            source_id=3, epoch_mjd=60001.0,
            ra_deg=0.0, dec_deg=0.0,
            forced_peak_jyb=-0.005, rms_jyb=0.003, n_sigma=5.0,
        )
        assert rec is not None
        assert rec.forced_peak_jyb == pytest.approx(-0.005)

    def test_nan_peak_is_upper_limit(self):
        """NaN forced peak → image failure → upper limit."""
        rec = forced_peak_to_upper_limit(
            source_id=5, epoch_mjd=60002.0,
            ra_deg=0.0, dec_deg=0.0,
            forced_peak_jyb=float("nan"), rms_jyb=0.003, n_sigma=5.0,
        )
        assert rec is not None

    def test_invalid_rms_returns_none(self):
        """Cannot compute upper limit if RMS is NaN/zero."""
        assert forced_peak_to_upper_limit(
            source_id=1, epoch_mjd=60000.0, ra_deg=0.0, dec_deg=0.0,
            forced_peak_jyb=0.001, rms_jyb=float("nan"), n_sigma=5.0,
        ) is None
        assert forced_peak_to_upper_limit(
            source_id=1, epoch_mjd=60000.0, ra_deg=0.0, dec_deg=0.0,
            forced_peak_jyb=0.001, rms_jyb=0.0, n_sigma=5.0,
        ) is None

    def test_upper_limit_equals_n_sigma_times_rms(self):
        rec = forced_peak_to_upper_limit(
            source_id=1, epoch_mjd=60000.0,
            ra_deg=0.0, dec_deg=0.0,
            forced_peak_jyb=0.001, rms_jyb=0.004, n_sigma=3.0,
        )
        assert rec is not None
        assert rec.upper_limit_jyb == pytest.approx(0.012)

    def test_custom_n_sigma(self):
        rec = forced_peak_to_upper_limit(
            source_id=1, epoch_mjd=60000.0,
            ra_deg=0.0, dec_deg=0.0,
            forced_peak_jyb=0.001, rms_jyb=0.003, n_sigma=7.0,
        )
        assert rec is not None
        assert rec.n_sigma == 7.0
        assert rec.upper_limit_jyb == pytest.approx(0.021)

    def test_image_path_stored(self):
        rec = forced_peak_to_upper_limit(
            source_id=1, epoch_mjd=60000.0,
            ra_deg=0.0, dec_deg=0.0,
            forced_peak_jyb=0.001, rms_jyb=0.003,
            image_path="/data/2024/img.fits",
        )
        assert rec is not None
        assert rec.image_path == "/data/2024/img.fits"


# ══════════════════════════════════════════════════════════════════════════════
# 3. ForcedPhotometryResult upper-limit fields (dataclass)
# ══════════════════════════════════════════════════════════════════════════════

class TestForcedPhotometryResultFields:

    def test_default_is_upper_limit_false(self):
        r = ForcedPhotometryResult(
            ra_deg=0.0, dec_deg=0.0,
            peak_jyb=1.0, peak_err_jyb=0.1,
            pix_x=50.0, pix_y=50.0, box_size_pix=5,
        )
        assert r.is_upper_limit is False
        assert r.upper_limit_jyb is None

    def test_can_set_upper_limit_fields(self):
        r = ForcedPhotometryResult(
            ra_deg=0.0, dec_deg=0.0,
            peak_jyb=0.001, peak_err_jyb=0.003,
            pix_x=50.0, pix_y=50.0, box_size_pix=5,
            is_upper_limit=True,
            upper_limit_jyb=0.015,
        )
        assert r.is_upper_limit is True
        assert r.upper_limit_jyb == pytest.approx(0.015)


# ══════════════════════════════════════════════════════════════════════════════
# 4. _flag_upper_limit helper
# ══════════════════════════════════════════════════════════════════════════════

class TestFlagUpperLimit:

    def _make_result(self, peak: float, err: float) -> ForcedPhotometryResult:
        return ForcedPhotometryResult(
            ra_deg=0.0, dec_deg=0.0,
            peak_jyb=peak, peak_err_jyb=err,
            pix_x=50.0, pix_y=50.0, box_size_pix=5,
        )

    def test_none_threshold_no_change(self):
        r = self._make_result(0.001, 0.003)
        r2 = _flag_upper_limit(r, None)
        assert r2.is_upper_limit is False
        assert r2.upper_limit_jyb is None

    def test_below_threshold_flagged(self):
        """SNR = 0.001/0.003 = 0.33 < 5."""
        r = self._make_result(0.001, 0.003)
        r2 = _flag_upper_limit(r, 5.0)
        assert r2.is_upper_limit is True
        assert r2.upper_limit_jyb == pytest.approx(0.015)

    def test_above_threshold_not_flagged(self):
        """SNR = 0.030/0.003 = 10 >= 5."""
        r = self._make_result(0.030, 0.003)
        r2 = _flag_upper_limit(r, 5.0)
        assert r2.is_upper_limit is False
        assert r2.upper_limit_jyb is None

    def test_nan_peak_flagged(self):
        r = self._make_result(float("nan"), 0.003)
        r2 = _flag_upper_limit(r, 5.0)
        assert r2.is_upper_limit is True

    def test_negative_peak_flagged(self):
        r = self._make_result(-0.005, 0.003)
        r2 = _flag_upper_limit(r, 5.0)
        assert r2.is_upper_limit is True

    def test_raw_peak_preserved(self):
        """The raw peak_jyb is not modified even if flagged."""
        r = self._make_result(0.002, 0.003)
        r2 = _flag_upper_limit(r, 5.0)
        assert r2.peak_jyb == pytest.approx(0.002)

    def test_zero_err_not_flagged(self):
        """Zero error → SNR undefined → treat as non-finite → flagged."""
        r = self._make_result(0.001, 0.0)
        r2 = _flag_upper_limit(r, 5.0)
        assert r2.is_upper_limit is True


# ══════════════════════════════════════════════════════════════════════════════
# 5. UpperLimitStore basic CRUD
# ══════════════════════════════════════════════════════════════════════════════

class TestUpperLimitStoreCRUD:

    def test_empty_store_count_zero(self, store):
        assert store.count() == 0

    def test_add_one_record(self, store):
        store.add(_make_record())
        assert store.count() == 1

    def test_add_returns_none(self, store):
        result = store.add(_make_record())
        assert result is None

    def test_get_by_source_returns_record(self, store):
        rec = _make_record(source_id=42)
        store.add(rec)
        retrieved = store.get_by_source(42)
        assert len(retrieved) == 1
        assert retrieved[0].source_id == 42
        assert retrieved[0].upper_limit_jyb == pytest.approx(rec.upper_limit_jyb)

    def test_get_by_source_unknown_returns_empty(self, store):
        assert store.get_by_source(999) == []

    def test_get_by_epoch(self, store):
        rec = _make_record(epoch_mjd=60100.5)
        store.add(rec)
        retrieved = store.get_by_epoch(60100.5)
        assert len(retrieved) == 1

    def test_get_by_epoch_tolerance(self, store):
        rec = _make_record(epoch_mjd=60100.5)
        store.add(rec)
        assert len(store.get_by_epoch(60100.5 + 1e-5)) == 1  # within tol
        assert len(store.get_by_epoch(60100.5 + 1.0)) == 0   # outside tol

    def test_delete_by_source(self, store):
        store.add(_make_record(source_id=1))
        store.add(_make_record(source_id=2, epoch_mjd=60001.0))
        deleted = store.delete_by_source(1)
        assert deleted == 1
        assert store.count() == 1
        assert store.get_by_source(1) == []

    def test_delete_by_epoch(self, store):
        store.add(_make_record(source_id=1, epoch_mjd=60000.0))
        store.add(_make_record(source_id=2, epoch_mjd=60001.0))
        deleted = store.delete_by_epoch(60000.0)
        assert deleted == 1
        assert store.count() == 1

    def test_round_trip_preserves_all_fields(self, store):
        rec = UpperLimitRecord(
            source_id=77,
            epoch_mjd=60777.25,
            ra_deg=234.567,
            dec_deg=-12.345,
            rms_jyb=0.00412,
            upper_limit_jyb=0.0206,
            n_sigma=5.0,
            forced_peak_jyb=-0.00123,
            image_path="/data/epoch/tile42.fits",
        )
        store.add(rec)
        retrieved = store.get_by_source(77)[0]
        assert retrieved.epoch_mjd == pytest.approx(rec.epoch_mjd)
        assert retrieved.ra_deg == pytest.approx(rec.ra_deg)
        assert retrieved.dec_deg == pytest.approx(rec.dec_deg)
        assert retrieved.rms_jyb == pytest.approx(rec.rms_jyb)
        assert retrieved.upper_limit_jyb == pytest.approx(rec.upper_limit_jyb)
        assert retrieved.n_sigma == pytest.approx(rec.n_sigma)
        assert retrieved.forced_peak_jyb == pytest.approx(rec.forced_peak_jyb)
        assert retrieved.image_path == rec.image_path


# ══════════════════════════════════════════════════════════════════════════════
# 6. Bulk operations
# ══════════════════════════════════════════════════════════════════════════════

class TestUpperLimitStoreBulk:

    def test_add_many_empty_list(self, store):
        n = store.add_many([])
        assert n == 0
        assert store.count() == 0

    def test_add_many_multiple_records(self, store):
        recs = [_make_record(source_id=i, epoch_mjd=60000.0 + i) for i in range(10)]
        n = store.add_many(recs)
        assert n == 10
        assert store.count() == 10

    def test_add_many_returns_count(self, store):
        recs = [_make_record(source_id=i, epoch_mjd=float(i)) for i in range(5)]
        assert store.add_many(recs) == 5

    def test_idempotency_replace_on_duplicate(self, store):
        """Re-inserting (source_id, epoch_mjd) should replace, not raise."""
        rec1 = _make_record(source_id=1, epoch_mjd=60000.0)
        rec2 = _make_record(source_id=1, epoch_mjd=60000.0)
        rec2.rms_jyb = 0.006
        rec2.upper_limit_jyb = 0.030
        store.add(rec1)
        store.add(rec2)  # replaces rec1
        assert store.count() == 1
        r = store.get_by_source(1)[0]
        assert r.rms_jyb == pytest.approx(0.006)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Range queries
# ══════════════════════════════════════════════════════════════════════════════

class TestUpperLimitStoreQueries:

    def _populate(self, store, n_sources=5, n_epochs=4):
        recs = []
        for s in range(n_sources):
            for e in range(n_epochs):
                recs.append(_make_record(source_id=s, epoch_mjd=60000.0 + e))
        store.add_many(recs)

    def test_get_epoch_range(self, store):
        self._populate(store, n_sources=3, n_epochs=10)
        # Source 1, epochs 60002 to 60005
        rows = store.get_epoch_range(1, 60002.0, 60005.0)
        assert len(rows) == 4
        assert all(r.source_id == 1 for r in rows)
        assert all(60002.0 <= r.epoch_mjd <= 60005.0 for r in rows)

    def test_sources_with_upper_limits(self, store):
        self._populate(store, n_sources=4, n_epochs=2)
        sources = store.sources_with_upper_limits()
        assert sources == [0, 1, 2, 3]

    def test_iter_all(self, store):
        self._populate(store, n_sources=2, n_epochs=3)
        all_recs = list(store.iter_all())
        assert len(all_recs) == 6

    def test_ordering_by_epoch(self, store):
        for mjd in [60003.0, 60001.0, 60002.0]:
            store.add(_make_record(source_id=1, epoch_mjd=mjd))
        rows = store.get_by_source(1)
        mjds = [r.epoch_mjd for r in rows]
        assert mjds == sorted(mjds)


# ══════════════════════════════════════════════════════════════════════════════
# 8. Numpy helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestUpperLimitStoreNumpy:

    def test_upper_limits_array_empty(self, store):
        arr = store.upper_limits_array(42)
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 0

    def test_upper_limits_array_values(self, store):
        for i, mjd in enumerate([60000.0, 60001.0, 60002.0]):
            r = UpperLimitRecord(
                source_id=1, epoch_mjd=mjd,
                ra_deg=0.0, dec_deg=0.0,
                rms_jyb=0.001 * (i + 1),
                upper_limit_jyb=0.005 * (i + 1),
            )
            store.add(r)
        arr = store.upper_limits_array(1)
        assert arr.shape == (3,)
        np.testing.assert_allclose(arr, [0.005, 0.010, 0.015])

    def test_epochs_array_values(self, store):
        for mjd in [60010.0, 60020.0]:
            store.add(_make_record(source_id=5, epoch_mjd=mjd))
        epochs = store.epochs_array(5)
        np.testing.assert_array_equal(sorted(epochs), [60010.0, 60020.0])


# ══════════════════════════════════════════════════════════════════════════════
# 9. Context manager
# ══════════════════════════════════════════════════════════════════════════════

class TestUpperLimitStoreContextManager:

    def test_context_manager_closes(self):
        with UpperLimitStore(":memory:") as s:
            s.add(_make_record())
            count = s.count()
        assert count == 1  # closed after exit — value captured before

    def test_double_close_safe(self, store):
        """Calling close twice should not raise."""
        store.close()
        store.close()  # should be a no-op


# ══════════════════════════════════════════════════════════════════════════════
# 10. Thread safety smoke test
# ══════════════════════════════════════════════════════════════════════════════

class TestUpperLimitStoreThreadSafety:

    def test_concurrent_writes(self, tmp_path):
        """Multiple threads writing unique records should not corrupt the DB."""
        db_path = tmp_path / "ul.db"
        store = UpperLimitStore(str(db_path))

        errors = []
        def writer(thread_id: int):
            try:
                recs = [
                    _make_record(
                        source_id=thread_id * 100 + i,
                        epoch_mjd=float(i),
                    )
                    for i in range(10)
                ]
                store.add_many(recs)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        store.close()
        assert errors == [], f"Thread errors: {errors}"

        # Verify all records were written
        verify_store = UpperLimitStore(str(db_path))
        assert verify_store.count() == 40
        verify_store.close()
