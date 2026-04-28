"""Tests for Batch D parallel forced photometry.

Behavior under test:
- Bit-for-bit equivalence between serial (workers=1) and parallel (workers>1).
- Deterministic output ordering matches input coord order.
- Worker-side exceptions propagate to the caller (no silent partial success).
- ``--workers`` and ``--chunk-size`` CLI flags wire through to the helper.
- ``_auto_chunk_size`` heuristic picks ~4 chunks per worker.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

# scripts/ on sys.path so we can import forced_photometry's CLI module
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

# noqa: E402 — sys.path mutation above must precede this import
from dsa110_continuum.photometry.two_stage import (  # noqa: E402
    _auto_chunk_size,
    _two_stage_chunk_worker,
    run_two_stage,
    run_two_stage_parallel,
)

# ─── Fixtures ───────────────────────────────────────────────────────────────


def _make_synthetic_fits(tmp_path: Path, n_sources: int = 30, seed: int = 42) -> tuple[Path, list[tuple[float, float]]]:
    """Build a minimal mosaic with `n_sources` injected point sources.

    Returns (fits_path, list_of_(ra,dec)_for_injected_sources). The synthetic
    image is small (200×200) but has valid WCS + beam keywords so the full
    two-stage path runs end-to-end.
    """
    rng = np.random.default_rng(seed)
    ny, nx = 200, 200
    data = rng.normal(0, 0.001, (ny, nx)).astype(np.float64)

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.crval = [180.0, 30.0]
    w.wcs.crpix = [nx // 2, ny // 2]
    w.wcs.cdelt = [-20.0 / 3600, 20.0 / 3600]  # 20 arcsec/pix

    coords: list[tuple[float, float]] = []
    for k in range(n_sources):
        # Spread sources roughly across the image — keep within central 80% to
        # avoid edge effects.
        xpix = float(rng.uniform(nx * 0.1, nx * 0.9))
        ypix = float(rng.uniform(ny * 0.1, ny * 0.9))
        sky = w.pixel_to_world(xpix, ypix)
        ra, dec = float(sky.ra.deg), float(sky.dec.deg)
        coords.append((ra, dec))
        # Inject a 0.05 Jy/beam point source at the nearest pixel
        ix, iy = int(round(xpix)), int(round(ypix))
        data[iy, ix] += 0.05

    hdr = w.to_header()
    hdr["BMAJ"] = 36.9 / 3600
    hdr["BMIN"] = 25.5 / 3600
    hdr["BPA"] = 0.0

    fpath = tmp_path / "synthetic.fits"
    fits.PrimaryHDU(data=data, header=hdr).writeto(fpath, overwrite=True)
    return fpath, coords


# ─── _auto_chunk_size ───────────────────────────────────────────────────────


def test_auto_chunk_size_serial_returns_full():
    """workers <= 1 means one chunk holds everything."""
    assert _auto_chunk_size(100, workers=1) == 100
    assert _auto_chunk_size(0, workers=4) == 1  # avoid 0-sized chunks


def test_auto_chunk_size_targets_four_chunks_per_worker():
    """Heuristic: chunk_size ≈ n / (4 * workers) so each worker gets ~4 chunks."""
    n, workers = 1000, 4
    chunk_size = _auto_chunk_size(n, workers)
    n_chunks = math.ceil(n / chunk_size)
    # ~4 chunks per worker, allowing rounding
    assert 3 * workers <= n_chunks <= 5 * workers


def test_auto_chunk_size_never_zero():
    """Even with tiny n, chunk_size stays at least 1."""
    assert _auto_chunk_size(1, workers=8) >= 1


# ─── workers <= 1 fallback ─────────────────────────────────────────────────


def test_serial_fallback_short_circuits(tmp_path):
    """workers=1 returns the same object that the underlying serial path returns."""
    fpath, coords = _make_synthetic_fits(tmp_path, n_sources=5)

    serial = run_two_stage(str(fpath), coords, snr_coarse_min=0.0)
    parallel = run_two_stage_parallel(str(fpath), coords, workers=1, snr_coarse_min=0.0)

    assert len(serial[0]) == len(parallel[0]) == len(coords)
    # element-wise: same fields, same values
    for s_res, p_res in zip(serial[0], parallel[0]):
        assert s_res.ra_deg == p_res.ra_deg
        assert s_res.dec_deg == p_res.dec_deg
        # NaN-safe equality on flux
        if math.isnan(s_res.peak_jyb):
            assert math.isnan(p_res.peak_jyb)
        else:
            assert s_res.peak_jyb == p_res.peak_jyb


def test_empty_coords_returns_empty(tmp_path):
    fpath, _ = _make_synthetic_fits(tmp_path, n_sources=1)
    results, augments = run_two_stage_parallel(str(fpath), [], workers=4)
    assert results == []
    assert augments == []


# ─── parallel == serial bit-for-bit ─────────────────────────────────────────


def _result_tuple(r):
    """Hashable comparable view of a ForcedPhotometryResult."""
    return (
        round(r.ra_deg, 8), round(r.dec_deg, 8),
        None if math.isnan(r.peak_jyb) else round(r.peak_jyb, 12),
        None if math.isnan(r.peak_err_jyb) else round(r.peak_err_jyb, 12),
    )


def _aug_tuple(a):
    return (
        round(a.ra_deg, 8), round(a.dec_deg, 8),
        None if math.isnan(a.coarse_peak_jyb) else round(a.coarse_peak_jyb, 12),
        None if math.isnan(a.coarse_snr) else round(a.coarse_snr, 12),
        a.passed_coarse,
    )


def test_parallel_matches_serial_bit_for_bit(tmp_path):
    """workers=4 produces identical results to workers=1 for the same input."""
    fpath, coords = _make_synthetic_fits(tmp_path, n_sources=24, seed=7)

    serial_r, serial_a = run_two_stage_parallel(
        str(fpath), coords, workers=1, snr_coarse_min=0.0,
    )
    parallel_r, parallel_a = run_two_stage_parallel(
        str(fpath), coords, workers=4, snr_coarse_min=0.0, chunk_size=5,
    )

    assert [_result_tuple(r) for r in serial_r] == [_result_tuple(r) for r in parallel_r]
    assert [_aug_tuple(a) for a in serial_a] == [_aug_tuple(a) for a in parallel_a]


def test_parallel_preserves_input_order(tmp_path):
    """Each output index corresponds to the same input coord index."""
    fpath, coords = _make_synthetic_fits(tmp_path, n_sources=20, seed=11)

    results, augments = run_two_stage_parallel(
        str(fpath), coords, workers=3, snr_coarse_min=0.0, chunk_size=4,
    )
    for i, (ra, dec) in enumerate(coords):
        assert augments[i].ra_deg == pytest.approx(ra)
        assert augments[i].dec_deg == pytest.approx(dec)
        # results[i] either has the survivor's coords (within rounding) or is
        # a NaN placeholder still tagged with the input ra/dec
        assert results[i].ra_deg == pytest.approx(ra, abs=1e-3)
        assert results[i].dec_deg == pytest.approx(dec, abs=1e-3)


def test_chunk_boundaries_dont_leak_state(tmp_path):
    """A chunk boundary that splits sources mid-list still produces correct results.

    Regression guard: if a future change introduces shared mutable state in the
    chunk worker, the parallel result for, say, chunk_size=1 (per-source chunks)
    should still equal the serial result.
    """
    fpath, coords = _make_synthetic_fits(tmp_path, n_sources=10, seed=3)

    serial_r, _ = run_two_stage_parallel(str(fpath), coords, workers=1, snr_coarse_min=0.0)
    parallel_r, _ = run_two_stage_parallel(
        str(fpath), coords, workers=2, chunk_size=1, snr_coarse_min=0.0,
    )

    assert [_result_tuple(r) for r in serial_r] == [_result_tuple(r) for r in parallel_r]


# ─── Error propagation ──────────────────────────────────────────────────────


def test_worker_exception_propagates(tmp_path):
    """A bad fits_path causes the worker to raise; exception reaches the caller."""
    bad_path = str(tmp_path / "does-not-exist.fits")
    coords = [(180.0, 30.0), (181.0, 31.0)]
    with pytest.raises((FileNotFoundError, OSError, ValueError, RuntimeError)):
        run_two_stage_parallel(bad_path, coords, workers=2, snr_coarse_min=0.0)


def test_chunk_worker_function_is_picklable():
    """``_two_stage_chunk_worker`` must be top-level so multiprocessing can pickle it."""
    import pickle

    pickled = pickle.dumps(_two_stage_chunk_worker)
    restored = pickle.loads(pickled)
    assert restored is _two_stage_chunk_worker


# ─── CLI wiring ─────────────────────────────────────────────────────────────


def test_cli_flags_wire_through(tmp_path, monkeypatch):
    """--workers and --chunk-size from argparse end up in run_forced_photometry."""
    import forced_photometry as fp

    captured: dict = {}

    def fake_run_forced_photometry(*args, **kwargs):
        captured.update(kwargs)
        return {
            "n_sources": 0, "median_ratio": float("nan"),
            "csv_path": "", "rms_mjy": float("nan"),
            "theoretical_rms_mjy": float("nan"),
            "rms_ratio": float("nan"), "noise_gate": "SKIP",
        }

    monkeypatch.setattr(fp, "run_forced_photometry", fake_run_forced_photometry)
    monkeypatch.setattr(
        sys, "argv",
        [
            "forced_photometry.py",
            "--mosaic", "/tmp/fake.fits",
            "--sim",
            "--workers", "4",
            "--chunk-size", "25",
            "--no-plots",
        ],
    )
    fp.main()

    assert captured.get("workers") == 4
    assert captured.get("chunk_size") == 25


def test_cli_chunk_size_zero_means_auto(tmp_path, monkeypatch):
    """--chunk-size 0 (the default) is forwarded as None so auto-sizing kicks in."""
    import forced_photometry as fp

    captured: dict = {}

    def fake_run_forced_photometry(*args, **kwargs):
        captured.update(kwargs)
        return {
            "n_sources": 0, "median_ratio": float("nan"),
            "csv_path": "", "rms_mjy": float("nan"),
            "theoretical_rms_mjy": float("nan"),
            "rms_ratio": float("nan"), "noise_gate": "SKIP",
        }

    monkeypatch.setattr(fp, "run_forced_photometry", fake_run_forced_photometry)
    monkeypatch.setattr(
        sys, "argv",
        [
            "forced_photometry.py",
            "--mosaic", "/tmp/fake.fits",
            "--sim",
            "--workers", "2",
            # chunk-size omitted → default 0 → forwarded as None
            "--no-plots",
        ],
    )
    fp.main()

    assert captured.get("workers") == 2
    assert captured.get("chunk_size") is None
