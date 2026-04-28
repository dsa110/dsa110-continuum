"""Two-stage forced photometry: coarse peak-in-box -> Condon fine pass."""
from __future__ import annotations

import logging
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from astropy.stats import mad_std
from astropy.wcs import WCS

from dsa110_continuum.photometry.simple_peak import measure_peak_box
from dsa110_continuum.photometry.forced import ForcedPhotometryResult, measure_many

_log = logging.getLogger(__name__)


@dataclass
class CoarseAugment:
    """Coarse-pass metadata attached alongside a ForcedPhotometryResult."""
    ra_deg: float
    dec_deg: float
    coarse_peak_jyb: float
    coarse_snr: float
    passed_coarse: bool


def beam_correction_factor(fits_path: str) -> float:
    """Return beam_area_sr / pixel_area_sr from a FITS header.
    Pixel area is |CDELT1| x |CDELT2| in steradians.
    For an unresolved point source: peak_jyb = S_total_Jy / correction_factor.
    So: S_total_Jy approximately peak_jyb * correction_factor.
    Returns 1.0 if beam keywords are absent or zero.
    """
    hdr = fits.getheader(fits_path)
    bmaj_deg = hdr.get("BMAJ")
    bmin_deg = hdr.get("BMIN")
    cdelt1_deg = abs(hdr.get("CDELT1", 0.0))
    cdelt2_deg = abs(hdr.get("CDELT2", 0.0))
    if bmaj_deg is None or bmin_deg is None or bmaj_deg <= 0.0 or bmin_deg <= 0.0 or cdelt1_deg == 0.0 or cdelt2_deg == 0.0:
        return 1.0
    bmaj_rad = math.radians(bmaj_deg)
    bmin_rad = math.radians(bmin_deg)
    beam_area_sr = (math.pi / (4.0 * math.log(2.0))) * bmaj_rad * bmin_rad
    pixel_area_sr = math.radians(cdelt1_deg) * math.radians(cdelt2_deg)
    return beam_area_sr / pixel_area_sr


def run_coarse_pass(
    fits_path: str,
    coords: list[tuple[float, float]],
    *,
    box_pix: int = 5,
    global_rms: float | None = None,
    snr_coarse_min: float = 3.0,
) -> list[CoarseAugment]:
    """Run peak-in-box measurement at each sky position.

    Parameters
    ----------
    fits_path : str
        Path to mosaic FITS file (Jy/beam).
    coords : list of (ra_deg, dec_deg)
        Source positions to measure.
    box_pix : int
        Half-width of the search box in pixels (default 5, giving 11x11 box).
    global_rms : float or None
        Noise estimate in Jy/beam.  If None, estimated from image MAD.
    snr_coarse_min : float
        Sources with coarse_snr >= this value have passed_coarse=True.

    Returns
    -------
    list[CoarseAugment]
        One entry per input coordinate, preserving order.
    """
    with fits.open(fits_path) as hdul:
        data = np.squeeze(np.asarray(hdul[0].data, dtype=float))
        wcs = WCS(hdul[0].header).celestial

    rms = global_rms
    if rms is None:
        finite = data[np.isfinite(data)]
        rms = float(mad_std(finite)) if finite.size > 0 else float("nan")

    if not (np.isfinite(rms) and rms > 0):
        return [
            CoarseAugment(
                ra_deg=ra, dec_deg=dec,
                coarse_peak_jyb=float("nan"),
                coarse_snr=float("nan"),
                passed_coarse=False,
            )
            for ra, dec in coords
        ]

    results = []
    for ra, dec in coords:
        peak, snr, _, _ = measure_peak_box(data, wcs, ra, dec, box_pix=box_pix, rms=rms)
        results.append(CoarseAugment(
            ra_deg=ra,
            dec_deg=dec,
            coarse_peak_jyb=peak,
            coarse_snr=snr,
            passed_coarse=bool(np.isfinite(snr) and snr >= snr_coarse_min),
        ))
    return results


def _nan_result(ra: float, dec: float, box_pix: int) -> "ForcedPhotometryResult":
    """Return a NaN-filled ForcedPhotometryResult placeholder."""
    return ForcedPhotometryResult(
        ra_deg=ra, dec_deg=dec,
        peak_jyb=float("nan"), peak_err_jyb=float("nan"),
        pix_x=float("nan"), pix_y=float("nan"),
        box_size_pix=box_pix,
    )


def run_two_stage(
    fits_path: str,
    coords: list[tuple[float, float]],
    *,
    snr_coarse_min: float = 3.0,
    box_pix: int = 5,
    global_rms: float | None = None,
    use_cluster_fitting: bool = False,
    noise_map_path: str | None = None,
) -> tuple[list[ForcedPhotometryResult], list[CoarseAugment]]:
    """Coarse peak-in-box pass, then Condon fine pass on survivors.

    Parameters
    ----------
    fits_path : str
        Path to mosaic FITS file.
    coords : list of (ra_deg, dec_deg)
        Source positions to measure.
    snr_coarse_min : float
        Coarse SNR gate threshold (default 3.0).
    box_pix : int
        Half-width of coarse measurement box in pixels (default 5).
    global_rms : float or None
        Noise estimate in Jy/beam.  If None, estimated from image MAD.
    use_cluster_fitting : bool
        Passed through to measure_many (default False).
    noise_map_path : str or None
        Optional noise map FITS path for fine pass.

    Returns
    -------
    results : list[ForcedPhotometryResult]
        One result per input coord, in order. NaN placeholder for coarse failures.
    augments : list[CoarseAugment]
        Coarse-pass metadata for each coord, in order.
    """
    if not coords:
        return [], []

    augments = run_coarse_pass(
        fits_path, coords,
        box_pix=box_pix,
        global_rms=global_rms,
        snr_coarse_min=snr_coarse_min,
    )

    # Collect survivors preserving their original indices
    survivor_indices = [i for i, aug in enumerate(augments) if aug.passed_coarse]
    survivor_coords = [coords[i] for i in survivor_indices]

    if survivor_coords:
        fine_results = measure_many(
            fits_path,
            survivor_coords,
            use_cluster_fitting=use_cluster_fitting,
            noise_map_path=noise_map_path,
        )
    else:
        fine_results = []

    # Map fine results back by original index
    fine_map = {survivor_indices[j]: fine_results[j] for j in range(len(fine_results))}
    assert len(fine_results) == len(survivor_coords), (
        f"measure_many returned {len(fine_results)} results for {len(survivor_coords)} survivors"
    )

    ordered_results = [
        fine_map.get(i, _nan_result(aug.ra_deg, aug.dec_deg, box_pix))
        for i, aug in enumerate(augments)
    ]

    return ordered_results, augments


# ─── Parallel orchestration ──────────────────────────────────────────────────
#
# The science kernels stay in run_two_stage. The functions below are a pure
# orchestration layer: they split the input coords into contiguous chunks,
# dispatch each chunk to a worker process, and re-concatenate results in the
# original order. workers<=1 falls through to the serial path with zero
# overhead.

def _compute_global_rms(fits_path: str) -> float:
    """Compute the image-wide MAD-based RMS once for all workers.

    Done in the parent so each worker doesn't redundantly load + scan the FITS
    file, and so the noise normalisation is bit-identical across chunks.
    Returns ``nan`` if the image has no finite pixels — workers will then
    short-circuit to NaN coarse results consistently.
    """
    with fits.open(fits_path) as hdul:
        data = np.squeeze(np.asarray(hdul[0].data, dtype=float))
    finite = data[np.isfinite(data)]
    return float(mad_std(finite)) if finite.size > 0 else float("nan")


def _two_stage_chunk_worker(
    args: tuple[int, str, list[tuple[float, float]], dict],
) -> tuple[int, list[ForcedPhotometryResult], list[CoarseAugment]]:
    """Run ``run_two_stage`` on a single chunk and tag the result with its index.

    Top-level function (not a closure) so it is picklable by multiprocessing.
    Returns ``(chunk_idx, results, augments)``; the index lets the parent
    re-assemble out-of-order completions deterministically.
    """
    chunk_idx, fits_path, coords_chunk, kwargs = args
    results, augments = run_two_stage(fits_path, coords_chunk, **kwargs)
    return chunk_idx, results, augments


def _auto_chunk_size(n_coords: int, workers: int) -> int:
    """Pick a chunk size targeting ~4 chunks per worker for load balance.

    Four chunks per worker balances coarse-pass + fine-pass cost variability
    (some chunks have many survivors, others few) without exploding the
    per-chunk overhead. Always at least 1.
    """
    if workers <= 1 or n_coords <= 0:
        return max(n_coords, 1)
    return max(1, n_coords // (4 * workers))


def run_two_stage_parallel(
    fits_path: str,
    coords: list[tuple[float, float]],
    *,
    workers: int = 1,
    chunk_size: int | None = None,
    snr_coarse_min: float = 3.0,
    box_pix: int = 5,
    global_rms: float | None = None,
    use_cluster_fitting: bool = False,
    noise_map_path: str | None = None,
) -> tuple[list[ForcedPhotometryResult], list[CoarseAugment]]:
    """Parallel deterministic wrapper around :func:`run_two_stage`.

    Splits ``coords`` into contiguous chunks (preserving order) and runs each
    chunk in a worker process. Results are concatenated in the original input
    order, so output is bit-for-bit identical to a serial run with the same
    kwargs (assuming the underlying photometry kernels are deterministic).

    Parameters
    ----------
    workers : int
        Number of worker processes. ``workers <= 1`` short-circuits to the
        existing serial :func:`run_two_stage` with zero pickling/process
        overhead.
    chunk_size : int or None
        If ``None`` or ``<= 0``, an auto size of ``max(1, len(coords) //
        (4 * workers))`` is used.
    Other kwargs are forwarded to :func:`run_two_stage` unchanged.

    Failure policy: any exception in a worker is re-raised by ``map``; the
    caller sees a normal traceback. No silent partial success.
    """
    if not coords:
        return [], []

    if workers <= 1:
        return run_two_stage(
            fits_path, coords,
            snr_coarse_min=snr_coarse_min,
            box_pix=box_pix,
            global_rms=global_rms,
            use_cluster_fitting=use_cluster_fitting,
            noise_map_path=noise_map_path,
        )

    # Compute RMS once for deterministic noise normalisation across chunks.
    if global_rms is None:
        global_rms = _compute_global_rms(fits_path)

    if chunk_size is None or chunk_size <= 0:
        chunk_size = _auto_chunk_size(len(coords), workers)

    chunks: list[list[tuple[float, float]]] = [
        coords[i : i + chunk_size]
        for i in range(0, len(coords), chunk_size)
    ]
    kwargs = {
        "snr_coarse_min": snr_coarse_min,
        "box_pix": box_pix,
        "global_rms": global_rms,
        "use_cluster_fitting": use_cluster_fitting,
        "noise_map_path": noise_map_path,
    }
    payloads = [
        (idx, fits_path, chunk, kwargs) for idx, chunk in enumerate(chunks)
    ]
    _log.info(
        "Forced photometry parallel: %d coords → %d chunks × ≤%d, %d workers",
        len(coords), len(chunks), chunk_size, workers,
    )

    results_by_chunk: list[list[ForcedPhotometryResult] | None] = [None] * len(chunks)
    augments_by_chunk: list[list[CoarseAugment] | None] = [None] * len(chunks)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        # map() preserves submission order even though tasks may complete
        # out of order. We additionally re-index by chunk_idx so any future
        # switch to as_completed() doesn't break determinism.
        for idx, results, augments in ex.map(_two_stage_chunk_worker, payloads):
            results_by_chunk[idx] = results
            augments_by_chunk[idx] = augments

    flat_results: list[ForcedPhotometryResult] = []
    flat_augments: list[CoarseAugment] = []
    for r, a in zip(results_by_chunk, augments_by_chunk):
        assert r is not None and a is not None, "chunk result missing"
        flat_results.extend(r)
        flat_augments.extend(a)

    assert len(flat_results) == len(coords), (
        f"parallel two-stage returned {len(flat_results)} results "
        f"for {len(coords)} input coords"
    )
    return flat_results, flat_augments
