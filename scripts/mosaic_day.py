#!/opt/miniforge/envs/casa6/bin/python
"""
Process DSA-110 drift observations and produce science mosaics.

Default (science) mode builds sliding-window mosaics (~1-hour products)
from time-ordered tiles, not one monolithic day coadd.  Use ``--full-day``
for the legacy all-day diagnostic mosaic.

Steps per MS:
  1. Phaseshift to median meridian (skip if *_meridian.ms already exists)
  2. Apply BP + G calibration (skip if CORRECTED_DATA ratio > 5)
  3. Image with WSClean

Then:
  4. Window tiles (default 12 tiles / stride 6) — or all-day with --full-day
  5. Reproject each window's tiles onto a common WCS
  6. Coadd with noise (1/σ²) weighting
  7. Write per-window mosaic FITS: {date}_w{NN}_mosaic.fits
"""

import argparse
import dataclasses
import glob
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

# casa_tables (wraps casatools) is imported lazily inside functions that
# need it so that the rest of this module remains importable in environments
# where casatools is not installed (e.g. CI, unit tests with mocks).

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Pipeline modules ────────────────────────────────────────────────────────
from dsa110_continuum.calibration.applycal import apply_to_target
from dsa110_continuum.calibration.runner import phaseshift_ms
from dsa110_continuum.imaging.cli_imaging import image_ms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Cal-path resolution (inlined to avoid dependency on untracked ensure.py) ─


def _resolve_cal_table_paths(ms_dir: str, cal_date: str) -> tuple[str, str]:
    """Find bandpass and gain table paths for a cal date.

    Globs for ``{ms_dir}/{cal_date}T*_0~23.{b,g}``; falls back to the
    legacy ``T22:26:05`` convention when no glob match is found.
    """
    bp_matches = sorted(glob.glob(os.path.join(ms_dir, f"{cal_date}T*_0~23.b")))
    g_matches = sorted(glob.glob(os.path.join(ms_dir, f"{cal_date}T*_0~23.g")))
    bp = bp_matches[0] if bp_matches else os.path.join(ms_dir, f"{cal_date}T22:26:05_0~23.b")
    g = g_matches[0] if g_matches else os.path.join(ms_dir, f"{cal_date}T22:26:05_0~23.g")
    return bp, g


# ── TileConfig: explicit pipeline configuration ─────────────────────────────


@dataclasses.dataclass(frozen=True)
class TileConfig:
    """Immutable configuration for a single pipeline run.

    Replaces the former mutable module-level globals (DATE, IMAGE_DIR, etc.).
    Passed explicitly to process_ms() and related functions so that pipeline
    steps are pure functions of their inputs — no hidden global state.
    """

    date: str
    ms_dir: str
    image_dir: str
    mosaic_out: str
    products_dir: str
    bp_table: str
    g_table: str

    @staticmethod
    def build(
        date: str,
        cal_date: str | None = None,
        ms_dir: str | None = None,
        image_dir: str | None = None,
        products_dir: str | None = None,
    ) -> "TileConfig":
        """Construct a TileConfig with standard DSA-110 path conventions.

        Parameters
        ----------
        date
            Observation date (YYYY-MM-DD).
        cal_date
            Date of calibration tables.  Defaults to *date*.
        ms_dir
            Measurement Set directory.  Defaults to ``$DSA110_MS_DIR`` or
            ``/stage/dsa110-contimg/ms``.
        image_dir
            Per-date image staging directory.  Derived from *date* if omitted.
        products_dir
            Final products directory.  Derived from *date* if omitted.
        """
        _cal = cal_date or date
        _ms = ms_dir or os.environ.get("DSA110_MS_DIR", "/stage/dsa110-contimg/ms")
        _img = image_dir or f"/stage/dsa110-contimg/images/mosaic_{date}"
        _prod = products_dir or (
            os.environ.get("DSA110_PRODUCTS_BASE", "/data/dsa110-proc/products/mosaics")
            + f"/{date}"
        )
        _bp, _g = _resolve_cal_table_paths(_ms, _cal)
        return TileConfig(
            date=date,
            ms_dir=_ms,
            image_dir=_img,
            mosaic_out=f"{_img}/full_mosaic.fits",
            products_dir=_prod,
            bp_table=_bp,
            g_table=_g,
        )

    def replace(self, **kwargs) -> "TileConfig":
        """Return a new TileConfig with selected fields overridden."""
        return dataclasses.replace(self, **kwargs)

    def to_dict(self) -> dict:
        """Serialize to a plain dict (for subprocess pickling / JSON)."""
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "TileConfig":
        """Reconstruct from a plain dict."""
        return TileConfig(**d)


@dataclasses.dataclass(frozen=True)
class TileResult:
    """Outcome of process_ms() for a single tile."""

    status: str  # "imaged" | "cached" | "failed"
    fits_path: str | None = None
    failed_stage: str | None = (
        None  # "phaseshift", "applycal", "allzero", "imaging", "no_output", "qa", "timeout"
    )
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Return True if the tile was successfully imaged or cached."""
        return self.status in ("imaged", "cached")

    def to_dict(self) -> dict:
        """Serialize to a plain dict for subprocess transport."""
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "TileResult":
        """Reconstruct a TileResult from a plain dict."""
        return TileResult(**d)


# ── Imaging parameters (constants — do not vary per run) ─────────────────────
IMSIZE = 2400
CELL_ARCSEC = 6.0
WEIGHTING = "briggs"
ROBUST = 0.5
NITER = 1000
THRESHOLD = "0.005Jy"

# Primary beam cutoff: blank tile pixels where the WSClean beam model is below
# this fraction of the peak response.  Values < PB_CUTOFF have high noise
# amplification in pb-corrected images and cause severe edge artefacts in the mosaic.
PB_CUTOFF = 0.2  # 20 % of peak response

# ── Helpers ──────────────────────────────────────────────────────────────────


def find_valid_ms(cfg: TileConfig) -> list[str]:
    """Return sorted list of valid (non-corrupt) raw MS paths for the configured date."""
    from dsa110_continuum.adapters.casa_tables import table  # noqa: PLC0415

    candidates = sorted(glob.glob(f"{cfg.ms_dir}/{cfg.date}T*.ms"))
    candidates = [p for p in candidates if "meridian" not in p and "flagversion" not in p]
    valid = []
    for path in candidates:
        field_path = os.path.join(path, "FIELD")
        if not os.path.exists(field_path):
            log.warning("Skipping corrupt MS (missing FIELD table): %s", path)
            continue
        if not os.path.isdir(field_path):
            log.warning("Skipping corrupt MS (FIELD table is not a directory): %s", path)
            continue
        try:
            with table(field_path, readonly=True, ack=False) as _:
                pass
            valid.append(path)
        except Exception as exc:
            log.warning(
                "Skipping corrupt MS (unreadable FIELD table: %s: %s): %s",
                type(exc).__name__,
                exc,
                path,
            )
    log.info("Found %d valid MS files", len(valid))
    return valid


def get_meridian_path(ms_path: str) -> str:
    """Return the meridian-phaseshifted MS path for a given raw MS."""
    return ms_path.replace(".ms", "_meridian.ms")


def needs_calibration(ms_path: str) -> bool:
    """Return True if CORRECTED_DATA doesn't exist or has ratio close to 1."""
    from dsa110_continuum.adapters.casa_tables import table  # noqa: PLC0415

    # NOTE: TypeError (e.g. wrong argument type) is intentionally NOT caught
    # here — it indicates a programming error and should propagate to the caller.
    try:
        with table(ms_path, readonly=True, ack=False) as t:
            if "CORRECTED_DATA" not in t.colnames():
                return True
            raw = t.getcol("DATA", nrow=1000)
            corr = t.getcol("CORRECTED_DATA", nrow=1000)
            flag = t.getcol("FLAG", nrow=1000)
            good = ~flag
            if not good.any():
                return True
            ratio = np.mean(np.abs(corr[good])) / np.mean(np.abs(raw[good]))
            return ratio < 5.0
    except (OSError, RuntimeError) as e:
        log.warning("needs_calibration(%s) failed: %s — assuming calibration needed", ms_path, e)
        return True


def _ms_is_valid(path: str) -> bool:
    """Return True only if path looks like a complete CASA Measurement Set."""
    import glob as _g

    return (
        os.path.isdir(path)
        and os.path.exists(os.path.join(path, "table.dat"))
        and len(_g.glob(os.path.join(path, "table.f*"))) > 0
    )


def _applycal_sentinel_path(meridian_ms: str) -> str:
    """Return the path of the applycal-completion sentinel for a meridian MS."""
    return meridian_ms.rstrip("/") + ".applycal_done"


def _fits_is_valid(fits_path: str) -> bool:
    """Return True if FITS file exists and has a readable, non-empty data array."""
    if not os.path.isfile(fits_path):
        return False
    try:
        with fits.open(fits_path) as hdul:
            if len(hdul) < 1 or hdul[0].data is None:
                return False
            _ = hdul[0].data.shape  # triggers decompression, catches truncation
            return True
    except Exception:
        return False


def generate_windows(
    tile_paths: list[str],
    window_tiles: int,
    stride_tiles: int,
) -> list[list[str]]:
    """Generate sliding windows over time-ordered tiles.

    Parameters
    ----------
    tile_paths
        Time-ordered list of tile FITS paths.
    window_tiles
        Maximum number of tiles per window.
    stride_tiles
        Step size between consecutive window starts.

    Returns
    -------
    list[list[str]]
        Each element is a list of tile paths for one mosaic window.
        Windows with fewer than 2 tiles are dropped (cannot mosaic).
    """
    if len(tile_paths) <= window_tiles:
        return [list(tile_paths)] if len(tile_paths) >= 2 else []
    windows: list[list[str]] = []
    for start in range(0, len(tile_paths), stride_tiles):
        window = tile_paths[start : start + window_tiles]
        if len(window) >= 2:
            windows.append(window)
        if start + window_tiles >= len(tile_paths):
            break
    return windows


def validate_window_params(window_tiles: int, stride_tiles: int) -> list[str]:
    """Validate windowing CLI parameters.

    Returns
    -------
    list[str]
        Empty if valid; otherwise one error message per violation.
    """
    errors: list[str] = []
    if window_tiles < 2:
        errors.append(f"--window-tiles must be >= 2, got {window_tiles}")
    if stride_tiles < 1:
        errors.append(f"--stride-tiles must be >= 1, got {stride_tiles}")
    if stride_tiles > window_tiles and window_tiles >= 2:
        errors.append(f"--stride-tiles ({stride_tiles}) must be <= --window-tiles ({window_tiles})")
    return errors


def process_ms(
    ms_path: str,
    cfg: TileConfig,
    keep_intermediates: bool = False,
    force_recal: bool = False,
) -> TileResult:
    """Phaseshift → applycal → image one MS. Returns a TileResult."""
    tag = Path(ms_path).stem  # e.g. 2026-01-25T21:17:33
    meridian_ms = get_meridian_path(ms_path)
    imagename = os.path.join(cfg.image_dir, tag)
    # With pbcor=True, WSClean produces {imagename}-image-pb.fits (primary-beam corrected)
    pbcor_fits = imagename + "-image-pb.fits"
    image_fits = imagename + "-image.fits"

    # Skip if already fully processed — unless force_recal requests a fresh run
    if not force_recal:
        if _fits_is_valid(pbcor_fits):
            log.info("[%s] PB-corrected image already exists — skipping", tag)
            return TileResult("cached", fits_path=pbcor_fits)
        if _fits_is_valid(image_fits):
            log.info("[%s] Image already exists (no pbcor) — skipping", tag)
            return TileResult("cached", fits_path=image_fits)
        # Remove partial/corrupt FITS so imaging reruns cleanly
        for partial in (pbcor_fits, image_fits):
            if os.path.isfile(partial):
                log.warning("[%s] Removing invalid/partial FITS: %s", tag, partial)
                try:
                    os.remove(partial)
                except OSError as e:
                    log.warning("[%s] Could not remove %s: %s", tag, partial, e)

    # ── Step 1: Phaseshift ────────────────────────────────────────────────
    sentinel = _applycal_sentinel_path(meridian_ms)
    if _ms_is_valid(meridian_ms):
        log.info("[%s] Meridian MS already exists", tag)
    else:
        # Invalidate stale sentinel — the MS it referred to is gone
        if os.path.isfile(sentinel):
            os.remove(sentinel)
        if os.path.isdir(meridian_ms):
            log.warning(
                "[%s] Corrupt or incomplete meridian MS detected — removing: %s", tag, meridian_ms
            )
            shutil.rmtree(meridian_ms)
        log.info("[%s] Phaseshifting to median meridian ...", tag)
        try:
            phaseshift_ms(
                ms_path=ms_path,
                mode="median_meridian",
                output_ms=meridian_ms,
            )
        except Exception as e:
            log.error("[%s] Phaseshift failed: %s", tag, e)
            return TileResult("failed", failed_stage="phaseshift", error=str(e))

    # ── Step 2: Calibration ────────────────────────────────────────────────
    applycal_needed = force_recal or not os.path.isfile(sentinel)
    if not applycal_needed and needs_calibration(meridian_ms):
        log.warning(
            "[%s] Applycal sentinel exists but ratio check disagrees — "
            "trusting sentinel (use --force-recal to override)",
            tag,
        )
    if applycal_needed:
        if os.path.isfile(sentinel):
            os.remove(sentinel)  # clear stale sentinel before starting
        log.info("[%s] Applying calibration (force_recal=%s) ...", tag, force_recal)
        try:
            apply_to_target(
                ms_target=meridian_ms,
                field="",
                gaintables=[cfg.bp_table, cfg.g_table],
                interp=["nearest", "linear"],
            )
        except Exception as e:
            log.error("[%s] Applycal failed: %s", tag, e)
            return TileResult("failed", failed_stage="applycal", error=str(e))

        # Verify CORRECTED_DATA isn't all zeros (silent applycal failure mode)
        try:
            from dsa110_continuum.adapters.casa_tables import table  # noqa: PLC0415

            with table(meridian_ms, readonly=True, ack=False) as t:
                if "CORRECTED_DATA" in t.colnames():
                    cd = t.getcol("CORRECTED_DATA", nrow=2048)
                    fl = t.getcol("FLAG", nrow=2048)
                    unflagged = cd[~fl]
                    if len(unflagged) > 0 and np.all(np.abs(unflagged) < 1e-10):
                        log.error("[%s] CORRECTED_DATA is all zeros after applycal", tag)
                        return TileResult(
                            "failed", failed_stage="allzero", error="CORRECTED_DATA all zeros"
                        )
        except (OSError, RuntimeError) as e:
            log.warning("[%s] Post-applycal check failed: %s — continuing", tag, e)

        # Write sentinel ONLY after applycal + verification both pass
        try:
            Path(sentinel).write_text("applycal completed\n")
        except OSError as e:
            log.warning("[%s] Could not write applycal sentinel: %s", tag, e)
    else:
        log.info("[%s] Calibration already applied (sentinel exists)", tag)

    # ── Step 3: Image ──────────────────────────────────────────────────────
    log.info("[%s] Imaging with WSClean ...", tag)
    try:
        image_ms(
            ms_path=meridian_ms,
            imagename=imagename,
            imsize=IMSIZE,
            cell_arcsec=CELL_ARCSEC,
            weighting=WEIGHTING,
            robust=ROBUST,
            niter=NITER,
            threshold=THRESHOLD,
            pbcor=True,
            gridder="wgridder",
            backend="wsclean",
            use_unicat_mask=False,
        )
    except Exception as e:
        log.error("[%s] Imaging failed: %s", tag, e)
        return TileResult("failed", failed_stage="imaging", error=str(e))

    result_fits = None
    if os.path.exists(pbcor_fits):
        log.info("[%s] PB-corrected image done: %s", tag, pbcor_fits)
        result_fits = pbcor_fits
    elif os.path.exists(image_fits):
        log.warning(
            "[%s] -image-pb.fits not found; falling back to plain image: %s", tag, image_fits
        )
        result_fits = image_fits
    else:
        log.error("[%s] WSClean finished but no image FITS found", tag)
        return TileResult("failed", failed_stage="no_output", error="no FITS after WSClean")

    # ── Per-tile image QA ─────────────────────────────────────────────────
    from dsa110_continuum.validation.image_validator import validate_image_quality

    tile_ok, tile_errors = validate_image_quality(
        Path(result_fits), min_snr=3.0, max_flagged_fraction=0.5
    )
    if not tile_ok:
        for err in tile_errors:
            log.warning("[%s] Tile QA: %s", tag, err)
        fatal = [
            e for e in tile_errors if "all zeros" in e.lower() or "no valid pixels" in e.lower()
        ]
        if fatal:
            log.error("[%s] Tile rejected by image QA", tag)
            return TileResult("failed", failed_stage="qa", error="; ".join(fatal))

    # ── Cleanup: delete meridian MS + sentinel now that imaging succeeded ─────
    if not keep_intermediates and os.path.isdir(meridian_ms):
        try:
            shutil.rmtree(meridian_ms)
            log.info("[%s] Deleted intermediate meridian MS: %s", tag, meridian_ms)
        except Exception as e:
            log.warning("[%s] Could not delete meridian MS %s: %s", tag, meridian_ms, e)
        if os.path.isfile(sentinel):
            try:
                os.remove(sentinel)
            except OSError:
                pass

    return TileResult("imaged", fits_path=result_fits)


# ── Mosaicking ────────────────────────────────────────────────────────────────


def _get_tile_center_ra(path: str) -> float:
    """Return the centre RA (deg) of a tile FITS image."""
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        wcs = WCS(hdr).celestial
        ny, nx = hdul[0].data.squeeze().shape
        center = wcs.pixel_to_world(nx / 2.0, ny / 2.0)
        return center.ra.deg


def group_tiles_by_ra(fits_paths: list[str], gap_deg: float = 10.0) -> list[list[str]]:
    """Split tiles into contiguous RA strips.

    Tiles are sorted by centre RA.  Wherever the gap between consecutive
    tiles exceeds *gap_deg*, a new strip begins.  This prevents mosaicking
    disjoint fields (e.g. 02h and 22h observations) into a single
    oversized grid.
    """
    if not fits_paths:
        return []
    centers = [(p, _get_tile_center_ra(p)) for p in fits_paths]
    centers.sort(key=lambda x: x[1])

    groups: list[list[str]] = [[centers[0][0]]]
    for i in range(1, len(centers)):
        delta = centers[i][1] - centers[i - 1][1]
        if delta > gap_deg:
            groups.append([])
        groups[-1].append(centers[i][0])

    # Also check wrap-around gap (last → first across 0°/360°)
    if len(groups) > 1:
        wrap_gap = (centers[0][1] + 360.0) - centers[-1][1]
        if wrap_gap < gap_deg:
            # First and last groups are actually contiguous across the wrap
            groups[0] = groups[-1] + groups[0]
            groups.pop()

    log.info("Grouped %d tiles into %d RA strip(s)", len(fits_paths), len(groups))
    for i, g in enumerate(groups):
        ras = [_get_tile_center_ra(p) for p in g]
        log.info("  Strip %d: %d tiles, RA %.1f–%.1f deg", i, len(g), min(ras), max(ras))
    return groups


def build_common_wcs(fits_paths: list[str], margin_deg: float = 0.5) -> tuple[WCS, int, int]:
    """Compute a common RA/Dec WCS that covers all input FITS images.

    Handles RA wrap by shifting all corner RAs relative to their mean
    before computing the bounding box, so tiles crossing 0°/360° produce
    a compact footprint instead of a ~360° span.
    """
    all_ras: list[float] = []
    dec_min, dec_max = 90.0, -90.0

    for path in fits_paths:
        with fits.open(path) as hdul:
            hdr = hdul[0].header
            wcs = WCS(hdr).celestial
            ny, nx = hdul[0].data.squeeze().shape
            corners = wcs.pixel_to_world([0, nx - 1, 0, nx - 1], [0, 0, ny - 1, ny - 1])
            ras = [c.ra.deg for c in corners]
            decs = [c.dec.deg for c in corners]
            all_ras.extend(ras)
            dec_min = min(dec_min, min(decs))
            dec_max = max(dec_max, max(decs))

    # ── RA wrap-safe bounding box ─────────────────────────────────────────
    # Use circular mean (atan2 of unit-vector average) so that tiles
    # crossing the 0°/360° boundary get a correct centre.  The arithmetic
    # mean of [359°, 1°] is 180° (wrong); the circular mean is 0° (correct).
    ra_rad = np.deg2rad(all_ras)
    mean_ra = (
        float(np.rad2deg(np.arctan2(np.mean(np.sin(ra_rad)), np.mean(np.cos(ra_rad))))) % 360.0
    )
    shifted = np.array([(ra - mean_ra + 180.0) % 360.0 - 180.0 for ra in all_ras])
    ra_min = mean_ra + float(shifted.min()) - margin_deg
    ra_max = mean_ra + float(shifted.max()) + margin_deg
    dec_min -= margin_deg
    dec_max += margin_deg

    # Normalise center RA to [0, 360)
    ra_center = ((ra_min + ra_max) / 2.0) % 360.0
    dec_center = (dec_min + dec_max) / 2.0

    pixel_scale_deg = CELL_ARCSEC / 3600.0
    nx = int(np.ceil((ra_max - ra_min) / pixel_scale_deg))
    ny = int(np.ceil((dec_max - dec_min) / pixel_scale_deg))
    # Round to even for WSClean compatibility
    nx = nx + (nx % 2)
    ny = ny + (ny % 2)

    out_wcs = WCS(naxis=2)
    out_wcs.wcs.crpix = [nx / 2 + 0.5, ny / 2 + 0.5]
    out_wcs.wcs.cdelt = [-pixel_scale_deg, pixel_scale_deg]
    out_wcs.wcs.crval = [ra_center, dec_center]
    out_wcs.wcs.ctype = ["RA---SIN", "DEC--SIN"]

    log.info(
        "Common WCS: RA %.2f–%.2f deg, Dec %.2f–%.2f deg, %d×%d px (center RA=%.2f)",
        ra_min,
        ra_max,
        dec_min,
        dec_max,
        nx,
        ny,
        ra_center,
    )
    return out_wcs, ny, nx


def coadd_tiles(fits_paths: list[str], out_wcs: WCS, ny: int, nx: int) -> np.ndarray:
    """Reproject each tile onto the common WCS and do noise-weighted coaddition."""
    from reproject import reproject_interp

    sum_image = np.zeros((ny, nx), dtype=np.float64)
    sum_weight = np.zeros((ny, nx), dtype=np.float64)
    out_header = out_wcs.to_header()
    out_header["NAXIS1"] = nx
    out_header["NAXIS2"] = ny

    for path in fits_paths:
        log.info("Reprojecting %s ...", Path(path).name)
        with fits.open(path) as hdul:
            data = hdul[0].data.squeeze().astype(np.float64)
            hdr = hdul[0].header

        # ── Primary beam cutoff ────────────────────────────────────────────
        # WSClean writes a companion *-pb.fits beam model alongside the
        # pb-corrected image.  Pixels where beam < PB_CUTOFF have been
        # noise-amplified by 1/beam and cause severe edge artefacts in the
        # mosaic; blank them before combining.
        # WSClean names the per-channel beam model "{tag}-beam-0.fits" (not -pb.fits)
        pb_path = path.replace("-image-pb.fits", "-beam-0.fits")
        if not os.path.exists(pb_path):
            # Fallback: plain image tile has no -image-pb suffix
            pb_path = path.replace("-image.fits", "-beam-0.fits")
        if PB_CUTOFF > 0 and os.path.exists(pb_path):
            with fits.open(pb_path) as pb_hdul:
                pb_data = pb_hdul[0].data.squeeze().astype(np.float64)
            # Normalise to peak in case WSClean stores absolute sensitivity
            pb_peak = np.nanmax(pb_data)
            if pb_peak > 0:
                pb_data /= pb_peak
            low_beam = (pb_data < PB_CUTOFF) | ~np.isfinite(pb_data)
            data[low_beam] = np.nan
            n_blanked = low_beam.sum()
            log.info("  PB cutoff (%.0f%%): blanked %d pixels", PB_CUTOFF * 100, n_blanked)
        elif PB_CUTOFF > 0:
            log.warning("  No pb.fits found for %s — skipping beam cutoff", Path(path).name)

        # Estimate per-tile noise from the central region (away from bright sources)
        cy, cx = data.shape[0] // 2, data.shape[1] // 2
        margin = 200  # pixels from center
        inner = data[
            max(0, cy - margin) : cy + margin,
            max(0, cx - margin) : cx + margin,
        ]
        noise = np.nanstd(inner[np.isfinite(inner)])
        if noise <= 0 or not np.isfinite(noise):
            noise = 1.0  # fallback unit weight
        weight = 1.0 / (noise**2)

        # Reproject onto common grid
        try:
            reprojected, footprint = reproject_interp(
                (data, WCS(hdr).celestial),
                out_header,
                shape_out=(ny, nx),
            )
        except Exception as e:
            log.warning("Reproject failed for %s: %s — skipping", Path(path).name, e)
            continue

        valid = footprint.astype(bool) & np.isfinite(reprojected)
        sum_image[valid] += weight * reprojected[valid]
        sum_weight[valid] += weight

    mosaic = np.where(sum_weight > 0, sum_image / sum_weight, np.nan)
    return mosaic


def write_mosaic(
    mosaic: np.ndarray, out_wcs: WCS, fits_paths: list[str], output_path: str, date: str = ""
) -> str:
    """Write mosaic to FITS using a representative header from the first tile."""
    out_path = output_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with fits.open(fits_paths[0]) as ref:
        ref_hdr = ref[0].header.copy()

    # Build minimal 2D header from common WCS
    new_hdr = fits.Header()
    new_hdr["SIMPLE"] = True
    new_hdr["BITPIX"] = -32
    new_hdr["NAXIS"] = 2
    new_hdr["NAXIS1"] = mosaic.shape[1]
    new_hdr["NAXIS2"] = mosaic.shape[0]
    # Copy beam parameters from reference
    for key in ("BMAJ", "BMIN", "BPA", "BUNIT", "RESTFRQ", "EQUINOX"):
        if key in ref_hdr:
            new_hdr[key] = ref_hdr[key]
    new_hdr.update(out_wcs.to_header())
    new_hdr["HISTORY"] = f"Mosaic of {len(fits_paths)} DSA-110 tiles ({date})"

    hdu = fits.PrimaryHDU(data=mosaic.astype(np.float32), header=new_hdr)
    hdu.writeto(out_path, overwrite=True)
    log.info("Mosaic written: %s", out_path)
    return out_path


def check_mosaic_quality(mosaic_path: str) -> bool:
    """Check noise consistency across horizontal strips of the mosaic."""
    with fits.open(mosaic_path) as hdul:
        data = hdul[0].data.squeeze()

    ny, nx = data.shape
    n_strips = 5
    strip_height = ny // n_strips
    noises = []
    for i in range(n_strips):
        strip = data[i * strip_height : (i + 1) * strip_height, :]
        valid = strip[np.isfinite(strip)]
        if len(valid) > 100:
            # Robust noise estimate via MAD
            mad = np.median(np.abs(valid - np.median(valid)))
            noises.append(1.4826 * mad)

    if not noises:
        log.warning("QA: no valid strips")
        return False

    log.info("QA noise per strip [Jy/beam]: %s", [f"{n:.4f}" for n in noises])
    max_ratio = max(noises) / min(noises)
    log.info("QA max/min noise ratio: %.2f (pass if < 3.0)", max_ratio)
    passed = max_ratio < 3.0
    if passed:
        log.info("QA PASSED: noise consistent across field")
    else:
        log.warning("QA FAILED: noise varies too much (ratio=%.2f)", max_ratio)
    return passed


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    """Parse CLI arguments and run the mosaic pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Produce science mosaics from DSA-110 drift observations. "
            "Default: sliding-window mosaics (~1 hour each). "
            "Use --full-day for legacy all-day diagnostic mosaic."
        ),
    )
    parser.add_argument(
        "--date",
        default="2026-01-25",
        metavar="YYYY-MM-DD",
        help="Observation date to process (default: %(default)s).",
    )
    parser.add_argument(
        "--cal-date",
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "Date whose calibration tables (BP/gain) to use. "
            "Defaults to --date if not provided. "
            "Use when processing a new date whose cal tables are symlinked from 2026-01-25."
        ),
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        default=False,
        help="Keep *_meridian.ms files and skip moving the mosaic to products/ (useful for debugging).",
    )
    parser.add_argument(
        "--window-tiles",
        type=int,
        default=12,
        metavar="N",
        help=(
            "Tiles per science mosaic window (default: %(default)s). "
            "Each window produces an independent ~1-hour mosaic product. "
            "Output: {date}_w{NN}_mosaic.fits"
        ),
    )
    parser.add_argument(
        "--stride-tiles",
        type=int,
        default=6,
        metavar="N",
        help=(
            "Step between consecutive window starts in tiles (default: %(default)s). "
            "stride < window-tiles gives overlapping windows for continuity."
        ),
    )
    parser.add_argument(
        "--full-day",
        action="store_true",
        default=False,
        help=(
            "Diagnostic/legacy: coadd ALL tiles for the date into monolithic "
            "full_mosaic.fits strip(s). Not the default science product."
        ),
    )
    args = parser.parse_args()
    keep = args.keep_intermediates

    # ── Validate windowing parameters ─────────────────────────────────────────
    if not args.full_day:
        win_errors = validate_window_params(args.window_tiles, args.stride_tiles)
        if win_errors:
            for e in win_errors:
                log.error("Invalid windowing parameter: %s", e)
            sys.exit(1)

    # Build immutable config from CLI args
    cfg = TileConfig.build(date=args.date, cal_date=args.cal_date)

    # ── Cal-table validation (ABORT early if missing) ─────────────────────────
    _missing = [t for t in [cfg.bp_table, cfg.g_table] if not os.path.exists(t)]
    if _missing:
        for _t in _missing:
            log.error("ABORT: calibration table not found: %s", _t)
        log.error("Available .b tables in %s:", cfg.ms_dir)
        for _f in sorted(os.listdir(cfg.ms_dir)):
            if _f.endswith(".b"):
                log.error("  %s", _f)
        cal_date = args.cal_date or args.date
        log.error(
            "To use a different date's tables, run with: --cal-date YYYY-MM-DD\n"
            "To symlink from 2026-01-25, run:\n"
            "  ln -s %s/2026-01-25T22:26:05_0~23.b %s\n"
            "  ln -s %s/2026-01-25T22:26:05_0~23.g %s",
            cfg.ms_dir,
            cfg.bp_table,
            cfg.ms_dir,
            cfg.g_table,
        )
        sys.exit(1)

    cal_date = args.cal_date or args.date
    if cal_date != cfg.date:
        log.info("Calibration tables from: %s", cal_date)
    log.info("Cal tables verified: %s, %s", cfg.bp_table, cfg.g_table)
    os.makedirs(cfg.image_dir, exist_ok=True)

    ms_list = find_valid_ms(cfg)
    if not ms_list:
        log.error("No valid MS files found — aborting")
        sys.exit(1)

    # ── Phase 1: Process each MS ──────────────────────────────────────────────
    tile_images = []
    for ms_path in ms_list:
        result = process_ms(ms_path, cfg, keep_intermediates=keep)
        if result.ok:
            tile_images.append(result.fits_path)
        else:
            log.warning(
                "Skipping failed MS (%s): %s — %s",
                result.failed_stage,
                ms_path,
                result.error or "unknown",
            )

    log.info("\n=== Processed %d/%d tiles successfully ===\n", len(tile_images), len(ms_list))

    if len(tile_images) < 2:
        log.error("Need at least 2 tile images to mosaic — aborting")
        sys.exit(1)

    # ── Phase 2: Build mosaics ──────────────────────────────────────────────
    mosaic_paths: list[str] = []

    if args.full_day:
        # Legacy/diagnostic: coadd ALL tiles into monolithic strip mosaic(s)
        log.warning(
            "FULL-DAY MODE: coadding all %d tiles into monolithic mosaic(s). "
            "This is diagnostic — not the default science product.",
            len(tile_images),
        )
        strips = group_tiles_by_ra(tile_images)
        for strip_idx, strip_tiles in enumerate(strips):
            if len(strip_tiles) < 2:
                log.warning("Strip %d has only %d tile — skipping", strip_idx, len(strip_tiles))
                continue
            if len(strips) == 1:
                strip_out = cfg.mosaic_out
            else:
                tile_ras = [_get_tile_center_ra(p) for p in strip_tiles]
                ra_rad = np.deg2rad(tile_ras)
                strip_ra = (
                    float(np.rad2deg(np.arctan2(np.mean(np.sin(ra_rad)), np.mean(np.cos(ra_rad)))))
                    % 360.0
                )
                strip_out = cfg.mosaic_out.replace(
                    ".fits", f"_ra{int(round(strip_ra)) % 360:03d}.fits"
                )
            if os.path.exists(strip_out):
                log.info("Strip %d mosaic already exists: %s", strip_idx, strip_out)
                mosaic_paths.append(strip_out)
                continue
            log.info(
                "\n=== [full-day] Building mosaic for strip %d (%d tiles) ===\n",
                strip_idx,
                len(strip_tiles),
            )
            out_wcs, ny, nx = build_common_wcs(strip_tiles)
            mosaic_data = coadd_tiles(strip_tiles, out_wcs, ny, nx)
            strip_path = write_mosaic(
                mosaic_data,
                out_wcs,
                strip_tiles,
                output_path=strip_out,
                date=cfg.date,
            )
            mosaic_paths.append(strip_path)
    else:
        # Science mode: sliding-window mosaics (~1 hour products)
        windows = generate_windows(tile_images, args.window_tiles, args.stride_tiles)
        if len(tile_images) < args.window_tiles:
            log.warning(
                "Only %d tiles available (< --window-tiles %d); "
                "producing single short-window mosaic",
                len(tile_images),
                args.window_tiles,
            )
        log.info(
            "Science mode: %d window(s) of <=%d tiles, stride %d",
            len(windows),
            args.window_tiles,
            args.stride_tiles,
        )
        for win_idx, win_tiles in enumerate(windows):
            strips = group_tiles_by_ra(win_tiles)
            for strip_idx, strip_tiles in enumerate(strips):
                if len(strip_tiles) < 2:
                    log.warning(
                        "Window %d strip %d has only %d tile — skipping",
                        win_idx,
                        strip_idx,
                        len(strip_tiles),
                    )
                    continue
                base = f"{cfg.date}_w{win_idx:02d}_mosaic"
                if len(strips) > 1:
                    tile_ras = [_get_tile_center_ra(p) for p in strip_tiles]
                    ra_rad = np.deg2rad(tile_ras)
                    strip_ra = (
                        float(
                            np.rad2deg(np.arctan2(np.mean(np.sin(ra_rad)), np.mean(np.cos(ra_rad))))
                        )
                        % 360.0
                    )
                    base += f"_ra{int(round(strip_ra)) % 360:03d}"
                win_out = os.path.join(cfg.image_dir, base + ".fits")
                if os.path.exists(win_out):
                    log.info("Window %d mosaic already exists: %s", win_idx, win_out)
                    mosaic_paths.append(win_out)
                    continue
                log.info(
                    "\n=== Building mosaic: window %d, strip %d (%d tiles) ===\n",
                    win_idx,
                    strip_idx,
                    len(strip_tiles),
                )
                out_wcs, ny, nx = build_common_wcs(strip_tiles)
                mosaic_data = coadd_tiles(strip_tiles, out_wcs, ny, nx)
                win_path = write_mosaic(
                    mosaic_data,
                    out_wcs,
                    strip_tiles,
                    output_path=win_out,
                    date=cfg.date,
                )
                mosaic_paths.append(win_path)

    if not mosaic_paths:
        log.error("No mosaics produced — aborting")
        sys.exit(1)

    # ── Phase 3: QA (per strip) ───────────────────────────────────────────────
    all_passed = True
    for mpath in mosaic_paths:
        log.info("\n=== Mosaic QA: %s ===\n", Path(mpath).name)
        passed = check_mosaic_quality(mpath)
        all_passed = all_passed and passed

        with fits.open(mpath) as hdul:
            data = hdul[0].data.squeeze()
            peak = np.nanmax(data)
            finite = data[np.isfinite(data)]
            rms = 1.4826 * np.nanmedian(np.abs(finite - np.nanmedian(finite)))
            log.info("  Peak: %.4f Jy/beam", peak)
            log.info("  RMS (MAD): %.4f Jy/beam", rms)
            if rms > 0 and np.isfinite(rms):
                log.info("  Dynamic range: %.0f", peak / rms)
            else:
                log.info("  Dynamic range: n/a (rms=0 or non-finite)")

    mode_label = "full-day" if args.full_day else "science-window"
    if all_passed:
        print(f"\nSUCCESS: {len(mosaic_paths)} {mode_label} mosaic(s) in {cfg.image_dir}")
        for mp in mosaic_paths:
            print(f"  {Path(mp).name}")
    else:
        print("\nWARNING: One or more mosaic QA checks failed — check noise consistency")

    # ── Phase 4: Move finished mosaics to products/ ──────────────────────────
    if not keep:
        _move_mosaic_to_products(cfg, full_day=args.full_day)

    return mosaic_paths[0] if len(mosaic_paths) == 1 else mosaic_paths


def _move_mosaic_to_products(cfg: TileConfig, *, full_day: bool) -> None:
    """Archive matching mosaic files from the staging area to products.

    Moves individual mosaic FITS files from *cfg.image_dir* into
    *cfg.products_dir*, selecting only files that match the active mode:
    ``full_mosaic*.fits`` when *full_day* is True, or
    ``{date}_w*_mosaic*.fits`` otherwise.  Other files in the staging
    directory (tile images, beam models) are left in place.

    Skips with a warning if *products_dir* already exists.
    """
    import glob as _g

    if full_day:
        mosaic_files = sorted(_g.glob(os.path.join(cfg.image_dir, "full_mosaic*.fits")))
    else:
        mosaic_files = sorted(_g.glob(os.path.join(cfg.image_dir, f"{cfg.date}_w*_mosaic*.fits")))
    if not mosaic_files:
        log.warning("Move skipped: no mosaic files found in %s", cfg.image_dir)
        return

    if os.path.exists(cfg.products_dir):
        log.warning(
            "Move skipped: destination already exists — remove it manually if you want to overwrite: %s",
            cfg.products_dir,
        )
        return

    os.makedirs(cfg.products_dir, exist_ok=True)

    mode_label = "full-day" if full_day else "science-window"
    log.info(
        "Archiving %d %s mosaic(s): %s → %s",
        len(mosaic_files),
        mode_label,
        cfg.image_dir,
        cfg.products_dir,
    )
    try:
        for src in mosaic_files:
            dst = os.path.join(cfg.products_dir, Path(src).name)
            shutil.move(src, dst)
            print(f"  Archived: {dst}")
        log.info("Archival complete: %s", cfg.products_dir)
    except Exception as e:
        log.error("Failed to archive mosaic to products: %s", e)
        print(f"\nERROR: Could not archive mosaic to {cfg.products_dir}: {e}")


if __name__ == "__main__":
    main()
