"""Simulated DSA-110 continuum pipeline for end-to-end testing.

Orchestrates all five pipeline stages using real production code where possible
(WSClean subprocess, astropy, pyuvdata, casacore.tables) and faithful simulation
where CASA is unavailable (calibration solve via Jacobi antenna factorisation).

Stages:
  1. Gain corruption  — corrupt_uvh5()
  2. Calibration      — _calibrate()
  3. Imaging          — _image()
  4. Mosaicking       — _mosaic()
  5. Photometry       — _photometry()
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyuvdata

if TYPE_CHECKING:
    from dsa110_continuum.simulation.harness import SimulationHarness
    from dsa110_continuum.simulation.ground_truth import GroundTruthRegistry

logger = logging.getLogger(__name__)


@dataclass
class SourceFluxResult:
    """Recovered vs. injected flux for one source."""
    source_id: str
    ra_deg: float
    dec_deg: float
    injected_flux_jy: float
    recovered_flux_jy: float   # NaN if not measurable
    snr: float                 # NaN if not measurable
    passed: bool               # True if |recovered - injected| / injected < tolerance


@dataclass
class SimulatedPipelineResult:
    """Outcome of a full simulated pipeline run."""
    work_dir: Path
    n_tiles: int
    calibration_passed: bool
    imaging_passed: bool
    mosaic_path: Path | None
    source_results: list[SourceFluxResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def n_recovered(self) -> int:
        return sum(1 for r in self.source_results if r.passed)

    @property
    def all_passed(self) -> bool:
        return (
            self.calibration_passed
            and self.imaging_passed
            and self.n_recovered == len(self.source_results)
        )


class SimulatedPipeline:
    """Orchestrate all five pipeline stages on simulated DSA-110 data.

    Parameters
    ----------
    work_dir:
        Root scratch directory; sub-directories created per stage.
    wsclean_bin:
        Path to wsclean binary (default: ``wsclean`` on PATH).
    niter:
        WSClean CLEAN iterations (default 1000).
    cell_arcsec:
        WSClean cell size in arcseconds (default 20.0).
    image_size:
        WSClean image size in pixels (default 512).
    """

    def __init__(
        self,
        work_dir: Path | str,
        *,
        wsclean_bin: str = "wsclean",
        niter: int = 1000,
        cell_arcsec: float = 20.0,
        image_size: int = 512,
        wsclean_mem_gb: float | None = None,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.wsclean_bin = wsclean_bin
        self.niter = niter
        self.cell_arcsec = cell_arcsec
        self.image_size = image_size
        self.wsclean_mem_gb = wsclean_mem_gb

    # ------------------------------------------------------------------ #
    # Stage 2 — Calibration (CASA-free Jacobi solver)                     #
    # ------------------------------------------------------------------ #

    def _calibrate(
        self,
        *,
        target_ms: Path | str,
        cal_uvh5: Path | str,
        cal_flux_jy: float,
        work_dir: Path | str,
    ) -> Path:
        """Derive per-antenna gains from calibrator UVH5 and apply to target MS.

        Algorithm:
          1. Read calibrator UVH5. For a known-flux source at phase centre,
             the model visibility is: V_model = flux_jy / 2  (real, XX = I/2).
             The harness stores vis as conj(V), so we read back conj(V_model)
             = V_model (real).
          2. Solve for per-antenna complex gains G_i using two Jacobi sweeps:
               G_i = mean_j [ V_ij_observed / (V_model * conj(G_j_prev)) ]
             Initialised with G_i = 1 for all i.
          3. Write CORRECTED_DATA = DATA / (G_i * conj(G_j)) to target MS
             via casacore.tables.

        Parameters
        ----------
        target_ms:
            Measurement Set to calibrate (DATA → CORRECTED_DATA written in place).
        cal_uvh5:
            Calibrator UVH5 (output of generate_calibrator_subband).
        cal_flux_jy:
            Known calibrator flux in Jy.
        work_dir:
            Scratch directory (reserved for future use).

        Returns
        -------
        Path
            Path to the modified target MS.
        """
        from dsa110_continuum.adapters import casa_tables as ct

        target_ms = Path(target_ms)
        cal_uvh5  = Path(cal_uvh5)

        # ── Step 1: Load calibrator visibilities ──────────────────────────
        uv_cal = pyuvdata.UVData()
        uv_cal.read(str(cal_uvh5))

        ant_nums = np.unique(
            np.concatenate([uv_cal.ant_1_array, uv_cal.ant_2_array])
        )
        n_ant  = len(ant_nums)
        n_freq = uv_cal.Nfreqs
        ant_idx = {int(a): i for i, a in enumerate(ant_nums)}

        # Model amplitude: source at phase centre → V_ij = flux/2 (real, XX=I/2)
        # Harness stores conj(V); conj of real = real, so observed ≈ flux/2.
        model_amp = cal_flux_jy / 2.0

        # ── Step 2: Jacobi gain solve (2 iterations) ──────────────────────
        # Initialise all gains to unity
        gains = np.ones((n_ant, n_freq), dtype=complex)

        for _iter in range(2):
            numerator   = np.zeros_like(gains)
            denominator = np.zeros((n_ant, n_freq), dtype=float)

            for row in range(uv_cal.Nblts):
                i_ant = int(uv_cal.ant_1_array[row])
                j_ant = int(uv_cal.ant_2_array[row])
                if i_ant == j_ant:
                    continue  # skip autocorrelations

                i = ant_idx[i_ant]
                j = ant_idx[j_ant]

                # Harness stores conj(V_ij); undo conjugation to get true V_ij
                vis = uv_cal.data_array[row, :, 0].conj()  # shape (n_freq,)

                # Update antenna i using antenna j's current gain
                numerator[i]   += vis * gains[j] / model_amp
                denominator[i] += np.abs(gains[j]) ** 2

                # Update antenna j using antenna i's current gain (conjugate relation)
                numerator[j]   += np.conj(vis) * gains[i] / model_amp
                denominator[j] += np.abs(gains[i]) ** 2

            gains = numerator / np.maximum(denominator, 1e-12)

        logger.info(
            "Gain solve complete: mean |G|=%.4f, mean phase=%.2f deg, n_ant=%d",
            float(np.abs(gains).mean()),
            float(np.degrees(np.angle(gains)).mean()),
            n_ant,
        )

        # ── Step 3: Apply gains → CORRECTED_DATA in target MS ─────────────
        with ct.table(str(target_ms), readonly=False, ack=False) as t:
            # Read antenna count from ANTENNA subtable
            with ct.table(str(target_ms) + "::ANTENNA", readonly=True, ack=False) as tant:
                n_ms_ant = tant.nrows()

            # Build per-MS-antenna gain lookup (0-based MS antenna index)
            # MS ANTENNA1/ANTENNA2 are 0-based indices into the ANTENNA table.
            # Map by position: the cal UVH5 antennas are in sorted order.
            ms_gains = np.ones((n_ms_ant, n_freq), dtype=complex)
            for ms_idx in range(min(n_ms_ant, n_ant)):
                ms_gains[ms_idx] = gains[ms_idx]

            data     = t.getcol("DATA")       # shape (Nrows, Nchans, Npols)
            ant1_col = t.getcol("ANTENNA1")   # 0-based
            ant2_col = t.getcol("ANTENNA2")   # 0-based
            n_rows, n_chan, n_pol = data.shape

            # Guard: calibrator and target must have the same channel count
            if n_chan != n_freq:
                raise ValueError(
                    f"Channel count mismatch: target MS has {n_chan} channels "
                    f"but calibrator UVH5 has {n_freq}. "
                    "Ensure calibrator and target use the same subband."
                )

            # Vectorised application: CORRECTED = DATA / (G_i * conj(G_j))
            # Clamp antenna indices to valid range (safety guard)
            idx1 = np.clip(ant1_col, 0, n_ms_ant - 1)
            idx2 = np.clip(ant2_col, 0, n_ms_ant - 1)
            gi = ms_gains[idx1]   # shape (Nrows, Nfreqs)
            gj = ms_gains[idx2]   # shape (Nrows, Nfreqs)
            denom = gi * np.conj(gj)   # shape (Nrows, Nfreqs)
            # Avoid division by near-zero
            safe_denom = np.where(np.abs(denom) > 1e-12, denom, 1.0)

            corrected = data.copy()
            corrected /= safe_denom[:, :, np.newaxis]

            # Add CORRECTED_DATA column if it doesn't exist
            if "CORRECTED_DATA" not in t.colnames():
                from dsa110_continuum.adapters.casa_tables import makearrcoldesc, maketabdesc
                cd = makearrcoldesc(
                    "CORRECTED_DATA",
                    data[0],
                    valuetype="complex",
                    comment="Gain-calibrated data",
                )
                t.addcols(maketabdesc(cd))

            t.putcol("CORRECTED_DATA", corrected.astype(np.complex64))

        logger.info("Wrote CORRECTED_DATA to %s", target_ms)
        return target_ms

    # ------------------------------------------------------------------ #
    # Stage 4 — Image-plane mosaicking (QUICKLOOK tier)                   #
    # ------------------------------------------------------------------ #

    def _mosaic(
        self,
        *,
        image_paths: list[Path | str],
        work_dir: Path | str,
        output_name: str = "epoch_mosaic.fits",
    ) -> Path:
        """Co-add tile FITS images into a single epoch mosaic.

        Delegates to the production ``build_mosaic()`` (PB-weighted linear
        coaddition, QUICKLOOK tier).  Primary beam correction is disabled
        because no beam model is available in the cloud sandbox; the SCIENCE
        tier would use WSClean visibility-domain joint deconvolution instead.

        Parameters
        ----------
        image_paths:
            List of tile FITS paths to co-add (minimum 2).
        work_dir:
            Output directory for the mosaic file.
        output_name:
            Filename for the output mosaic FITS (default ``epoch_mosaic.fits``).

        Returns
        -------
        Path
            Path to the written mosaic FITS file.
        """
        from dsa110_continuum.mosaic.builder import build_mosaic

        work_dir    = Path(work_dir)
        output_path = work_dir / output_name

        result = build_mosaic(
            image_paths=[Path(p) for p in image_paths],
            output_path=output_path,
            apply_pb_correction=False,  # no beam model in cloud sandbox
            write_weight_map=False,
        )

        logger.info(
            "Mosaic: %d tiles \u2192 %s  (median RMS %.3f mJy/beam)",
            len(image_paths),
            result.output_path,
            result.median_rms * 1e3,
        )
        return result.output_path

    # ------------------------------------------------------------------ #
    # Stage 3 — WSClean imaging with CLEAN deconvolution                  #
    # ------------------------------------------------------------------ #

    def _image(
        self,
        *,
        ms_path: Path | str,
        work_dir: Path | str,
        data_column: str = "CORRECTED_DATA",
    ) -> dict[str, Path]:
        """Run WSClean with CLEAN iterations on a calibrated MS.

        Reads ``data_column`` (default ``CORRECTED_DATA``) and produces
        restored, dirty, residual, and PSF FITS images.

        Parameters
        ----------
        ms_path:
            Calibrated Measurement Set.  Must contain a ``CORRECTED_DATA``
            column (or whichever column is specified in ``data_column``).
        work_dir:
            Directory for WSClean output files.  A ``wsclean_out/`` sub-
            directory is created inside it.
        data_column:
            MS column to image (default ``CORRECTED_DATA``).

        Returns
        -------
        dict[str, Path]
            Keys: ``'restored'``, ``'dirty'``, ``'residual'``, ``'psf'``.
            Values are Paths to the respective FITS files; each is checked
            for existence and a warning is logged if missing.

        Raises
        ------
        RuntimeError
            If WSClean exits with a non-zero return code.
        """
        work_dir = Path(work_dir)
        img_dir  = work_dir / "wsclean_out"
        img_dir.mkdir(parents=True, exist_ok=True)
        prefix   = str(img_dir / "wsclean")

        cmd = [
            self.wsclean_bin,
            "-name", prefix,
            "-size", str(self.image_size), str(self.image_size),
            "-scale", f"{self.cell_arcsec}asec",
            "-weight", "briggs", "0.0",
            "-niter", str(self.niter),
            "-mgain", "0.8",
            "-auto-threshold", "1.0",
            "-pol", "I",
            "-data-column", data_column,
            "-make-psf",
            "-no-update-model-required",
        ]
        if self.wsclean_mem_gb is not None:
            cmd += ["-abs-mem", str(self.wsclean_mem_gb)]
        cmd.append(str(ms_path))
        logger.info("Running WSClean: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("WSClean stderr (last 3000 chars):\n%s",
                         result.stderr[-3000:])
            raise RuntimeError(
                f"WSClean failed with return code {result.returncode}"
            )

        outputs = {
            "restored":  Path(f"{prefix}-image.fits"),
            "dirty":     Path(f"{prefix}-dirty.fits"),
            "residual":  Path(f"{prefix}-residual.fits"),
            "psf":       Path(f"{prefix}-psf.fits"),
        }
        for key, path in outputs.items():
            if not path.exists():
                logger.warning("Expected WSClean output missing: %s (%s)", key, path)
        return outputs

    # ------------------------------------------------------------------ #
    # Stage 5 — Forced photometry vs. ground truth                        #
    # ------------------------------------------------------------------ #

    def _photometry(
        self,
        *,
        image_path: "Path | str",
        ground_truth: "GroundTruthRegistry",
        mjd: float,
        noise_jy_beam: float,
        box_pix: int = 5,
        flux_tolerance: float = 0.20,
    ) -> "list[SourceFluxResult]":
        """Forced photometry at injected source positions vs. ground truth.

        Uses the production ``measure_peak_box()`` from
        ``dsa110_continuum.photometry.simple_peak``.  For each source in
        the ground-truth registry, measures peak flux in a pixel box centred
        on the known position and compares to the expected flux at ``mjd``.

        Parameters
        ----------
        image_path:
            FITS image (Jy/beam) to measure.
        ground_truth:
            Registry of injected sources (positions + fluxes).
        mjd:
            Observation MJD for flux prediction via variability model.
        noise_jy_beam:
            Global noise estimate (Jy/beam) for SNR computation and the
            ``passed`` threshold.
        box_pix:
            Half-width of the pixel search box (default 5 → 11×11 box).
        flux_tolerance:
            Fractional tolerance for ``passed`` flag:
            ``|recovered - expected| / expected < flux_tolerance``.
            Default 0.20 (20 %).

        Returns
        -------
        list[SourceFluxResult]
            One entry per source in the ground-truth registry.
        """
        from astropy.io import fits as astrofits
        from astropy.wcs import WCS
        from dsa110_continuum.photometry.simple_peak import measure_peak_box

        image_path = Path(image_path)
        with astrofits.open(str(image_path)) as hdul:
            data = np.squeeze(hdul[0].data).astype(float)
            wcs  = WCS(hdul[0].header).celestial

        results: list[SourceFluxResult] = []
        for src in ground_truth.sources.values():
            expected = ground_truth.get_expected_flux(src.source_id, mjd)
            if expected is None:
                expected = src.baseline_flux_jy

            try:
                peak, snr, _xp, _yp = measure_peak_box(
                    data, wcs, src.ra_deg, src.dec_deg,
                    box_pix=box_pix, rms=noise_jy_beam,
                )
            except (ValueError, OverflowError):
                # Position projects to NaN pixels (outside WCS domain)
                peak = snr = float("nan")

            if np.isnan(peak):
                passed = False
            else:
                frac_err = abs(peak - expected) / max(abs(expected), 1e-12)
                passed   = frac_err < flux_tolerance

            results.append(SourceFluxResult(
                source_id=src.source_id,
                ra_deg=src.ra_deg,
                dec_deg=src.dec_deg,
                injected_flux_jy=expected,
                recovered_flux_jy=peak,
                snr=snr,
                passed=passed,
            ))
            logger.info(
                "  Phot %s: injected=%.3f Jy, recovered=%.3f Jy, SNR=%.1f  [%s]",
                src.source_id, expected, peak, snr,
                "PASS" if passed else "FAIL",
            )

        return results

    # ------------------------------------------------------------------ #
    # Top-level orchestrator                                               #
    # ------------------------------------------------------------------ #

    def run(
        self,
        harness: "SimulationHarness",
        *,
        n_tiles: int = 2,
        n_subbands: int = 4,
        amp_scatter: float = 0.05,
        phase_scatter_deg: float = 5.0,
        cal_flux_jy: float = 10.0,
        mjd: float = 60310.0,
    ) -> "SimulatedPipelineResult":
        """Run all five pipeline stages end-to-end on simulated data.

        Parameters
        ----------
        harness:
            Configured SimulationHarness (sky model + antenna positions).
        n_tiles:
            Number of simulated transit tiles (default 2).  Each tile is
            imaged independently; tiles are then mosaicked together.
        n_subbands:
            Subbands per tile (default 4).  Each tile produces n_subbands
            UVH5 files which are concatenated into one MS before imaging.
        amp_scatter:
            Per-antenna amplitude gain error (fractional). Default 0.05.
        phase_scatter_deg:
            Per-antenna phase gain error (degrees). Default 5.
        cal_flux_jy:
            Calibrator source flux (Jy). Default 10.
        mjd:
            Observation MJD for ground-truth flux prediction.

        Returns
        -------
        SimulatedPipelineResult
        """
        from astropy.io import fits as astrofits
        from dsa110_continuum.simulation.gain_corruption import corrupt_uvh5
        from dsa110_continuum.simulation.ground_truth import GroundTruthRegistry
        import gc

        # Cap WSClean memory to leave headroom for Python / pyuvdata data.
        # If wsclean_mem_gb is not explicitly set, use half of available RAM
        # (floor at 1.5 GB, cap at 3 GB) to avoid OOM kills in CI.
        if self.wsclean_mem_gb is None:
            try:
                import psutil
                avail_gb = psutil.virtual_memory().available / 2**30
                _mem_cap = max(1.5, min(3.0, avail_gb / 2))
            except ImportError:
                _mem_cap = 2.0
            self.wsclean_mem_gb = _mem_cap
            _reset_mem = True  # restore None after run() exits
        else:
            _reset_mem = False

        errors: list[str] = []
        tile_images: list[Path] = []
        calibration_passed = True
        imaging_passed = True
        mosaic_path: Path | None = None

        # ── Build ground-truth registry from harness sky model ─────────────
        # IMPORTANT: generate the sky ONCE and reuse it for both the registry
        # and generate_subbands().  If make_sky_model() were called a second
        # time inside generate_subbands(), the RNG would advance and produce a
        # *different* sky — so WSClean would image sources at positions the
        # photometry stage never looks for.
        registry = GroundTruthRegistry(test_run_id="simulated_pipeline")
        sky = harness.make_sky_model()
        for idx in range(sky.Ncomponents):
            ra_deg  = float(sky.ra[idx].deg)
            dec_deg = float(sky.dec[idx].deg)
            # Stokes I from sky model.  WSClean -pol I from linear XX/YY feeds
            # outputs (XX + YY) / 2 = I/2 (since V_XX = V_YY = I/2 by convention).
            # Register the *image* flux (I/2) as the ground truth so that the
            # photometry comparison is in the same units as the WSClean FITS.
            stokes_i = float(sky.stokes[0, 0, idx].value)
            flux_jy  = stokes_i / 2.0   # WSClean I-pol image units: (XX+YY)/2
            registry.register_source(
                source_id=f"SIM_S{idx:03d}",
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                baseline_flux_jy=flux_jy,
            )
        registry.register_epoch(mjd)

        # Pre-generate clean calibrator subbands (shared across tiles).
        # Each subband is corrupted inside the per-tile loop using the same
        # seeds as the target — modelling simultaneous instrument gain errors.
        cal_dir = self.work_dir / "calibrator"
        cal_dir.mkdir(parents=True, exist_ok=True)
        cal_clean_paths = [
            harness.generate_calibrator_subband(
                cal_dir / f"sb{sb:02d}",
                flux_jy=cal_flux_jy,
                subband_index=sb,
            )
            for sb in range(n_subbands)
        ]

        # ── Per-tile loop ──────────────────────────────────────────────────
        for tile_idx in range(n_tiles):
            tile_dir = self.work_dir / f"tile_{tile_idx:02d}"
            tile_dir.mkdir(parents=True, exist_ok=True)

            # Stage 1: Generate and corrupt target visibilities.
            # The same per-subband seeds are applied to the calibrator below
            # (same physical instrument, same gain errors).
            seed = 100 * (tile_idx + 1)
            uvh5_paths = harness.generate_subbands(
                output_dir=tile_dir, n_subbands=n_subbands, sky=sky
            )
            corrupted_paths = [
                corrupt_uvh5(
                    p,
                    amp_scatter=amp_scatter,
                    phase_scatter_deg=phase_scatter_deg,
                    seed=seed + i,
                )
                for i, p in enumerate(uvh5_paths)
            ]

            # Corrupt calibrator subbands with the same gain errors as the target
            # so that the Jacobi solver can recover the instrument gains.
            corrupted_cal_paths = [
                corrupt_uvh5(
                    cp,
                    amp_scatter=amp_scatter,
                    phase_scatter_deg=phase_scatter_deg,
                    seed=seed + i,
                    output_path=tile_dir / f"sim_cal_sb{i:02d}_corrupted.uvh5",
                )
                for i, cp in enumerate(cal_clean_paths)
            ]

            # Concatenate corrupted calibrator subbands (freq axis, same timestamps)
            cal_uvs_tile: list[pyuvdata.UVData] = []
            for cp in corrupted_cal_paths:
                uv = pyuvdata.UVData()
                uv.read(str(cp))
                cal_uvs_tile.append(uv)
            for uv in cal_uvs_tile:
                for key in uv.phase_center_catalog:
                    uv.phase_center_catalog[key]["cat_name"] = "SIM_TILE"
            combined_cal = cal_uvs_tile[0].fast_concat(
                cal_uvs_tile[1:], "freq", inplace=False
            ) if len(cal_uvs_tile) > 1 else cal_uvs_tile[0]
            cal_uvh5 = tile_dir / "sim_cal_combined.uvh5"
            combined_cal.write_uvh5(str(cal_uvh5))
            del combined_cal, cal_uvs_tile
            # Remove corrupted calibrator UVH5 intermediates
            for cp in corrupted_cal_paths:
                try:
                    Path(cp).unlink(missing_ok=True)
                except Exception:
                    pass
            gc.collect()

            # Concatenate corrupted subbands → single MS
            uvs: list[pyuvdata.UVData] = []
            for cp in corrupted_paths:
                uv = pyuvdata.UVData()
                uv.read(str(cp))
                uvs.append(uv)
            # Harmonise phase_center_catalog cat_name before concatenation
            for uv in uvs:
                for key in uv.phase_center_catalog:
                    uv.phase_center_catalog[key]["cat_name"] = "SIM_TILE"
            combined = uvs[0]
            for uv in uvs[1:]:
                combined = combined + uv

            ms_path = tile_dir / f"tile_{tile_idx:02d}.ms"
            combined.write_ms(str(ms_path))
            # Free visibility data from memory before calibration/imaging
            del combined, uvs
            gc.collect()
            # Remove intermediate UVH5 files to recover disk space
            import shutil
            for p_uvh5 in uvh5_paths + corrupted_paths:
                try:
                    Path(p_uvh5).unlink(missing_ok=True)
                except Exception:
                    pass

            # UVW sign convention note:
            # pyuvdata.UVData stores uvw_array with a sign convention such that
            # write_ms() negates the UVW (MS_uvw = -uvh5_uvw_array).  The
            # harness generates data with uvh5_uvw_array pointing in the
            # *opposite* direction to the standard CASA/WSClean convention, so
            # after write_ms the sign becomes correct for WSClean.  No further
            # correction is needed.

            # Stage 2: Calibrate using the per-tile corrupted calibrator
            try:
                self._calibrate(
                    target_ms=ms_path,
                    cal_uvh5=cal_uvh5,
                    cal_flux_jy=cal_flux_jy,
                    work_dir=tile_dir,
                )
            except Exception as exc:
                errors.append(f"Tile {tile_idx} calibration: {exc}")
                calibration_passed = False
                continue  # skip imaging for this tile

            # Stage 3: Image (with CLEAN deconvolution)
            try:
                img_results = self._image(ms_path=ms_path, work_dir=tile_dir)
                restored = img_results.get("restored")
                if restored and restored.exists():
                    tile_images.append(restored)
                else:
                    errors.append(f"Tile {tile_idx}: restored image missing")
                    imaging_passed = False
            except Exception as exc:
                errors.append(f"Tile {tile_idx} imaging: {exc}")
                imaging_passed = False

        # Stage 4: Mosaic tile images
        if len(tile_images) >= 2:
            try:
                mosaic_path = self._mosaic(
                    image_paths=tile_images, work_dir=self.work_dir
                )
            except Exception as exc:
                errors.append(f"Mosaicking failed: {exc}")
                # Fall back to first tile image for photometry
                mosaic_path = tile_images[0] if tile_images else None
        elif len(tile_images) == 1:
            # Only one tile — use it directly (no mosaic needed)
            mosaic_path = tile_images[0]
            logger.info("Only 1 tile image; skipping mosaic, using tile directly")
        else:
            errors.append("No tile images produced; skipping mosaic and photometry")

        # Stage 5: Forced photometry on mosaic (or best available image)
        source_results: list[SourceFluxResult] = []
        if mosaic_path and mosaic_path.exists():
            try:
                with astrofits.open(str(mosaic_path)) as hdul:
                    img_data = np.squeeze(hdul[0].data)
                finite_vals = img_data[np.isfinite(img_data)]
                noise = float(np.std(finite_vals)) if finite_vals.size > 0 else 1e-3

                source_results = self._photometry(
                    image_path=mosaic_path,
                    ground_truth=registry,
                    mjd=mjd,
                    noise_jy_beam=noise,
                    box_pix=10,           # wider search box for imperfect WCS
                    flux_tolerance=0.40,  # relaxed: CLEAN won't perfectly recover flux
                )
            except Exception as exc:
                errors.append(f"Photometry failed: {exc}")

        if _reset_mem:
            self.wsclean_mem_gb = None

        return SimulatedPipelineResult(
            work_dir=self.work_dir,
            n_tiles=n_tiles,
            calibration_passed=calibration_passed,
            imaging_passed=imaging_passed,
            mosaic_path=mosaic_path,
            source_results=source_results,
            errors=errors,
        )
