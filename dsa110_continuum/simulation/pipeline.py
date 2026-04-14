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
    ) -> None:
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.wsclean_bin = wsclean_bin
        self.niter = niter
        self.cell_arcsec = cell_arcsec
        self.image_size = image_size

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
        import casacore.tables as ct

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
                numerator[i]   += vis * np.conj(gains[j]) / model_amp
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
            for p in range(n_pol):
                corrected[:, :, p] /= safe_denom

            # Add CORRECTED_DATA column if it doesn't exist
            if "CORRECTED_DATA" not in t.colnames():
                from casacore.tables import makearrcoldesc, maketabdesc
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
