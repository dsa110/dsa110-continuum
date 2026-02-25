"""Production self-calibration routine orchestration.

This module provides a lightweight, explicit self-calibration loop that:
1) runs WSClean imaging with ``-save-model`` to populate MODEL_DATA,
2) solves gains with CASA ``gaincal`` (phase first, optional amp+phase),
3) applies cumulative tables with CASA ``applycal``.

References
----------
- knowledge/SELFCAL_DESIGN.md
- knowledge/wsclean_docs.md
- knowledge/casa_calibration.md
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SelfCalRoutineError(RuntimeError):
    """Raised when the self-calibration routine fails."""


@dataclass(frozen=True)
class SelfCalIterationConfig:
    """Per-iteration self-cal parameters."""

    solint: str
    calmode: str  # "p" or "ap"
    auto_mask_sigma: float
    minsnr: float = 3.0


@dataclass
class SelfCalRoutineConfig:
    """Configuration for self-calibration routine."""

    field: str = "0"
    refant: str = "0"
    wsclean_path: str = "wsclean"
    image_size: tuple[int, int] = (4096, 4096)
    pixel_scale: str = "2.5asec"
    niter: int = 50_000
    mgain: float = 0.9
    auto_threshold_sigma: float = 1.0
    weighting: str | None = None
    robust: float | None = None
    wsclean_threads: int | None = None
    wsclean_mem_gb: float | None = None
    gaincal_combine: str = ""
    applycal_interp: list[str] = dc_field(default_factory=lambda: ["linear"])
    applycal_calwt: bool = False
    applycal_flagbackup: bool = True
    phase_iterations: list[SelfCalIterationConfig] = dc_field(
        default_factory=lambda: [
            SelfCalIterationConfig(solint="inf", calmode="p", auto_mask_sigma=5.0),
            SelfCalIterationConfig(solint="60s", calmode="p", auto_mask_sigma=4.0),
        ]
    )
    do_amp_phase: bool = False
    amp_phase_iteration: SelfCalIterationConfig = dc_field(
        default_factory=lambda: SelfCalIterationConfig(
            solint="int", calmode="ap", auto_mask_sigma=3.0, minsnr=5.0
        )
    )


@dataclass(frozen=True)
class SelfCalIterationResult:
    """Result for one iteration."""

    iteration: int
    calmode: str
    solint: str
    image_prefix: str
    gaintable: str


@dataclass(frozen=True)
class SelfCalRoutineResult:
    """Final self-calibration routine output."""

    ms_path: str
    output_dir: str
    applied_gaintables: list[str]
    iterations: list[SelfCalIterationResult]


def _ensure_exists(path: str, kind: str) -> None:
    if not os.path.exists(path):
        raise SelfCalRoutineError(f"{kind} does not exist: {path}")


def _run_command(cmd: list[str], *, cwd: str | None = None) -> None:
    logger.info("Running command: %s", " ".join(cmd))
    env = os.environ.copy()
    # WSClean + OpenBLAS may oversubscribe CPU threads if unset; keep this explicit.
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        if proc.stdout:
            logger.debug("Command stdout:\n%s", proc.stdout)
        if proc.stderr:
            logger.debug("Command stderr:\n%s", proc.stderr)
    except FileNotFoundError as e:
        raise SelfCalRoutineError(f"Required executable not found: {cmd[0]}") from e
    except subprocess.CalledProcessError as e:
        raise SelfCalRoutineError(
            f"Command failed with exit code {e.returncode}: {' '.join(cmd)}\n"
            f"stdout:\n{e.stdout}\n"
            f"stderr:\n{e.stderr}"
        ) from e


def _build_wsclean_command(
    *,
    ms_path: str,
    image_prefix: str,
    data_column: str,
    iteration_cfg: SelfCalIterationConfig,
    cfg: SelfCalRoutineConfig,
) -> list[str]:
    cmd = [
        cfg.wsclean_path,
        "-name",
        image_prefix,
        "-size",
        str(cfg.image_size[0]),
        str(cfg.image_size[1]),
        "-scale",
        cfg.pixel_scale,
        "-niter",
        str(cfg.niter),
        "-mgain",
        str(cfg.mgain),
        "-auto-mask",
        str(iteration_cfg.auto_mask_sigma),
        "-auto-threshold",
        str(cfg.auto_threshold_sigma),
        "-data-column",
        data_column,
        "-save-model",
    ]

    if cfg.weighting:
        cmd.extend(["-weight", cfg.weighting])
        if cfg.weighting.lower() == "briggs" and cfg.robust is not None:
            cmd.append(str(cfg.robust))
    if cfg.wsclean_threads is not None and cfg.wsclean_threads > 0:
        cmd.extend(["-j", str(cfg.wsclean_threads)])
    if cfg.wsclean_mem_gb is not None and cfg.wsclean_mem_gb > 0:
        cmd.extend(["-mem", str(cfg.wsclean_mem_gb)])

    cmd.append(ms_path)
    return cmd


def _remove_existing_caltable(caltable: str) -> None:
    path = Path(caltable)
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _call_gaincal_guarded(**kwargs: Any) -> None:
    """Call CASA gaincal with guarded import.

    Prefer direct casatasks import when available. Fall back to CASAService.
    """
    try:
        from casatasks import gaincal  # type: ignore

        gaincal(**kwargs)
        return
    except ImportError:
        logger.debug("casatasks.gaincal unavailable; falling back to CASAService")

    try:
        from dsa110_contimg.core.calibration.casa_service import CASAService

        CASAService().gaincal(**kwargs)
    except Exception as e:  # noqa: BLE001
        raise SelfCalRoutineError(f"CASA gaincal invocation failed: {e}") from e


def _call_applycal_guarded(**kwargs: Any) -> None:
    """Call CASA applycal with guarded import.

    Prefer direct casatasks import when available. Fall back to CASAService.
    """
    try:
        from casatasks import applycal  # type: ignore

        applycal(**kwargs)
        return
    except ImportError:
        logger.debug("casatasks.applycal unavailable; falling back to CASAService")

    try:
        from dsa110_contimg.core.calibration.casa_service import CASAService

        CASAService().applycal(**kwargs)
    except Exception as e:  # noqa: BLE001
        raise SelfCalRoutineError(f"CASA applycal invocation failed: {e}") from e


def _run_gaincal(
    *,
    ms_path: str,
    field: str,
    refant: str,
    iteration_cfg: SelfCalIterationConfig,
    gaintable_out: str,
    gaincal_combine: str,
) -> str:
    _remove_existing_caltable(gaintable_out)

    kwargs: dict[str, Any] = {
        "vis": ms_path,
        "caltable": gaintable_out,
        "field": field,
        "refant": refant,
        "calmode": iteration_cfg.calmode,
        "solint": iteration_cfg.solint,
        "minsnr": iteration_cfg.minsnr,
        "gaintable": [],
        "parang": False,
    }
    if gaincal_combine:
        kwargs["combine"] = gaincal_combine

    logger.info(
        "Running gaincal: mode=%s, solint=%s, minsnr=%.2f, out=%s",
        iteration_cfg.calmode,
        iteration_cfg.solint,
        iteration_cfg.minsnr,
        gaintable_out,
    )
    _call_gaincal_guarded(**kwargs)

    if not os.path.exists(gaintable_out):
        raise SelfCalRoutineError(
            f"gaincal completed but caltable was not created: {gaintable_out}"
        )
    return gaintable_out


def _run_applycal(
    *,
    ms_path: str,
    field: str,
    gaintables: list[str],
    cfg: SelfCalRoutineConfig,
) -> None:
    logger.info("Applying %d cumulative self-cal table(s)", len(gaintables))
    _call_applycal_guarded(
        vis=ms_path,
        field=field,
        gaintable=gaintables,
        interp=cfg.applycal_interp,
        calwt=cfg.applycal_calwt,
        flagbackup=cfg.applycal_flagbackup,
    )


def run_selfcal_routine(
    ms_path: str,
    output_dir: str,
    config: SelfCalRoutineConfig | None = None,
) -> SelfCalRoutineResult:
    """Run iterative self-calibration on a Measurement Set.

    Parameters
    ----------
    ms_path
        Input Measurement Set path.
    output_dir
        Directory for self-cal imaging products and gain tables.
    config
        Optional routine configuration.
    """
    cfg = config or SelfCalRoutineConfig()
    ms_path_abs = os.path.abspath(ms_path)
    out_dir_abs = os.path.abspath(output_dir)

    _ensure_exists(ms_path_abs, "Measurement Set")
    os.makedirs(out_dir_abs, exist_ok=True)

    iterations_cfg = list(cfg.phase_iterations)
    if cfg.do_amp_phase:
        iterations_cfg.append(cfg.amp_phase_iteration)

    if not iterations_cfg:
        raise SelfCalRoutineError("No self-cal iterations configured")

    logger.info(
        "Starting self-cal routine: ms=%s, output_dir=%s, iterations=%d",
        ms_path_abs,
        out_dir_abs,
        len(iterations_cfg),
    )

    applied_gaintables: list[str] = []
    results: list[SelfCalIterationResult] = []

    for idx, iteration_cfg in enumerate(iterations_cfg, start=1):
        is_first = idx == 1
        data_column = "DATA" if is_first else "CORRECTED_DATA"
        image_prefix = os.path.join(out_dir_abs, f"selfcal_iter{idx:02d}")
        gaintable = os.path.join(out_dir_abs, f"selfcal_iter{idx:02d}.gcal")

        logger.info(
            "Self-cal iteration %d/%d: data_column=%s, calmode=%s, solint=%s",
            idx,
            len(iterations_cfg),
            data_column,
            iteration_cfg.calmode,
            iteration_cfg.solint,
        )

        wsclean_cmd = _build_wsclean_command(
            ms_path=ms_path_abs,
            image_prefix=image_prefix,
            data_column=data_column,
            iteration_cfg=iteration_cfg,
            cfg=cfg,
        )
        _run_command(wsclean_cmd)

        solved_table = _run_gaincal(
            ms_path=ms_path_abs,
            field=cfg.field,
            refant=cfg.refant,
            iteration_cfg=iteration_cfg,
            gaintable_out=gaintable,
            gaincal_combine=cfg.gaincal_combine,
        )
        applied_gaintables.append(solved_table)

        _run_applycal(
            ms_path=ms_path_abs,
            field=cfg.field,
            gaintables=applied_gaintables,
            cfg=cfg,
        )

        results.append(
            SelfCalIterationResult(
                iteration=idx,
                calmode=iteration_cfg.calmode,
                solint=iteration_cfg.solint,
                image_prefix=image_prefix,
                gaintable=solved_table,
            )
        )

    logger.info(
        "Self-cal routine completed successfully with %d applied table(s)",
        len(applied_gaintables),
    )

    return SelfCalRoutineResult(
        ms_path=ms_path_abs,
        output_dir=out_dir_abs,
        applied_gaintables=applied_gaintables,
        iterations=results,
    )


__all__ = [
    "SelfCalIterationConfig",
    "SelfCalIterationResult",
    "SelfCalRoutineConfig",
    "SelfCalRoutineError",
    "SelfCalRoutineResult",
    "run_selfcal_routine",
]
