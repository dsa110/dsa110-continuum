"""K-calibration delay plotting utilities.

This module is intentionally small and focused so the general-purpose
`calibration_plots.py` module does not grow unbounded.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from dsa110_contimg.core.visualization.config import FigureConfig, PlotStyle

logger = logging.getLogger(__name__)


def _setup_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")


def _find_ms_for_caltable(caltable: Path, ms_path: str | Path | None) -> Path | None:
    if ms_path is not None:
        candidate = Path(ms_path)
        return candidate if candidate.exists() else None

    ms_files = sorted(caltable.parent.glob("*.ms"))
    return ms_files[0] if ms_files else None


def _get_ref_frequency_hz(
    ms_candidate: Path | None,
    default_ref_frequency_hz: float,
) -> float:
    ref_frequency_hz = float(default_ref_frequency_hz)
    if ms_candidate is None:
        return ref_frequency_hz

    try:
        from casacore.tables import table

        with table(f"{ms_candidate}::SPECTRAL_WINDOW", readonly=True) as spw_tb:
            ref_freqs = spw_tb.getcol("REF_FREQUENCY")
            if getattr(ref_freqs, "size", 0) > 0:
                ref_frequency_hz = float(ref_freqs[0])
    except Exception:
        return ref_frequency_hz

    return ref_frequency_hz


def _read_kcal_table_columns(
    caltable: Path,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    from casacore.tables import table

    with table(str(caltable), readonly=True) as tb:
        colnames = set(tb.colnames())
        if "CPARAM" not in colnames:
            return None, None, None

        cparam = tb.getcol("CPARAM")
        flags = tb.getcol("FLAG") if "FLAG" in colnames else None
        antenna_ids = tb.getcol("ANTENNA1") if "ANTENNA1" in colnames else None

    return cparam, flags, antenna_ids


def _first_unflagged_cparam_value(
    cparam: np.ndarray,
    flags: np.ndarray | None,
    indices: np.ndarray,
) -> complex | None:
    for idx in indices:
        if flags is not None:
            try:
                if bool(np.asarray(flags[idx]).ravel()[0]):
                    continue
            except Exception:
                pass

        try:
            return cparam[idx, 0, 0]
        except Exception:
            return np.asarray(cparam[idx]).ravel()[0]

    return None


def _extract_kcal_delays_ns(
    cparam: np.ndarray,
    flags: np.ndarray | None,
    antenna_ids: np.ndarray,
    ref_frequency_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    antenna_ids = np.asarray(antenna_ids)
    delays_by_ant: dict[int, float] = {}

    for ant_id in np.unique(antenna_ids):
        ant_mask = antenna_ids == ant_id
        indices = np.where(ant_mask)[0]
        if not indices.size:
            continue

        chosen_val = _first_unflagged_cparam_value(cparam, flags, indices)
        if chosen_val is None:
            continue

        phase_rad = float(np.angle(chosen_val))
        delay_sec = phase_rad / (2.0 * np.pi * ref_frequency_hz)
        delays_by_ant[int(ant_id)] = delay_sec * 1e9

    if not delays_by_ant:
        return np.array([], dtype=int), np.array([], dtype=float)

    ants = np.array(sorted(delays_by_ant.keys()), dtype=int)
    delays_ns = np.array([delays_by_ant[a] for a in ants], dtype=float)
    return ants, delays_ns


def _save_kcal_delay_plots(
    *,
    ants: np.ndarray,
    delays_ns: np.ndarray,
    caltable_name: str,
    ref_frequency_hz: float,
    output: Path,
    config: FigureConfig,
) -> list[Path]:
    import matplotlib.pyplot as plt

    generated: list[Path] = []

    fig, axis = plt.subplots(figsize=config.figsize)
    axis.plot(ants, delays_ns, ".", alpha=0.8)
    axis.axhline(float(np.median(delays_ns)), linestyle="--", linewidth=1)
    axis.set_xlabel("Antenna ID")
    axis.set_ylabel("Estimated delay (ns)")
    axis.set_title(f"K-cal delays: {caltable_name} (ref={ref_frequency_hz / 1e6:.1f} MHz)")
    fig.tight_layout()

    out_path = output / f"{caltable_name}_delay.png"
    fig.savefig(out_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    generated.append(out_path)

    fig, axis = plt.subplots(figsize=config.figsize)
    axis.hist(delays_ns, bins=min(50, max(5, int(np.sqrt(len(delays_ns))))))
    axis.set_xlabel("Estimated delay (ns)")
    axis.set_ylabel("Count")
    axis.set_title(f"K-cal delay distribution: {caltable_name}")
    fig.tight_layout()

    hist_path = output / f"{caltable_name}_delay_hist.png"
    fig.savefig(hist_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    generated.append(hist_path)

    return generated


def plot_kcal_delays(
    caltable: str | Path,
    ms_path: str | Path | None = None,
    output: str | Path | None = None,
    config: FigureConfig | None = None,
    default_ref_frequency_hz: float = 1400e6,
) -> list[Path]:
    """Plot K-calibration (delay) solutions from a CASA calibration table.

    This reads complex per-antenna solutions from the caltable's CPARAM column
    and estimates delays from the phase at a reference frequency.

    Returns a list of generated PNG file paths.
    """
    _setup_matplotlib()

    if config is None:
        config = FigureConfig(style=PlotStyle.QUICKLOOK)

    caltable = Path(caltable)
    if output is None:
        output = caltable.parent / f"{caltable.name}_plots"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    ms_candidate = _find_ms_for_caltable(caltable, ms_path)
    ref_frequency_hz = _get_ref_frequency_hz(ms_candidate, default_ref_frequency_hz)

    try:
        cparam, flags, antenna_ids = _read_kcal_table_columns(caltable)
    except ImportError as exc:
        raise ImportError("casacore is required for plot_kcal_delays") from exc
    if cparam is None or antenna_ids is None:
        logger.info("K-cal table %s missing CPARAM/ANTENNA1; skipping", caltable)
        return []

    ants, delays_ns = _extract_kcal_delays_ns(cparam, flags, antenna_ids, ref_frequency_hz)
    if not ants.size:
        return []

    return _save_kcal_delay_plots(
        ants=ants,
        delays_ns=delays_ns,
        caltable_name=caltable.name,
        ref_frequency_hz=ref_frequency_hz,
        output=output,
        config=config,
    )
