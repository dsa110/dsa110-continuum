"""Quality metrics computation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from dsa110_contimg.common.unified_config import settings
from dsa110_contimg.common.utils.gpu_utils import get_array_module

logger = logging.getLogger(__name__)


def compute_image_metrics(image_path: Path) -> dict[str, Any]:
    """Compute image quality metrics.

    Parameters
    ----------
    image_path : str
        Path to FITS image

    Returns
    -------
        dict
        Dict of metrics
    """
    from astropy.io import fits

    metrics = {}

    with fits.open(image_path) as hdul:
        data_cpu = np.squeeze(hdul[0].data)
        header = hdul[0].header

    xp, is_gpu = get_array_module(
        prefer_gpu=settings.gpu.prefer_gpu,
        min_elements=settings.gpu.min_array_size,
    )
    data = xp.asarray(data_cpu) if is_gpu else data_cpu

    # Valid data
    valid_mask = xp.isfinite(data)
    valid_data = data[valid_mask]

    # Basic statistics
    metrics["peak_flux"] = float(xp.max(valid_data))
    metrics["min_flux"] = float(xp.min(valid_data))
    metrics["mean_flux"] = float(xp.mean(valid_data))
    metrics["median_flux"] = float(xp.median(valid_data))
    metrics["std_flux"] = float(xp.std(valid_data))

    # Off-source noise estimate
    off_source = valid_data[valid_data < 3 * xp.median(valid_data)]
    if len(off_source) > 100:
        metrics["rms_noise"] = float(xp.std(off_source))
    else:
        metrics["rms_noise"] = float(xp.std(valid_data))

    # SNR
    rms_noise = metrics["rms_noise"]
    peak_flux = metrics["peak_flux"]
    metrics["snr"] = peak_flux / rms_noise if rms_noise > 0 else 0.0
    metrics["dynamic_range"] = peak_flux / rms_noise if rms_noise > 0 else 0.0

    # NaN fraction
    metrics["nan_fraction"] = float(xp.mean(xp.isnan(data)))

    # Image size
    metrics["image_shape"] = list(data.shape)

    # Beam info (if available)
    if "BMAJ" in header:
        metrics["beam_major_deg"] = float(header["BMAJ"])
    if "BMIN" in header:
        metrics["beam_minor_deg"] = float(header["BMIN"])
    if "BPA" in header:
        metrics["beam_pa_deg"] = float(header["BPA"])

    return metrics


def compute_ms_metrics(ms_path: Path) -> dict[str, Any]:
    """Compute MS quality metrics.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set

    Returns
    -------
        dict
        Dict of metrics
    """
    from casacore.tables import table

    metrics = {}

    # Main table stats
    with table(str(ms_path), readonly=True, ack=False) as tb:
        metrics["nrows"] = tb.nrows()

        # Sample data for statistics
        sample_size = min(1000, tb.nrows())
        data = tb.getcol("DATA", startrow=0, nrow=sample_size)
        flags = tb.getcol("FLAG", startrow=0, nrow=sample_size)
        uvw = tb.getcol("UVW", startrow=0, nrow=sample_size)
        times = tb.getcol("TIME", startrow=0, nrow=sample_size)

    # Visibility statistics
    amp = np.abs(data)
    metrics["vis_amp_mean"] = float(np.mean(amp))
    metrics["vis_amp_std"] = float(np.std(amp))
    metrics["vis_amp_max"] = float(np.max(amp))

    # Flagging
    metrics["flag_fraction"] = float(np.mean(flags))

    # Baselines
    baselines = np.linalg.norm(uvw, axis=1)
    metrics["baseline_min_m"] = float(np.min(baselines[baselines > 0]))
    metrics["baseline_max_m"] = float(np.max(baselines))

    # Time range
    metrics["time_range_s"] = float(times.max() - times.min())

    # Antenna count
    with table(str(ms_path / "ANTENNA"), readonly=True, ack=False) as tb:
        metrics["num_antennas"] = tb.nrows()

    # Spectral windows
    with table(str(ms_path / "SPECTRAL_WINDOW"), readonly=True, ack=False) as tb:
        metrics["num_spw"] = tb.nrows()
        chan_freq = tb.getcol("CHAN_FREQ")
        metrics["total_channels"] = int(chan_freq.size)
        metrics["min_freq_hz"] = float(chan_freq.min())
        metrics["max_freq_hz"] = float(chan_freq.max())
        metrics["bandwidth_hz"] = float(chan_freq.max() - chan_freq.min())

    return metrics
