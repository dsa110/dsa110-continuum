"""Helper functions for writing UVData objects to subband files."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
from astropy.time import Time
from pyuvdata import UVData

from dsa110_contimg.core.simulation.make_synthetic_uvh5 import TelescopeConfig
from dsa110_contimg.core.simulation.source_selection import SyntheticSource


def write_uvdata_to_subbands(
    uv: UVData,
    config: TelescopeConfig,
    output_dir: Path,
    obs_time: Time,
    sources: Sequence[SyntheticSource] | None = None,
) -> list[Path]:
    """Write a populated UVData object to 16 subband HDF5 files.

    This is a simplified interface for when you already have a complete UVData
    object with visibilities populated. It splits the data across 16 subbands
    and writes them to disk.

    Parameters
    ----------
    uv :
        Complete UVData object with data_array already populated
    config :
        Telescope configuration
    output_dir :
        Output directory for subband files
    obs_time :
        Observation time
    sources :
        Optional list of sources for metadata

    Returns
    -------
        List of paths to the 16 created HDF5 files

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    subband_files = []

    nfreqs_per_subband = config.channels_per_subband
    channel_width_hz = config.channel_width_hz
    channel_width_signed = -abs(channel_width_hz)  # DSA-110 uses negative widths

    for sb in range(16):
        # Select frequency slice for this subband
        start_ch = sb * nfreqs_per_subband
        end_ch = start_ch + nfreqs_per_subband

        # Extract frequency slice for this subband
        # Instead of using uv.select() which has shape issues, manually extract the slice
        # and build a new UVData object from scratch with correct dimensions
        full_data = uv.data_array
        full_flags = uv.flag_array
        full_nsample = uv.nsample_array

        # Handle both (nblts, nspws, nfreqs, npols) and (nblts, nfreqs, npols) shapes
        if full_data.ndim == 4:
            subband_data = full_data[:, :, start_ch:end_ch, :]
        else:
            subband_data = full_data[:, start_ch:end_ch, :]

        if full_flags.ndim == 4:
            subband_flags = full_flags[:, :, start_ch:end_ch, :]
        else:
            subband_flags = full_flags[:, start_ch:end_ch, :]

        if full_nsample.ndim == 4:
            subband_nsample = full_nsample[:, :, start_ch:end_ch, :]
        else:
            subband_nsample = full_nsample[:, start_ch:end_ch, :]

        # Create new UVData with subband metadata
        uv_sb = uv.copy()

        # Assign sliced data arrays to the copy
        # Squeeze nspws=1 dimension if present
        if subband_data.ndim == 4:
            uv_sb.data_array = subband_data.squeeze(axis=1)
            uv_sb.flag_array = subband_flags.squeeze(axis=1)
            uv_sb.nsample_array = subband_nsample.squeeze(axis=1)
        else:
            uv_sb.data_array = subband_data
            uv_sb.flag_array = subband_flags
            uv_sb.nsample_array = subband_nsample

        # Recompute per-subband frequency axis and widths to enforce sign convention
        subband_center_hz = config.freq_min_hz + (sb + 0.5) * (
            nfreqs_per_subband * channel_width_hz
        )
        freq_array = np.arange(nfreqs_per_subband) * channel_width_hz + (
            subband_center_hz - (nfreqs_per_subband / 2.0) * channel_width_hz
        )
        uv_sb.freq_array = freq_array.reshape(1, -1)
        uv_sb._Nfreqs.value = nfreqs_per_subband
        uv_sb._Nspws.value = 1
        uv_sb.spw_array = np.array([0], dtype=int)
        # pyuvdata expects channel_width to be 1D (Nfreqs,) even when freq_array is 2D
        uv_sb.channel_width = np.full(nfreqs_per_subband, channel_width_signed, dtype=np.float64)
        uv_sb.flex_spw_id_array = np.zeros(nfreqs_per_subband, dtype=int)

        # Add metadata
        uv_sb.extra_keywords["synthetic"] = True
        uv_sb.extra_keywords["subband_index"] = sb
        if sources:
            uv_sb.extra_keywords["nsources"] = len(sources)

        # Ensure blt_order is set so combined subbands do not emit pyuvdata warnings
        uv_sb.blt_order = ("time", "baseline")

        # Write file
        anchor_str = obs_time.strftime("%Y-%m-%dT%H:%M:%S")
        filename = f"{anchor_str}_sb{sb:02d}.hdf5"
        output_path = output_dir / filename
        uv_sb.write_uvh5(output_path, run_check=False, clobber=True)

        # Post-process header to ensure per-subband metadata matches the sliced data
        with h5py.File(output_path, "a") as handle:
            header = handle["Header"]
            header["Nfreqs"][...] = nfreqs_per_subband

            if "flex_spw_id_array" in header:
                del header["flex_spw_id_array"]
            header.create_dataset(
                "flex_spw_id_array",
                data=np.zeros(nfreqs_per_subband, dtype=np.int64),
            )

            if "channel_width" in header:
                del header["channel_width"]
            # Store as scalar (shape ()) to match real DSA-110 files
            header.create_dataset(
                "channel_width",
                data=channel_width_signed,
                dtype=np.float64,
            )

            # Ensure freq_array is stored with the per-subband values
            if "freq_array" in header:
                del header["freq_array"]
            header.create_dataset("freq_array", data=freq_array.reshape(1, -1))

        subband_files.append(output_path)

    return subband_files
