"""Time-domain synthetic data generation for multi-epoch pipeline testing.

    This module extends the single-epoch synthetic UVH5 generator to support
    multi-epoch observations with time-varying sources. It enables end-to-end
    testing of the lightcurve and transient detection pipeline stages.

    Key Features
------------
    - Generate multiple observation epochs with configurable time spacing
    - Inject variable sources using variability models (flares, ESE, periodic)
    - Maintain consistent source populations across epochs
    - Embed ground truth metadata in UVH5 files for validation
    - Compatible with production pipeline code (conversion, calibration, imaging)

    Example
-------
    >>> from dsa110_contimg.core.simulation.time_domain import generate_multi_epoch_uvh5
    >>> from dsa110_contimg.core.simulation.variability_models import FlareModel
    >>>
    >>> # Define epochs (4 observations over 1 week)
    >>> epochs = [
    ...     datetime(2025, 1, 15, 12, 0, 0),
    ...     datetime(2025, 1, 16, 12, 0, 0),
    ...     datetime(2025, 1, 17, 12, 0, 0),
    ...     datetime(2025, 1, 22, 12, 0, 0),
    ... ]
    >>>
    >>> # Inject a flare in source NVSS_J123456+420000
    >>> variability_models = {
    ...     "NVSS_J123456+420000": FlareModel(
    ...         peak_time_mjd=60300.5,  # Peak at epoch 1
    ...         rise_time_hours=6.0,
    ...         decay_time_hours=24.0,
    ...         peak_flux_jy=5.0,
    ...         baseline_flux_jy=1.0,
    ...     )
    ... }
    >>>
    >>> # Generate synthetic data for all epochs
    >>> from dsa110_contimg.common.utils import TempPaths
    >>> result = generate_multi_epoch_uvh5(
    ...     epochs=epochs,
    ...     output_dir=TempPaths.test_output("synthetic_multiepoch"),
    ...     variability_models=variability_models,
    ...     catalog_type="nvss",
    ...     region_ra_deg=188.0,
    ...     region_dec_deg=42.0,
    ... )
    >>>
    >>> print(f"Generated {len(result['epochs'])} epochs")
    >>> for epoch_data in result['epochs']:
    ...     print(f"  Epoch {epoch_data['mjd']}: {len(epoch_data['files'])} files")
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy.time import Time

from dsa110_contimg.core.simulation.source_selection import (
    CatalogRegion,
    SourceSelector,
    SyntheticSource,
)
from dsa110_contimg.core.simulation.variability_models import (
    VariabilityModel,
    compute_flux_at_time,
)

logger = logging.getLogger(__name__)


@dataclass
class EpochData:
    """Data for a single observation epoch."""

    epoch_datetime: datetime
    mjd: float
    files: list[Path]
    sources: list[SyntheticSource]
    ground_truth_file: Path | None = None


@dataclass
class MultiEpochResult:
    """Result from multi-epoch UVH5 generation."""

    epochs: list[EpochData]
    output_dir: Path
    variability_models: dict[str, VariabilityModel]
    catalog_region: CatalogRegion
    success: bool
    error_message: str | None = None


def generate_multi_epoch_uvh5(
    epochs: Sequence[datetime],
    output_dir: Path,
    variability_models: dict[str, VariabilityModel] | None = None,
    catalog_type: str = "nvss",
    region_ra_deg: float = 188.0,
    region_dec_deg: float = 42.0,
    region_radius_deg: float = 2.0,
    min_flux_mjy: float = 100.0,
    max_sources: int | None = 50,
    nants: int = 117,  # Full DSA-110 array (use 8 for fast testing)
    ntimes: int = 24,
    add_noise: bool = True,
    system_temp_k: float = 50.0,
    seed: int = 42,
    catalog_path: Path | None = None,
    pyuvsim_beam_type: str = "airy",
) -> MultiEpochResult:
    """Generate synthetic UVH5 files for multiple observation epochs.

        This function generates a complete multi-epoch dataset with time-varying
        sources. The same source catalog is used across all epochs, but individual
        source fluxes are modulated by their variability models.

    Parameters
    ----------
    epochs : sequence of datetime
        List of observation times (datetime objects)
    output_dir : Path
        Directory for all output files (organized by epoch)
    variability_models : dict of str to VariabilityModel or None, optional
        Dict mapping source_id to variability model (default is None)
    catalog_type : str, optional
        Catalog to query ("nvss", "first", "vlass", "racs") (default is "nvss")
    region_ra_deg : float, optional
        Field center RA in degrees (default is 188.0)
    region_dec_deg : float, optional
        Field center Dec in degrees (default is 42.0)
    region_radius_deg : float, optional
        Field radius in degrees (default is 2.0)
    min_flux_mjy : float, optional
        Minimum source flux in mJy (default is 100.0)
    max_sources : int or None, optional
        Maximum number of sources to include (default is 50)
    nants : int, optional
        Number of antennas (default is 63)
    ntimes : int, optional
        Number of time integrations per epoch (default is 24)
    add_noise : bool, optional
        Whether to add thermal noise (default is True)
    system_temp_k : float, optional
        System temperature for noise in Kelvin (default is 50.0)
    seed : int, optional
        Random seed for reproducibility (default is 42)
    catalog_path : Path or None, optional
        Optional path to catalog database (default is None)
    pyuvsim_beam_type : str, optional
        Beam model type for pyuvsim: "airy" or "gaussian" (default is "airy")

    Returns
    -------
        MultiEpochResult
        Object containing epoch data and metadata.

    Examples
    --------
        >>> from datetime import datetime, timedelta
        >>> from dsa110_contimg.common.utils import TempPaths
        >>> epochs = [datetime(2025, 1, 15) + timedelta(days=i) for i in range(4)]
        >>> result = generate_multi_epoch_uvh5(
        ...     epochs=epochs,
        ...     output_dir=TempPaths.test_output("test"),
        ...     catalog_type="nvss",
        ... )
        >>> assert len(result.epochs) == 4
    """
    logger.debug("[DEBUG] generate_multi_epoch_uvh5 called")
    import time as time_module

    logger.debug("[DEBUG] time module imported")

    from dsa110_contimg.core.simulation.make_synthetic_uvh5 import (
        CONFIG_DIR,
        PYUVSIM_DIR,
        build_uvdata_from_scratch,
        load_reference_layout,
        load_telescope_config,
    )
    from dsa110_contimg.core.simulation.uvdata_writer import write_uvdata_to_subbands

    logger.debug("[DEBUG] make_synthetic_uvh5 imports complete")

    if variability_models is None:
        variability_models = {}

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug("[DEBUG] output_dir created")

    logger.info("=" * 70)
    logger.info("Multi-Epoch Synthetic UVH5 Generation")
    logger.info("=" * 70)
    logger.info("  Output: %s", output_dir)
    logger.info("  Epochs: %d", len(epochs))
    logger.info("  Variable sources: %d", len(variability_models))
    logger.info("  Catalog: %s", catalog_type)
    logger.info(
        "  Region: RA=%.2f°, Dec=%.2f°, R=%.2f°",
        region_ra_deg,
        region_dec_deg,
        region_radius_deg,
    )
    logger.debug("[DEBUG] logger statements complete")

    try:
        # Select source catalog (same for all epochs)
        logger.debug("[DEBUG] Creating CatalogRegion...")
        region = CatalogRegion(
            ra_deg=region_ra_deg,
            dec_deg=region_dec_deg,
            radius_deg=region_radius_deg,
        )
        logger.debug("[DEBUG] Creating SourceSelector...")
        selector = SourceSelector(region, catalog_type, catalog_path=catalog_path)
        logger.debug("[DEBUG] Selecting sources...")
        base_sources = selector.select_sources(
            min_flux_mjy=min_flux_mjy,
            max_sources=max_sources,
        )
        logger.debug("[DEBUG] Selected %d sources", len(base_sources))

        if not base_sources:
            logger.warning("No sources found in catalog query")
            return MultiEpochResult(
                epochs=[],
                output_dir=output_dir,
                variability_models=variability_models,
                catalog_region=region,
                success=False,
                error_message="No sources found in catalog",
            )

        logger.info("  Selected %d sources from %s", len(base_sources), catalog_type.upper())

        # Load telescope configuration once
        logger.debug("[DEBUG] Loading reference layout...")
        layout_meta = load_reference_layout(CONFIG_DIR / "reference_layout.parquet")
        logger.debug("[DEBUG] Loading telescope config...")
        config = load_telescope_config(PYUVSIM_DIR / "telescope.yaml", layout_meta, "desc")
        logger.debug("[DEBUG] Telescope config loaded")

        epoch_results = []

        # Generate data for each epoch
        logger.debug("[DEBUG] Starting epoch loop (%d epochs)...", len(epochs))
        for epoch_idx, epoch_dt in enumerate(epochs):
            logger.debug("[DEBUG] Epoch %d: %s", epoch_idx, epoch_dt)
            epoch_start_time = time_module.time()

            # Convert to astropy Time and MJD
            obs_time = Time(epoch_dt, scale="utc")
            epoch_mjd = obs_time.mjd
            logger.debug("[DEBUG] Epoch MJD: %s", epoch_mjd)

            logger.info(
                "\n--- Epoch %d/%d: %s (MJD %.4f) ---",
                epoch_idx + 1,
                len(epochs),
                epoch_dt.isoformat(),
                epoch_mjd,
            )

            # Create epoch subdirectory
            epoch_dir = output_dir / f"epoch_{epoch_idx:02d}_{epoch_dt.strftime('%Y%m%d_%H%M%S')}"
            epoch_dir.mkdir(parents=True, exist_ok=True)

            # Apply variability models to update source fluxes
            epoch_sources = []
            flux_changes = []
            for source in base_sources:
                source_id = source.source_id or f"source_{source.ra_deg:.4f}_{source.dec_deg:.4f}"

                if source_id in variability_models:
                    model = variability_models[source_id]
                    new_flux_jy = compute_flux_at_time(source.flux_ref_jy, model, epoch_mjd)
                    flux_change = new_flux_jy / source.flux_ref_jy
                    flux_changes.append(
                        f"{source_id}: {source.flux_ref_jy:.2f} → {new_flux_jy:.2f} Jy ({flux_change:.2f}x)"
                    )

                    # Create modified source with new flux
                    from dataclasses import replace

                    epoch_sources.append(replace(source, flux_ref_jy=new_flux_jy))
                else:
                    # No variability - use baseline flux
                    epoch_sources.append(source)

            if flux_changes:
                logger.info("  Applied %d variability models:", len(flux_changes))
                for change in flux_changes[:5]:  # Show first 5
                    logger.info("    %s", change)
                if len(flux_changes) > 5:
                    logger.info("    ... and %d more", len(flux_changes) - 5)

            # Build UVData for this epoch
            logger.debug("[DEBUG] Building UVData (nants=%d, ntimes=%d)...", nants, ntimes)
            uv = build_uvdata_from_scratch(config, nants=nants, ntimes=ntimes, start_time=obs_time)
            logger.debug("[DEBUG] UVData built, shape=%s", uv.data_array.shape)

            # UVW coordinates are already computed in build_uvdata_from_scratch
            # Just extract them from the UVData object
            logger.debug("[DEBUG] Extracting UVW coordinates from UVData...")
            uvw_m = uv.uvw_array  # Already computed, shape (nblts, 3)

            # Generate visibilities with epoch-specific fluxes
            # Priority: pyuvsim (if requested) > GPU > CPU chunked
            logger.debug("[DEBUG] Starting visibility computation...")
            
            # Generate visibilities
            # Priority: pyuvsim (required)
            logger.debug("[DEBUG] Starting visibility computation...")
            
            # Use pyuvsim for high-precision visibility simulation
            logger.debug("[DEBUG] Using pyuvsim for visibility simulation...")
            try:
                from dsa110_contimg.core.simulation.pyuvsim_adapter import (
                    check_pyuvsim_available,
                    simulate_visibilities,
                )
                
                if not check_pyuvsim_available():
                    raise RuntimeError("pyuvsim is required for synthetic data generation")

                uv = simulate_visibilities(
                    uv, epoch_sources, beam_type=pyuvsim_beam_type, quiet=True
                )
                vis = uv.data_array
                logger.info("  Using pyuvsim for visibility simulation")
            except Exception as e:
                logger.error("pyuvsim simulation failed: %s", e)
                raise

            # Add noise if requested (use shared visibility_models for consistency)
            if add_noise:
                from dsa110_contimg.core.simulation.visibility_models import add_thermal_noise as add_thermal_noise_vis

                rng = np.random.default_rng(seed + epoch_idx)
                mean_freq_hz = (
                    np.mean(uv.freq_array) if uv.freq_array.size > 0
                    else config.reference_frequency_hz
                )
                uv.data_array = add_thermal_noise_vis(
                    uv.data_array,
                    config.integration_time_sec,
                    abs(config.channel_width_hz),
                    system_temperature_k=system_temp_k,
                    frequency_hz=float(mean_freq_hz),
                    rng=rng,
                )

            # Add epoch metadata
            uv.extra_keywords["EPOCH_IDX"] = epoch_idx
            uv.extra_keywords["EPOCH_MJD"] = epoch_mjd
            uv.extra_keywords["NVARIABLE"] = len(variability_models)

            # Write subband files
            subband_files = write_uvdata_to_subbands(
                uv,
                config,
                epoch_dir,
                obs_time,
                epoch_sources,
            )

            # Save ground truth for this epoch
            ground_truth_file = epoch_dir / "ground_truth.json"
            ground_truth = {
                "epoch_idx": epoch_idx,
                "epoch_datetime": epoch_dt.isoformat(),
                "epoch_mjd": epoch_mjd,
                "sources": [
                    {
                        "source_id": s.source_id,
                        "ra_deg": s.ra_deg,
                        "dec_deg": s.dec_deg,
                        "flux_jy": s.flux_ref_jy,
                        "has_variability": s.source_id in variability_models,
                    }
                    for s in epoch_sources
                ],
                "variability_models": {
                    source_id: model.to_dict() for source_id, model in variability_models.items()
                },
            }
            ground_truth_file.write_text(json.dumps(ground_truth, indent=2))

            elapsed = time_module.time() - epoch_start_time
            logger.info("   Generated %d subband files in %.1fs", len(subband_files), elapsed)
            logger.info("   Ground truth: %s", ground_truth_file)

            epoch_results.append(
                EpochData(
                    epoch_datetime=epoch_dt,
                    mjd=epoch_mjd,
                    files=subband_files,
                    sources=epoch_sources,
                    ground_truth_file=ground_truth_file,
                )
            )

        logger.info("\n" + "=" * 70)
        logger.info(" Multi-epoch generation complete: %d epochs", len(epoch_results))
        logger.info("=" * 70)

        return MultiEpochResult(
            epochs=epoch_results,
            output_dir=output_dir,
            variability_models=variability_models,
            catalog_region=region,
            success=True,
        )

    except Exception as e:
        logger.error("Multi-epoch generation failed: %s", e, exc_info=True)
        return MultiEpochResult(
            epochs=[],
            output_dir=output_dir,
            variability_models=variability_models or {},
            catalog_region=CatalogRegion(region_ra_deg, region_dec_deg, region_radius_deg),
            success=False,
            error_message=str(e),
        )
