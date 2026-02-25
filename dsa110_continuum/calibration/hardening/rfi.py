"""RFI mitigation logic (Issue #8)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RFIStats:
    """RFI flagging statistics."""

    original_flagged_fraction: float
    new_flagged_fraction: float
    rfi_detected_fraction: float
    channels_flagged: int
    baselines_flagged: int
    processing_time_s: float


def preflag_rfi(
    ms_path: str,
    *,
    backend: str = "aoflagger",
    strategy: str = "tfcrop",
    aggressive: bool = False,
    aoflagger_strategy: str | None = None,
) -> RFIStats:
    """Pre-flag RFI before calibration.

        This fixes Issue #8: Inadequate RFI mitigation.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set
    backend : str, optional
        Flagging backend ('aoflagger', 'casa', or 'gpu'), by default "aoflagger"
    strategy : str, optional
        CASA flagging strategy ('tfcrop', 'rflag'), only used when backend='casa', by default "tfcrop"
    aggressive : bool, optional
        If True, use more aggressive thresholds, by default False
    aoflagger_strategy : Optional[str], optional
        Path to AOFlagger Lua strategy file, by default None
    """
    import casacore.tables as casatables
    import numpy as np

    start_time = time.time()

    # Get initial flag state
    with casatables.table(ms_path, readonly=True) as tb:
        flags = tb.getcol("FLAG")
        original_flagged = np.sum(flags) / flags.size

    # Apply flagging based on backend
    if backend == "aoflagger":
        # Use AOFlagger (preferred, faster)
        try:
            from dsa110_contimg.core.calibration.flagging import flag_rfi

            flag_rfi(
                ms_path,
                backend="aoflagger",
                strategy=aoflagger_strategy,
            )
            logger.info("Pre-flagging with AOFlagger complete")
        except Exception as e:
            logger.warning(f"AOFlagger pre-flagging failed: {e}, falling back to CASA")
            backend = "casa"  # Fall through to CASA

    if backend == "gpu":
        # Use GPU-accelerated RFI detection
        try:
            from dsa110_contimg.core.rfi import RFIDetectionConfig, gpu_rfi_detection

            threshold = 4.0 if aggressive else 5.0
            config = RFIDetectionConfig(
                threshold=threshold,
                apply_flags=True,
            )
            result = gpu_rfi_detection(ms_path, config=config)

            if result.success:
                logger.info(f"GPU pre-flagging complete: {result.flag_percent:.2f}% flagged")
            else:
                logger.warning(f"GPU pre-flagging failed: {result.error}, falling back to CASA")
                backend = "casa"  # Fall through to CASA
        except ImportError:
            logger.warning("GPU RFI module not available, falling back to CASA")
            backend = "casa"
        except Exception as e:
            logger.warning(f"GPU pre-flagging failed: {e}, falling back to CASA")
            backend = "casa"

    if backend == "casa":
        # Use CASA flagdata modes
        # Import flagdata with CASA log environment protection
        try:
            from dsa110_contimg.common.utils.casa_init import casa_log_environment

            with casa_log_environment():
                from casatasks import flagdata
        except ImportError:
            from casatasks import flagdata

        def _call_flagdata(**kwargs):
            """Call flagdata with CASA log environment protection."""
            try:
                from dsa110_contimg.common.utils.casa_init import casa_log_environment

                with casa_log_environment():
                    return flagdata(**kwargs)
            except ImportError:
                return flagdata(**kwargs)

        # Apply flagging based on strategy
        if strategy == "tfcrop":
            # Time-frequency crop: good for broadband RFI
            threshold = 3.0 if aggressive else 4.0
            _call_flagdata(
                vis=ms_path,
                mode="tfcrop",
                datacolumn="DATA",
                timecutoff=threshold,
                freqcutoff=threshold,
                action="apply",
            )

        elif strategy == "rflag":
            # R-flag: statistical outlier detection
            threshold = 4.0 if aggressive else 5.0
            _call_flagdata(
                vis=ms_path,
                mode="rflag",
                datacolumn="DATA",
                timedevscale=threshold,
                freqdevscale=threshold,
                action="apply",
            )

    # Get final flag state
    with casatables.table(ms_path, readonly=True) as tb:
        flags = tb.getcol("FLAG")
        new_flagged = np.sum(flags) / flags.size

    rfi_fraction = new_flagged - original_flagged

    return RFIStats(
        original_flagged_fraction=original_flagged,
        new_flagged_fraction=new_flagged,
        rfi_detected_fraction=rfi_fraction,
        channels_flagged=0,  # Would need more detailed analysis
        baselines_flagged=0,
        processing_time_s=time.time() - start_time,
    )


def preflag_rfi_adaptive(
    ms_path: str,
    *,
    enable_pass2: bool = True,
    max_pass2_iterations: int = 2,
    apply_spw_safeguards: bool = True,
    enable_provenance: bool = True,
    output_dir: str | None = None,
    fallback_to_simple: bool = True,
) -> RFIStats:
    """Enhanced RFI pre-flagging using adaptive two-pass system.

        This is an enhanced version of preflag_rfi() that uses the new adaptive
        RFI flagging system with QA-driven iteration and SPW safeguards.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set
    enable_pass2 : bool, optional
        Enable surgical Pass 2 refinement, by default True
    max_pass2_iterations : int, optional
        Maximum Pass 2 iterations, by default 2
    apply_spw_safeguards : bool, optional
        Enable per-SPW QA analysis and safeguards, by default True
    enable_provenance : bool, optional
        Enable provenance logging (JSON), by default True
    output_dir : Optional[str], optional
        Output directory for provenance, by default None (MS parent dir)
    fallback_to_simple : bool, optional
        If True, fall back to simple preflag_rfi on error, by default True
    """
    start_time = time.time()

    try:
        # Import adaptive RFI modules
        # Get initial flag state
        import casacore.tables as casatables
        import numpy as np

        from dsa110_contimg.core.calibration.rfi_adaptive_enhanced import (
            AdaptiveRFIConfig,
            RFIQAThresholds,
            flag_rfi_adaptive_enhanced,
        )
        from dsa110_contimg.core.calibration.spw_safeguards import (
            SPWThresholds,
        )
        from dsa110_contimg.core.calibration.spw_safeguards import (
            apply_spw_safeguards as apply_spw_analysis,
        )

        with casatables.table(ms_path, readonly=True) as tb:
            flags = tb.getcol("FLAG")
            original_flagged = np.sum(flags) / flags.size

        # Configure adaptive RFI
        rfi_config = AdaptiveRFIConfig(
            enable_pass2=enable_pass2,
            max_pass2_iterations=max_pass2_iterations,
            enable_provenance=enable_provenance,
            qa_thresholds=RFIQAThresholds(),  # Use defaults
        )

        # Set output directory
        if output_dir is None:
            output_dir = str(Path(ms_path).parent)

        # Run adaptive RFI flagging
        # Note: function accepts individual params, not a config object
        flag_rfi_adaptive_enhanced(
            ms=ms_path,
            thresholds=rfi_config.qa_thresholds,
            enable_pass2=rfi_config.enable_pass2,
            max_iterations_pass2=rfi_config.max_pass2_iterations,
            enable_provenance=rfi_config.enable_provenance,
            output_dir=output_dir,
        )

        # Optionally run SPW safeguards analysis
        if apply_spw_safeguards:
            spw_result = apply_spw_analysis(
                ms=ms_path,
                thresholds=SPWThresholds(),  # Use defaults
                output_dir=output_dir,
            )
            logger.info(
                f"SPW safeguards: {len(spw_result.decisions)} SPWs analyzed, "
                f"{sum(1 for d in spw_result.decisions.values() if d.action == 'DROP')} dropped"
            )

        # Get final flag state
        with casatables.table(ms_path, readonly=True) as tb:
            flags = tb.getcol("FLAG")
            new_flagged = np.sum(flags) / flags.size

        # Convert to RFIStats for backward compatibility
        return RFIStats(
            original_flagged_fraction=original_flagged,
            new_flagged_fraction=new_flagged,
            rfi_detected_fraction=new_flagged - original_flagged,
            channels_flagged=0,  # Detailed channel count not available
            baselines_flagged=0,
            processing_time_s=time.time() - start_time,
        )

    except Exception as e:
        logger.error(f"Adaptive RFI pre-flagging failed: {e}", exc_info=True)

        if fallback_to_simple:
            logger.warning("Falling back to simple preflag_rfi")
            return preflag_rfi(
                ms_path,
                backend="aoflagger",
                aggressive=False,
            )
        else:
            raise
