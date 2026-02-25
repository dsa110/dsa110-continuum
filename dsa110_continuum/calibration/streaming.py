# pylint: disable=no-member  # astropy.units uses dynamic attributes (deg, etc.)
"""Streaming calibration utilities for autonomous pipeline operation.

This module provides functions for calibrator detection and calibration solving
in the streaming converter context, borrowing from batch mode implementations.
"""

import structlog

logger = structlog.get_logger(__name__)


def solve_calibration_for_ms(
    ms_path: str,
    cal_field: str | None = None,
    refant: str | None = None,
    do_k: bool = False,
    catalog_path: str | None = None,
    calibrator_name: str | None = None,
) -> tuple[bool, str | None]:
    """Solve calibration for a single MS file.

    Orchestrates K, BP, and G calibration solves using existing batch mode functions.
    Auto-detects calibrator field and reference antenna if not provided.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    cal_field :
        Calibrator field name/index (auto-detected if None)
    refant :
        Reference antenna ID (auto-detected if None)
    do_k :
        If True, perform K-calibration (delay). Default False for DSA-110.
    catalog_path :
        Optional path to calibrator catalog (auto-resolved if None)
    calibrator_name :
        Expected calibrator name (e.g., "0834+555"). If provided,
        used for model lookup instead of auto-detection.

    Returns
    -------
    Tuple of (success
        bool, error_message: Optional[str])
    On success
        (True, None)
    On failure
        (False, error_message_string)

    """
    try:
        from dsa110_contimg.core.calibration.runner import run_calibrator
        from dsa110_contimg.core.calibration.refant_selection import (
            get_default_outrigger_refants,
        )
        from dsa110_contimg.core.calibration.selection import select_bandpass_from_catalog

        # Auto-detect calibrator field if not provided
        if cal_field is None:
            logger.info(f"Auto-detecting calibrator field for {ms_path}")
            try:
                # If calibrator_name is specified, use it to filter the selection
                # Otherwise, the nearest calibrator will be selected
                field_sel_str, _, _, calinfo, _ = select_bandpass_from_catalog(
                    ms_path,
                    catalog_path=catalog_path,
                    search_radius_deg=1.0,
                    window=3,
                    calibrator_name=calibrator_name,  # Pass expected calibrator
                )
                if not field_sel_str:
                    error_msg = (
                        f"Could not auto-detect calibrator field in {ms_path}. "
                        "No calibrator found in catalog within search radius."
                    )
                    logger.error(error_msg)
                    return False, error_msg
                cal_field = field_sel_str
                name, ra_deg, dec_deg, flux_jy = calinfo
                logger.info(
                    f"Auto-detected calibrator field '{cal_field}' "
                    f"for calibrator {name} (RA={ra_deg:.4f}, Dec={dec_deg:.4f})"
                )
            except Exception as e:
                error_msg = f"Failed to auto-detect calibrator field: {e}"
                logger.error(error_msg, exc_info=True)
                return False, error_msg

        # Auto-detect reference antenna if not provided
        if refant is None:
            logger.info(f"Auto-detecting reference antenna for {ms_path}")
            try:
                # Use default outrigger chain (CASA will auto-fallback)
                refant = get_default_outrigger_refants()
                logger.info(f"Using default outrigger refant chain: {refant}")
            except Exception as e:
                error_msg = f"Failed to auto-detect reference antenna: {e}"
                logger.error(error_msg, exc_info=True)
                return False, error_msg

        # Issue #8: Pre-calibration RFI flagging
        try:
            from dsa110_contimg.core.calibration.hardening import preflag_rfi

            preflag_rfi(ms_path, backend="aoflagger")
            logger.info(f"Pre-calibration RFI flagging complete for {ms_path}")
        except ImportError:
            logger.debug("preflag_rfi not available (hardening module)")
        except Exception as e:
            logger.warning(f"Pre-calibration RFI flagging failed (non-fatal): {e}")

        # Run calibration solves
        logger.info(
            f"Solving calibration for {ms_path} "
            f"(field={cal_field}, refant={refant}, do_k={do_k}, calibrator_name={calibrator_name})"
        )
        caltables = run_calibrator(
            ms_path,
            cal_field,
            refant,
            do_flagging=True,
            do_k=do_k,
            calibrator_name=calibrator_name,
        )

        if not caltables:
            error_msg = "Calibration solve completed but no calibration tables were produced"
            logger.error(error_msg)
            return False, error_msg

        # Issue #5: Automated QA assessment of calibration solutions
        qa_warnings: list[str] = []
        try:
            from dsa110_contimg.core.calibration.qa import assess_calibration_quality

            for caltable in caltables:
                qa_result = assess_calibration_quality(
                    caltable_path=caltable,
                    snr_min=3.0,  # Default minimum SNR threshold
                    flagged_max=0.5,  # Max 50% flagged
                )

                # Extract issues by severity
                errors = [i for i in qa_result.issues if i.severity == "error"]
                warnings = [i for i in qa_result.issues if i.severity == "warning"]

                if not qa_result.passed:
                    issue_msgs = [i.message for i in errors]
                    qa_warnings.append(f"QA failed for {caltable}: {', '.join(issue_msgs)}")
                    logger.warning("Calibration QA failed for %s: %s", caltable, issue_msgs)
                elif warnings:
                    warning_msgs = [i.message for i in warnings]
                    qa_warnings.extend(warning_msgs)
                    logger.info("Calibration QA warnings for %s: %s", caltable, warning_msgs)
                else:
                    # Summarize metrics from all calibration metrics
                    if qa_result.metrics:
                        median_snr = qa_result.metrics[0].median_snr or 0.0
                        flag_pct = qa_result.metrics[0].flag_fraction * 100
                        logger.debug(
                            "Calibration QA passed for %s (SNR=%.1f, flagged=%.1f%%)",
                            caltable,
                            median_snr,
                            flag_pct,
                        )
                    else:
                        logger.debug("Calibration QA passed for %s", caltable)
        except ImportError:
            logger.debug("QA module not available, skipping calibration QA")
        except Exception as qa_err:
            logger.warning("QA assessment error: %s", qa_err)

        logger.info(
            "Successfully solved calibration for %s: produced %d calibration table(s)%s",
            ms_path,
            len(caltables),
            f" (QA warnings: {len(qa_warnings)})" if qa_warnings else "",
        )
        return True, None

    except Exception as e:
        error_msg = f"Calibration solve failed for {ms_path}: {e}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg
