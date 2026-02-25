"""
Pipeline quality wrappers for stage-level QA checks.

These functions delegate to existing QA utilities so the pipeline can call a
single module for MS, calibration, and imaging quality assessment.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from dsa110_contimg.core.qa.calibration_quality import (
    check_caltable_completeness,
    validate_caltable_quality,
)
from dsa110_contimg.core.qa.image_metrics import (
    calculate_dynamic_range,
    calculate_psf_correlation,
    calculate_residual_stats,
)

LOG = logging.getLogger(__name__)


def check_ms_after_conversion(
    ms_path: str | Path,
    *,
    quick_check_only: bool = False,
    alert_on_issues: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """Basic MS QA after conversion: existence, size, row count (if casacore available)."""
    ms_path = Path(ms_path)
    result: dict[str, Any] = {
        "ms_path": str(ms_path),
        "exists": ms_path.exists(),
        "size_bytes": ms_path.stat().st_size if ms_path.exists() else 0,
        "nrows": None,
        "has_issues": False,
        "warnings": [],
    }

    if not ms_path.exists():
        result["has_issues"] = True
        result["warnings"].append("MS not found")
        LOG.warning("Conversion QA: MS not found at %s", ms_path)
        return False, result

    if not quick_check_only:
        try:
            import casacore.tables as tb  # type: ignore

            with tb.table(str(ms_path), readonly=True) as t:
                result["nrows"] = int(t.nrows())
        except Exception as exc:  # pragma: no cover - optional dependency
            result["warnings"].append(f"Row count unavailable: {exc}")
            LOG.debug("Conversion QA: could not read MS rows: %s", exc)

    qa_passed = not result["has_issues"]
    if alert_on_issues and (result["has_issues"] or result["warnings"]):
        LOG.warning("Conversion QA alerts for %s: %s", ms_path, result)

    return qa_passed, result


def check_calibration_quality(
    caltables: str | list[str] | None = None,
    *,
    ms_path: str | Path | None = None,
    caltable_set: str | None = None,
    alert_on_issues: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """
    Calibration QA: check expected caltables and validate each table.

    Returns a dict with completeness info and per-table quality metrics.
    """
    if ms_path is None:
        raise ValueError("ms_path is required for calibration QA")

    ms_path = Path(ms_path)
    summary: dict[str, Any] = {
        "ms_path": str(ms_path),
        "caltable_set": caltable_set,
        "completeness": {},
        "tables": [],
        "has_issues": False,
    }

    try:
        completeness = check_caltable_completeness(str(ms_path), caltable_set)
        summary["completeness"] = completeness
        if completeness.get("has_issues"):
            summary["has_issues"] = True
    except Exception as exc:
        summary["has_issues"] = True
        summary["completeness_error"] = str(exc)
        LOG.warning("Calibration QA: completeness check failed: %s", exc)
        return False, summary

    existing: list[str] = []
    if caltables:
        if isinstance(caltables, (list, tuple, set)):
            existing.extend([str(p) for p in caltables])
        else:
            existing.append(str(caltables))
    existing_from_completeness = completeness.get("existing_tables", [])
    for cal_path in existing_from_completeness:
        cal_path_str = str(cal_path)
        if cal_path_str not in existing:
            existing.append(cal_path_str)

    for cal_path in existing:
        try:
            metrics = validate_caltable_quality(cal_path)
            summary["tables"].append(metrics.to_dict())
            if metrics.has_issues:
                summary["has_issues"] = True
        except Exception as exc:
            summary["has_issues"] = True
            summary["tables"].append(
                {
                    "caltable": cal_path,
                    "error": str(exc),
                    "has_issues": True,
                }
            )
            LOG.warning("Calibration QA: validation failed for %s: %s", cal_path, exc)

    qa_passed = not summary["has_issues"]
    if alert_on_issues and not qa_passed:
        LOG.warning("Calibration QA issues detected for %s: %s", ms_path, summary)

    return qa_passed, summary


def check_image_quality(
    image_path: str | Path,
    *,
    psf_path: str | Path | None = None,
    residual_path: str | Path | None = None,
    peak_flux: float | None = None,
    alert_on_issues: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """Imaging QA: compute basic metrics (dynamic range, residual stats, PSF correlation)."""
    image_path = Path(image_path)
    summary: dict[str, Any] = {
        "image_path": str(image_path),
        "dynamic_range": None,
        "psf_correlation": None,
        "residual_stats": None,
        "has_issues": False,
        "warnings": [],
    }

    if not image_path.exists():
        summary["has_issues"] = True
        summary["warnings"].append("Image not found")
        LOG.warning("Image QA: image not found at %s", image_path)
        return False, summary

    try:
        summary["dynamic_range"] = calculate_dynamic_range(image_path, peak_flux=peak_flux)
    except Exception as exc:
        summary["warnings"].append(f"Dynamic range unavailable: {exc}")
        LOG.debug("Image QA: dynamic range failed: %s", exc)

    if psf_path:
        try:
            summary["psf_correlation"] = calculate_psf_correlation(image_path, psf_path)
        except Exception as exc:
            summary["warnings"].append(f"PSF correlation unavailable: {exc}")
            LOG.debug("Image QA: PSF correlation failed: %s", exc)

    if residual_path:
        try:
            summary["residual_stats"] = calculate_residual_stats(residual_path)
        except Exception as exc:
            summary["warnings"].append(f"Residual stats unavailable: {exc}")
            LOG.debug("Image QA: residual stats failed: %s", exc)

    qa_passed = not summary["has_issues"]
    if alert_on_issues and (summary["has_issues"] or summary["warnings"]):
        LOG.warning("Image QA alerts for %s: %s", image_path, summary)

    return qa_passed, summary
