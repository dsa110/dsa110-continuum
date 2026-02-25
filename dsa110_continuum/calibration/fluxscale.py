"""
Flux Calibration Bootstrap using CASA fluxscale.

This module provides absolute flux calibration for DSA-110 data by bootstrapping
the VLA flux scale from primary calibrators (3C286, 3C48, 3C147, 3C138) to
secondary/transfer calibrators.

Notes
-----
Workflow:

1. setjy: Set MODEL_DATA for primary flux calibrator using Perley-Butler 2017
2. gaincal: Solve gains on primary calibrator (with MODEL_DATA set)
3. gaincal: Solve gains on secondary calibrator (without flux scale)
4. fluxscale: Transfer flux scale from primary → secondary calibrator gains
5. applycal: Apply flux-scaled gains to target data

References
----------
- Perley & Butler (2017), ApJS 230, 7: VLA flux density scale
- CASA documentation: fluxscale task
- VLA calibrator manual: https://www.vla.nrao.edu/astro/calib/manual/

Examples
--------
>>> from dsa110_contimg.core.calibration.fluxscale import (
...     bootstrap_flux_scale,
...     is_primary_flux_calibrator,
...     set_model_primary_calibrator,
... )
>>>
>>> # Check if calibrator is primary
>>> is_primary_flux_calibrator("3C286")  # True
>>> is_primary_flux_calibrator("0834+555")  # False
>>>
>>> # Full bootstrap workflow
>>> result = bootstrap_flux_scale(
...     ms_primary="/path/to/3c286.ms",
...     ms_secondary="/path/to/0834+555.ms",
...     primary_name="3C286",
...     secondary_name="0834+555",
...     refant="104,105,106",
...     output_dir="/path/to/caltables",
... )
>>> print(f"Bootstrapped flux: {result.derived_flux_jy:.3f} Jy")
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from dsa110_contimg.core.calibration.casa_service import CASAService
from dsa110_contimg.common.utils.casa_init import ensure_casa_path

ensure_casa_path()

logger = logging.getLogger(__name__)

# =============================================================================
# Primary Flux Calibrator Definitions
# =============================================================================

# Primary VLA flux calibrators with Perley-Butler 2017 models in CASA
# These are the ONLY sources that can be used with setjy standard="Perley-Butler 2017"
# Format: name -> (alt_names, typical_flux_1400MHz_Jy, spectral_index, notes)
PRIMARY_FLUX_CALIBRATORS: dict[str, dict[str, Any]] = {
    "3C286": {
        "alt_names": ["J1331+3030", "1331+305", "1328+307"],
        "flux_1400mhz_jy": 14.86,
        "spectral_index": -0.467,
        "ra_j2000": "13h31m08.288s",
        "dec_j2000": "+30d30m32.96s",
        "notes": "Primary flux calibrator, highly polarized (~10%)",
    },
    "3C48": {
        "alt_names": ["J0137+3309", "0134+329", "0137+331"],
        "flux_1400mhz_jy": 16.23,
        "spectral_index": -0.491,
        "ra_j2000": "01h37m41.299s",
        "dec_j2000": "+33d09m35.13s",
        "notes": "Primary flux calibrator, slightly resolved",
    },
    "3C147": {
        "alt_names": ["J0542+4951", "0538+498", "0542+498"],
        "flux_1400mhz_jy": 22.45,
        "spectral_index": -0.518,
        "ra_j2000": "05h42m36.138s",
        "dec_j2000": "+49d51m07.23s",
        "notes": "Primary flux calibrator, compact",
    },
    "3C138": {
        "alt_names": ["J0521+1638", "0518+165", "0521+166"],
        "flux_1400mhz_jy": 8.36,
        "spectral_index": -0.433,
        "ra_j2000": "05h21m09.886s",
        "dec_j2000": "+16d38m22.05s",
        "notes": "Primary flux calibrator, polarized",
    },
    "3C295": {
        "alt_names": ["J1411+5212", "1409+524"],
        "flux_1400mhz_jy": 22.0,
        "spectral_index": -0.58,
        "ra_j2000": "14h11m20.647s",
        "dec_j2000": "+52d12m09.14s",
        "notes": "Primary flux calibrator, resolved double",
    },
    "3C196": {
        "alt_names": ["J0813+4813", "0809+483"],
        "flux_1400mhz_jy": 12.0,
        "spectral_index": -0.79,
        "ra_j2000": "08h13m36.033s",
        "dec_j2000": "+48d13m02.64s",
        "notes": "Primary flux calibrator",
    },
}


def is_primary_flux_calibrator(name: str) -> bool:
    """Check if a calibrator is a VLA primary flux calibrator.

    Primary flux calibrators have accurately known flux densities from the
    Perley-Butler 2017 scale and can be used with CASA setjy.

    Parameters
    ----------
    name :
        Calibrator name (e.g., "3C286", "J1331+3030")

    Returns
    -------
        True if the calibrator is a primary flux calibrator

    """
    name_upper = name.upper().strip()

    for primary_name, info in PRIMARY_FLUX_CALIBRATORS.items():
        if name_upper == primary_name.upper():
            return True
        for alt in info.get("alt_names", []):
            if name_upper == alt.upper():
                return True

    return False


def get_primary_calibrator_info(name: str) -> dict[str, Any] | None:
    """Get information about a primary flux calibrator.

    Parameters
    ----------
    name :
        Calibrator name

    Returns
    -------
        Dictionary with calibrator info, or None if not a primary calibrator

    """
    name_upper = name.upper().strip()

    for primary_name, info in PRIMARY_FLUX_CALIBRATORS.items():
        if name_upper == primary_name.upper():
            return {"canonical_name": primary_name, **info}
        for alt in info.get("alt_names", []):
            if name_upper == alt.upper():
                return {"canonical_name": primary_name, **info}

    return None


def list_primary_flux_calibrators() -> list[str]:
    """List all primary flux calibrators.

    Returns
    -------
        List of primary calibrator names

    """
    return list(PRIMARY_FLUX_CALIBRATORS.keys())


# =============================================================================
# Data Classes for Results
# =============================================================================


@dataclass
class SetjyResult:
    """Result from setjy task."""

    ms_path: str
    field: str
    calibrator_name: str
    standard: str
    model_flux_jy: float
    spw_fluxes: dict[str, float] = None  # type: ignore
    success: bool = True
    error_message: str | None = None

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.spw_fluxes is None:
            self.spw_fluxes = {}


@dataclass
class FluxscaleResult:
    """Result from fluxscale task."""

    caltable_in: str
    fluxtable_out: str
    reference_field: str
    transfer_fields: list[str]
    derived_fluxes: dict[str, float] = None  # type: ignore
    spectral_indices: dict[str, float] = None  # type: ignore
    fit_reffreq_hz: float | None = None
    success: bool = True
    error_message: str | None = None

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.derived_fluxes is None:
            self.derived_fluxes = {}
        if self.spectral_indices is None:
            self.spectral_indices = {}


@dataclass
class FluxBootstrapResult:
    """Complete result from flux scale bootstrap workflow."""

    primary_name: str
    secondary_name: str
    primary_flux_jy: float
    derived_flux_jy: float
    flux_ratio: float
    spectral_index: float | None
    reference_freq_hz: float
    gain_table_primary: str | None
    gain_table_secondary: str | None
    flux_table: str | None
    setjy_result: SetjyResult | None = None
    fluxscale_result: FluxscaleResult | None = None
    success: bool = True
    error_message: str | None = None
    timestamp: str = None  # type: ignore

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "primary_name": self.primary_name,
            "secondary_name": self.secondary_name,
            "primary_flux_jy": self.primary_flux_jy,
            "derived_flux_jy": self.derived_flux_jy,
            "flux_ratio": self.flux_ratio,
            "spectral_index": self.spectral_index,
            "reference_freq_hz": self.reference_freq_hz,
            "gain_table_primary": self.gain_table_primary,
            "gain_table_secondary": self.gain_table_secondary,
            "flux_table": self.flux_table,
            "success": self.success,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
        }


# =============================================================================
# Core Functions
# =============================================================================


def set_model_primary_calibrator(
    ms_path: str,
    field_sel: str = "",
    calibrator_name: str | None = None,
    standard: str = "Perley-Butler 2017",
    spw: str = "",
    scalebychan: bool = True,
    usescratch: bool = True,
) -> SetjyResult:
    """Set MODEL_DATA for a primary flux calibrator using setjy.

    This uses CASA's built-in flux density models from Perley-Butler 2017
    to set the MODEL_DATA column with accurate flux values.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    field_sel :
        Field selection (default: "" for all fields)
    calibrator_name :
        Calibrator name (e.g., "3C286"). If None, auto-detect
        from MS field names.
    standard :
        Flux density standard (default: "Perley-Butler 2017")
    spw :
        Spectral window selection (default: "" for all)
    scalebychan :
        Scale flux by channel frequency (default: True)
    usescratch :
        Write to scratch MODEL_DATA column (default: True)

    Returns
    -------
        SetjyResult with model flux information

    Raises
    ------
    ValueError
        If calibrator is not a primary flux calibrator
    RuntimeError
        If setjy fails

    """
    # Validate calibrator is primary
    if calibrator_name:
        info = get_primary_calibrator_info(calibrator_name)
        if info is None:
            raise ValueError(
                f"'{calibrator_name}' is not a primary flux calibrator. "
                f"Primary calibrators are: {list_primary_flux_calibrators()}"
            )
        canonical_name = info["canonical_name"]
    else:
        # Auto-detect from field names
        canonical_name = _detect_primary_calibrator_in_ms(ms_path)
        if canonical_name is None:
            raise ValueError(
                "No primary flux calibrator found in MS field names. "
                "Specify calibrator_name explicitly."
            )

    logger.info(
        "Setting MODEL_DATA for primary calibrator %s using %s",
        canonical_name,
        standard,
    )

    try:
        service = CASAService()

        # Run setjy
        result_dict = service.setjy(
            vis=ms_path,
            field=field_sel if field_sel else canonical_name,
            standard=standard,
            spw=spw,
            scalebychan=scalebychan,
            usescratch=usescratch,
        )

        # Parse setjy output to get flux values
        # setjy returns a dict with spw fluxes
        spw_fluxes = {}
        total_flux = 0.0

        if isinstance(result_dict, dict):
            # Extract flux values from nested dict structure
            for key, value in result_dict.items():
                if isinstance(value, dict) and "fluxd" in value:
                    # fluxd is [I, Q, U, V] - take Stokes I
                    flux_i = value["fluxd"][0]
                    spw_fluxes[key] = flux_i
                    if flux_i > 0:
                        total_flux = flux_i  # Use last valid

        if total_flux == 0:
            # Fallback: use expected flux from our table
            info = PRIMARY_FLUX_CALIBRATORS.get(canonical_name, {})
            total_flux = info.get("flux_1400mhz_jy", 1.0)

        logger.info(
            "setjy complete: %s model flux = %.3f Jy",
            canonical_name,
            total_flux,
        )

        return SetjyResult(
            ms_path=ms_path,
            field=field_sel if field_sel else canonical_name,
            calibrator_name=canonical_name,
            standard=standard,
            model_flux_jy=total_flux,
            spw_fluxes=spw_fluxes,
            success=True,
        )

    except Exception as exc:
        logger.error("setjy failed: %s", exc)
        return SetjyResult(
            ms_path=ms_path,
            field=field_sel,
            calibrator_name=canonical_name,
            standard=standard,
            model_flux_jy=0.0,
            success=False,
            error_message=str(exc),
        )


def run_fluxscale(
    caltable: str,
    fluxtable: str,
    reference: str | list[str],
    transfer: str | list[str],
    refspwmap: list[int] | None = None,
    incremental: bool = False,
    fitorder: int = 1,
) -> FluxscaleResult:
    """Run CASA fluxscale to transfer flux scale from reference to transfer fields.

    The fluxscale task computes flux densities for transfer calibrators by
    comparing their gains to a reference calibrator with known flux density.

    Parameters
    ----------
    caltable :
        Input calibration table (from gaincal with all calibrators)
    fluxtable :
        Output flux-scaled calibration table
    reference :
        Reference field(s) - primary flux calibrator(s)
    transfer :
        Transfer field(s) - secondary calibrator(s) to bootstrap
    refspwmap :
        SPW mapping for reference (default: identity)
    incremental :
        Produce incremental table (default: False)
    fitorder :
        Polynomial order for spectral index fit (default: 1)

    Returns
    -------
        FluxscaleResult with derived fluxes and spectral indices

    """
    # Normalize to lists
    if isinstance(reference, str):
        reference = [reference]
    if isinstance(transfer, str):
        transfer = [transfer]

    logger.info(
        "Running fluxscale: reference=%s, transfer=%s",
        reference,
        transfer,
    )

    try:
        service = CASAService()

        result_dict = service.fluxscale(
            vis="",  # Not needed when using caltable
            caltable=caltable,
            fluxtable=fluxtable,
            reference=reference,
            transfer=transfer,
            refspwmap=refspwmap if refspwmap else [],
            incremental=incremental,
            fitorder=fitorder,
        )

        # Parse fluxscale output
        derived_fluxes = {}
        spectral_indices = {}
        fit_reffreq = None

        if isinstance(result_dict, dict):
            for field_key, field_data in result_dict.items():
                if isinstance(field_data, dict):
                    # Extract flux density (first Stokes I value across SPWs)
                    if "fitFluxd" in field_data:
                        derived_fluxes[field_key] = float(field_data["fitFluxd"])
                    elif "fluxd" in field_data:
                        # fluxd might be per-spw
                        fluxd = field_data["fluxd"]
                        if isinstance(fluxd, (list, np.ndarray)) and len(fluxd) > 0:
                            derived_fluxes[field_key] = float(fluxd[0])

                    # Extract spectral index
                    if "spidx" in field_data:
                        spidx = field_data["spidx"]
                        if isinstance(spidx, (list, np.ndarray)) and len(spidx) > 0:
                            spectral_indices[field_key] = float(spidx[0])

                    # Reference frequency
                    if "fitRefFreq" in field_data:
                        fit_reffreq = float(field_data["fitRefFreq"])

        logger.info(
            "fluxscale complete: derived fluxes = %s",
            derived_fluxes,
        )

        return FluxscaleResult(
            caltable_in=caltable,
            fluxtable_out=fluxtable,
            reference_field=",".join(reference),
            transfer_fields=transfer,
            derived_fluxes=derived_fluxes,
            spectral_indices=spectral_indices,
            fit_reffreq_hz=fit_reffreq,
            success=True,
        )

    except Exception as e:
        logger.error("fluxscale failed: %s", e)
        return FluxscaleResult(
            caltable_in=caltable,
            fluxtable_out=fluxtable,
            reference_field=",".join(reference),
            transfer_fields=transfer,
            success=False,
            error_message=str(e),
        )


def bootstrap_flux_scale(
    ms_primary: str,
    ms_secondary: str,
    primary_name: str,
    secondary_name: str,
    refant: str,
    output_dir: str,
    *,
    primary_field: str = "",
    secondary_field: str = "",
    solint: str = "inf",
    minsnr: float = 3.0,
    combine_spw: bool = False,
) -> FluxBootstrapResult:
    """Bootstrap flux scale from primary to secondary calibrator.

    This is the main workflow function that:
    1. Sets MODEL_DATA on primary calibrator using setjy (Perley-Butler 2017)
    2. Solves gains on primary calibrator
    3. Solves gains on secondary calibrator (using catalog model)
    4. Runs fluxscale to derive secondary calibrator flux

    Parameters
    ----------
    ms_primary :
        Path to MS containing primary flux calibrator observation
    ms_secondary :
        Path to MS containing secondary calibrator observation
    primary_name :
        Primary calibrator name (e.g., "3C286")
    secondary_name :
        Secondary calibrator name (e.g., "0834+555")
    refant :
        Reference antenna chain
    output_dir :
        Directory for output calibration tables
    primary_field :
        Field selection for primary (default: auto)
    secondary_field :
        Field selection for secondary (default: auto)
    solint :
        Solution interval (default: "inf")
    minsnr :
        Minimum SNR for solutions (default: 3.0)
    combine_spw :
        Combine SPWs for gain solution (default: False)

    Returns
    -------
        FluxBootstrapResult with derived flux and calibration tables

    """
    from dsa110_contimg.core.calibration.model import populate_model_from_catalog

    # Validate primary calibrator
    primary_info = get_primary_calibrator_info(primary_name)
    if primary_info is None:
        return FluxBootstrapResult(
            primary_name=primary_name,
            secondary_name=secondary_name,
            primary_flux_jy=0.0,
            derived_flux_jy=0.0,
            flux_ratio=0.0,
            spectral_index=None,
            reference_freq_hz=1.4e9,
            gain_table_primary=None,
            gain_table_secondary=None,
            flux_table=None,
            success=False,
            error_message=f"'{primary_name}' is not a primary flux calibrator",
        )

    canonical_primary = primary_info["canonical_name"]
    primary_flux = primary_info["flux_1400mhz_jy"]

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Set MODEL_DATA on primary calibrator
    logger.info("Step 1: Setting MODEL_DATA for primary calibrator %s", canonical_primary)
    setjy_result = set_model_primary_calibrator(
        ms_path=ms_primary,
        field_sel=primary_field,
        calibrator_name=canonical_primary,
    )

    if not setjy_result.success:
        return FluxBootstrapResult(
            primary_name=canonical_primary,
            secondary_name=secondary_name,
            primary_flux_jy=primary_flux,
            derived_flux_jy=0.0,
            flux_ratio=0.0,
            spectral_index=None,
            reference_freq_hz=1.4e9,
            gain_table_primary=None,
            gain_table_secondary=None,
            flux_table=None,
            setjy_result=setjy_result,
            success=False,
            error_message=f"setjy failed: {setjy_result.error_message}",
        )

    # Step 2: Solve gains on primary calibrator
    logger.info("Step 2: Solving gains on primary calibrator")
    primary_gaintable = os.path.join(output_dir, f"{canonical_primary}_fluxcal.G")

    try:
        service = CASAService()
        service.gaincal(
            vis=ms_primary,
            caltable=primary_gaintable,
            field=primary_field if primary_field else canonical_primary,
            refant=refant,
            solint=solint,
            minsnr=minsnr,
            gaintype="G",
            calmode="ap",
            combine="spw" if combine_spw else "",
        )
    except Exception as e:
        return FluxBootstrapResult(
            primary_name=canonical_primary,
            secondary_name=secondary_name,
            primary_flux_jy=primary_flux,
            derived_flux_jy=0.0,
            flux_ratio=0.0,
            spectral_index=None,
            reference_freq_hz=1.4e9,
            gain_table_primary=None,
            gain_table_secondary=None,
            flux_table=None,
            setjy_result=setjy_result,
            success=False,
            error_message=f"gaincal on primary failed: {e}",
        )

    # Step 3: Set MODEL_DATA on secondary calibrator (from catalog)
    logger.info("Step 3: Setting MODEL_DATA for secondary calibrator %s", secondary_name)
    try:
        populate_model_from_catalog(
            ms_path=ms_secondary,
            field=secondary_field if secondary_field else "0",
            calibrator_name=secondary_name,
            use_unified_model=False,  # Single source for flux calibration
        )
    except Exception as e:
        logger.warning("Failed to set model from catalog: %s. Using unity model.", e)
        # Continue anyway - fluxscale can still derive flux without model

    # Step 4: Solve gains on secondary calibrator
    logger.info("Step 4: Solving gains on secondary calibrator")
    secondary_gaintable = os.path.join(output_dir, f"{secondary_name}_fluxcal.G")

    try:
        service = CASAService()
        service.gaincal(
            vis=ms_secondary,
            caltable=secondary_gaintable,
            field=secondary_field if secondary_field else "0",
            refant=refant,
            solint=solint,
            minsnr=minsnr,
            gaintype="G",
            calmode="ap",
            combine="spw" if combine_spw else "",
        )
    except Exception as e:
        return FluxBootstrapResult(
            primary_name=canonical_primary,
            secondary_name=secondary_name,
            primary_flux_jy=primary_flux,
            derived_flux_jy=0.0,
            flux_ratio=0.0,
            spectral_index=None,
            reference_freq_hz=1.4e9,
            gain_table_primary=primary_gaintable,
            gain_table_secondary=None,
            flux_table=None,
            setjy_result=setjy_result,
            success=False,
            error_message=f"gaincal on secondary failed: {e}",
        )

    # Step 5: Run fluxscale to bootstrap
    # NOTE: For DSA-110, primary and secondary may be in different MS files
    # Standard fluxscale requires them in the same caltable
    # We need to merge the gain tables first

    logger.info("Step 5: Merging gain tables and running fluxscale")
    merged_gaintable = os.path.join(output_dir, "merged_fluxcal.G")
    flux_table = os.path.join(output_dir, f"{secondary_name}_fluxscaled.G")

    try:
        # Merge calibration tables
        _merge_caltables(
            [primary_gaintable, secondary_gaintable],
            merged_gaintable,
        )

        # Determine field names in merged table
        primary_field_name = _get_field_name_from_caltable(primary_gaintable)
        secondary_field_name = _get_field_name_from_caltable(secondary_gaintable)

        # Run fluxscale
        fluxscale_result = run_fluxscale(
            caltable=merged_gaintable,
            fluxtable=flux_table,
            reference=primary_field_name or "0",
            transfer=secondary_field_name or "1",
        )

        if not fluxscale_result.success:
            return FluxBootstrapResult(
                primary_name=canonical_primary,
                secondary_name=secondary_name,
                primary_flux_jy=primary_flux,
                derived_flux_jy=0.0,
                flux_ratio=0.0,
                spectral_index=None,
                reference_freq_hz=1.4e9,
                gain_table_primary=primary_gaintable,
                gain_table_secondary=secondary_gaintable,
                flux_table=None,
                setjy_result=setjy_result,
                fluxscale_result=fluxscale_result,
                success=False,
                error_message=f"fluxscale failed: {fluxscale_result.error_message}",
            )

        # Extract derived flux
        derived_flux = 0.0
        derived_spidx = None

        for field_name, flux in fluxscale_result.derived_fluxes.items():
            if secondary_name.lower() in field_name.lower() or field_name == secondary_field_name:
                derived_flux = flux
                break
        else:
            # Take first transfer field flux
            if fluxscale_result.derived_fluxes:
                derived_flux = list(fluxscale_result.derived_fluxes.values())[0]

        for field_name, spidx in fluxscale_result.spectral_indices.items():
            if secondary_name.lower() in field_name.lower() or field_name == secondary_field_name:
                derived_spidx = spidx
                break
        else:
            if fluxscale_result.spectral_indices:
                derived_spidx = list(fluxscale_result.spectral_indices.values())[0]

        flux_ratio = derived_flux / primary_flux if primary_flux > 0 else 0.0

        logger.info(
            "Flux bootstrap complete: %s = %.3f Jy (ratio to %s: %.3f)",
            secondary_name,
            derived_flux,
            canonical_primary,
            flux_ratio,
        )

        return FluxBootstrapResult(
            primary_name=canonical_primary,
            secondary_name=secondary_name,
            primary_flux_jy=primary_flux,
            derived_flux_jy=derived_flux,
            flux_ratio=flux_ratio,
            spectral_index=derived_spidx,
            reference_freq_hz=fluxscale_result.fit_reffreq_hz or 1.4e9,
            gain_table_primary=primary_gaintable,
            gain_table_secondary=secondary_gaintable,
            flux_table=flux_table,
            setjy_result=setjy_result,
            fluxscale_result=fluxscale_result,
            success=True,
        )

    except Exception as e:
        logger.error("Flux bootstrap failed: %s", e)
        return FluxBootstrapResult(
            primary_name=canonical_primary,
            secondary_name=secondary_name,
            primary_flux_jy=primary_flux,
            derived_flux_jy=0.0,
            flux_ratio=0.0,
            spectral_index=None,
            reference_freq_hz=1.4e9,
            gain_table_primary=primary_gaintable,
            gain_table_secondary=secondary_gaintable,
            flux_table=None,
            setjy_result=setjy_result,
            success=False,
            error_message=str(e),
        )


def bootstrap_flux_scale_single_ms(
    ms_path: str,
    primary_field: str,
    secondary_field: str,
    primary_name: str,
    secondary_name: str,
    refant: str,
    output_dir: str,
    *,
    solint: str = "inf",
    minsnr: float = 3.0,
    combine_spw: bool = False,
    apply_to_secondary: bool = True,
) -> FluxBootstrapResult:
    """Bootstrap flux scale when primary and secondary are in the same MS.

    This is the simpler case where both calibrators were observed in the
    same session and are in the same Measurement Set.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set
    primary_field :
        Field selection for primary calibrator
    secondary_field :
        Field selection for secondary calibrator
    primary_name :
        Primary calibrator name (e.g., "3C286")
    secondary_name :
        Secondary calibrator name (e.g., "0834+555")
    refant :
        Reference antenna chain
    output_dir :
        Directory for output calibration tables
    solint :
        Solution interval (default: "inf")
    minsnr :
        Minimum SNR for solutions (default: 3.0)
    combine_spw :
        Combine SPWs for gain solution (default: False)
    apply_to_secondary :
        Apply flux-scaled gains to secondary field (default: True)

    Returns
    -------
        FluxBootstrapResult with derived flux and calibration tables

    """
    # Validate primary calibrator
    primary_info = get_primary_calibrator_info(primary_name)
    if primary_info is None:
        return FluxBootstrapResult(
            primary_name=primary_name,
            secondary_name=secondary_name,
            primary_flux_jy=0.0,
            derived_flux_jy=0.0,
            flux_ratio=0.0,
            spectral_index=None,
            reference_freq_hz=1.4e9,
            gain_table_primary=None,
            gain_table_secondary=None,
            flux_table=None,
            success=False,
            error_message=f"'{primary_name}' is not a primary flux calibrator",
        )

    canonical_primary = primary_info["canonical_name"]
    primary_flux = primary_info["flux_1400mhz_jy"]

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Set MODEL_DATA on primary calibrator
    logger.info("Step 1: Setting MODEL_DATA for primary calibrator %s", canonical_primary)
    setjy_result = set_model_primary_calibrator(
        ms_path=ms_path,
        field_sel=primary_field,
        calibrator_name=canonical_primary,
    )

    if not setjy_result.success:
        return FluxBootstrapResult(
            primary_name=canonical_primary,
            secondary_name=secondary_name,
            primary_flux_jy=primary_flux,
            derived_flux_jy=0.0,
            flux_ratio=0.0,
            spectral_index=None,
            reference_freq_hz=1.4e9,
            gain_table_primary=None,
            gain_table_secondary=None,
            flux_table=None,
            setjy_result=setjy_result,
            success=False,
            error_message=f"setjy failed: {setjy_result.error_message}",
        )

    # Step 2: Solve gains on both calibrators together
    logger.info("Step 2: Solving gains on both calibrators")
    combined_gaintable = os.path.join(output_dir, "combined_fluxcal.G")
    flux_table = os.path.join(output_dir, f"{secondary_name}_fluxscaled.G")

    try:
        service = CASAService()
        service.gaincal(
            vis=ms_path,
            caltable=combined_gaintable,
            field=f"{primary_field},{secondary_field}",
            refant=refant,
            solint=solint,
            minsnr=minsnr,
            gaintype="G",
            calmode="ap",
            combine="spw" if combine_spw else "",
        )
    except Exception as e:
        return FluxBootstrapResult(
            primary_name=canonical_primary,
            secondary_name=secondary_name,
            primary_flux_jy=primary_flux,
            derived_flux_jy=0.0,
            flux_ratio=0.0,
            spectral_index=None,
            reference_freq_hz=1.4e9,
            gain_table_primary=None,
            gain_table_secondary=None,
            flux_table=None,
            setjy_result=setjy_result,
            success=False,
            error_message=f"gaincal failed: {e}",
        )

    # Step 3: Run fluxscale
    logger.info("Step 3: Running fluxscale to bootstrap flux")
    fluxscale_result = run_fluxscale(
        caltable=combined_gaintable,
        fluxtable=flux_table,
        reference=primary_field,
        transfer=secondary_field,
    )

    if not fluxscale_result.success:
        return FluxBootstrapResult(
            primary_name=canonical_primary,
            secondary_name=secondary_name,
            primary_flux_jy=primary_flux,
            derived_flux_jy=0.0,
            flux_ratio=0.0,
            spectral_index=None,
            reference_freq_hz=1.4e9,
            gain_table_primary=combined_gaintable,
            gain_table_secondary=None,
            flux_table=None,
            setjy_result=setjy_result,
            fluxscale_result=fluxscale_result,
            success=False,
            error_message=f"fluxscale failed: {fluxscale_result.error_message}",
        )

    # Extract derived flux for secondary
    derived_flux = 0.0
    derived_spidx = None

    if fluxscale_result.derived_fluxes:
        # Try to find by field name
        for field_key, flux in fluxscale_result.derived_fluxes.items():
            derived_flux = flux
            break

    if fluxscale_result.spectral_indices:
        for field_key, spidx in fluxscale_result.spectral_indices.items():
            derived_spidx = spidx
            break

    flux_ratio = derived_flux / primary_flux if primary_flux > 0 else 0.0

    # Step 4: Optionally apply flux-scaled gains
    if apply_to_secondary:
        logger.info("Step 4: Applying flux-scaled gains to secondary field")
        try:
            service = CASAService()
            service.applycal(
                vis=ms_path,
                field=secondary_field,
                gaintable=[flux_table],
                interp=["linear"],
                calwt=True,
            )
            logger.info("Applied flux-scaled gains to field %s", secondary_field)
        except Exception as e:
            logger.warning("Failed to apply gains: %s", e)

    logger.info(
        "Flux bootstrap complete: %s = %.3f Jy (%.2f%% of %s)",
        secondary_name,
        derived_flux,
        flux_ratio * 100,
        canonical_primary,
    )

    return FluxBootstrapResult(
        primary_name=canonical_primary,
        secondary_name=secondary_name,
        primary_flux_jy=primary_flux,
        derived_flux_jy=derived_flux,
        flux_ratio=flux_ratio,
        spectral_index=derived_spidx,
        reference_freq_hz=fluxscale_result.fit_reffreq_hz or 1.4e9,
        gain_table_primary=combined_gaintable,
        gain_table_secondary=None,
        flux_table=flux_table,
        setjy_result=setjy_result,
        fluxscale_result=fluxscale_result,
        success=True,
    )


# =============================================================================
# Database Recording Functions
# =============================================================================


def record_flux_bootstrap(
    result: FluxBootstrapResult,
    db_path: str = os.environ.get("PIPELINE_DB", "/data/dsa110-contimg/state/db/pipeline.sqlite3"),
) -> bool:
    """Record flux bootstrap result to database.

    Parameters
    ----------
    result :
        FluxBootstrapResult from bootstrap workflow
    db_path :
        Path to pipeline database

    Returns
    -------
        True if recorded successfully

    """
    if not result.success:
        logger.warning("Not recording failed flux bootstrap to database")
        return False

    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        cur = conn.cursor()

        # Create table if needed
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS flux_bootstrap_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                primary_name TEXT NOT NULL,
                secondary_name TEXT NOT NULL,
                primary_flux_jy REAL NOT NULL,
                derived_flux_jy REAL NOT NULL,
                flux_ratio REAL NOT NULL,
                spectral_index REAL,
                reference_freq_hz REAL NOT NULL,
                gain_table_primary TEXT,
                gain_table_secondary TEXT,
                flux_table TEXT,
                timestamp_iso TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )

        cur.execute(
            """
            INSERT INTO flux_bootstrap_history
            (primary_name, secondary_name, primary_flux_jy, derived_flux_jy,
             flux_ratio, spectral_index, reference_freq_hz, gain_table_primary,
             gain_table_secondary, flux_table, timestamp_iso, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.primary_name,
                result.secondary_name,
                result.primary_flux_jy,
                result.derived_flux_jy,
                result.flux_ratio,
                result.spectral_index,
                result.reference_freq_hz,
                result.gain_table_primary,
                result.gain_table_secondary,
                result.flux_table,
                result.timestamp,
                time.time(),
            ),
        )

        conn.commit()
        conn.close()

        logger.info(
            "Recorded flux bootstrap: %s → %s (%.3f Jy)",
            result.primary_name,
            result.secondary_name,
            result.derived_flux_jy,
        )
        return True

    except Exception as e:
        logger.error("Failed to record flux bootstrap: %s", e)
        return False


def get_latest_flux_for_calibrator(
    calibrator_name: str,
    db_path: str = os.environ.get("PIPELINE_DB", "/data/dsa110-contimg/state/db/pipeline.sqlite3"),
) -> dict[str, Any] | None:
    """Get the most recent bootstrapped flux for a calibrator.

    Parameters
    ----------
    calibrator_name :
        Secondary calibrator name
    db_path :
        Path to pipeline database

    Returns
    -------
        Dictionary with flux info, or None if no record exists

    """
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute(
            """
            SELECT * FROM flux_bootstrap_history
            WHERE secondary_name = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (calibrator_name,),
        )

        row = cur.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    except Exception as e:
        logger.warning("Failed to query flux history: %s", e)
        return None


def update_calibrator_catalog_flux(
    calibrator_name: str,
    flux_jy: float,
    spectral_index: float | None = None,
    reference_freq_hz: float = 1.4e9,
    db_path: str = os.environ.get(
        "CAL_CATALOG_DB", "/data/dsa110-contimg/state/db/vla_calibrator_catalog.sqlite3"
    ),
) -> bool:
    """Update the calibrator catalog with bootstrapped flux.

    This allows the pipeline to use bootstrapped flux values for future
    calibrations instead of default catalog values.

    Parameters
    ----------
    calibrator_name :
        Calibrator name
    flux_jy :
        Bootstrapped flux density in Jy
    spectral_index :
        Spectral index (optional)
    reference_freq_hz :
        Reference frequency in Hz
    db_path :
        Path to calibrator registry database

    Returns
    -------
        True if updated successfully

    """
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        cur = conn.cursor()

        # Check if calibrators table exists
        cur.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='calibrators'
            """
        )
        if not cur.fetchone():
            logger.warning("calibrators table not found in %s", db_path)
            conn.close()
            return False

        # Update or insert
        cur.execute(
            """
            UPDATE calibrators
            SET flux_jy = ?, spectral_index = ?, flux_freq_hz = ?, flux_updated_at = ?
            WHERE UPPER(name) = UPPER(?)
            """,
            (
                flux_jy,
                spectral_index,
                reference_freq_hz,
                datetime.now(UTC).isoformat(),
                calibrator_name,
            ),
        )

        if cur.rowcount == 0:
            logger.warning("Calibrator %s not found in catalog", calibrator_name)
            conn.close()
            return False

        conn.commit()
        conn.close()

        logger.info(
            "Updated catalog flux for %s: %.3f Jy at %.1f MHz",
            calibrator_name,
            flux_jy,
            reference_freq_hz / 1e6,
        )
        return True

    except Exception as e:
        logger.error("Failed to update catalog flux: %s", e)
        return False


# =============================================================================
# Helper Functions
# =============================================================================


def _detect_primary_calibrator_in_ms(ms_path: str) -> str | None:
    """Detect primary flux calibrator from MS field names.

    Parameters
    ----------
    ms_path :
        Path to Measurement Set

    Returns
    -------
        Canonical name of primary calibrator if found, else None

    """
    import casacore.tables as ct

    try:
        with ct.table(f"{ms_path}::FIELD", readonly=True) as field_tb:
            field_names = field_tb.getcol("NAME")

        for field_name in field_names:
            for primary_name, info in PRIMARY_FLUX_CALIBRATORS.items():
                if primary_name.lower() in field_name.lower():
                    return primary_name
                for alt in info.get("alt_names", []):
                    if alt.lower() in field_name.lower():
                        return primary_name

        return None

    except Exception as e:
        logger.warning("Failed to detect primary calibrator: %s", e)
        return None


def _get_field_name_from_caltable(caltable_path: str) -> str | None:
    """Get field name from calibration table.

    Parameters
    ----------
    caltable_path :
        Path to calibration table

    Returns
    -------
        Field name, or None if not found

    """
    import casacore.tables as ct

    try:
        with ct.table(f"{caltable_path}::FIELD", readonly=True) as field_tb:
            names = field_tb.getcol("NAME")
            if names is not None and len(names) > 0:
                return str(names[0])
        return None
    except Exception:
        return None


def _merge_caltables(
    caltables: list[str],
    output_table: str,
) -> None:
    """Merge multiple calibration tables into one.

    This is needed when primary and secondary calibrators are in different MS
    files and we need to run fluxscale on a combined table.

    Parameters
    ----------
    caltables :
        List of input calibration table paths
    output_table :
        Output merged table path
    caltables: List[str] :

    Raises
    ------
    RuntimeError
        If merge fails

    """
    import shutil

    import casacore.tables as ct

    if not caltables:
        raise ValueError("No calibration tables provided")

    if len(caltables) == 1:
        # Just copy
        shutil.copytree(caltables[0], output_table)
        return

    # Copy first table as base
    shutil.copytree(caltables[0], output_table)

    # Append rows from remaining tables
    with ct.table(output_table, readonly=False) as out_tb:
        for caltable in caltables[1:]:
            with ct.table(caltable, readonly=True) as in_tb:
                # Read all columns
                colnames = in_tb.colnames()
                nrows_in = in_tb.nrows()

                if nrows_in == 0:
                    continue

                # Add rows to output
                start_row = out_tb.nrows()
                out_tb.addrows(nrows_in)

                # Copy column data
                for col in colnames:
                    try:
                        data = in_tb.getcol(col)
                        out_tb.putcol(col, data, startrow=start_row)
                    except Exception as e:
                        logger.warning("Failed to copy column %s: %s", col, e)

    logger.info("Merged %d calibration tables into %s", len(caltables), output_table)


def calculate_flux_at_frequency(
    flux_ref: float,
    freq_ref_hz: float,
    freq_target_hz: float,
    spectral_index: float,
) -> float:
    """Calculate flux at a target frequency using spectral index.

    Uses power-law model: S(ν) = S_ref * (ν / ν_ref)^α

    Parameters
    ----------
    flux_ref :
        Flux at reference frequency [Jy]
    freq_ref_hz :
        Reference frequency [Hz]
    freq_target_hz :
        Target frequency [Hz]
    spectral_index :
        Spectral index α

    Returns
    -------
        Flux at target frequency [Jy]

    """
    return flux_ref * (freq_target_hz / freq_ref_hz) ** spectral_index


# =============================================================================
# CLI Support Functions
# =============================================================================


def cli_flux_bootstrap(args) -> int:
    """CLI handler for flux bootstrap command.

    Parameters
    ----------
    args :


    """
    from dsa110_contimg.core.calibration.cli import setup_logging

    setup_logging(args.verbose)

    if args.single_ms:
        # Both calibrators in same MS
        result = bootstrap_flux_scale_single_ms(
            ms_path=args.ms_primary,
            primary_field=args.primary_field,
            secondary_field=args.secondary_field,
            primary_name=args.primary_name,
            secondary_name=args.secondary_name,
            refant=args.refant,
            output_dir=args.output_dir,
            solint=args.solint,
            minsnr=args.minsnr,
            combine_spw=args.combine_spw,
            apply_to_secondary=not args.no_apply,
        )
    else:
        # Separate MS files
        result = bootstrap_flux_scale(
            ms_primary=args.ms_primary,
            ms_secondary=args.ms_secondary,
            primary_name=args.primary_name,
            secondary_name=args.secondary_name,
            refant=args.refant,
            output_dir=args.output_dir,
            primary_field=args.primary_field,
            secondary_field=args.secondary_field,
            solint=args.solint,
            minsnr=args.minsnr,
            combine_spw=args.combine_spw,
        )

    if result.success:
        logger.info(" Flux bootstrap successful")
        logger.info("  Primary: %s (%.3f Jy)", result.primary_name, result.primary_flux_jy)
        logger.info("  Secondary: %s (%.3f Jy)", result.secondary_name, result.derived_flux_jy)
        logger.info("  Flux ratio: %.3f", result.flux_ratio)
        if result.spectral_index is not None:
            logger.info("  Spectral index: %.3f", result.spectral_index)
        logger.info("  Flux table: %s", result.flux_table)

        # Record to database
        if args.record_db:
            record_flux_bootstrap(result)

        # Update catalog if requested
        if args.update_catalog:
            update_calibrator_catalog_flux(
                result.secondary_name,
                result.derived_flux_jy,
                result.spectral_index,
                result.reference_freq_hz,
            )

        # Output JSON if requested
        if args.json_output:
            print(json.dumps(result.to_dict(), indent=2))

        return 0
    else:
        logger.error(" Flux bootstrap failed: %s", result.error_message)
        return 1


def cli_list_primary_calibrators(args) -> int:
    """CLI handler for list-flux-calibrators command.

    Parameters
    ----------
    args :


    """
    print("Primary VLA Flux Calibrators (Perley-Butler 2017)")
    print("=" * 60)
    for name, info in PRIMARY_FLUX_CALIBRATORS.items():
        print(f"\n{name}")
        print(f"  Flux (1.4 GHz): {info['flux_1400mhz_jy']:.2f} Jy")
        print(f"  Spectral index: {info['spectral_index']:.3f}")
        print(f"  Position: {info['ra_j2000']}, {info['dec_j2000']}")
        print(f"  Alt names: {', '.join(info['alt_names'])}")
        print(f"  Notes: {info['notes']}")
    return 0


def cli_setjy(args) -> int:
    """CLI handler for setjy command.

    Parameters
    ----------
    args :


    """
    from dsa110_contimg.core.calibration.cli import setup_logging

    setup_logging(args.verbose)

    result = set_model_primary_calibrator(
        ms_path=args.ms,
        field_sel=args.field,
        calibrator_name=args.calibrator,
        standard=args.standard,
    )

    if result.success:
        logger.info(" setjy successful")
        logger.info("  Calibrator: %s", result.calibrator_name)
        logger.info("  Model flux: %.3f Jy", result.model_flux_jy)
        return 0
    else:
        logger.error(" setjy failed: %s", result.error_message)
        return 1
