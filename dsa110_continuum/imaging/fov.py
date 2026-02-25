"""Field-of-view helpers for imaging.

Provides an optional derived FoV based on telescope parameters instead of
hard-coding a fixed extent. Defaults remain unchanged unless the caller opts in
via settings.imaging.derive_extent_from_telescope.
"""

from __future__ import annotations

import logging
import math
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from dsa110_contimg.common.unified_config import settings
from astropy import constants as const

LOG = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    from dsa110_contimg.common.utils.yaml_loader import load_yaml_with_env

    return load_yaml_with_env(path, expand_vars=True) or {}


def _default_telescope_path() -> Path:
    """Return packaged telescope YAML path."""
    return Path(resources.files("dsa110_contimg.simulation.pyuvsim") / "telescope.yaml")


def _default_beam_path() -> Path:
    return Path(resources.files("dsa110_contimg.simulation.pyuvsim") / "beams.yaml")


def derive_extent_deg() -> float:
    """Compute image FoV in degrees using telescope parameters.

    Derivation (simple, conservative):
        FWHM_deg = kappa * lambda / D * (180/pi)
        extent_deg = FWHM_deg * padding_factor

    Clamped to [min_extent_deg, max_extent_deg]. On any failure, falls back to
    settings.imaging.fixed_extent_deg.
    """
    cfg = settings.imaging

    # Fast exit if disabled
    if not cfg.derive_extent_from_telescope:
        return cfg.fixed_extent_deg

    try:
        tel_path = (
            Path(cfg.telescope_yaml_path) if cfg.telescope_yaml_path else _default_telescope_path()
        )
        tel = _load_yaml(tel_path)

        # Reference frequency
        freq_ref = tel.get("spectral", {}).get("reference_frequency_hz")
        if not freq_ref:
            fmin = tel.get("spectral", {}).get("freq_min_hz")
            fmax = tel.get("spectral", {}).get("freq_max_hz")
            if fmin and fmax:
                freq_ref = 0.5 * (float(fmin) + float(fmax))
        if not freq_ref:
            LOG.warning("Falling back to fixed FoV: reference frequency missing in telescope YAML")
            return cfg.fixed_extent_deg

        # Beam diameter from beams.yaml if referenced
        beam_cfg = tel.get("beam_model", {}) or {}
        beam_path = beam_cfg.get("config")
        diameter_m: float | None = None
        if beam_path:
            try:
                beam_file = Path(beam_path)
                if not beam_file.is_absolute():
                    # Relative to telescope yaml directory
                    beam_file = tel_path.parent / beam_file
                beams = _load_yaml(beam_file)
                beams_list = beams.get("beams", []) or []
                if beams_list:
                    diameter_m = beams_list[0].get("parameters", {}).get("diameter_m")
            except Exception as exc:  # pragma: no cover - defensive
                LOG.warning("Failed to read beams.yaml; using fallback diameter", exc_info=exc)
        if diameter_m is None:
            # Try bundled beams.yaml
            try:
                beams = _load_yaml(_default_beam_path())
                beams_list = beams.get("beams", []) or []
                if beams_list:
                    diameter_m = beams_list[0].get("parameters", {}).get("diameter_m")
            except Exception:  # pragma: no cover - defensive
                pass

        if not diameter_m:
            LOG.warning("Falling back to fixed FoV: dish diameter missing in beams.yaml")
            return cfg.fixed_extent_deg

        wavelength_m = const.c.value / float(freq_ref)
        fwhm_rad = cfg.primary_beam_kappa * wavelength_m / float(diameter_m)
        fwhm_deg = math.degrees(fwhm_rad)
        extent = fwhm_deg * cfg.fov_padding_factor

        clamped = max(cfg.min_extent_deg, min(cfg.max_extent_deg, extent))

        LOG.info(
            "Derived FoV: fwhm_deg=%.3f, extent_deg=%.3f, clamped=%.3f (kappa=%.2f, pad=%.2f)",
            fwhm_deg,
            extent,
            clamped,
            cfg.primary_beam_kappa,
            cfg.fov_padding_factor,
        )
        return clamped
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("Falling back to fixed FoV after derivation error", exc_info=exc)
        return cfg.fixed_extent_deg
