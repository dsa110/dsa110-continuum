"""Shared configuration loaders for evaluation modules."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dsa110_contimg.common.utils import get_env_path

logger = logging.getLogger(__name__)


def load_thresholds_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load evaluation thresholds from YAML config.

    Parameters
    ----------
    config_path : Optional[Path], optional
        Path to config file, defaults to config/evaluation_thresholds.yaml
    """
    if config_path is None:
        contimg_base = get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg")
        config_path = contimg_base / "config" / "evaluation_thresholds.yaml"

    if not config_path.exists():
        logger.warning("Thresholds config not found: %s, using defaults", config_path)
        return {}

    try:
        from dsa110_contimg.common.utils.yaml_loader import load_yaml_with_env

        config = load_yaml_with_env(config_path, expand_vars=True) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Unable to load thresholds config %s: %s", config_path, exc)
        return {}

    if not isinstance(config, dict):
        logger.warning("Thresholds config %s did not parse to a dict; using defaults", config_path)
        return {}

    return config
