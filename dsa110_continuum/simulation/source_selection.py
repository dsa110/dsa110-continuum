"""Utilities for selecting real catalog sources for synthetic simulations.

This module bridges the catalog querying layer and the simulation toolkit.
It provides lightweight dataclasses that capture the metadata needed to
generate visibilities together with helpers that query existing catalog
SQLite databases (NVSS, FIRST, VLASS, RACS, etc.).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dsa110_contimg.core.catalog.coverage import CATALOG_COVERAGE
from dsa110_contimg.core.catalog.query import query_sources


@dataclass(frozen=True)
class CatalogRegion:
    """Simple sky-region descriptor (circular field-of-view)."""

    ra_deg: float
    dec_deg: float
    radius_deg: float


@dataclass
class SyntheticSource:
    """Container describing a single catalog-based source."""

    source_id: str | None
    ra_deg: float
    dec_deg: float
    flux_ref_jy: float
    reference_freq_hz: float | None
    spectral_index: float | None = None
    major_axis_arcsec: float | None = None
    minor_axis_arcsec: float | None = None
    position_angle_deg: float | None = None
    catalog_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    variability_model: Any | None = None  # VariabilityModel (avoid circular import)

    def to_metadata_json(self) -> str:
        """Serialize minimal metadata for embedding inside UVH5 keywords."""
        payload = {
            "source_id": self.source_id,
            "ra_deg": self.ra_deg,
            "dec_deg": self.dec_deg,
            "flux_ref_jy": self.flux_ref_jy,
            "reference_freq_hz": self.reference_freq_hz,
            "spectral_index": self.spectral_index,
            "major_axis_arcsec": self.major_axis_arcsec,
            "minor_axis_arcsec": self.minor_axis_arcsec,
            "position_angle_deg": self.position_angle_deg,
            "catalog_type": self.catalog_type,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)


class SourceSelector:
    """Select catalog sources for simulation inputs."""

    def __init__(
        self,
        region: CatalogRegion,
        catalog_type: str,
        *,
        catalog_path: Path | None = None,
    ) -> None:
        self.region = region
        self.catalog_type = catalog_type.lower()
        self.catalog_path = catalog_path
        self._reference_freq_hz = self._lookup_reference_frequency()

    def select_sources(
        self,
        *,
        min_flux_mjy: float | None = None,
        max_sources: int | None = None,
    ) -> list[SyntheticSource]:
        """Query the requested catalog and convert rows to SyntheticSource objects."""
        df = query_sources(
            catalog_type=self.catalog_type,
            ra_center=self.region.ra_deg,
            dec_center=self.region.dec_deg,
            radius_deg=self.region.radius_deg,
            min_flux_mjy=min_flux_mjy,
            max_sources=max_sources,
            catalog_path=str(self.catalog_path) if self.catalog_path else None,
        )

        if df.empty:
            return []

        return self._from_dataframe(df)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _from_dataframe(self, df: pd.DataFrame) -> list[SyntheticSource]:
        """Convert catalog rows (DataFrame) to SyntheticSource objects."""
        sources: list[SyntheticSource] = []
        for _, row in df.iterrows():
            flux_mjy = float(row["flux_mjy"])
            flux_jy = flux_mjy / 1000.0

            source_id = None
            for key in ("source_id", "name", "component_id"):
                if key in row and pd.notna(row[key]):
                    source_id = str(row[key])
                    break

            metadata = {}
            for key in ("catalog_name", "dataset", "mosaic_id"):
                if key in row and pd.notna(row[key]):
                    metadata[key] = row[key]

            major_axis = self._safe_get(row, "maj_arcsec")
            minor_axis = self._safe_get(row, "min_arcsec")
            pa_deg = self._safe_get(row, "pa_deg")
            spectral_index = self._safe_get(row, "spectral_index")

            sources.append(
                SyntheticSource(
                    source_id=source_id,
                    ra_deg=float(row["ra_deg"]),
                    dec_deg=float(row["dec_deg"]),
                    flux_ref_jy=flux_jy,
                    reference_freq_hz=self._reference_freq_hz,
                    spectral_index=spectral_index,
                    major_axis_arcsec=major_axis,
                    minor_axis_arcsec=minor_axis,
                    position_angle_deg=pa_deg,
                    catalog_type=self.catalog_type,
                    metadata=metadata,
                )
            )
        return sources

    def _lookup_reference_frequency(self) -> float | None:
        """Derive the survey reference frequency in Hz from coverage metadata."""
        coverage = CATALOG_COVERAGE.get(self.catalog_type)
        if not coverage:
            return None
        freq_ghz = coverage.get("frequency_ghz")
        if freq_ghz is None:
            return None
        return float(freq_ghz) * 1e9

    @staticmethod
    def _safe_get(row: pd.Series, column: str) -> float | None:
        if column not in row or pd.isna(row[column]):
            return None
        value = row[column]
        if isinstance(value, (np.floating, float, int, np.integer)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def summarize_sources(sources: Sequence[SyntheticSource]) -> dict[str, Any]:
    """Return a lightweight summary for logging/metadata."""
    if not sources:
        return {"count": 0}

    fluxes = np.array([src.flux_ref_jy for src in sources], dtype=float)
    return {
        "count": len(sources),
        "total_flux_jy": float(fluxes.sum()),
        "brightest_flux_jy": float(fluxes.max()),
        "faintest_flux_jy": float(fluxes.min()),
    }
