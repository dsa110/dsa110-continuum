"""Ground truth tracking for synthetic data validation.

    This module provides a registry system to track what was injected into
    synthetic datasets, enabling validation of pipeline outputs against known
    ground truth. The registry stores source properties, variability models,
    and expected fluxes at each observation epoch.

    Key Features
------------
    - Track injected sources across multiple epochs
    - Store variability models for time-varying sources
    - Query expected flux at any MJD for validation
    - Export ground truth for comparison with pipeline outputs
    - Persist to JSON and optionally to database

    Example
-------
    >>> from dsa110_contimg.core.simulation.ground_truth import GroundTruthRegistry
    >>> from dsa110_contimg.core.simulation.variability_models import FlareModel
    >>>
    >>> # Create registry
    >>> registry = GroundTruthRegistry(test_run_id="test_2025-01-15")
    >>>
    >>> # Register a variable source
    >>> flare = FlareModel(
    ...     peak_time_mjd=60300.5,
    ...     peak_flux_jy=5.0,
    ...     baseline_flux_jy=1.0,
    ... )
    >>> registry.register_source(
    ...     source_id="NVSS_J123456+420000",
    ...     ra_deg=188.5,
    ...     dec_deg=42.0,
    ...     baseline_flux_jy=1.0,
    ...     variability_model=flare,
    ...     catalog_origin="NVSS",
    ... )
    >>>
    >>> # Query expected flux at specific time
    >>> expected_flux = registry.get_expected_flux("NVSS_J123456+420000", mjd=60300.5)
    >>> print(f"Expected flux at peak: {expected_flux:.2f} Jy")
    >>>
    >>> # Export for validation
    >>> from dsa110_contimg.common.utils import TempPaths
    >>> registry.export_to_json(TempPaths.report("ground_truth.json"))
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dsa110_contimg.core.simulation.variability_models import (
    VariabilityModel,
    compute_flux_at_time,
)

logger = logging.getLogger(__name__)


@dataclass
class GroundTruthSource:
    """Known injected source with time-varying properties.

    This represents a source that was injected into synthetic data,
    including its position, baseline flux, and optional variability model.

    Parameters
    ----------
    source_id :
        Unique source identifier (e.g., "NVSS_J123456+420000")
    ra_deg :
        Right ascension (degrees)
    dec_deg :
        Declination (degrees)
    baseline_flux_jy :
        Quiescent flux density (Jy)
    variability_model :
        Optional variability model
    catalog_origin :
        Source catalog (e.g., "NVSS", "FIRST")
    spectral_index :
        Spectral index (optional)
    metadata :
        Additional metadata

    """

    source_id: str
    ra_deg: float
    dec_deg: float
    baseline_flux_jy: float
    variability_model: VariabilityModel | None = None
    catalog_origin: str = "synthetic"
    spectral_index: float | None = None
    metadata: dict = field(default_factory=dict)

    def get_flux_at_time(self, mjd: float) -> float:
        """Compute expected flux at specified MJD."""
        return compute_flux_at_time(self.baseline_flux_jy, self.variability_model, mjd)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        result = {
            "source_id": self.source_id,
            "ra_deg": self.ra_deg,
            "dec_deg": self.dec_deg,
            "baseline_flux_jy": self.baseline_flux_jy,
            "catalog_origin": self.catalog_origin,
            "spectral_index": self.spectral_index,
            "metadata": self.metadata,
        }
        if self.variability_model:
            result["variability_model"] = self.variability_model.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict) -> GroundTruthSource:
        """Deserialize from dictionary."""
        var_model = None
        if "variability_model" in data:
            from dsa110_contimg.core.simulation.variability_models import VariabilityModel

            var_model = VariabilityModel.from_dict(data["variability_model"])

        return cls(
            source_id=data["source_id"],
            ra_deg=data["ra_deg"],
            dec_deg=data["dec_deg"],
            baseline_flux_jy=data["baseline_flux_jy"],
            variability_model=var_model,
            catalog_origin=data.get("catalog_origin", "synthetic"),
            spectral_index=data.get("spectral_index"),
            metadata=data.get("metadata", {}),
        )


class GroundTruthRegistry:
    """Registry for tracking injected sources in synthetic datasets.

    This class maintains a record of what was injected into synthetic data,
    enabling validation of pipeline outputs against known ground truth.

    Parameters
    ----------
    test_run_id :
        Unique identifier for this test run
    sources : optional
        Pre-populated sources

    Returns
    -------
    None

    Examples
    --------
    >>> registry = GroundTruthRegistry(test_run_id="validation_2025")
    >>> registry.register_source("src1", 180.0, 45.0, 2.0)
    >>> flux = registry.get_expected_flux("src1", mjd=60000.0)
    """

    def __init__(self, test_run_id: str, sources: list[GroundTruthSource] | None = None):
        self.test_run_id = test_run_id
        self.sources: dict[str, GroundTruthSource] = {}
        self.epochs: list[float] = []  # MJD values for observed epochs

        if sources:
            for source in sources:
                self.sources[source.source_id] = source

    def add_source(self, source: GroundTruthSource) -> None:
        """Add a GroundTruthSource object to the registry."""
        self.sources[source.source_id] = source
        logger.debug("Added source: %s", source.source_id)

    def register_source(
        self,
        source_id: str,
        ra_deg: float,
        dec_deg: float,
        baseline_flux_jy: float,
        variability_model: VariabilityModel | None = None,
        catalog_origin: str = "synthetic",
        spectral_index: float | None = None,
        **metadata: Any,
    ) -> None:
        """Register a source in the ground truth registry.

        Parameters
        ----------
        source_id :
            Unique source identifier
        ra_deg :
            Right ascension (degrees)
        dec_deg :
            Declination (degrees)
        baseline_flux_jy :
            Quiescent flux density (Jy)
        variability_model :
            Optional variability model
        catalog_origin :
            Source catalog
        spectral_index :
            Spectral index
        **metadata :
            Additional metadata
        """
        source = GroundTruthSource(
            source_id=source_id,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            baseline_flux_jy=baseline_flux_jy,
            variability_model=variability_model,
            catalog_origin=catalog_origin,
            spectral_index=spectral_index,
            metadata=metadata,
        )
        self.sources[source_id] = source
        logger.debug("Registered source: %s", source_id)

    def register_epoch(self, mjd: float) -> None:
        """Register an observation epoch."""
        if mjd not in self.epochs:
            self.epochs.append(mjd)
            self.epochs.sort()
            logger.debug("Registered epoch: MJD %.4f", mjd)

    def get_expected_flux(self, source_id: str, mjd: float) -> float | None:
        """Get expected flux for a source at specified MJD.

        Parameters
        ----------
        source_id :
            Source identifier
        mjd :
            Modified Julian Date

        Returns
        -------
            Expected flux in Jy, or None if source not found

        """
        source = self.sources.get(source_id)
        if source is None:
            logger.warning("Source not found in ground truth: %s", source_id)
            return None
        return source.get_flux_at_time(mjd)

    def get_variable_sources(self) -> list[GroundTruthSource]:
        """Get list of sources with variability models."""
        return [s for s in self.sources.values() if s.variability_model is not None]

    def get_constant_sources(self) -> list[GroundTruthSource]:
        """Get list of constant sources (no variability)."""
        return [s for s in self.sources.values() if s.variability_model is None]

    def export_to_json(self, output_path: Path) -> None:
        """Export ground truth to JSON file.

        Parameters
        ----------
        output_path :
            Path for output JSON file
        """
        data = {
            "test_run_id": self.test_run_id,
            "created_at": time.time(),
            "n_sources": len(self.sources),
            "n_variable": len(self.get_variable_sources()),
            "epochs": self.epochs,
            "sources": [source.to_dict() for source in self.sources.values()],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(data, indent=2))
        logger.info("Exported ground truth to %s", output_path)

    @classmethod
    def from_json(cls, input_path: Path) -> GroundTruthRegistry:
        """Load ground truth from JSON file.

        Parameters
        ----------
        input_path :
            Path to JSON file

        Returns
        -------
            Loaded GroundTruthRegistry

        """
        data = json.loads(input_path.read_text())

        sources = [GroundTruthSource.from_dict(s) for s in data["sources"]]
        registry = cls(test_run_id=data["test_run_id"], sources=sources)
        registry.epochs = data.get("epochs", [])

        logger.info("Loaded ground truth from %s: %d sources", input_path, len(sources))
        return registry

    def export_to_database(
        self,
        db_path: Path,
        table_name: str = "synthetic_ground_truth",
    ) -> None:
        """Export ground truth to database table.

        Parameters
        ----------
        db_path :
            Path to SQLite database
        table_name :
            Table name (default: synthetic_ground_truth)
        """
        conn = sqlite3.connect(db_path, timeout=30.0)
        cursor = conn.cursor()

        # Create table if it doesn't exist
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_run_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                ra_deg REAL NOT NULL,
                dec_deg REAL NOT NULL,
                baseline_flux_jy REAL NOT NULL,
                variability_model TEXT,
                catalog_origin TEXT,
                spectral_index REAL,
                metadata TEXT,
                created_at REAL NOT NULL
            )
        """)

        # Insert sources
        current_time = time.time()
        for source in self.sources.values():
            var_model_json = None
            if source.variability_model:
                var_model_json = json.dumps(source.variability_model.to_dict())

            cursor.execute(
                f"""
                INSERT INTO {table_name}
                (test_run_id, source_id, ra_deg, dec_deg, baseline_flux_jy,
                 variability_model, catalog_origin, spectral_index, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.test_run_id,
                    source.source_id,
                    source.ra_deg,
                    source.dec_deg,
                    source.baseline_flux_jy,
                    var_model_json,
                    source.catalog_origin,
                    source.spectral_index,
                    json.dumps(source.metadata),
                    current_time,
                ),
            )

        conn.commit()
        conn.close()
        logger.info("Exported %d sources to %s:%s", len(self.sources), db_path, table_name)

    @classmethod
    def from_database(
        cls,
        db_path: Path,
        test_run_id: str,
        table_name: str = "synthetic_ground_truth",
    ) -> GroundTruthRegistry:
        """Load ground truth from database.

        Parameters
        ----------
        db_path :
            Path to SQLite database
        test_run_id :
            Test run ID to load
        table_name :
            Table name

        Returns
        -------
            Loaded GroundTruthRegistry

        """
        conn = sqlite3.connect(db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            f"""
            SELECT * FROM {table_name}
            WHERE test_run_id = ?
            ORDER BY source_id
            """,
            (test_run_id,),
        )

        sources = []
        for row in cursor.fetchall():
            var_model = None
            if row["variability_model"]:
                from dsa110_contimg.core.simulation.variability_models import VariabilityModel

                var_model = VariabilityModel.from_dict(json.loads(row["variability_model"]))

            source = GroundTruthSource(
                source_id=row["source_id"],
                ra_deg=row["ra_deg"],
                dec_deg=row["dec_deg"],
                baseline_flux_jy=row["baseline_flux_jy"],
                variability_model=var_model,
                catalog_origin=row["catalog_origin"],
                spectral_index=row["spectral_index"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
            sources.append(source)

        conn.close()

        registry = cls(test_run_id=test_run_id, sources=sources)
        logger.info("Loaded %d sources from %s for run %s", len(sources), db_path, test_run_id)
        return registry
