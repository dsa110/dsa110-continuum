"""
Calibration Checkpointing System.

Provides caching/checkpointing for expensive calibration operations:
- Phaseshifted MS files
- MODEL_DATA population
- Flagging strategy tracking

This prevents redundant work when calibration retries occur, saving
10+ minutes per retry.

Usage:
    from dsa110_contimg.core.calibration.checkpoints import CalibrationCheckpoint

    checkpoint = CalibrationCheckpoint(ms_path)

    # Get or create phaseshifted MS
    cal_ms = checkpoint.get_or_create_phaseshifted_ms(
        field="0~23",
        calibrator_name="0834+555",
        phaseshift_fn=lambda ms, fld, cal: phaseshift_ms(
            ms_path=ms, field=fld, mode="calibrator", calibrator_name=cal
        )
    )

    # Track which strategies have been tried
    checkpoint.mark_strategy_tried("default")
    untried = checkpoint.get_untried_strategies(["default", "aggressive", "conservative"])
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MSIntegrityError(Exception):
    """ """

    def __init__(self, ms_path: str, reason: str, suggestions: list[str] | None = None):
        self.ms_path = ms_path
        self.reason = reason
        self.suggestions = suggestions or []
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        msg = f"MS integrity check failed for {self.ms_path}: {self.reason}"
        if self.suggestions:
            msg += "\n\nSuggested actions:\n"
            msg += "\n".join(f"  - {s}" for s in self.suggestions)
        return msg


def validate_ms_integrity(ms_path: str, check_writable: bool = False) -> None:
    """Validate measurement set integrity before trusting cached state.

        This is a FAIL-FAST validation - it raises MSIntegrityError with
        actionable diagnostics on first problem found, rather than silently
        falling back to alternatives.

    Parameters
    ----------
    ms_path : str
        Path to measurement set directory.
    check_writable : bool, optional
        If True, also verify the MS is writable. Default is False.

    Returns
    -------
        None
    """
    ms = Path(ms_path)

    # Check 1: MS directory exists
    if not ms.exists():
        raise MSIntegrityError(
            ms_path,
            "Measurement set directory does not exist",
            ["Verify the path is correct", "Check if MS was deleted or moved"],
        )

    if not ms.is_dir():
        raise MSIntegrityError(
            ms_path,
            "Path exists but is not a directory (MS must be a directory)",
            ["Check path - should point to .ms directory, not a file"],
        )

    # Check 2: Required subtables exist
    required_subtables = ["ANTENNA", "SPECTRAL_WINDOW", "FIELD", "POLARIZATION"]
    missing = [t for t in required_subtables if not (ms / t).is_dir()]
    if missing:
        raise MSIntegrityError(
            ms_path,
            f"Required subtables missing: {missing}",
            [
                "MS may be corrupted or incomplete",
                "Re-run conversion from source data",
                f"Check if {ms_path} is a valid MS directory",
            ],
        )

    # Check 3: STATE table exists (this is what broke phaseshift)
    state_table = ms / "STATE"
    if not state_table.is_dir():
        raise MSIntegrityError(
            ms_path,
            "STATE subtable missing - required for CASA operations",
            [
                "MS likely corrupted during copy or phaseshift operation",
                "Re-run phaseshift_ms() to create a fresh phaseshifted copy",
                "Delete checkpoint file to force recreation",
            ],
        )

    # Check 4: Main table is readable and has required columns
    try:
        import casacore.tables as casatables

        with casatables.table(ms_path, readonly=True) as tb:
            nrows = tb.nrows()
            if nrows == 0:
                raise MSIntegrityError(
                    ms_path,
                    "Main table has zero rows - MS is empty",
                    ["Re-run MS creation/conversion", "Check source data"],
                )

            # Check 5: Required columns exist
            required_columns = ["DATA", "FLAG", "UVW", "TIME", "ANTENNA1", "ANTENNA2"]
            existing_columns = tb.colnames()
            missing_cols = [c for c in required_columns if c not in existing_columns]
            if missing_cols:
                raise MSIntegrityError(
                    ms_path,
                    f"Required columns missing: {missing_cols}",
                    [
                        "MS may be corrupted or an incomplete conversion",
                        f"Available columns: {existing_columns[:10]}...",
                        "Re-run conversion from source data",
                    ],
                )
    except ImportError:
        logger.warning("casacore not available - skipping table read check")
    except Exception as e:
        if "MSIntegrityError" not in str(type(e)):
            raise MSIntegrityError(
                ms_path,
                f"Cannot open main table: {e}",
                [
                    "MS may be corrupted or locked by another process",
                    "Try: ls -la {ms_path}",
                    "Check disk space and permissions",
                ],
            )
        raise

    # Check 6: Writable check if requested
    if check_writable:
        test_file = ms / ".write_test"
        try:
            test_file.touch()
            test_file.unlink()
        except (PermissionError, OSError) as e:
            raise MSIntegrityError(
                ms_path,
                f"MS is not writable: {e}",
                [
                    "Check file permissions (common with Docker AOFlagger)",
                    f"Run: chmod -R u+w {ms_path}",
                    "Check disk space",
                ],
            )

    logger.debug(f"MS integrity validated: {ms_path}")


@dataclass
class CheckpointState:
    """Persistent state for calibration checkpointing."""

    # Phaseshifted MS info
    phaseshifted_ms: str | None = None
    phaseshift_calibrator: str | None = None
    phaseshift_field: str | None = None
    phaseshift_phasecenter: str | None = None
    phaseshift_timestamp: str | None = None

    # MODEL_DATA info
    model_populated: bool = False
    model_calibrator: str | None = None
    model_flux_jy: float | None = None
    model_timestamp: str | None = None

    # Flagging strategy tracking
    tried_strategies: list[str] = field(default_factory=list)
    successful_strategy: str | None = None
    strategy_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Custom data for generic key-value storage
    custom_data: dict[str, Any] = field(default_factory=dict)

    # Overall state
    created_at: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointState:
        """Create from dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary of data to create the object from.

        Returns
        -------
            Any
            Created object instance.
        """
        # Handle missing fields gracefully
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


class CalibrationCheckpoint:
    """Checkpoint manager for calibration operations.

    Tracks intermediate work to avoid redundant computation on retries:
    - Phaseshifted MS creation (expensive, ~1-2 min)
    - MODEL_DATA population (moderate, ~30s)
    - Which flagging strategies have been tried

    Checkpoints are stored as JSON files alongside the MS.

    """

    def __init__(self, ms_path: str, staging_dir: str | None = None):
        """Initialize checkpoint manager.

        Parameters
        ----------
        ms_path : str
            Path to the original measurement set.
        staging_dir : str, optional
            Directory for calibration staging files.
            If not specified, uses a 'cal_staging' subdirectory alongside the input MS
            to preserve I/O characteristics.

        Returns
        -------
            None
        """
        self.ms_path = Path(ms_path)
        if staging_dir:
            self.staging_dir = Path(staging_dir)
        else:
            # Default: cal_staging subdirectory alongside input MS
            # This preserves I/O performance (e.g., if MS is in /dev/shm, staging stays there)
            self.staging_dir = self.ms_path.parent / "cal_staging"
        self.staging_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint file stored in staging directory
        ms_name = self.ms_path.stem
        self.checkpoint_file = self.staging_dir / f"{ms_name}_checkpoint.json"

        # Load existing state or create new
        self.state = self._load_state()

    def _load_state(self) -> CheckpointState:
        """Load checkpoint state from disk."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file) as f:
                    data = json.load(f)
                logger.info(f"Loaded calibration checkpoint: {self.checkpoint_file}")
                return CheckpointState.from_dict(data)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load checkpoint, creating new: {e}")

        # Create new state
        state = CheckpointState(created_at=datetime.now().isoformat())
        return state

    def _save_state(self) -> None:
        """Save checkpoint state to disk."""
        self.state.updated_at = datetime.now().isoformat()
        try:
            with open(self.checkpoint_file, "w") as f:
                json.dump(self.state.to_dict(), f, indent=2)
            logger.debug(f"Saved calibration checkpoint: {self.checkpoint_file}")
        except OSError as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def _invalidate_phaseshift_cache(self) -> None:
        """Invalidate cached phaseshift state when MS is corrupted.

        Clears phaseshift-related state fields and optionally deletes the
        corrupted MS file to ensure a clean recreation.

        """
        import shutil

        corrupted_ms = self.state.phaseshifted_ms
        if corrupted_ms and Path(corrupted_ms).exists():
            logger.info(f"Deleting corrupted phaseshifted MS: {corrupted_ms}")
            try:
                shutil.rmtree(corrupted_ms)
            except OSError as e:
                logger.warning(f"Failed to delete corrupted MS (will try to overwrite): {e}")

        # Clear cached state
        self.state.phaseshifted_ms = None
        self.state.phaseshift_calibrator = None
        self.state.phaseshift_field = None
        self.state.phaseshift_phasecenter = None
        self.state.phaseshift_timestamp = None
        # Also reset flagging state since it was based on corrupted MS
        self.state.tried_strategies = []
        self.state.successful_strategy = None
        self.state.strategy_results = {}
        self._save_state()
        logger.info("Phaseshift cache invalidated, will recreate from original MS")

    def get_or_create_phaseshifted_ms(
        self,
        field: str,
        calibrator_name: str,
        phaseshift_fn: Callable[[str, str, str], tuple[str, str]],
    ) -> tuple[str, str]:
        """Get cached phaseshifted MS or create new one.

        FAIL-FAST: Validates MS integrity before trusting cached state.
        Raises MSIntegrityError with actionable diagnostics if MS is corrupted.

        Parameters
        ----------
        field : str
            Field selection string (e.g., "0~23").
        calibrator_name : str
            Calibrator name for phaseshift.
        phaseshift_fn : Callable[[str, str, str], Tuple[str, str]]
            Function to call for phaseshift. Signature: (ms_path, field, calibrator_name) -> (phaseshifted_ms, phasecenter).
        """
        # Check if we have a cached phaseshift that matches
        if (
            self.state.phaseshifted_ms
            and self.state.phaseshift_calibrator == calibrator_name
            and self.state.phaseshift_field == field
            and Path(self.state.phaseshifted_ms).exists()
        ):
            # FAIL-FAST: Validate integrity before trusting cached state
            try:
                validate_ms_integrity(self.state.phaseshifted_ms, check_writable=True)
                logger.info(
                    f"Using cached phaseshifted MS (validated): {self.state.phaseshifted_ms} "
                    f"(calibrator={calibrator_name}, field={field})"
                )
                return self.state.phaseshifted_ms, self.state.phaseshift_phasecenter or ""
            except MSIntegrityError as e:
                # Cached MS is corrupted - invalidate cache and recreate
                logger.warning(
                    f"Cached phaseshifted MS failed integrity check, will recreate: {e.reason}"
                )
                self._invalidate_phaseshift_cache()
                # Fall through to create new phaseshift

        # Need to create new phaseshift
        logger.info(f"Creating new phaseshifted MS for calibrator {calibrator_name}, field {field}")
        start = time.time()
        phaseshifted_ms, phasecenter = phaseshift_fn(str(self.ms_path), field, calibrator_name)
        elapsed = time.time() - start

        # Cache result
        self.state.phaseshifted_ms = phaseshifted_ms
        self.state.phaseshift_calibrator = calibrator_name
        self.state.phaseshift_field = field
        self.state.phaseshift_phasecenter = phasecenter
        self.state.phaseshift_timestamp = datetime.now().isoformat()
        self._save_state()

        logger.info(f"Phaseshift completed in {elapsed:.1f}s, cached to checkpoint")
        return phaseshifted_ms, phasecenter

    def ensure_model_populated(
        self,
        cal_ms_path: str,
        calibrator_name: str,
        field: str,
        populate_fn: Callable[..., None],
        **populate_kwargs: Any,
    ) -> bool:
        """Ensure MODEL_DATA is populated, using cache if available.

        Parameters
        ----------
        cal_ms_path : str
            Path to calibration MS
        calibrator_name : str
            Calibrator name
        field : str
            Field to populate
        populate_fn : Callable[..., None]
            Function to call for population
            **populate_kwargs : Any
            Additional kwargs for populate_fn

        """
        # Check if model is already populated for this calibrator
        if self.state.model_populated and self.state.model_calibrator == calibrator_name:
            logger.info(
                f"MODEL_DATA already populated for {calibrator_name} "
                f"(flux={self.state.model_flux_jy} Jy)"
            )
            return False

        # Need to populate model
        logger.info(f"Populating MODEL_DATA for calibrator {calibrator_name}")
        start = time.time()
        populate_fn(
            cal_ms_path,
            field=field,
            calibrator_name=calibrator_name,
            **populate_kwargs,
        )
        elapsed = time.time() - start

        # Cache result
        self.state.model_populated = True
        self.state.model_calibrator = calibrator_name
        self.state.model_flux_jy = populate_kwargs.get("cal_flux_jy")
        self.state.model_timestamp = datetime.now().isoformat()
        self._save_state()

        logger.info(f"MODEL_DATA populated in {elapsed:.1f}s, cached to checkpoint")
        return True

    def mark_strategy_tried(
        self,
        strategy_name: str,
        success: bool = False,
        error: str | None = None,
        flagged_fraction: float | None = None,
        duration_s: float | None = None,
    ) -> None:
        """Mark a flagging strategy as tried.

        Parameters
        ----------
        strategy_name : str
            Name of the strategy
        success : bool, optional
            Whether calibration succeeded with this strategy
            Default is False
        error : Optional[str], optional
            Error message if failed
            Default is None
        flagged_fraction : Optional[float], optional
            Fraction of data flagged
            Default is None
        duration_s : Optional[float], optional
            How long the strategy took
            Default is None

        """
        if strategy_name not in self.state.tried_strategies:
            self.state.tried_strategies.append(strategy_name)

        self.state.strategy_results[strategy_name] = {
            "success": success,
            "error": error,
            "flagged_fraction": flagged_fraction,
            "duration_s": duration_s,
            "timestamp": datetime.now().isoformat(),
        }

        if success:
            self.state.successful_strategy = strategy_name

        self._save_state()
        logger.info(
            f"Strategy '{strategy_name}' marked as tried "
            f"(success={success}, flagged={flagged_fraction})"
        )

    def get_untried_strategies(self, all_strategies: list[str]) -> list[str]:
        """Get strategies that haven't been tried yet.

        Parameters
        ----------
        all_strategies : List[str]
            List of all available strategy names

        """
        tried = set(self.state.tried_strategies)
        return [s for s in all_strategies if s not in tried]

    def has_successful_strategy(self) -> bool:
        """Check if any strategy succeeded."""
        return self.state.successful_strategy is not None

    def get_successful_strategy(self) -> str | None:
        """Get the name of the successful strategy, if any."""
        return self.state.successful_strategy

    def reset_strategies(self) -> None:
        """Reset strategy tracking (for full retry)."""
        self.state.tried_strategies = []
        self.state.successful_strategy = None
        self.state.strategy_results = {}
        self._save_state()
        logger.info("Strategy tracking reset")

    def reset_all(self) -> None:
        """Reset all checkpoint state."""
        self.state = CheckpointState(created_at=datetime.now().isoformat())
        self._save_state()
        logger.info("All checkpoint state reset")

    def cleanup(self, remove_phaseshifted: bool = False) -> None:
        """Clean up checkpoint files.

        Parameters
        ----------
        remove_phaseshifted :
            Also remove the phaseshifted MS
        remove_phaseshifted : bool :
            (Default value = False)
        """
        # Remove checkpoint file
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info(f"Removed checkpoint file: {self.checkpoint_file}")

        # Optionally remove phaseshifted MS
        if remove_phaseshifted and self.state.phaseshifted_ms:
            ps_path = Path(self.state.phaseshifted_ms)
            if ps_path.exists():
                import shutil

                shutil.rmtree(ps_path)
                logger.info(f"Removed phaseshifted MS: {ps_path}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from checkpoint state.

        Supports legacy/helper keys by mapping them to structured state fields,
        or retrieves from custom_data.

        Parameters
        ----------
        key : str
            Key to retrieve
        default : Any, optional
            Default value if key not found

        Returns
        -------
        Any
            Value associated with key
        """
        # Map legacy/helper keys to structured state fields
        if key == "phaseshift_result":
            return self.state.phaseshifted_ms or default
        elif key == "model_populated":
            return self.state.model_populated

        # Check custom data
        return self.state.custom_data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value in checkpoint state.

        Maps legacy/helper keys to structured state fields,
        or stores in custom_data.

        Parameters
        ----------
        key : str
            Key to set
        value : Any
            Value to store
        """
        if key == "phaseshift_result":
            self.state.phaseshifted_ms = value
            if value:
                self.state.phaseshift_timestamp = datetime.now().isoformat()
        elif key == "model_populated":
            self.state.model_populated = bool(value)
            if value:
                self.state.model_timestamp = datetime.now().isoformat()
        else:
            self.state.custom_data[key] = value

        self._save_state()

    def exists(self, key: str) -> bool:
        """Check if key exists in checkpoint.

        Parameters
        ----------
        key : str
            Key to check

        Returns
        -------
        bool
            True if key exists and is truthy (for backward compatibility), False otherwise
        """
        val = self.get(key)
        return val is not None and val is not False
