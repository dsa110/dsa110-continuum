"""
Plot organization and path management for QA plots.

This module provides utilities for organizing plots in a hierarchical structure
with automatic path detection, metadata tracking, and archival management.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from dsa110_contimg.common.utils import get_env_path

logger = logging.getLogger(__name__)


@dataclass
class PlotInfo:
    """Metadata for a single plot file."""

    filename: str
    plot_type: str
    format: str
    size_bytes: int
    generated_at: str  # ISO 8601 timestamp
    context: str
    generation_time_s: float | None = None
    path: str | None = None  # Full path for database tracking


@dataclass
class PlotMetadata:
    """Metadata for a collection of plots for an observation."""

    observation_id: str
    timestamp: str  # ISO 8601
    mid_mjd: float
    plots: list[PlotInfo]
    calibrator: str | None = None
    field_id: int | None = None
    ra_deg: float | None = None
    dec_deg: float | None = None


class PlotOrganizer:
    """Manages hierarchical organization of QA plots with automatic path detection.

    Creates and maintains directory structure:
    - by-date/YYYY/MM/DD/obs_id/
    - by-observation/obs_id/ (symlinks)
    - by-type/plot_type/ (symlinks)
    - archive/YYYY/MM/DD/ (archived plots)

    Features:
    - Automatic path generation based on observation metadata
    - Context-aware organization using detect_context_from_path()
    - Metadata companion file generation
    - Symlink creation for multiple access patterns
    - Archival management with configurable retention

    """

    def __init__(
        self,
        base_dir: Path,
        retention_days: int = 7,
        archive_retention_days: int = 90,
    ):
        """Initialize plot organizer.

        Parameters
        ----------
        base_dir : str
            Base directory for plots (e.g., /data/dsa110-contimg/products/qa/plots)
        retention_days : int
            Days before archiving plots (default: 7)
        archive_retention_days : int
            Days before deleting archived plots (default: 90)
        """
        self.base_dir = Path(base_dir)
        self.retention_days = retention_days
        self.archive_retention_days = archive_retention_days

        # Directory structure
        self.by_date_dir = self.base_dir / "by-date"
        self.by_obs_dir = self.base_dir / "by-observation"
        self.by_type_dir = self.base_dir / "by-type"
        self.archive_dir = self.base_dir / "archive"

    def create_directory_structure(self):
        """Create base directory structure for organized plots."""
        for directory in [
            self.by_date_dir,
            self.by_obs_dir,
            self.by_type_dir,
            self.archive_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created plot directory structure in {self.base_dir}")

    def get_observation_dir(
        self,
        observation_id: str,
        timestamp: datetime,
        create: bool = True,
    ) -> Path:
        """Get primary directory path for an observation's plots.

        Parameters
        ----------
        observation_id : str
            Unique observation identifier
        timestamp : datetime
            Observation timestamp
        create : bool
            Create directory if it doesn't exist (Default value = True)

        """
        date_path = self.by_date_dir / timestamp.strftime("%Y/%m/%d") / observation_id

        if create:
            date_path.mkdir(parents=True, exist_ok=True)

        return date_path

    def generate_plot_filename(
        self,
        observation_id: str,
        timestamp: datetime,
        plot_type: str,
        format: str,
    ) -> str:
        """Generate standardized plot filename.

        Format: {observation_id}_{timestamp}_{plot_type}.{extension}

        Parameters
        ----------
        observation_id : str
            Unique observation identifier
        timestamp : datetime
            Observation timestamp
        plot_type : str
            Type of plot (rfi_spectrum, psf_correlation, etc.)
        format : str
            File format (png, pdf, vega.json)

        """
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
        extension = "vega.json" if format == "vega" else format
        return f"{observation_id}_{ts_str}_{plot_type}.{extension}"

    def get_plot_path(
        self,
        observation_id: str,
        timestamp: datetime,
        plot_type: str,
        format: str,
        create_dirs: bool = True,
    ) -> Path:
        """Get full path for a plot file.

        Parameters
        ----------
        observation_id : str
            Unique observation identifier
        timestamp : datetime
            Observation timestamp
        plot_type : str
            Type of plot
        format : str
            File format
        create_dirs : bool
            Create parent directories if they don't exist (Default value = True)

        """
        obs_dir = self.get_observation_dir(observation_id, timestamp, create=create_dirs)
        filename = self.generate_plot_filename(observation_id, timestamp, plot_type, format)
        return obs_dir / filename

    def create_symlinks(
        self,
        plot_path: Path,
        observation_id: str,
        plot_type: str,
    ):
        """Create symlinks for by-observation and by-type access patterns.

        Parameters
        ----------
        plot_path : Path
            Full path to plot file in by-date directory
        observation_id : str
            Unique observation identifier
        plot_type : str
            Type of plot

        """
        # by-observation symlink
        obs_link_dir = self.by_obs_dir / observation_id
        obs_link_dir.mkdir(parents=True, exist_ok=True)
        obs_link = obs_link_dir / plot_path.name

        if not obs_link.exists():
            try:
                obs_link.symlink_to(plot_path)
                logger.debug(f"Created by-observation symlink: {obs_link}")
            except FileExistsError:
                pass  # Symlink already exists
            except Exception as e:
                logger.warning(f"Failed to create by-observation symlink: {e}")

        # by-type symlink
        type_link_dir = self.by_type_dir / plot_type
        type_link_dir.mkdir(parents=True, exist_ok=True)
        type_link = type_link_dir / plot_path.name

        if not type_link.exists():
            try:
                type_link.symlink_to(plot_path)
                logger.debug(f"Created by-type symlink: {type_link}")
            except FileExistsError:
                pass  # Symlink already exists
            except Exception as e:
                logger.warning(f"Failed to create by-type symlink: {e}")

    def save_metadata(
        self,
        observation_dir: Path,
        metadata: PlotMetadata,
    ):
        """Save metadata companion file for an observation's plots.

        Parameters
        ----------
        observation_dir : Path
            Directory containing the plots
        metadata : PlotMetadata
            Plot metadata to save

        """
        metadata_path = observation_dir / "metadata.json"

        try:
            with open(metadata_path, "w") as f:
                json.dump(asdict(metadata), f, indent=2)
            logger.debug(f"Saved plot metadata: {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save plot metadata: {e}")

    def load_metadata(self, observation_dir: Path) -> PlotMetadata | None:
        """Load metadata companion file from observation directory.

        Parameters
        ----------
        observation_dir : Path
            Directory containing metadata.json

        """
        metadata_path = observation_dir / "metadata.json"

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path) as f:
                data = json.load(f)

            # Convert plot dicts back to PlotInfo objects
            plots = [PlotInfo(**p) for p in data.pop("plots")]
            return PlotMetadata(plots=plots, **data)
        except Exception as e:
            logger.error(f"Failed to load plot metadata: {e}")
            return None

    def find_plots_by_observation(self, observation_id: str) -> list[Path]:
        """Find all plots for an observation.

        Parameters
        ----------
        observation_id : str
            Unique observation identifier

        """
        obs_dir = self.by_obs_dir / observation_id

        if not obs_dir.exists():
            return []

        plots = []
        for pattern in ["*.png", "*.pdf", "*.vega.json"]:
            plots.extend(obs_dir.glob(pattern))

        return plots

    def find_plots_by_type(self, plot_type: str) -> list[Path]:
        """Find all plots of a specific type.

        Parameters
        ----------
        plot_type : str
            Type of plot (rfi_spectrum, psf_correlation, etc.)

        """
        type_dir = self.by_type_dir / plot_type

        if not type_dir.exists():
            return []

        plots = []
        for pattern in ["*.png", "*.pdf", "*.vega.json"]:
            plots.extend(type_dir.glob(pattern))

        return plots

    def find_plots_by_date(
        self,
        start_date: datetime,
        end_date: datetime | None = None,
    ) -> list[Path]:
        """Find all plots within a date range.

        Parameters
        ----------
        start_date : datetime
            Start of date range
        end_date : Optional[datetime]
            End of date range (default: start_date) (Default value = None)

        """
        if end_date is None:
            end_date = start_date

        plots = []
        current = start_date

        while current <= end_date:
            date_path = self.by_date_dir / current.strftime("%Y/%m/%d")

            if date_path.exists():
                for obs_dir in date_path.iterdir():
                    if obs_dir.is_dir():
                        for pattern in ["*.png", "*.pdf", "*.vega.json"]:
                            plots.extend(obs_dir.glob(pattern))

            current += timedelta(days=1)

        return plots

    def archive_old_plots(self, dry_run: bool = False) -> int:
        """Archive plots older than retention period.

        Moves plots from by-date/ to archive/, preserving date structure.
        Removes symlinks from by-observation/ and by-type/.

        Parameters
        ----------
        dry_run :
            If True, only report what would be archived
        dry_run : bool :
            (Default value = False)
        dry_run : bool :
            (Default value = False)
        """
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        archived_count = 0

        # Find old plots in by-date directory
        for year_dir in self.by_date_dir.iterdir():
            if not year_dir.is_dir():
                continue

            for month_dir in year_dir.iterdir():
                if not month_dir.is_dir():
                    continue

                for day_dir in month_dir.iterdir():
                    if not day_dir.is_dir():
                        continue

                    # Parse date from path
                    try:
                        date_str = f"{year_dir.name}/{month_dir.name}/{day_dir.name}"
                        dir_date = datetime.strptime(date_str, "%Y/%m/%d")
                    except ValueError:
                        continue

                    if dir_date < cutoff_date:
                        # Archive this entire day
                        for obs_dir in day_dir.iterdir():
                            if not obs_dir.is_dir():
                                continue

                            if dry_run:
                                logger.info(f"Would archive: {obs_dir}")
                                archived_count += len(list(obs_dir.glob("*")))
                            else:
                                archived_count += self._archive_observation_dir(obs_dir)

        logger.info(f"Archived {archived_count} plots older than {self.retention_days} days")
        return archived_count

    def _archive_observation_dir(self, obs_dir: Path) -> int:
        """Archive an entire observation directory.

        Parameters
        ----------
        obs_dir : Path
            Observation directory to archive

        """
        # Extract date from path: by-date/YYYY/MM/DD/obs_id
        date_parts = obs_dir.parts[-4:-1]  # YYYY, MM, DD
        observation_id = obs_dir.name

        # Create archive destination
        archive_dest = self.archive_dir / "/".join(date_parts) / observation_id
        archive_dest.mkdir(parents=True, exist_ok=True)

        # Move all files
        archived_count = 0
        for file_path in obs_dir.iterdir():
            if file_path.is_file():
                dest_path = archive_dest / file_path.name
                try:
                    file_path.rename(dest_path)
                    archived_count += 1
                except Exception as e:
                    logger.error(f"Failed to archive {file_path}: {e}")

        # Remove symlinks
        self._remove_symlinks(observation_id)

        # Remove empty observation directory
        try:
            obs_dir.rmdir()
        except OSError:
            pass  # Directory not empty

        logger.info(f"Archived {archived_count} files from {obs_dir} to {archive_dest}")
        return archived_count

    def _remove_symlinks(self, observation_id: str):
        """Remove symlinks for an archived observation.

        Parameters
        ----------
        observation_id : str
            Observation identifier

        """
        # Remove by-observation symlinks
        obs_link_dir = self.by_obs_dir / observation_id
        if obs_link_dir.exists():
            try:
                for link in obs_link_dir.iterdir():
                    if link.is_symlink():
                        link.unlink()
                obs_link_dir.rmdir()
            except Exception as e:
                logger.warning(f"Failed to remove by-observation symlinks: {e}")

        # Remove by-type symlinks (find symlinks pointing to archived obs)
        for type_dir in self.by_type_dir.iterdir():
            if not type_dir.is_dir():
                continue

            for link in type_dir.iterdir():
                if link.is_symlink():
                    # Check if symlink target is for this observation
                    if observation_id in str(link.resolve()):
                        try:
                            link.unlink()
                        except Exception as e:
                            logger.warning(f"Failed to remove by-type symlink {link}: {e}")

    def cleanup_old_archives(self, dry_run: bool = False) -> int:
        """Delete archived plots older than archive retention period.

        Parameters
        ----------
        dry_run :
            If True, only report what would be deleted
        dry_run : bool :
            (Default value = False)
        dry_run : bool :
            (Default value = False)
        """
        cutoff_date = datetime.now() - timedelta(days=self.archive_retention_days)
        deleted_count = 0

        for year_dir in self.archive_dir.iterdir():
            if not year_dir.is_dir():
                continue

            for month_dir in year_dir.iterdir():
                if not month_dir.is_dir():
                    continue

                for day_dir in month_dir.iterdir():
                    if not day_dir.is_dir():
                        continue

                    # Parse date from path
                    try:
                        date_str = f"{year_dir.name}/{month_dir.name}/{day_dir.name}"
                        dir_date = datetime.strptime(date_str, "%Y/%m/%d")
                    except ValueError:
                        continue

                    if dir_date < cutoff_date:
                        # Delete this entire day
                        for obs_dir in day_dir.iterdir():
                            if not obs_dir.is_dir():
                                continue

                            if dry_run:
                                logger.info(f"Would delete: {obs_dir}")
                                deleted_count += len(list(obs_dir.glob("*")))
                            else:
                                deleted_count += self._delete_directory(obs_dir)

        logger.info(
            f"Deleted {deleted_count} archived plots older than {self.archive_retention_days} days"
        )
        return deleted_count

    def _delete_directory(self, directory: Path) -> int:
        """Recursively delete a directory and all its contents.

        Parameters
        ----------
        directory : Path
            Directory to delete
        """
        deleted_count = 0

        for item in directory.iterdir():
            if item.is_file():
                try:
                    item.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete {item}: {e}")
            elif item.is_dir():
                deleted_count += self._delete_directory(item)

        try:
            directory.rmdir()
        except Exception as e:
            logger.error(f"Failed to remove directory {directory}: {e}")

        return deleted_count

    def get_retention_config(self) -> dict:
        """Get current retention configuration."""
        return {
            "retention_days": self.retention_days,
            "archive_retention_days": self.archive_retention_days,
            "base_dir": str(self.base_dir),
        }


def get_plot_organizer_from_env() -> PlotOrganizer:
    """Create PlotOrganizer from environment variables.

    Environment variables:
    - QA_PLOT_BASE_DIR (default: /data/dsa110-contimg/products/qa/plots)
    - QA_PLOT_RETENTION_DAYS (default: 7)
    - QA_PLOT_ARCHIVE_RETENTION_DAYS (default: 90)

    """
    from dsa110_contimg.common.utils import get_env_int

    contimg_base = str(get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg"))
    base_dir = Path(os.environ.get("QA_PLOT_BASE_DIR", f"{contimg_base}/products/qa/plots"))

    retention_days = get_env_int("QA_PLOT_RETENTION_DAYS", default=7)
    archive_retention_days = get_env_int("QA_PLOT_ARCHIVE_RETENTION_DAYS", default=90)

    return PlotOrganizer(
        base_dir=base_dir,
        retention_days=retention_days,
        archive_retention_days=archive_retention_days,
    )
