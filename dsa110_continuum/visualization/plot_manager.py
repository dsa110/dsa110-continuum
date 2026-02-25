"""
Plot manager for database-tracked QA plot organization.

Integrates PlotOrganizer with database tracking for complete plot lifecycle management.
"""

import logging
import time
from datetime import datetime
from pathlib import Path

from dsa110_contimg.infrastructure.database.models import QAPlot
from dsa110_contimg.infrastructure.database.session import get_session

from .plot_organization import PlotInfo, PlotMetadata, PlotOrganizer
from dsa110_contimg.common.utils import get_env_path

logger = logging.getLogger(__name__)


class PlotManager:
    """Manages QA plots with database tracking and organized storage.

    Combines PlotOrganizer for hierarchical storage with database
    tracking for searchability and lifecycle management.

    Features:
    - Register plots with automatic path organization
    - Track plot metadata in database
    - Search plots by observation, type, date, etc.
    - Manage archival and cleanup with database updates
    - Generate metadata companion files

    """

    def __init__(
        self,
        organizer: PlotOrganizer | None = None,
        db_path: str | None = None,
    ):
        """Initialize plot manager.

        Parameters
        ----------
        organizer : Any, optional
            PlotOrganizer instance (default: from environment)
        db_path : str, optional
            Path to SQLite database (default: pipeline.sqlite3)
        """
        import os

        from .plot_organization import get_plot_organizer_from_env

        self.organizer = organizer or get_plot_organizer_from_env()
        contimg_base = str(get_env_path("CONTIMG_BASE_DIR", default="/data/dsa110-contimg"))
        self.db_path = db_path or os.environ.get(
            "PIPELINE_DB", f"{contimg_base}/state/db/pipeline.sqlite3"
        )

        # Ensure directory structure exists
        self.organizer.create_directory_structure()

    def register_plot(
        self,
        observation_id: str,
        timestamp: datetime,
        plot_type: str,
        format: str,
        plot_path: Path,
        context: str | None = None,
        generation_time_s: float | None = None,
        mid_mjd: float | None = None,
        calibrator: str | None = None,
        field_id: int | None = None,
        ra_deg: float | None = None,
        dec_deg: float | None = None,
        ms_path: str | None = None,
        image_path: str | None = None,
    ) -> QAPlot:
        """Register a plot in the database with hierarchical organization.

            This method:
            1. Moves plot to organized location (by-date directory)
            2. Creates symlinks for by-observation and by-type access
            3. Registers plot in database
            4. Updates metadata companion file

        Parameters
        ----------
        observation_id : str
            Unique observation identifier
        timestamp : datetime
            Observation timestamp
        plot_type : str
            Type of plot
        format : str
            File format (png, pdf, vega)
        plot_path : Path
            Current path to plot file
        context : Optional[str], optional
            Generation context (batch, web_api, etc.), by default None
        generation_time_s : Optional[float], optional
            Time taken to generate plot, by default None
        mid_mjd : Optional[float], optional
            Observation MJD, by default None
        calibrator : Optional[str], optional
            Calibrator name if applicable, by default None
        field_id : Optional[int], optional
            Field ID if applicable, by default None
        ra_deg : Optional[float], optional
            RA in degrees, by default None
        dec_deg : Optional[float], optional
            Dec in degrees, by default None
        ms_path : Optional[str], optional
            Associated MS path, by default None
        image_path : Optional[str], optional
            Associated image path, by default None

        """
        # Get organized path
        organized_path = self.organizer.get_plot_path(
            observation_id=observation_id,
            timestamp=timestamp,
            plot_type=plot_type,
            format=format,
            create_dirs=True,
        )

        # Move plot to organized location (if not already there)
        if plot_path != organized_path:
            if plot_path.exists():
                plot_path.rename(organized_path)
                logger.debug(f"Moved plot to organized location: {organized_path}")
            else:
                logger.warning(f"Plot file not found: {plot_path}")
                # Continue with registration using organized_path

        # Create symlinks
        self.organizer.create_symlinks(
            plot_path=organized_path,
            observation_id=observation_id,
            plot_type=plot_type,
        )

        # Get file stats
        size_bytes = organized_path.stat().st_size if organized_path.exists() else 0

        # Create database record
        with get_session("pipeline") as session:
            qa_plot = QAPlot(
                path=str(organized_path),
                filename=organized_path.name,
                observation_id=observation_id,
                plot_type=plot_type,
                format=format,
                size_bytes=size_bytes,
                generated_at=time.time(),
                context=context,
                generation_time_s=generation_time_s,
                timestamp=timestamp.isoformat(),
                mid_mjd=mid_mjd,
                calibrator=calibrator,
                field_id=field_id,
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                storage_location="by-date",
                is_archived=0,
                ms_path=ms_path,
                image_path=image_path,
            )

            session.add(qa_plot)
            session.commit()

            # Get ID before session closes
            plot_id = qa_plot.id
            plot_name = qa_plot.filename

        logger.info(f"Registered plot: {plot_name} (ID: {plot_id})")

        # Update metadata companion file
        self._update_metadata_file(observation_id, timestamp)

        # Return a detached object with key info
        return qa_plot

    def _update_metadata_file(self, observation_id: str, timestamp: datetime):
        """Update metadata companion file for an observation.

        Parameters
        ----------
        observation_id : str
            Unique observation identifier
        timestamp : datetime
            Observation timestamp

        """
        obs_dir = self.organizer.get_observation_dir(observation_id, timestamp, create=False)

        if not obs_dir.exists():
            logger.warning(f"Observation directory not found: {obs_dir}")
            return

        # Query all plots for this observation
        with get_session("pipeline") as session:
            plots = (
                session.query(QAPlot)
                .filter_by(
                    observation_id=observation_id,
                    is_archived=0,
                )
                .all()
            )

            if not plots:
                logger.debug(f"No plots found for observation: {observation_id}")
                return

            # Build plot info list
            plot_infos = []
            for plot in plots:
                plot_infos.append(
                    PlotInfo(
                        filename=plot.filename,
                        plot_type=plot.plot_type,
                        format=plot.format,
                        size_bytes=plot.size_bytes,
                        generated_at=datetime.fromtimestamp(plot.generated_at).isoformat(),
                        context=plot.context or "unknown",
                        generation_time_s=plot.generation_time_s,
                        path=plot.path,
                    )
                )

            # Use first plot's metadata for observation-level fields
            first_plot = plots[0]
            metadata = PlotMetadata(
                observation_id=observation_id,
                timestamp=first_plot.timestamp or timestamp.isoformat(),
                mid_mjd=first_plot.mid_mjd or 0.0,
                plots=plot_infos,
                calibrator=first_plot.calibrator,
                field_id=first_plot.field_id,
                ra_deg=first_plot.ra_deg,
                dec_deg=first_plot.dec_deg,
            )

            # Save metadata file
            self.organizer.save_metadata(obs_dir, metadata)

    def find_plots(
        self,
        observation_id: str | None = None,
        plot_type: str | None = None,
        format: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        context: str | None = None,
        is_archived: bool | None = None,
        limit: int | None = None,
    ) -> list[QAPlot]:
        """Search for plots in database.

        Parameters
        ----------
        observation_id :
            Filter by observation ID
        plot_type :
            Filter by plot type
        format :
            Filter by format
        start_date :
            Filter by generation date (start)
        end_date :
            Filter by generation date (end)
        context :
            Filter by context
        is_archived :
            Filter by archived status
        limit :
            Maximum number of results
        observation_id : Optional[str] :
            (Default value = None)
        plot_type : Optional[str] :
            (Default value = None)
        format : Optional[str] :
            (Default value = None)
        start_date : Optional[datetime] :
            (Default value = None)
        end_date : Optional[datetime] :
            (Default value = None)
        context : Optional[str] :
            (Default value = None)
        is_archived : Optional[bool] :
            (Default value = None)
        limit : Optional[int] :
            (Default value = None)
        observation_id : Optional[str] :
            (Default value = None)
        plot_type : Optional[str] :
            (Default value = None)
        format : Optional[str] :
            (Default value = None)
        start_date : Optional[datetime] :
            (Default value = None)
        end_date : Optional[datetime] :
            (Default value = None)
        context : Optional[str] :
            (Default value = None)
        is_archived : Optional[bool] :
            (Default value = None)
        limit : Optional[int] :
            (Default value = None)
        observation_id: Optional[str] :
             (Default value = None)
        plot_type: Optional[str] :
             (Default value = None)
        format: Optional[str] :
             (Default value = None)
        start_date: Optional[datetime] :
             (Default value = None)
        end_date: Optional[datetime] :
             (Default value = None)
        context: Optional[str] :
             (Default value = None)
        is_archived: Optional[bool] :
             (Default value = None)
        limit: Optional[int] :
             (Default value = None)

        """
        with get_session("pipeline") as session:
            query = session.query(QAPlot)

            if observation_id:
                query = query.filter(QAPlot.observation_id == observation_id)

            if plot_type:
                query = query.filter(QAPlot.plot_type == plot_type)

            if format:
                query = query.filter(QAPlot.format == format)

            if start_date:
                start_ts = start_date.timestamp()
                query = query.filter(QAPlot.generated_at >= start_ts)

            if end_date:
                end_ts = end_date.timestamp()
                query = query.filter(QAPlot.generated_at <= end_ts)

            if context:
                query = query.filter(QAPlot.context == context)

            if is_archived is not None:
                query = query.filter(QAPlot.is_archived == (1 if is_archived else 0))

            # Order by most recent first
            query = query.order_by(QAPlot.generated_at.desc())

            if limit:
                query = query.limit(limit)

            # Get results and expunge from session so they can be used after session closes
            results = query.all()
            for result in results:
                session.expunge(result)

            return results

    def get_plot_by_id(self, plot_id: int) -> QAPlot | None:
        """Get plot by database ID.

        Parameters
        ----------
        plot_id : int
            Database ID

        """
        with get_session("pipeline") as session:
            return session.query(QAPlot).filter_by(id=plot_id).first()

    def get_plot_by_path(self, path: str) -> QAPlot | None:
        """Get plot by file path.

        Parameters
        ----------
        path : str
            Full path to plot file

        """
        with get_session("pipeline") as session:
            return session.query(QAPlot).filter_by(path=path).first()

    def get_plots_for_observation(self, observation_id: str) -> list[QAPlot]:
        """Get all plots for an observation (non-archived only).

        Parameters
        ----------
        observation_id : str
            Unique observation identifier

        """
        return self.find_plots(observation_id=observation_id, is_archived=False)

    def get_plots_by_type(self, plot_type: str, limit: int | None = None) -> list[QAPlot]:
        """Get plots of a specific type.

        Parameters
        ----------
        plot_type : str
            Plot type (rfi_spectrum, psf_correlation, etc.)
        limit : Optional[int]
            Maximum number of results (Default value = None)

        """
        return self.find_plots(plot_type=plot_type, limit=limit)

    def archive_old_plots(self, dry_run: bool = False) -> int:
        """Archive plots older than retention period with database updates.

        Parameters
        ----------
        dry_run :
            If True, only report what would be archived
        dry_run : bool :
            (Default value = False)
        dry_run : bool :
            (Default value = False)
        """
        # Use organizer to move files
        archived_count = self.organizer.archive_old_plots(dry_run=dry_run)

        if not dry_run and archived_count > 0:
            # Update database records
            with get_session("pipeline") as session:
                # Find plots that were archived
                cutoff_date = datetime.now() - timedelta(days=self.organizer.retention_days)
                cutoff_ts = cutoff_date.timestamp()

                archived_plots = (
                    session.query(QAPlot)
                    .filter(
                        QAPlot.generated_at < cutoff_ts,
                        QAPlot.is_archived == 0,
                    )
                    .all()
                )

                for plot in archived_plots:
                    plot.is_archived = 1
                    plot.archived_at = time.time()
                    plot.storage_location = "archive"

                session.commit()
                logger.info(f"Updated {len(archived_plots)} plot records as archived")

        return archived_count

    def cleanup_old_archives(self, dry_run: bool = False) -> int:
        """Delete archived plots older than archive retention period with database updates.

        Parameters
        ----------
        dry_run :
            If True, only report what would be deleted
        dry_run : bool :
            (Default value = False)
        dry_run : bool :
            (Default value = False)
        """
        # Use organizer to delete files
        deleted_count = self.organizer.cleanup_old_archives(dry_run=dry_run)

        if not dry_run and deleted_count > 0:
            # Delete database records
            with get_session("pipeline") as session:
                # Find archived plots older than archive retention
                cutoff_date = datetime.now() - timedelta(days=self.organizer.archive_retention_days)
                cutoff_ts = cutoff_date.timestamp()

                deleted_plots = (
                    session.query(QAPlot)
                    .filter(
                        QAPlot.is_archived == 1,
                        QAPlot.archived_at < cutoff_ts,
                    )
                    .delete()
                )

                session.commit()
                logger.info(f"Deleted {deleted_plots} plot records from database")

        return deleted_count

    def get_stats(self) -> dict:
        """Get statistics about tracked plots."""
        with get_session("pipeline") as session:
            total = session.query(QAPlot).count()
            active = session.query(QAPlot).filter_by(is_archived=0).count()
            archived = session.query(QAPlot).filter_by(is_archived=1).count()

            # Plot types
            plot_types = (
                session.query(QAPlot.plot_type, func.count(QAPlot.id))
                .group_by(QAPlot.plot_type)
                .all()
            )

            # Formats
            formats = (
                session.query(QAPlot.format, func.count(QAPlot.id)).group_by(QAPlot.format).all()
            )

            return {
                "total_plots": total,
                "active_plots": active,
                "archived_plots": archived,
                "by_type": dict(plot_types),
                "by_format": dict(formats),
                "retention_days": self.organizer.retention_days,
                "archive_retention_days": self.organizer.archive_retention_days,
            }


# Import for timedelta
from datetime import timedelta

from sqlalchemy import func
