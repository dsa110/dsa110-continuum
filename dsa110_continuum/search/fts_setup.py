"""Full-Text Search Setup.

Creates and manages SQLite FTS5 virtual tables for searching jobs, alerts, and sources.
"""

import logging

from dsa110_contimg.infrastructure.database.connection import get_db_connection

logger = logging.getLogger(__name__)


def setup_fts_tables():
    """Create FTS5 virtual tables for search."""
    conn = get_db_connection()
    try:
        # Jobs search table
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS jobs_fts USING fts5(
                job_id UNINDEXED,
                name,
                stage,
                error_message,
                worker_id,
                content=jobs,
                content_rowid=id
            )
        """)

        # Alerts search table
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS alerts_fts USING fts5(
                alert_id UNINDEXED,
                title,
                message,
                source,
                severity,
                content=alerts,
                content_rowid=id
            )
        """)

        # Sources search table
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS sources_fts USING fts5(
                source_id UNINDEXED,
                name,
                source_type,
                nvss_name,
                vlass_name,
                content=sources,
                content_rowid=id
            )
        """)

        # Create triggers to keep FTS tables in sync
        _create_sync_triggers(conn)

        conn.commit()
        logger.info("FTS5 search tables created successfully")
    finally:
        conn.close()


def _create_sync_triggers(conn):
    """Create triggers to auto-update FTS tables."""
    # Jobs triggers
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS jobs_ai AFTER INSERT ON jobs BEGIN
            INSERT INTO jobs_fts(rowid, job_id, name, stage, error_message, worker_id)
            VALUES (new.id, new.id, new.name, new.stage, new.error_message, new.worker_id);
        END
    """)

    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS jobs_ad AFTER DELETE ON jobs BEGIN
            DELETE FROM jobs_fts WHERE rowid = old.id;
        END
    """)

    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS jobs_au AFTER UPDATE ON jobs BEGIN
            UPDATE jobs_fts SET 
                name = new.name,
                stage = new.stage,
                error_message = new.error_message,
                worker_id = new.worker_id
            WHERE rowid = old.id;
        END
    """)

    # Alerts triggers
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS alerts_ai AFTER INSERT ON alerts BEGIN
            INSERT INTO alerts_fts(rowid, alert_id, title, message, source, severity)
            VALUES (new.id, new.id, new.title, new.message, new.source, new.severity);
        END
    """)

    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS alerts_ad AFTER DELETE ON alerts BEGIN
            DELETE FROM alerts_fts WHERE rowid = old.id;
        END
    """)

    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS alerts_au AFTER UPDATE ON alerts BEGIN
            UPDATE alerts_fts SET 
                title = new.title,
                message = new.message,
                source = new.source,
                severity = new.severity
            WHERE rowid = old.id;
        END
    """)

    # Sources triggers
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS sources_ai AFTER INSERT ON sources BEGIN
            INSERT INTO sources_fts(rowid, source_id, name, source_type, nvss_name, vlass_name)
            VALUES (new.id, new.id, new.name, new.source_type, new.nvss_name, new.vlass_name);
        END
    """)

    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS sources_ad AFTER DELETE ON sources BEGIN
            DELETE FROM sources_fts WHERE rowid = old.id;
        END
    """)

    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS sources_au AFTER UPDATE ON sources BEGIN
            UPDATE sources_fts SET 
                name = new.name,
                source_type = new.source_type,
                nvss_name = new.nvss_name,
                vlass_name = new.vlass_name
            WHERE rowid = old.id;
        END
    """)


def rebuild_fts_indexes():
    """Rebuild FTS5 indexes from current database content."""
    conn = get_db_connection()
    try:
        # Clear and rebuild jobs_fts
        conn.execute("DELETE FROM jobs_fts")
        conn.execute("""
            INSERT INTO jobs_fts(rowid, job_id, name, stage, error_message, worker_id)
            SELECT id, id, name, stage, error_message, worker_id FROM jobs
        """)

        # Clear and rebuild alerts_fts
        conn.execute("DELETE FROM alerts_fts")
        conn.execute("""
            INSERT INTO alerts_fts(rowid, alert_id, title, message, source, severity)
            SELECT id, id, title, message, source, severity FROM alerts
        """)

        # Clear and rebuild sources_fts
        conn.execute("DELETE FROM sources_fts")
        conn.execute("""
            INSERT INTO sources_fts(rowid, source_id, name, source_type, nvss_name, vlass_name)
            SELECT id, id, name, source_type, nvss_name, vlass_name FROM sources
        """)

        conn.commit()
        logger.info("FTS5 indexes rebuilt successfully")
    finally:
        conn.close()
