"""
Sliding-window mosaic trigger for real-time streaming.

This module implements automatic mosaic triggering based on tile accumulation.
It creates 12-tile mosaics every 6 tiles (50% overlap), ensuring edge tiles
in one mosaic become center tiles in the next.

Design:
- STRIDE = 6: Trigger every 6 new tiles
- WINDOW = 12: Each mosaic contains 12 tiles
- DEC_TOLERANCE = 1.0°: Threshold for detecting pointing changes
- SLEW_STABILITY = 3: Consecutive tiles needed to confirm new pointing

Key features:
- Strict dec-band handling: discard tiles during slew transitions
- State recovery: reconstruct from images table on service restart
- Single active dec-band: DSA-110 observes at fixed dec per session

Usage:
    from dsa110_contimg.core.mosaic.trigger import SlidingWindowTrigger

    trigger = SlidingWindowTrigger(db_path=Path("pipeline.sqlite3"))

    # Record a new tile after imaging completes
    result = trigger.record_tile(image_id=123, dec_deg=30.0, mjd=60000.5)

    if result.should_trigger:
        # Get the 12 tiles for this mosaic
        tile_ids = result.mosaic_tile_ids
        # Trigger mosaic creation...
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration Constants
# =============================================================================

# Number of tiles between mosaic triggers
STRIDE = 6

# Number of tiles per mosaic
WINDOW = 12

# Declination tolerance for same pointing (degrees)
DEC_TOLERANCE = 1.0

# Consecutive tiles required to confirm stable pointing after slew
SLEW_STABILITY_COUNT = 3


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TileRecordResult:
    """Result of recording a new tile."""

    # Whether a mosaic should be triggered
    should_trigger: bool = False

    # Image IDs for the mosaic (WINDOW tiles)
    mosaic_tile_ids: list[int] | None = None

    # Current tile count toward next trigger
    current_tile_count: int = 0

    # Whether the tile was discarded (slew transition)
    discarded: bool = False

    # Reason for discarding (if applicable)
    discard_reason: str | None = None

    # Current pointing state
    pointing_stable: bool = False
    active_dec_deg: float | None = None


@dataclass
class TriggerState:
    """Current state of the sliding-window trigger."""

    active_dec_deg: float | None
    tile_count: int
    last_tile_mjd: float | None
    last_tile_image_id: int | None
    pointing_stable_since_mjd: float | None
    consecutive_dec_count: int
    pointing_stable: bool
    last_mosaic_id: int | None
    last_mosaic_mjd: float | None
    updated_at: float


# =============================================================================
# Sliding Window Trigger
# =============================================================================


class SlidingWindowTrigger:
    """Manages sliding-window mosaic triggering.

    This class tracks tile accumulation and determines when to trigger
    mosaic creation. It handles:
    - Counting tiles toward the stride threshold
    - Detecting pointing changes and discarding slew tiles
    - Recovering state from the database on restart

    Parameters
    ----------
    db_path : Path
        Path to the unified pipeline database.
    stride : int
        Number of tiles between mosaic triggers (default: 6).
    window : int
        Number of tiles per mosaic (default: 12).
    dec_tolerance : float
        Declination tolerance for same pointing in degrees (default: 0.5).
    slew_stability_count : int
        Consecutive tiles needed to confirm stable pointing (default: 3).
    """

    def __init__(
        self,
        db_path: Path,
        stride: int | None = None,
        window: int | None = None,
        dec_tolerance: float = DEC_TOLERANCE,
        slew_stability_count: int = SLEW_STABILITY_COUNT,
    ):
        self.db_path = Path(db_path)
        self.window = window if window is not None else WINDOW
        self.stride = stride if stride is not None else STRIDE
        self.dec_tolerance = dec_tolerance
        self.slew_stability_count = slew_stability_count

        logger.info(
            "SlidingWindowTrigger initialized: stride=%s, window=%s, "
            "dec_tolerance=%s°, slew_stability=%s",
            self.stride,
            self.window,
            dec_tolerance,
            slew_stability_count,
        )

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_state(self) -> TriggerState | None:
        """Get current trigger state from database.

        Returns
        -------
        TriggerState or None
            Current state, or None if no state exists.
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM mosaic_trigger_state ORDER BY id DESC LIMIT 1"
            ).fetchone()

            if row is None:
                return None

            return TriggerState(
                active_dec_deg=row["active_dec_deg"],
                tile_count=row["tile_count"],
                last_tile_mjd=row["last_tile_mjd"],
                last_tile_image_id=row["last_tile_image_id"],
                pointing_stable_since_mjd=row["pointing_stable_since_mjd"],
                consecutive_dec_count=row["consecutive_dec_count"],
                pointing_stable=bool(row["pointing_stable"]),
                last_mosaic_id=row["last_mosaic_id"],
                last_mosaic_mjd=row["last_mosaic_mjd"],
                updated_at=row["updated_at"],
            )

    def _update_state(
        self,
        conn: sqlite3.Connection,
        active_dec_deg: float | None,
        tile_count: int,
        last_tile_mjd: float | None,
        last_tile_image_id: int | None,
        pointing_stable_since_mjd: float | None,
        consecutive_dec_count: int,
        pointing_stable: bool,
        last_mosaic_id: int | None = None,
        last_mosaic_mjd: float | None = None,
    ) -> None:
        """Update trigger state in database."""
        now = time.time()

        # Check if state row exists
        existing = conn.execute("SELECT id FROM mosaic_trigger_state LIMIT 1").fetchone()

        if existing:
            conn.execute(
                """
                UPDATE mosaic_trigger_state SET
                    active_dec_deg = ?,
                    tile_count = ?,
                    last_tile_mjd = ?,
                    last_tile_image_id = ?,
                    pointing_stable_since_mjd = ?,
                    consecutive_dec_count = ?,
                    pointing_stable = ?,
                    last_mosaic_id = COALESCE(?, last_mosaic_id),
                    last_mosaic_mjd = COALESCE(?, last_mosaic_mjd),
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    active_dec_deg,
                    tile_count,
                    last_tile_mjd,
                    last_tile_image_id,
                    pointing_stable_since_mjd,
                    consecutive_dec_count,
                    pointing_stable,
                    last_mosaic_id,
                    last_mosaic_mjd,
                    now,
                    existing["id"],
                ),
            )
        else:
            conn.execute(
                """
                INSERT INTO mosaic_trigger_state (
                    active_dec_deg, tile_count, last_tile_mjd, last_tile_image_id,
                    pointing_stable_since_mjd, consecutive_dec_count, pointing_stable,
                    last_mosaic_id, last_mosaic_mjd, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    active_dec_deg,
                    tile_count,
                    last_tile_mjd,
                    last_tile_image_id,
                    pointing_stable_since_mjd,
                    consecutive_dec_count,
                    pointing_stable,
                    last_mosaic_id,
                    last_mosaic_mjd,
                    now,
                ),
            )

    def _register_tile(
        self,
        conn: sqlite3.Connection,
        image_id: int,
        dec_deg: float,
        mjd: float,
    ) -> None:
        """Register a tile in the trigger tiles table."""
        conn.execute(
            """
            INSERT INTO mosaic_trigger_tiles (image_id, dec_deg, mjd, registered_at)
            VALUES (?, ?, ?, ?)
            """,
            (image_id, dec_deg, mjd, time.time()),
        )

    def _get_pending_tiles(
        self,
        conn: sqlite3.Connection,
        dec_deg: float,
        limit: int,
    ) -> list[int]:
        """Get pending (un-mosaicked) tile image IDs for a declination band.

        Returns tiles in MJD order (oldest first).
        """
        rows = conn.execute(
            """
            SELECT image_id FROM mosaic_trigger_tiles
            WHERE used_in_mosaic = 0
              AND ABS(dec_deg - ?) < ?
            ORDER BY mjd ASC
            LIMIT ?
            """,
            (dec_deg, self.dec_tolerance, limit),
        ).fetchall()

        return [row["image_id"] for row in rows]

    def _count_pending_tiles(
        self,
        conn: sqlite3.Connection,
        dec_deg: float,
    ) -> int:
        """Count pending tiles for a declination band."""
        row = conn.execute(
            """
            SELECT COUNT(*) as cnt FROM mosaic_trigger_tiles
            WHERE used_in_mosaic = 0
              AND ABS(dec_deg - ?) < ?
            """,
            (dec_deg, self.dec_tolerance),
        ).fetchone()

        return row["cnt"] if row else 0

    def _mark_tiles_used(
        self,
        conn: sqlite3.Connection,
        image_ids: list[int],
        mosaic_id: int | None,
    ) -> None:
        """Mark tiles as used in a mosaic.

        Note: For sliding window, only mark the oldest STRIDE tiles as used,
        keeping WINDOW - STRIDE tiles for the next mosaic.
        """
        # Only mark the first STRIDE tiles as "used"
        # The remaining tiles will be reused in the next mosaic
        tiles_to_mark = image_ids[: self.stride]

        for image_id in tiles_to_mark:
            conn.execute(
                """
                UPDATE mosaic_trigger_tiles
                SET used_in_mosaic = 1, mosaic_id = ?
                WHERE image_id = ?
                """,
                (mosaic_id, image_id),
            )

    def record_tile(
        self,
        image_id: int,
        dec_deg: float,
        mjd: float,
    ) -> TileRecordResult:
        """Record a new tile and check if mosaic should be triggered.

        This is the main entry point called after each imaging task completes.

        Parameters
        ----------
        image_id : int
            ID of the newly created image.
        dec_deg : float
            Declination of the tile in degrees.
        mjd : float
            Modified Julian Date of the observation.

        Returns
        -------
        TileRecordResult
            Result indicating whether to trigger a mosaic and related state.
        """
        with self._get_connection() as conn:
            # Get current state
            state = self.get_state()

            # Handle first tile ever
            if state is None:
                logger.info(f"First tile recorded: image_id={image_id}, dec={dec_deg}°")
                self._register_tile(conn, image_id, dec_deg, mjd)
                self._update_state(
                    conn,
                    active_dec_deg=dec_deg,
                    tile_count=1,
                    last_tile_mjd=mjd,
                    last_tile_image_id=image_id,
                    pointing_stable_since_mjd=mjd,
                    consecutive_dec_count=1,
                    pointing_stable=False,  # Need SLEW_STABILITY_COUNT to confirm
                )
                conn.commit()

                return TileRecordResult(
                    should_trigger=False,
                    current_tile_count=1,
                    discarded=False,
                    pointing_stable=False,
                    active_dec_deg=dec_deg,
                )

            # Check for pointing change
            dec_changed = (
                state.active_dec_deg is None
                or abs(dec_deg - state.active_dec_deg) > self.dec_tolerance
            )

            if dec_changed:
                # Pointing has changed - check if transitioning or establishing new stable
                if state.pointing_stable or state.active_dec_deg is None:
                    # Was stable, now transitioning to new dec
                    logger.info(
                        f"Pointing change detected: {state.active_dec_deg}° -> {dec_deg}°. "
                        f"Starting slew transition."
                    )
                    # Reset to new dec, start counting stability
                    self._update_state(
                        conn,
                        active_dec_deg=dec_deg,
                        tile_count=0,  # Reset tile count
                        last_tile_mjd=mjd,
                        last_tile_image_id=image_id,
                        pointing_stable_since_mjd=None,  # Not stable yet
                        consecutive_dec_count=1,
                        pointing_stable=False,
                    )
                    conn.commit()

                    return TileRecordResult(
                        should_trigger=False,
                        current_tile_count=0,
                        discarded=True,
                        discard_reason=f"Pointing change from {state.active_dec_deg}° to {dec_deg}°",
                        pointing_stable=False,
                        active_dec_deg=dec_deg,
                    )
                else:
                    # Was already transitioning, check if this is same new dec
                    logger.warning(
                        f"Slew continues: expected dec={state.active_dec_deg}°, got {dec_deg}°"
                    )
                    # Update to new dec, reset consecutive count
                    self._update_state(
                        conn,
                        active_dec_deg=dec_deg,
                        tile_count=0,
                        last_tile_mjd=mjd,
                        last_tile_image_id=image_id,
                        pointing_stable_since_mjd=None,
                        consecutive_dec_count=1,
                        pointing_stable=False,
                    )
                    conn.commit()

                    return TileRecordResult(
                        should_trigger=False,
                        current_tile_count=0,
                        discarded=True,
                        discard_reason=f"Still slewing, dec={dec_deg}°",
                        pointing_stable=False,
                        active_dec_deg=dec_deg,
                    )

            # Pointing is consistent with active dec
            new_consecutive = state.consecutive_dec_count + 1

            # Check if pointing just became stable
            just_became_stable = (
                not state.pointing_stable and new_consecutive >= self.slew_stability_count
            )

            if just_became_stable:
                logger.info(f"Pointing now stable at dec={dec_deg}° after {new_consecutive} tiles")

            pointing_stable = state.pointing_stable or just_became_stable
            pointing_stable_since = (
                state.pointing_stable_since_mjd if state.pointing_stable else mjd
            )

            # Only count tiles if pointing is stable
            if not pointing_stable:
                # Still in slew transition, discard tile
                self._update_state(
                    conn,
                    active_dec_deg=dec_deg,
                    tile_count=0,
                    last_tile_mjd=mjd,
                    last_tile_image_id=image_id,
                    pointing_stable_since_mjd=None,
                    consecutive_dec_count=new_consecutive,
                    pointing_stable=False,
                )
                conn.commit()

                return TileRecordResult(
                    should_trigger=False,
                    current_tile_count=0,
                    discarded=True,
                    discard_reason=f"Stabilizing pointing ({new_consecutive}/{self.slew_stability_count})",
                    pointing_stable=False,
                    active_dec_deg=dec_deg,
                )

            # Pointing is stable - register tile and check trigger
            self._register_tile(conn, image_id, dec_deg, mjd)

            # Count pending tiles
            pending_count = self._count_pending_tiles(conn, dec_deg)

            # Update state
            self._update_state(
                conn,
                active_dec_deg=dec_deg,
                tile_count=pending_count,
                last_tile_mjd=mjd,
                last_tile_image_id=image_id,
                pointing_stable_since_mjd=pointing_stable_since,
                consecutive_dec_count=new_consecutive,
                pointing_stable=True,
            )

            # Check if we should trigger a mosaic
            # Trigger when we have at least WINDOW tiles, and new tiles >= STRIDE
            if pending_count >= self.window:
                # Get the tiles for this mosaic
                mosaic_tile_ids = self._get_pending_tiles(conn, dec_deg, self.window)

                logger.info(
                    f"Mosaic trigger! {pending_count} tiles pending, "
                    f"creating mosaic with {len(mosaic_tile_ids)} tiles"
                )

                conn.commit()

                return TileRecordResult(
                    should_trigger=True,
                    mosaic_tile_ids=mosaic_tile_ids,
                    current_tile_count=pending_count,
                    discarded=False,
                    pointing_stable=True,
                    active_dec_deg=dec_deg,
                )

            conn.commit()

            tiles_until_trigger = self.window - pending_count
            logger.debug(
                f"Tile recorded: image_id={image_id}, pending={pending_count}, "
                f"need {tiles_until_trigger} more for trigger"
            )

            return TileRecordResult(
                should_trigger=False,
                current_tile_count=pending_count,
                discarded=False,
                pointing_stable=True,
                active_dec_deg=dec_deg,
            )

    def mark_mosaic_complete(
        self,
        mosaic_id: int,
        tile_image_ids: list[int],
        mosaic_mjd: float,
    ) -> None:
        """Mark a mosaic as complete, updating tile and state tracking.

        This should be called after mosaic creation succeeds. It marks
        the oldest STRIDE tiles as used (but keeps WINDOW - STRIDE tiles
        pending for the next mosaic, implementing the sliding window).

        Parameters
        ----------
        mosaic_id : int
            Database ID of the created mosaic.
        tile_image_ids : List[int]
            Image IDs of tiles included in the mosaic.
        mosaic_mjd : float
            MJD of the mosaic.
        """
        with self._get_connection() as conn:
            # Mark oldest STRIDE tiles as used
            self._mark_tiles_used(conn, tile_image_ids, mosaic_id)

            # Update state with mosaic info
            state = self.get_state()
            if state:
                new_pending = self._count_pending_tiles(conn, state.active_dec_deg or 0.0)
                self._update_state(
                    conn,
                    active_dec_deg=state.active_dec_deg,
                    tile_count=new_pending,
                    last_tile_mjd=state.last_tile_mjd,
                    last_tile_image_id=state.last_tile_image_id,
                    pointing_stable_since_mjd=state.pointing_stable_since_mjd,
                    consecutive_dec_count=state.consecutive_dec_count,
                    pointing_stable=state.pointing_stable,
                    last_mosaic_id=mosaic_id,
                    last_mosaic_mjd=mosaic_mjd,
                )

            conn.commit()

            logger.info(
                f"Mosaic {mosaic_id} complete: marked {self.stride} tiles as used, "
                f"{self.window - self.stride} tiles retained for next mosaic"
            )

    def reconstruct_state(self) -> TriggerState:
        """Reconstruct trigger state from images table on service restart.

        This queries the images table to find recent tiles and rebuilds
        the trigger state. Called during worker initialization.

        Returns
        -------
        TriggerState
            Reconstructed state.
        """
        logger.info("Reconstructing trigger state from images table...")

        with self._get_connection() as conn:
            # Find most recent images with dec info
            # Look back ~2 hours worth of tiles (24 tiles at 5 min each)
            rows = conn.execute(
                """
                SELECT
                    i.id,
                    i.center_dec_deg,
                    m.mid_mjd
                FROM images i
                JOIN ms_index m ON i.ms_path = m.path
                WHERE i.center_dec_deg IS NOT NULL
                  AND m.mid_mjd IS NOT NULL
                ORDER BY m.mid_mjd DESC
                LIMIT 24
                """
            ).fetchall()

            if not rows:
                logger.info("No recent images found, starting with clean state")
                self._update_state(
                    conn,
                    active_dec_deg=None,
                    tile_count=0,
                    last_tile_mjd=None,
                    last_tile_image_id=None,
                    pointing_stable_since_mjd=None,
                    consecutive_dec_count=0,
                    pointing_stable=False,
                )
                conn.commit()
                return self.get_state()

            # Analyze pointing stability
            decs = [row["center_dec_deg"] for row in rows]
            mjds = [row["mid_mjd"] for row in rows]

            # Find most recent stable dec (consecutive tiles within tolerance)
            stable_dec = decs[0]
            consecutive = 1
            for i in range(1, len(decs)):
                if abs(decs[i] - stable_dec) <= self.dec_tolerance:
                    consecutive += 1
                else:
                    break

            pointing_stable = consecutive >= self.slew_stability_count

            # Check which tiles are already registered
            existing_tiles = set()
            for row in conn.execute("SELECT image_id FROM mosaic_trigger_tiles"):
                existing_tiles.add(row["image_id"])

            # Register any unregistered tiles that match stable dec
            registered_count = 0
            for row in reversed(rows):  # Process oldest first
                if row["id"] not in existing_tiles:
                    if abs(row["center_dec_deg"] - stable_dec) <= self.dec_tolerance:
                        self._register_tile(conn, row["id"], row["center_dec_deg"], row["mid_mjd"])
                        registered_count += 1

            if registered_count > 0:
                logger.info(f"Registered {registered_count} tiles during state reconstruction")

            # Count pending tiles
            pending_count = self._count_pending_tiles(conn, stable_dec)

            # Update state
            self._update_state(
                conn,
                active_dec_deg=stable_dec,
                tile_count=pending_count,
                last_tile_mjd=mjds[0],
                last_tile_image_id=rows[0]["id"],
                pointing_stable_since_mjd=mjds[consecutive - 1] if pointing_stable else None,
                consecutive_dec_count=consecutive,
                pointing_stable=pointing_stable,
            )

            conn.commit()

            logger.info(
                f"State reconstructed: dec={stable_dec}°, pending={pending_count}, "
                f"stable={pointing_stable}"
            )

            return self.get_state()

    def reset_state(self) -> None:
        """Reset all trigger state. Use with caution.

        This clears both the state table and all pending tiles.
        Primarily for testing or manual intervention.
        """
        with self._get_connection() as conn:
            conn.execute("DELETE FROM mosaic_trigger_tiles")
            conn.execute("DELETE FROM mosaic_trigger_state")
            conn.commit()
            logger.warning("Trigger state reset - all pending tiles cleared")
