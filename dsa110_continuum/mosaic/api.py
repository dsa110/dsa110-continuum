"""
FastAPI endpoints for mosaic operations.

Two endpoints:
- POST /api/v1/mosaics/create - Start mosaic creation
- GET /api/v1/mosaics/status/{name} - Check status
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mosaics", tags=["mosaic"])


class MosaicRequest(BaseModel):
    """Request to create a mosaic."""

    name: str = Field(..., description="Unique name for the mosaic")
    start_time: int = Field(..., description="Start time (Unix timestamp)")
    end_time: int = Field(..., description="End time (Unix timestamp)")
    tier: str = Field(default="science", description="Mosaic tier: quicklook, science, or deep")


class MosaicResponse(BaseModel):
    """Response for mosaic creation request."""

    status: str
    execution_id: str | None = None
    message: str


class MosaicStatusResponse(BaseModel):
    """Response for mosaic status query."""

    name: str
    status: str
    tier: str
    n_images: int
    mosaic_path: str | None = None
    qa_status: str | None = None
    created_at: int | None = None


# Configuration - will be set by app startup
_config: dict[str, Any] = {}


def configure_mosaic_api(
    database_path: Path,
    mosaic_dir: Path,
    images_table: str = "images",
) -> None:
    """Configure the mosaic API with paths.

    Parameters
    ----------
    database_path : Path
        Path to the unified database
    mosaic_dir : Path
        Directory for output mosaics
    images_table : str, optional
        Name of the images table (default is "images")
    """
    global _config
    _config = {
        "database_path": database_path,
        "mosaic_dir": mosaic_dir,
        "images_table": images_table,
    }
    logger.info(f"Mosaic API configured: db={database_path}, dir={mosaic_dir}")


def _get_config() -> dict[str, Any]:
    """Get current configuration."""
    if not _config:
        raise HTTPException(
            status_code=500, detail="Mosaic API not configured. Call configure_mosaic_api() first."
        )
    return _config


@router.get("")
async def list_mosaics() -> dict[str, Any]:
    """
    List all mosaics.

    Returns paginated list of mosaics.
    """
    config = _get_config()

    try:
        conn = sqlite3.connect(str(config["database_path"]), timeout=30.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT DISTINCT name, status, tier, COUNT(*) as n_images
            FROM mosaics
            GROUP BY name, status, tier
            ORDER BY created_at DESC
            LIMIT 100
            """
        )
        rows = cursor.fetchall()
        conn.close()

        return {
            "total": len(rows),
            "items": [
                {
                    "name": row["name"],
                    "status": row["status"],
                    "tier": row["tier"],
                    "n_images": row["n_images"],
                }
                for row in rows
            ],
        }
    except sqlite3.OperationalError:
        # Mosaics table might not exist yet
        return {"total": 0, "items": []}


@router.post("/create", response_model=MosaicResponse)
async def create_mosaic(request: MosaicRequest) -> MosaicResponse:
    """Create a mosaic from a time range.

        This is the ONLY mosaic creation endpoint. Simple API.

        The mosaic will be created asynchronously. Use the status endpoint
        to check progress and get the result.

    Parameters
    ----------
    request : Mosaic creation request
        Request with name, time range, and tier

    Returns
    -------
        Response
        Response with execution status

    Raises
    ------
        HTTPException
        If request is invalid or creation fails
    """
    config = _get_config()

    # Validate time range
    if request.end_time <= request.start_time:
        raise HTTPException(
            status_code=400, detail="Invalid time range: end_time must be after start_time"
        )

    # Validate tier
    valid_tiers = ["quicklook", "science", "deep"]
    if request.tier.lower() not in valid_tiers:
        raise HTTPException(
            status_code=400, detail=f"Invalid tier: {request.tier}. Must be one of: {valid_tiers}"
        )

    # Check if name already exists
    try:
        conn = sqlite3.connect(str(config["database_path"]))
        conn.row_factory = sqlite3.Row

        cursor = conn.execute("SELECT id FROM mosaic_plans WHERE name = ?", (request.name,))
        existing = cursor.fetchone()

        if existing:
            conn.close()
            raise HTTPException(
                status_code=409, detail=f"Mosaic with name '{request.name}' already exists"
            )

        conn.close()
    except sqlite3.Error as e:
        logger.warning(f"Database check failed: {e}")
        # Continue anyway - the pipeline will handle duplicates

    # Start pipeline asynchronously
    try:
        # For now, execute synchronously in background
        # In production, would use Dagster asset materialization
        import asyncio

        params = {
            "database_path": str(config["database_path"]),
            "mosaic_dir": str(config["mosaic_dir"]),
            "images_table": config["images_table"],
            "name": request.name,
            "start_time": request.start_time,
            "end_time": request.end_time,
            "tier": request.tier.lower(),
        }

        # Create background task
        asyncio.create_task(_run_mosaic_background(params))

        return MosaicResponse(
            status="accepted",
            execution_id=request.name,
            message=f"Mosaic creation started: {request.name}",
        )

    except Exception as e:
        logger.exception(f"Failed to start mosaic creation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start mosaic creation: {e}")


async def _run_mosaic_background(params: dict[str, Any]) -> None:
    """Run mosaic pipeline in background."""
    from .pipeline import execute_mosaic_pipeline_task

    try:
        result = await execute_mosaic_pipeline_task(params)
        logger.info(f"Background mosaic complete: {result}")
    except Exception as e:
        logger.exception(f"Background mosaic failed: {e}")


@router.get("/status/{name}", response_model=MosaicStatusResponse)
async def get_mosaic_status(name: str) -> MosaicStatusResponse:
    """Query mosaic build status.

    Parameters
    ----------
    name : str
        Mosaic name to query

    Returns
    -------
        Status response
        Current state and results

    Raises
    ------
        HTTPException
        If mosaic not found
    """
    config = _get_config()

    try:
        conn = sqlite3.connect(str(config["database_path"]))
        conn.row_factory = sqlite3.Row

        # Get plan
        cursor = conn.execute("SELECT * FROM mosaic_plans WHERE name = ?", (name,))
        plan = cursor.fetchone()

        if not plan:
            conn.close()
            raise HTTPException(status_code=404, detail=f"Mosaic '{name}' not found")

        # Get mosaic if completed
        mosaic = None
        if plan["status"] == "completed":
            cursor = conn.execute("SELECT * FROM mosaics WHERE plan_id = ?", (plan["id"],))
            mosaic = cursor.fetchone()

        conn.close()

        return MosaicStatusResponse(
            name=name,
            status=plan["status"],
            tier=plan["tier"],
            n_images=plan["n_images"],
            mosaic_path=mosaic["path"] if mosaic else None,
            qa_status=mosaic["qa_status"] if mosaic else None,
            created_at=plan["created_at"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get mosaic status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get mosaic status: {e}")


@router.get("/list")
async def list_mosaic_plans(
    tier: str | None = None,
    status: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """List mosaics with optional filtering.

    Parameters
    ----------
    tier : str, optional
        Filter by tier (quicklook, science, deep)
    status : str, optional
        Filter by status (pending, building, completed, failed)
    limit : int, optional
        Maximum number of results

    Returns
    -------
        list
        List of mosaic records
    """
    config = _get_config()

    try:
        conn = sqlite3.connect(str(config["database_path"]))
        conn.row_factory = sqlite3.Row

        # Build query
        query = "SELECT * FROM mosaic_plans WHERE 1=1"
        params = []

        if tier:
            query += " AND tier = ?"
            params.append(tier.lower())

        if status:
            query += " AND status = ?"
            params.append(status.lower())

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(query, params)
        plans = cursor.fetchall()

        # Get mosaic details for completed plans
        results = []
        for plan in plans:
            record = dict(plan)

            if plan["status"] == "completed":
                cursor = conn.execute(
                    "SELECT path, qa_status FROM mosaics WHERE plan_id = ?", (plan["id"],)
                )
                mosaic = cursor.fetchone()
                if mosaic:
                    record["mosaic_path"] = mosaic["path"]
                    record["qa_status"] = mosaic["qa_status"]

            # Parse image_ids JSON
            record["image_ids"] = json.loads(record.get("image_ids", "[]"))

            results.append(record)

        conn.close()
        return results

    except Exception as e:
        logger.exception(f"Failed to list mosaics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list mosaics: {e}")


@router.delete("/{name}")
async def delete_mosaic(name: str, delete_file: bool = False) -> dict[str, str]:
    """Delete a mosaic record (and optionally the file).

    Parameters
    ----------
    name : str
        Mosaic name to delete
    delete_file : bool, optional
        Whether to also delete the FITS file

    Returns
    -------
        str
        Confirmation message
    """
    config = _get_config()

    try:
        conn = sqlite3.connect(str(config["database_path"]))
        conn.row_factory = sqlite3.Row

        # Get plan
        cursor = conn.execute("SELECT * FROM mosaic_plans WHERE name = ?", (name,))
        plan = cursor.fetchone()

        if not plan:
            conn.close()
            raise HTTPException(status_code=404, detail=f"Mosaic '{name}' not found")

        plan_id = plan["id"]

        # Get mosaic record if exists
        cursor = conn.execute("SELECT * FROM mosaics WHERE plan_id = ?", (plan_id,))
        mosaic = cursor.fetchone()

        # Delete file if requested
        if delete_file and mosaic and mosaic["path"]:
            mosaic_path = Path(mosaic["path"])
            if mosaic_path.exists():
                mosaic_path.unlink()
                logger.info(f"Deleted mosaic file: {mosaic_path}")

        # Delete database records
        if mosaic:
            conn.execute("DELETE FROM mosaic_qa WHERE mosaic_id = ?", (mosaic["id"],))
            conn.execute("DELETE FROM mosaics WHERE id = ?", (mosaic["id"],))

        conn.execute("DELETE FROM mosaic_plans WHERE id = ?", (plan_id,))

        conn.commit()
        conn.close()

        return {"message": f"Deleted mosaic: {name}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to delete mosaic: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete mosaic: {e}")


@router.get("/{name}")
async def get_mosaic_detail(name: str) -> dict[str, Any]:
    """Get detailed information about a mosaic.

    Parameters
    ----------
    name : str
        Mosaic name

    Returns
    -------
        dict
        Mosaic details including path and metadata
    """
    config = _get_config()

    try:
        conn = sqlite3.connect(str(config["database_path"]))
        conn.row_factory = sqlite3.Row

        # Get plan
        cursor = conn.execute("SELECT * FROM mosaic_plans WHERE name = ?", (name,))
        plan = cursor.fetchone()

        if not plan:
            conn.close()
            raise HTTPException(status_code=404, detail=f"Mosaic '{name}' not found")

        # Get mosaic
        cursor = conn.execute("SELECT * FROM mosaics WHERE plan_id = ?", (plan["id"],))
        mosaic = cursor.fetchone()

        if not mosaic:
            conn.close()
            raise HTTPException(
                status_code=404,
                detail=f"Mosaic data for '{name}' not found (status: {plan['status']})",
            )

        result = dict(mosaic)
        result["name"] = plan["name"]
        result["status"] = plan["status"]
        result["tier"] = plan["tier"]
        result["created_at"] = plan["created_at"]
        result["image_ids"] = json.loads(plan["image_ids"])

        if result["qa_details"]:
            result["qa_details"] = json.loads(result["qa_details"])

        conn.close()
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get mosaic detail: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get mosaic detail: {e}")


@router.get("/{name}/fits")
async def download_mosaic_fits(name: str):
    """Download mosaic FITS file.

    Parameters
    ----------
    name : str
        Mosaic name

    Returns
    -------
        FileResponse
        Response with the FITS file
    """
    config = _get_config()

    try:
        conn = sqlite3.connect(str(config["database_path"]))
        conn.row_factory = sqlite3.Row

        # Get plan
        cursor = conn.execute("SELECT id FROM mosaic_plans WHERE name = ?", (name,))
        plan = cursor.fetchone()

        if not plan:
            conn.close()
            raise HTTPException(status_code=404, detail=f"Mosaic '{name}' not found")

        # Get mosaic
        cursor = conn.execute("SELECT path FROM mosaics WHERE plan_id = ?", (plan["id"],))
        mosaic = cursor.fetchone()
        conn.close()

        if not mosaic or not mosaic["path"]:
            raise HTTPException(status_code=404, detail=f"Mosaic file for '{name}' not found")

        file_path = Path(mosaic["path"])
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Mosaic file not found on disk")

        return FileResponse(
            path=file_path,
            media_type="application/fits",
            filename=f"{name}.fits",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to download mosaic FITS: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download mosaic FITS: {e}")
