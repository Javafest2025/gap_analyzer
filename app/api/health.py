"""
Health check endpoints for the Gap Analyzer API.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from loguru import logger
from datetime import datetime, timezone

from app.core.database import get_db
from app.core.config import settings

router = APIRouter()


@router.get("/")
async def health_check():
    """
    Basic health check endpoint.
    Returns the current status of the API.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": settings.APP_VERSION,
        "service": settings.APP_NAME
    }


@router.get("/ready")
async def readiness_check(db: AsyncSession = Depends(get_db)):
    """
    Readiness check endpoint for Kubernetes/container orchestration.
    Returns 200 if the service is ready to accept traffic.
    """
    try:
        # Check if database is accessible
        await db.execute(text("SELECT 1"))
        return {"status": "ready", "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/live")
async def liveness_check():
    """
    Liveness check endpoint for Kubernetes/container orchestration.
    Returns 200 if the service is alive.
    """
    return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}
