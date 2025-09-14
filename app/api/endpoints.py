"""
FastAPI endpoints for monitoring and management.
app/api/endpoints.py
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from typing import Optional, List
from datetime import datetime, timedelta, timezone
from uuid import UUID

from app.core.database import get_db
from app.core.config import settings
from app.model.gap_models import GapAnalysis, ResearchGap, GapStatus
from app.model.paper import Paper
from app.schemas.gap_schemas import GapAnalysisResponse, GapDetail

# Create router
router = APIRouter()


# Health check endpoints
@router.get("/health")
async def health_check():
    """Basic health check."""
    return {"status": "healthy", "service": settings.APP_NAME}


@router.get("/health/detailed")
async def detailed_health_check(db: AsyncSession = Depends(get_db)):
    """Detailed health check with service status."""
    health_status = {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {}
    }
    
    # Check database
    try:
        await db.execute(text("SELECT 1"))
        health_status["checks"]["database"] = {
            "status": "up",
            "message": "Database connection successful"
        }
    except Exception as e:
        health_status["checks"]["database"] = {
            "status": "down",
            "message": str(e)
        }
        health_status["status"] = "unhealthy"
    
    # Check GROBID
    try:
        from app.main import grobid_client
        if grobid_client:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{settings.GROBID_URL}/api/isalive", timeout=5)
                if response.status_code == 200:
                    health_status["checks"]["grobid"] = {
                        "status": "up",
                        "message": "GROBID service is responsive",
                        "url": settings.GROBID_URL
                    }
                else:
                    health_status["checks"]["grobid"] = {
                        "status": "down",
                        "message": f"GROBID returned status {response.status_code}",
                        "url": settings.GROBID_URL
                    }
        else:
            health_status["checks"]["grobid"] = {
                "status": "not_initialized",
                "message": "GROBID client not initialized",
                "url": settings.GROBID_URL
            }
    except Exception as e:
        health_status["checks"]["grobid"] = {
            "status": "down",
            "message": str(e),
            "url": settings.GROBID_URL
        }
    
    # Check RabbitMQ
    try:
        from app.main import rabbitmq_service
        if rabbitmq_service and rabbitmq_service.connection:
            health_status["checks"]["rabbitmq"] = {
                "status": "up",
                "message": "RabbitMQ service is connected",
                "url": settings.rabbitmq_url
            }
        else:
            health_status["checks"]["rabbitmq"] = {
                "status": "not_connected",
                "message": "RabbitMQ service not connected",
                "url": settings.rabbitmq_url
            }
    except Exception as e:
        health_status["checks"]["rabbitmq"] = {
            "status": "down",
            "message": str(e),
            "url": settings.rabbitmq_url
        }
    
    # Check Gemini API
    try:
        if settings.GA_GEMINI_API_KEY:
            health_status["checks"]["gemini"] = {
                "status": "configured",
                "message": "Gemini API key is configured",
                "model": settings.GA_GEMINI_MODEL
            }
        else:
            health_status["checks"]["gemini"] = {
                "status": "not_configured",
                "message": "Gemini API key is not configured"
            }
    except Exception as e:
        health_status["checks"]["gemini"] = {
            "status": "error",
            "message": str(e)
        }
    
    return health_status


# Gap Analysis Management Endpoints
@router.get("/gap-analyses")
async def list_gap_analyses(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """List gap analyses with pagination."""
    query = select(GapAnalysis).order_by(GapAnalysis.created_at.desc())
    
    if status:
        query = query.where(GapAnalysis.status == status)
    
    query = query.limit(limit).offset(offset)
    
    result = await db.execute(query)
    analyses = result.scalars().all()
    
    # Get total count
    count_query = select(func.count()).select_from(GapAnalysis)
    if status:
        count_query = count_query.where(GapAnalysis.status == status)
    count_result = await db.execute(count_query)
    total = count_result.scalar()
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "analyses": [
            {
                "id": str(analysis.id),
                "paper_id": str(analysis.paper_id),
                "status": analysis.status,
                "total_gaps": analysis.total_gaps_identified,
                "valid_gaps": analysis.valid_gaps_count,
                "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
                "completed_at": analysis.completed_at.isoformat() if analysis.completed_at else None
            }
            for analysis in analyses
        ]
    }


@router.get("/gap-analyses/{analysis_id}")
async def get_gap_analysis(
    analysis_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get detailed gap analysis by ID."""
    # Fetch analysis
    result = await db.execute(
        select(GapAnalysis).where(GapAnalysis.id == analysis_id)
    )
    analysis = result.scalar_one_or_none()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Gap analysis not found")
    
    # Fetch gaps
    gaps_result = await db.execute(
        select(ResearchGap)
        .where(ResearchGap.gap_analysis_id == analysis_id)
        .order_by(ResearchGap.order_index)
    )
    gaps = gaps_result.scalars().all()
    
    return {
        "id": str(analysis.id),
        "paper_id": str(analysis.paper_id),
        "status": analysis.status,
        "total_gaps": analysis.total_gaps_identified,
        "valid_gaps": analysis.valid_gaps_count,
        "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
        "completed_at": analysis.completed_at.isoformat() if analysis.completed_at else None,
        "gaps": [
            {
                "id": str(gap.id),
                "name": gap.name,
                "category": gap.category,
                "validation_status": gap.validation_status,
                "confidence": gap.validation_confidence
            }
            for gap in gaps
        ]
    }


@router.get("/gaps/{gap_id}")
async def get_gap_details(
    gap_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get detailed information about a specific gap."""
    result = await db.execute(
        select(ResearchGap).where(ResearchGap.id == gap_id)
    )
    gap = result.scalar_one_or_none()
    
    if not gap:
        raise HTTPException(status_code=404, detail="Gap not found")
    
    return {
        "id": str(gap.id),
        "gap_id": gap.gap_id,
        "name": gap.name,
        "description": gap.description,
        "category": gap.category,
        "validation_status": gap.validation_status,
        "confidence": gap.validation_confidence,
        "potential_impact": gap.potential_impact,
        "research_hints": gap.research_hints,
        "implementation_suggestions": gap.implementation_suggestions,
        "risks_and_challenges": gap.risks_and_challenges,
        "required_resources": gap.required_resources,
        "estimated_difficulty": gap.estimated_difficulty,
        "estimated_timeline": gap.estimated_timeline,
        "evidence_anchors": gap.evidence_anchors,
        "suggested_topics": gap.suggested_topics,
        "papers_analyzed": gap.papers_analyzed_count,
        "created_at": gap.created_at.isoformat() if gap.created_at else None,
        "validated_at": gap.validated_at.isoformat() if gap.validated_at else None
    }


# Statistics Endpoints
@router.get("/stats")
async def get_statistics(
    days: int = Query(7, description="Number of days to look back"),
    db: AsyncSession = Depends(get_db)
):
    """Get service statistics."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    
    # Total analyses
    total_result = await db.execute(
        select(func.count()).select_from(GapAnalysis)
    )
    total_analyses = total_result.scalar()
    
    # Recent analyses
    recent_result = await db.execute(
        select(func.count())
        .select_from(GapAnalysis)
        .where(GapAnalysis.created_at >= since)
    )
    recent_analyses = recent_result.scalar()
    
    # Status breakdown
    status_result = await db.execute(
        select(
            GapAnalysis.status,
            func.count().label('count')
        )
        .select_from(GapAnalysis)
        .where(GapAnalysis.created_at >= since)
        .group_by(GapAnalysis.status)
    )
    status_breakdown = {row.status: row.count for row in status_result}
    
    # Gap statistics
    gap_stats_result = await db.execute(
        select(
            func.sum(GapAnalysis.total_gaps_identified).label('total_gaps'),
            func.sum(GapAnalysis.valid_gaps_count).label('valid_gaps'),
            func.avg(GapAnalysis.valid_gaps_count).label('avg_valid_gaps')
        )
        .select_from(GapAnalysis)
        .where(GapAnalysis.status == GapStatus.COMPLETED)
    )
    gap_stats = gap_stats_result.one()
    
    return {
        "period_days": days,
        "since": since.isoformat(),
        "total_analyses": total_analyses,
        "recent_analyses": recent_analyses,
        "status_breakdown": status_breakdown,
        "gap_statistics": {
            "total_gaps_identified": gap_stats.total_gaps or 0,
            "total_valid_gaps": gap_stats.valid_gaps or 0,
            "average_valid_gaps_per_paper": float(gap_stats.avg_valid_gaps or 0)
        }
    }


@router.post("/gap-analyses/{analysis_id}/retry")
async def retry_gap_analysis(
    analysis_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Retry a failed gap analysis."""
    # Fetch analysis
    result = await db.execute(
        select(GapAnalysis).where(GapAnalysis.id == analysis_id)
    )
    analysis = result.scalar_one_or_none()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Gap analysis not found")
    
    if analysis.status != GapStatus.FAILED:
        raise HTTPException(
            status_code=400,
            detail="Can only retry failed analyses"
        )
    
    # Reset status
    analysis.status = GapStatus.PENDING
    analysis.error_message = None
    analysis.started_at = None
    analysis.completed_at = None
    
    await db.commit()
    
    # Note: Re-publishing to RabbitMQ queue for processing would be implemented here
    # This would require integration with the RabbitMQ service
    
    return {
        "message": "Gap analysis queued for retry",
        "analysis_id": str(analysis_id)
    }
