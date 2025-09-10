"""
Pydantic schemas for gap analysis.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from uuid import UUID
from enum import Enum


class GapAnalysisRequest(BaseModel):
    """Request model for gap analysis from RabbitMQ"""
    paperId: str
    paperExtractionId: str
    correlationId: str
    requestId: str
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    class Config:
        # Allow field name aliases for backward compatibility
        populate_by_name = True


class GapAnalysisResponse(BaseModel):
    """Response model for gap analysis to RabbitMQ"""
    request_id: str
    correlation_id: str
    status: str
    message: str
    gap_analysis_id: Optional[str] = None
    total_gaps: int = 0
    valid_gaps: int = 0
    gaps: Optional[List['GapDetail']] = None
    error: Optional[str] = None
    completed_at: Optional[datetime] = None


class GapDetail(BaseModel):
    """Detailed information about a research gap"""
    gap_id: str
    name: str
    description: str
    category: str
    validation_status: str
    confidence_score: float
    
    # Expanded information
    potential_impact: Optional[str] = None
    research_hints: Optional[str] = None
    implementation_suggestions: Optional[str] = None
    risks_and_challenges: Optional[str] = None
    required_resources: Optional[str] = None
    estimated_difficulty: Optional[str] = None
    estimated_timeline: Optional[str] = None
    
    # Evidence
    evidence_anchors: List[Dict[str, str]] = Field(default_factory=list)
    supporting_papers_count: int = 0
    conflicting_papers_count: int = 0
    
    # Topics
    suggested_topics: List['ResearchTopic'] = Field(default_factory=list)


class ResearchTopic(BaseModel):
    """Suggested research topic based on a gap"""
    title: str
    description: str
    research_questions: List[str]
    methodology_suggestions: Optional[str] = None
    expected_outcomes: Optional[str] = None
    relevance_score: float = 0.0


class InitialGap(BaseModel):
    """Initial gap identified by AI"""
    name: str
    description: str
    category: str
    reasoning: str
    evidence: str


class ValidationResult(BaseModel):
    """Result of gap validation"""
    is_valid: bool
    confidence: float
    reasoning: str
    should_modify: bool
    modification_suggestion: Optional[str] = None
    supporting_papers: List[Dict[str, str]] = Field(default_factory=list)
    conflicting_papers: List[Dict[str, str]] = Field(default_factory=list)


class SearchQuery(BaseModel):
    """Search query for finding related papers"""
    query: str
    filters: Dict[str, Any] = Field(default_factory=dict)
    max_results: int = 10


class PaperSearchResult(BaseModel):
    """Result from paper search"""
    title: str
    abstract: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    publication_date: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    venue: Optional[str] = None


class ExtractedContent(BaseModel):
    """Content extracted from a paper"""
    title: str
    abstract: Optional[str] = None
    sections: List[Dict[str, str]] = Field(default_factory=list)
    conclusion: Optional[str] = None
    methods: Optional[str] = None
    results: Optional[str] = None
    extraction_success: bool = True
    error: Optional[str] = None


# Forward references are handled automatically in Pydantic v2