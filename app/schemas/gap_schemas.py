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
    requestId: str = Field(alias="request_id")
    correlationId: str = Field(alias="correlation_id")
    status: str
    message: str
    gapAnalysisId: Optional[str] = Field(default=None, alias="gap_analysis_id")
    totalGaps: int = Field(default=0, alias="total_gaps")
    validGaps: int = Field(default=0, alias="valid_gaps")
    gaps: Optional[List['GapDetail']] = None
    error: Optional[str] = None
    completedAt: Optional[datetime] = Field(default=None, alias="completed_at")
    
    class Config:
        populate_by_name = True  # Allow both camelCase and snake_case


class GapDetail(BaseModel):
    """Detailed information about a research gap"""
    gapId: str = Field(alias="gap_id")
    name: str
    description: str
    category: str
    validationStatus: str = Field(alias="validation_status")
    confidenceScore: float = Field(alias="confidence_score")
    
    # Expanded information
    potentialImpact: Optional[str] = Field(default=None, alias="potential_impact")
    researchHints: Optional[str] = Field(default=None, alias="research_hints")
    implementationSuggestions: Optional[str] = Field(default=None, alias="implementation_suggestions")
    risksAndChallenges: Optional[str] = Field(default=None, alias="risks_and_challenges")
    requiredResources: Optional[str] = Field(default=None, alias="required_resources")
    estimatedDifficulty: Optional[str] = Field(default=None, alias="estimated_difficulty")
    estimatedTimeline: Optional[str] = Field(default=None, alias="estimated_timeline")
    
    # Evidence
    evidenceAnchors: List[Dict[str, str]] = Field(default_factory=list, alias="evidence_anchors")
    supportingPapersCount: int = Field(default=0, alias="supporting_papers_count")
    conflictingPapersCount: int = Field(default=0, alias="conflicting_papers_count")
    
    # Topics
    suggestedTopics: List['ResearchTopic'] = Field(default_factory=list, alias="suggested_topics")
    
    class Config:
        populate_by_name = True  # Allow both camelCase and snake_case


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