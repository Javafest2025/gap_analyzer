"""
Gap analysis models for storing research gaps.
"""

from sqlalchemy import Column, String, Text, Integer, Double, Boolean, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from enum import Enum
import uuid

from app.core.database import Base


class GapStatus(str, Enum):
    """Status of gap analysis"""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class GapValidationStatus(str, Enum):
    """Validation status of individual gaps"""
    INITIAL = "INITIAL"
    VALIDATING = "VALIDATING"
    VALID = "VALID"
    INVALID = "INVALID"
    MODIFIED = "MODIFIED"


class GapAnalysis(Base):
    """
    Main gap analysis entity that holds the analysis process and results.
    """
    __tablename__ = "gap_analyses"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Paper reference
    paper_id = Column(UUID(as_uuid=True), ForeignKey("papers.id"), nullable=False)
    paper_extraction_id = Column(UUID(as_uuid=True), ForeignKey("paper_extractions.id"), nullable=False)
    
    # Request tracking
    correlation_id = Column(Text, unique=True, nullable=False)
    request_id = Column(Text, unique=True, nullable=False)
    
    # Analysis metadata
    status = Column(Text, default=GapStatus.PENDING)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    error_message = Column(Text)
    
    # Analysis configuration
    config = Column(JSON)  # Store analysis parameters
    
    # Summary statistics
    total_gaps_identified = Column(Integer, default=0)
    valid_gaps_count = Column(Integer, default=0)
    invalid_gaps_count = Column(Integer, default=0)
    modified_gaps_count = Column(Integer, default=0)
    
    # Relationships
    gaps = relationship("ResearchGap", back_populates="gap_analysis", cascade="all, delete-orphan")
    paper = relationship("Paper", backref="gap_analyses")
    paper_extraction = relationship("PaperExtraction", backref="gap_analyses")


class ResearchGap(Base):
    """
    Individual research gap identified in the analysis.
    """
    __tablename__ = "research_gaps"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    gap_analysis_id = Column(UUID(as_uuid=True), ForeignKey("gap_analyses.id"), nullable=False)
    
    # Gap identification
    gap_id = Column(Text, unique=True, nullable=False)
    order_index = Column(Integer)
    
    # Core gap information
    name = Column(Text)
    description = Column(Text)
    category = Column(Text)  # theoretical, methodological, empirical, etc.
    
    # Validation status
    validation_status = Column(Text, default=GapValidationStatus.INITIAL)
    validation_confidence = Column(Double)  # 0-1 confidence score
    
    # Initial analysis
    initial_reasoning = Column(Text)
    initial_evidence = Column(Text)
    
    # Validation details
    validation_query = Column(Text)  # Search query used for validation
    papers_analyzed_count = Column(Integer, default=0)
    validation_reasoning = Column(Text)
    modification_history = Column(JSON)  # Track how the gap was modified
    
    # Expanded information (after validation)
    potential_impact = Column(Text)
    research_hints = Column(Text)
    implementation_suggestions = Column(Text)
    risks_and_challenges = Column(Text)
    required_resources = Column(Text)
    estimated_difficulty = Column(Text)  # low, medium, high
    estimated_timeline = Column(Text)  # e.g., "6-12 months"
    
    # Evidence and references
    evidence_anchors = Column(JSON)  # Links to papers analyzed
    supporting_papers = Column(JSON)  # Papers that support this gap
    conflicting_papers = Column(JSON)  # Papers that conflict with this gap
    
    # Suggested research topics
    suggested_topics = Column(JSON)  # List of topic suggestions
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    validated_at = Column(DateTime(timezone=True))
    
    # Relationships
    gap_analysis = relationship("GapAnalysis", back_populates="gaps")
    validation_papers = relationship("GapValidationPaper", back_populates="research_gap", cascade="all, delete-orphan")


class GapValidationPaper(Base):
    """
    Papers analyzed during gap validation.
    """
    __tablename__ = "gap_validation_papers"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    research_gap_id = Column(UUID(as_uuid=True), ForeignKey("research_gaps.id"), nullable=False)
    
    # Paper information
    title = Column(Text)
    doi = Column(Text)
    url = Column(Text)
    publication_date = Column(DateTime)
    
    # Extraction status
    extraction_status = Column(Text)
    extracted_text = Column(Text)
    extraction_error = Column(Text)
    
    # Relevance analysis
    relevance_score = Column(Double)  # 0-1
    relevance_reasoning = Column(Text)
    supports_gap = Column(Boolean)
    conflicts_with_gap = Column(Boolean)
    key_findings = Column(Text)
    
    # Relationships
    research_gap = relationship("ResearchGap", back_populates="validation_papers")


class GapTopic(Base):
    """
    Suggested research topics based on gaps.
    """
    __tablename__ = "gap_topics"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    research_gap_id = Column(UUID(as_uuid=True), ForeignKey("research_gaps.id"), nullable=False)
    
    # Topic information
    title = Column(Text)
    description = Column(Text)
    research_questions = Column(JSON)  # List of research questions
    methodology_suggestions = Column(Text)
    expected_outcomes = Column(Text)
    relevance_score = Column(Double)  # 0-1
    
    # Relationships
    research_gap = relationship("ResearchGap", backref="topics")