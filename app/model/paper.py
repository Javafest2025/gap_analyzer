"""
Paper model for the Gap Analyzer application.
Based on the Spring backend Paper entity.
"""

from sqlalchemy import Column, String, Text, DateTime, Boolean, Double, Date
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.core.database import Base


class Paper(Base):
    """
    Paper entity representing academic papers.
    Based on the Spring backend Paper entity structure.
    """
    __tablename__ = "papers"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Correlation ID reference - each paper belongs to a search operation
    correlation_id = Column(String(100), nullable=False)
    
    # Core Fields
    title = Column(String(500), nullable=False)
    abstract_text = Column(Text)
    publication_date = Column(Date)
    doi = Column(String(100))
    
    # Identifiers and Source Information
    semantic_scholar_id = Column(String(100))
    source = Column(String(50))
    
    # PDF and Access Information
    pdf_content_url = Column(String(500))
    pdf_url = Column(String(500))
    is_open_access = Column(Boolean)
    paper_url = Column(String(500))
    
    # Publication Types (stored as comma-separated values for simplicity)
    publication_types = Column(String(200))
    
    # Fields of Study (stored as comma-separated values for simplicity)
    fields_of_study = Column(Text)
    
    # Extraction-related fields
    is_extracted = Column(Boolean, default=False)
    extraction_status = Column(String(50))  # PENDING, PROCESSING, COMPLETED, FAILED
    extraction_job_id = Column(String(100))
    extraction_started_at = Column(DateTime(timezone=True))
    extraction_completed_at = Column(DateTime(timezone=True))
    extraction_error = Column(Text)
    extraction_coverage = Column(Double)  # 0-100%
    
    # Summarization-related fields
    is_summarized = Column(Boolean, default=False)
    summarization_status = Column(String(50))  # PENDING, PROCESSING, COMPLETED, FAILED
    summarization_started_at = Column(DateTime(timezone=True))
    summarization_completed_at = Column(DateTime(timezone=True))
    summarization_error = Column(Text)
    
    # LaTeX Context field - indicates if paper is added to LaTeX context for a project
    is_latex_context = Column(Boolean, default=False)
    
    # Relationships (keeping only the essential one for now)
    extractions = relationship("PaperExtraction", backref="paper", cascade="all, delete-orphan")
