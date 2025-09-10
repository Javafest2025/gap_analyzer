"""
Paper extraction models based on the Spring backend schemas.
"""

from sqlalchemy import Column, String, Text, Integer, Double, Boolean, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.core.database import Base


class PaperExtraction(Base):
    """
    Main extraction entity that holds metadata about the extraction process
    and links to all extracted content.
    """
    __tablename__ = "paper_extractions"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Paper reference (assuming we have a Paper model)
    paper_id = Column(UUID(as_uuid=True), ForeignKey("papers.id"), nullable=False)
    
    # Extraction metadata
    extraction_id = Column(String(100), unique=True, nullable=False)  # UUID from extractor service
    pdf_hash = Column(String(255))
    extraction_timestamp = Column(DateTime(timezone=True))
    
    # Core metadata
    title = Column(String(1000))
    abstract_text = Column(Text)
    language = Column(String(10))
    page_count = Column(Integer)
    
    # Processing metadata
    extraction_methods = Column(Text)  # JSON array of methods used
    processing_time = Column(Double)  # in seconds
    errors = Column(Text)  # JSON array of errors
    warnings = Column(Text)  # JSON array of warnings
    
    # Quality metrics
    extraction_coverage = Column(Double)  # 0-100%
    confidence_scores = Column(Text)  # JSON object with confidence scores
    
    # Relationships to extracted content
    sections = relationship("ExtractedSection", back_populates="paper_extraction", cascade="all, delete-orphan")
    figures = relationship("ExtractedFigure", back_populates="paper_extraction", cascade="all, delete-orphan")
    tables = relationship("ExtractedTable", back_populates="paper_extraction", cascade="all, delete-orphan")
    equations = relationship("ExtractedEquation", back_populates="paper_extraction", cascade="all, delete-orphan")
    code_blocks = relationship("ExtractedCodeBlock", back_populates="paper_extraction", cascade="all, delete-orphan")
    references = relationship("ExtractedReference", back_populates="paper_extraction", cascade="all, delete-orphan")
    entities = relationship("ExtractedEntity", back_populates="paper_extraction", cascade="all, delete-orphan")


class ExtractedSection(Base):
    """
    Entity representing extracted document sections with hierarchical structure.
    """
    __tablename__ = "extracted_sections"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    paper_extraction_id = Column(UUID(as_uuid=True), ForeignKey("paper_extractions.id"), nullable=False)
    
    # Section metadata
    section_id = Column(String(100))  # ID from extractor
    label = Column(String(50))  # e.g., "1.1", "A.1"
    title = Column(String(1000))
    section_type = Column(String(50))  # introduction, methods, results, etc.
    level = Column(Integer)  # heading level
    page_start = Column(Integer)
    page_end = Column(Integer)
    order_index = Column(Integer)  # for maintaining section order
    
    # Self-referencing for hierarchical structure
    parent_section_id = Column(UUID(as_uuid=True), ForeignKey("extracted_sections.id"))
    
    # Relationships
    paper_extraction = relationship("PaperExtraction", back_populates="sections")
    parent_section = relationship("ExtractedSection", remote_side="ExtractedSection.id", backref="subsections")
    paragraphs = relationship("ExtractedParagraph", back_populates="section", cascade="all, delete-orphan")


class ExtractedParagraph(Base):
    """
    Entity representing text paragraphs within sections.
    """
    __tablename__ = "extracted_paragraphs"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    section_id = Column(UUID(as_uuid=True), ForeignKey("extracted_sections.id"), nullable=False)
    
    # Paragraph content
    text = Column(Text)
    page = Column(Integer)
    order_index = Column(Integer)  # for maintaining paragraph order within section
    
    # Bounding box information (optional)
    bbox_x1 = Column(Double)
    bbox_y1 = Column(Double)
    bbox_x2 = Column(Double)
    bbox_y2 = Column(Double)
    
    # Style information (stored as JSON)
    style = Column(Text)  # font, size, etc. as JSON
    
    # Relationships
    section = relationship("ExtractedSection", back_populates="paragraphs")


class ExtractedFigure(Base):
    """
    Entity representing extracted figures and images.
    """
    __tablename__ = "extracted_figures"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    paper_extraction_id = Column(UUID(as_uuid=True), ForeignKey("paper_extractions.id"), nullable=False)
    
    # Figure metadata
    figure_id = Column(String(100))  # ID from extractor
    label = Column(String(100))  # e.g., "Figure 1", "Fig. 2"
    caption = Column(Text)
    page = Column(Integer)
    figure_type = Column(String(50))  # figure, chart, diagram, etc.
    
    # Bounding box
    bbox_x1 = Column(Double)
    bbox_y1 = Column(Double)
    bbox_x2 = Column(Double)
    bbox_y2 = Column(Double)
    bbox_confidence = Column(Double)
    
    # File paths
    image_path = Column(String(500))
    thumbnail_path = Column(String(500))
    
    # Note: references column removed to match actual database schema
    
    # OCR extracted text for LLM processing
    ocr_text = Column(Text)  # text extracted from the figure image
    ocr_confidence = Column(Double)  # OCR confidence score (0-1)
    order_index = Column(Integer)  # for maintaining figure order
    
    # Relationships
    paper_extraction = relationship("PaperExtraction", back_populates="figures")


class ExtractedTable(Base):
    """
    Entity representing extracted tables.
    """
    __tablename__ = "extracted_tables"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    paper_extraction_id = Column(UUID(as_uuid=True), ForeignKey("paper_extractions.id"), nullable=False)
    
    # Table metadata
    table_id = Column(String(100))  # ID from extractor
    label = Column(String(100))  # e.g., "Table 1", "Tab. 2"
    caption = Column(Text)
    page = Column(Integer)
    
    # Bounding box
    bbox_x1 = Column(Double)
    bbox_y1 = Column(Double)
    bbox_x2 = Column(Double)
    bbox_y2 = Column(Double)
    bbox_confidence = Column(Double)
    
    # Table structure (stored as JSON)
    headers = Column(Text)  # JSON array of header rows
    rows = Column(Text)  # JSON array of data rows
    structure = Column(Text)  # detailed structure information as JSON
    
    # Export formats
    csv_path = Column(String(500))
    html = Column(Text)  # HTML representation
    
    # Note: references column removed to match actual database schema
    order_index = Column(Integer)  # for maintaining table order
    
    # Relationships
    paper_extraction = relationship("PaperExtraction", back_populates="tables")


class ExtractedEquation(Base):
    """
    Entity representing extracted mathematical equations.
    """
    __tablename__ = "extracted_equations"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    paper_extraction_id = Column(UUID(as_uuid=True), ForeignKey("paper_extractions.id"), nullable=False)
    
    # Equation metadata
    equation_id = Column(String(100))  # ID from extractor
    label = Column(String(100))  # e.g., "Equation 1", "Eq. (2)"
    latex = Column(Text)  # LaTeX representation
    mathml = Column(Text)  # MathML representation (optional)
    page = Column(Integer)
    is_inline = Column(Boolean, default=False)  # inline vs display equation
    
    # Bounding box (optional)
    bbox_x1 = Column(Double)
    bbox_y1 = Column(Double)
    bbox_x2 = Column(Double)
    bbox_y2 = Column(Double)
    
    order_index = Column(Integer)  # for maintaining equation order
    
    # Relationships
    paper_extraction = relationship("PaperExtraction", back_populates="equations")


class ExtractedCodeBlock(Base):
    """
    Entity representing extracted code blocks and algorithms.
    """
    __tablename__ = "extracted_code_blocks"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    paper_extraction_id = Column(UUID(as_uuid=True), ForeignKey("paper_extractions.id"), nullable=False)
    
    # Code metadata
    code_id = Column(String(100))  # ID from extractor
    language = Column(String(50))  # programming language
    code = Column(Text)  # the actual code content
    page = Column(Integer)
    context = Column(Text)  # surrounding text for context
    has_line_numbers = Column(Boolean, default=False)
    
    # Bounding box (optional)
    bbox_x1 = Column(Double)
    bbox_y1 = Column(Double)
    bbox_x2 = Column(Double)
    bbox_y2 = Column(Double)
    
    order_index = Column(Integer)  # for maintaining code block order
    
    # Relationships
    paper_extraction = relationship("PaperExtraction", back_populates="code_blocks")


class ExtractedReference(Base):
    """
    Entity representing extracted bibliographic references.
    """
    __tablename__ = "extracted_references"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    paper_extraction_id = Column(UUID(as_uuid=True), ForeignKey("paper_extractions.id"), nullable=False)
    
    # Reference metadata
    reference_id = Column(String(100))  # ID from extractor
    raw_text = Column(Text)  # original citation text
    title = Column(String(1000))
    authors = Column(Text)  # JSON array of author names
    year = Column(Integer)
    venue = Column(String(500))  # journal, conference, etc.
    doi = Column(String(200))
    url = Column(String(1000))
    arxiv_id = Column(String(50))
    
    # Enrichment data from external APIs (stored as JSON)
    crossref_data = Column(Text)
    openalex_data = Column(Text)
    unpaywall_data = Column(Text)
    
    # Citation context
    cited_by_sections = Column(Text)  # JSON array of section IDs
    citation_count = Column(Integer, default=0)
    order_index = Column(Integer)  # for maintaining reference order
    
    # Relationships
    paper_extraction = relationship("PaperExtraction", back_populates="references")


class ExtractedEntity(Base):
    """
    Entity representing extracted named entities (organizations, locations, etc.).
    """
    __tablename__ = "extracted_entities"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    paper_extraction_id = Column(UUID(as_uuid=True), ForeignKey("paper_extractions.id"), nullable=False)
    
    # Entity metadata
    entity_id = Column(String(100))  # ID from extractor
    entity_type = Column(String(50))  # PERSON, ORGANIZATION, LOCATION, etc.
    name = Column(String(500))
    uri = Column(String(1000))  # linked data URI (optional)
    page = Column(Integer)
    context = Column(Text)  # surrounding text
    confidence = Column(Double, default=1.0)  # confidence score 0-1
    order_index = Column(Integer)  # for maintaining entity order
    
    # Relationships
    paper_extraction = relationship("PaperExtraction", back_populates="entities")
