"""
Main gap analysis service that orchestrates the entire process.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from loguru import logger
import json

from app.model.gap_models import (
    GapAnalysis, ResearchGap, GapValidationPaper,
    GapStatus, GapValidationStatus, GapTopic
)
from app.model.paper import Paper
from app.model.paper_extraction import (
    PaperExtraction, ExtractedSection, ExtractedParagraph,
    ExtractedFigure, ExtractedTable
)
from app.schemas.gap_schemas import (
    GapAnalysisRequest, GapAnalysisResponse,
    GapDetail, ResearchTopic, InitialGap
)
from app.services.gemini_service import GeminiService
from app.services.search_service import WebSearchService
from app.services.grobid_client import GrobidClient
from app.utils.helpers import AsyncBatchProcessor


class GapAnalysisService:
    """Main service for performing gap analysis on papers."""
    
    def __init__(
        self,
        gemini_api_key: str,
        grobid_url: str
    ):
        self.gemini_service = GeminiService(gemini_api_key)
        self.search_service = WebSearchService()
        self.grobid_client = GrobidClient(grobid_url)
        
        # Initialize batch processor for gap validation
        self.batch_processor = AsyncBatchProcessor(
            batch_size=5,  # Process 5 gaps at a time
            max_concurrent=2  # Maximum 2 concurrent batches
        )
    
    async def analyze_paper(
        self,
        request: GapAnalysisRequest,
        session: AsyncSession
    ) -> GapAnalysisResponse:
        """Main method to analyze a paper for research gaps."""
        analysis = None
        
        try:
            # Create gap analysis record
            analysis = await self._create_gap_analysis(request, session)
            
            # Fetch paper and extraction data
            paper_data, extracted_content = await self._fetch_paper_data(
                request.paper_id,
                request.paper_extraction_id,
                session
            )
            
            if not paper_data:
                raise ValueError("Paper not found")
            
            # Generate initial gaps
            logger.info("Generating initial gaps...")
            initial_gaps = await self.gemini_service.generate_initial_gaps(
                paper_data,
                extracted_content
            )
            
            if not initial_gaps:
                raise ValueError("No gaps could be identified")
            
            # Process gaps in batches for better performance
            gap_processing_tasks = []
            for i, gap in enumerate(initial_gaps):
                task = self._process_single_gap(
                    analysis.id,
                    gap,
                    i,
                    session
                )
                gap_processing_tasks.append(task)
            
            # Process all gaps concurrently using batch processor
            gap_results = await self.batch_processor.process(
                gap_processing_tasks,
                lambda task: task  # Identity function since tasks are already coroutines
            )
            
            # Filter valid gaps
            valid_gaps = [result for result in gap_results if result is not None]
            
            # Update analysis summary
            await self._update_analysis_summary(
                analysis,
                len(initial_gaps),
                len(valid_gaps),
                session
            )
            
            # Prepare response
            response = await self._prepare_response(
                analysis,
                valid_gaps,
                session
            )
            
            logger.info(f"Gap analysis completed: {len(valid_gaps)}/{len(initial_gaps)} valid gaps")
            return response
            
        except Exception as e:
            logger.error(f"Gap analysis failed: {e}")
            
            if analysis:
                await self._mark_analysis_failed(analysis, str(e), session)
            
            return GapAnalysisResponse(
                request_id=request.request_id,
                correlation_id=request.correlation_id,
                status="FAILED",
                message=f"Analysis failed: {str(e)}",
                error=str(e)
            )
        
        finally:
            # Cleanup
            await self.search_service.close()
            await self.grobid_client.close()
    
    async def _create_gap_analysis(
        self,
        request: GapAnalysisRequest,
        session: AsyncSession
    ) -> GapAnalysis:
        """Create initial gap analysis record."""
        analysis = GapAnalysis(
            paper_id=request.paper_id,
            paper_extraction_id=request.paper_extraction_id,
            correlation_id=request.correlation_id,
            request_id=request.request_id,
            status=GapStatus.PROCESSING,
            started_at=datetime.now(timezone.utc),
            config=request.config
        )
        
        session.add(analysis)
        await session.commit()
        await session.refresh(analysis)
        
        return analysis
    
    async def _fetch_paper_data(
        self,
        paper_id: str,
        extraction_id: str,
        session: AsyncSession
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Fetch paper and extraction data from database."""
        # Fetch paper
        paper_result = await session.execute(
            select(Paper).where(Paper.id == paper_id)
        )
        paper = paper_result.scalar_one_or_none()
        
        if not paper:
            return None, None
        
        paper_data = {
            'title': paper.title,
            'abstract_text': paper.abstract_text,
            'doi': paper.doi,
            'publication_date': paper.publication_date
        }
        
        # Fetch extraction data
        extraction_result = await session.execute(
            select(PaperExtraction).where(PaperExtraction.id == extraction_id)
        )
        extraction = extraction_result.scalar_one_or_none()
        
        if not extraction:
            return paper_data, {}
        
        # Fetch sections with paragraphs
        sections_result = await session.execute(
            select(ExtractedSection)
            .where(ExtractedSection.paper_extraction_id == extraction_id)
            .order_by(ExtractedSection.order_index)
        )
        sections = sections_result.scalars().all()
        
        # Fetch figures
        figures_result = await session.execute(
            select(ExtractedFigure)
            .where(ExtractedFigure.paper_extraction_id == extraction_id)
            .order_by(ExtractedFigure.order_index)
        )
        figures = figures_result.scalars().all()
        
        # Fetch tables
        tables_result = await session.execute(
            select(ExtractedTable)
            .where(ExtractedTable.paper_extraction_id == extraction_id)
            .order_by(ExtractedTable.order_index)
        )
        tables = tables_result.scalars().all()
        
        # Prepare extracted content
        extracted_content = {
            'sections': [],
            'figures': [],
            'tables': [],
            'conclusion': None
        }
        
        for section in sections:
            # Fetch paragraphs for this section
            paragraphs_result = await session.execute(
                select(ExtractedParagraph)
                .where(ExtractedParagraph.section_id == section.id)
                .order_by(ExtractedParagraph.order_index)
            )
            paragraphs = paragraphs_result.scalars().all()
            
            section_data = {
                'title': section.title,
                'type': section.section_type,
                'paragraphs': [{'text': p.text} for p in paragraphs]
            }
            extracted_content['sections'].append(section_data)
            
            # Check for conclusion
            if section.title and 'conclusion' in section.title.lower():
                extracted_content['conclusion'] = ' '.join([p.text for p in paragraphs if p.text])
        
        for figure in figures:
            extracted_content['figures'].append({
                'caption': figure.caption,
                'label': figure.label
            })
        
        for table in tables:
            extracted_content['tables'].append({
                'caption': table.caption,
                'label': table.label
            })
        
        return paper_data, extracted_content
    
    async def _create_gap_record(
        self,
        analysis_id: str,
        gap: InitialGap,
        index: int,
        session: AsyncSession
    ) -> ResearchGap:
        """Create a research gap record."""
        gap_record = ResearchGap(
            gap_analysis_id=analysis_id,
            gap_id=str(uuid4()),
            order_index=index,
            name=gap.name,
            description=gap.description,
            category=gap.category,
            initial_reasoning=gap.reasoning,
            initial_evidence=gap.evidence,
            validation_status=GapValidationStatus.INITIAL
        )
        
        session.add(gap_record)
        await session.commit()
        await session.refresh(gap_record)
        
        return gap_record
    
    async def _process_single_gap(
        self,
        analysis_id: str,
        gap: InitialGap,
        index: int,
        session: AsyncSession
    ) -> Optional[ResearchGap]:
        """Process a single gap (create, validate, and expand)."""
        try:
            logger.info(f"Processing gap {index+1}: {gap.name}")
            
            # Create gap record
            gap_record = await self._create_gap_record(
                analysis_id,
                gap,
                index,
                session
            )
            
            # Validate the gap
            is_valid = await self._validate_gap(
                gap,
                gap_record,
                session
            )
            
            if is_valid:
                # Expand gap details
                await self._expand_gap_details(
                    gap,
                    gap_record,
                    session
                )
                return gap_record
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error processing gap {gap.name}: {e}")
            return None
    
    async def _validate_gap(
        self,
        gap: InitialGap,
        gap_record: ResearchGap,
        session: AsyncSession
    ) -> bool:
        """Validate a research gap by searching for related work."""
        try:
            # Update status
            gap_record.validation_status = GapValidationStatus.VALIDATING
            await session.commit()
            
            # Generate search query
            search_query = await self.gemini_service.generate_search_query(gap)
            gap_record.validation_query = search_query
            
            # Search for related papers
            logger.info(f"Searching for papers: {search_query}")
            related_papers = await self.search_service.search_papers(
                search_query,
                max_results=10
            )
            
            if not related_papers:
                logger.warning("No related papers found, assuming gap is valid")
                gap_record.validation_status = GapValidationStatus.VALID
                gap_record.validation_confidence = 0.5
                gap_record.validated_at = datetime.now(timezone.utc)
                await session.commit()
                return True
            
            # Extract content from papers
            logger.info(f"Extracting content from {len(related_papers)} papers")
            extracted_contents = await self.grobid_client.extract_batch(related_papers)
            
            # Store validation papers
            for paper, content in zip(related_papers, extracted_contents):
                validation_paper = GapValidationPaper(
                    research_gap_id=gap_record.id,
                    title=paper.title,
                    doi=paper.doi,
                    url=paper.url,
                    extraction_status="SUCCESS" if content.extraction_success else "FAILED",
                    extracted_text=content.abstract if content.abstract else None
                )
                session.add(validation_paper)
            
            gap_record.papers_analyzed_count = len(related_papers)
            
            # Validate the gap using AI
            validation_result = await self.gemini_service.validate_gap(
                gap,
                extracted_contents
            )
            
            # Update gap record based on validation
            gap_record.validation_confidence = validation_result.confidence
            gap_record.validation_reasoning = validation_result.reasoning
            gap_record.validated_at = datetime.now(timezone.utc)
            
            if validation_result.is_valid:
                if validation_result.should_modify:
                    gap_record.validation_status = GapValidationStatus.MODIFIED
                    gap_record.modification_history = [{
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'original': gap.description,
                        'modification': validation_result.modification_suggestion
                    }]
                    if validation_result.modification_suggestion:
                        gap_record.description = validation_result.modification_suggestion
                else:
                    gap_record.validation_status = GapValidationStatus.VALID
                
                # Store supporting and conflicting papers
                gap_record.supporting_papers = validation_result.supporting_papers
                gap_record.conflicting_papers = validation_result.conflicting_papers
                
                await session.commit()
                return True
            else:
                gap_record.validation_status = GapValidationStatus.INVALID
                await session.commit()
                return False
                
        except Exception as e:
            logger.error(f"Error validating gap: {e}")
            gap_record.validation_status = GapValidationStatus.VALID
            gap_record.validation_confidence = 0.3
            gap_record.validation_reasoning = f"Validation error: {str(e)}"
            await session.commit()
            return True
    
    async def _expand_gap_details(
        self,
        gap: InitialGap,
        gap_record: ResearchGap,
        session: AsyncSession
    ):
        """Expand gap with detailed information."""
        try:
            # Get validation result for context
            validation_result = {
                'confidence': gap_record.validation_confidence,
                'reasoning': gap_record.validation_reasoning
            }
            
            # Generate expanded details
            expanded_details = await self.gemini_service.expand_gap_details(
                gap,
                validation_result
            )
            
            # Update gap record
            gap_record.potential_impact = expanded_details.get('potential_impact')
            gap_record.research_hints = expanded_details.get('research_hints')
            gap_record.implementation_suggestions = expanded_details.get('implementation_suggestions')
            gap_record.risks_and_challenges = expanded_details.get('risks_and_challenges')
            gap_record.required_resources = expanded_details.get('required_resources')
            gap_record.estimated_difficulty = expanded_details.get('estimated_difficulty')
            gap_record.estimated_timeline = expanded_details.get('estimated_timeline')
            
            # Store suggested topics
            suggested_topics = []
            for topic_data in expanded_details.get('suggested_topics', []):
                topic = GapTopic(
                    research_gap_id=gap_record.id,
                    title=topic_data.get('title'),
                    description=topic_data.get('description'),
                    research_questions=topic_data.get('research_questions', []),
                    methodology_suggestions=topic_data.get('methodology_suggestions'),
                    expected_outcomes=topic_data.get('expected_outcomes'),
                    relevance_score=topic_data.get('relevance_score', 0.5)
                )
                session.add(topic)
                suggested_topics.append(topic_data)
            
            gap_record.suggested_topics = suggested_topics
            
            # Prepare evidence anchors
            evidence_anchors = []
            if gap_record.supporting_papers:
                for paper in gap_record.supporting_papers:
                    evidence_anchors.append({
                        'title': paper.get('title'),
                        'url': paper.get('url', ''),
                        'type': 'supporting'
                    })
            
            if gap_record.conflicting_papers:
                for paper in gap_record.conflicting_papers:
                    evidence_anchors.append({
                        'title': paper.get('title'),
                        'url': paper.get('url', ''),
                        'type': 'conflicting'
                    })
            
            gap_record.evidence_anchors = evidence_anchors
            
            await session.commit()
            
        except Exception as e:
            logger.error(f"Error expanding gap details: {e}")
    
    async def _update_analysis_summary(
        self,
        analysis: GapAnalysis,
        total_gaps: int,
        valid_gaps: int,
        session: AsyncSession
    ):
        """Update analysis summary statistics."""
        analysis.total_gaps_identified = total_gaps
        analysis.valid_gaps_count = valid_gaps
        analysis.invalid_gaps_count = total_gaps - valid_gaps
        analysis.status = GapStatus.COMPLETED
        analysis.completed_at = datetime.now(timezone.utc)
        
        await session.commit()
    
    async def _prepare_response(
        self,
        analysis: GapAnalysis,
        valid_gaps: List[ResearchGap],
        session: AsyncSession
    ) -> GapAnalysisResponse:
        """Prepare the final response."""
        gap_details = []
        
        for gap in valid_gaps:
            # Fetch topics for this gap
            topics_result = await session.execute(
                select(GapTopic).where(GapTopic.research_gap_id == gap.id)
            )
            topics = topics_result.scalars().all()
            
            research_topics = [
                ResearchTopic(
                    title=topic.title,
                    description=topic.description,
                    research_questions=topic.research_questions or [],
                    methodology_suggestions=topic.methodology_suggestions,
                    expected_outcomes=topic.expected_outcomes,
                    relevance_score=topic.relevance_score or 0.5
                )
                for topic in topics
            ]
            
            gap_detail = GapDetail(
                gap_id=gap.gap_id,
                name=gap.name,
                description=gap.description,
                category=gap.category,
                validation_status=gap.validation_status,
                confidence_score=gap.validation_confidence or 0.5,
                potential_impact=gap.potential_impact,
                research_hints=gap.research_hints,
                implementation_suggestions=gap.implementation_suggestions,
                risks_and_challenges=gap.risks_and_challenges,
                required_resources=gap.required_resources,
                estimated_difficulty=gap.estimated_difficulty,
                estimated_timeline=gap.estimated_timeline,
                evidence_anchors=gap.evidence_anchors or [],
                supporting_papers_count=len(gap.supporting_papers) if gap.supporting_papers else 0,
                conflicting_papers_count=len(gap.conflicting_papers) if gap.conflicting_papers else 0,
                suggested_topics=research_topics
            )
            
            gap_details.append(gap_detail)
        
        return GapAnalysisResponse(
            request_id=analysis.request_id,
            correlation_id=analysis.correlation_id,
            status="SUCCESS",
            message=f"Successfully identified {len(valid_gaps)} valid research gaps",
            gap_analysis_id=str(analysis.id),
            total_gaps=analysis.total_gaps_identified,
            valid_gaps=analysis.valid_gaps_count,
            gaps=gap_details,
            completed_at=analysis.completed_at
        )
    
    async def _mark_analysis_failed(
        self,
        analysis: GapAnalysis,
        error: str,
        session: AsyncSession
    ):
        """Mark analysis as failed."""
        analysis.status = GapStatus.FAILED
        analysis.error_message = error
        analysis.completed_at = datetime.now(timezone.utc)
        await session.commit()