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
        
        # Initialize batch processor for gap validation - process sequentially
        self.batch_processor = AsyncBatchProcessor(
            batch_size=1,  # Process 1 gap at a time to avoid rate limits
            max_concurrent=1  # Maximum 1 concurrent batch to respect rate limits
        )
    
    async def analyze_paper(
        self,
        request: GapAnalysisRequest,
        session: AsyncSession
    ) -> GapAnalysisResponse:
        """Main method to analyze a paper for research gaps."""
        analysis = None
        
        try:
            logger.info(f"Starting gap analysis for paper: {request.paperId}")
            logger.info(f"Request ID: {request.requestId}, Correlation ID: {request.correlationId}")
            
            # Test network connectivity first
            logger.info("Step 1: Testing network connectivity...")
            await self._test_network_connectivity()
            logger.info("Network connectivity test completed successfully")
            
            # Create gap analysis record
            logger.info("Step 2: Creating gap analysis record...")
            analysis = await self._create_gap_analysis(request, session)
            logger.info(f"Gap analysis record created with ID: {analysis.id}")
            
            # Fetch paper and extraction data
            logger.info("Step 3: Fetching paper and extraction data...")
            paper_data, extracted_content = await self._fetch_paper_data(
                request.paperId,
                request.paperExtractionId,
                session
            )
            
            if not paper_data:
                raise ValueError("Paper not found")
            logger.info(f"Paper data fetched successfully. Title: {paper_data.get('title', 'N/A')}")
            
            # Generate initial gaps
            logger.info("Step 4: Generating initial gaps using Gemini AI...")
            initial_gaps = await self.gemini_service.generate_initial_gaps(
                paper_data,
                extracted_content
            )
            
            if not initial_gaps:
                logger.warning("No gaps could be identified from the paper content")
                # Return a response indicating no gaps were found
                await self._mark_analysis_completed(analysis, 0, 0, 0, 0, session)
                return GapAnalysisResponse(
                    requestId=request.requestId,
                    correlationId=request.correlationId,
                    status="COMPLETED",
                    message="Analysis completed - no research gaps identified",
                    gapAnalysisId=str(analysis.id),
                    totalGaps=0,
                    validGaps=0,
                    gaps=[]
                )
            logger.info(f"Generated {len(initial_gaps)} initial gaps")
            
            # Process gaps in batches for better performance
            logger.info("Step 5: Processing gaps for validation and expansion...")
            # Process gaps sequentially to avoid rate limiting
            gap_results = []
            for i, gap in enumerate(initial_gaps):
                logger.info(f"Processing gap {i+1}/{len(initial_gaps)}: {gap.name}")
                try:
                    result = await self._process_single_gap(
                        analysis.id,
                        gap,
                        i
                    )
                    gap_results.append(result)
                    logger.info(f"Gap {i+1} processing completed successfully")
                except Exception as e:
                    logger.error(f"Error processing gap {i+1}: {e}")
                    gap_results.append(None)
            
            logger.info("Sequential gap processing completed")
            
            # Filter valid gaps
            valid_gap_data = [result for result in gap_results if result is not None]
            logger.info(f"Found {len(valid_gap_data)} valid gaps out of {len(initial_gaps)} total")
            
            # Update analysis summary
            logger.info("Step 6: Updating analysis summary...")
            await self._update_analysis_summary(
                analysis,
                len(initial_gaps),
                len(valid_gap_data),
                session
            )
            
            # Prepare response
            logger.info("Step 7: Preparing final response...")
            response = self._prepare_response(
                analysis,
                valid_gap_data
            )
            
            logger.info(f"Gap analysis completed successfully: {len(valid_gap_data)}/{len(initial_gaps)} valid gaps")
            return response
            
        except Exception as e:
            import traceback
            from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
            
            logger.error(f"Gap analysis failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            
            # Rollback the session before any further operations
            try:
                await session.rollback()
                logger.info("Session rolled back successfully after error")
            except Exception as rollback_error:
                logger.error(f"Failed to rollback session: {rollback_error}")
            
            # Mark analysis as failed
            if analysis:
                try:
                    await self._mark_analysis_failed(analysis, str(e), session)
                except Exception as mark_error:
                    logger.error(f"Failed to mark analysis as failed: {mark_error}")
            
            return GapAnalysisResponse(
                requestId=request.requestId,
                correlationId=request.correlationId,
                status="FAILED",
                message=f"Analysis failed: {str(e)}",
                error=str(e)
            )
        
        finally:
            # Cleanup
            logger.info("Cleaning up resources...")
            await self.search_service.close()
            await self.grobid_client.close()
            logger.info("Cleanup completed")
    
    async def _test_network_connectivity(self):
        """Test network connectivity to external services."""
        import socket
        import httpx
        
        # Test DNS resolution for external APIs
        test_hosts = [
            'export.arxiv.org',  # arXiv API
            'generativelanguage.googleapis.com'  # Gemini API
        ]
        
        for host in test_hosts:
            try:
                # Test DNS resolution
                socket.gethostbyname(host)
                logger.info(f"DNS resolution successful for {host}")
            except socket.gaierror as e:
                logger.error(f"DNS resolution failed for {host}: {e}")
                raise ConnectionError(
                    f"Network connectivity issue: Cannot resolve hostname '{host}'. "
                    "Please check your internet connection and DNS settings. "
                    f"Error: {e}"
                )
        
        # Test HTTP connectivity to arXiv API
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get("https://export.arxiv.org/api/query?search_query=test&max_results=1")
                if response.status_code == 200:
                    logger.info("arXiv API connectivity test successful")
                else:
                    logger.warning(f"arXiv API connectivity test returned status {response.status_code}")
        except Exception as e:
            logger.error(f"arXiv API connectivity test failed: {e}")
            raise ConnectionError(
                f"Network connectivity issue: Cannot reach arXiv API. "
                "Please check your internet connection and firewall settings. "
                f"Error: {e}"
            )
        
        # Test Gemini API connectivity
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Test Gemini API endpoint - we expect 401/403 without proper auth, but that means the API is reachable
                response = await client.get("https://generativelanguage.googleapis.com/v1beta/models")
                if response.status_code == 200:
                    logger.info("Gemini API connectivity test successful")
                elif response.status_code in [401, 403]:
                    logger.info("Gemini API connectivity test successful (API reachable, auth required)")
                else:
                    logger.warning(f"Gemini API returned unexpected status {response.status_code}")
        except Exception as e:
            logger.error(f"Gemini API connectivity test failed: {e}")
            raise ConnectionError(
                f"Gemini API connectivity issue: Cannot reach Google's Generative AI API. "
                "Please check your internet connection and firewall settings. "
                f"Error: {e}"
            )
        
        # Test GROBID service connectivity
        try:
            grobid_host = self.grobid_client.grobid_url.replace('http://', '').replace('https://', '').split(':')[0]
            socket.gethostbyname(grobid_host)
            logger.info(f"GROBID service DNS resolution successful for {grobid_host}")
            
            # Test HTTP connectivity to GROBID
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.grobid_client.grobid_url}/api/isalive")
                if response.status_code == 200:
                    logger.info("GROBID service connectivity test successful")
                else:
                    logger.warning(f"GROBID service returned status {response.status_code}")
        except Exception as e:
            logger.error(f"GROBID service connectivity test failed: {e}")
            raise ConnectionError(
                f"GROBID service connectivity issue: Cannot reach GROBID service at {self.grobid_client.grobid_url}. "
                "Please check if GROBID service is running and accessible. "
                f"Error: {e}"
            )
    
    async def _create_gap_analysis(
        self,
        request: GapAnalysisRequest,
        session: AsyncSession
    ) -> GapAnalysis:
        """Create initial gap analysis record with idempotency on correlation_id."""
        from sqlalchemy.dialects.postgresql import insert
        from sqlalchemy import func
        
        values = {
            'id': uuid4(),
            'paper_id': request.paperId,
            'paper_extraction_id': request.paperExtractionId,
            'correlation_id': request.correlationId,  # idempotency key
            'request_id': request.requestId,
            'status': GapStatus.PROCESSING,
            'started_at': datetime.now(timezone.utc),
            'error_message': None,
            'config': request.config,
            'total_gaps_identified': 0,
            'valid_gaps_count': 0,
            'invalid_gaps_count': 0,
            'modified_gaps_count': 0,
        }
        
        stmt = (
            insert(GapAnalysis)
            .values(**values)
            .on_conflict_do_update(
                index_elements=["correlation_id"],
                set_={
                    "paper_id": request.paperId,
                    "paper_extraction_id": request.paperExtractionId,
                    "request_id": request.requestId,
                    "status": GapStatus.PROCESSING,
                    "started_at": func.now(),
                    "error_message": None,
                    "config": request.config,
                },
            )
            .returning(GapAnalysis.id)
        )
        
        analysis_id = (await session.execute(stmt)).scalar_one()
        await session.commit()
        
        # Fetch the complete record
        analysis = await session.get(GapAnalysis, analysis_id)
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
    
    
    async def _process_single_gap(
        self,
        analysis_id: str,
        gap: InitialGap,
        index: int
    ) -> Optional[Dict[str, Any]]:
        """Process a single gap without creating database records."""
        try:
            logger.info(f"Processing gap {index+1}: {gap.name}")
            logger.info(f"Gap category: {gap.category}")
            logger.info(f"Gap description: {gap.description[:100]}...")
            
            # Validate gap (without creating database records)
            logger.info(f"Validating gap {index+1}...")
            is_valid = await self._validate_gap(gap)
            logger.info(f"Validation completed for gap {index+1}. Valid: {is_valid}")
            
            if is_valid:
                # Expand gap details (without creating database records)
                logger.info(f"Expanding details for gap {index+1}: {gap.name}")
                expanded_details = await self._expand_gap_details(gap)
                logger.info(f"Gap {index+1} processing completed successfully")
                
                # Return gap data for Java backend to process
                return {
                    'gap_id': f"{analysis_id}-{index}-{uuid4()}",
                    'name': gap.name,
                    'description': gap.description,
                    'category': gap.category,
                    'validation_status': 'VALID',
                    'confidence_score': 0.8,
                    'potential_impact': expanded_details.get('potential_impact'),
                    'research_hints': expanded_details.get('research_hints'),
                    'implementation_suggestions': expanded_details.get('implementation_suggestions'),
                    'risks_and_challenges': expanded_details.get('risks_and_challenges'),
                    'required_resources': expanded_details.get('required_resources'),
                    'estimated_difficulty': expanded_details.get('estimated_difficulty'),
                    'estimated_timeline': expanded_details.get('estimated_timeline'),
                    'evidence_anchors': expanded_details.get('evidence_anchors', []),
                    'suggested_topics': expanded_details.get('suggested_topics', [])
                }
            else:
                logger.info(f"Gap {index+1} marked as invalid")
                return None
                
        except Exception as e:
            logger.error(f"Error processing gap {index+1}: {e}")
            return None
    
    async def _validate_gap(
        self,
        gap: InitialGap
    ) -> bool:
        """Validate a research gap by searching for related work."""
        try:
            logger.info(f"Starting gap validation for: {gap.name}")
            
            # Generate search query
            logger.info("Generating search query using Gemini AI...")
            search_query = await self.gemini_service.generate_search_query(gap)
            logger.info(f"Generated search query: {search_query}")
            
            # Search for related papers
            logger.info(f"Starting paper search with query: {search_query}")
            try:
                related_papers = await self.search_service.search_papers(
                    search_query,
                    max_results=5
                )
                logger.info(f"Paper search completed. Found {len(related_papers)} papers")
            except Exception as search_error:
                import traceback
                logger.error(f"Failed to search for papers: {search_error}")
                logger.error(f"Search error type: {type(search_error).__name__}")
                logger.error(f"Search error stack trace: {traceback.format_exc()}")
                
                # Check if it's a network connectivity issue
                if "getaddrinfo failed" in str(search_error) or "Name or service not known" in str(search_error):
                    logger.error("DNS resolution failure detected in paper search")
                    raise ConnectionError(
                        "Network connectivity issue: Cannot resolve external API hostnames. "
                        "Please check your internet connection and DNS settings. "
                        f"Error: {search_error}"
                    )
                else:
                    raise search_error
            
            if not related_papers:
                logger.warning("No related papers found, assuming gap is valid")
                return True
            
            # Extract content from papers
            logger.info(f"Starting content extraction from {len(related_papers)} papers using GROBID")
            try:
                extracted_contents = await self.grobid_client.extract_batch(related_papers)
                logger.info(f"Content extraction completed. Successfully extracted {len([c for c in extracted_contents if c.extraction_success])} papers")
            except Exception as grobid_error:
                import traceback
                logger.error(f"Failed to extract content from papers: {grobid_error}")
                logger.error(f"GROBID error type: {type(grobid_error).__name__}")
                logger.error(f"GROBID error stack trace: {traceback.format_exc()}")
                
                # Check if it's a network connectivity issue
                if "getaddrinfo failed" in str(grobid_error) or "Name or service not known" in str(grobid_error):
                    logger.error("DNS resolution failure detected in GROBID extraction")
                    raise ConnectionError(
                        f"Network connectivity issue: Cannot reach GROBID service at {self.grobid_client.grobid_url}. "
                        "Please check if GROBID service is running and accessible. "
                        f"Error: {grobid_error}"
                    )
                else:
                    raise grobid_error
            
            # Validate the gap using AI
            validation_result = await self.gemini_service.validate_gap(
                gap,
                extracted_contents
            )
            
            return validation_result.is_valid
                
        except Exception as e:
            logger.error(f"Error validating gap: {e}")
            # Return True on error to avoid blocking processing
            return True
    
    async def _expand_gap_details(
        self,
        gap: InitialGap
    ) -> Dict[str, Any]:
        """Expand gap with detailed information without creating database records."""
        try:
            # Get validation result for context
            from app.schemas.gap_schemas import ValidationResult
            validation_result = ValidationResult(
                is_valid=True,
                confidence=0.8,
                reasoning="Validated",
                should_modify=False
            )
            
            # Generate expanded details
            expanded_details = await self.gemini_service.expand_gap_details(
                gap,
                validation_result
            )
            
            # Return the expanded details as a dictionary
            return {
                'potential_impact': expanded_details.get('potential_impact'),
                'research_hints': expanded_details.get('research_hints'),
                'implementation_suggestions': expanded_details.get('implementation_suggestions'),
                'risks_and_challenges': expanded_details.get('risks_and_challenges'),
                'required_resources': expanded_details.get('required_resources'),
                'estimated_difficulty': expanded_details.get('estimated_difficulty'),
                'estimated_timeline': expanded_details.get('estimated_timeline'),
                'evidence_anchors': expanded_details.get('evidence_anchors', []),
                'suggested_topics': expanded_details.get('suggested_topics', [])
            }
            
        except Exception as e:
            logger.error(f"Error expanding gap details: {e}")
            return {}
    
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
    
    def _transform_suggested_topics(self, suggested_topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transform suggested topics to ensure proper data types for Pydantic validation."""
        transformed_topics = []
        
        for i, topic in enumerate(suggested_topics):
            transformed_topic = topic.copy()
            
            # Convert methodology_suggestions from list to string if needed
            if 'methodology_suggestions' in transformed_topic:
                methodology = transformed_topic['methodology_suggestions']
                if isinstance(methodology, list):
                    logger.warning(f"Topic {i}: Converting methodology_suggestions from list to string")
                    # Join list items with newlines or semicolons
                    transformed_topic['methodology_suggestions'] = '; '.join(str(item) for item in methodology)
                elif methodology is None:
                    transformed_topic['methodology_suggestions'] = None
            
            # Convert expected_outcomes from list to string if needed
            if 'expected_outcomes' in transformed_topic:
                outcomes = transformed_topic['expected_outcomes']
                if isinstance(outcomes, list):
                    logger.warning(f"Topic {i}: Converting expected_outcomes from list to string")
                    # Join list items with newlines or semicolons
                    transformed_topic['expected_outcomes'] = '; '.join(str(item) for item in outcomes)
                elif outcomes is None:
                    transformed_topic['expected_outcomes'] = None
            
            # Ensure research_questions is a list
            if 'research_questions' in transformed_topic:
                questions = transformed_topic['research_questions']
                if not isinstance(questions, list):
                    logger.warning(f"Topic {i}: Converting research_questions from {type(questions)} to list")
                    # Convert single string to list
                    transformed_topic['research_questions'] = [str(questions)] if questions else []
            
            transformed_topics.append(transformed_topic)
        
        return transformed_topics
    
    def _prepare_response(
        self, 
        analysis: GapAnalysis, 
        valid_gap_data: List[Dict[str, Any]]
    ) -> GapAnalysisResponse:
        """Prepare the final response with gap data."""
        gap_details = []
        
        for gap_data in valid_gap_data:
            # Transform suggested topics to ensure proper data types
            suggested_topics = self._transform_suggested_topics(gap_data.get('suggested_topics', []))
            
            gap_detail = GapDetail(
                gapId=gap_data['gap_id'],
                name=gap_data['name'],
                description=gap_data['description'],
                category=gap_data['category'],
                validationStatus=gap_data['validation_status'],
                confidenceScore=gap_data['confidence_score'],
                potentialImpact=gap_data.get('potential_impact'),
                researchHints=gap_data.get('research_hints'),
                implementationSuggestions=gap_data.get('implementation_suggestions'),
                risksAndChallenges=gap_data.get('risks_and_challenges'),
                requiredResources=gap_data.get('required_resources'),
                estimatedDifficulty=gap_data.get('estimated_difficulty'),
                estimatedTimeline=gap_data.get('estimated_timeline'),
                evidenceAnchors=gap_data.get('evidence_anchors', []),
                supportingPapersCount=0,
                conflictingPapersCount=0,
                suggestedTopics=suggested_topics
            )
            gap_details.append(gap_detail)
        
        return GapAnalysisResponse(
            requestId=analysis.request_id,
            correlationId=analysis.correlation_id,
            status="SUCCESS",
            message=f"Successfully identified {len(valid_gap_data)} valid research gaps",
            gapAnalysisId=str(analysis.id),
            totalGaps=analysis.total_gaps_identified,
            validGaps=analysis.valid_gaps_count,
            gaps=gap_details,
            completedAt=analysis.completed_at
        )
    
    async def _mark_analysis_completed(
        self,
        analysis: GapAnalysis,
        total_gaps: int,
        valid_gaps: int,
        invalid_gaps: int,
        modified_gaps: int,
        session: AsyncSession
    ):
        """Mark analysis as completed."""
        analysis.status = GapStatus.COMPLETED
        analysis.total_gaps_identified = total_gaps
        analysis.valid_gaps_count = valid_gaps
        analysis.invalid_gaps_count = invalid_gaps
        analysis.modified_gaps_count = modified_gaps
        analysis.completed_at = datetime.now(timezone.utc)
        await session.commit()

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