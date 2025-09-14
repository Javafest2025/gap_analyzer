"""
Gemini AI service for gap analysis.
"""

import google.generativeai as genai
from typing import List, Dict, Any, Optional
import json
import asyncio
from loguru import logger

from app.schemas.gap_schemas import (
    InitialGap, ValidationResult, ResearchTopic,
    ExtractedContent
)
from app.utils.helpers import RateLimiter, retry_async, parse_json_safely
from app.core.config import settings


class GeminiService:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(settings.GA_GEMINI_MODEL)
        
        # Initialize rate limiter for Gemini API
        self.rate_limiter = RateLimiter(
            max_calls=settings.GA_GEMINI_RATE_LIMIT,
            time_window=60  # 1 minute window
        )
        
        # Circuit breaker state
        self.circuit_breaker_failures = 0
        self.circuit_breaker_threshold = 3
        self.circuit_breaker_timeout = 300  # 5 minutes
        self.circuit_breaker_last_failure = None
        self.circuit_breaker_state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows the request."""
        import time
        
        if self.circuit_breaker_state == "CLOSED":
            return True
        elif self.circuit_breaker_state == "OPEN":
            if time.time() - self.circuit_breaker_last_failure > self.circuit_breaker_timeout:
                self.circuit_breaker_state = "HALF_OPEN"
                logger.info("Circuit breaker moved to HALF_OPEN state")
                return True
            return False
        elif self.circuit_breaker_state == "HALF_OPEN":
            return True
        return False
    
    def _record_success(self):
        """Record a successful API call."""
        if self.circuit_breaker_state == "HALF_OPEN":
            self.circuit_breaker_state = "CLOSED"
            self.circuit_breaker_failures = 0
            logger.info("Circuit breaker moved to CLOSED state after success")
    
    def _record_failure(self):
        """Record a failed API call."""
        import time
        
        self.circuit_breaker_failures += 1
        self.circuit_breaker_last_failure = time.time()
        
        if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
            self.circuit_breaker_state = "OPEN"
            logger.warning(f"Circuit breaker opened after {self.circuit_breaker_failures} failures")
    
    async def _exponential_backoff(self, attempt: int, base_delay: float = 1.0):
        """Implement exponential backoff with jitter."""
        import random
        
        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
        max_delay = 60  # Maximum 60 seconds
        delay = min(delay, max_delay)
        
        logger.info(f"Exponential backoff: waiting {delay:.2f} seconds before retry {attempt + 1}")
        await asyncio.sleep(delay)
        
    async def generate_initial_gaps(
        self, 
        paper_data: Dict[str, Any],
        extracted_content: Dict[str, Any]
    ) -> List[InitialGap]:
        """Generate initial research gaps from paper content."""
        # Check circuit breaker
        if not self._check_circuit_breaker():
            logger.warning("Circuit breaker is OPEN, skipping API call")
            return []
        
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Apply rate limiting
                await self.rate_limiter.wait_if_needed()
                
                # Prepare paper context
                context = self._prepare_paper_context(paper_data, extracted_content)
                
                prompt = f"""
            Analyze the following academic paper and identify research gaps:

            {context}

            Identify 3-7 significant research gaps in this paper. For each gap, provide:
            1. A concise name (max 100 characters)
            2. A detailed description of the gap
            3. Category (theoretical, methodological, empirical, application, or interdisciplinary)
            4. Reasoning why this is a gap
            5. Evidence from the paper supporting this gap

            Format your response as a JSON array with objects containing:
            {{
                "name": "gap name",
                "description": "detailed description",
                "category": "category",
                "reasoning": "why this is a gap",
                "evidence": "evidence from paper"
            }}

            Focus on:
            - Limitations explicitly mentioned by authors
            - Future work suggestions
            - Unexplored methodologies or approaches
            - Missing comparative analyses
            - Scalability or generalization issues
            - Theoretical gaps or assumptions
            - Interdisciplinary opportunities
            
            Respond ONLY with valid JSON array.
            """
                
                response = await asyncio.to_thread(
                    self.model.generate_content, prompt
                )
                
                # Parse response
                gaps_data = parse_json_safely(response.text, [])
                gaps = [InitialGap(**gap) for gap in gaps_data]
                
                # Record success
                self._record_success()
                logger.info(f"Generated {len(gaps)} initial gaps")
                return gaps
                
            except Exception as e:
                error_str = str(e)
                logger.error(f"Error generating initial gaps (attempt {attempt + 1}): {e}")
                
                # Record failure
                self._record_failure()
                
                # Check if it's a rate limit error
                if "429" in error_str or "quota" in error_str.lower():
                    logger.warning(f"Rate limit exceeded for gap generation: {e}")
                    if attempt < max_attempts - 1:
                        await self._exponential_backoff(attempt, base_delay=30.0)  # Longer delay for rate limits
                        continue
                    else:
                        return []
                else:
                    if attempt < max_attempts - 1:
                        await self._exponential_backoff(attempt)
                        continue
                    else:
                        return []
        
        return []
    
    async def generate_search_query(self, gap: InitialGap) -> str:
        """Generate a simple search query optimized for arXiv."""
        try:
            prompt = f"""
            Generate a simple search query for arXiv to find papers related to this research gap:

            Gap Name: {gap.name}
            Description: {gap.description}
            Category: {gap.category}

            Create a simple search query that:
            1. Uses only 2-4 key terms (no boolean operators)
            2. Focuses on the main topic/domain
            3. Uses common academic terminology
            4. Is suitable for arXiv's simple search

            Examples of good queries:
            - "machine learning protein structure"
            - "neural networks computer vision"
            - "quantum computing algorithms"
            - "natural language processing"

            Return ONLY the search terms separated by spaces, nothing else.
            """
            
            response = await asyncio.to_thread(
                self.model.generate_content, prompt
            )
            
            query = response.text.strip().strip('"')
            logger.info(f"Generated arXiv search query for gap: {query}")
            return query
            
        except Exception as e:
            logger.error(f"Error generating search query: {e}")
            # Fallback to basic query using gap name and category
            fallback_query = f"{gap.name} {gap.category}".lower()
            # Clean up the query for arXiv
            fallback_query = ' '.join(fallback_query.split()[:4])  # Limit to 4 words
            logger.info(f"Using fallback query: {fallback_query}")
            return fallback_query
    
    @retry_async(max_attempts=3, delay=2)
    async def validate_gap(
        self,
        gap: InitialGap,
        related_papers: List[ExtractedContent]
    ) -> ValidationResult:
        """Validate if a gap is still valid based on related papers."""
        try:
            # Apply rate limiting
            await self.rate_limiter.wait_if_needed()
            
            # Prepare context from related papers
            papers_context = self._prepare_validation_context(related_papers)
            
            prompt = f"""
            Validate if the following research gap is still valid based on recent papers:

            RESEARCH GAP:
            Name: {gap.name}
            Description: {gap.description}
            Category: {gap.category}
            Reasoning: {gap.reasoning}

            RELATED PAPERS ANALYZED:
            {papers_context}

            Analyze whether this gap:
            1. Has been fully addressed by any of these papers
            2. Has been partially addressed
            3. Remains completely unaddressed
            4. Should be modified based on new findings

            Provide your analysis as JSON:
            {{
                "is_valid": true/false,
                "confidence": 0.0-1.0,
                "reasoning": "detailed reasoning",
                "should_modify": true/false,
                "modification_suggestion": "suggestion if modification needed or null",
                "supporting_papers": [
                    {{"title": "paper title", "reason": "why it supports the gap"}}
                ],
                "conflicting_papers": [
                    {{"title": "paper title", "reason": "why it conflicts with the gap"}}
                ]
            }}

            Be critical and thorough. A gap is only invalid if it has been comprehensively addressed.
            Respond ONLY with valid JSON.
            """
            
            response = await asyncio.to_thread(
                self.model.generate_content, prompt
            )
            
            validation_data = parse_json_safely(response.text, {})
            return ValidationResult(**validation_data)
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                logger.warning(f"Rate limit exceeded for gap validation: {e}")
                # Wait longer for rate limit
                await asyncio.sleep(60)  # Wait 1 minute for rate limit reset
                # Return default validation
                return ValidationResult(
                    is_valid=True,
                    confidence=0.3,
                    reasoning="Rate limited - assuming valid with low confidence",
                    should_modify=False
                )
            else:
                logger.error(f"Error validating gap: {e}")
                # Return default validation (assume valid with low confidence)
                return ValidationResult(
                    is_valid=True,
                    confidence=0.3,
                    reasoning="Could not validate due to error",
                    should_modify=False
                )
    
    @retry_async(max_attempts=3, delay=2)
    async def expand_gap_details(
        self,
        gap: InitialGap,
        validation: ValidationResult
    ) -> Dict[str, Any]:
        """Expand gap with detailed information for users."""
        try:
            # Apply rate limiting
            await self.rate_limiter.wait_if_needed()
            
            prompt = f"""
            Provide comprehensive details about this validated research gap:

            GAP INFORMATION:
            Name: {gap.name}
            Description: {gap.description}
            Category: {gap.category}
            Validation Confidence: {validation.confidence}

            Generate detailed information in JSON format:
            {{
                "potential_impact": "Explain the potential scientific and practical impact",
                "research_hints": "Provide specific hints and directions for researchers",
                "implementation_suggestions": "Suggest concrete steps to address this gap",
                "risks_and_challenges": "Identify potential risks and challenges",
                "required_resources": "List required resources (expertise, equipment, data, etc.)",
                "estimated_difficulty": "low/medium/high with justification",
                "estimated_timeline": "Realistic timeline estimate with milestones",
                "suggested_topics": [
                    {{
                        "title": "Research topic title",
                        "description": "Topic description",
                        "research_questions": ["question1", "question2"],
                        "methodology_suggestions": "Suggested methodologies",
                        "expected_outcomes": "Expected outcomes",
                        "relevance_score": 0.0-1.0
                    }}
                ]
            }}

            Provide at least 3-5 suggested research topics.
            Be specific, practical, and actionable.
            Respond ONLY with valid JSON.
            """
            
            response = await asyncio.to_thread(
                self.model.generate_content, prompt
            )
            
            expanded_data = parse_json_safely(response.text, {})
            logger.info(f"Expanded gap details with {len(expanded_data.get('suggested_topics', []))} topics")
            return expanded_data
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                logger.warning(f"Rate limit exceeded for gap expansion: {e}")
                # Wait longer for rate limit
                await asyncio.sleep(60)  # Wait 1 minute for rate limit reset
            else:
                logger.error(f"Error expanding gap details: {e}")
            
            return {
                "potential_impact": "Unable to generate impact analysis due to rate limiting",
                "research_hints": "Unable to generate hints due to rate limiting",
                "implementation_suggestions": "Unable to generate suggestions due to rate limiting",
                "risks_and_challenges": "Unable to identify risks due to rate limiting",
                "required_resources": "Unable to identify resources due to rate limiting",
                "estimated_difficulty": "unknown",
                "estimated_timeline": "unknown",
                "suggested_topics": []
            }
    
    def _prepare_paper_context(
        self,
        paper_data: Dict[str, Any],
        extracted_content: Dict[str, Any]
    ) -> str:
        """Prepare paper context for AI analysis."""
        context_parts = []
        
        # Basic metadata
        context_parts.append(f"Title: {paper_data.get('title', 'N/A')}")
        context_parts.append(f"Abstract: {paper_data.get('abstract_text', 'N/A')}")
        
        # Extracted sections
        if extracted_content.get('sections'):
            context_parts.append("\nKEY SECTIONS:")
            for section in extracted_content['sections'][:10]:  # Limit to 10 sections
                if section.get('title'):
                    context_parts.append(f"\n{section['title']}:")
                    if section.get('paragraphs'):
                        # Combine first few paragraphs
                        text = ' '.join([p.get('text', '') for p in section['paragraphs'][:3]])
                        context_parts.append(text[:1000])  # Limit text length
        
        # Conclusion
        if extracted_content.get('conclusion'):
            context_parts.append(f"\nCONCLUSION:\n{extracted_content['conclusion'][:1000]}")
        
        # Figures and tables captions (often contain important info)
        if extracted_content.get('figures'):
            context_parts.append("\nFIGURE CAPTIONS:")
            for fig in extracted_content['figures'][:5]:
                if fig.get('caption'):
                    context_parts.append(f"- {fig['caption']}")
        
        if extracted_content.get('tables'):
            context_parts.append("\nTABLE CAPTIONS:")
            for table in extracted_content['tables'][:5]:
                if table.get('caption'):
                    context_parts.append(f"- {table['caption']}")
        
        return '\n'.join(context_parts)
    
    def _prepare_validation_context(self, papers: List[ExtractedContent]) -> str:
        """Prepare context from related papers for validation."""
        context_parts = []
        
        for i, paper in enumerate(papers[:10], 1):  # Analyze up to 10 papers
            context_parts.append(f"\nPAPER {i}:")
            context_parts.append(f"Title: {paper.title}")
            
            if paper.abstract:
                context_parts.append(f"Abstract: {paper.abstract[:500]}")
            
            if paper.methods:
                context_parts.append(f"Methods: {paper.methods[:500]}")
            
            if paper.results:
                context_parts.append(f"Results: {paper.results[:500]}")
            
            if paper.conclusion:
                context_parts.append(f"Conclusion: {paper.conclusion[:500]}")
        
        return '\n'.join(context_parts)