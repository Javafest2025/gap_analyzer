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
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
        
        # Initialize rate limiter for Gemini API
        self.rate_limiter = RateLimiter(
            max_calls=settings.GEMINI_RATE_LIMIT,
            time_window=60  # 1 minute window
        )
        
    @retry_async(max_attempts=3, delay=2)
    async def generate_initial_gaps(
        self, 
        paper_data: Dict[str, Any],
        extracted_content: Dict[str, Any]
    ) -> List[InitialGap]:
        """Generate initial research gaps from paper content."""
        try:
            # Apply rate limiting
            await self.rate_limiter.wait_if_needed()
            
            # Prepare paper context
            context = self._prepare_paper_context(paper_data, extracted_content)
            
            prompt = f"""
            Analyze the following academic paper and identify research gaps:

            {context}

            Identify 5-10 significant research gaps in this paper. For each gap, provide:
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
            
            logger.info(f"Generated {len(gaps)} initial gaps")
            return gaps
            
        except Exception as e:
            logger.error(f"Error generating initial gaps: {e}")
            return []
    
    async def generate_search_query(self, gap: InitialGap) -> str:
        """Generate an advanced search query for validating a gap."""
        try:
            prompt = f"""
            Generate an advanced academic search query to find papers that might have addressed this research gap:

            Gap Name: {gap.name}
            Description: {gap.description}
            Category: {gap.category}

            Create a search query that will find:
            1. Papers that directly address this gap
            2. Related work in this area
            3. Similar methodologies or approaches
            4. Recent advances that might fill this gap

            The query should be:
            - Specific enough to find relevant papers
            - Include key technical terms
            - Use boolean operators if helpful
            - Be optimized for academic search engines

            Return ONLY the search query string, nothing else.
            """
            
            response = await asyncio.to_thread(
                self.model.generate_content, prompt
            )
            
            query = response.text.strip().strip('"')
            logger.info(f"Generated search query for gap: {query}")
            return query
            
        except Exception as e:
            logger.error(f"Error generating search query: {e}")
            # Fallback to basic query
            return f"{gap.name} {gap.category}"
    
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
            logger.error(f"Error expanding gap details: {e}")
            return {
                "potential_impact": "Unable to generate impact analysis",
                "research_hints": "Unable to generate hints",
                "implementation_suggestions": "Unable to generate suggestions",
                "risks_and_challenges": "Unable to identify risks",
                "required_resources": "Unable to identify resources",
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