"""
Web search service for finding academic papers.
"""

import httpx
from typing import List, Dict, Any, Optional
import asyncio
from loguru import logger
from datetime import datetime
import json

from app.schemas.gap_schemas import PaperSearchResult
from app.utils.helpers import retry_async, parse_json_safely, clean_text, calculate_similarity, RateLimiter


class WebSearchService:
    """Service for searching academic papers on the web."""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)  # Standard timeout
        # Using only arXiv for now to avoid rate limiting issues
        self.search_apis = {
            'arxiv': 'https://export.arxiv.org/api/query'
        }
        # Rate limiter for arXiv only
        self.rate_limiters = {
            'arxiv': RateLimiter(max_calls=5, time_window=60)  # 5 calls per minute for arXiv
        }
    
    async def search_papers(
        self,
        query: str,
        max_results: int = 5
    ) -> List[PaperSearchResult]:
        """Search for papers using arXiv API with fallback strategies."""
        logger.info(f"Starting paper search with query: '{query}' and max_results: {max_results}")
        all_results = []
        
        # Try the original query first
        logger.info("Trying original query on arXiv")
        results = await self._search_arxiv(query, max_results)
        all_results.extend(results)
        
        # If no results, try fallback strategies
        if len(all_results) == 0:
            logger.info("No results with original query, trying fallback strategies")
            
            # Strategy 1: Use only the first 2 words of the query
            words = query.split()
            if len(words) > 2:
                fallback_query = ' '.join(words[:2])
                logger.info(f"Trying fallback query with first 2 words: '{fallback_query}'")
                fallback_results = await self._search_arxiv(fallback_query, max_results)
                all_results.extend(fallback_results)
            
            # Strategy 2: Use only the first word if still no results
            if len(all_results) == 0 and len(words) > 1:
                single_word_query = words[0]
                logger.info(f"Trying single word query: '{single_word_query}'")
                single_word_results = await self._search_arxiv(single_word_query, max_results)
                all_results.extend(single_word_results)
        
        # Remove duplicates based on title similarity
        logger.info(f"Removing duplicates from {len(all_results)} total results")
        unique_results = self._remove_duplicates(all_results)
        logger.info(f"Found {len(unique_results)} unique papers for query: {query}")
        
        # Return top results
        return unique_results[:max_results]
    
    def _remove_duplicates(self, results: List[PaperSearchResult]) -> List[PaperSearchResult]:
        """Remove duplicate papers based on title similarity."""
        if not results:
            return []
        
        unique_results = [results[0]]  # Start with first result
        
        for result in results[1:]:
            is_duplicate = False
            for unique_result in unique_results:
                # Check if titles are similar (threshold: 0.8)
                if calculate_similarity(result.title, unique_result.title) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_results.append(result)
        
        return unique_results
    
    @retry_async(max_attempts=3, delay=1)
    async def _search_arxiv(
        self,
        query: str,
        limit: int
    ) -> List[PaperSearchResult]:
        """Search using arXiv API."""
        try:
            # Apply rate limiting
            await self.rate_limiters['arxiv'].wait_if_needed()
            
            logger.info(f"Starting arXiv search with query: '{query}' and limit: {limit}")
            # Clean the query for arXiv
            clean_query = query.strip().lower()
            logger.info(f"Cleaned query for arXiv: '{clean_query}'")
            
            params = {
                'search_query': f'all:{clean_query}',
                'start': 0,
                'max_results': limit,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            logger.info(f"Making HTTP request to: {self.search_apis['arxiv']}")
            response = await self.client.get(
                self.search_apis['arxiv'],
                params=params
            )
            logger.info(f"arXiv API response status: {response.status_code}")
            
            if response.status_code == 200:
                # Parse XML response
                import xml.etree.ElementTree as ET
                try:
                    root = ET.fromstring(response.text)
                    logger.info("Successfully parsed arXiv XML response")
                    
                    results = []
                    entries = root.findall('{http://www.w3.org/2005/Atom}entry')
                    logger.info(f"Found {len(entries)} entries in arXiv response")
                    
                    for i, entry in enumerate(entries):
                        try:
                            title_elem = entry.find('{http://www.w3.org/2005/Atom}title')
                            summary_elem = entry.find('{http://www.w3.org/2005/Atom}summary')
                            link_elem = entry.find('{http://www.w3.org/2005/Atom}link[@type="application/pdf"]')
                            published_elem = entry.find('{http://www.w3.org/2005/Atom}published')
                            
                            # Extract authors
                            authors = []
                            for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
                                name_elem = author.find('{http://www.w3.org/2005/Atom}name')
                                if name_elem is not None and name_elem.text:
                                    authors.append(name_elem.text.strip())
                            
                            title = title_elem.text.strip() if title_elem is not None and title_elem.text else f"Paper {i+1}"
                            abstract = summary_elem.text.strip() if summary_elem is not None and summary_elem.text else ""
                            
                            result = PaperSearchResult(
                                title=title,
                                abstract=abstract,
                                url=link_elem.get('href') if link_elem is not None else None,
                                pdf_url=link_elem.get('href') if link_elem is not None else None,
                                publication_date=published_elem.text[:10] if published_elem is not None and published_elem.text else None,
                                authors=authors,
                                venue='arXiv'
                            )
                            results.append(result)
                            logger.debug(f"Added paper {i+1}: {title[:50]}...")
                            
                        except Exception as entry_error:
                            logger.warning(f"Error processing entry {i+1}: {entry_error}")
                            continue
                    
                    logger.info(f"arXiv search completed successfully with {len(results)} results")
                    return results
                    
                except ET.ParseError as parse_error:
                    logger.error(f"Failed to parse arXiv XML response: {parse_error}")
                    logger.error(f"Response content (first 500 chars): {response.text[:500]}")
                    return []
            elif response.status_code == 301:
                logger.warning("arXiv API returned 301 redirect - this should be fixed with HTTPS URL")
                return []
            else:
                logger.warning(f"arXiv search failed with status: {response.status_code}")
                return []
                
        except Exception as e:
            import traceback
            logger.error(f"arXiv search error: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return []
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()