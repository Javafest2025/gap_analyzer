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
from app.utils.helpers import retry_async, parse_json_safely, clean_text, calculate_similarity


class WebSearchService:
    """Service for searching academic papers on the web."""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        # Using multiple search APIs for better coverage
        self.search_apis = {
            'semantic_scholar': 'https://api.semanticscholar.org/graph/v1/paper/search',
            'crossref': 'https://api.crossref.org/works',
            'arxiv': 'http://export.arxiv.org/api/query'
        }
    
    async def search_papers(
        self,
        query: str,
        max_results: int = 10
    ) -> List[PaperSearchResult]:
        """Search for papers using multiple academic search APIs."""
        all_results = []
        
        # Search using multiple APIs in parallel
        tasks = [
            self._search_semantic_scholar(query, max_results),
            self._search_crossref(query, max_results),
            self._search_arxiv(query, max_results)
        ]
        
        results_lists = await asyncio.gather(*tasks, return_exceptions=True)
        
        for results in results_lists:
            if isinstance(results, list):
                all_results.extend(results)
            elif isinstance(results, Exception):
                logger.error(f"Search error: {results}")
        
        # Remove duplicates based on title similarity
        unique_results = self._deduplicate_results(all_results)
        
        logger.info(f"Found {len(unique_results)} unique papers for query: {query}")
        return unique_results[:max_results]
    
    @retry_async(max_attempts=3, delay=1)
    async def _search_semantic_scholar(
        self,
        query: str,
        limit: int
    ) -> List[PaperSearchResult]:
        """Search using Semantic Scholar API."""
        try:
            params = {
                'query': query,
                'limit': limit,
                'fields': 'paperId,title,abstract,url,venue,year,authors,isOpenAccess,openAccessPdf'
            }
            
            response = await self.client.get(
                self.search_apis['semantic_scholar'],
                params=params
            )
            
            if response.status_code == 200:
                data = parse_json_safely(response.text, {})
                results = []
                
                for paper in data.get('data', []):
                    pdf_url = None
                    if paper.get('openAccessPdf'):
                        pdf_url = paper['openAccessPdf'].get('url')
                    
                    result = PaperSearchResult(
                        title=paper.get('title', ''),
                        abstract=paper.get('abstract'),
                        url=paper.get('url'),
                        pdf_url=pdf_url,
                        publication_date=str(paper.get('year')) if paper.get('year') else None,
                        authors=[a.get('name', '') for a in paper.get('authors', [])],
                        venue=paper.get('venue')
                    )
                    results.append(result)
                
                return results
            else:
                logger.warning(f"Semantic Scholar search failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Semantic Scholar search error: {e}")
            return []
    
    @retry_async(max_attempts=3, delay=1)
    async def _search_crossref(
        self,
        query: str,
        limit: int
    ) -> List[PaperSearchResult]:
        """Search using CrossRef API."""
        try:
            params = {
                'query': query,
                'rows': limit,
                'select': 'DOI,title,abstract,URL,published-print,author,container-title'
            }
            
            response = await self.client.get(
                self.search_apis['crossref'],
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for item in data.get('message', {}).get('items', []):
                    # Extract publication date
                    pub_date = None
                    if item.get('published-print'):
                        date_parts = item['published-print'].get('date-parts', [[]])[0]
                        if date_parts:
                            pub_date = '-'.join(map(str, date_parts))
                    
                    # Extract authors
                    authors = []
                    for author in item.get('author', []):
                        name = f"{author.get('given', '')} {author.get('family', '')}".strip()
                        if name:
                            authors.append(name)
                    
                    result = PaperSearchResult(
                        title=' '.join(item.get('title', [])),
                        abstract=item.get('abstract'),
                        doi=item.get('DOI'),
                        url=item.get('URL'),
                        publication_date=pub_date,
                        authors=authors,
                        venue=item.get('container-title', [None])[0] if item.get('container-title') else None
                    )
                    results.append(result)
                
                return results
            else:
                logger.warning(f"CrossRef search failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"CrossRef search error: {e}")
            return []
    
    @retry_async(max_attempts=3, delay=1)
    async def _search_arxiv(
        self,
        query: str,
        limit: int
    ) -> List[PaperSearchResult]:
        """Search using arXiv API."""
        try:
            params = {
                'search_query': f'all:{query}',
                'max_results': limit,
                'sortBy': 'relevance'
            }
            
            response = await self.client.get(
                self.search_apis['arxiv'],
                params=params
            )
            
            if response.status_code == 200:
                # Parse XML response (simplified parsing)
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response.text)
                
                # Define namespaces
                ns = {
                    'atom': 'http://www.w3.org/2005/Atom',
                    'arxiv': 'http://arxiv.org/schemas/atom'
                }
                
                results = []
                for entry in root.findall('atom:entry', ns):
                    # Extract title
                    title_elem = entry.find('atom:title', ns)
                    title = title_elem.text.strip() if title_elem is not None else ''
                    
                    # Extract abstract
                    summary_elem = entry.find('atom:summary', ns)
                    abstract = summary_elem.text.strip() if summary_elem is not None else None
                    
                    # Extract URL
                    id_elem = entry.find('atom:id', ns)
                    url = id_elem.text if id_elem is not None else None
                    
                    # Extract PDF URL
                    pdf_url = None
                    for link in entry.findall('atom:link', ns):
                        if link.get('type') == 'application/pdf':
                            pdf_url = link.get('href')
                            break
                    
                    # Extract authors
                    authors = []
                    for author in entry.findall('atom:author', ns):
                        name_elem = author.find('atom:name', ns)
                        if name_elem is not None:
                            authors.append(name_elem.text)
                    
                    # Extract publication date
                    published_elem = entry.find('atom:published', ns)
                    pub_date = None
                    if published_elem is not None:
                        pub_date = published_elem.text.split('T')[0]
                    
                    result = PaperSearchResult(
                        title=title,
                        abstract=abstract,
                        url=url,
                        pdf_url=pdf_url,
                        publication_date=pub_date,
                        authors=authors,
                        venue='arXiv'
                    )
                    results.append(result)
                
                return results
            else:
                logger.warning(f"arXiv search failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"arXiv search error: {e}")
            return []
    
    def _deduplicate_results(
        self,
        results: List[PaperSearchResult]
    ) -> List[PaperSearchResult]:
        """Remove duplicate papers based on title similarity."""
        unique_results = []
        
        for result in results:
            # Clean and normalize title for comparison
            normalized_title = clean_text(result.title).lower()
            
            # Check similarity with existing results
            is_duplicate = False
            for existing_result in unique_results:
                existing_title = clean_text(existing_result.title).lower()
                similarity = calculate_similarity(normalized_title, existing_title)
                
                # If similarity is high (>0.8), consider it a duplicate
                if similarity > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_results.append(result)
        
        return unique_results
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()