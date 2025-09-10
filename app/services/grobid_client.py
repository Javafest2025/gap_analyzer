"""
GROBID client for extracting text from PDFs.
"""

import httpx
from typing import Dict, Any, Optional, List
import xml.etree.ElementTree as ET
from loguru import logger
import asyncio

from app.schemas.gap_schemas import ExtractedContent, PaperSearchResult
from app.utils.helpers import retry_async, parse_json_safely


class GrobidClient:
    """Client for interacting with GROBID service."""
    
    def __init__(self, grobid_url: str):
        self.grobid_url = grobid_url.rstrip('/')
        # Increased timeout significantly for large PDFs and GROBID processing
        self.client = httpx.AsyncClient(timeout=300.0)  # 5 minutes timeout
        # Semaphore to limit concurrent GROBID requests
        self.semaphore = asyncio.Semaphore(2)  # Max 2 concurrent requests
    
    @retry_async(max_attempts=3, delay=2)
    async def extract_from_url(
        self,
        pdf_url: str
    ) -> ExtractedContent:
        """Extract content from a PDF URL."""
        try:
            # First, download the PDF
            pdf_content = await self._download_pdf(pdf_url)
            if not pdf_content:
                return ExtractedContent(
                    title="",
                    extraction_success=False,
                    error="Failed to download PDF"
                )
            
            # Then extract using GROBID
            return await self.extract_from_bytes(pdf_content)
            
        except Exception as e:
            logger.error(f"Error extracting from URL {pdf_url}: {e}")
            return ExtractedContent(
                title="",
                extraction_success=False,
                error=str(e)
            )
    
    async def extract_from_bytes(
        self,
        pdf_bytes: bytes
    ) -> ExtractedContent:
        """Extract content from PDF bytes using GROBID with rate limiting."""
        async with self.semaphore:  # Limit concurrent requests
            return await self._extract_with_retry(pdf_bytes)
    
    async def _extract_with_retry(
        self,
        pdf_bytes: bytes,
        max_attempts: int = 3
    ) -> ExtractedContent:
        """Extract content with retry logic for 503 errors."""
        # Validate PDF size before processing
        if len(pdf_bytes) < 1000:  # Less than 1KB is likely not a valid PDF
            logger.warning(f"PDF too small ({len(pdf_bytes)} bytes), likely invalid or error page")
            return ExtractedContent(
                title="",
                extraction_success=False,
                error=f"PDF too small ({len(pdf_bytes)} bytes) - likely invalid or error page"
            )
        
        for attempt in range(max_attempts):
            try:
                logger.info(f"Calling GROBID API at: {self.grobid_url}/api/processFulltextDocument (attempt {attempt + 1})")
                logger.info(f"PDF size: {len(pdf_bytes)} bytes")
                
                # Call GROBID processFulltextDocument
                response = await self.client.post(
                    f"{self.grobid_url}/api/processFulltextDocument",
                    files={'input': ('document.pdf', pdf_bytes, 'application/pdf')},
                    data={
                        'consolidateHeader': '1',
                        'consolidateCitations': '0',
                        'includeRawCitations': '0',
                        'includeRawAffiliations': '0'
                    }
                )
                
                logger.info(f"GROBID API response status: {response.status_code}")
                if response.status_code == 200:
                    logger.info("GROBID extraction successful, parsing TEI XML response")
                    # Parse TEI XML response
                    return self._parse_tei_xml(response.text)
                elif response.status_code == 503:
                    # Service unavailable - wait longer and retry
                    wait_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
                    logger.warning(f"GROBID service unavailable (503), waiting {wait_time}s before retry {attempt + 1}/{max_attempts}")
                    if attempt < max_attempts - 1:  # Don't wait on last attempt
                        await asyncio.sleep(wait_time)
                        continue
                elif response.status_code == 500:
                    # Internal server error - likely invalid PDF
                    logger.error(f"GROBID internal server error (500) - PDF may be corrupted or invalid")
                    return ExtractedContent(
                        title="",
                        extraction_success=False,
                        error="GROBID internal server error - PDF may be corrupted or invalid"
                    )
                else:
                    logger.error(f"GROBID returned status {response.status_code}")
                    return ExtractedContent(
                        title="",
                        extraction_success=False,
                        error=f"GROBID error: {response.status_code}"
                    )
                    
            except Exception as e:
                import traceback
                logger.error(f"Error calling GROBID (attempt {attempt + 1}): {e}")
                logger.error(f"Error type: {type(e).__name__}")
                if attempt < max_attempts - 1:  # Don't log full trace on last attempt
                    logger.error(f"Stack trace: {traceback.format_exc()}")
                
                if attempt < max_attempts - 1:
                    wait_time = (2 ** attempt) * 2  # Exponential backoff: 2s, 4s
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
        
        # All attempts failed
        return ExtractedContent(
            title="",
            extraction_success=False,
            error="GROBID extraction failed after all retry attempts"
        )
    
    async def extract_batch(
        self,
        papers: List[PaperSearchResult]
    ) -> List[ExtractedContent]:
        """Extract content from multiple papers with controlled concurrency."""
        logger.info(f"Starting batch extraction for {len(papers)} papers")
        extracted_contents = []
        successful_extractions = 0
        
        # Process papers in smaller batches sequentially to avoid overwhelming GROBID
        batch_size = 3  # Process 3 papers at a time
        for batch_start in range(0, len(papers), batch_size):
            batch_end = min(batch_start + batch_size, len(papers))
            batch_papers = papers[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}: papers {batch_start + 1}-{batch_end}")
            
            # Create tasks for this batch
            tasks = []
            for i, paper in enumerate(batch_papers):
                paper_index = batch_start + i
                if paper.pdf_url:
                    logger.info(f"Paper {paper_index + 1}: '{paper.title}' - PDF URL available: {paper.pdf_url}")
                    tasks.append(self.extract_from_url(paper.pdf_url))
                else:
                    logger.info(f"Paper {paper_index + 1}: '{paper.title}' - No PDF URL available, using metadata only")
                    # Create empty extraction for papers without PDFs
                    tasks.append(asyncio.create_task(
                        self._create_extraction_from_metadata(paper)
                    ))
            
            # Process this batch concurrently (but limited by semaphore)
            logger.info(f"Starting extraction of {len(tasks)} papers in current batch")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results for this batch
            batch_successful = 0
            for i, result in enumerate(results):
                paper_index = batch_start + i
                if isinstance(result, ExtractedContent):
                    extracted_contents.append(result)
                    if result.extraction_success:
                        successful_extractions += 1
                        batch_successful += 1
                        logger.info("Successfully extracted paper %d: %s", paper_index + 1, result.title)
                    else:
                        logger.warning(f"Failed to extract paper {paper_index + 1}: {result.error}")
                elif isinstance(result, Exception):
                    import traceback
                    logger.error(f"Extraction error for paper {paper_index + 1}: {result}")
                    logger.error(f"Error type: {type(result).__name__}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")
                    extracted_contents.append(ExtractedContent(
                        title=papers[paper_index].title if paper_index < len(papers) else "",
                        extraction_success=False,
                        error=str(result)
                    ))
            
            # Log batch completion and wait before next batch
            logger.info("Batch %d completed: %d/%d successful", batch_start//batch_size + 1, batch_successful, len(tasks))
            if batch_end < len(papers):
                logger.info("Waiting 3 seconds before processing next batch to ensure GROBID stability...")
                await asyncio.sleep(3)
        
        logger.info(f"Batch extraction completed: {successful_extractions}/{len(papers)} successful")
        return extracted_contents
    
    async def _download_pdf(self, url: str) -> Optional[bytes]:
        """Download PDF from URL with multiple fallback strategies."""
        # Try multiple approaches for PDF download
        download_attempts = [
            self._try_direct_download,
            self._try_with_user_agent,
            self._try_alternative_urls
        ]
        
        for attempt_func in download_attempts:
            try:
                result = await attempt_func(url)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"Download attempt failed: {e}")
                continue
        
        logger.warning(f"All download attempts failed for URL: {url}")
        return None
    
    async def _try_direct_download(self, url: str) -> Optional[bytes]:
        """Try direct download without special headers."""
        logger.info(f"Attempting direct download from: {url}")
        response = await self.client.get(url, follow_redirects=True)
        logger.info(f"Direct download response status: {response.status_code}")
        
        if response.status_code == 200:
            content_size = len(response.content)
            logger.info(f"PDF downloaded successfully, size: {content_size} bytes")
            
            # Validate PDF size
            if content_size < 1000:  # Less than 1KB is likely not a valid PDF
                logger.warning(f"Downloaded file too small ({content_size} bytes), likely not a valid PDF")
                return None
            
            return response.content
        elif response.status_code == 403:
            logger.warning(f"Direct download failed with 403 Forbidden")
            return None
        else:
            logger.warning(f"Direct download failed with status: {response.status_code}")
            return None
    
    async def _try_with_user_agent(self, url: str) -> Optional[bytes]:
        """Try download with browser-like user agent."""
        logger.info(f"Attempting download with user agent from: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = await self.client.get(url, headers=headers, follow_redirects=True)
        logger.info(f"User agent download response status: {response.status_code}")
        
        if response.status_code == 200:
            content_size = len(response.content)
            logger.info(f"PDF downloaded with user agent, size: {content_size} bytes")
            
            # Validate PDF size
            if content_size < 1000:  # Less than 1KB is likely not a valid PDF
                logger.warning(f"Downloaded file too small ({content_size} bytes), likely not a valid PDF")
                return None
            
            return response.content
        else:
            logger.warning(f"User agent download failed with status: {response.status_code}")
            return None
    
    async def _try_alternative_urls(self, url: str) -> Optional[bytes]:
        """Try alternative URL formats for common academic repositories."""
        # Generate alternative URLs for common patterns
        alternative_urls = []
        
        # For arXiv URLs, try different formats
        if 'arxiv.org' in url:
            # Extract arXiv ID and try different URL formats
            import re
            arxiv_match = re.search(r'arxiv\.org/abs/(\d+\.\d+)', url)
            if arxiv_match:
                arxiv_id = arxiv_match.group(1)
                alternative_urls.extend([
                    f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                    f"https://arxiv.org/e-print/{arxiv_id}",
                ])
        
        # For PMC URLs, try different formats
        elif 'ncbi.nlm.nih.gov' in url:
            pmc_match = re.search(r'pmc/articles/(PMC\d+)', url)
            if pmc_match:
                pmc_id = pmc_match.group(1)
                alternative_urls.extend([
                    f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/",
                    f"https://europepmc.org/articles/{pmc_id}?pdf=render",
                ])
        
        # Try alternative URLs
        for alt_url in alternative_urls:
            try:
                logger.info(f"Trying alternative URL: {alt_url}")
                response = await self.client.get(alt_url, follow_redirects=True)
                if response.status_code == 200:
                    content_size = len(response.content)
                    logger.info(f"PDF downloaded from alternative URL, size: {content_size} bytes")
                    return response.content
            except Exception as e:
                logger.warning(f"Alternative URL failed: {e}")
                continue
        
        return None
    
    def _parse_tei_xml(self, xml_content: str) -> ExtractedContent:
        """Parse TEI XML from GROBID to extract relevant content."""
        try:
            # Parse XML
            root = ET.fromstring(xml_content)
            
            # Define namespace
            ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
            
            # Extract title
            title = ""
            title_elem = root.find('.//tei:titleStmt/tei:title', ns)
            if title_elem is not None and title_elem.text:
                title = title_elem.text.strip()
            
            # Extract abstract
            abstract = None
            abstract_elem = root.find('.//tei:abstract', ns)
            if abstract_elem is not None:
                abstract_text = ''.join(abstract_elem.itertext()).strip()
                if abstract_text:
                    abstract = abstract_text
            
            # Extract sections
            sections = []
            body = root.find('.//tei:body', ns)
            if body is not None:
                for div in body.findall('.//tei:div', ns):
                    section = self._extract_section(div, ns)
                    if section:
                        sections.append(section)
            
            # Extract specific sections
            methods = None
            results = None
            conclusion = None
            
            for section in sections:
                title_lower = section.get('title', '').lower()
                content = section.get('content', '')
                
                if 'method' in title_lower or 'approach' in title_lower:
                    methods = content
                elif 'result' in title_lower or 'experiment' in title_lower:
                    results = content
                elif 'conclusion' in title_lower or 'discussion' in title_lower:
                    conclusion = content
            
            return ExtractedContent(
                title=title,
                abstract=abstract,
                sections=sections,
                methods=methods,
                results=results,
                conclusion=conclusion,
                extraction_success=True
            )
            
        except Exception as e:
            logger.error(f"Error parsing TEI XML: {e}")
            return ExtractedContent(
                title="",
                extraction_success=False,
                error=f"XML parsing error: {str(e)}"
            )
    
    def _extract_section(
        self,
        div_elem: ET.Element,
        ns: Dict[str, str]
    ) -> Optional[Dict[str, str]]:
        """Extract a section from a TEI div element."""
        try:
            section = {}
            
            # Extract section title
            head = div_elem.find('tei:head', ns)
            if head is not None and head.text:
                section['title'] = head.text.strip()
            
            # Extract section content
            paragraphs = []
            for p in div_elem.findall('tei:p', ns):
                text = ''.join(p.itertext()).strip()
                if text:
                    paragraphs.append(text)
            
            if paragraphs:
                section['content'] = ' '.join(paragraphs)
                return section
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting section: {e}")
            return None
    
    async def _create_extraction_from_metadata(
        self,
        paper: PaperSearchResult
    ) -> ExtractedContent:
        """Create extraction from paper metadata when PDF is not available."""
        # Create a basic section from the abstract for better validation
        sections = []
        if paper.abstract:
            sections.append({
                'heading': 'Abstract',
                'content': paper.abstract,
                'level': '1'
            })
        
        return ExtractedContent(
            title=paper.title,
            abstract=paper.abstract,
            sections=sections,
            extraction_success=True,  # Mark as successful since we have metadata
            error=None
        )
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()