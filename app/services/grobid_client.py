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
        self.client = httpx.AsyncClient(timeout=60.0)
    
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
    
    @retry_async(max_attempts=3, delay=2)
    async def extract_from_bytes(
        self,
        pdf_bytes: bytes
    ) -> ExtractedContent:
        """Extract content from PDF bytes using GROBID."""
        try:
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
            
            if response.status_code == 200:
                # Parse TEI XML response
                return self._parse_tei_xml(response.text)
            else:
                logger.error(f"GROBID returned status {response.status_code}")
                return ExtractedContent(
                    title="",
                    extraction_success=False,
                    error=f"GROBID error: {response.status_code}"
                )
                
        except Exception as e:
            logger.error(f"Error calling GROBID: {e}")
            return ExtractedContent(
                title="",
                extraction_success=False,
                error=str(e)
            )
    
    async def extract_batch(
        self,
        papers: List[PaperSearchResult]
    ) -> List[ExtractedContent]:
        """Extract content from multiple papers in batch."""
        tasks = []
        
        for paper in papers:
            if paper.pdf_url:
                tasks.append(self.extract_from_url(paper.pdf_url))
            else:
                # Create empty extraction for papers without PDFs
                tasks.append(asyncio.create_task(
                    self._create_extraction_from_metadata(paper)
                ))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        extracted_contents = []
        for i, result in enumerate(results):
            if isinstance(result, ExtractedContent):
                extracted_contents.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Extraction error for paper {i}: {result}")
                extracted_contents.append(ExtractedContent(
                    title=papers[i].title if i < len(papers) else "",
                    extraction_success=False,
                    error=str(result)
                ))
        
        return extracted_contents
    
    async def _download_pdf(self, url: str) -> Optional[bytes]:
        """Download PDF from URL."""
        try:
            response = await self.client.get(url, follow_redirects=True)
            if response.status_code == 200:
                return response.content
            else:
                logger.warning(f"Failed to download PDF: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error downloading PDF: {e}")
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
        return ExtractedContent(
            title=paper.title,
            abstract=paper.abstract,
            sections=[],
            extraction_success=False,
            error="No PDF available"
        )
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()