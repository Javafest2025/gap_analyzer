"""
Utility functions for the Gap Analysis Service.
app/utils/helpers.py
"""

import hashlib
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
import asyncio
from functools import wraps
from loguru import logger
import json


class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, max_calls: int, time_window: int):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    async def wait_if_needed(self):
        """Wait if rate limit is exceeded."""
        now = datetime.now(timezone.utc)
        
        # Remove old calls outside the time window
        cutoff = now - timedelta(seconds=self.time_window)
        self.calls = [call_time for call_time in self.calls if call_time > cutoff]
        
        # Check if we need to wait
        if len(self.calls) >= self.max_calls:
            oldest_call = min(self.calls)
            wait_time = (oldest_call + timedelta(seconds=self.time_window) - now).total_seconds()
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
                # Recursive call to clean up and check again
                await self.wait_if_needed()
        
        # Record this call
        self.calls.append(now)


def retry_async(max_attempts: int = 3, delay: int = 5):
    """
    Decorator for retrying async functions.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Delay between attempts in seconds
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )
            raise last_exception
        return wrapper
    return decorator


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\-.,;:!?\'"()]', '', text)
    
    # Trim
    text = text.strip()
    
    return text


def truncate_text(text: str, max_length: int, add_ellipsis: bool = True) -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        add_ellipsis: Whether to add ellipsis
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    
    if add_ellipsis and max_length > 3:
        return text[:max_length - 3] + "..."
    
    return text[:max_length]


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text.
    
    Args:
        text: Input text
        max_keywords: Maximum number of keywords
        
    Returns:
        List of keywords
    """
    if not text:
        return []
    
    # Simple keyword extraction based on word frequency
    # In production, consider using NLTK or spaCy
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter out common words (stopwords)
    stopwords = {
        'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 'was',
        'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
        'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'just', 'and', 'or', 'but', 'if', 'while',
        'with', 'about', 'against', 'between', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
        'in', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
        'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
        'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'for', 'of', 'by'
    }
    
    # Filter and count
    word_freq = {}
    for word in words:
        if word not in stopwords and len(word) > 2:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:max_keywords]]


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using Jaccard similarity.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score (0-1)
    """
    if not text1 or not text2:
        return 0.0
    
    # Convert to sets of words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)


def generate_hash(content: str) -> str:
    """
    Generate SHA256 hash of content.
    
    Args:
        content: Content to hash
        
    Returns:
        Hash string
    """
    return hashlib.sha256(content.encode()).hexdigest()


def format_paper_citation(
    title: str,
    authors: List[str],
    year: Optional[int] = None,
    venue: Optional[str] = None
) -> str:
    """
    Format paper citation.
    
    Args:
        title: Paper title
        authors: List of authors
        year: Publication year
        venue: Publication venue
        
    Returns:
        Formatted citation
    """
    citation_parts = []
    
    # Authors
    if authors:
        if len(authors) == 1:
            citation_parts.append(authors[0])
        elif len(authors) == 2:
            citation_parts.append(f"{authors[0]} and {authors[1]}")
        else:
            citation_parts.append(f"{authors[0]} et al.")
    
    # Year
    if year:
        citation_parts.append(f"({year})")
    
    # Title
    citation_parts.append(f'"{title}"')
    
    # Venue
    if venue:
        citation_parts.append(f"In {venue}")
    
    return ". ".join(citation_parts) + "."


def parse_json_safely(json_str: str, default: Any = None) -> Any:
    """
    Safely parse JSON string, handling markdown code blocks.
    
    Args:
        json_str: JSON string (may be wrapped in markdown code blocks)
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        # First try direct parsing
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        # Try to extract JSON from markdown code blocks
        try:
            import re
            
            # Look for JSON in markdown code blocks - improved pattern
            json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
            matches = re.findall(json_pattern, json_str, re.DOTALL | re.IGNORECASE)
            
            # Also try a more flexible pattern for incomplete code blocks
            if not matches:
                json_pattern_flexible = r'```(?:json)?\s*\n?(.*?)(?:\n?```|$)'
                matches = re.findall(json_pattern_flexible, json_str, re.DOTALL | re.IGNORECASE)
            
            if matches:
                # Use the first match
                json_content = matches[0].strip()
                return json.loads(json_content)
            
            # If no code blocks, try to find JSON-like content
            # Look for content between curly braces
            brace_pattern = r'\{.*\}'
            brace_matches = re.findall(brace_pattern, json_str, re.DOTALL)
            
            if brace_matches:
                # Try the largest match (most likely to be complete JSON)
                largest_match = max(brace_matches, key=len)
                return json.loads(largest_match)
            
            # If still no luck, try to find array-like content
            array_pattern = r'\[.*\]'
            array_matches = re.findall(array_pattern, json_str, re.DOTALL)
            
            if array_matches:
                largest_match = max(array_matches, key=len)
                return json.loads(largest_match)
                
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
        
        logger.warning(f"Failed to parse JSON: {json_str[:100]}...")
        return default


def batch_list(items: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split list into batches.
    
    Args:
        items: List of items
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    return batches


class AsyncBatchProcessor:
    """Process items in batches asynchronously."""
    
    def __init__(self, batch_size: int = 5, max_concurrent: int = 3):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Items per batch
            max_concurrent: Maximum concurrent batches
        """
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process(
        self,
        items: List[Any],
        process_func,
        *args,
        **kwargs
    ) -> List[Any]:
        """
        Process items in batches.
        
        Args:
            items: Items to process
            process_func: Async function to process each item
            *args: Additional arguments for process_func
            **kwargs: Additional keyword arguments for process_func
            
        Returns:
            List of results
        """
        batches = batch_list(items, self.batch_size)
        tasks = []
        
        for batch in batches:
            task = self._process_batch(batch, process_func, *args, **kwargs)
            tasks.append(task)
        
        batch_results = await asyncio.gather(*tasks)
        
        # Flatten results
        results = []
        for batch_result in batch_results:
            results.extend(batch_result)
        
        return results
    
    async def _process_batch(
        self,
        batch: List[Any],
        process_func,
        *args,
        **kwargs
    ) -> List[Any]:
        """Process a single batch."""
        async with self.semaphore:
            tasks = [process_func(item, *args, **kwargs) for item in batch]
            return await asyncio.gather(*tasks, return_exceptions=True)


# Export
__all__ = [
    'RateLimiter',
    'retry_async',
    'clean_text',
    'truncate_text',
    'extract_keywords',
    'calculate_similarity',
    'generate_hash',
    'format_paper_citation',
    'parse_json_safely',
    'batch_list',
    'AsyncBatchProcessor'
]