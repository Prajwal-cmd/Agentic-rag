"""
Semantic Scholar Research Paper Search Service

Pattern: Academic database integration with lazy loading and citation awareness
Source: Semantic Scholar API best practices, Enterprise RAG patterns
"""

import requests
from typing import List, Dict, Optional, Any
from ..config import settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class ResearchSearchService:
    """
    Semantic Scholar API integration for academic paper search.
    
    Industry Pattern: Multi-source academic retrieval with metadata enrichment
    Source: Semantic Scholar API documentation + Enterprise RAG best practices
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Semantic Scholar search client.
        
        Args:
            api_key: Semantic Scholar API key (optional but recommended)
        """
        self.api_key = api_key or settings.semantic_scholar_api_key
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = {}
        
        if self.api_key:
            self.headers["x-api-key"] = self.api_key
            logger.info("Semantic Scholar service initialized with API key")
        else:
            logger.warning("Semantic Scholar service initialized WITHOUT API key (rate limits apply)")
    
    def search_papers(
        self,
        query: str,
        limit: int = 5,
        year_from: Optional[int] = None,
        min_citations: Optional[int] = None,
        fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for academic papers using Semantic Scholar API.
        
        Pattern: Relevance-based search with metadata filtering
        
        Args:
            query: Search query
            limit: Maximum number of papers to return
            year_from: Only papers from this year onwards
            min_citations: Minimum citation count filter
            fields: Specific fields to retrieve
            
        Returns:
            List of paper dictionaries with metadata
        """
        # Default fields for RAG context
        if fields is None:
            fields = [
                "paperId",
                "title",
                "abstract",
                "year",
                "authors",
                "citationCount",
                "venue",
                "url",
                "openAccessPdf",
                "publicationTypes",
                "publicationDate"
            ]
        
        try:
            # Build search URL
            search_url = f"{self.base_url}/paper/search"
            
            params = {
                "query": query,
                "limit": limit,
                "fields": ",".join(fields)
            }
            
            # Add year filter if specified
            if year_from:
                params["year"] = f"{year_from}-"
            
            logger.info(f"Searching Semantic Scholar: query='{query}', limit={limit}")
            
            response = requests.get(
                search_url,
                params=params,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                papers = data.get("data", [])
                
                # Filter by minimum citations if specified
                if min_citations:
                    papers = [
                        p for p in papers
                        if p.get("citationCount", 0) >= min_citations
                    ]
                
                logger.info(f"Found {len(papers)} papers from Semantic Scholar")
                return papers
                
            elif response.status_code == 429:
                logger.error("Semantic Scholar rate limit exceeded")
                return []
            else:
                logger.error(f"Semantic Scholar API error: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching Semantic Scholar: {e}")
            return []
    
    def get_paper_details(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch detailed information about a specific paper.
        
        Args:
            paper_id: Semantic Scholar paper ID
            
        Returns:
            Paper details dictionary or None
        """
        try:
            url = f"{self.base_url}/paper/{paper_id}"
            params = {
                "fields": "paperId,title,abstract,year,authors,citationCount,venue,url,openAccessPdf,tldr,citations,references"
            }
            
            response = requests.get(
                url,
                params=params,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to fetch paper {paper_id}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching paper details: {e}")
            return None
    
    def format_paper_for_context(self, paper: Dict[str, Any]) -> str:
        """
        Format paper metadata into context string for LLM.
        
        Pattern: Structured context with citation metadata
        
        Args:
            paper: Paper dictionary from API
            
        Returns:
            Formatted string for RAG context
        """
        title = paper.get("title", "Unknown Title")
        authors = paper.get("authors", [])
        author_names = ", ".join([a.get("name", "") for a in authors[:3]])
        if len(authors) > 3:
            author_names += " et al."
        
        year = paper.get("year", "N/A")
        abstract = paper.get("abstract", "No abstract available")
        citations = paper.get("citationCount", 0)
        venue = paper.get("venue", "Unknown Venue")
        
        # Include TLDR if available (Semantic Scholar's AI-generated summary)
        tldr = paper.get("tldr", {})
        tldr_text = tldr.get("text", "") if isinstance(tldr, dict) else ""
        
        context = f"""
**Research Paper: {title}**
Authors: {author_names}
Year: {year} | Venue: {venue} | Citations: {citations}

{f"Summary: {tldr_text}" if tldr_text else ""}

Abstract: {abstract[:500]}{"..." if len(abstract) > 500 else ""}
"""
        return context.strip()
    
    def check_connection(self) -> bool:
        """
        Check if Semantic Scholar API is accessible.
        
        Returns:
            True if API is reachable, False otherwise
        """
        try:
            url = f"{self.base_url}/paper/search"
            params = {"query": "test", "limit": 1}
            
            response = requests.get(
                url,
                params=params,
                headers=self.headers,
                timeout=5
            )
            
            return response.status_code in [200, 429]  # 429 means API is up but rate limited
            
        except:
            return False

def get_research_search_service(api_key: Optional[str] = None) -> ResearchSearchService:
    """
    Factory function to get research search service instance.
    
    Args:
        api_key: Optional API key override
        
    Returns:
        ResearchSearchService instance
    """
    return ResearchSearchService(api_key)
