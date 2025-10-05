"""
Tavily Web Search Service
Pattern: AI-optimized search API with clean, summarized results
Source: Tavily API (1,000 searches/month free tier)
"""
from tavily import TavilyClient
from typing import List, Dict
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class WebSearchService:
    """
    Tavily web search integration for supplemental information retrieval.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize Tavily client.
        
        Args:
            api_key: Tavily API key
        """
        self.client = TavilyClient(api_key=api_key)
        logger.info("Tavily search service initialized")
    
    def search(self, query: str, max_results: int = 3) -> List[Dict]:
        """
        Perform web search and return clean results.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of search results with content, title, URL
        """
        try:
            logger.info(f"Performing web search for: {query}")
            
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",  # "basic" or "advanced"
                include_answer=False,   # We'll generate our own answer
                include_raw_content=False  # We only need summaries
            )
            
            results = []
            for result in response.get("results", []):
                results.append({
                    "content": result.get("content", ""),
                    "title": result.get("title", "Untitled"),
                    "url": result.get("url", ""),
                    "score": result.get("score", 0.0)
                })
            
            logger.info(f"Retrieved {len(results)} search results")
            return results
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []

# Global instance
web_search_service = None

def get_web_search_service(api_key: str) -> WebSearchService:
    """Get or create global web search service instance"""
    global web_search_service
    if web_search_service is None:
        web_search_service = WebSearchService(api_key)
    return web_search_service