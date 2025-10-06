"""
Semantic Scholar Research Paper Search Service

Pattern: Query rewriting + decomposition for academic search
Source: Rewrite-Retrieve-Read (EMNLP 2023), ParallelSearch (2025)
"""

import requests
from typing import List, Dict, Optional, Any
import re
from ..config import settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class ResearchSearchService:
    """
    Semantic Scholar API integration with intelligent query rewriting.
    
    Industry Pattern: Rewrite-Retrieve-Read (proven 15-20% retrieval improvement)
    Source: Query Rewriting for RAG (Microsoft Research EMNLP 2023)
    """
    
    def __init__(self, api_key: Optional[str] = None, llm_service=None):
        """
        Initialize Semantic Scholar search client.
        
        Args:
            api_key: Semantic Scholar API key (optional but recommended)
            llm_service: LLM service for query rewriting
        """
        self.api_key = api_key or settings.semantic_scholar_api_key
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = {}
        
        if self.api_key:
            self.headers["x-api-key"] = self.api_key
            logger.info("Semantic Scholar service initialized with API key")
        else:
            logger.warning("Semantic Scholar service initialized WITHOUT API key (rate limits apply)")
        
        self.llm_service = llm_service
    
    def extract_research_keywords(self, query: str) -> str:
        """
        Extract research-focused keywords from complex queries.
        Pattern: Keyword extraction for academic search (Google Scholar, Semantic Scholar best practices)
        
        Examples:
            "compare uploaded paper with latest findings in recommendation systems"
            → "recommendation systems collaborative filtering neural networks recent"
        """
        query_lower = query.lower()
        
        # Remove comparison/meta words that don't help search
        stopwords = [
            'compare', 'uploaded', 'paper', 'with', 'latest', 'findings', 'in', 'the', 'field', 'of',
            'show', 'me', 'find', 'search', 'for', 'about', 'regarding', 'related', 'to'
        ]
        
        # Extract words
        words = re.findall(r'\b\w+\b', query_lower)
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        # Keep only significant keywords
        keyword_string = ' '.join(keywords[:8])  # Limit to 8 keywords
        
        logger.info(f"Extracted keywords: '{query}' → '{keyword_string}'")
        return keyword_string
    
    def rewrite_query_for_academic_search(self, query: str, llm_service=None) -> Dict[str, Any]:
        """
        Rewrite user query into academic search-optimized query.
        Pattern: Rewrite-Retrieve-Read (Microsoft Research EMNLP 2023)
        
        Returns:
            {
                "primary_query": str,  # Main search query
                "sub_queries": List[str],  # Decomposed sub-queries for parallel search
                "keywords": List[str],  # Extracted keywords
                "year_filter": Optional[int]  # Year threshold if temporal query
            }
        """
        try:
            # Use LLM for intelligent rewriting if available
            if llm_service or self.llm_service:
                service = llm_service or self.llm_service
                return self._llm_query_rewrite(query, service)
            
            # FALLBACK: Rule-based query rewriting
            return self._rule_based_query_rewrite(query)
        
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}, using fallback")
            return self._rule_based_query_rewrite(query)
    
    def _llm_query_rewrite(self, query: str, llm_service) -> Dict[str, Any]:
        """
        LLM-powered query rewriting for academic search.
        Pattern: Rewrite-Retrieve-Read (proven 15-20% improvement)
        """
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at rewriting queries for academic paper search (Semantic Scholar API).

**Your task:** Transform the user query into optimized search queries for finding relevant research papers.

**Rules:**
1. Extract the CORE research topic (e.g., "recommendation systems", "neural networks")
2. Add relevant academic keywords (e.g., "algorithm", "model", "approach")
3. Remove meta-words like "compare", "latest", "uploaded paper", "findings"
4. If query mentions "latest" or "recent", suggest year filter
5. If query is complex, decompose into 2-3 parallel sub-queries

**Output format (JSON-like):**
Primary query: [main search query - concise, keyword-focused]
Sub-queries: [sub-query 1], [sub-query 2], [sub-query 3]
Keywords: [keyword1], [keyword2], [keyword3]
Year filter: [2020 or null]

**Examples:**

Input: "compare the uploaded paper with the latest findings in recommendation systems"
Primary query: recommendation systems algorithms recent advances
Sub-queries: collaborative filtering deep learning, neural collaborative filtering, recommendation algorithms survey
Keywords: recommendation, collaborative filtering, neural networks, deep learning
Year filter: 2022

Input: "what are recent studies on transformers for NLP"
Primary query: transformers natural language processing recent
Sub-queries: transformer architecture NLP, BERT GPT models, attention mechanism language
Keywords: transformers, NLP, attention, BERT, GPT
Year filter: 2021

Input: "machine learning for healthcare"
Primary query: machine learning healthcare medical diagnosis
Sub-queries: clinical decision support, medical image analysis, predictive healthcare models
Keywords: machine learning, healthcare, clinical, diagnosis
Year filter: null"""),
            ("human", "Query: {query}\n\nRewrite for academic search:")
        ])
        
        try:
            llm = llm_service.get_llm(settings.routing_model, temperature=0.3)
            chain = prompt | llm | StrOutputParser()
            
            result = chain.invoke({"query": query})
            
            # Parse LLM output
            parsed = self._parse_llm_rewrite_output(result, query)
            logger.info(f"LLM query rewrite: {parsed['primary_query']}")
            
            return parsed
        
        except Exception as e:
            logger.error(f"LLM rewriting failed: {e}, using rule-based fallback")
            return self._rule_based_query_rewrite(query)
    
    def _parse_llm_rewrite_output(self, llm_output: str, original_query: str) -> Dict[str, Any]:
        """Parse LLM output into structured format."""
        lines = llm_output.strip().split('\n')
        
        primary_query = ""
        sub_queries = []
        keywords = []
        year_filter = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            if line_lower.startswith('primary query:'):
                primary_query = line.split(':', 1)[1].strip()
            elif line_lower.startswith('sub-queries:') or line_lower.startswith('sub-query'):
                # Extract comma-separated sub-queries
                parts = line.split(':', 1)[1].strip()
                sub_queries = [sq.strip() for sq in parts.split(',') if sq.strip()]
            elif line_lower.startswith('keywords:'):
                parts = line.split(':', 1)[1].strip()
                keywords = [kw.strip() for kw in parts.split(',') if kw.strip()]
            elif line_lower.startswith('year filter:'):
                year_str = line.split(':', 1)[1].strip()
                try:
                    if year_str and year_str.lower() != 'null':
                        year_filter = int(re.search(r'\d{4}', year_str).group())
                except:
                    year_filter = None
        
        # Fallback to rule-based if parsing failed
        if not primary_query:
            primary_query = self.extract_research_keywords(original_query)
        
        if not sub_queries:
            sub_queries = [primary_query]
        
        return {
            "primary_query": primary_query,
            "sub_queries": sub_queries[:3],  # Limit to 3 for parallel search
            "keywords": keywords[:5],
            "year_filter": year_filter
        }
    
    def _rule_based_query_rewrite(self, query: str) -> Dict[str, Any]:
        """
        Rule-based query rewriting (fallback).
        Pattern: Keyword extraction + decomposition for academic APIs
        """
        query_lower = query.lower()
        
        # Extract temporal context
        year_filter = None
        temporal_keywords = ['latest', 'recent', 'new', '2024', '2025', 'current']
        if any(kw in query_lower for kw in temporal_keywords):
            # Extract year if mentioned, otherwise use 2020 as threshold for "recent"
            year_match = re.search(r'20\d{2}', query)
            year_filter = int(year_match.group()) if year_match else 2022
        
        # Extract primary keywords
        primary_query = self.extract_research_keywords(query)
        
        # Decompose complex queries
        sub_queries = [primary_query]
        
        # If query mentions comparison, create sub-queries
        if 'compare' in query_lower or 'vs' in query_lower or 'versus' in query_lower:
            # Extract compared entities
            entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|\b[a-z]+\s+systems?\b|\b[a-z]+\s+networks?\b', query)
            if entities:
                for entity in entities[:2]:  # Max 2 sub-queries
                    sub_queries.append(f"{entity} {primary_query}")
        
        keywords = primary_query.split()[:5]
        
        return {
            "primary_query": primary_query,
            "sub_queries": sub_queries[:3],
            "keywords": keywords,
            "year_filter": year_filter
        }
    
    def search_papers(
        self,
        query: str,
        limit: int = 5,
        year_from: Optional[int] = None,
        min_citations: Optional[int] = None,
        fields: Optional[List[str]] = None,
        use_query_rewriting: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for academic papers using Semantic Scholar API with query rewriting.
        
        Pattern: Rewrite-Retrieve-Read (proven 15-20% retrieval improvement)
        
        Args:
            query: Search query
            limit: Maximum number of papers to return
            year_from: Only papers from this year onwards
            min_citations: Minimum citation count filter
            fields: Specific fields to retrieve
            use_query_rewriting: Whether to rewrite query before search
        
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
                "publicationDate",
                "tldr"
            ]
        
        try:
            # STEP 1: Query Rewriting (NEW - Industry Standard)
            if use_query_rewriting:
                rewrite_result = self.rewrite_query_for_academic_search(query, self.llm_service)
                search_query = rewrite_result["primary_query"]
                sub_queries = rewrite_result["sub_queries"]
                
                # Use year filter from rewriting if not explicitly provided
                if year_from is None and rewrite_result.get("year_filter"):
                    year_from = rewrite_result["year_filter"]
                
                logger.info(f"Rewritten query: '{query}' → '{search_query}'")
                logger.info(f"Sub-queries: {sub_queries}")
            else:
                search_query = query
                sub_queries = [query]
            
            # STEP 2: Parallel Search (if multiple sub-queries)
            all_papers = []
            seen_paper_ids = set()
            
            # Search with primary query
            papers = self._search_semantic_scholar(
                search_query,
                limit=limit,
                year_from=year_from,
                fields=fields
            )
            
            for paper in papers:
                paper_id = paper.get("paperId")
                if paper_id and paper_id not in seen_paper_ids:
                    all_papers.append(paper)
                    seen_paper_ids.add(paper_id)
            
            # If primary query returned nothing, try sub-queries
            if len(all_papers) == 0 and len(sub_queries) > 1:
                logger.info("Primary query returned 0 results, trying sub-queries...")
                
                for sub_query in sub_queries[1:]:  # Skip first (primary)
                    if len(all_papers) >= limit:
                        break
                    
                    sub_papers = self._search_semantic_scholar(
                        sub_query,
                        limit=max(2, limit // 2),  # Smaller limit for sub-queries
                        year_from=year_from,
                        fields=fields
                    )
                    
                    for paper in sub_papers:
                        paper_id = paper.get("paperId")
                        if paper_id and paper_id not in seen_paper_ids:
                            all_papers.append(paper)
                            seen_paper_ids.add(paper_id)
                            if len(all_papers) >= limit:
                                break
            
            # STEP 3: Filter by minimum citations if specified
            if min_citations:
                all_papers = [
                    p for p in all_papers
                    if p.get("citationCount", 0) >= min_citations
                ]
            
            # STEP 4: Sort by relevance (citation count + year)
            all_papers = self._rank_papers(all_papers)
            
            logger.info(f"Found {len(all_papers)} papers from Semantic Scholar")
            return all_papers[:limit]
        
        except Exception as e:
            logger.error(f"Error searching Semantic Scholar: {e}")
            return []
    
    def _search_semantic_scholar(
        self,
        query: str,
        limit: int,
        year_from: Optional[int],
        fields: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Direct Semantic Scholar API call (internal helper).
        """
        try:
            search_url = f"{self.base_url}/paper/search"
            params = {
                "query": query,
                "limit": limit,
                "fields": ",".join(fields)
            }
            
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
                logger.info(f"  → Found {len(papers)} papers")
                return papers
            elif response.status_code == 429:
                logger.error("Semantic Scholar rate limit exceeded")
                return []
            else:
                logger.error(f"Semantic Scholar API error: {response.status_code}")
                return []
        
        except Exception as e:
            logger.error(f"Semantic Scholar request error: {e}")
            return []
    
    def _rank_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank papers by relevance score (combination of citations and recency).
        Pattern: Citation-aware ranking (Google Scholar approach)
        """
        def relevance_score(paper):
            citations = paper.get("citationCount", 0)
            year = paper.get("year", 2000)
            
            # Newer papers get boost
            recency_boost = (year - 2000) / 25.0  # 0.0 to 1.0
            
            # Citation score (log scale to avoid dominance of highly-cited papers)
            import math
            citation_score = math.log(citations + 1)
            
            # Combined score
            return citation_score * 0.7 + recency_boost * 0.3
        
        return sorted(papers, key=relevance_score, reverse=True)
    
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

def get_research_search_service(api_key: Optional[str] = None, llm_service=None) -> ResearchSearchService:
    """
    Factory function to get research search service instance.
    
    Args:
        api_key: Optional API key override
        llm_service: Optional LLM service for query rewriting
    
    Returns:
        ResearchSearchService instance
    """
    return ResearchSearchService(api_key, llm_service)
