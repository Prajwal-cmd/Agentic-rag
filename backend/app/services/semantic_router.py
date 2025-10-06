"""
Semantic Router for Fast Query Classification

Pattern: Embedding-based routing (80% of queries)
Source: Semantic Router, Advanced RAG Retrieval Strategies
"""

from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class SemanticRouter:
    """
    Fast semantic routing using embedding similarity.
    Replaces LLM calls for 80% of queries, reducing latency by 200-400ms.
    """
    
    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
        
        # UPDATED: Added follow-up/continuation templates
        self.route_templates = {
            "vectorstore": [
                "what does the document say about X",
                "summarize the uploaded file",
                "explain section 3 of the paper",
                "what's in my document",
                "analyze the methodology in the PDF",
                "what does my document contain",
                "review the uploaded research paper"
            ],
            "web_search": [
                "what's the weather today",
                "latest news about X",
                "current stock price",
                "who won the election",
                "recent events in",
                "what happened today",
                "breaking news about"
            ],
            "research": [
                "show me research papers on X",
                "academic studies about Y",
                "scientific publications on Z",
                "what does research say",
                "peer-reviewed papers about",
                "arxiv papers on topic X",
                "seminal work in field Y"
            ],
            "hybrid_web_research": [
                "what are recent research and studies on X",
                "latest academic findings in Y",
                "recent scientific discoveries about Z",
                "what do recent papers say about X",
                "current research trends in Y",
                "latest studies and breakthroughs in Z",
                "new research published in 2024 on X",
                "recent survey papers about Y",
                "state of the art in Z field",
                "what are the latest findings about X"
            ],
            "hybrid": [
                "compare the document with web information",
                "supplement my docs with online data",
                "combine uploaded info with web search",
                "check my document against current data"
            ],
            "direct_llm": [
                "hello",
                "what is X",
                "explain the concept of Y",
                "how does Z work",
                "define artificial intelligence",
                "tell me about machine learning",
                # NEW: Follow-up templates
                "multiply it by 3",
                "what is that plus 5",
                "use that value",
                "calculate the result times 2",
                "what about it",
                "solve it again",
                "with the same approach"
            ],
            "computational": [
                "calculate 5 + 3",
                "solve the equation x + 2 = 5",
                "what is 15% of 200",
                "compute the derivative of x squared",
                # NEW: Follow-up computational
                "x + 5",
                "multiply by 2",
                "that divided by 3"
            ]
        }
        
        # Pre-compute embeddings for route templates
        self.route_embeddings = self._compute_route_embeddings()
        logger.info(f"Semantic Router initialized with {len(self.route_templates)} routes")
    
    def _compute_route_embeddings(self) -> Dict[str, np.ndarray]:
        """Pre-compute embeddings for all route templates."""
        route_embeddings = {}
        for route, examples in self.route_templates.items():
            # Compute embeddings for all examples
            embeddings = [
                self.embedding_service.embed_text(example)
                for example in examples
            ]
            
            # Average embedding represents the route
            route_embeddings[route] = np.mean(embeddings, axis=0)
        
        return route_embeddings
    
    def route(
        self,
        query: str,
        has_documents: bool = False
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Route query using semantic similarity.
        
        Args:
            query: User query string
            has_documents: Whether documents are available
        
        Returns:
            (route_name, confidence_score, all_similarities)
        """
        # Compute query embedding
        query_embedding = self.embedding_service.embed_text(query)
        
        # Calculate similarity with each route
        similarities = {}
        for route, route_embedding in self.route_embeddings.items():
            similarity = cosine_similarity(
                [query_embedding],
                [route_embedding]
            )[0][0]
            similarities[route] = float(similarity)
        
        # Get best route
        best_route = max(similarities, key=similarities.get)
        confidence = similarities[best_route]
        
        # Log top 3 matches for debugging
        top_3 = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]
        logger.info(f"Top 3 semantic matches: {top_3}")
        
        # Apply document availability constraints
        if best_route in ["vectorstore", "hybrid", "hybrid_research"] and not has_documents:
            # Fallback logic
            if "research" in best_route:
                best_route = "hybrid_web_research"
                logger.info(f"Fallback: {best_route} (no documents)")
            else:
                best_route = "web_search"
                confidence = confidence * 0.8
                logger.info(f"Fallback: web_search (no documents, confidence reduced)")
        
        logger.info(f"Semantic routing: {best_route} (confidence={confidence:.3f})")
        
        return best_route, confidence, similarities

def get_semantic_router(embedding_service) -> SemanticRouter:
    """Factory function for semantic router."""
    return SemanticRouter(embedding_service)
