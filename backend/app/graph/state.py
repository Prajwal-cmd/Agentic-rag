"""
LangGraph State Definition
Pattern: TypedDict for type-safe state management with dialog management
Source: Official LangGraph documentation + Microsoft Bot Framework patterns
"""

from typing import TypedDict, List, Dict, Optional, Literal
from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    State object passed through the graph workflow.
    Each node can read and update specific fields.
    TypedDict provides type hints for IDE support and validation.
    """
    
    # User input
    question: str  # Current query (may be transformed)
    original_question: str  # Original query before any transformations
    enriched_question: Optional[str]  # FIXED: Context-enriched query for persistence
    
    # Retrieved data
    documents: List[Document]  # Documents from RAG or web search
    research_papers: Optional[List[Dict]]  # Research papers from Semantic Scholar
    
    # Output
    generation: str  # Final answer
    
    # Control flow flags
    web_search_needed: bool  # Whether to supplement with web search
    research_needed: bool  # Whether to fetch academic papers
    route_decision: str  # Routing: vectorstore/web_search/hybrid/direct_llm/clarification/research/hybrid_research
    
    # Context
    conversation_history: List[Dict]  # Chat history (may be summarized)
    session_id: str  # Session identifier
    
    # Enhanced routing metadata
    routing_confidence: Optional[float]  # Confidence in routing decision
    query_complexity: Optional[str]  # Query complexity level
    routing_reason: Optional[str]  # Explanation for routing choice
    
    # Dialog management (for clarification flows)
    needs_clarification: Optional[bool]  # Whether clarification is needed
    clarification_type: Optional[str]  # Type of clarification needed
    clarification_message: Optional[str]  # Message to show user
    clarification_options: Optional[List[str]]  # Suggested options
    dialog_state: Optional[str]  # Dialog state: normal/awaiting_clarification/clarified
    
    # FIXED: Working memory (for context persistence)
    working_memory: Optional[Dict[str, str]]  # Variables, facts from conversation
    
    # Sources and metadata
    sources: Optional[List[Dict]]  # Source citations
    
    # Context note (for soft clarifications)
    context_note: Optional[str]  # Additional context or notes
