"""
LangGraph Conditional Edges with Clarification Support
Pattern: State-based routing with dialog management
"""

from typing import Dict, Any, Literal
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


def decide_to_generate(state: Dict[str, Any]) -> Literal["transform_query", "generate"]:
    """
    Decide whether to generate answer or seek more information.
    Pattern: Adaptive routing based on retrieval quality
    Source: CRAG paper - confidence-based branching
    
    Args:
        state: Current graph state
    
    Returns:
        Next node name: "transform_query" or "generate"
    """
    web_search_needed = state.get("web_search_needed", False)
    
    if web_search_needed:
        logger.info("EDGE: Routing to transform_query (low confidence)")
        return "transform_query"
    else:
        logger.info("EDGE: Routing to generate (high confidence)")
        return "generate"


def route_question_edge(state: Dict[str, Any]) -> Literal[
    "retrieve_documents",
    "web_search",
    "direct_llm_generate",
    "generate_clarification",
    "wait_for_upload",
    "research_search",
    "hybrid_web_research_generate"  # NEW
]:
    """
    FIXED: Route based on initial query classification with all routing options.
    """
    route_decision = state.get("route_decision", "direct_llm_generate")
    logger.info(f"EDGE: route_question_edge -> {route_decision}")
    
    if route_decision == "clarification":
        return "generate_clarification"
    
    elif route_decision == "wait_for_upload":
        return "wait_for_upload"
    
    elif route_decision in ["web_search", "websearch"]:
        return "web_search"
    
    elif route_decision == "vectorstore":
        return "retrieve_documents"
    
    elif route_decision == "hybrid":
        return "retrieve_documents"
    
    elif route_decision == "research":
        return "research_search"
    
    elif route_decision == "hybrid_research":
        return "retrieve_documents"
    
    elif route_decision == "hybrid_web_research":  # NEW
        return "hybrid_web_research_generate"
    
    else:
        return "direct_llm_generate"


def clarification_edge(state: Dict[str, Any]) -> Literal["route_question", "END"]:
    """
    Handle clarification response routing.
    
    Args:
        state: Current graph state
    
    Returns:
        Next node name
    """
    dialog_state = state.get("dialog_state", "normal")
    
    if dialog_state == "clarified":
        logger.info("EDGE: Clarification complete, resuming routing")
        return "route_question"
    else:
        logger.info("EDGE: Awaiting user clarification")
        return "END"


def decide_research_hybrid(state: Dict[str, Any]) -> Literal["research_search", "generate"]:
    """
    Decide whether to fetch research papers after retrieving documents.
    Pattern: Multi-source retrieval for comprehensive answers
    
    Args:
        state: Current graph state
    
    Returns:
        Next node name
    """
    route_decision = state.get("route_decision", "")
    research_needed = state.get("research_needed", False)
    
    if "research" in route_decision or research_needed:
        logger.info("EDGE: Fetching research papers for hybrid retrieval")
        return "research_search"
    else:
        logger.info("EDGE: Proceeding to generation")
        return "generate"
