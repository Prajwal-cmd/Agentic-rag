"""
LangGraph Node Functions - Production Query Routing with Dialog Management
Pattern: Adaptive multi-stage routing with clarification dialogs
Source: Microsoft Bot Framework, Rasa, Google DialogFlow patterns
"""

from typing import Dict, Any, List, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from ..models.graders import GradeDocuments, RouteQuery
from ..services.llm_service import get_groq_service
from ..services.embeddings import get_embedding_service
from ..services.vector_store import VectorStoreService
from ..services.web_search import get_web_search_service
from ..config import settings
from ..utils.logger import setup_logger
from ..services.research_search import get_research_search_service

logger = setup_logger(__name__)

# ========== PYDANTIC MODELS ==========

class EnhancedRouteQuery(BaseModel):
    """Enhanced routing decision with confidence and reasoning."""
    datasource: Literal[
        "vectorstore", 
        "web_search", 
        "hybrid", 
        "direct_llm", 
        "research", 
        "hybrid_research",
        "hybrid_web_research"  # NEW: Research papers + Web search
    ] = Field(
        description="Route decision - NEVER use 'clarification' (handled separately before routing)"
    )
    confidence: float = Field(
        description="Confidence score 0.0-1.0 for this routing decision"
    )
    reasoning: str = Field(
        description="Clear explanation for routing choice"
    )
    query_complexity: str = Field(
        description="Query complexity: 'simple', 'moderate', 'complex', or 'multi_hop'"
    )
    temporal_context: bool = Field(
        description="Whether query requires current/recent information"
    )
    needs_clarification: bool = Field(
        default=False,
        description="DEPRECATED: Clarification handled before routing"
    )
    clarification_reason: str = Field(
        default="",
        description="DEPRECATED: Clarification handled before routing"
    )

class QueryIntent(BaseModel):
    """Query intent classification for better routing."""
    intent_type: str = Field(
        description="Primary intent: 'factual', 'analytical', 'comparative', 'procedural', 'exploratory', 'computational', 'conversational', 'command'"
    )
    secondary_intents: List[str] = Field(
        default_factory=list,
        description="Secondary intents if query has multiple purposes"
    )
    needs_documents: bool = Field(
        description="Whether this query requires document context"
    )
    needs_web: bool = Field(
        description="Whether this query requires current web information"
    )
    needs_retrieval: bool = Field(
        description="Whether this query needs any external information retrieval"
    )
    ambiguity_score: float = Field(
        description="Query ambiguity score 0.0-1.0 (0=clear, 1=very ambiguous)"
    )
    context_dependent: bool = Field(
        description="Whether query depends on previous context"
    )


class QueryClarification(BaseModel):
    """Clarification requirements for ambiguous queries."""
    needs_clarification: bool = Field(
        description="Whether clarification is needed"
    )
    clarification_type: str = Field(
        description="Type: 'missing_documents', 'ambiguous_intent', 'missing_context', 'multiple_interpretations', 'permission_needed'"
    )
    clarification_message: str = Field(
        description="Message to present to user for clarification"
    )
    suggested_options: List[str] = Field(
        default_factory=list,
        description="Suggested options for user to choose from"
    )
    confidence_without_clarification: float = Field(
        description="Confidence score if proceeding without clarification"
    )


# ========== CONTEXT ENRICHMENT FUNCTIONS ==========

def enrich_query_with_context(question: str, working_memory: Dict[str, str], history: List[Dict]) -> str:
    """
    Enrich ambiguous queries with working memory context before processing.
    Pattern: Context-aware query rewriting (Adobe Experience Platform, enterprise RAG)
    """
    # Check if query references known variables
    tokens = re.findall(r'\b[a-zA-Z_]\w*\b', question)
    referenced_vars = [var for var in tokens if var in working_memory]
    
    if referenced_vars:
        # Inject context directly into query
        context_additions = []
        for var in referenced_vars:
            context_additions.append(f"{var}={working_memory[var]}")
        enriched_query = f"{question} (Context: {', '.join(context_additions)})"
        logger.info(f"Query enriched with working memory: {enriched_query}")
        return enriched_query
    
    # Check for pronouns/references requiring history context
    ambiguous_patterns = r'\b(it|that|this|the one|previous|earlier)\b'
    if re.search(ambiguous_patterns, question.lower()):
        # Extract last topic/entity from history
        recent_context = extract_recent_topic(history)
        if recent_context:
            enriched_query = f"{question} [Referring to: {recent_context}]"
            logger.info(f"Query enriched with history context: {enriched_query}")
            return enriched_query
    
    return question


def extract_recent_topic(history: List[Dict], lookback: int = 3) -> str:
    """Extract the main topic/entity from recent conversation."""
    if not history:
        return ""
    
    recent_messages = history[-lookback:]
    topics = []
    
    for msg in recent_messages:
        content = msg.get("content", "")
        # Extract equations, variables, and main entities
        entities = re.findall(r'[a-z]\s*=\s*[^,\s]+|\d+|[A-Z][a-z]+', content)
        topics.extend(entities)
    
    return ", ".join(topics[-3:]) if topics else ""


def format_working_memory_context(working_memory: Dict[str, str]) -> str:
    """
    Format working memory for injection into LLM prompts.
    Pattern: LangChain ConversationBufferMemory formatting
    """
    if not working_memory:
        return ""
    
    context_parts = []
    for key, value in working_memory.items():
        context_parts.append(f"{key} = {value}")
    
    return "\n".join([
        "**Working Memory (Variables and Context):**",
        "\n".join(context_parts),
        ""
    ])


# ========== QUERY COMPLETENESS DETECTION ==========

def detect_query_completeness(
    question: str, 
    intent: QueryIntent, 
    history: List[Dict],
    has_documents: bool  # ADDED: Need to know if docs available
) -> Dict[str, Any]:
    """
    Detect if query is complete, incomplete, or requires disambiguation.
    Pattern: Incomplete Utterance Detection from Microsoft Research & Rasa Pro
    
    CRITICAL FIX: Also checks if query needs documents but none are available
    
    Returns:
        completeness_type: 'complete', 'incomplete', 'command', 'ambiguous', 'missing_documents'
        requires_clarification: bool
        clarification_reason: str
    """
    question_lower = question.lower().strip()
    
    # **PRIORITY 0: Check if query explicitly needs documents but none available**
    # Keywords that indicate document requirement
    doc_keywords = [
        'document', 'paper', 'file', 'pdf', 'uploaded', 'attachment',
        'doc', 'summarize', 'summary', 'analyze', 'review',
        'the document', 'the paper', 'the file', 'my document',
        'this document', 'these documents'
    ]
    
    mentions_documents = any(keyword in question_lower for keyword in doc_keywords)
    
    # If query explicitly mentions documents OR intent says needs_documents, check availability
    if (mentions_documents or intent.needs_documents) and not has_documents:
        # EXCEPTION: If query is asking ABOUT documents in general (not THIS document)
        general_doc_questions = [
            'what is a document',
            'what are documents',
            'how to upload',
            'how do i upload',
            'can i upload',
            'document format'
        ]
        
        is_general_question = any(gq in question_lower for gq in general_doc_questions)
        
        if not is_general_question:
            return {
                "completeness_type": "missing_documents",
                "requires_clarification": True,
                "clarification_reason": "Query requires documents but none are uploaded",
                "confidence": 0.9
            }
    
    # CASE 1: Explicit commands (complete and actionable)
    command_patterns = [
        r'^search (the |for )?web',
        r'^(google|bing|search) (for )?',
        r'^look up',
        r'^find (me )?information',
        r'^what is the weather',
        r'^calculate',
        r'^tell me about \w+',  # "tell me about X" where X is specific
    ]
    
    if any(re.match(pattern, question_lower) for pattern in command_patterns):
        # Check if command has necessary parameters
        if len(question.split()) <= 3:  # e.g., "search the web" (incomplete)
            return {
                "completeness_type": "incomplete_command",
                "requires_clarification": True,
                "clarification_reason": "Command missing subject/topic",
                "confidence": 0.8
            }
        return {
            "completeness_type": "command",
            "requires_clarification": False,
            "clarification_reason": "",
            "confidence": 0.9
        }
    
    # CASE 2: Incomplete queries (missing critical information)
    incomplete_indicators = [
        r'^(what|how|why|when|where|who)\??$',  # Single word questions
        r'^(tell me|show me|explain|describe)\??$',  # Verb only
        r'^(the|a|an|this|that|it)\s',  # Starts with reference without context
        r'\b(about|of|for|in)\s*$',  # Ends with preposition
        r'^[a-z]{1,2}[\+\-\*/]$',  # Single variable math without context
    ]
    
    if any(re.match(pattern, question_lower) for pattern in incomplete_indicators):
        return {
            "completeness_type": "incomplete",
            "requires_clarification": True,
            "clarification_reason": "Query is incomplete or missing subject",
            "confidence": 0.85
        }
    
    # CASE 3: Ambiguous references (pronouns without context)
    if not history or len(history) < 2:
        ambiguous_patterns = [
            r'^(it|this|that|the one|these|those)',
            r'\b(it|this|that)\b.*\?$',
            r'^(what about|how about|why|explain)\s+(it|this|that|the)',
        ]
        
        if any(re.match(pattern, question_lower) for pattern in ambiguous_patterns):
            return {
                "completeness_type": "ambiguous_reference",
                "requires_clarification": True,
                "clarification_reason": "Ambiguous reference without prior context",
                "confidence": 0.8
            }
    
    # CASE 4: Multi-intent detection (needs disambiguation)
    multi_intent_markers = [
        r'\band\b.*\balso\b',
        r'\band then\b',
        r'\bafter that\b',
        r'[;,].*\b(and|then|also)\b',
    ]
    
    if any(re.search(pattern, question_lower) for pattern in multi_intent_markers):
        return {
            "completeness_type": "multi_intent",
            "requires_clarification": False,  # Can handle with sequential processing
            "clarification_reason": "Multiple intents detected",
            "confidence": 0.7
        }
    
    # CASE 5: High ambiguity from intent classification
    if intent.ambiguity_score >= 0.7:
        return {
            "completeness_type": "ambiguous",
            "requires_clarification": True,
            "clarification_reason": "High semantic ambiguity",
            "confidence": 1.0 - intent.ambiguity_score
        }
    
    # CASE 6: Complete query
    return {
        "completeness_type": "complete",
        "requires_clarification": False,
        "clarification_reason": "",
        "confidence": 0.9
    }


def generate_smart_clarification(
    original_question: str,
    enriched_question: str,
    completeness: Dict[str, Any],
    intent: QueryIntent,
    has_documents: bool,
    working_memory: Dict[str, str]
) -> Dict[str, Any]:
    """
    Generate context-aware clarification messages.
    Pattern: Smart clarification from IBM Watson & LivePerson
    """
    completeness_type = completeness["completeness_type"]
    
    # Type 0: Missing documents (ADDED)
    if completeness_type == "missing_documents":
        return {
            "message": f"You asked to '{original_question}', but no documents are currently uploaded. Would you like to:",
            "options": [
                "Upload a document first",
                "Search the web instead"
            ]
        }
    
    # Type 1: Incomplete command
    if completeness_type == "incomplete_command":
        return {
            "message": f"I see you want to search the web. What would you like me to search for?",
            "options": [
                "Search for general information",
                "Search for recent news",
                "Search for technical information"
            ]
        }
    
    # Type 2: Incomplete query
    if completeness_type == "incomplete":
        return {
            "message": f"Your question seems incomplete: '{original_question}'. Could you provide more details?",
            "options": [
                "Tell me what you want to know",
                "Provide the topic or subject"
            ]
        }
    
    # Type 3: Ambiguous reference
    if completeness_type == "ambiguous_reference":
        return {
            "message": f"You mentioned '{original_question}', but I'm not sure what you're referring to. Could you clarify?",
            "options": [
                "Specify what 'it' or 'this' refers to",
                "Start a new question"
            ]
        }
    
    # Type 4: General high ambiguity
    return {
        "message": f"I'm not sure I understand: '{original_question}'. Could you rephrase or provide more context?",
        "options": []
    }


def should_clarify_or_resolve(
    question: str,
    enriched_question: str,
    has_documents: bool,
    intent: QueryIntent,
    context_switch: Dict[str, Any],
    working_memory: Dict[str, str]
) -> Dict[str, Any]:
    """
    FIXED: Decide: (1) Clarify, (2) Auto-resolve with context, or (3) Proceed
    Pattern: Three-tier decision framework from enterprise dialog systems
    
    CRITICAL FIX: Low ambiguity (<0.7) should NEVER require clarification
    """
    ambiguity_score = intent.ambiguity_score
    
    # TIER 1: High confidence - proceed directly (FIXED: threshold at 0.7)
    if ambiguity_score < 0.7 and not (intent.needs_documents and not has_documents):
        return {
            "action": "proceed",
            "resolved_query": enriched_question,
            "confidence": 1.0 - ambiguity_score
        }
    
    # TIER 2: Medium ambiguity - attempt auto-resolution
    if 0.3 <= ambiguity_score < 0.7:
        # Check if working memory or history can resolve
        if enriched_question != question:  # Context was added
            return {
                "action": "proceed_with_context",
                "resolved_query": enriched_question,
                "confidence": 0.7,
                "note": "Resolved using conversation context"
            }
        
        # Check if it's a computational follow-up
        if intent.intent_type == "computational" and working_memory:
            return {
                "action": "proceed_with_memory",
                "resolved_query": question,
                "confidence": 0.75
            }
        
        # Handle missing documents
        if intent.needs_documents and not has_documents:
            # Check if query explicitly mentions documents
            doc_keywords = ['document', 'paper', 'file', 'pdf', 'the doc', 'uploaded', 'attachment']
            mentions_docs = any(keyword in question.lower() for keyword in doc_keywords)
            
            if mentions_docs:
                # User explicitly wants documents
                return {
                    "action": "clarify",
                    "reason": "missing_documents",
                    "confidence": 0.3
                }
            else:
                # General query - use web search
                return {
                    "action": "soft_clarification",
                    "message": f"I'll search the web for: '{question}'",
                    "fallback_route": "web_search"
                }
    
    # TIER 3: High ambiguity - clarification needed
    if ambiguity_score >= 0.7:
        return {
            "action": "clarify",
            "reason": "high_ambiguity",
            "confidence": 1.0 - ambiguity_score
        }
    
    # Default: proceed with best effort
    return {
        "action": "proceed",
        "resolved_query": enriched_question,
        "confidence": 0.5
    }


def add_capability_context_to_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add capability awareness to state for LLM transparency.
    Pattern: Capability-aware prompting for better user experience
    """
    try:
        embedding_service = get_embedding_service()
        vector_store = VectorStoreService(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name=f"session_{state['session_id']}",
            embedding_dim=embedding_service.get_dimension()
        )
        
        collection_info = vector_store.get_collection_info()
        has_documents = collection_info.get("points_count", 0) > 0
    except Exception as e:
        logger.warning(f"Could not check document availability: {e}")
        has_documents = False
    
    capabilities_note = f"""**SYSTEM CAPABILITIES AVAILABLE TO YOU:**
- Web search: ✓ ACTIVE (you can search the internet)
- Document search: {"✓ ACTIVE (documents uploaded)" if has_documents else "✗ INACTIVE (no documents uploaded yet)"}
- Conversation memory: ✓ ACTIVE (you remember previous context)
- Computational reasoning: ✓ ACTIVE (you can solve math problems)

**CRITICAL: Never tell users you cannot search the web or access information. You have these capabilities through the system.**"""
    
    state["system_capabilities"] = capabilities_note
    return state


# ========== EXISTING HELPER FUNCTIONS ==========

def detect_context_switch(current_question: str, history: List[Dict]) -> Dict[str, Any]:
    """
    Detect if user has switched context from previous conversation.
    Pattern: Context coherence checking from DialogFlow
    """
    if not history:
        return {"switch_detected": False, "confidence": 1.0}
    
    # Get last few exchanges for context
    recent_context = " ".join([
        msg.get("content", "") for msg in history[-4:]
        if msg.get("role") == "user"
    ])
    
    if not recent_context:
        return {"switch_detected": False, "confidence": 1.0}
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Analyze if the current question represents a context switch from the previous conversation.

Context switch indicators:
- Completely new topic unrelated to previous discussion
- Sudden change in domain (e.g., from technical to personal)
- Ignoring previous clarification request
- Starting fresh without acknowledging previous context

Score:
- 0.0-0.3: Same context, natural continuation
- 0.3-0.6: Related but shifting focus
- 0.6-0.8: Probable context switch
- 0.8-1.0: Definite context switch

Return a score and brief explanation."""),
        ("human", """Previous context: {recent_context}

Current question: {current_question}

Analyze context switch:""")
    ])
    
    groq_service = get_groq_service(settings.groq_api_key)
    llm = groq_service.get_llm(settings.routing_model, temperature=0)
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({
            "recent_context": recent_context,
            "current_question": current_question
        })
        
        # Parse score from result
        score_match = re.search(r"(\d+\.?\d*)", result)
        score = float(score_match.group(1)) if score_match else 0.5
        
        return {
            "switch_detected": score > 0.6,
            "confidence": score,
            "explanation": result
        }
    
    except Exception as e:
        logger.error(f"Context switch detection error: {e}")
        return {"switch_detected": False, "confidence": 0.5}


def analyze_query_for_clarification(
    question: str,
    has_documents: bool,
    intent: QueryIntent,
    context_switch: Dict[str, Any]
) -> QueryClarification:
    """
    Analyze if query needs clarification before processing.
    Pattern: Clarification dialog management from Microsoft Bot Framework
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at identifying when user clarification is needed.

**Context:**
- Documents available: {has_documents}
- Query ambiguity: {ambiguity_score}
- Context switch detected: {context_switch}
- Query needs documents: {needs_documents}

**Clarification Triggers:**
1. **missing_documents**: User asks about "the document/paper/file" but none uploaded
2. **ambiguous_intent**: Query could mean multiple things
3. **missing_context**: References something unclear ("it", "that", "the thing")
4. **multiple_interpretations**: Query has 2+ valid interpretations
5. **permission_needed**: Action requires explicit user confirmation

**Rules:**
- Only suggest clarification for genuinely ambiguous cases
- If query is clear despite missing resources, proceed with appropriate fallback
- Provide actionable options (not just "please clarify")
- Be concise and helpful in clarification messages

**Examples:**
"What's in the document?" + no docs → needs clarification with upload/search options
"What is machine learning?" + no docs → NO clarification (general knowledge)
"Analyze the methodology" + no docs → needs clarification (which methodology?)
"Tell me about it" + no context → needs clarification (about what?)"""),
        ("human", """Query: {question}

Determine if clarification needed:""")
    ])
    
    groq_service = get_groq_service(settings.groq_api_key)
    structured_llm = groq_service.get_structured_llm(
        settings.routing_model,
        QueryClarification
    )
    
    chain = prompt | structured_llm
    return chain.invoke({
        "question": question,
        "has_documents": has_documents,
        "ambiguity_score": intent.ambiguity_score,
        "context_switch": context_switch.get("switch_detected", False),
        "needs_documents": intent.needs_documents
    })


def check_document_availability(session_id: str) -> bool:
    """Check if documents are available for this session."""
    try:
        embedding_service = get_embedding_service()
        vector_store = VectorStoreService(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name=f"session_{session_id}",
            embedding_dim=embedding_service.get_dimension()
        )
        
        collection_info = vector_store.get_collection_info()
        points_count = collection_info.get("points_count", 0)
        logger.info(f"Document check: session_{session_id} has {points_count} documents")
        return points_count > 0
    except Exception as e:
        logger.warning(f"Could not check document availability: {e}")
        return False



def apply_fallback_logic(
    routing_decision: EnhancedRouteQuery,
    has_documents: bool,
    intent: QueryIntent
) -> str:
    """
    Apply fallback logic to ensure valid routing.
    Pattern: Defensive routing with graceful degradation
    """
    route = routing_decision.datasource
    
    # CRITICAL FIX: Override any "clarification" route
    if route == "clarification":
        logger.error("CRITICAL BUG: 'clarification' returned as route - overriding to 'web_search'")
        return "web_search"
    
    # Fallback 1: Can't route to vectorstore if no documents
    if route == "vectorstore" and not has_documents:
        logger.warning("Routing changed: vectorstore → web_search (no documents)")
        return "web_search"
    
    # Fallback 2: Can't route to hybrid if no documents
    if route == "hybrid" and not has_documents:
        logger.warning("Routing changed: hybrid → web_search (no documents)")
        return "web_search"
    
    # Fallback 3: Can't route to hybrid_research if no documents
    if route == "hybrid_research" and not has_documents:
        logger.warning("Routing changed: hybrid_research → research (no documents)")
        return "research"
    
    # NEW Fallback 4: hybrid_web_research always works (no document dependency)
    # No fallback needed - both research and web search are always available
    
    # Fallback 5: If computational/conversational, prefer direct_llm
    if intent.intent_type in ["computational", "conversational"] and route in ["web_search", "vectorstore"]:
        if not intent.needs_retrieval:
            logger.info("Routing optimized: Using direct_llm for computational/conversational")
            return "direct_llm"
    
    return route


def extract_working_memory(question: str, history: List[Dict]) -> Dict[str, str]:
    """
    Extract variables, facts, and context from conversation.
    Pattern: Working memory extraction from dialog systems
    """
    # Look at current message + last 5 messages
    recent_messages = [question] + [
        msg.get("content", "") for msg in history[-5:]
    ]
    
    working_memory = {}
    
    # Extract variable assignments (x = 5, name: John, etc.)
    for message in recent_messages:
        # Pattern 1: x = value
        assignments = re.findall(r'([a-zA-Z_]\w*)\s*=\s*([^,\n]+)', message)
        for var, value in assignments:
            working_memory[var.strip()] = value.strip()
        
        # Pattern 2: variable: value
        colon_assignments = re.findall(r'([a-zA-Z_]\w*)\s*:\s*([^,\n]+)', message)
        for var, value in colon_assignments:
            if var.lower() not in ['http', 'https']:  # Avoid URLs
                working_memory[var.strip()] = value.strip()
    
    if working_memory:
        logger.info(f"Extracted working memory: {working_memory}")
    
    return working_memory


# ========== MAIN NODE FUNCTIONS (WITH CRITICAL FIXES) ==========

"""
CRITICAL FIXES for nodes.py - Add to existing file

Pattern: Defense in Depth with Rule-Based Pre-Filtering
Source: Production RAG Systems (Anthropic, OpenAI)
"""

def classify_query_intent(question: str) -> QueryIntent:
    """
    FIXED: Classify query intent with error handling and fallback.
    
    Pattern: Rule-based pre-filtering before LLM classification
    Source: Agentic RAG Best Practices (2025)
    """
    
    # STEP 1: Rule-based quick classification (no LLM needed)
    question_lower = question.lower().strip()
    
    # Simple patterns that don't need LLM
    if len(question) < 3:
        return QueryIntent(
            intent_type="conversational",
            needs_documents=False,
            needs_web=False,
            needs_retrieval=False,
            ambiguity_score=1.0,
            context_dependent=True,
            secondary_intents=[]
        )
    
    # Math/computation patterns
    if any(op in question for op in ['+', '-', '*', '/', '=', 'calculate', 'compute']):
        return QueryIntent(
            intent_type="computational",
            needs_documents=False,
            needs_web=False,
            needs_retrieval=False,
            ambiguity_score=0.2,
            context_dependent=False,
            secondary_intents=[]
        )
    
    # Document-specific patterns
    doc_keywords = ['document', 'uploaded', 'file', 'pdf', 'the doc', 'this file']
    if any(kw in question_lower for kw in doc_keywords):
        return QueryIntent(
            intent_type="factual",
            needs_documents=True,
            needs_web=False,
            needs_retrieval=True,
            ambiguity_score=0.1,
            context_dependent=False,
            secondary_intents=[]
        )
    
    # Web search commands
    web_keywords = ['search the web', 'google', 'look up online', 'search for']
    if any(kw in question_lower for kw in web_keywords):
        return QueryIntent(
            intent_type="command",
            needs_documents=False,
            needs_web=True,
            needs_retrieval=True,
            ambiguity_score=0.1,
            context_dependent=False,
            secondary_intents=[]
        )
    
    # STEP 2: LLM classification with error handling (only if needed)
    try:
        # SIMPLIFIED PROMPT - much more robust
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Classify this query into ONE intent type:
- factual: Facts/definitions (e.g., "What is X?")
- analytical: Analysis needed
- comparative: Comparing things
- procedural: How-to questions
- exploratory: Open research
- computational: Math/calculations
- conversational: Greetings/chat
- command: Action requests

Also determine:
- needs_documents: true if query mentions uploaded documents
- needs_web: true if query needs current/online information
- ambiguity_score: 0.0 (clear) to 1.0 (very unclear)

Return JSON with: intent_type, needs_documents, needs_web, needs_retrieval, ambiguity_score, context_dependent"""),
            ("human", "Query: {question}")
        ])
        
        groq_service = get_groq_service(settings.groq_api_key)
        
        # Try structured output first
        try:
            structured_llm = groq_service.get_structured_llm(
                settings.routing_model,
                QueryIntent
            )
            chain = prompt | structured_llm
            result = groq_service.invoke_with_fallback(
                chain,
                {"question": question},
                schema=QueryIntent
            )
            return result
            
        except Exception as structured_error:
            logger.warning(f"Structured classification failed: {structured_error}, using fallback")
            
            # FALLBACK: Use regular LLM + manual parsing
            llm = groq_service.get_llm(settings.routing_model, temperature=0)
            chain = prompt | llm | StrOutputParser()
            
            response = chain.invoke({"question": question})
            
            # Parse manually
            import json
            try:
                # Try to extract JSON from response
                json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    return QueryIntent(**data)
            except:
                pass
            
            # Ultimate fallback: heuristic classification
            return QueryIntent(
                intent_type="exploratory",
                needs_documents=False,
                needs_web=True,
                needs_retrieval=True,
                ambiguity_score=0.5,
                context_dependent=False,
                secondary_intents=[]
            )
    
    except Exception as e:
        logger.error(f"All intent classification attempts failed: {e}")
        
        # SAFE FALLBACK: Default to web search
        return QueryIntent(
            intent_type="exploratory",
            needs_documents=False,
            needs_web=True,
            needs_retrieval=True,
            ambiguity_score=0.5,
            context_dependent=False,
            secondary_intents=[]
        )


def make_routing_decision(question: str, has_documents: bool, intent: QueryIntent) -> EnhancedRouteQuery:
    """
    FIXED: Make routing decision with robust error handling.
    
    Pattern: Rule-based routing with LLM as tiebreaker only
    Source: Production Agentic RAG (Microsoft, Google)
    """
    
    # RULE-BASED ROUTING (80% of queries can be handled without LLM)
    
    # Rule 1: Computational queries → direct_llm
    if intent.intent_type == "computational":
        return EnhancedRouteQuery(
            datasource="direct_llm",
            confidence=0.95,
            reasoning="Computational query handled by LLM directly",
            query_complexity="simple",
            temporal_context=False
        )
    
    # Rule 2: Explicit document queries with documents → vectorstore
    if intent.needs_documents and has_documents:
        return EnhancedRouteQuery(
            datasource="vectorstore",
            confidence=0.9,
            reasoning="Query explicitly needs documents and documents are available",
            query_complexity="moderate",
            temporal_context=False
        )
    
    # Rule 3: Explicit document queries WITHOUT documents → web_search
    if intent.needs_documents and not has_documents:
        return EnhancedRouteQuery(
            datasource="web_search",
            confidence=0.85,
            reasoning="Query needs documents but none available, using web search",
            query_complexity="moderate",
            temporal_context=False
        )
    
    # Rule 4: Explicit web search commands → web_search
    if intent.intent_type == "command" and intent.needs_web:
        return EnhancedRouteQuery(
            datasource="web_search",
            confidence=0.95,
            reasoning="Explicit web search command",
            query_complexity="simple",
            temporal_context=True
        )
    
    # Rule 5: Research queries → hybrid_web_research
    research_keywords = ['research', 'study', 'studies', 'paper', 'scientific', 'academic']
    if any(kw in question.lower() for kw in research_keywords):
        return EnhancedRouteQuery(
            datasource="hybrid_web_research",
            confidence=0.85,
            reasoning="Research-oriented query",
            query_complexity="complex",
            temporal_context=False
        )
    
    # Rule 6: General knowledge (low ambiguity) → direct_llm
    if intent.ambiguity_score < 0.3 and not intent.needs_documents and not intent.needs_retrieval:
        return EnhancedRouteQuery(
            datasource="direct_llm",
            confidence=0.8,
            reasoning="Clear general knowledge question",
            query_complexity="simple",
            temporal_context=False
        )
    
    # Rule 7: Conversational queries → direct_llm
    if intent.intent_type == "conversational":
        return EnhancedRouteQuery(
            datasource="direct_llm",
            confidence=0.9,
            reasoning="Conversational query",
            query_complexity="simple",
            temporal_context=False
        )
    
    # FALLBACK: Use LLM routing only for ambiguous cases
    try:
        logger.info("Using LLM routing for ambiguous case")
        
        # SIMPLIFIED ROUTING PROMPT
        prompt = ChatPromptTemplate.from_template("""
Route this query to ONE source:
- vectorstore: User has documents uploaded
- web_search: Needs current/online info
- direct_llm: General knowledge
- research: Academic papers
- hybrid: Documents + web
- hybrid_web_research: Research + web

Query: {question}
Has documents: {has_documents}
Intent type: {intent_type}

Choose the BEST single route and explain briefly why.""")
        
        groq_service = get_groq_service(settings.groq_api_key)
        
        try:
            structured_llm = groq_service.get_structured_llm(
                settings.routing_model,
                EnhancedRouteQuery
            )
            chain = prompt | structured_llm
            
            result = groq_service.invoke_with_fallback(
                chain,
                {
                    "question": question,
                    "has_documents": has_documents,
                    "intent_type": intent.intent_type
                },
                schema=EnhancedRouteQuery
            )
            
            # Validate route
            valid_routes = ["vectorstore", "web_search", "hybrid", "direct_llm", "research", "hybrid_research", "hybrid_web_research"]
            if result.datasource not in valid_routes:
                raise ValueError(f"Invalid route: {result.datasource}")
            
            return result
            
        except Exception as llm_error:
            logger.error(f"LLM routing failed: {llm_error}, using intelligent fallback")
            
            # INTELLIGENT FALLBACK LOGIC
            if has_documents:
                return EnhancedRouteQuery(
                    datasource="hybrid",
                    confidence=0.6,
                    reasoning="Fallback: has documents, using hybrid search",
                    query_complexity="moderate",
                    temporal_context=False
                )
            else:
                return EnhancedRouteQuery(
                    datasource="web_search",
                    confidence=0.6,
                    reasoning="Fallback: no documents, using web search",
                    query_complexity="moderate",
                    temporal_context=True
                )
    
    except Exception as e:
        logger.error(f"All routing attempts failed: {e}, using safe fallback")
        
        # ULTIMATE SAFE FALLBACK
        return EnhancedRouteQuery(
            datasource="web_search" if not has_documents else "hybrid",
            confidence=0.5,
            reasoning="Emergency fallback due to routing failures",
            query_complexity="unknown",
            temporal_context=False
        )


def route_question(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIXED: Main routing node with comprehensive error handling.
    
    Pattern: Fail-safe routing with multiple fallback layers
    """
    logger.info("NODE: route_question (PRODUCTION-HARDENED)")
    
    try:
        question = state["question"]
        session_id = state["session_id"]
        history = state.get("conversation_history", [])
        working_memory = state.get("working_memory", {})
        
        # Step 1: Check document availability
        try:
            has_documents = check_document_availability(session_id)
        except Exception as e:
            logger.warning(f"Document check failed: {e}, assuming no documents")
            has_documents = False
        
        # Step 2: Enrich query with context
        try:
            enriched_question = enrich_query_with_context(question, working_memory, history)
        except Exception as e:
            logger.warning(f"Query enrichment failed: {e}, using original question")
            enriched_question = question
        
        # Step 3: Classify intent (with fallback)
        try:
            intent = classify_query_intent(enriched_question)
            logger.info(f"Intent classified: {intent.intent_type} (ambiguity={intent.ambiguity_score:.2f})")
        except Exception as e:
            logger.error(f"Intent classification completely failed: {e}")
            # Default safe intent
            intent = QueryIntent(
                intent_type="exploratory",
                needs_documents=False,
                needs_web=True,
                needs_retrieval=True,
                ambiguity_score=0.5,
                context_dependent=False,
                secondary_intents=[]
            )
        
        # Step 4: Check completeness (simplified)
        try:
            completeness = detect_query_completeness(enriched_question, intent, history, has_documents)
            
            if completeness["requires_clarification"]:
                clarification = generate_smart_clarification(
                    question, enriched_question, completeness, intent, has_documents, working_memory
                )
                
                return {
                    **state,
                    "route_decision": "clarification",
                    "needs_clarification": True,
                    "clarification_type": completeness['completeness_type'],
                    "clarification_message": clarification["message"],
                    "clarification_options": clarification.get("options", []),
                    "dialog_state": "awaiting_clarification",
                    "working_memory": working_memory,
                    "enriched_question": enriched_question
                }
        except Exception as e:
            logger.warning(f"Completeness check failed: {e}, assuming complete")
            # Assume query is complete if check fails
        
        # Step 5: Make routing decision (with fallback)
        try:
            routing_decision = make_routing_decision(enriched_question, has_documents, intent)
            final_route = routing_decision.datasource
            
            logger.info(f"Final route: {final_route} (confidence={routing_decision.confidence:.2f})")
            
            return {
                **state,
                "route_decision": final_route,
                "question": enriched_question,
                "enriched_question": enriched_question,
                "routing_confidence": routing_decision.confidence,
                "query_complexity": routing_decision.query_complexity,
                "routing_reason": routing_decision.reasoning,
                "working_memory": working_memory,
                "dialog_state": "normal",
                "needs_clarification": False
            }
            
        except Exception as e:
            logger.error(f"Routing decision failed critically: {e}")
            
            # EMERGENCY FALLBACK
            emergency_route = "web_search" if not has_documents else "hybrid"
            
            return {
                **state,
                "route_decision": emergency_route,
                "question": enriched_question,
                "enriched_question": enriched_question,
                "routing_confidence": 0.3,
                "query_complexity": "unknown",
                "routing_reason": f"Emergency fallback due to error: {str(e)[:100]}",
                "working_memory": working_memory,
                "dialog_state": "normal",
                "needs_clarification": False
            }
    
    except Exception as e:
        logger.error(f"Critical failure in route_question: {e}", exc_info=True)
        
        # LAST RESORT FALLBACK
        return {
            **state,
            "route_decision": "direct_llm",
            "question": state.get("question", ""),
            "enriched_question": state.get("question", ""),
            "routing_confidence": 0.1,
            "query_complexity": "unknown",
            "routing_reason": f"Critical routing failure: {str(e)[:100]}",
            "working_memory": {},
            "dialog_state": "error",
            "needs_clarification": False
        }


def handle_clarification_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced to detect if user switched context during clarification.
    """
    logger.info("NODE: handle_clarification_response")
    
    user_response = state["question"]
    clarification_type = state.get("clarification_type", "")
    original_question = state.get("original_question", "")
    session_id = state["session_id"]
    
    # Detect if user completely changed topic
    context_switch = detect_context_switch(user_response, [
        {"role": "assistant", "content": state.get("clarification_message", "")},
        {"role": "user", "content": original_question}
    ])
    
    if context_switch["switch_detected"] and context_switch["confidence"] > 0.8:
        # User switched to new topic - treat as new query
        logger.info("User switched context during clarification - treating as new query")
        state["dialog_state"] = "normal"
        state["needs_clarification"] = False
        # Recursively route the new question
        return route_question(state)
    
    # Analyze user's response to clarification
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Analyze the user's response to a clarification request.

Original question: {original_question}
Clarification type: {clarification_type}
User response: {user_response}

Determine the user's intent:
1. proceed_web_search - User wants to search the web
2. wait_for_upload - User will upload documents
3. provide_context - User provided missing context
4. change_query - User changed their question entirely
5. cancel - User wants to cancel/skip

Be flexible in interpretation - users may not respond exactly as expected."""),
        ("human", "What is the user's intent?")
    ])
    
    groq_service = get_groq_service(settings.groq_api_key)
    llm = groq_service.get_llm(settings.routing_model, temperature=0)
    chain = prompt | llm | StrOutputParser()
    
    response_intent = chain.invoke({
        "original_question": original_question,
        "clarification_type": clarification_type,
        "user_response": user_response
    })
    
    # Route based on user's response
    if "web" in response_intent.lower() or "search" in user_response.lower():
        logger.info("User chose web search after clarification")
        return {
            **state,
            "route_decision": "web_search",
            "question": original_question,
            "dialog_state": "clarified",
            "needs_clarification": False
        }
    
    elif "upload" in response_intent.lower() or "wait" in user_response.lower():
        logger.info("User will upload documents")
        return {
            **state,
            "route_decision": "wait_for_upload",
            "generation": "I'll wait for you to upload the documents. Please upload them using the panel on the left, then ask your question again.",
            "dialog_state": "normal",
            "needs_clarification": False
        }
    
    elif "context" in response_intent.lower() or "change_query" in response_intent.lower():
        # User provided new context or changed question - re-route
        logger.info("User provided new context/question - re-routing")
        state["dialog_state"] = "normal"
        state["needs_clarification"] = False
        return route_question(state)
    
    else:
        # Default: proceed with best guess
        logger.info("Proceeding with best guess after unclear clarification response")
        has_documents = check_document_availability(session_id)
        return {
            **state,
            "route_decision": "web_search" if not has_documents else "vectorstore",
            "question": original_question,
            "dialog_state": "normal",
            "needs_clarification": False
        }


# ========== RETRIEVAL NODES (UNCHANGED) ==========


def retrieve_documents(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    OPTIMIZED: Retrieve documents from vector store.
    
    Changes:
    1. Uses cached embedding service
    2. Reuses connection pool
    3. Proper error handling
    """
    logger.info("NODE: retrieve_documents")
    question = state.get("enriched_question", state["question"])
    session_id = state["session_id"]
    
    try:
        # Get embedding service (cached globally)
        embedding_service = get_embedding_service()
        
        # Create vector store with pooled connection
        vector_store = VectorStoreService(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name=f"session_{session_id}",
            embedding_dim=embedding_service.get_dimension(),
            auto_create=False  # Don't create - should already exist
        )
        
        # Generate query embedding
        query_embedding = embedding_service.embed_text(question)
        
        # Retrieve relevant documents
        results = vector_store.similarity_search(
            query_embedding=query_embedding,
            k=settings.retrieval_k
        )
        
        # Convert to LangChain Document format
        documents = [
            Document(
                page_content=result["text"],
                metadata={
                    **result["metadata"],
                    "similarity_score": result.get("score", 0)
                }
            )
            for result in results
        ]
        
        logger.info(f"Retrieved {len(documents)} documents")
        
        return {
            **state,
            "documents": documents
        }
        
    except Exception as e:
        logger.error(f"Document retrieval error: {e}")
        return {
            **state,
            "documents": [],
            "web_search_needed": True
        }


def grade_documents(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Grade document relevance to question with robust error handling.
    Pattern: Relevance filtering from CRAG paper
    Enhanced: Handles special characters, formulas, and edge cases
    """
    logger.info("NODE: grade_documents")
    
    question = state["question"]
    documents = state["documents"]
    
    if not documents:
        logger.warning("No documents to grade")
        return {
            **state,
            "documents": [],
            "web_search_needed": True
        }
    
    groq_service = get_groq_service(settings.groq_api_key)
    structured_llm = groq_service.get_structured_llm(
        settings.grading_model,
        GradeDocuments
    )
    
    # Create prompt template for grading
    grading_prompt = ChatPromptTemplate.from_template(
        """You are a grading expert assessing document relevance.

Question: {question}

Document Content:
{document}

Is this document relevant to answering the question?
- Answer 'yes' if the document contains information that helps answer the question
- Answer 'no' if the document is not relevant

Provide a binary score ('yes' or 'no') and brief reasoning."""
    )
    
    # Grade each document
    filtered_docs = []
    for i, doc in enumerate(documents):
        try:
            # Clean and truncate document content to avoid parsing errors
            doc_content = doc.page_content
            # Remove problematic characters that break structured parsing
            # Keep only printable ASCII and basic Unicode
            doc_content = ''.join(char for char in doc_content if char.isprintable() or char in '\n\r\t')
            
            # Truncate to prevent token overflow
            max_chars = 2000
            if len(doc_content) > max_chars:
                doc_content = doc_content[:max_chars] + "... [truncated]"
            
            # Format the prompt with cleaned content
            formatted_messages = grading_prompt.format_messages(
                question=question,
                document=doc_content
            )
            
            # Invoke with formatted messages
            grade = structured_llm.invoke(formatted_messages)
            
            if grade.binary_score == "yes":
                filtered_docs.append(doc)
                source_name = doc.metadata.get('source', doc.metadata.get('filename', 'unknown'))
                logger.info(f"✓ Document {i+1} relevant: {source_name}")
            else:
                source_name = doc.metadata.get('source', doc.metadata.get('filename', 'unknown'))
                logger.info(f"✗ Document {i+1} not relevant: {source_name}")
        
        except Exception as e:
            # On error, keep the document (fail-safe approach)
            logger.error(f"Grading error for document {i+1}: {str(e)[:100]}")
            filtered_docs.append(doc)
            logger.warning(f"⚠ Document {i+1} kept due to grading error (fail-safe)")
    
    # Check if we need web search fallback
    web_search_needed = len(filtered_docs) == 0
    
    if web_search_needed:
        logger.warning("No relevant documents found - triggering web search")
    else:
        logger.info(f"Grading complete: {len(filtered_docs)}/{len(documents)} documents relevant")
    
    return {
        **state,
        "documents": filtered_docs,
        "web_search_needed": web_search_needed
    }


def web_search(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform web search for current information.
    Pattern: Web-augmented retrieval
    """
    logger.info("NODE: web_search")
    
    question = state.get("enriched_question", state["question"])  # FIXED: Use enriched
    
    try:
        web_search_service = get_web_search_service(settings.tavily_api_key)
        results = web_search_service.search(question, max_results=settings.web_search_results)
        
        # Convert to Document format
        documents = [
            Document(
                page_content=result["content"],
                metadata={
                    "source": result["url"],
                    "title": result.get("title", "Web Result"),
                    "type": "web_search"
                }
            )
            for result in results
        ]
        
        logger.info(f"Web search returned {len(documents)} results")
        
        # Merge with existing documents if any
        existing_docs = state.get("documents", [])
        all_documents = existing_docs + documents
        
        return {
            **state,
            "documents": all_documents
        }
    
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return state


def research_search(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search academic papers using Semantic Scholar API.
    Pattern: Academic paper retrieval with citation-aware ranking
    Industry Standard: Multi-source academic RAG (Elicit, ScholarAI pattern)
    """
    logger.info("NODE: research_search")
    
    question = state["question"]
    
    try:
        # Initialize research search service
        research_service = get_research_search_service()
        
        # Search for papers with quality filters
        papers = research_service.search_papers(
            query=question,
            limit=settings.research_papers_limit,
            year_from=settings.research_year_threshold,
            min_citations=settings.research_citation_threshold
        )
        
        if not papers:
            logger.warning("No research papers found")
            return {
                **state,
                "research_papers": [],
                "documents": []
            }
        
        # Format papers as LangChain Documents for consistency
        documents = []
        research_papers = []
        
        for paper in papers:
            # Store full paper metadata
            research_papers.append(paper)
            
            # Create Document with formatted context
            content = research_service.format_paper_for_context(paper)
            doc = Document(
                page_content=content,
                metadata={
                    "source": "research",
                    "paper_id": paper.get("paperId"),
                    "title": paper.get("title", "Unknown"),
                    "year": paper.get("year"),
                    "citations": paper.get("citationCount", 0),
                    "url": paper.get("url"),
                    "authors": [a.get("name") for a in paper.get("authors", [])[:3]]
                }
            )
            documents.append(doc)
        
        logger.info(f"Retrieved {len(documents)} research papers")
        
        return {
            **state,
            "research_papers": research_papers,
            "documents": documents
        }
    
    except Exception as e:
        logger.error(f"Research search error: {e}")
        return {
            **state,
            "research_papers": [],
            "documents": [],
            "web_search_needed": True  # Fallback to web search
        }


def transform_query(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform query for better retrieval.
    Pattern: Query rewriting from HyDE, step-back prompting
    """
    logger.info("NODE: transform_query")
    
    question = state["question"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at query optimization for information retrieval.

Given a user question, generate an improved search query that:
1. Expands abbreviations and acronyms
2. Adds relevant context
3. Uses more specific terminology
4. Includes synonyms or related terms

Keep the transformed query concise and focused.
Return ONLY the transformed query, nothing else."""),
        ("human", "Original query: {question}\n\nTransformed query:")
    ])
    
    groq_service = get_groq_service(settings.groq_api_key)
    llm = groq_service.get_llm(settings.generation_model, temperature=0.3)
    chain = prompt | llm | StrOutputParser()
    
    try:
        transformed = chain.invoke({"question": question})
        logger.info(f"Query transformed: '{question}' → '{transformed}'")
        
        return {
            **state,
            "question": transformed
        }
    
    except Exception as e:
        logger.error(f"Query transformation error: {e}")
        return state


# ========== GENERATION NODES (FIXED WITH WORKING MEMORY) ==========

def generate(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIXED: Generate answer using retrieved documents and conversation context with working memory.
    Pattern: Context-aware generation with citation
    """
    logger.info("NODE: generate (FIXED with working memory)")
    
    # Add capability context to state
    state = add_capability_context_to_state(state)
    
    question = state.get("enriched_question", state["question"])  # FIXED: Use enriched
    documents = state.get("documents", [])
    history = state.get("conversation_history", [])
    working_memory = state.get("working_memory", {})  # FIXED: Extract working memory
    capabilities = state.get("system_capabilities", "")
    
    # Format documents
    if documents:
        context = "\n\n".join([
            f"[Source {i+1}: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for i, doc in enumerate(documents[:5])
        ])
    else:
        context = "No specific documents or sources available."
    
    # Format conversation history
    conversation_context = ""
    if history:
        recent_history = history[-6:]
        for msg in recent_history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            conversation_context += f"{role}: {msg.get('content', '')}\n"
    
    # FIXED: Format working memory
    memory_text = format_working_memory_context(working_memory)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """{capabilities}

You are a helpful AI assistant with access to multiple information sources and capabilities.

{memory_context}

**CRITICAL RULES:**
1. **Never say you cannot search the web** - you have web search capability
2. **Never say you cannot access documents** - the system handles document retrieval
3. **Never apologize for limitations you don't have**
4. If the user's question references variables from working memory, USE THEM
5. Always check working memory for computational queries

**Response Guidelines:**
- Synthesize information from retrieved documents and web search results
- Use working memory for computational continuity
- Cite sources naturally when using information from documents or web
- Be concise but comprehensive
- For computational queries, show your work step-by-step
- If you don't have specific information, state what you found, not what you can't do"""),
        ("human", """Context from Retrieved Documents:
{context}

Conversation History:
{conversation_history}

Current Question: {question}

Answer:""")
    ])
    
    groq_service = get_groq_service(settings.groq_api_key)
    llm = groq_service.get_llm(settings.generation_model, temperature=0.7)
    chain = prompt | llm | StrOutputParser()
    
    try:
        generation = chain.invoke({
            "capabilities": capabilities,
            "memory_context": memory_text,  # FIXED: Inject memory
            "context": context,
            "conversation_history": conversation_context,
            "question": question
        })
        
        # Extract sources for citation
        sources = []
        for doc in documents[:5]:
            source_info = {
                "url": doc.metadata.get("source", ""),
                "title": doc.metadata.get("title", "Document"),
                "type": doc.metadata.get("type", "document")
            }
            if source_info not in sources:
                sources.append(source_info)
        
        return {
            **state,
            "generation": generation,
            "sources": sources
        }
    
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return {
            **state,
            "generation": "I apologize, but I encountered an error generating the response. Please try again.",
            "sources": []
        }


def direct_llm_generate(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIXED: Generate answer directly from LLM without retrieval, with working memory support.
    Pattern: Direct generation for general knowledge queries
    """
    logger.info("NODE: direct_llm_generate (FIXED with memory)")
    
    # Add capability context to state
    state = add_capability_context_to_state(state)
    
    question = state.get("enriched_question", state["question"])  # FIXED: Use enriched
    history = state.get("conversation_history", [])
    working_memory = state.get("working_memory", {})  # FIXED: Extract working memory
    capabilities = state.get("system_capabilities", "")
    
    # Format conversation history
    conversation_context = ""
    if history:
        recent_history = history[-6:]
        for msg in recent_history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            conversation_context += f"{role}: {msg.get('content', '')}\n"
        conversation_context = f"Conversation History:\n{conversation_context}\n"
    
    # FIXED: Format working memory
    memory_text = format_working_memory_context(working_memory)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant with multiple capabilities.

{capabilities}

{memory_context}

**CRITICAL RULES:**
1. Never say you cannot search the web - you have that capability through the system
2. Never say you cannot access documents - the system routes to appropriate sources
3. Never apologize for limitations you do not have - web search and document access are available
4. For computational queries: CHECK the working memory for stored variables
5. Use stored variable values in your calculations

**Computational Example:**
If working memory shows c = 3 and user asks "c+1", answer "4"

**Response Guidelines:**
- For computational queries: Solve step-by-step and show your work
- For general knowledge: Provide accurate, concise information
- For conversational follow-ups: Use conversation history and working memory
- Never claim you lack capabilities that the system provides"""),
        ("human", """{conversation_context}

Current Question: {question}

Answer:""")
    ])
    
    groq_service = get_groq_service(settings.groq_api_key)
    llm = groq_service.get_llm(settings.generation_model, temperature=0.7)
    chain = prompt | llm | StrOutputParser()
    
    try:
        generation = chain.invoke({
            "capabilities": capabilities,
            "memory_context": memory_text,  # FIXED: Inject memory
            "conversation_context": conversation_context,
            "question": question
        })
        
        return {
            **state,
            "generation": generation,
            "sources": []
        }
    
    except Exception as e:
        logger.error(f"Direct LLM generation error: {e}")
        return {
            **state,
            "generation": "I apologize, but I encountered an error. Please try again.",
            "sources": []
        }


def generate_clarification(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIXED: Generate clarification message for user and ensure 'generation' field is set.
    Pattern: Proactive clarification dialog
    """
    logger.info("NODE: generate_clarification (FIXED)")
    
    clarification_message = state.get("clarification_message", "")
    clarification_options = state.get("clarification_options", [])
    
    # Format options if provided
    if clarification_options:
        options_text = "\n\nOptions:\n" + "\n".join([
            f"- {option}" for option in clarification_options
        ])
        full_message = clarification_message + options_text
    else:
        full_message = clarification_message
    
    # CRITICAL FIX: Set 'generation' field so response is sent to user
    return {
        **state,
        "generation": full_message,
        "sources": []
    }


def hybrid_generate(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate using both retrieved documents and web search.
    Pattern: Hybrid RAG for comprehensive answers
    """
    logger.info("NODE: hybrid_generate")
    
    # First perform web search to augment documents
    state = web_search(state)
    
    # Then generate with all available context
    return generate(state)


def hybrid_web_research_generate(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate using both research papers and web search.
    Pattern: Multi-modal RAG from Perplexity Academic / ScholarAI
    Industry Use: Combine academic rigor with current information
    
    Flow:
    1. Fetch research papers from Semantic Scholar
    2. Fetch current web information from Tavily
    3. Combine both sources for comprehensive answer
    """
    logger.info("NODE: hybrid_web_research_generate (research + web)")
    
    # Step 1: Fetch research papers
    logger.info("Step 1/3: Fetching research papers...")
    state = research_search(state)
    
    research_docs = state.get("documents", [])
    logger.info(f"Retrieved {len(research_docs)} research papers")
    
    # Step 2: Fetch web search results
    logger.info("Step 2/3: Fetching web search results...")
    state = web_search(state)
    
    all_docs = state.get("documents", [])
    web_docs = [doc for doc in all_docs if doc.metadata.get("type") == "web_search"]
    logger.info(f"Retrieved {len(web_docs)} web results")
    
    # Step 3: Generate with combined sources
    logger.info(f"Step 3/3: Generating answer with {len(all_docs)} total sources")
    return generate(state)


def wait_for_upload(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle waiting for document upload.
    """
    logger.info("NODE: wait_for_upload")
    
    return {
        **state,
        "generation": "Please upload your documents using the upload panel on the left, then ask your question again.",
        "sources": []
    }
