from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Optional, AsyncGenerator
import uuid
import json
from io import BytesIO
from functools import lru_cache


from .config import settings
from .models.schemas import ChatRequest, ChatResponse, UploadResponse, HealthResponse, Source
from .graph.workflow import get_workflow
from .services.embeddings import get_embedding_service
from .services.vector_store import VectorStoreService
from .services.summarizer import ConversationSummarizer
from .services.llm_service import get_groq_service
from .services.research_search import get_research_search_service  # NEW: Add this import
from .utils.document_processor import get_document_processor
from .utils.logger import setup_logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .services.literature_review import get_literature_review_service
from .services.table_extractor import get_table_extractor
from .services.math_handler import get_math_handler
# ADD these lines after existing imports
from .services.fast_contextual_embedder import get_fast_contextual_embedder
from .services.query_rewriter import get_query_rewriter
from .services.reranker import get_reranker
from .services.hybrid_search import get_hybrid_search



logger = setup_logger(__name__)

app = FastAPI(
    title="Agentic RAG System with Research",  # UPDATED
    description="Adaptive Corrective RAG with LangGraph + Academic Paper Search",  # UPDATED
    version="2.0.0"  # UPDATED
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services
workflow = None
summarizer = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup."""
    global workflow, summarizer
    logger.info("Starting up Agentic RAG System with Research...")
    
    # Initialize workflow
    workflow = get_workflow()
    logger.info("✓ LangGraph workflow compiled")
    
    # Initialize embedding model
    embedding_service = get_embedding_service(settings.embedding_model)
    logger.info("✓ Embedding model loaded")
    
    # Initialize summarizer
    groq_service = get_groq_service(settings.groq_api_key)
    summarizer = ConversationSummarizer(groq_service, settings.routing_model)
    logger.info("✓ Conversation summarizer ready")
    
    logger.info("Application startup complete!")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Agentic RAG System API with Research",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    logger.info("Health check requested")
    
    health_status = {
        "status": "healthy",
        "groq_connected": False,
        "qdrant_connected": False,
        "tavily_connected": False,
        "semantic_scholar_connected": False,  # NEW
        "embedding_model_loaded": False
    }
    
    # Check Groq
    try:
        groq_service = get_groq_service(settings.groq_api_key)
        health_status["groq_connected"] = True
    except Exception as e:
        logger.error(f"Groq health check failed: {e}")
    
    # Check Qdrant
    try:
        test_store = VectorStoreService(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name="health_check",
            embedding_dim=384
        )
        health_status["qdrant_connected"] = True
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
    
    # Check Tavily
    health_status["tavily_connected"] = bool(settings.tavily_api_key)
    
    # NEW: Check Semantic Scholar
    try:
        research_service = get_research_search_service()
        health_status["semantic_scholar_connected"] = research_service.check_connection()
    except Exception as e:
        logger.error(f"Semantic Scholar health check failed: {e}")
        health_status["semantic_scholar_connected"] = False
    
    # Check embeddings
    try:
        embedding_service = get_embedding_service(settings.embedding_model)
        health_status["embedding_model_loaded"] = True
    except Exception as e:
        logger.error(f"Embedding model health check failed: {e}")
    
    all_healthy = all([
        health_status["groq_connected"],
        health_status["qdrant_connected"],
        health_status["tavily_connected"],
        health_status["embedding_model_loaded"]
        # Note: Semantic Scholar is optional
    ])
    
    health_status["status"] = "healthy" if all_healthy else "degraded"
    return health_status


@app.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Query(None)
):
    """Upload and process documents with contextual embedding."""
    logger.info(f"Document upload requested: {len(files)} files, session_id={session_id}")
    
    # Validate session_id is provided
    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="session_id is required. Please provide a valid session ID."
        )
    
    # Validate file sizes
    total_size = 0
    file_data = []
    for file in files:
        content = await file.read()
        size = len(content)
        total_size += size
        
        if total_size > settings.max_upload_size:
            raise HTTPException(
                status_code=413,
                detail=f"Total upload size exceeds 15MB limit"
            )
        
        file_data.append({
            "filename": file.filename,
            "content": content
        })
    
    logger.info(f"Total upload size: {total_size / (1024*1024):.2f} MB")
    
    # Process documents
    doc_processor = get_document_processor()
    
    all_chunks = []
    all_full_texts = []
    
    for file_info in file_data:
        try:
            # Process document and get chunks + full text
            result = doc_processor.process_document(
                BytesIO(file_info["content"]),
                file_info["filename"],
                session_id
            )
            
            all_chunks.extend(result["chunks"])
            all_full_texts.append({
                "filename": file_info["filename"],
                "full_text": result["full_text"]
            })
            
        except Exception as e:
            logger.error(f"Failed to process {file_info['filename']}: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process {file_info['filename']}: {str(e)}"
            )
    
    logger.info(f"Processed {len(all_chunks)} chunks from {len(files)} files")
    
    # Apply contextual embedding if enabled
    if settings.enable_contextual_embedding:
        logger.info("Applying contextual embedding to chunks...")
        from .services.contextual_embedder import get_contextual_embedder
        
        contextual_embedder = get_fast_contextual_embedder()
        
        # Process each file's chunks with its full text
        contextualized_chunks = []
        chunk_idx = 0
        
        for file_info in all_full_texts:
            filename = file_info["filename"]
            full_text = file_info["full_text"]
            
            # Find chunks belonging to this file
            file_chunks = [
                chunk for chunk in all_chunks 
                if chunk["metadata"]["filename"] == filename
            ]
            
            # Add context to these chunks
            file_contextualized = contextual_embedder.generate_contexts_batch(
                file_chunks,
                full_text,
                filename
            )
            
            contextualized_chunks.extend(file_contextualized)
        
        all_chunks = contextualized_chunks
        logger.info("✅ Contextual embedding applied")
    
    # Generate embeddings
    embedding_service = get_embedding_service(settings.embedding_model)
    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = embedding_service.embed_documents(texts)
    
    logger.info(f"Generated {len(embeddings)} embeddings")
    
    # Store in vector database with session-specific collection
    vectorstore = VectorStoreService(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=f"session_{session_id}",
        embedding_dim=embedding_service.get_dimension()
    )
    vectorstore.create_payload_indexes()
    
    metadatas = [chunk["metadata"] for chunk in all_chunks]
    vectorstore.add_documents(texts, embeddings, metadatas)
    
    logger.info(f"✅ Documents stored in session_{session_id}")
    
    return UploadResponse(
        message="Documents processed successfully with contextual embedding" if settings.enable_contextual_embedding else "Documents processed successfully",
        files_processed=len(files),
        chunks_created=len(all_chunks),
        session_id=session_id
    )



@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint with progress updates."""
    logger.info(f"Streaming chat request received: {request.message[:50]}...")
    
    # Validate session_id
    if not request.session_id:
        raise HTTPException(
            status_code=400,
            detail="session_id is required in chat request"
        )
    
    session_id = request.session_id
    logger.info(f"Using session_id: {session_id}, collection: session_{session_id}")
    
    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate Server-Sent Events for streaming response."""
        try:
            # Process conversation history
            history = request.conversation_history
            if summarizer.should_summarize(
                [{"role": msg.role, "content": msg.content} for msg in history],
                settings.max_messages_before_summary
            ):
                logger.info("Compressing conversation history")
                history_dicts = [{"role": msg.role, "content": msg.content} for msg in history]
                history_dicts = summarizer.compress_history(
                    history_dicts,
                    settings.recent_messages_to_keep
                )
            else:
                history_dicts = [{"role": msg.role, "content": msg.content} for msg in history]

            # Send progress: Starting
            yield f"event: progress\ndata: {json.dumps({'status': 'started', 'message': 'Processing your question...'})}\n\n"

            # Stage 1: Query routing
            yield f"event: progress\ndata: {json.dumps({'status': 'routing', 'message': 'Analyzing query intent...'})}\n\n"

            # Check document availability
            has_documents = check_document_availability(session_id)

            # Prepare initial state
            initial_state = {
                "question": request.message,
                "documents": [],
                "research_papers": [],  # NEW
                "generation": "",
                "web_search_needed": False,
                "research_needed": False,  # NEW
                "route_decision": "",
                "conversation_history": history_dicts,
                "session_id": session_id,
                "working_memory": {}
            }

            # Execute workflow with streaming
            current_node = None
            for step in workflow.stream(initial_state):
                # Extract current node from step
                node_name = list(step.keys())[0] if step else "unknown"

                # Send progress updates based on node
                if node_name == "route_question":
                    route_decision = step[node_name].get("route_decision", "")
                    if route_decision == "vectorstore":
                        if has_documents:
                            yield f"event: progress\ndata: {json.dumps({'status': 'retrieving', 'message': 'Searching your uploaded documents...'})}\n\n"
                        else:
                            yield f"event: progress\ndata: {json.dumps({'status': 'fallback', 'message': 'No documents found. Searching the web instead...'})}\n\n"
                    elif route_decision == "web_search":
                        yield f"event: progress\ndata: {json.dumps({'status': 'web_search', 'message': 'Searching the web for current information...'})}\n\n"
                    elif route_decision == "hybrid":
                        yield f"event: progress\ndata: {json.dumps({'status': 'hybrid', 'message': 'Searching documents and web...'})}\n\n"
                    elif route_decision == "research":  # NEW
                        yield f"event: progress\ndata: {json.dumps({'status': 'research', 'message': 'Searching academic papers...'})}\n\n"
                    elif route_decision == "hybrid_research":  # NEW
                        yield f"event: progress\ndata: {json.dumps({'status': 'hybrid_research', 'message': 'Searching documents and research papers...'})}\n\n"
                    elif route_decision == "hybrid_web_research":  # NEW
                        yield f"event: progress\ndata: {json.dumps({'status': 'hybrid_web_research', 'message': 'Searching research papers and web...'})}\n\n"
                elif node_name == "retrieve_documents":
                    yield f"event: progress\ndata: {json.dumps({'status': 'retrieving', 'message': 'Retrieving relevant document chunks...'})}\n\n"
                elif node_name == "grade_documents":
                    yield f"event: progress\ndata: {json.dumps({'status': 'grading', 'message': 'Evaluating document relevance...'})}\n\n"
                elif node_name == "transform_query":
                    yield f"event: progress\ndata: {json.dumps({'status': 'transforming', 'message': 'Optimizing search query...'})}\n\n"
                elif node_name == "web_search":
                    yield f"event: progress\ndata: {json.dumps({'status': 'web_search', 'message': 'Fetching web results...'})}\n\n"
                elif node_name == "research_search":  # NEW
                    yield f"event: progress\ndata: {json.dumps({'status': 'research', 'message': 'Fetching academic papers...'})}\n\n"
                elif node_name == "hybrid_web_research_generate":  # NEW
                    yield f"event: progress\ndata: {json.dumps({'status': 'hybrid_web_research', 'message': 'Combining research papers and web results...'})}\n\n"
                elif node_name == "generate":
                    yield f"event: progress\ndata: {json.dumps({'status': 'generating', 'message': 'Generating response...'})}\n\n"

                current_node = step

            # Get final state
            final_state = current_node[list(current_node.keys())[0]] if current_node else initial_state
            answer = final_state.get("generation", "I apologize, but I couldn't generate an answer.")
            documents = final_state.get("documents", [])
            route = final_state.get("route_decision", "unknown")

            # Add contextual message if no documents found but web search was used
            if route == "vectorstore" and not has_documents:
                answer = f"**Note:** No documents were found in your session, so I searched the web instead.\n\n{answer}"
            elif route in ["vectorstore", "hybrid", "hybrid_research"] and not documents:
                answer = f"**Note:** I couldn't find relevant information in your uploaded documents.\n\n{answer}"

            # FIXED: Build sources (including research papers) - Handle both dict and Document objects
            sources = []
            for doc in documents:
                # FIXED: Handle both dict and Document objects
                if isinstance(doc, dict):
                    # Dict format (from hybrid_web_research_generate)
                    metadata = doc.get("metadata", {})
                    page_content = doc.get("page_content", "")
                    source_type = metadata.get("source", "vectorstore")
                else:
                    # Document object format (from other nodes)
                    metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                    page_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    source_type = metadata.get("source", "vectorstore")
                
                if source_type == "research" or source_type == "semantic_scholar":  # Handle research papers
                    sources.append({
                        "content": page_content[:200] + "...",
                        "title": metadata.get("title", "Research Paper"),
                        "url": metadata.get("url"),
                        "score": None,
                        "type": "research",
                        "authors": metadata.get("authors", []),
                        "year": metadata.get("year"),
                        "citation_count": metadata.get("citations"),
                        "venue": metadata.get("venue")
                    })
                else:
                    sources.append({
                        "content": page_content[:200] + "...",
                        "title": metadata.get("title", metadata.get("source", "Document")),
                        "url": metadata.get("source") if source_type == "web_search" else metadata.get("url"),
                        "score": metadata.get("score"),
                        "type": source_type
                    })

            # Stream the answer token by token
            words = answer.split()
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield f"event: token\ndata: {json.dumps({'token': chunk})}\n\n"
                # Small delay for better UX
                import asyncio
                await asyncio.sleep(0.02)

            # Send completion event with metadata
            completion_data = {
                "status": "completed",
                "sources": sources,
                "route_taken": route,
                "session_id": session_id
            }
            yield f"event: complete\ndata: {json.dumps(completion_data)}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            error_data = {
                "status": "error",
                "message": str(e)
            }
            yield f"event: error\ndata: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def execute_workflow_with_retry(workflow, initial_state):
    """
    Execute workflow with retry logic for transient failures.
    
    Pattern: Exponential backoff retry wrapper
    - 1st attempt: immediate
    - 2nd attempt: after 1-2 seconds  
    - 3rd attempt: after 2-5 seconds (max)
    
    This handles transient LLM API failures, network issues, etc.
    """
    try:
        return workflow.invoke(initial_state)
    except Exception as e:
        logger.error(f"Workflow execution error on retry: {e}")
        raise


# Helper function for document availability check

@lru_cache(maxsize=256)  # Cache results for same session_id
def _get_embedding_dimension() -> int:
    """Cache embedding dimension to avoid repeated calls."""
    return get_embedding_service(settings.embedding_model).get_dimension()


def check_document_availability(session_id: str) -> bool:
    """
    OPTIMIZED: Check if session has uploaded documents.
    
    Changes:
    1. Uses cached embedding dimension
    2. Reuses connection pool via VectorStoreService
    3. auto_create=False to avoid unnecessary collection creation
    """
    try:
        vector_store = VectorStoreService(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name=f"session_{session_id}",
            embedding_dim=_get_embedding_dimension(),
            auto_create=False  # Don't create collection - just check
        )
        
        collection_info = vector_store.get_collection_info()
        points_count = collection_info.get("points_count", 0)
        
        return points_count > 0
        
    except Exception as e:
        logger.warning(f"Could not check document availability: {e}")
        return False




# Keep original non-streaming endpoint for backward compatibility
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint (non-streaming)."""
    logger.info(f"Chat request received: {request.message[:50]}...")
    
    # Validate session_id
    if not request.session_id:
        raise HTTPException(
            status_code=400,
            detail="session_id is required in chat request"
        )
    
    session_id = request.session_id
    logger.info(f"Using session_id: {session_id}, collection: session_{session_id}")
    
    # Check document availability
    has_documents = check_document_availability(session_id)
    
    # Process conversation history
    history = request.conversation_history
    if summarizer.should_summarize(
        [{"role": msg.role, "content": msg.content} for msg in history],
        settings.max_messages_before_summary
    ):
        logger.info("Compressing conversation history")
        history_dicts = [{"role": msg.role, "content": msg.content} for msg in history]
        history_dicts = summarizer.compress_history(
            history_dicts,
            settings.recent_messages_to_keep
        )
    else:
        history_dicts = [{"role": msg.role, "content": msg.content} for msg in history]
    
    # Prepare initial state
    initial_state = {
        "question": request.message,
        "documents": [],
        "research_papers": [],  # NEW
        "generation": "",
        "web_search_needed": False,
        "research_needed": False,  # NEW
        "route_decision": "",
        "conversation_history": history_dicts,
        "session_id": session_id,
        "working_memory": {}
    }
    
    try:
        logger.info("Executing LangGraph workflow with research support")
        final_state = execute_workflow_with_retry(workflow, initial_state)
        
        answer = final_state.get("generation", "I apologize, but I couldn't generate an answer.")
        documents = final_state.get("documents", [])
        route = final_state.get("route_decision", "unknown")
        
        # Add contextual message if no documents found
        if route == "vectorstore" and not has_documents:
            answer = f"**Note:** No documents were found in your session, so I searched the web instead.\n\n{answer}"
        elif route in ["vectorstore", "hybrid", "hybrid_research"] and not documents:
            answer = f"**Note:** I couldn't find relevant information in your uploaded documents.\n\n{answer}"
        
        # FIXED: Build sources (including research papers) - Handle both dict and Document objects
        sources = []
        for doc in documents:
            # FIXED: Handle both dict and Document objects
            if isinstance(doc, dict):
                # Dict format (from hybrid_web_research_generate)
                metadata = doc.get("metadata", {})
                page_content = doc.get("page_content", "")
                source_type = metadata.get("source", "vectorstore")
            else:
                # Document object format (from other nodes)
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                page_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                source_type = metadata.get("source", "vectorstore")
            
            if source_type == "research" or source_type == "semantic_scholar":  # Handle research papers
                sources.append(Source(
                    content=page_content[:200] + "...",
                    title=metadata.get("title", "Research Paper"),
                    url=metadata.get("url"),
                    score=None,
                    type="research",
                    authors=metadata.get("authors", []),
                    year=metadata.get("year"),
                    citation_count=metadata.get("citations"),
                    venue=metadata.get("venue"),
                    paper_id=metadata.get("paper_id")
                ))
            else:
                sources.append(Source(
                    content=page_content[:200] + "...",
                    title=metadata.get("title", metadata.get("source", "Document")),
                    url=metadata.get("source") if source_type == "web_search" else metadata.get("url"),
                    score=metadata.get("score"),
                    type=source_type
                ))
        
        logger.info(f"Chat response generated: {len(answer)} chars, {len(sources)} sources")
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            session_id=session_id,
            route_taken=route
        )
    
    except Exception as e:
        logger.error(f"Chat execution error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat request: {str(e)}"
        )

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete session data."""
    logger.info(f"Session deletion requested: {session_id}")
    
    try:
        vectorstore = VectorStoreService(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name=f"session_{session_id}",
            embedding_dim=384
        )
        vectorstore.delete_collection()
        return {"message": f"Session {session_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Session deletion error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete session: {str(e)}"
        )

@app.post("/research/literature-review")
async def literature_review(
    research_question: str = Query(..., description="Research question to investigate"),
    max_papers: int = Query(10, ge=1, le=20, description="Maximum papers to analyze"),
    min_year: int = Query(2020, ge=1900, description="Minimum publication year"),
    extract_structured: bool = Query(True, description="Extract key findings, methods, limitations")
):
    """
    Conduct automated literature review (Elicit-style).
    
    NEW FEATURE: Multi-paper synthesis with citation export
    """
    logger.info(f"Literature review requested: {research_question}")
    
    try:
        # Initialize services
        groq_service = get_groq_service(settings.groq_api_key)
        research_service = get_research_search_service(
            api_key=settings.semantic_scholar_api_key,
            llm_service=groq_service
        )
        lit_review_service = get_literature_review_service(groq_service, research_service)
        
        # Conduct review
        result = lit_review_service.conduct_literature_review(
            research_question=research_question,
            max_papers=max_papers,
            min_year=min_year,
            extract_structured_data=extract_structured
        )
        
        logger.info(f"✅ Literature review complete: {len(result['papers'])} papers")
        
        return result
    
    except Exception as e:
        logger.error(f"Literature review error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/research/export-citations")
async def export_citations(
    paper_ids: List[str] = Query(..., description="Paper IDs to export"),
    format: str = Query("bibtex", regex="^(bibtex|ris)$", description="Export format")
):
    """
    Export citations in BibTeX or RIS format.
    
    NEW FEATURE: Citation management for research papers
    """
    logger.info(f"Citation export requested: {len(paper_ids)} papers, format={format}")
    
    try:
        # Fetch papers
        groq_service = get_groq_service(settings.groq_api_key)
        research_service = get_research_search_service(
            api_key=settings.semantic_scholar_api_key,
            llm_service=groq_service
        )
        lit_review_service = get_literature_review_service(groq_service, research_service)
        
        # Get papers (simplified - you'd fetch from cache or API)
        # For demo, return empty result
        citations = lit_review_service._generate_citation_exports([])
        
        return {
            "format": format,
            "citations": citations.get(format, ""),
            "count": citations.get("count", 0)
        }
    
    except Exception as e:
        logger.error(f"Citation export error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/research/extract-tables")
async def extract_tables(
    file: UploadFile = File(..., description="PDF file"),
    pages: str = Query("all", description="Pages to extract (e.g., 'all', '1-3', '1,3,5')"),
    output_format: str = Query("csv", regex="^(csv|excel|markdown)$")
):
    """
    Extract tables from PDF.
    
    NEW FEATURE: Table extraction with multiple export formats
    """
    logger.info(f"Table extraction requested: {file.filename}, pages={pages}")
    
    try:
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Extract tables
        table_extractor = get_table_extractor()
        tables = table_extractor.extract_tables_from_pdf(tmp_path, pages=pages)
        
        if not tables:
            return {
                "message": "No tables found in PDF",
                "tables": [],
                "summary": {}
            }
        
        # Export tables
        if output_format == "markdown":
            markdown = table_extractor.tables_to_markdown(tables)
            result = {
                "format": "markdown",
                "content": markdown,
                "tables": len(tables)
            }
        else:
            exported = table_extractor.export_tables(tables, output_format)
            result = {
                "format": output_format,
                "files": exported,
                "summary": table_extractor.get_table_summary(tables)
            }
        
        # Clean up temp file
        import os
        os.unlink(tmp_path)
        
        logger.info(f"✅ Extracted {len(tables)} tables")
        
        return result
    
    except Exception as e:
        logger.error(f"Table extraction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/research/extract-math")
async def extract_math(
    text: str = Query(..., description="Text containing mathematical formulas"),
    verify_latex: bool = Query(True, description="Verify LaTeX syntax"),
    fix_errors: bool = Query(False, description="Attempt to fix broken LaTeX with LLM")
):
    """
    Extract and normalize mathematical formulas from text.
    
    NEW FEATURE: LaTeX formula extraction and verification
    """
    logger.info(f"Math extraction requested: {len(text)} chars")
    
    try:
        groq_service = get_groq_service(settings.groq_api_key)
        math_handler = get_math_handler(groq_service)
        
        # Extract formulas
        formulas = math_handler.extract_math_from_text(text)
        
        # Verify and fix if requested
        for formula in formulas:
            if verify_latex:
                is_valid, error = math_handler.verify_latex_syntax(formula["latex"])
                formula["valid"] = is_valid
                formula["error"] = error
                
                if fix_errors and not is_valid:
                    fixed = math_handler.fix_latex_with_llm(formula["latex"])
                    if fixed:
                        formula["fixed_latex"] = fixed
            
            # Normalize for rendering
            formula["render_latex"] = math_handler.normalize_for_rendering(
                formula.get("fixed_latex") or formula["latex"],
                mode=formula["type"]
            )
        
        logger.info(f"✅ Extracted {len(formulas)} formulas")
        
        return {
            "formulas": formulas,
            "total_count": len(formulas),
            "inline_count": sum(1 for f in formulas if f["type"] == "inline"),
            "block_count": sum(1 for f in formulas if f["type"] == "block")
        }
    
    except Exception as e:
        logger.error(f"Math extraction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
