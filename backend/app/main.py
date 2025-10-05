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
    """Upload and process documents."""
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
    doc_processor = get_document_processor(
        settings.chunk_size,
        settings.chunk_overlap
    )
    
    all_chunks = []
    for file_info in file_data:
        try:
            chunks = doc_processor.process_file(
                BytesIO(file_info["content"]),
                file_info["filename"]
            )
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Failed to process {file_info['filename']}: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process {file_info['filename']}: {str(e)}"
            )
    
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
    
    metadatas = [chunk["metadata"] for chunk in all_chunks]
    vectorstore.add_documents(texts, embeddings, metadatas)
    
    logger.info(f"Documents stored in session_{session_id}")
    
    return UploadResponse(
        message="Documents processed successfully",
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
                    elif route_decision == "websearch":
                        yield f"event: progress\ndata: {json.dumps({'status': 'web_search', 'message': 'Searching the web for current information...'})}\n\n"
                    elif route_decision == "hybrid":
                        yield f"event: progress\ndata: {json.dumps({'status': 'hybrid', 'message': 'Searching documents and web...'})}\n\n"
                    elif route_decision == "research":  # NEW
                        yield f"event: progress\ndata: {json.dumps({'status': 'research', 'message': 'Searching academic papers...'})}\n\n"
                    elif route_decision == "hybrid_research":  # NEW
                        yield f"event: progress\ndata: {json.dumps({'status': 'hybrid_research', 'message': 'Searching documents and research papers...'})}\n\n"
                
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
            
            # NEW: Build sources (including research papers)
            sources = []
            for doc in documents:
                source_type = doc.metadata.get("source", "vectorstore")
                
                if source_type == "research":  # NEW: Handle research papers
                    sources.append({
                        "content": doc.page_content[:200] + "...",
                        "title": doc.metadata.get("title", "Research Paper"),
                        "url": doc.metadata.get("url"),
                        "score": None,
                        "type": "research",
                        "authors": doc.metadata.get("authors", []),
                        "year": doc.metadata.get("year"),
                        "citation_count": doc.metadata.get("citations"),
                        "venue": doc.metadata.get("venue")
                    })
                else:
                    sources.append({
                        "content": doc.page_content[:200] + "...",
                        "title": doc.metadata.get("title", doc.metadata.get("source", "Document")),
                        "url": doc.metadata.get("source") if source_type == "web_search" else None,
                        "score": doc.metadata.get("score"),
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
    logger.info(f"Using session_id: {session_id}, collection: session_{session_id}...")
    
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
        
        # NEW: Build sources (including research papers)
        sources = []
        for doc in documents:
            source_type = doc.metadata.get("source", "vectorstore")
            
            if source_type == "research":  # NEW: Handle research papers
                sources.append(Source(
                    content=doc.page_content[:200] + "...",
                    title=doc.metadata.get("title", "Research Paper"),
                    url=doc.metadata.get("url"),
                    score=None,
                    type="research",
                    authors=doc.metadata.get("authors", []),
                    year=doc.metadata.get("year"),
                    citation_count=doc.metadata.get("citations"),
                    venue=doc.metadata.get("venue"),
                    paper_id=doc.metadata.get("paper_id")
                ))
            else:
                sources.append(Source(
                    content=doc.page_content[:200] + "...",
                    title=doc.metadata.get("title", doc.metadata.get("source", "Document")),
                    url=doc.metadata.get("source") if source_type == "web_search" else None,
                    score=doc.metadata.get("score"),
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
