"""
Configuration Management

Source: Pydantic Settings pattern - Industry standard for environment management
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Uses Pydantic for validation and type safety.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # ===== API Keys =====
    groq_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    tavily_api_key: str
    semantic_scholar_api_key: Optional[str] = None  # Optional for Semantic Scholar
    
    # ===== Model Configuration =====
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    routing_model: str = "llama-3.1-8b-instant"
    grading_model: str = "gemma2-9b-it"
    generation_model: str = "llama-3.3-70b-versatile"
    
    # ===== Vector Store Settings =====
    qdrant_collection_name: str = "rag_documents"
    
    # ===== Document Processing =====
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_upload_size: int = 15728640  # 15MB in bytes
    
    # ===== Conversation Management =====
    max_messages_before_summary: int = 10
    recent_messages_to_keep: int = 4
    
    # ===== Retrieval Configuration =====
    retrieval_k: int = 4  # Number of documents to retrieve
    web_search_results: int = 3
    
    # ===== Research Configuration =====
    research_papers_limit: int = 5  # Max papers to retrieve from Semantic Scholar
    research_citation_threshold: int = 10  # Minimum citations for paper relevance
    research_year_threshold: int = 2018  # Papers after this year preferred
    
    # ===== NEW: Fallback Configuration =====
    # Web search fallback chain: Tavily → DuckDuckGo → Brave
    enable_web_search_fallback: bool = True
    use_duckduckgo_fallback: bool = True  # Free, unlimited
    
    # Research fallback chain: Semantic Scholar → arXiv → CORE
    enable_research_fallback: bool = True
    use_arxiv_fallback: bool = True  # Free, unlimited
    use_core_fallback: bool = True  # Free, 1000 req/day
    
    # ===== Retry and Error Handling =====
    llm_max_retries: int = 3
    llm_retry_base_delay: float = 1.0
    llm_retry_max_delay: float = 10.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    
    # Fallback model (faster, more reliable for errors)
    fallback_model: str = "llama-3.1-8b-instant"
    
    # Timeout settings
    llm_timeout: int = 30  # seconds
    api_timeout: int = 60  # seconds

settings = Settings()
