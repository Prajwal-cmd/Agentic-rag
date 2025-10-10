"""
Contextual Embedding Service

Pattern: Anthropic's Contextual Retrieval (September 2024)
Source: https://www.anthropic.com/news/contextual-retrieval

Generates chunk-specific context before embedding to improve retrieval accuracy.
Research shows 49% reduction in retrieval failures.
"""

from typing import List, Dict, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..config import settings
from ..services.llm_service import get_groq_service
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class ContextualEmbedder:
    """
    Adds document-level context to each chunk before embedding.
    
    Pattern: Contextual Retrieval
    - Before: "The findings showed a 15% improvement"
    - After: "[Document: Study on RAG Performance] [Section: Results] The findings showed a 15% improvement"
    """
    
    def __init__(self):
        self.groq_service = get_groq_service(settings.groq_api_key)
        self.llm = self.groq_service.get_llm(
            settings.context_generation_model,
            temperature=0.0  # Deterministic context generation
        )
        
        self.context_prompt = ChatPromptTemplate.from_template("""<document> 
{doc_content}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context (50-100 tokens) to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.""")
        
        self.chain = self.context_prompt | self.llm | StrOutputParser()
    
    def generate_context_for_chunk(
        self,
        chunk: str,
        full_document: str,
        filename: str
    ) -> str:
        """
        Generate contextual prefix for a single chunk.
        
        Args:
            chunk: The chunk text
            full_document: Complete document text (or summary for long docs)
            filename: Source filename
            
        Returns:
            Contextual prefix string
        """
        try:
            # For very long documents, use first 3000 chars as context
            doc_context = full_document[:3000] if len(full_document) > 3000 else full_document
            
            context = self.chain.invoke({
                "doc_content": doc_context,
                "chunk_content": chunk
            })
            
            # Format: [Source: filename] [Context] Original chunk
            contextual_chunk = f"[Source: {filename}] {context.strip()}\n\n{chunk}"
            
            logger.debug(f"Generated context for chunk from {filename}")
            return contextual_chunk
            
        except Exception as e:
            logger.warning(f"Context generation failed: {e}. Using chunk without context.")
            # Fallback: Just add filename
            return f"[Source: {filename}]\n\n{chunk}"
    
    def generate_contexts_batch(
        self,
        chunks: List[Dict],
        full_document: str,
        filename: str
    ) -> List[Dict]:
        """
        Generate contextual embeddings for multiple chunks.
        
        Args:
            chunks: List of chunk dicts with 'text' and 'metadata'
            full_document: Complete document text
            filename: Source filename
            
        Returns:
            List of chunk dicts with contextualized text
        """
        if not settings.enable_contextual_embedding:
            logger.info("Contextual embedding disabled. Using original chunks.")
            return chunks
        
        contextualized_chunks = []
        
        for i, chunk in enumerate(chunks):
            try:
                contextual_text = self.generate_context_for_chunk(
                    chunk["text"],
                    full_document,
                    filename
                )
                
                # Create new chunk dict with contextualized text
                contextualized_chunk = {
                    "text": contextual_text,
                    "original_text": chunk["text"],  # Keep original for reference
                    "metadata": {
                        **chunk["metadata"],
                        "has_context": True
                    }
                }
                contextualized_chunks.append(contextualized_chunk)
                
                # Log progress every 10 chunks
                if (i + 1) % 10 == 0:
                    logger.info(f"Contextualized {i + 1}/{len(chunks)} chunks")
                    
            except Exception as e:
                logger.error(f"Failed to contextualize chunk {i}: {e}")
                # Fallback to original chunk
                contextualized_chunks.append(chunk)
        
        logger.info(f"✅ Successfully contextualized {len(contextualized_chunks)} chunks")
        return contextualized_chunks


# Global instance
_contextual_embedder = None

def get_contextual_embedder() -> ContextualEmbedder:
    """Get or create global contextual embedder instance."""
    global _contextual_embedder
    if _contextual_embedder is None:
        _contextual_embedder = ContextualEmbedder()
    return _contextual_embedder
