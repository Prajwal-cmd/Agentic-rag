"""
Qdrant Vector Store Service
Pattern: Production-quality vector database with HNSW indexing
Source: Qdrant Cloud free tier (1GB storage)
"""
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)
from typing import List, Dict, Optional
from uuid import uuid4
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class VectorStoreService:
    """
    Qdrant vector store operations for document storage and retrieval.
    """
    
    def __init__(self, url: str, api_key: str, collection_name: str, embedding_dim: int = 384):
        """
        Initialize Qdrant client and ensure collection exists.
        
        Args:
            url: Qdrant cluster URL
            api_key: Qdrant API key
            collection_name: Collection name for this session
            embedding_dim: Embedding vector dimension
        """
        logger.info(f"Connecting to Qdrant: {url}")
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        
        # Ensure collection exists
        self._ensure_collection()
    
    def _ensure_collection(self):
        """
        Create collection if it doesn't exist.
        Uses cosine distance for semantic similarity.
        """
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE  # Cosine similarity for semantic search
                    )
                )
                logger.info("Collection created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict]] = None
    ) -> List[str]:
        """
        Add documents to vector store.
        
        Args:
            texts: Document texts
            embeddings: Embedding vectors
            metadatas: Optional metadata for each document
            
        Returns:
            List of assigned document IDs
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        points = []
        ids = []
        
        for text, embedding, metadata in zip(texts, embeddings, metadatas):
            point_id = str(uuid4())
            ids.append(point_id)
            
            # Store text in payload for retrieval
            payload = {
                "text": text,
                **metadata
            }
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
            )
        
        # Batch upsert
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        logger.info(f"Added {len(points)} documents to vector store")
        return ids
    
    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 4,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Search for similar documents using cosine similarity.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            score_threshold: Minimum similarity score (optional)
            
        Returns:
            List of documents with scores
        """
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k,
            score_threshold=score_threshold
        )
        
        documents = []
        for hit in search_result:
            documents.append({
                "id": hit.id,
                "text": hit.payload.get("text", ""),
                "score": hit.score,
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
            })
        
        logger.info(f"Retrieved {len(documents)} documents from vector store")
        return documents
    
    def delete_collection(self):
        """
        Delete the entire collection (for cleanup after conversation).
        """
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
    
    def get_collection_info(self) -> Dict:
        """Get collection statistics"""
        try:
            # Check if collection exists first
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                logger.warning(f"Collection {self.collection_name} does not exist")
                return {
                    "name": self.collection_name,
                    "vectors_count": 0,
                    "points_count": 0,
                    "exists": False
                }
            
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count if hasattr(info, 'vectors_count') else 0,
                "points_count": info.points_count if hasattr(info, 'points_count') else 0,
                "exists": True
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {
                "name": self.collection_name,
                "vectors_count": 0,
                "points_count": 0,
                "exists": False
            }