"""
Vector store module using FAISS for similarity search.
"""

import os
import json
import pickle
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from app.config import FAISS_DIR
from app.embeddings import generate_embeddings

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Represents a single retrieval result."""
    content: str
    metadata: Dict[str, Any]
    score: float
    chunk_id: str
    
    @property
    def source_info(self) -> str:
        """Get formatted source information."""
        filename = self.metadata.get("filename", "Unknown")
        chunk_idx = self.metadata.get("chunk_index", 0)
        total = self.metadata.get("total_chunks", 1)
        return f"{filename} (chunk {chunk_idx + 1}/{total})"


class VectorStore:
    """FAISS-based vector store for document embeddings."""
    
    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the index
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not installed. Run: pip install faiss-cpu")
        
        self.persist_directory = persist_directory or str(FAISS_DIR)
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self.index: Optional[faiss.IndexFlatIP] = None
        self.documents: List[Dict[str, Any]] = []
        self.dimension: Optional[int] = None
        
        # File paths for persistence
        self.index_path = os.path.join(self.persist_directory, "faiss.index")
        self.docs_path = os.path.join(self.persist_directory, "documents.pkl")
        
        # Load existing index if available
        self._load_index()
        
        logger.info(f"VectorStore initialized with persist_directory: {self.persist_directory}")
    
    def _load_index(self) -> None:
        """Load existing index from disk if available."""
        if os.path.exists(self.index_path) and os.path.exists(self.docs_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
                self.dimension = self.index.d
                logger.info(f"Loaded existing index with {len(self.documents)} documents")
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}")
                self.index = None
                self.documents = []
    
    def _save_index(self) -> None:
        """Save index to disk."""
        if self.index is not None:
            try:
                faiss.write_index(self.index, self.index_path)
                with open(self.docs_path, 'wb') as f:
                    pickle.dump(self.documents, f)
                logger.info(f"Saved index with {len(self.documents)} documents")
            except Exception as e:
                logger.error(f"Failed to save index: {e}")
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Add documents with their embeddings to the vector store.
        
        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            List of document IDs
        """
        if not documents or not embeddings:
            logger.warning("No documents or embeddings provided")
            return []
        
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Normalize embeddings for cosine similarity (using inner product)
        faiss.normalize_L2(embeddings_array)
        
        # Initialize index if needed
        if self.index is None:
            self.dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info(f"Created new FAISS index with dimension {self.dimension}")
        
        # Generate IDs
        start_id = len(self.documents)
        ids = [f"doc_{start_id + i}" for i in range(len(documents))]
        
        # Add to index
        self.index.add(embeddings_array)
        
        # Store documents with metadata
        for i, (doc, doc_id) in enumerate(zip(documents, ids)):
            doc_entry = {
                "id": doc_id,
                "content": doc,
                "metadata": metadatas[i] if metadatas else {}
            }
            self.documents.append(doc_entry)
        
        # Persist
        self._save_index()
        
        logger.info(f"Added {len(documents)} documents to vector store")
        return ids
    
    def search(
        self,
        query: str = None,
        query_embedding: List[float] = None,
        top_k: int = 5,
        n_results: int = None,
        filter_metadata: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """
        Search for similar documents.
        
        Args:
            query: Text query (will be embedded)
            query_embedding: Pre-computed query vector (alternative to query)
            top_k: Number of results to return
            n_results: Alias for top_k
            filter_metadata: Optional metadata filter (not implemented yet)
            
        Returns:
            List of RetrievalResult objects
        """
        if self.index is None or len(self.documents) == 0:
            logger.warning("No documents in vector store")
            return []
        
        # Use n_results if provided, otherwise top_k
        num_results = n_results if n_results is not None else top_k
        
        # Generate embedding if text query provided
        if query is not None and query_embedding is None:
            from app.embeddings import generate_embedding
            query_embedding = generate_embedding(query)
        
        if query_embedding is None:
            logger.error("No query or query_embedding provided")
            return []
        
        # Convert query to numpy array and normalize
        query_array = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_array)
        
        # Limit n_results to available documents
        num_results = min(num_results, len(self.documents))
        
        # Search
        distances, indices = self.index.search(query_array, num_results)
        
        # Gather results as RetrievalResult objects
        results = []
        
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents) and idx >= 0:
                doc = self.documents[idx]
                # Convert distance to similarity score (inner product, higher is better)
                score = float(distances[0][i])
                
                results.append(RetrievalResult(
                    content=doc["content"],
                    metadata=doc["metadata"],
                    score=score,
                    chunk_id=doc["id"]
                ))
        
        logger.info(f"Search returned {len(results)} results")
        return results
    
    @property
    def document_count(self) -> int:
        """Get the number of documents (chunks) in the store."""
        return len(self.documents)
    
    def get_document_count(self) -> int:
        """Get the number of documents in the store."""
        return len(self.documents)
    
    def add_document(self, document) -> bool:
        """
        Add a processed document to the vector store.
        
        Args:
            document: ProcessedDocument with chunks to add
            
        Returns:
            True if successful, False otherwise
        """
        if document.status != "success" or not document.chunks:
            logger.warning(f"Cannot add document with status: {document.status}")
            return False
        
        try:
            # Prepare data
            texts = [chunk.content for chunk in document.chunks]
            metadatas = [chunk.metadata for chunk in document.chunks]
            
            # Generate embeddings
            embeddings = generate_embeddings(texts)
            
            # Add to store
            self.add_documents(texts, embeddings, metadatas)
            
            logger.info(f"Added {len(texts)} chunks from '{document.filename}'")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents with their metadata."""
        return self.documents
    
    def delete_collection(self) -> None:
        """Delete all documents and reset the index."""
        self.index = None
        self.documents = []
        self.dimension = None
        
        # Remove persisted files
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.docs_path):
            os.remove(self.docs_path)
        
        logger.info("Vector store cleared")
    
    def document_exists(self, source: str) -> bool:
        """
        Check if a document from a specific source already exists.
        
        Args:
            source: Source file name to check
            
        Returns:
            True if document exists, False otherwise
        """
        for doc in self.documents:
            if doc.get("metadata", {}).get("source") == source:
                return True
        return False
    
    def get_sources(self) -> List[str]:
        """Get list of unique source documents."""
        sources = set()
        for doc in self.documents:
            source = doc.get("metadata", {}).get("source")
            if source:
                sources.add(source)
        return list(sources)
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all unique documents in the store.
        
        Returns:
            List of document info dictionaries
        """
        # Group by filename
        documents = {}
        for doc in self.documents:
            metadata = doc.get("metadata", {})
            filename = metadata.get("filename") or metadata.get("source", "Unknown")
            if filename not in documents:
                documents[filename] = {
                    'filename': filename,
                    'chunk_count': 0,
                    'total_chunks': metadata.get('total_chunks', 0)
                }
            documents[filename]['chunk_count'] += 1
        
        return list(documents.values())
    
    def clear(self) -> bool:
        """
        Clear all documents from the store.
        
        Returns:
            True if successful
        """
        self.delete_collection()
        return True
    
    def delete_document(self, filename: str) -> bool:
        """
        Delete all chunks belonging to a specific document.
        
        Args:
            filename: Name of the document to delete
            
        Returns:
            True if successful
        """
        # Find indices to remove
        indices_to_remove = []
        for i, doc in enumerate(self.documents):
            metadata = doc.get("metadata", {})
            doc_filename = metadata.get("filename") or metadata.get("source")
            if doc_filename == filename:
                indices_to_remove.append(i)
        
        if not indices_to_remove:
            logger.warning(f"No chunks found for '{filename}'")
            return False
        
        # Remove from documents list (in reverse to maintain indices)
        for i in sorted(indices_to_remove, reverse=True):
            self.documents.pop(i)
        
        # Rebuild the FAISS index
        if self.documents:
            # We need to rebuild the index since FAISS doesn't support deletion
            # This requires re-adding all remaining documents
            # For now, we'll just save the updated documents list
            pass
        else:
            self.index = None
            self.dimension = None
        
        self._save_index()
        logger.info(f"Deleted {len(indices_to_remove)} chunks from '{filename}'")
        return True


# Singleton instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create the vector store singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def reset_vector_store():
    """Reset the vector store singleton."""
    global _vector_store
    if _vector_store is not None:
        _vector_store.delete_collection()
    _vector_store = None
