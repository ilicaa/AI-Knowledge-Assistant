"""
Embeddings Module

Handles the generation of vector embeddings for document chunks.
Uses sentence-transformers for local, fast embedding generation.

Key features:
- Uses lightweight all-MiniLM-L6-v2 model for speed
- Batch processing for efficiency
- Caching to avoid recomputation
"""

import time
from typing import List, Optional
from functools import lru_cache
from loguru import logger

from app.config import EMBEDDING_MODEL, EMBEDDING_DIMENSION, CACHE_EMBEDDINGS


class EmbeddingGenerator:
    """
    Generates embeddings using sentence-transformers.
    
    Uses all-MiniLM-L6-v2 by default which provides:
    - Fast inference (ideal for <5s requirement)
    - Good quality embeddings (384 dimensions)
    - Small model size (~80MB)
    """
    
    _instance = None
    _model = None
    
    def __new__(cls):
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the embedding model."""
        if EmbeddingGenerator._model is None:
            self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        start_time = time.time()
        
        try:
            EmbeddingGenerator._model = SentenceTransformer(EMBEDDING_MODEL)
            load_time = time.time() - start_time
            logger.success(f"Embedding model loaded in {load_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Could not load embedding model: {str(e)}")
    
    @property
    def model(self):
        """Get the loaded model."""
        return EmbeddingGenerator._model
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return EMBEDDING_DIMENSION
    
    def generate(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * self.dimension
        
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Filter empty texts and track indices
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
        
        if not valid_texts:
            return [[0.0] * self.dimension for _ in texts]
        
        try:
            start_time = time.time()
            
            embeddings = self.model.encode(
                valid_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32  # Optimize batch size for performance
            )
            
            elapsed = time.time() - start_time
            logger.debug(f"Generated {len(valid_texts)} embeddings in {elapsed:.2f}s")
            
            # Build result list with zero vectors for empty texts
            result = [[0.0] * self.dimension for _ in texts]
            for idx, embedding in zip(valid_indices, embeddings):
                result[idx] = embedding.tolist()
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise


# Global instance for caching
_embedding_generator: Optional[EmbeddingGenerator] = None


def get_embedding_generator() -> EmbeddingGenerator:
    """Get or create the embedding generator singleton."""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    return _embedding_generator


def generate_embedding(text: str) -> List[float]:
    """Convenience function to generate a single embedding."""
    generator = get_embedding_generator()
    return generator.generate(text)


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Convenience function to generate multiple embeddings."""
    generator = get_embedding_generator()
    return generator.generate_batch(texts)


# Optional: Cache frequently used embeddings
if CACHE_EMBEDDINGS:
    @lru_cache(maxsize=1000)
    def cached_generate_embedding(text: str) -> tuple:
        """
        Cached version of embedding generation.
        Returns tuple for hashability with lru_cache.
        """
        embedding = generate_embedding(text)
        return tuple(embedding)
    
    def get_cached_embedding(text: str) -> List[float]:
        """Get embedding from cache or generate new one."""
        return list(cached_generate_embedding(text))
