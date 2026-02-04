"""
RAG Pipeline Module

Orchestrates the complete RAG (Retrieval-Augmented Generation) workflow.
Combines document retrieval with LLM generation for grounded answers.

Key features:
- End-to-end question answering
- Source citation
- "I don't know" handling when information is not found
- Performance tracking
"""

import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from loguru import logger

from app.config import (
    TOP_K_RESULTS,
    MAX_CONTEXT_LENGTH,
    INFERENCE_WARNING_THRESHOLD,
)
from app.vector_store import get_vector_store, RetrievalResult
from app.llm_client import get_llm_client, LLMResponse


@dataclass
class RAGResponse:
    """Complete RAG response with answer and sources."""
    answer: str
    sources: List[Dict[str, Any]]
    retrieval_time: float
    generation_time: float
    total_time: float
    success: bool
    error_message: Optional[str] = None
    model_used: str = ""
    chunks_retrieved: int = 0


# System prompt for the RAG assistant
RAG_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based ONLY on the provided context.

IMPORTANT RULES:
1. Answer the question using ONLY the information in the provided context.
2. If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided documents to answer this question."
3. Be concise and direct in your answers.
4. When referencing information, mention which source it came from (e.g., "According to [filename]...").
5. Do NOT make up information or use knowledge outside of the provided context.
6. If the context is partially relevant, provide what you can and note the limitations.

Your goal is to be helpful while staying grounded in the provided documents."""


def build_context_prompt(question: str, results: List[RetrievalResult]) -> str:
    """
    Build the prompt with retrieved context for the LLM.
    
    Args:
        question: User's question
        results: Retrieved document chunks
        
    Returns:
        Formatted prompt with context
    """
    if not results:
        return f"""Context: No relevant documents found.

Question: {question}

Answer:"""
    
    # Build context from retrieved chunks
    context_parts = []
    total_length = 0
    
    for i, result in enumerate(results, 1):
        source = result.metadata.get('filename', 'Unknown')
        chunk_info = f"[Source {i}: {source}]"
        chunk_text = result.content
        
        # Check if adding this chunk would exceed max context
        chunk_full = f"{chunk_info}\n{chunk_text}\n"
        if total_length + len(chunk_full) > MAX_CONTEXT_LENGTH:
            # Truncate if necessary
            remaining = MAX_CONTEXT_LENGTH - total_length - len(chunk_info) - 50
            if remaining > 100:
                chunk_text = chunk_text[:remaining] + "..."
                context_parts.append(f"{chunk_info}\n{chunk_text}")
            break
        
        context_parts.append(chunk_full)
        total_length += len(chunk_full)
    
    context = "\n".join(context_parts)
    
    prompt = f"""Context from uploaded documents:
---
{context}
---

Question: {question}

Based on the context above, provide a clear and accurate answer. Remember to cite your sources."""
    
    return prompt


class RAGPipeline:
    """
    Main RAG pipeline that combines retrieval and generation.
    
    Workflow:
    1. Retrieve relevant chunks from vector store
    2. Build context prompt
    3. Generate answer using LLM
    4. Format response with sources
    """
    
    def __init__(self):
        """Initialize the RAG pipeline."""
        self.vector_store = get_vector_store()
        self.llm_client = get_llm_client()
        logger.info("RAG Pipeline initialized")
    
    def is_ready(self) -> Dict[str, bool]:
        """Check if the pipeline is ready to answer questions."""
        return {
            "llm_configured": self.llm_client.is_configured(),
            "documents_loaded": self.vector_store.document_count > 0,
            "ready": self.llm_client.is_configured() and self.vector_store.document_count > 0
        }
    
    def ask(
        self,
        question: str,
        top_k: int = TOP_K_RESULTS,
        filter_filename: Optional[str] = None
    ) -> RAGResponse:
        """
        Answer a question using RAG.
        
        Args:
            question: The user's question
            top_k: Number of chunks to retrieve
            filter_filename: Optional filter to search only in specific document
            
        Returns:
            RAGResponse with answer and metadata
        """
        total_start = time.time()
        
        # Validate input
        if not question or not question.strip():
            return RAGResponse(
                answer="",
                sources=[],
                retrieval_time=0,
                generation_time=0,
                total_time=0,
                success=False,
                error_message="Please enter a question."
            )
        
        question = question.strip()
        
        # Check if ready
        status = self.is_ready()
        if not status["llm_configured"]:
            return RAGResponse(
                answer="",
                sources=[],
                retrieval_time=0,
                generation_time=0,
                total_time=0,
                success=False,
                error_message="LLM not configured. Please set your API key in the .env file."
            )
        
        if not status["documents_loaded"]:
            return RAGResponse(
                answer="",
                sources=[],
                retrieval_time=0,
                generation_time=0,
                total_time=0,
                success=False,
                error_message="No documents loaded. Please upload documents first."
            )
        
        try:
            # Step 1: Retrieve relevant chunks
            retrieval_start = time.time()
            
            filter_metadata = None
            if filter_filename:
                filter_metadata = {"filename": filter_filename}
            
            results = self.vector_store.search(
                query=question,
                top_k=top_k,
                filter_metadata=filter_metadata
            )
            
            retrieval_time = time.time() - retrieval_start
            logger.debug(f"Retrieved {len(results)} chunks in {retrieval_time:.3f}s")
            
            # Step 2: Build context prompt
            prompt = build_context_prompt(question, results)
            
            # Step 3: Generate answer
            generation_start = time.time()
            
            llm_response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=RAG_SYSTEM_PROMPT
            )
            
            generation_time = time.time() - generation_start
            total_time = time.time() - total_start
            
            # Log performance
            if total_time > INFERENCE_WARNING_THRESHOLD:
                logger.warning(f"Inference time ({total_time:.2f}s) exceeded threshold ({INFERENCE_WARNING_THRESHOLD}s)")
            else:
                logger.info(f"Question answered in {total_time:.2f}s")
            
            # Handle LLM errors
            if not llm_response.success:
                return RAGResponse(
                    answer="",
                    sources=[],
                    retrieval_time=retrieval_time,
                    generation_time=generation_time,
                    total_time=total_time,
                    success=False,
                    error_message=llm_response.error_message,
                    model_used=llm_response.model,
                    chunks_retrieved=len(results)
                )
            
            # Format sources
            sources = self._format_sources(results)
            
            return RAGResponse(
                answer=llm_response.content,
                sources=sources,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                total_time=total_time,
                success=True,
                model_used=llm_response.model,
                chunks_retrieved=len(results)
            )
            
        except Exception as e:
            total_time = time.time() - total_start
            logger.error(f"RAG pipeline error: {e}")
            
            return RAGResponse(
                answer="",
                sources=[],
                retrieval_time=0,
                generation_time=0,
                total_time=total_time,
                success=False,
                error_message=f"An error occurred: {str(e)}"
            )
    
    def _format_sources(self, results: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """Format retrieval results as source citations."""
        sources = []
        for result in results:
            sources.append({
                "filename": result.metadata.get("filename", "Unknown"),
                "chunk_id": result.chunk_id,
                "chunk_index": result.metadata.get("chunk_index", 0),
                "total_chunks": result.metadata.get("total_chunks", 1),
                "relevance_score": round(result.score, 3),
                "snippet": result.content[:200] + "..." if len(result.content) > 200 else result.content
            })
        return sources


# Singleton instance
_rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create the RAG pipeline singleton."""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


def ask_question(question: str, top_k: int = TOP_K_RESULTS) -> RAGResponse:
    """Convenience function to ask a question."""
    pipeline = get_rag_pipeline()
    return pipeline.ask(question, top_k)
