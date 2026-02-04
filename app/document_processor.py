"""
Document Processor Module

Handles document loading, parsing, and chunking for various file formats.
Supports PDF, TXT, and Markdown files.

Key features:
- Efficient chunking with configurable size and overlap
- Metadata preservation (filename, chunk_id, page numbers)
- Error handling for corrupted or unsupported files
"""

import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from loguru import logger

from app.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SUPPORTED_EXTENSIONS,
)


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata."""
    content: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""
    
    def __post_init__(self):
        """Generate chunk_id if not provided."""
        if not self.chunk_id:
            # Create unique ID from content hash + metadata
            hash_input = f"{self.content}{self.metadata.get('filename', '')}{self.metadata.get('chunk_index', 0)}"
            self.chunk_id = hashlib.md5(hash_input.encode()).hexdigest()[:12]


@dataclass
class ProcessedDocument:
    """Represents a fully processed document."""
    filename: str
    chunks: List[DocumentChunk]
    total_chunks: int
    status: str = "success"
    error_message: Optional[str] = None


class DocumentProcessor:
    """
    Processes documents into chunks suitable for embedding and retrieval.
    
    Features:
    - Supports PDF, TXT, and MD files
    - Configurable chunk size and overlap
    - Preserves source information in metadata
    """
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"DocumentProcessor initialized (chunk_size={chunk_size}, overlap={chunk_overlap})")
    
    def is_supported(self, filename: str) -> bool:
        """Check if the file type is supported."""
        ext = Path(filename).suffix.lower()
        return ext in SUPPORTED_EXTENSIONS
    
    def process_file(self, file_path: str, file_content: bytes = None) -> ProcessedDocument:
        """
        Process a single file into chunks.
        
        Args:
            file_path: Path to the file or filename
            file_content: Optional raw file content (for uploaded files)
            
        Returns:
            ProcessedDocument with chunks and metadata
        """
        filename = Path(file_path).name
        extension = Path(file_path).suffix.lower()
        
        logger.info(f"Processing file: {filename}")
        
        if not self.is_supported(filename):
            logger.error(f"Unsupported file type: {extension}")
            return ProcessedDocument(
                filename=filename,
                chunks=[],
                total_chunks=0,
                status="error",
                error_message=f"Unsupported file type: {extension}. Supported: {SUPPORTED_EXTENSIONS}"
            )
        
        try:
            # Extract text based on file type
            if extension == ".pdf":
                text = self._extract_pdf(file_path, file_content)
            elif extension in [".txt", ".md"]:
                text = self._extract_text(file_path, file_content)
            else:
                raise ValueError(f"Unsupported extension: {extension}")
            
            if not text or not text.strip():
                return ProcessedDocument(
                    filename=filename,
                    chunks=[],
                    total_chunks=0,
                    status="error",
                    error_message="No text content found in document"
                )
            
            # Create chunks
            chunks = self._create_chunks(text, filename)
            
            logger.success(f"Successfully processed {filename}: {len(chunks)} chunks created")
            
            return ProcessedDocument(
                filename=filename,
                chunks=chunks,
                total_chunks=len(chunks),
                status="success"
            )
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            return ProcessedDocument(
                filename=filename,
                chunks=[],
                total_chunks=0,
                status="error",
                error_message=str(e)
            )
    
    def _extract_pdf(self, file_path: str, file_content: bytes = None) -> str:
        """Extract text from a PDF file."""
        from pypdf import PdfReader
        from io import BytesIO
        
        try:
            if file_content:
                reader = PdfReader(BytesIO(file_content))
            else:
                reader = PdfReader(file_path)
            
            text_parts = []
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    # Add page marker for reference
                    text_parts.append(f"[Page {page_num}]\n{page_text}")
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            raise ValueError(f"Failed to extract PDF content: {str(e)}")
    
    def _extract_text(self, file_path: str, file_content: bytes = None) -> str:
        """Extract text from a TXT or MD file."""
        try:
            if file_content:
                # Try common encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        return file_content.decode(encoding)
                    except UnicodeDecodeError:
                        continue
                raise ValueError("Could not decode file with supported encodings")
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Text extraction error: {e}")
            raise ValueError(f"Failed to read text file: {str(e)}")
    
    def _create_chunks(self, text: str, filename: str) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks.
        
        Uses a simple but effective chunking strategy:
        - Fixed size chunks with overlap
        - Tries to break at sentence boundaries when possible
        - Preserves metadata for each chunk
        """
        chunks = []
        
        # Clean the text
        text = self._clean_text(text)
        
        if len(text) <= self.chunk_size:
            # Small document - single chunk
            chunks.append(DocumentChunk(
                content=text,
                metadata={
                    "filename": filename,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "start_char": 0,
                    "end_char": len(text)
                }
            ))
            return chunks
        
        # Create overlapping chunks
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Determine end position
            end = min(start + self.chunk_size, len(text))
            
            # Try to find a good break point (sentence end)
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                break_chars = ['. ', '.\n', '! ', '!\n', '? ', '?\n', '\n\n']
                best_break = end
                
                # Search in the last 20% of the chunk for a break point
                search_start = end - int(self.chunk_size * 0.2)
                for break_char in break_chars:
                    pos = text.rfind(break_char, search_start, end)
                    if pos != -1:
                        best_break = pos + len(break_char)
                        break
                
                end = best_break
            
            # Extract chunk content
            chunk_content = text[start:end].strip()
            
            if chunk_content:  # Only add non-empty chunks
                chunks.append(DocumentChunk(
                    content=chunk_content,
                    metadata={
                        "filename": filename,
                        "chunk_index": chunk_index,
                        "start_char": start,
                        "end_char": end
                    }
                ))
                chunk_index += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            
            # Prevent infinite loop
            if start >= len(text) - self.chunk_overlap:
                break
        
        # Update total_chunks in metadata
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        import re
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\t+', ' ', text)
        
        return text.strip()
    
    def process_multiple(self, files: List[tuple]) -> List[ProcessedDocument]:
        """
        Process multiple files.
        
        Args:
            files: List of (filename, file_content) tuples
            
        Returns:
            List of ProcessedDocument objects
        """
        results = []
        for filename, content in files:
            result = self.process_file(filename, content)
            results.append(result)
        return results


# Convenience function for single file processing
def process_document(file_path: str, file_content: bytes = None) -> ProcessedDocument:
    """Process a single document file."""
    processor = DocumentProcessor()
    return processor.process_file(file_path, file_content)
