"""
Unit tests for the document processor module.
"""

import pytest
from app.document_processor import DocumentProcessor, DocumentChunk, ProcessedDocument


class TestDocumentProcessor:
    """Tests for DocumentProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    
    def test_is_supported_pdf(self):
        """Test PDF file support check."""
        assert self.processor.is_supported("test.pdf") is True
        assert self.processor.is_supported("test.PDF") is True
    
    def test_is_supported_txt(self):
        """Test TXT file support check."""
        assert self.processor.is_supported("test.txt") is True
        assert self.processor.is_supported("test.TXT") is True
    
    def test_is_supported_md(self):
        """Test MD file support check."""
        assert self.processor.is_supported("test.md") is True
        assert self.processor.is_supported("test.MD") is True
    
    def test_is_supported_unsupported(self):
        """Test unsupported file types."""
        assert self.processor.is_supported("test.docx") is False
        assert self.processor.is_supported("test.jpg") is False
    
    def test_process_text_file(self):
        """Test processing a simple text file."""
        content = b"This is a test document with some content for testing."
        result = self.processor.process_file("test.txt", content)
        
        assert result.status == "success"
        assert result.filename == "test.txt"
        assert len(result.chunks) > 0
    
    def test_process_empty_file(self):
        """Test processing an empty file."""
        content = b""
        result = self.processor.process_file("empty.txt", content)
        
        assert result.status == "error"
        # Error can be "No text content" or file read error
        assert result.error_message is not None
    
    def test_chunk_metadata(self):
        """Test that chunks have proper metadata."""
        content = b"This is a test document with some content."
        result = self.processor.process_file("test.txt", content)
        
        if result.chunks:
            chunk = result.chunks[0]
            assert "filename" in chunk.metadata
            assert "chunk_index" in chunk.metadata
            assert chunk.metadata["filename"] == "test.txt"


class TestDocumentChunk:
    """Tests for DocumentChunk dataclass."""
    
    def test_chunk_id_generation(self):
        """Test that chunk IDs are generated automatically."""
        chunk = DocumentChunk(
            content="Test content",
            metadata={"filename": "test.txt", "chunk_index": 0}
        )
        assert chunk.chunk_id != ""
        assert len(chunk.chunk_id) == 12
    
    def test_chunk_id_uniqueness(self):
        """Test that different chunks get different IDs."""
        chunk1 = DocumentChunk(
            content="Content 1",
            metadata={"filename": "test.txt", "chunk_index": 0}
        )
        chunk2 = DocumentChunk(
            content="Content 2",
            metadata={"filename": "test.txt", "chunk_index": 1}
        )
        assert chunk1.chunk_id != chunk2.chunk_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
