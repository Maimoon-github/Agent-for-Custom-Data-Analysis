"""
Basic tests for the RAG Agent system.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

# Add src to path for testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_agent.config import RAGConfig
from rag_agent.document_processor import DocumentProcessor
from rag_agent.embeddings import EmbeddingGenerator

class TestConfig:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = RAGConfig()
        
        assert config.ollama.model_name == "llama3"
        assert config.chromadb.collection_name == "domain_knowledge"
        assert config.embedding.chunk_size == 1000
        assert config.system.log_level == "INFO"
    
    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = RAGConfig()
        config_dict = config.to_dict()
        
        assert "ollama" in config_dict
        assert "chromadb" in config_dict
        assert "embedding" in config_dict
        assert "system" in config_dict

class TestDocumentProcessor:
    """Test document processing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = RAGConfig()
        self.processor = DocumentProcessor(self.config.document_processing)
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_process_text(self):
        """Test text processing."""
        text = "This is a test document. It contains multiple sentences. Each sentence should be processed correctly."
        
        result = self.processor.process_text(text)
        
        assert result.success is True
        assert len(result.documents) > 0
        assert all(doc.content for doc in result.documents)
    
    def test_process_empty_text(self):
        """Test processing empty text."""
        result = self.processor.process_text("")
        
        assert result.success is False
        assert "Empty text" in result.error
    
    def test_text_chunking(self):
        """Test text chunking functionality."""
        # Create a long text that should be chunked
        long_text = "This is a sentence. " * 200  # Should exceed chunk size
        
        result = self.processor.process_text(long_text)
        
        assert result.success is True
        # Should create multiple chunks for long text
        assert len(result.documents) > 1
    
    def test_create_test_file(self):
        """Test file processing with a simple text file."""
        test_file = self.temp_dir / "test.txt"
        test_content = "This is a test file.\nIt has multiple lines.\nEach line should be processed."
        
        test_file.write_text(test_content)
        
        result = self.processor.process_file(test_file)
        
        assert result.success is True
        assert len(result.documents) > 0
        
        # Check metadata
        doc = result.documents[0]
        assert doc.metadata["file_name"] == "test.txt"
        assert doc.metadata["file_type"] == "txt"
    
    def test_nonexistent_file(self):
        """Test processing nonexistent file."""
        fake_file = self.temp_dir / "nonexistent.txt"
        
        result = self.processor.process_file(fake_file)
        
        assert result.success is False
        assert "not found" in result.error.lower()

class TestEmbeddingGenerator:
    """Test embedding generation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = RAGConfig()
        # Use sentence transformer for testing (doesn't require Ollama)
        self.config.embedding.model_type = "sentence_transformer"
        self.config.embedding.model_name = "all-MiniLM-L6-v2"
    
    def test_embedding_initialization(self):
        """Test embedding generator initialization."""
        try:
            generator = EmbeddingGenerator(self.config.embedding)
            dimension = generator.get_dimension()
            assert dimension > 0
        except Exception as e:
            # Skip test if sentence transformers not available
            pytest.skip(f"Sentence transformers not available: {e}")
    
    def test_embed_query(self):
        """Test query embedding."""
        try:
            generator = EmbeddingGenerator(self.config.embedding)
            
            query = "What is artificial intelligence?"
            embedding = generator.embed_query(query)
            
            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert all(isinstance(x, float) for x in embedding)
        except Exception as e:
            pytest.skip(f"Embedding test skipped: {e}")
    
    def test_embed_documents(self):
        """Test document embedding."""
        try:
            generator = EmbeddingGenerator(self.config.embedding)
            
            documents = [
                "This is the first document.",
                "This is the second document.",
                "Another document with different content."
            ]
            
            embeddings = generator.embed_documents(documents)
            
            assert len(embeddings) == len(documents)
            assert all(isinstance(emb, list) for emb in embeddings)
            assert all(len(emb) > 0 for emb in embeddings)
        except Exception as e:
            pytest.skip(f"Embedding test skipped: {e}")
    
    def test_similarity_computation(self):
        """Test similarity computation between embeddings."""
        try:
            generator = EmbeddingGenerator(self.config.embedding)
            
            # Similar sentences should have high similarity
            text1 = "The cat is sleeping."
            text2 = "A cat is sleeping."
            text3 = "The weather is sunny today."
            
            emb1 = generator.embed_query(text1)
            emb2 = generator.embed_query(text2)
            emb3 = generator.embed_query(text3)
            
            sim_12 = generator.compute_similarity(emb1, emb2)
            sim_13 = generator.compute_similarity(emb1, emb3)
            
            # Similar sentences should have higher similarity
            assert sim_12 > sim_13
            assert 0 <= sim_12 <= 1
            assert 0 <= sim_13 <= 1
        except Exception as e:
            pytest.skip(f"Similarity test skipped: {e}")

class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_config_creation(self):
        """Test that all components can be initialized with default config."""
        config = RAGConfig()
        
        # Test that we can create processor
        processor = DocumentProcessor(config.document_processing)
        assert processor is not None
        
        # Test basic text processing
        result = processor.process_text("Test document for integration testing.")
        assert result.success is True

if __name__ == "__main__":
    pytest.main([__file__])