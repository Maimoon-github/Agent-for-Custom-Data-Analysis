"""
Embedding generation for the RAG Agent system.
Supports both Ollama and Hugging Face embedding models.
"""

import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
from abc import ABC, abstractmethod

# Ollama embeddings
from langchain_ollama import OllamaEmbeddings

# Hugging Face embeddings
from langchain.embeddings import HuggingFaceEmbeddings

# Sentence transformers for local embeddings
from sentence_transformers import SentenceTransformer

from .config import EmbeddingConfig

logger = logging.getLogger(__name__)

class EmbeddingInterface(ABC):
    """Abstract interface for embedding models."""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        pass

class OllamaEmbeddingModel(EmbeddingInterface):
    """Ollama embedding model implementation."""
    
    def __init__(self, config: EmbeddingConfig, base_url: str = "http://localhost:11434"):
        """Initialize Ollama embedding model.
        
        Args:
            config: Embedding configuration
            base_url: Ollama server base URL
        """
        self.config = config
        self.base_url = base_url
        
        try:
            self.embeddings = OllamaEmbeddings(
                base_url=base_url,
                model=config.model_name
            )
            logger.info(f"Initialized Ollama embeddings with model: {config.model_name}")
        except Exception as e:
            logger.error(f"Error initializing Ollama embeddings: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Process in batches to avoid memory issues
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.embeddings.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                
                logger.debug(f"Embedded batch {i // batch_size + 1}: {len(batch)} texts")
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        # Try to get dimension by embedding a test string
        try:
            test_embedding = self.embed_query("test")
            return len(test_embedding)
        except Exception:
            # Default dimension for common models
            return 384

class HuggingFaceEmbeddingModel(EmbeddingInterface):
    """Hugging Face embedding model implementation."""
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize Hugging Face embedding model.
        
        Args:
            config: Embedding configuration
        """
        self.config = config
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.model_name,
                model_kwargs={'device': 'cpu'},  # Use CPU by default
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info(f"Initialized HuggingFace embeddings with model: {config.model_name}")
        except Exception as e:
            logger.error(f"Error initializing HuggingFace embeddings: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        try:
            test_embedding = self.embed_query("test")
            return len(test_embedding)
        except Exception:
            return 384  # Default for all-MiniLM-L6-v2

class SentenceTransformerEmbeddingModel(EmbeddingInterface):
    """Sentence Transformer embedding model implementation."""
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize Sentence Transformer embedding model.
        
        Args:
            config: Embedding configuration
        """
        self.config = config
        
        try:
            self.model = SentenceTransformer(config.model_name)
            logger.info(f"Initialized SentenceTransformer with model: {config.model_name}")
        except Exception as e:
            logger.error(f"Error initializing SentenceTransformer: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        try:
            embedding = self.model.encode([text], convert_to_tensor=False)
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

class EmbeddingGenerator:
    """Main embedding generator class."""
    
    def __init__(self, config: EmbeddingConfig, ollama_base_url: str = "http://localhost:11434"):
        """Initialize embedding generator.
        
        Args:
            config: Embedding configuration
            ollama_base_url: Ollama server base URL
        """
        self.config = config
        self.ollama_base_url = ollama_base_url
        self.model = self._initialize_model()
        
    def _initialize_model(self) -> EmbeddingInterface:
        """Initialize the embedding model based on configuration."""
        model_type = self.config.model_type.lower()
        
        if model_type == "ollama":
            return OllamaEmbeddingModel(self.config, self.ollama_base_url)
        elif model_type == "huggingface":
            return HuggingFaceEmbeddingModel(self.config)
        elif model_type == "sentence_transformer":
            return SentenceTransformerEmbeddingModel(self.config)
        else:
            # Default to sentence transformers
            logger.warning(f"Unknown model type: {model_type}. Using sentence_transformer.")
            config_copy = EmbeddingConfig()
            config_copy.model_name = "all-MiniLM-L6-v2"
            return SentenceTransformerEmbeddingModel(config_copy)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Filter out empty texts
        non_empty_texts = [text for text in texts if text.strip()]
        if not non_empty_texts:
            return []
        
        # Truncate texts if they're too long
        processed_texts = []
        for text in non_empty_texts:
            if len(text) > self.config.max_tokens * 4:  # Rough token estimation
                truncated_text = text[:self.config.max_tokens * 4]
                processed_texts.append(truncated_text)
                logger.debug(f"Truncated text from {len(text)} to {len(truncated_text)} characters")
            else:
                processed_texts.append(text)
        
        return self.model.embed_documents(processed_texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        if not text.strip():
            raise ValueError("Query text cannot be empty")
        
        # Truncate if too long
        if len(text) > self.config.max_tokens * 4:
            text = text[:self.config.max_tokens * 4]
            logger.debug(f"Truncated query text to {len(text)} characters")
        
        return self.model.embed_query(text)
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_dimension()
    
    def compute_similarity(self, 
                          embedding1: List[float], 
                          embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            
            if norm_product == 0:
                return 0.0
            
            similarity = dot_product / norm_product
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def batch_similarity(self, 
                        query_embedding: List[float], 
                        document_embeddings: List[List[float]]) -> List[float]:
        """Compute similarities between a query and multiple documents.
        
        Args:
            query_embedding: Query embedding vector
            document_embeddings: List of document embedding vectors
            
        Returns:
            List of similarity scores
        """
        similarities = []
        for doc_embedding in document_embeddings:
            similarity = self.compute_similarity(query_embedding, doc_embedding)
            similarities.append(similarity)
        
        return similarities
    
    def find_most_similar(self, 
                         query_embedding: List[float], 
                         document_embeddings: List[List[float]], 
                         top_k: int = 5) -> List[tuple]:
        """Find the most similar documents to a query.
        
        Args:
            query_embedding: Query embedding vector
            document_embeddings: List of document embedding vectors
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples, sorted by similarity
        """
        similarities = self.batch_similarity(query_embedding, document_embeddings)
        
        # Create (index, similarity) pairs and sort by similarity
        indexed_similarities = [(i, sim) for i, sim in enumerate(similarities)]
        indexed_similarities.sort(key=lambda x: x[1], reverse=True)
        
        return indexed_similarities[:top_k]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_type': self.config.model_type,
            'model_name': self.config.model_name,
            'dimension': self.get_dimension(),
            'max_tokens': self.config.max_tokens,
            'chunk_size': self.config.chunk_size,
            'chunk_overlap': self.config.chunk_overlap
        }

class EmbeddingCache:
    """Cache for embedding vectors to avoid recomputation."""
    
    def __init__(self, max_size: int = 10000):
        """Initialize embedding cache.
        
        Args:
            max_size: Maximum number of cached embeddings
        """
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
    
    def _compute_hash(self, text: str) -> str:
        """Compute hash for text."""
        import hashlib
        return hashlib.sha256(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Cached embedding or None if not found
        """
        text_hash = self._compute_hash(text)
        
        if text_hash in self.cache:
            # Move to end to mark as recently accessed
            self.access_order.remove(text_hash)
            self.access_order.append(text_hash)
            return self.cache[text_hash]
        
        return None
    
    def put(self, text: str, embedding: List[float]) -> None:
        """Store embedding in cache.
        
        Args:
            text: Text that was embedded
            embedding: Embedding vector
        """
        text_hash = self._compute_hash(text)
        
        # Remove if already exists
        if text_hash in self.cache:
            self.access_order.remove(text_hash)
        
        # Add to cache
        self.cache[text_hash] = embedding
        self.access_order.append(text_hash)
        
        # Evict oldest if cache is full
        while len(self.cache) > self.max_size:
            oldest_hash = self.access_order.pop(0)
            del self.cache[oldest_hash]
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)

class CachedEmbeddingGenerator(EmbeddingGenerator):
    """Embedding generator with caching support."""
    
    def __init__(self, config: EmbeddingConfig, 
                 ollama_base_url: str = "http://localhost:11434",
                 cache_size: int = 10000):
        """Initialize cached embedding generator.
        
        Args:
            config: Embedding configuration
            ollama_base_url: Ollama server base URL
            cache_size: Maximum cache size
        """
        super().__init__(config, ollama_base_url)
        self.cache = EmbeddingCache(cache_size)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query with caching."""
        # Check cache first
        cached_embedding = self.cache.get(text)
        if cached_embedding is not None:
            logger.debug("Using cached embedding for query")
            return cached_embedding
        
        # Compute embedding
        embedding = super().embed_query(text)
        
        # Store in cache
        self.cache.put(text, embedding)
        
        return embedding
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': self.cache.size(),
            'max_cache_size': self.cache.max_size,
            'cache_utilization': self.cache.size() / self.cache.max_size
        }