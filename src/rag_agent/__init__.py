"""
RAG Agent package initialization.
"""

__version__ = "1.0.0"
__author__ = "RAG Agent Development Team"
__description__ = "Privacy-focused local RAG agent for custom data analysis"

from .config import RAGConfig, config
from .llm_interface import OllamaLLM
from .vector_store import ChromaVectorStore
from .document_processor import DocumentProcessor
from .embeddings import EmbeddingGenerator
from .retriever import DocumentRetriever
from .rag_agent import RAGAgent

__all__ = [
    "RAGConfig",
    "config", 
    "OllamaLLM",
    "ChromaVectorStore",
    "DocumentProcessor",
    "EmbeddingGenerator",
    "DocumentRetriever",
    "RAGAgent"
]