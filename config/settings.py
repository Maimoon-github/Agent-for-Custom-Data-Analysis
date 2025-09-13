"""
Configuration management for the RAG Agent
"""

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class OllamaConfig(BaseModel):
    """Ollama LLM configuration"""
    base_url: str = Field(default="http://localhost:11434")
    model: str = Field(default="llama3")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, gt=0)
    context_window: int = Field(default=4096, gt=0)

class EmbeddingConfig(BaseModel):
    """Embedding model configuration"""
    model: str = Field(default="all-MiniLM-L6-v2")
    dimension: int = Field(default=384, gt=0)

class ChromaConfig(BaseModel):
    """ChromaDB configuration"""
    db_path: str = Field(default="./data/vectordb")
    collection_name: str = Field(default="knowledge_base")

class DocumentConfig(BaseModel):
    """Document processing configuration"""
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    max_chunk_size: int = Field(default=1500, gt=0)
    supported_formats: list = Field(default=[".pdf", ".txt", ".docx", ".csv"])

class RetrievalConfig(BaseModel):
    """Retrieval configuration"""
    k: int = Field(default=5, gt=0)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    rerank_top_k: int = Field(default=3, gt=0)

class PerformanceConfig(BaseModel):
    """Performance and optimization settings"""
    batch_size: int = Field(default=100, gt=0)
    max_workers: int = Field(default=4, gt=0)

class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = Field(default="INFO")
    file: str = Field(default="./logs/rag_agent.log")

class SecurityConfig(BaseModel):
    """Security settings"""
    encrypt_sensitive_data: bool = Field(default=True)

class RAGConfig(BaseModel):
    """Main RAG system configuration"""
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    document: DocumentConfig = Field(default_factory=DocumentConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Load configuration from environment variables"""
        return cls(
            ollama=OllamaConfig(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                model=os.getenv("OLLAMA_MODEL", "llama3"),
                temperature=float(os.getenv("TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("MAX_TOKENS", "2048")),
                context_window=int(os.getenv("CONTEXT_WINDOW", "4096"))
            ),
            embedding=EmbeddingConfig(
                model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
                dimension=int(os.getenv("EMBEDDING_DIMENSION", "384"))
            ),
            chroma=ChromaConfig(
                db_path=os.getenv("CHROMA_DB_PATH", "./data/vectordb"),
                collection_name=os.getenv("COLLECTION_NAME", "knowledge_base")
            ),
            document=DocumentConfig(
                chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
                chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
                max_chunk_size=int(os.getenv("MAX_CHUNK_SIZE", "1500"))
            ),
            retrieval=RetrievalConfig(
                k=int(os.getenv("RETRIEVAL_K", "5")),
                similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.7")),
                rerank_top_k=int(os.getenv("RERANK_TOP_K", "3"))
            ),
            performance=PerformanceConfig(
                batch_size=int(os.getenv("BATCH_SIZE", "100")),
                max_workers=int(os.getenv("MAX_WORKERS", "4"))
            ),
            logging=LoggingConfig(
                level=os.getenv("LOG_LEVEL", "INFO"),
                file=os.getenv("LOG_FILE", "./logs/rag_agent.log")
            ),
            security=SecurityConfig(
                encrypt_sensitive_data=os.getenv("ENCRYPT_SENSITIVE_DATA", "true").lower() == "true"
            )
        )

# Global configuration instance
config = RAGConfig.from_env()
