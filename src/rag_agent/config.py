"""
Configuration management for the RAG Agent system.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OllamaConfig(BaseSettings):
    """Configuration for Ollama LLM integration."""
    
    base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    model_name: str = Field(default="llama3", description="Default model name")
    temperature: float = Field(default=0.7, description="Model temperature")
    context_window: int = Field(default=4096, description="Model context window")
    num_gpu: int = Field(default=1, description="Number of GPUs to use")
    timeout: int = Field(default=300, description="Request timeout in seconds")

class ChromaDBConfig(BaseSettings):
    """Configuration for ChromaDB vector database."""
    
    persist_directory: str = Field(default="./chroma_db", description="ChromaDB persistence directory")
    collection_name: str = Field(default="domain_knowledge", description="Default collection name")
    distance_metric: str = Field(default="cosine", description="Distance metric for similarity search")
    max_batch_size: int = Field(default=100, description="Maximum batch size for operations")

class EmbeddingConfig(BaseSettings):
    """Configuration for embedding models."""
    
    model_name: str = Field(default="nomic-embed-text", description="Embedding model name")
    model_type: str = Field(default="ollama", description="Embedding model type (ollama/huggingface)")
    chunk_size: int = Field(default=1000, description="Text chunk size")
    chunk_overlap: int = Field(default=200, description="Text chunk overlap")
    max_tokens: int = Field(default=8192, description="Maximum tokens per chunk")

class DocumentProcessingConfig(BaseSettings):
    """Configuration for document processing."""
    
    supported_formats: list = Field(default=["pdf", "txt", "md", "csv", "docx"], description="Supported file formats")
    max_file_size_mb: int = Field(default=100, description="Maximum file size in MB")
    batch_size: int = Field(default=50, description="Batch size for processing")
    parallel_workers: int = Field(default=4, description="Number of parallel workers")

class RetrievalConfig(BaseSettings):
    """Configuration for document retrieval."""
    
    top_k: int = Field(default=5, description="Number of top documents to retrieve")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity threshold")
    rerank_top_k: int = Field(default=10, description="Number of documents to rerank")
    enable_reranking: bool = Field(default=True, description="Enable result reranking")

class SystemConfig(BaseSettings):
    """Main system configuration."""
    
    # Directories
    data_dir: str = Field(default="./data", description="Data directory")
    logs_dir: str = Field(default="./logs", description="Logs directory")
    temp_dir: str = Field(default="./temp", description="Temporary directory")
    
    # Security
    max_query_length: int = Field(default=1000, description="Maximum query length")
    enable_content_filtering: bool = Field(default=True, description="Enable content filtering")
    
    # Performance
    cache_size: int = Field(default=1000, description="Cache size for queries")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    enable_query_logging: bool = Field(default=True, description="Enable query logging")
    
    class Config:
        env_prefix = "RAG_"

class RAGConfig:
    """Main configuration class for the RAG Agent system."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.ollama = OllamaConfig()
        self.chromadb = ChromaDBConfig()
        self.embedding = EmbeddingConfig()
        self.document_processing = DocumentProcessingConfig()
        self.retrieval = RetrievalConfig()
        self.system = SystemConfig()
        
        # Load from config file if provided
        if config_file and Path(config_file).exists():
            self._load_from_file(config_file)
        
        # Create necessary directories
        self._create_directories()
    
    def _load_from_file(self, config_file: str):
        """Load configuration from YAML file."""
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Update configurations with file data
        for section, values in config_data.items():
            if hasattr(self, section):
                config_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            self.system.data_dir,
            self.system.logs_dir,
            self.system.temp_dir,
            self.chromadb.persist_directory
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "ollama": self.ollama.dict(),
            "chromadb": self.chromadb.dict(),
            "embedding": self.embedding.dict(),
            "document_processing": self.document_processing.dict(),
            "retrieval": self.retrieval.dict(),
            "system": self.system.dict()
        }
    
    def save_to_file(self, config_file: str):
        """Save configuration to YAML file."""
        with open(config_file, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

# Global configuration instance
config = RAGConfig()

# Configuration validation functions
def validate_ollama_connection(config: OllamaConfig) -> bool:
    """Validate Ollama connection."""
    try:
        import requests
        response = requests.get(f"{config.base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def validate_model_availability(config: OllamaConfig) -> bool:
    """Check if the specified model is available."""
    try:
        import requests
        response = requests.get(f"{config.base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return any(model["name"].startswith(config.model_name) for model in models)
        return False
    except Exception:
        return False