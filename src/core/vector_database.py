"""
ChromaDB Vector Database manager for the RAG system
Handles vector storage, retrieval, and collection management
"""

import uuid
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

from config.settings import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class VectorDatabase:
    """ChromaDB vector database manager"""
    
    def __init__(self):
        self.config = config.chroma
        self.embedding_config = config.embedding
        self.client = None
        self.collection = None
        self.embedding_function = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Ensure database directory exists
            db_path = Path(self.config.db_path)
            db_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client with persistent storage
            self.client = chromadb.PersistentClient(
                path=str(db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Initialize embedding function
            self._setup_embedding_function()
            
            # Create or get collection
            self._setup_collection()
            
            logger.info(f"ChromaDB initialized successfully at {db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
    
    def _setup_embedding_function(self):
        """Setup the embedding function for the collection"""
        try:
            # Use SentenceTransformer embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_config.model
            )
            
            logger.info(f"Embedding function initialized with model: {self.embedding_config.model}")
            
        except Exception as e:
            logger.error(f"Failed to setup embedding function: {str(e)}")
            raise
    
    def _setup_collection(self):
        """Create or get the document collection"""
        try:
            # Try to get existing collection
            try:
                self.collection = self.client.get_collection(
                    name=self.config.collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"Retrieved existing collection: {self.config.collection_name}")
                
            except ValueError:
                # Collection doesn't exist, create new one
                self.collection = self.client.create_collection(
                    name=self.config.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={
                        "hnsw:space": "cosine",
                        "hnsw:construction_ef": 200,
                        "hnsw:M": 16,
                        "created_at": datetime.now().isoformat()
                    }
                )
                logger.info(f"Created new collection: {self.config.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to setup collection: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Add documents to the vector database"""
        try:
            if not documents:
                logger.warning("No documents provided for indexing")
                return {"added": 0, "errors": 0}
            
            texts = []
            metadatas = []
            ids = []
            errors = 0
            
            for doc in documents:
                try:
                    # Generate unique ID for the document chunk
                    doc_id = doc.metadata.get('chunk_id', str(uuid.uuid4()))
                    
                    # Prepare metadata (ChromaDB requires string values)
                    metadata = self._prepare_metadata(doc.metadata)
                    
                    texts.append(doc.page_content)
                    metadatas.append(metadata)
                    ids.append(doc_id)
                    
                except Exception as e:
                    logger.error(f"Error preparing document for indexing: {str(e)}")
                    errors += 1
                    continue
            
            if not texts:
                logger.warning("No valid documents to add after processing")
                return {"added": 0, "errors": errors}
            
            # Add documents to collection in batches
            batch_size = config.performance.batch_size
            added_count = 0
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                
                try:
                    self.collection.add(
                        documents=batch_texts,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                    added_count += len(batch_texts)
                    logger.debug(f"Added batch {i//batch_size + 1}: {len(batch_texts)} documents")
                    
                except Exception as e:
                    logger.error(f"Error adding batch {i//batch_size + 1}: {str(e)}")
                    errors += len(batch_texts)
            
            result = {
                "added": added_count,
                "errors": errors,
                "total_documents": self.get_collection_count()
            }
            
            logger.info(f"Document indexing completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise
    
    def _prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, str]:
        """Prepare metadata for ChromaDB (convert all values to strings)"""
        prepared = {}
        
        for key, value in metadata.items():
            if isinstance(value, (datetime, int, float, bool)):
                prepared[key] = str(value)
            elif isinstance(value, str):
                prepared[key] = value
            else:
                # Convert other types to string representation
                prepared[key] = str(value)
        
        return prepared
    
    def search(self, query: str, k: int = None, filter_dict: Dict[str, Any] = None) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        try:
            if k is None:
                k = config.retrieval.k
            
            # Prepare query filter
            where_clause = None
            if filter_dict:
                where_clause = {key: {"$eq": str(value)} for key, value in filter_dict.items()}
            
            # Perform similarity search
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert results to Document objects with scores
            documents_with_scores = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc_text, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (1 - cosine_distance)
                    similarity_score = 1.0 - distance
                    
                    # Create Document object
                    document = Document(
                        page_content=doc_text,
                        metadata=metadata
                    )
                    
                    documents_with_scores.append((document, similarity_score))
            
            logger.debug(f"Search returned {len(documents_with_scores)} results for query: {query[:50]}...")
            return documents_with_scores
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise
    
    def search_by_metadata(self, metadata_filter: Dict[str, str], k: int = 10) -> List[Document]:
        """Search documents by metadata filters"""
        try:
            where_clause = {key: {"$eq": value} for key, value in metadata_filter.items()}
            
            results = self.collection.get(
                where=where_clause,
                limit=k,
                include=["documents", "metadatas"]
            )
            
            documents = []
            if results['documents']:
                for doc_text, metadata in zip(results['documents'], results['metadatas']):
                    documents.append(Document(page_content=doc_text, metadata=metadata))
            
            logger.debug(f"Metadata search returned {len(documents)} results")
            return documents
            
        except Exception as e:
            logger.error(f"Metadata search failed: {str(e)}")
            raise
    
    def delete_documents(self, filter_dict: Dict[str, str] = None, ids: List[str] = None) -> int:
        """Delete documents from the collection"""
        try:
            deleted_count = 0
            
            if ids:
                # Delete by IDs
                self.collection.delete(ids=ids)
                deleted_count = len(ids)
                logger.info(f"Deleted {deleted_count} documents by IDs")
                
            elif filter_dict:
                # Delete by metadata filter
                where_clause = {key: {"$eq": value} for key, value in filter_dict.items()}
                
                # First, get the IDs to delete
                results = self.collection.get(
                    where=where_clause,
                    include=["metadatas"]
                )
                
                if results['ids']:
                    self.collection.delete(ids=results['ids'])
                    deleted_count = len(results['ids'])
                    logger.info(f"Deleted {deleted_count} documents by metadata filter")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            raise
    
    def update_document(self, doc_id: str, document: Document) -> bool:
        """Update a specific document in the collection"""
        try:
            metadata = self._prepare_metadata(document.metadata)
            
            self.collection.update(
                ids=[doc_id],
                documents=[document.page_content],
                metadatas=[metadata]
            )
            
            logger.debug(f"Updated document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {str(e)}")
            return False
    
    def get_collection_count(self) -> int:
        """Get the total number of documents in the collection"""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Failed to get collection count: {str(e)}")
            return 0
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the collection"""
        try:
            count = self.get_collection_count()
            
            if count == 0:
                return {"total_documents": 0}
            
            # Get sample of documents to analyze
            sample_size = min(100, count)
            sample_results = self.collection.get(
                limit=sample_size,
                include=["metadatas"]
            )
            
            # Analyze metadata
            file_types = {}
            sources = set()
            
            for metadata in sample_results['metadatas']:
                doc_type = metadata.get('document_type', 'unknown')
                file_types[doc_type] = file_types.get(doc_type, 0) + 1
                
                source = metadata.get('file_path', 'unknown')
                sources.add(source)
            
            return {
                "total_documents": count,
                "unique_sources": len(sources),
                "file_types": file_types,
                "sample_size": sample_size
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {"error": str(e)}
    
    def reset_collection(self) -> bool:
        """Reset the collection (delete all documents)"""
        try:
            # Delete the existing collection
            self.client.delete_collection(name=self.config.collection_name)
            
            # Recreate the collection
            self._setup_collection()
            
            logger.info(f"Collection {self.config.collection_name} has been reset")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {str(e)}")
            return False
    
    def close(self):
        """Close the database connection"""
        try:
            if self.client:
                # ChromaDB doesn't require explicit closing
                self.client = None
                self.collection = None
                logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database: {str(e)}")

# Create global vector database instance
vector_db = VectorDatabase()
