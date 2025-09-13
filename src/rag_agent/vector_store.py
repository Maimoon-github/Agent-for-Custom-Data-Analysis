"""
ChromaDB vector store implementation for the RAG Agent system.
Handles vector storage, retrieval, and similarity search.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from .config import ChromaDBConfig

logger = logging.getLogger(__name__)

class ChromaVectorStore:
    """ChromaDB vector store implementation."""
    
    def __init__(self, config: ChromaDBConfig, embedding_function=None):
        """Initialize ChromaDB vector store.
        
        Args:
            config: ChromaDB configuration object
            embedding_function: Custom embedding function (optional)
        """
        self.config = config
        self.client = None
        self.collection = None
        self.embedding_function = embedding_function
        self._initialize_client()
        self._initialize_collection()
    
    def _initialize_client(self) -> None:
        """Initialize ChromaDB client."""
        try:
            # Ensure persist directory exists
            Path(self.config.persist_directory).mkdir(parents=True, exist_ok=True)
            
            # Initialize persistent client
            self.client = chromadb.PersistentClient(
                path=self.config.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"ChromaDB client initialized with persistence at {self.config.persist_directory}")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB client: {e}")
            raise
    
    def _initialize_collection(self) -> None:
        """Initialize or get existing collection."""
        try:
            # Set up embedding function if not provided
            if self.embedding_function is None:
                # Use default sentence transformer embeddings
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.config.collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"Loaded existing collection: {self.config.collection_name}")
            except ValueError:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.config.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": self.config.distance_metric}
                )
                logger.info(f"Created new collection: {self.config.collection_name}")
                
        except Exception as e:
            logger.error(f"Error initializing collection: {e}")
            raise
    
    def add_documents(self, 
                     documents: List[str], 
                     metadatas: List[Dict[str, Any]], 
                     ids: Optional[List[str]] = None) -> bool:
        """Add documents to the vector store.
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
            ids: Optional list of document IDs
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if len(documents) != len(metadatas):
                raise ValueError("Number of documents must match number of metadatas")
            
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in documents]
            
            # Process in batches to avoid memory issues
            batch_size = self.config.max_batch_size
            
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_metas = metadatas[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                
                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
                
                logger.info(f"Added batch {i // batch_size + 1}: {len(batch_docs)} documents")
            
            logger.info(f"Successfully added {len(documents)} documents to collection")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    def similarity_search(self, 
                         query: str, 
                         n_results: int = 5,
                         where: Optional[Dict[str, Any]] = None,
                         where_document: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform similarity search.
        
        Args:
            query: Query text
            n_results: Number of results to return
            where: Metadata filter conditions
            where_document: Document content filter conditions
            
        Returns:
            List of search results with documents, metadata, and distances
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            
            if results['documents'] and len(results['documents']) > 0:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(documents)
                distances = results['distances'][0] if results['distances'] else [0.0] * len(documents)
                ids = results['ids'][0] if results['ids'] else [''] * len(documents)
                
                for i, doc in enumerate(documents):
                    formatted_results.append({
                        'id': ids[i],
                        'document': doc,
                        'metadata': metadatas[i],
                        'distance': distances[i],
                        'similarity_score': 1 - distances[i]  # Convert distance to similarity
                    })
            
            logger.info(f"Found {len(formatted_results)} results for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def similarity_search_with_score_threshold(self,
                                             query: str,
                                             score_threshold: float = 0.7,
                                             n_results: int = 10) -> List[Dict[str, Any]]:
        """Perform similarity search with score threshold filtering.
        
        Args:
            query: Query text
            score_threshold: Minimum similarity score threshold
            n_results: Maximum number of results to retrieve before filtering
            
        Returns:
            List of filtered search results
        """
        # Get more results than needed for filtering
        results = self.similarity_search(query, n_results=n_results)
        
        # Filter by score threshold
        filtered_results = [
            result for result in results 
            if result['similarity_score'] >= score_threshold
        ]
        
        logger.info(f"Filtered {len(results)} results to {len(filtered_results)} above threshold {score_threshold}")
        return filtered_results
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by its ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document data or None if not found
        """
        try:
            results = self.collection.get(
                ids=[doc_id],
                include=["documents", "metadatas"]
            )
            
            if results['documents'] and len(results['documents']) > 0:
                return {
                    'id': doc_id,
                    'document': results['documents'][0],
                    'metadata': results['metadatas'][0] if results['metadatas'] else {}
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {e}")
            return None
    
    def update_document(self, 
                       doc_id: str, 
                       document: str, 
                       metadata: Dict[str, Any]) -> bool:
        """Update an existing document.
        
        Args:
            doc_id: Document ID
            document: New document text
            metadata: New metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.update(
                ids=[doc_id],
                documents=[document],
                metadatas=[metadata]
            )
            logger.info(f"Updated document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Deleted document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False
    
    def delete_collection(self) -> bool:
        """Delete the entire collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete_collection(name=self.config.collection_name)
            logger.info(f"Deleted collection {self.config.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics.
        
        Returns:
            Dictionary containing collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get sample documents to analyze
            sample_results = self.collection.get(limit=min(10, count), include=["metadatas"])
            
            stats = {
                'total_documents': count,
                'collection_name': self.config.collection_name,
                'distance_metric': self.config.distance_metric,
                'sample_metadata_keys': []
            }
            
            # Analyze metadata structure
            if sample_results['metadatas']:
                all_keys = set()
                for metadata in sample_results['metadatas']:
                    if metadata:
                        all_keys.update(metadata.keys())
                stats['sample_metadata_keys'] = list(all_keys)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}
    
    def search_by_metadata(self, 
                          metadata_filter: Dict[str, Any], 
                          limit: int = 100) -> List[Dict[str, Any]]:
        """Search documents by metadata filters.
        
        Args:
            metadata_filter: Metadata filter conditions
            limit: Maximum number of results
            
        Returns:
            List of matching documents
        """
        try:
            results = self.collection.get(
                where=metadata_filter,
                limit=limit,
                include=["documents", "metadatas"]
            )
            
            formatted_results = []
            if results['documents']:
                documents = results['documents']
                metadatas = results['metadatas'] if results['metadatas'] else [{}] * len(documents)
                ids = results['ids'] if results['ids'] else [''] * len(documents)
                
                for i, doc in enumerate(documents):
                    formatted_results.append({
                        'id': ids[i],
                        'document': doc,
                        'metadata': metadatas[i]
                    })
            
            logger.info(f"Found {len(formatted_results)} documents matching metadata filter")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            return []
    
    def backup_collection(self, backup_path: str) -> bool:
        """Create a backup of the collection.
        
        Args:
            backup_path: Path to save backup
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all documents
            all_docs = self.collection.get(include=["documents", "metadatas"])
            
            backup_data = {
                'collection_name': self.config.collection_name,
                'documents': all_docs['documents'],
                'metadatas': all_docs['metadatas'],
                'ids': all_docs['ids']
            }
            
            import json
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            logger.info(f"Collection backed up to {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False
    
    def restore_collection(self, backup_path: str) -> bool:
        """Restore collection from backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import json
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
            
            # Clear existing collection
            self.delete_collection()
            self._initialize_collection()
            
            # Restore documents
            if backup_data['documents']:
                return self.add_documents(
                    documents=backup_data['documents'],
                    metadatas=backup_data['metadatas'],
                    ids=backup_data['ids']
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error restoring from backup: {e}")
            return False