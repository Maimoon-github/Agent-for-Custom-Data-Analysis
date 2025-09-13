"""
Main RAG Agent class that orchestrates all components.
Provides the main interface for document ingestion and query processing.
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from .config import RAGConfig
from .llm_interface import OllamaLLM
from .vector_store import ChromaVectorStore
from .document_processor import DocumentProcessor, ProcessingResult
from .embeddings import EmbeddingGenerator, CachedEmbeddingGenerator
from .retriever import DocumentRetriever, RetrievalResponse

logger = logging.getLogger(__name__)

@dataclass
class QueryResponse:
    """Response from the RAG agent."""
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    response_time: float
    num_sources_used: int
    retrieval_stats: Dict[str, Any]

class RAGAgent:
    """Main RAG Agent class."""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize RAG Agent.
        
        Args:
            config: RAG configuration (uses default if None)
        """
        self.config = config or RAGConfig()
        self._setup_logging()
        
        # Initialize components
        self.llm = None
        self.vector_store = None
        self.document_processor = None
        self.embedding_generator = None
        self.retriever = None
        
        # State
        self.is_initialized = False
        self.conversation_history = []
        
        logger.info("RAG Agent initialized")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.system.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.config.system.logs_dir}/rag_agent.log"),
                logging.StreamHandler()
            ]
        )
    
    def initialize(self) -> bool:
        """Initialize all components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing RAG Agent components...")
            
            # Initialize embedding generator
            self.embedding_generator = CachedEmbeddingGenerator(
                config=self.config.embedding,
                ollama_base_url=self.config.ollama.base_url,
                cache_size=self.config.system.cache_size
            )
            logger.info("✓ Embedding generator initialized")
            
            # Initialize vector store
            self.vector_store = ChromaVectorStore(
                config=self.config.chromadb,
                embedding_function=None  # Will use default
            )
            logger.info("✓ Vector store initialized")
            
            # Initialize document processor
            self.document_processor = DocumentProcessor(
                config=self.config.document_processing
            )
            logger.info("✓ Document processor initialized")
            
            # Initialize retriever
            self.retriever = DocumentRetriever(
                config=self.config.retrieval,
                vector_store=self.vector_store,
                embedding_generator=self.embedding_generator
            )
            logger.info("✓ Document retriever initialized")
            
            # Initialize LLM
            self.llm = OllamaLLM(config=self.config.ollama)
            logger.info("✓ LLM interface initialized")
            
            self.is_initialized = True
            logger.info("RAG Agent initialization complete!")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing RAG Agent: {e}")
            return False
    
    def ingest_document(self, file_path: Union[str, Path]) -> bool:
        """Ingest a single document.
        
        Args:
            file_path: Path to document to ingest
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_initialized:
            logger.error("RAG Agent not initialized. Call initialize() first.")
            return False
        
        try:
            logger.info(f"Ingesting document: {file_path}")
            
            # Process document
            result = self.document_processor.process_file(file_path)
            
            if not result.success:
                logger.error(f"Failed to process document: {result.error}")
                return False
            
            # Generate embeddings and store in vector database
            documents = [doc.content for doc in result.documents]
            metadatas = [doc.metadata for doc in result.documents]
            ids = [doc.doc_id for doc in result.documents]
            
            success = self.vector_store.add_documents(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            if success:
                logger.info(f"Successfully ingested {len(result.documents)} chunks from {file_path}")
                return True
            else:
                logger.error(f"Failed to add documents to vector store")
                return False
                
        except Exception as e:
            logger.error(f"Error ingesting document {file_path}: {e}")
            return False
    
    def ingest_directory(self, directory_path: Union[str, Path], recursive: bool = True) -> Dict[str, Any]:
        """Ingest all documents in a directory.
        
        Args:
            directory_path: Path to directory
            recursive: Whether to process subdirectories
            
        Returns:
            Dictionary with ingestion statistics
        """
        if not self.is_initialized:
            logger.error("RAG Agent not initialized. Call initialize() first.")
            return {'success': False, 'error': 'Agent not initialized'}
        
        try:
            logger.info(f"Ingesting directory: {directory_path}")
            
            # Process all files in directory
            results = self.document_processor.process_directory(directory_path, recursive)
            
            # Collect all successful documents
            all_documents = []
            all_metadatas = []
            all_ids = []
            
            for result in results:
                if result.success:
                    for doc in result.documents:
                        all_documents.append(doc.content)
                        all_metadatas.append(doc.metadata)
                        all_ids.append(doc.doc_id)
            
            # Add to vector store in batches
            if all_documents:
                success = self.vector_store.add_documents(
                    documents=all_documents,
                    metadatas=all_metadatas,
                    ids=all_ids
                )
                
                if success:
                    stats = self.document_processor.get_processing_stats(results)
                    stats['success'] = True
                    logger.info(f"Successfully ingested directory: {stats}")
                    return stats
                else:
                    return {'success': False, 'error': 'Failed to add documents to vector store'}
            else:
                return {'success': False, 'error': 'No valid documents found'}
                
        except Exception as e:
            logger.error(f"Error ingesting directory {directory_path}: {e}")
            return {'success': False, 'error': str(e)}
    
    def ingest_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Ingest raw text.
        
        Args:
            text: Text content to ingest
            metadata: Optional metadata
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_initialized:
            logger.error("RAG Agent not initialized. Call initialize() first.")
            return False
        
        try:
            # Process text
            result = self.document_processor.process_text(text, metadata)
            
            if not result.success:
                logger.error(f"Failed to process text: {result.error}")
                return False
            
            # Add to vector store
            documents = [doc.content for doc in result.documents]
            metadatas = [doc.metadata for doc in result.documents]
            ids = [doc.doc_id for doc in result.documents]
            
            success = self.vector_store.add_documents(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            if success:
                logger.info(f"Successfully ingested text: {len(result.documents)} chunks")
                return True
            else:
                logger.error("Failed to add text to vector store")
                return False
                
        except Exception as e:
            logger.error(f"Error ingesting text: {e}")
            return False
    
    def query(self, 
             question: str, 
             include_sources: bool = True,
             max_sources: Optional[int] = None) -> QueryResponse:
        """Query the RAG agent.
        
        Args:
            question: Question to ask
            include_sources: Whether to include source documents
            max_sources: Maximum number of sources to retrieve
            
        Returns:
            QueryResponse with answer and sources
        """
        if not self.is_initialized:
            error_response = QueryResponse(
                query=question,
                answer="Error: RAG Agent not initialized. Please call initialize() first.",
                sources=[],
                response_time=0,
                num_sources_used=0,
                retrieval_stats={}
            )
            return error_response
        
        start_time = time.time()
        
        try:
            # Validate query
            if not question.strip():
                return QueryResponse(
                    query=question,
                    answer="Please provide a valid question.",
                    sources=[],
                    response_time=time.time() - start_time,
                    num_sources_used=0,
                    retrieval_stats={}
                )
            
            # Check query length
            if len(question) > self.config.system.max_query_length:
                question = question[:self.config.system.max_query_length]
                logger.warning(f"Query truncated to {self.config.system.max_query_length} characters")
            
            # Retrieve relevant documents
            max_sources = max_sources or self.config.retrieval.top_k
            retrieval_response = self.retriever.retrieve(question, top_k=max_sources)
            
            # Generate answer
            if retrieval_response.results:
                answer = self._generate_answer(question, retrieval_response.results)
                sources = self._format_sources(retrieval_response.results) if include_sources else []
            else:
                answer = "I couldn't find any relevant information to answer your question. Please try rephrasing or check if the relevant documents have been ingested."
                sources = []
            
            response_time = time.time() - start_time
            
            # Create response
            query_response = QueryResponse(
                query=question,
                answer=answer,
                sources=sources,
                response_time=response_time,
                num_sources_used=len(retrieval_response.results),
                retrieval_stats={
                    'total_found': retrieval_response.total_found,
                    'retrieval_time': retrieval_response.retrieval_time,
                    'reranked': retrieval_response.reranked
                }
            )
            
            # Store in conversation history
            self._store_conversation(query_response)
            
            logger.info(f"Query processed in {response_time:.3f}s with {len(retrieval_response.results)} sources")
            
            return query_response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return QueryResponse(
                query=question,
                answer=f"Error processing query: {str(e)}",
                sources=[],
                response_time=time.time() - start_time,
                num_sources_used=0,
                retrieval_stats={}
            )
    
    def _generate_answer(self, question: str, sources: List) -> str:
        """Generate answer using LLM with retrieved context.
        
        Args:
            question: User question
            sources: Retrieved source documents
            
        Returns:
            Generated answer
        """
        try:
            # Prepare context from sources
            context_parts = []
            for i, source in enumerate(sources):
                source_text = f"Source {i+1}:\n{source.content}\n"
                context_parts.append(source_text)
            
            context = "\n".join(context_parts)
            
            # Truncate context if too long
            context = self.llm.truncate_to_context_window(context)
            
            # Create prompt
            prompt = self.llm.create_rag_prompt(
                query=question,
                context=context,
                system_message="""You are a helpful AI assistant that answers questions based on the provided context. 
                Use only the information from the context to answer the question. 
                If the context doesn't contain enough information to answer the question, say so clearly.
                Always cite the sources when possible using "Source X" notation.
                Be concise but comprehensive in your answer."""
            )
            
            # Generate response
            answer = self.llm.generate(prompt, temperature=0.3)
            
            # Validate response
            if not self.llm.validate_response(answer):
                return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I encountered an error while generating the answer. Please try again."
    
    def _format_sources(self, sources: List) -> List[Dict[str, Any]]:
        """Format sources for response.
        
        Args:
            sources: List of retrieval results
            
        Returns:
            List of formatted source dictionaries
        """
        formatted_sources = []
        
        for i, source in enumerate(sources):
            formatted_source = {
                'id': source.document_id,
                'content': source.content[:500] + "..." if len(source.content) > 500 else source.content,
                'similarity_score': round(source.similarity_score, 3),
                'rank': source.rank,
                'metadata': {
                    'file_name': source.metadata.get('file_name', 'Unknown'),
                    'source': source.metadata.get('source', 'Unknown'),
                    'chunk_index': source.metadata.get('chunk_index', 0),
                    'file_type': source.metadata.get('file_type', 'Unknown')
                }
            }
            formatted_sources.append(formatted_source)
        
        return formatted_sources
    
    def _store_conversation(self, response: QueryResponse) -> None:
        """Store conversation in history.
        
        Args:
            response: Query response to store
        """
        conversation_entry = {
            'timestamp': time.time(),
            'query': response.query,
            'answer': response.answer,
            'num_sources': response.num_sources_used,
            'response_time': response.response_time
        }
        
        self.conversation_history.append(conversation_entry)
        
        # Keep only last 100 conversations
        if len(self.conversation_history) > 100:
            self.conversation_history = self.conversation_history[-100:]
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation history.
        
        Args:
            limit: Maximum number of conversations to return
            
        Returns:
            List of recent conversations
        """
        return self.conversation_history[-limit:]
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics.
        
        Returns:
            Dictionary containing system statistics
        """
        stats = {
            'is_initialized': self.is_initialized,
            'conversation_count': len(self.conversation_history),
            'config': self.config.to_dict()
        }
        
        if self.is_initialized:
            # Vector store stats
            if self.vector_store:
                stats['vector_store'] = self.vector_store.get_collection_stats()
            
            # Retriever stats
            if self.retriever:
                stats['retrieval'] = self.retriever.get_retrieval_stats()
            
            # Embedding stats
            if hasattr(self.embedding_generator, 'get_cache_stats'):
                stats['embedding_cache'] = self.embedding_generator.get_cache_stats()
            
            # LLM stats
            if self.llm:
                stats['llm'] = self.llm.get_model_info()
        
        return stats
    
    def clear_conversation_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")
    
    def backup_data(self, backup_path: str) -> bool:
        """Backup the vector database.
        
        Args:
            backup_path: Path to save backup
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_initialized or not self.vector_store:
            logger.error("Cannot backup: RAG Agent not initialized")
            return False
        
        try:
            success = self.vector_store.backup_collection(backup_path)
            if success:
                logger.info(f"Data backed up to {backup_path}")
            return success
        except Exception as e:
            logger.error(f"Error backing up data: {e}")
            return False
    
    def restore_data(self, backup_path: str) -> bool:
        """Restore the vector database from backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_initialized or not self.vector_store:
            logger.error("Cannot restore: RAG Agent not initialized")
            return False
        
        try:
            success = self.vector_store.restore_collection(backup_path)
            if success:
                logger.info(f"Data restored from {backup_path}")
            return success
        except Exception as e:
            logger.error(f"Error restoring data: {e}")
            return False
    
    def reset_vector_store(self) -> bool:
        """Reset (clear) the vector store.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_initialized or not self.vector_store:
            logger.error("Cannot reset: RAG Agent not initialized")
            return False
        
        try:
            success = self.vector_store.delete_collection()
            if success:
                # Reinitialize collection
                self.vector_store._initialize_collection()
                logger.info("Vector store reset successfully")
            return success
        except Exception as e:
            logger.error(f"Error resetting vector store: {e}")
            return False