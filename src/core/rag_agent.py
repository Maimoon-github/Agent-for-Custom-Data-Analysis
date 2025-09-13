"""
Main RAG Agent that orchestrates all components
This is the primary interface for the RAG system
"""

import asyncio
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from langchain.schema import Document

from config.settings import config
from src.core.document_processor import document_processor
from src.core.vector_database import vector_db
from src.core.ollama_manager import ollama_manager
from src.core.retrieval_engine import retrieval_engine
from src.core.response_generator import response_generator, GenerationResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class RAGResponse:
    """Complete response from the RAG system"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    query: str
    processing_time: float
    retrieval_stats: Dict[str, Any]
    generation_stats: Dict[str, Any]
    metadata: Dict[str, Any]

class RAGAgent:
    """Main RAG Agent class that coordinates all components"""
    
    def __init__(self):
        self.config = config
        self.is_initialized = False
        self.knowledge_base_stats = {}
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all RAG system components"""
        try:
            logger.info("Initializing RAG Agent...")
            
            # Test Ollama connection
            ollama_status = ollama_manager.test_connection()
            if not ollama_status['connection_ok']:
                raise ConnectionError("Ollama connection failed. Please ensure Ollama is running.")
            
            if not ollama_status['model_available']:
                logger.warning(f"Model {config.ollama.model} not available. Attempting to pull...")
                # The OllamaManager will handle model pulling automatically
            
            # Initialize vector database (already done in vector_db initialization)
            self.knowledge_base_stats = vector_db.get_collection_stats()
            
            self.is_initialized = True
            logger.info(f"RAG Agent initialized successfully. Knowledge base: {self.knowledge_base_stats.get('total_documents', 0)} documents")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG Agent: {str(e)}")
            raise
    
    def add_documents(self, source_path: str, recursive: bool = True) -> Dict[str, Any]:
        """Add documents to the knowledge base"""
        try:
            logger.info(f"Adding documents from: {source_path}")
            start_time = datetime.now()
            
            # Check if source is file or directory
            source = Path(source_path)
            
            if not source.exists():
                raise FileNotFoundError(f"Source path does not exist: {source_path}")
            
            # Load documents
            if source.is_file():
                documents = document_processor.load_document(str(source))
            else:
                documents = document_processor.load_directory(str(source), recursive=recursive)
            
            if not documents:
                logger.warning(f"No documents found in {source_path}")
                return {"status": "warning", "message": "No documents found", "added": 0}
            
            # Validate documents
            valid_documents = document_processor.validate_documents(documents)
            
            # Add to vector database
            indexing_result = vector_db.add_documents(valid_documents)
            
            # Update stats
            self.knowledge_base_stats = vector_db.get_collection_stats()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "status": "success",
                "documents_processed": len(documents),
                "documents_added": indexing_result["added"],
                "documents_failed": indexing_result["errors"],
                "processing_time": processing_time,
                "total_documents_in_kb": self.knowledge_base_stats.get("total_documents", 0),
                "source_path": str(source),
                "document_stats": document_processor.get_document_stats(valid_documents)
            }
            
            logger.info(f"Document addition completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def query(self, question: str, 
              query_type: str = 'general',
              retrieval_method: str = 'semantic',
              k: int = None,
              filters: Dict[str, Any] = None,
              **generation_kwargs) -> RAGResponse:
        """Query the RAG system"""
        try:
            if not self.is_initialized:
                raise RuntimeError("RAG Agent not properly initialized")
            
            logger.info(f"Processing query: {question}")
            start_time = datetime.now()
            
            # Determine query type if not specified
            if query_type == 'auto':
                query_type = self._classify_query(question)
            
            # Retrieve relevant documents
            if retrieval_method == 'hybrid':
                retrieval_result = retrieval_engine.hybrid_search(question, k=k)
            else:
                retrieval_result = retrieval_engine.retrieve_documents(
                    query=question,
                    k=k,
                    filters=filters,
                    rerank=True
                )
            
            if not retrieval_result.documents:
                return RAGResponse(
                    answer="I couldn't find any relevant information in the knowledge base to answer your question. Please try rephrasing your query or add more documents to the knowledge base.",
                    sources=[],
                    confidence_score=0.0,
                    query=question,
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    retrieval_stats={"documents_found": 0},
                    generation_stats={},
                    metadata={"query_type": query_type, "retrieval_method": retrieval_method}
                )
            
            # Generate response
            generation_result = response_generator.generate_response(
                question=question,
                retrieval_result=retrieval_result,
                query_type=query_type,
                **generation_kwargs
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            response = RAGResponse(
                answer=generation_result.response,
                sources=generation_result.sources,
                confidence_score=generation_result.confidence_score,
                query=question,
                processing_time=processing_time,
                retrieval_stats={
                    "documents_found": retrieval_result.total_found,
                    "documents_used": len(retrieval_result.documents),
                    "retrieval_time": retrieval_result.retrieval_time,
                    "retrieval_method": retrieval_result.retrieval_method,
                    "average_score": sum(retrieval_result.scores) / len(retrieval_result.scores) if retrieval_result.scores else 0
                },
                generation_stats={
                    "generation_time": generation_result.generation_time,
                    "token_usage": generation_result.token_usage,
                    "model": generation_result.metadata.get("model", "")
                },
                metadata={
                    "query_type": query_type,
                    "retrieval_method": retrieval_method,
                    "timestamp": datetime.now().isoformat(),
                    "knowledge_base_size": self.knowledge_base_stats.get("total_documents", 0)
                }
            )
            
            logger.info(f"Query processed successfully in {processing_time:.2f}s (confidence: {generation_result.confidence_score:.2f})")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    def stream_query(self, question: str, 
                    query_type: str = 'general',
                    retrieval_method: str = 'semantic',
                    k: int = None,
                    filters: Dict[str, Any] = None,
                    **generation_kwargs):
        """Stream a response to a query"""
        try:
            if not self.is_initialized:
                raise RuntimeError("RAG Agent not properly initialized")
            
            logger.info(f"Processing streaming query: {question}")
            
            # Retrieve relevant documents
            if retrieval_method == 'hybrid':
                retrieval_result = retrieval_engine.hybrid_search(question, k=k)
            else:
                retrieval_result = retrieval_engine.retrieve_documents(
                    query=question,
                    k=k,
                    filters=filters,
                    rerank=True
                )
            
            if not retrieval_result.documents:
                yield "I couldn't find any relevant information in the knowledge base to answer your question."
                return
            
            # Stream response
            for chunk in response_generator.generate_streaming_response(
                question=question,
                retrieval_result=retrieval_result,
                query_type=query_type,
                **generation_kwargs
            ):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error in streaming query: {str(e)}")
            yield f"Error processing query: {str(e)}"
    
    def _classify_query(self, question: str) -> str:
        """Automatically classify the query type"""
        question_lower = question.lower()
        
        # Simple rule-based classification
        if any(word in question_lower for word in ['what is', 'define', 'explain', 'describe']):
            return 'factual'
        elif any(word in question_lower for word in ['analyze', 'why', 'how does', 'impact', 'effect']):
            return 'analytical'
        elif any(word in question_lower for word in ['compare', 'difference', 'versus', 'vs']):
            return 'comparative'
        elif any(word in question_lower for word in ['summarize', 'summary', 'overview']):
            return 'summary'
        else:
            return 'general'
    
    def get_similar_documents(self, document_id: str, k: int = 5) -> List[Document]:
        """Find documents similar to a given document"""
        try:
            return retrieval_engine.get_similar_documents(document_id, k)
        except Exception as e:
            logger.error(f"Error finding similar documents: {str(e)}")
            return []
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge base"""
        try:
            stats = vector_db.get_collection_stats()
            
            # Add system status
            ollama_status = ollama_manager.test_connection()
            
            return {
                "knowledge_base": stats,
                "system_status": {
                    "ollama_connected": ollama_status.get("connection_ok", False),
                    "model_available": ollama_status.get("model_available", False),
                    "current_model": config.ollama.model,
                    "vector_db_connected": True,  # If we got here, it's connected
                },
                "configuration": {
                    "chunk_size": config.document.chunk_size,
                    "chunk_overlap": config.document.chunk_overlap,
                    "retrieval_k": config.retrieval.k,
                    "embedding_model": config.embedding.model
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting knowledge base stats: {str(e)}")
            return {"error": str(e)}
    
    def search_knowledge_base(self, query: str, k: int = 10, 
                            filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search the knowledge base and return formatted results"""
        try:
            retrieval_result = retrieval_engine.retrieve_documents(
                query=query,
                k=k,
                filters=filters,
                rerank=True
            )
            
            results = []
            for i, (doc, score) in enumerate(zip(retrieval_result.documents, retrieval_result.scores)):
                results.append({
                    "rank": i + 1,
                    "content": doc.page_content,
                    "score": round(score, 3),
                    "source": doc.metadata.get("file_name", "Unknown"),
                    "document_type": doc.metadata.get("document_type", ""),
                    "chunk_index": doc.metadata.get("chunk_index", 0),
                    "metadata": doc.metadata
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {str(e)}")
            return []
    
    def delete_documents(self, filters: Dict[str, str] = None, 
                        document_ids: List[str] = None) -> Dict[str, Any]:
        """Delete documents from the knowledge base"""
        try:
            deleted_count = vector_db.delete_documents(filter_dict=filters, ids=document_ids)
            
            # Update stats
            self.knowledge_base_stats = vector_db.get_collection_stats()
            
            return {
                "status": "success",
                "deleted_count": deleted_count,
                "remaining_documents": self.knowledge_base_stats.get("total_documents", 0)
            }
            
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def reset_knowledge_base(self) -> bool:
        """Reset the entire knowledge base"""
        try:
            logger.warning("Resetting knowledge base - all documents will be deleted")
            success = vector_db.reset_collection()
            
            if success:
                self.knowledge_base_stats = {"total_documents": 0}
                logger.info("Knowledge base reset successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Error resetting knowledge base: {str(e)}")
            return False
    
    def explain_answer(self, question: str, answer: RAGResponse) -> str:
        """Generate an explanation of how the answer was derived"""
        try:
            # Create a mock retrieval result for the explanation
            from src.core.retrieval_engine import RetrievalResult
            
            retrieval_result = RetrievalResult(
                documents=[],  # We don't need documents for explanation
                scores=[],
                query=question,
                retrieval_method=answer.metadata.get("retrieval_method", "unknown"),
                total_found=answer.retrieval_stats.get("documents_found", 0),
                retrieval_time=answer.retrieval_stats.get("retrieval_time", 0),
                metadata=answer.metadata
            )
            
            return response_generator.explain_reasoning(question, retrieval_result)
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return "Unable to generate explanation for this answer."
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check of the system"""
        try:
            health = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "components": {}
            }
            
            # Check Ollama
            ollama_status = ollama_manager.test_connection()
            health["components"]["ollama"] = {
                "status": "healthy" if ollama_status["connection_ok"] else "unhealthy",
                "model_available": ollama_status["model_available"],
                "generation_working": ollama_status["generation_ok"]
            }
            
            # Check Vector Database
            try:
                doc_count = vector_db.get_collection_count()
                health["components"]["vector_database"] = {
                    "status": "healthy",
                    "document_count": doc_count
                }
            except Exception as e:
                health["components"]["vector_database"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
            
            # Check Document Processor
            health["components"]["document_processor"] = {
                "status": "healthy",
                "supported_formats": config.document.supported_formats
            }
            
            # Overall status
            if any(comp["status"] == "unhealthy" for comp in health["components"].values()):
                health["status"] = "degraded"
            
            return health
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Create global RAG agent instance
rag_agent = RAGAgent()
