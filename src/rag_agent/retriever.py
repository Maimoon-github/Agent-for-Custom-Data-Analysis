"""
Document retriever for the RAG Agent system.
Handles document retrieval, ranking, and filtering.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from .config import RetrievalConfig
from .vector_store import ChromaVectorStore
from .embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Single retrieval result."""
    document_id: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    rank: int

@dataclass
class RetrievalResponse:
    """Complete retrieval response."""
    query: str
    results: List[RetrievalResult]
    total_found: int
    retrieval_time: float
    reranked: bool = False

class DocumentRetriever:
    """Document retrieval system."""
    
    def __init__(self, 
                 config: RetrievalConfig,
                 vector_store: ChromaVectorStore,
                 embedding_generator: EmbeddingGenerator):
        """Initialize document retriever.
        
        Args:
            config: Retrieval configuration
            vector_store: Vector store instance
            embedding_generator: Embedding generator instance
        """
        self.config = config
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.query_history = []
    
    def retrieve(self, 
                query: str, 
                top_k: Optional[int] = None,
                similarity_threshold: Optional[float] = None,
                metadata_filter: Optional[Dict[str, Any]] = None) -> RetrievalResponse:
        """Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity threshold
            metadata_filter: Optional metadata filter
            
        Returns:
            RetrievalResponse containing results
        """
        import time
        start_time = time.time()
        
        # Use config defaults if not provided
        top_k = top_k or self.config.top_k
        similarity_threshold = similarity_threshold or self.config.similarity_threshold
        
        try:
            # Expand query if needed
            expanded_query = self._expand_query(query)
            
            # Get initial results (more than needed for reranking)
            initial_k = self.config.rerank_top_k if self.config.enable_reranking else top_k
            
            # Perform vector similarity search
            search_results = self.vector_store.similarity_search_with_score_threshold(
                query=expanded_query,
                score_threshold=similarity_threshold,
                n_results=initial_k
            )
            
            # Convert to RetrievalResult objects
            retrieval_results = []
            for i, result in enumerate(search_results):
                retrieval_result = RetrievalResult(
                    document_id=result['id'],
                    content=result['document'],
                    metadata=result['metadata'],
                    similarity_score=result['similarity_score'],
                    rank=i + 1
                )
                retrieval_results.append(retrieval_result)
            
            # Apply reranking if enabled
            reranked = False
            if self.config.enable_reranking and len(retrieval_results) > top_k:
                retrieval_results = self._rerank_results(query, retrieval_results)
                retrieval_results = retrieval_results[:top_k]
                reranked = True
                
                # Update ranks after reranking
                for i, result in enumerate(retrieval_results):
                    result.rank = i + 1
            
            retrieval_time = time.time() - start_time
            
            # Create response
            response = RetrievalResponse(
                query=query,
                results=retrieval_results,
                total_found=len(search_results),
                retrieval_time=retrieval_time,
                reranked=reranked
            )
            
            # Store in query history
            self._store_query_history(query, response)
            
            logger.info(f"Retrieved {len(retrieval_results)} documents for query in {retrieval_time:.3f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return RetrievalResponse(
                query=query,
                results=[],
                total_found=0,
                retrieval_time=time.time() - start_time
            )
    
    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms or related terms.
        
        Args:
            query: Original query
            
        Returns:
            Expanded query
        """
        # Basic query expansion - in practice, you might use more sophisticated methods
        expanded_terms = []
        
        # Add original query
        expanded_terms.append(query)
        
        # Simple synonym expansion (you could use WordNet, word embeddings, etc.)
        synonyms = {
            'artificial intelligence': ['AI', 'machine learning', 'ML'],
            'machine learning': ['ML', 'artificial intelligence', 'AI'],
            'natural language processing': ['NLP', 'text processing'],
            'neural network': ['neural net', 'deep learning'],
            'database': ['DB', 'data store', 'storage']
        }
        
        query_lower = query.lower()
        for term, syns in synonyms.items():
            if term in query_lower:
                expanded_terms.extend(syns)
        
        return ' '.join(expanded_terms)
    
    def _rerank_results(self, 
                       query: str, 
                       results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Rerank results using additional scoring methods.
        
        Args:
            query: Original query
            results: Initial retrieval results
            
        Returns:
            Reranked results
        """
        try:
            # Calculate additional scores
            for result in results:
                # Keyword matching score
                keyword_score = self._calculate_keyword_score(query, result.content)
                
                # Length penalty (prefer more substantial content)
                length_score = self._calculate_length_score(result.content)
                
                # Metadata boost (e.g., prefer certain file types or sources)
                metadata_score = self._calculate_metadata_score(result.metadata)
                
                # Combine scores
                combined_score = (
                    0.6 * result.similarity_score +
                    0.2 * keyword_score +
                    0.1 * length_score +
                    0.1 * metadata_score
                )
                
                result.similarity_score = combined_score
            
            # Sort by combined score
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            logger.debug(f"Reranked {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Error reranking results: {e}")
            return results
    
    def _calculate_keyword_score(self, query: str, content: str) -> float:
        """Calculate keyword matching score.
        
        Args:
            query: Search query
            content: Document content
            
        Returns:
            Keyword matching score between 0 and 1
        """
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.0
        
        # Calculate intersection
        matches = len(query_words.intersection(content_words))
        return matches / len(query_words)
    
    def _calculate_length_score(self, content: str) -> float:
        """Calculate length-based score.
        
        Args:
            content: Document content
            
        Returns:
            Length score between 0 and 1
        """
        length = len(content)
        
        # Prefer content between 200-2000 characters
        if 200 <= length <= 2000:
            return 1.0
        elif length < 200:
            return length / 200.0
        else:
            return max(0.5, 2000.0 / length)
    
    def _calculate_metadata_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate metadata-based score boost.
        
        Args:
            metadata: Document metadata
            
        Returns:
            Metadata score between 0 and 1
        """
        score = 0.5  # Base score
        
        # Boost for certain file types
        file_type = metadata.get('file_type', '').lower()
        if file_type in ['pdf', 'docx']:
            score += 0.2
        elif file_type in ['txt', 'md']:
            score += 0.1
        
        # Boost for recent documents
        timestamp = metadata.get('processing_timestamp')
        if timestamp:
            import time
            current_time = time.time()
            age_days = (current_time - timestamp) / (24 * 3600)
            if age_days < 30:  # Recent documents
                score += 0.2
            elif age_days < 90:
                score += 0.1
        
        return min(1.0, score)
    
    def _store_query_history(self, query: str, response: RetrievalResponse) -> None:
        """Store query in history for analysis.
        
        Args:
            query: Search query
            response: Retrieval response
        """
        history_entry = {
            'query': query,
            'timestamp': time.time(),
            'num_results': len(response.results),
            'retrieval_time': response.retrieval_time,
            'reranked': response.reranked
        }
        
        self.query_history.append(history_entry)
        
        # Keep only last 1000 queries
        if len(self.query_history) > 1000:
            self.query_history = self.query_history[-1000:]
    
    def get_query_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent query history.
        
        Args:
            limit: Maximum number of queries to return
            
        Returns:
            List of recent queries with metadata
        """
        return self.query_history[-limit:]
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics.
        
        Returns:
            Dictionary containing retrieval statistics
        """
        if not self.query_history:
            return {
                'total_queries': 0,
                'avg_retrieval_time': 0,
                'avg_results_per_query': 0,
                'reranking_usage': 0
            }
        
        total_queries = len(self.query_history)
        total_time = sum(q['retrieval_time'] for q in self.query_history)
        total_results = sum(q['num_results'] for q in self.query_history)
        reranked_queries = sum(1 for q in self.query_history if q['reranked'])
        
        return {
            'total_queries': total_queries,
            'avg_retrieval_time': total_time / total_queries,
            'avg_results_per_query': total_results / total_queries,
            'reranking_usage': reranked_queries / total_queries,
            'config': {
                'top_k': self.config.top_k,
                'similarity_threshold': self.config.similarity_threshold,
                'reranking_enabled': self.config.enable_reranking
            }
        }
    
    def hybrid_search(self, 
                     query: str,
                     keyword_weight: float = 0.3,
                     semantic_weight: float = 0.7) -> RetrievalResponse:
        """Perform hybrid search combining keyword and semantic search.
        
        Args:
            query: Search query
            keyword_weight: Weight for keyword search
            semantic_weight: Weight for semantic search
            
        Returns:
            RetrievalResponse with hybrid results
        """
        import time
        start_time = time.time()
        
        try:
            # Semantic search
            semantic_results = self.retrieve(query)
            
            # Keyword search (simplified - searches in metadata and content)
            keyword_results = self._keyword_search(query)
            
            # Combine and reweight results
            combined_results = self._combine_search_results(
                semantic_results.results,
                keyword_results,
                semantic_weight,
                keyword_weight
            )
            
            retrieval_time = time.time() - start_time
            
            return RetrievalResponse(
                query=query,
                results=combined_results[:self.config.top_k],
                total_found=len(combined_results),
                retrieval_time=retrieval_time,
                reranked=True  # Hybrid search includes reranking
            )
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return self.retrieve(query)  # Fallback to regular search
    
    def _keyword_search(self, query: str) -> List[RetrievalResult]:
        """Perform keyword-based search.
        
        Args:
            query: Search query
            
        Returns:
            List of keyword search results
        """
        # Simplified keyword search - in practice, you might use BM25 or similar
        query_terms = query.lower().split()
        
        # Get all documents (this is inefficient for large collections)
        # In practice, you'd maintain a separate keyword index
        all_results = self.vector_store.search_by_metadata({})
        
        keyword_results = []
        for result in all_results:
            content_lower = result['document'].lower()
            score = 0.0
            
            for term in query_terms:
                # Count occurrences of each term
                occurrences = content_lower.count(term)
                score += occurrences
            
            if score > 0:
                # Normalize by document length
                score = score / len(result['document'].split())
                
                keyword_result = RetrievalResult(
                    document_id=result['id'],
                    content=result['document'],
                    metadata=result['metadata'],
                    similarity_score=score,
                    rank=0  # Will be set later
                )
                keyword_results.append(keyword_result)
        
        # Sort by keyword score
        keyword_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return keyword_results
    
    def _combine_search_results(self,
                               semantic_results: List[RetrievalResult],
                               keyword_results: List[RetrievalResult],
                               semantic_weight: float,
                               keyword_weight: float) -> List[RetrievalResult]:
        """Combine semantic and keyword search results.
        
        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            semantic_weight: Weight for semantic scores
            keyword_weight: Weight for keyword scores
            
        Returns:
            Combined and reranked results
        """
        # Create lookup for keyword scores
        keyword_scores = {result.document_id: result.similarity_score for result in keyword_results}
        
        # Combine scores
        combined_results = []
        processed_ids = set()
        
        # Process semantic results first
        for result in semantic_results:
            keyword_score = keyword_scores.get(result.document_id, 0.0)
            combined_score = (semantic_weight * result.similarity_score + 
                            keyword_weight * keyword_score)
            
            result.similarity_score = combined_score
            combined_results.append(result)
            processed_ids.add(result.document_id)
        
        # Add keyword-only results
        for result in keyword_results:
            if result.document_id not in processed_ids:
                combined_score = keyword_weight * result.similarity_score
                result.similarity_score = combined_score
                combined_results.append(result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(combined_results):
            result.rank = i + 1
        
        return combined_results
    
    def get_similar_documents(self, 
                             document_id: str, 
                             top_k: int = 5) -> List[RetrievalResult]:
        """Find documents similar to a given document.
        
        Args:
            document_id: ID of the reference document
            top_k: Number of similar documents to return
            
        Returns:
            List of similar documents
        """
        try:
            # Get the reference document
            ref_doc = self.vector_store.get_document_by_id(document_id)
            if not ref_doc:
                logger.error(f"Document not found: {document_id}")
                return []
            
            # Use the document content as query
            response = self.retrieve(ref_doc['document'], top_k=top_k + 1)
            
            # Filter out the reference document itself
            similar_docs = [
                result for result in response.results 
                if result.document_id != document_id
            ]
            
            return similar_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
            return []

class AdvancedRetriever(DocumentRetriever):
    """Advanced retriever with additional features."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    def retrieve_with_cache(self, query: str, **kwargs) -> RetrievalResponse:
        """Retrieve with caching support.
        
        Args:
            query: Search query
            **kwargs: Additional retrieval parameters
            
        Returns:
            RetrievalResponse (potentially from cache)
        """
        import time
        current_time = time.time()
        
        # Check cache
        cache_key = f"{query}_{str(sorted(kwargs.items()))}"
        if cache_key in self.query_cache:
            cached_result, timestamp = self.query_cache[cache_key]
            if current_time - timestamp < self.cache_ttl:
                logger.debug("Returning cached result")
                return cached_result
        
        # Retrieve and cache
        result = self.retrieve(query, **kwargs)
        self.query_cache[cache_key] = (result, current_time)
        
        # Clean old cache entries
        self._clean_cache(current_time)
        
        return result
    
    def _clean_cache(self, current_time: float) -> None:
        """Clean expired cache entries."""
        expired_keys = [
            key for key, (_, timestamp) in self.query_cache.items()
            if current_time - timestamp > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.query_cache[key]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self.query_cache),
            'cache_ttl': self.cache_ttl
        }