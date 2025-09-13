"""
Retrieval mechanism for the RAG system
Handles similarity search, reranking, and context optimization
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

from langchain.schema import Document

from config.settings import config
from src.core.vector_database import vector_db
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class RetrievalResult:
    """Result from document retrieval"""
    documents: List[Document]
    scores: List[float]
    query: str
    retrieval_method: str
    total_found: int
    retrieval_time: float
    metadata: Dict[str, Any]

class QueryProcessor:
    """Processes and expands user queries"""
    
    def __init__(self):
        self.synonyms = {
            # Add domain-specific synonyms
            'analyze': ['examine', 'study', 'investigate', 'review'],
            'data': ['information', 'dataset', 'records', 'statistics'],
            'trend': ['pattern', 'tendency', 'direction', 'movement'],
            'increase': ['rise', 'growth', 'improvement', 'boost'],
            'decrease': ['decline', 'reduction', 'drop', 'fall']
        }
    
    def preprocess_query(self, query: str) -> str:
        """Clean and preprocess the user query"""
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        # Convert to lowercase for processing
        query = query.lower()
        
        # Remove special characters but keep important punctuation
        query = re.sub(r'[^\w\s\-\?\!\.]', ' ', query)
        
        return query.strip()
    
    def expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms"""
        expanded_terms = []
        words = query.split()
        
        for word in words:
            expanded_terms.append(word)
            # Add synonyms if available
            if word in self.synonyms:
                expanded_terms.extend(self.synonyms[word][:2])  # Add top 2 synonyms
        
        return ' '.join(expanded_terms)
    
    def extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query"""
        # Simple keyword extraction (can be enhanced with NLP libraries)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        words = query.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def categorize_query(self, query: str) -> str:
        """Categorize the query to determine retrieval strategy"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what', 'define', 'explain', 'describe']):
            return 'factual'
        elif any(word in query_lower for word in ['how', 'why', 'analyze', 'compare']):
            return 'analytical'
        elif any(word in query_lower for word in ['list', 'show', 'find', 'search']):
            return 'listing'
        elif any(word in query_lower for word in ['trend', 'pattern', 'correlation']):
            return 'pattern'
        else:
            return 'general'

class DocumentReranker:
    """Reranks retrieved documents based on relevance and quality"""
    
    def __init__(self):
        self.quality_weights = {
            'length_score': 0.2,
            'keyword_density': 0.3,
            'document_quality': 0.2,
            'freshness': 0.1,
            'source_reliability': 0.2
        }
    
    def calculate_quality_score(self, document: Document, query_keywords: List[str]) -> float:
        """Calculate document quality score"""
        content = document.page_content.lower()
        metadata = document.metadata
        
        # Length score (optimal length gets higher score)
        length = len(content)
        if 200 <= length <= 2000:
            length_score = 1.0
        elif length < 200:
            length_score = length / 200
        else:
            length_score = max(0.5, 2000 / length)
        
        # Keyword density score
        keyword_matches = sum(1 for keyword in query_keywords if keyword in content)
        keyword_density = keyword_matches / max(1, len(query_keywords))
        
        # Document quality based on structure
        sentences = content.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        quality_score = min(1.0, avg_sentence_length / 20)  # Optimal ~20 words per sentence
        
        # Freshness score (if date available)
        freshness_score = 0.5  # Default neutral score
        if 'modified_at' in metadata:
            try:
                modified_date = datetime.fromisoformat(str(metadata['modified_at']))
                days_old = (datetime.now() - modified_date).days
                freshness_score = max(0.1, 1.0 - (days_old / 365))  # Decay over a year
            except:
                pass
        
        # Source reliability (based on file type and size)
        source_score = 0.5  # Default
        doc_type = metadata.get('document_type', '')
        if doc_type == '.pdf':
            source_score = 0.8
        elif doc_type in ['.docx', '.txt']:
            source_score = 0.7
        elif doc_type == '.csv':
            source_score = 0.6
        
        # Combine scores
        total_score = (
            self.quality_weights['length_score'] * length_score +
            self.quality_weights['keyword_density'] * keyword_density +
            self.quality_weights['document_quality'] * quality_score +
            self.quality_weights['freshness'] * freshness_score +
            self.quality_weights['source_reliability'] * source_score
        )
        
        return min(1.0, total_score)
    
    def rerank_documents(self, documents_with_scores: List[Tuple[Document, float]], 
                        query: str, query_keywords: List[str]) -> List[Tuple[Document, float]]:
        """Rerank documents based on quality and relevance"""
        reranked = []
        
        for document, similarity_score in documents_with_scores:
            quality_score = self.calculate_quality_score(document, query_keywords)
            
            # Combine similarity and quality scores
            combined_score = 0.7 * similarity_score + 0.3 * quality_score
            
            reranked.append((document, combined_score))
        
        # Sort by combined score
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked

class RetrievalEngine:
    """Main retrieval engine that orchestrates the search process"""
    
    def __init__(self):
        self.config = config.retrieval
        self.query_processor = QueryProcessor()
        self.reranker = DocumentReranker()
    
    def retrieve_documents(self, query: str, k: int = None, 
                         filters: Dict[str, Any] = None,
                         rerank: bool = True) -> RetrievalResult:
        """Retrieve relevant documents for a query"""
        start_time = datetime.now()
        
        try:
            if k is None:
                k = self.config.k
            
            # Process the query
            processed_query = self.query_processor.preprocess_query(query)
            expanded_query = self.query_processor.expand_query(processed_query)
            keywords = self.query_processor.extract_keywords(processed_query)
            query_category = self.query_processor.categorize_query(processed_query)
            
            logger.debug(f"Query processing - Original: {query}, Processed: {processed_query}")
            logger.debug(f"Keywords: {keywords}, Category: {query_category}")
            
            # Determine retrieval strategy based on query category
            retrieval_k = self._get_retrieval_k(query_category, k)
            
            # Perform vector similarity search
            documents_with_scores = vector_db.search(
                query=expanded_query,
                k=retrieval_k,
                filter_dict=filters
            )
            
            if not documents_with_scores:
                logger.warning(f"No documents found for query: {query}")
                return RetrievalResult(
                    documents=[],
                    scores=[],
                    query=query,
                    retrieval_method="vector_similarity",
                    total_found=0,
                    retrieval_time=0.0,
                    metadata={"query_category": query_category, "keywords": keywords}
                )
            
            # Filter by similarity threshold
            filtered_docs = [
                (doc, score) for doc, score in documents_with_scores
                if score >= self.config.similarity_threshold
            ]
            
            if not filtered_docs:
                logger.warning(f"No documents above similarity threshold for query: {query}")
                # Return top documents anyway but with lower confidence
                filtered_docs = documents_with_scores[:min(3, len(documents_with_scores))]
            
            # Apply reranking if requested
            if rerank:
                filtered_docs = self.reranker.rerank_documents(filtered_docs, processed_query, keywords)
            
            # Take top k results
            final_docs = filtered_docs[:k]
            
            # Extract documents and scores
            documents = [doc for doc, score in final_docs]
            scores = [score for doc, score in final_docs]
            
            retrieval_time = (datetime.now() - start_time).total_seconds()
            
            # Add retrieval metadata to documents
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    'retrieval_score': scores[i],
                    'retrieval_rank': i + 1,
                    'query_keywords': keywords,
                    'retrieval_timestamp': datetime.now().isoformat()
                })
            
            result = RetrievalResult(
                documents=documents,
                scores=scores,
                query=query,
                retrieval_method="vector_similarity_with_rerank" if rerank else "vector_similarity",
                total_found=len(documents_with_scores),
                retrieval_time=retrieval_time,
                metadata={
                    "processed_query": processed_query,
                    "expanded_query": expanded_query,
                    "keywords": keywords,
                    "query_category": query_category,
                    "similarity_threshold": self.config.similarity_threshold,
                    "reranked": rerank
                }
            )
            
            logger.info(f"Retrieved {len(documents)} documents for query '{query}' in {retrieval_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error during document retrieval: {str(e)}")
            raise
    
    def _get_retrieval_k(self, query_category: str, requested_k: int) -> int:
        """Determine how many documents to retrieve based on query category"""
        multipliers = {
            'factual': 1.5,      # Get more for factual queries
            'analytical': 2.0,    # Get even more for analysis
            'listing': 1.2,       # Slightly more for listing
            'pattern': 2.0,       # More for pattern recognition
            'general': 1.0        # Default
        }
        
        multiplier = multipliers.get(query_category, 1.0)
        return min(20, int(requested_k * multiplier))  # Cap at 20
    
    def hybrid_search(self, query: str, k: int = None, 
                     keyword_weight: float = 0.3,
                     semantic_weight: float = 0.7) -> RetrievalResult:
        """Perform hybrid search combining keyword and semantic search"""
        try:
            if k is None:
                k = self.config.k
            
            # Extract keywords for keyword search
            keywords = self.query_processor.extract_keywords(query)
            
            # Perform semantic search
            semantic_results = self.retrieve_documents(query, k=k*2, rerank=False)
            
            # Perform keyword-based metadata search
            keyword_documents = []
            for keyword in keywords[:3]:  # Use top 3 keywords
                metadata_results = vector_db.search_by_metadata(
                    metadata_filter={'content_keywords': keyword},
                    k=5
                )
                keyword_documents.extend(metadata_results)
            
            # Combine and score results
            combined_docs = {}
            
            # Add semantic results
            for doc, score in zip(semantic_results.documents, semantic_results.scores):
                doc_id = doc.metadata.get('chunk_id', id(doc))
                combined_docs[doc_id] = {
                    'document': doc,
                    'semantic_score': score,
                    'keyword_score': 0.0
                }
            
            # Add keyword results
            for doc in keyword_documents:
                doc_id = doc.metadata.get('chunk_id', id(doc))
                if doc_id in combined_docs:
                    combined_docs[doc_id]['keyword_score'] = 0.8  # High keyword relevance
                else:
                    combined_docs[doc_id] = {
                        'document': doc,
                        'semantic_score': 0.0,
                        'keyword_score': 0.8
                    }
            
            # Calculate combined scores
            final_results = []
            for doc_data in combined_docs.values():
                combined_score = (
                    semantic_weight * doc_data['semantic_score'] +
                    keyword_weight * doc_data['keyword_score']
                )
                final_results.append((doc_data['document'], combined_score))
            
            # Sort and take top k
            final_results.sort(key=lambda x: x[1], reverse=True)
            final_results = final_results[:k]
            
            documents = [doc for doc, score in final_results]
            scores = [score for doc, score in final_results]
            
            return RetrievalResult(
                documents=documents,
                scores=scores,
                query=query,
                retrieval_method="hybrid_search",
                total_found=len(combined_docs),
                retrieval_time=semantic_results.retrieval_time,
                metadata={
                    "keywords": keywords,
                    "semantic_weight": semantic_weight,
                    "keyword_weight": keyword_weight
                }
            )
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            raise
    
    def get_similar_documents(self, document_id: str, k: int = 5) -> List[Document]:
        """Find documents similar to a given document"""
        try:
            # Get the reference document
            ref_docs = vector_db.search_by_metadata({'chunk_id': document_id}, k=1)
            
            if not ref_docs:
                logger.warning(f"Reference document {document_id} not found")
                return []
            
            ref_doc = ref_docs[0]
            
            # Use the document content as query
            similar_docs_with_scores = vector_db.search(
                query=ref_doc.page_content,
                k=k+1  # +1 to exclude the reference document itself
            )
            
            # Filter out the reference document
            similar_docs = [
                doc for doc, score in similar_docs_with_scores
                if doc.metadata.get('chunk_id') != document_id
            ]
            
            return similar_docs[:k]
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {str(e)}")
            return []

# Create global retrieval engine instance
retrieval_engine = RetrievalEngine()
