"""
Response generation system for the RAG agent
Handles prompt engineering, context optimization, and response formatting
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from langchain.schema import Document

from config.settings import config
from src.core.ollama_manager import ollama_manager
from src.core.retrieval_engine import RetrievalResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class GenerationResult:
    """Result from response generation"""
    response: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    generation_time: float
    token_usage: Dict[str, int]
    metadata: Dict[str, Any]

class PromptTemplate:
    """Manages prompt templates for different types of queries"""
    
    def __init__(self):
        self.templates = {
            'factual': """You are an expert data analyst assistant. Based on the provided context, answer the user's question with accurate, factual information.

Context:
{context}

Question: {question}

Instructions:
- Provide a clear, accurate answer based solely on the provided context
- If the context doesn't contain enough information, state this clearly
- Include specific details and numbers when available
- Cite sources using [Source: filename] format
- Keep the response concise but comprehensive

Answer:""",

            'analytical': """You are an expert data analyst. Analyze the provided information to answer the user's question with detailed insights and reasoning.

Context:
{context}

Question: {question}

Instructions:
- Provide thorough analysis based on the context
- Break down complex concepts into understandable parts
- Identify patterns, trends, and relationships in the data
- Support your analysis with evidence from the sources
- Use [Source: filename] to cite specific information
- Include recommendations if appropriate

Analysis:""",

            'comparative': """You are an expert analyst skilled in comparative analysis. Compare and contrast the information in the context to answer the user's question.

Context:
{context}

Question: {question}

Instructions:
- Compare relevant aspects mentioned in the context
- Highlight similarities and differences
- Use data and examples from the sources to support comparisons
- Organize the comparison in a clear, structured manner
- Cite sources using [Source: filename] format
- Provide a balanced perspective

Comparison:""",

            'summary': """You are an expert at synthesizing information. Summarize the key information from the context to answer the user's question.

Context:
{context}

Question: {question}

Instructions:
- Provide a comprehensive summary of relevant information
- Organize information logically and coherently
- Include the most important points and key findings
- Maintain accuracy to the source material
- Use [Source: filename] to cite information
- Keep the summary focused on the question asked

Summary:""",

            'general': """You are a helpful data analyst assistant. Use the provided context to answer the user's question accurately and helpfully.

Context:
{context}

Question: {question}

Instructions:
- Answer the question using information from the provided context
- Be accurate and stick to the facts presented in the sources
- If information is incomplete, acknowledge this limitation
- Provide clear, well-structured responses
- Cite sources using [Source: filename] format
- Be helpful and comprehensive in your response

Answer:"""
        }
        
        self.system_message = """You are an expert data analyst AI assistant specializing in answering questions based on document analysis. 

Key principles:
1. Base your answers strictly on the provided context
2. Be accurate and factual in all responses
3. Clearly cite sources for information used
4. Acknowledge when information is insufficient
5. Provide clear, well-structured, and helpful responses
6. Maintain professional and analytical tone

When citing sources, use the format: [Source: filename]
If you cannot find relevant information in the context, clearly state this limitation."""
    
    def get_template(self, query_type: str) -> str:
        """Get the appropriate template for a query type"""
        return self.templates.get(query_type, self.templates['general'])
    
    def format_prompt(self, question: str, context: str, query_type: str = 'general') -> str:
        """Format the prompt with question and context"""
        template = self.get_template(query_type)
        return template.format(question=question, context=context)

class ContextOptimizer:
    """Optimizes context for better response generation"""
    
    def __init__(self):
        self.max_context_length = config.ollama.context_window - 1000  # Leave room for response
    
    def prepare_context(self, retrieval_result: RetrievalResult) -> Tuple[str, List[Dict[str, Any]]]:
        """Prepare optimized context from retrieved documents"""
        if not retrieval_result.documents:
            return "No relevant information found in the knowledge base.", []
        
        context_parts = []
        sources = []
        total_length = 0
        
        for i, (document, score) in enumerate(zip(retrieval_result.documents, retrieval_result.scores)):
            # Extract source information
            source_info = {
                'filename': document.metadata.get('file_name', 'Unknown'),
                'file_path': document.metadata.get('file_path', ''),
                'chunk_index': document.metadata.get('chunk_index', 0),
                'relevance_score': round(score, 3),
                'document_type': document.metadata.get('document_type', ''),
                'rank': i + 1
            }
            
            # Format content with source reference
            content = document.page_content.strip()
            
            # Check if adding this content would exceed context limit
            content_with_source = f"[Source: {source_info['filename']}]\n{content}\n\n"
            
            if total_length + len(content_with_source) > self.max_context_length:
                # Try to include partial content if this is important
                if i < 3:  # For top 3 results, try to include partial content
                    remaining_space = self.max_context_length - total_length - 100
                    if remaining_space > 200:
                        truncated_content = content[:remaining_space] + "..."
                        content_with_source = f"[Source: {source_info['filename']}]\n{truncated_content}\n\n"
                        context_parts.append(content_with_source)
                        sources.append(source_info)
                        total_length += len(content_with_source)
                break
            
            context_parts.append(content_with_source)
            sources.append(source_info)
            total_length += len(content_with_source)
        
        context = "".join(context_parts)
        
        # Add metadata about the search
        if retrieval_result.metadata:
            context_header = f"Search results for: '{retrieval_result.query}'\n"
            context_header += f"Found {retrieval_result.total_found} total documents, showing top {len(sources)} most relevant.\n\n"
            context = context_header + context
        
        logger.debug(f"Prepared context: {len(context)} characters from {len(sources)} sources")
        return context, sources
    
    def optimize_for_query_type(self, context: str, query_type: str) -> str:
        """Optimize context based on query type"""
        if query_type == 'summary':
            # For summaries, ensure all sources are represented
            return context
        elif query_type == 'analytical':
            # For analysis, prioritize quantitative data
            return self._highlight_numerical_data(context)
        elif query_type == 'comparative':
            # For comparisons, structure for easy comparison
            return self._structure_for_comparison(context)
        else:
            return context
    
    def _highlight_numerical_data(self, context: str) -> str:
        """Highlight numerical data in context"""
        # Simple implementation - can be enhanced
        import re
        
        # Find numbers and percentages
        number_pattern = r'\b\d+(?:\.\d+)?(?:%|\s*percent)?\b'
        highlighted = re.sub(number_pattern, lambda m: f"**{m.group()}**", context)
        
        return highlighted
    
    def _structure_for_comparison(self, context: str) -> str:
        """Structure context for easier comparison"""
        # Simple implementation - group by sources
        sources = context.split('[Source:')
        if len(sources) > 1:
            structured = sources[0]  # Header
            for i, source in enumerate(sources[1:], 1):
                structured += f"\n--- Source {i} ---\n[Source:{source}"
            return structured
        return context

class ResponseGenerator:
    """Main response generation system"""
    
    def __init__(self):
        self.prompt_template = PromptTemplate()
        self.context_optimizer = ContextOptimizer()
    
    def generate_response(self, question: str, retrieval_result: RetrievalResult,
                         query_type: str = 'general', **generation_kwargs) -> GenerationResult:
        """Generate a response based on retrieved documents"""
        start_time = datetime.now()
        
        try:
            # Prepare context from retrieved documents
            context, sources = self.context_optimizer.prepare_context(retrieval_result)
            
            # Optimize context for query type
            optimized_context = self.context_optimizer.optimize_for_query_type(context, query_type)
            
            # Create the prompt
            prompt = self.prompt_template.format_prompt(question, optimized_context, query_type)
            system_message = self.prompt_template.system_message
            
            logger.debug(f"Generated prompt length: {len(prompt)} characters")
            
            # Generate response using Ollama
            generation_result = ollama_manager.generate_response(
                prompt=prompt,
                system_message=system_message,
                **generation_kwargs
            )
            
            response_text = generation_result['response']
            
            # Calculate confidence score based on various factors
            confidence_score = self._calculate_confidence_score(
                retrieval_result, len(sources), response_text
            )
            
            # Post-process response
            processed_response = self._post_process_response(response_text, sources)
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            result = GenerationResult(
                response=processed_response,
                sources=sources,
                confidence_score=confidence_score,
                generation_time=generation_time,
                token_usage={
                    'input_tokens': generation_result.get('input_tokens', 0),
                    'output_tokens': generation_result.get('output_tokens', 0)
                },
                metadata={
                    'query_type': query_type,
                    'context_length': len(optimized_context),
                    'sources_used': len(sources),
                    'retrieval_method': retrieval_result.retrieval_method,
                    'generation_params': generation_result.get('parameters', {}),
                    'model': generation_result.get('model', ''),
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            logger.info(f"Generated response for '{question}' in {generation_time:.2f}s "
                       f"(confidence: {confidence_score:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def _calculate_confidence_score(self, retrieval_result: RetrievalResult, 
                                  num_sources: int, response_text: str) -> float:
        """Calculate confidence score for the response"""
        factors = []
        
        # Retrieval quality factor
        if retrieval_result.scores:
            avg_retrieval_score = sum(retrieval_result.scores) / len(retrieval_result.scores)
            factors.append(min(1.0, avg_retrieval_score))
        else:
            factors.append(0.0)
        
        # Source count factor
        source_factor = min(1.0, num_sources / 3.0)  # Optimal around 3 sources
        factors.append(source_factor)
        
        # Response length factor (not too short, not too long)
        response_length = len(response_text.split())
        if 50 <= response_length <= 300:
            length_factor = 1.0
        elif response_length < 50:
            length_factor = response_length / 50.0
        else:
            length_factor = max(0.5, 300.0 / response_length)
        factors.append(length_factor)
        
        # Citation factor (check if response includes source citations)
        citation_count = response_text.count('[Source:')
        citation_factor = min(1.0, citation_count / max(1, num_sources))
        factors.append(citation_factor)
        
        # Calculate weighted average
        weights = [0.4, 0.2, 0.2, 0.2]  # Emphasize retrieval quality
        confidence = sum(f * w for f, w in zip(factors, weights))
        
        return round(min(1.0, max(0.0, confidence)), 3)
    
    def _post_process_response(self, response: str, sources: List[Dict[str, Any]]) -> str:
        """Post-process the generated response"""
        # Clean up the response
        response = response.strip()
        
        # Ensure proper source citation format
        response = self._fix_source_citations(response, sources)
        
        # Add source list at the end if not already present
        if '[Source:' in response and not response.endswith('\n\n**Sources:**'):
            response = self._append_source_list(response, sources)
        
        return response
    
    def _fix_source_citations(self, response: str, sources: List[Dict[str, Any]]) -> str:
        """Fix and standardize source citations in the response"""
        # This is a simple implementation - can be enhanced
        import re
        
        # Find all source citations
        citation_pattern = r'\[Source:\s*([^\]]+)\]'
        
        def replace_citation(match):
            cited_name = match.group(1).strip()
            # Find the matching source
            for source in sources:
                if source['filename'] in cited_name or cited_name in source['filename']:
                    return f"[Source: {source['filename']}]"
            return match.group(0)  # Keep original if no match found
        
        return re.sub(citation_pattern, replace_citation, response)
    
    def _append_source_list(self, response: str, sources: List[Dict[str, Any]]) -> str:
        """Append a formatted source list to the response"""
        if not sources:
            return response
        
        source_list = "\n\n**Sources:**\n"
        for i, source in enumerate(sources, 1):
            source_list += f"{i}. {source['filename']}"
            if source.get('document_type'):
                source_list += f" ({source['document_type']})"
            if source.get('relevance_score'):
                source_list += f" - Relevance: {source['relevance_score']}"
            source_list += "\n"
        
        return response + source_list
    
    def generate_streaming_response(self, question: str, retrieval_result: RetrievalResult,
                                  query_type: str = 'general', **generation_kwargs):
        """Generate a streaming response"""
        try:
            # Prepare context
            context, sources = self.context_optimizer.prepare_context(retrieval_result)
            optimized_context = self.context_optimizer.optimize_for_query_type(context, query_type)
            
            # Create prompt
            prompt = self.prompt_template.format_prompt(question, optimized_context, query_type)
            system_message = self.prompt_template.system_message
            
            # Stream response
            for chunk in ollama_manager.generate_streaming_response(
                prompt=prompt,
                system_message=system_message,
                **generation_kwargs
            ):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            yield f"Error generating response: {str(e)}"
    
    def explain_reasoning(self, question: str, retrieval_result: RetrievalResult) -> str:
        """Generate an explanation of the reasoning process"""
        explanation_prompt = f"""Explain your reasoning process for answering this question based on the retrieved information:

Question: {question}

Retrieved Information Summary:
- Found {len(retrieval_result.documents)} relevant documents
- Average relevance score: {sum(retrieval_result.scores) / len(retrieval_result.scores) if retrieval_result.scores else 0:.3f}
- Retrieval method: {retrieval_result.retrieval_method}

Please explain:
1. How you interpreted the question
2. Which sources were most relevant and why
3. How you synthesized the information
4. Any limitations or uncertainties in your answer

Reasoning:"""
        
        try:
            result = ollama_manager.generate_response(
                prompt=explanation_prompt,
                temperature=0.3  # Lower temperature for more structured reasoning
            )
            return result['response']
        except Exception as e:
            logger.error(f"Error generating reasoning explanation: {str(e)}")
            return "Unable to generate reasoning explanation."

# Create global response generator instance
response_generator = ResponseGenerator()
